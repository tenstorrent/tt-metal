# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device weight preparation and loading for the Qwen Gated DeltaNet (GDN) layer.

Host preparation turns the canonical layer-local weights (``reference/gdn/weights.py``) into the per-TP-rank device
layout; materialization places, shards and caches them. With ``Hk_r = Hk / TP`` K heads and ``HV_r = HV / TP`` V heads
per rank (rank ``r`` owns K heads ``[r Hk_r, (r + 1) Hk_r)`` and exactly the V heads that read them):

* ``input_projection`` ``[H, TP * W]``, sharded on the last dim: per rank the columns
  ``[q_r | k_r | v_r | z_r | a_r, zero-padded to a tile | b_r]`` (``W = 2 Hk_r K + 2 HV_r V + pad(HV_r) + HV_r``), so
  every block starts tile-aligned.
* ``convolution_taps``: four ``[1, 1, conv_dim]`` taps, per rank ``[q_r | k_r | v_r]`` (the fused HF ``conv1d`` split).
* ``decay_scale`` ``= -exp(A_log)`` and ``decay_bias`` ``= dt_bias``, ``[1, 1, HV]`` FP32, sharded on the last dim:
  ``g = decay_scale * softplus(a + decay_bias)``.
* ``norm`` ``[V]`` replicated; ``output_projection`` ``= out_proj^T`` ``[HV V, H]``, sharded on dim -2.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig
from models.demos.deepseek_v3_d_p.reference.gdn.weights import validate_gdn_weights
from models.demos.deepseek_v3_d_p.tt.kda.weights import group_projection_rows_by_tp_rank, tensor_parallel_mesh_mapper
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import FastCacheChecker

# Covers the host layout and dtypes above; bump with any change to what a cached tensorbin holds.
_CACHE_SCHEMA_VERSION = 1
_TILE = 32
# Projection, convolution and norm weights are bf16 (design decision D4; bfp8 is a later perf experiment). The gate
# parameters stay fp32: the decay is formed in fp32 on device.
_WEIGHT_DTYPE = ttnn.bfloat16
_GATE_DTYPE = ttnn.float32


@dataclass(frozen=True)
class _Artifact:
    name: str
    dtype: ttnn.DataType
    shard_dim: int | None  # None: replicated over the TP axis


def _artifacts(config: GDNConfig) -> tuple[_Artifact, ...]:
    return (
        _Artifact("input_projection", _WEIGHT_DTYPE, -1),
        _Artifact("output_projection", _WEIGHT_DTYPE, -2),
        _Artifact("decay_scale", _GATE_DTYPE, -1),
        _Artifact("decay_bias", _GATE_DTYPE, -1),
        _Artifact("norm", _WEIGHT_DTYPE, None),
    ) + tuple(_Artifact(f"conv_tap_{tap}", _WEIGHT_DTYPE, -1) for tap in range(config.conv_kernel_size))


@dataclass(frozen=True)
class GDNHostWeights:
    """Per-TP-rank device layout of one layer, on the host (see the module docstring)."""

    input_projection: torch.Tensor
    output_projection: torch.Tensor
    decay_scale: torch.Tensor
    decay_bias: torch.Tensor
    norm: torch.Tensor
    convolution_taps: tuple[torch.Tensor, ...]

    def tensor(self, name: str) -> torch.Tensor:
        if name.startswith("conv_tap_"):
            return self.convolution_taps[int(name.removeprefix("conv_tap_"))]
        return getattr(self, name)


def gdn_input_projection_widths(config: GDNConfig, tensor_parallel_size: int) -> dict[str, int]:
    """Column widths of one rank's input-projection block, in order: q, k, v, z, a (with its tile padding), b."""
    key_heads, value_heads = _heads_per_rank(config, tensor_parallel_size)
    return {
        "q": key_heads * config.head_k_dim,
        "k": key_heads * config.head_k_dim,
        "v": value_heads * config.head_v_dim,
        "z": value_heads * config.head_v_dim,
        "a": -(-value_heads // _TILE) * _TILE,
        "b": value_heads,
    }


def prepare_gdn_host_weights(
    state_dict: Mapping[str, torch.Tensor], config: GDNConfig, tensor_parallel_size: int
) -> GDNHostWeights:
    """Validate the canonical layer-local weights and lay them out for ``tensor_parallel_size`` ranks."""
    validate_gdn_weights(state_dict, config)
    _, value_heads = _heads_per_rank(config, tensor_parallel_size)
    qkv = state_dict["in_proj_qkv.weight"]
    q, k, v = qkv.split([config.q_dim, config.k_dim, config.v_dim])
    a = state_dict["in_proj_a.weight"]
    padding = gdn_input_projection_widths(config, tensor_parallel_size)["a"] - value_heads
    a_padded = torch.cat(
        [
            torch.cat([rows, rows.new_zeros(padding, rows.shape[1])])
            for rows in a.split(value_heads)  # one block per rank
        ]
    )
    input_projection = group_projection_rows_by_tp_rank(
        (q, k, v, state_dict["in_proj_z.weight"], a_padded, state_dict["in_proj_b.weight"]), tensor_parallel_size
    ).T
    conv = state_dict["conv1d.weight"][:, 0, :]
    convolution_taps = tuple(
        group_projection_rows_by_tp_rank(
            tuple(conv[:, tap].split([config.q_dim, config.k_dim, config.v_dim])), tensor_parallel_size
        ).reshape(1, 1, -1)
        for tap in range(config.conv_kernel_size)
    )
    heads = config.num_value_heads
    return GDNHostWeights(
        input_projection=input_projection,
        output_projection=state_dict["out_proj.weight"].T,
        decay_scale=-state_dict["A_log"].float().exp().reshape(1, 1, heads),
        decay_bias=state_dict["dt_bias"].float().reshape(1, 1, heads),
        norm=state_dict["norm.weight"],
        convolution_taps=convolution_taps,
    )


def _heads_per_rank(config: GDNConfig, tensor_parallel_size: int) -> tuple[int, int]:
    """K and V heads per rank; a rank must hold whole K-head groups."""
    if tensor_parallel_size <= 0 or config.num_key_heads % tensor_parallel_size:
        raise ValueError(
            f"num_key_heads {config.num_key_heads} must be divisible by tensor parallel size {tensor_parallel_size}"
        )
    return config.num_key_heads // tensor_parallel_size, config.num_value_heads // tensor_parallel_size


def _tensor_parallel_size(mesh_shape: tuple[int, int], config: GDNConfig, tensor_parallel_axis: int) -> int:
    if tensor_parallel_axis not in (0, 1):
        raise ValueError(f"tensor_parallel_axis must be 0 or 1, got {tensor_parallel_axis}")
    tensor_parallel_size = mesh_shape[tensor_parallel_axis]
    _heads_per_rank(config, tensor_parallel_size)
    return tensor_parallel_size


def _mesh_shape(device: ttnn.Device | ttnn.MeshDevice) -> tuple[int, int]:
    return tuple(device.shape) if isinstance(device, ttnn.MeshDevice) else (1, 1)


def _cache_stem(
    cache_name_prefix: str, name: str, config: GDNConfig, mesh_shape: tuple[int, int], tensor_parallel_axis: int
) -> str:
    """Transformation identity of one tensorbin. The source identity (checkpoint digest, slice, synthetic seed) is the
    caller's, carried by ``cache_path`` / ``cache_name_prefix``."""
    config_payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    config_digest = hashlib.sha256(config_payload.encode("utf-8")).hexdigest()[:16]
    return (
        f"{cache_name_prefix}.{name}.v{_CACHE_SCHEMA_VERSION}.{config_digest}."
        f"mesh{mesh_shape[0]}x{mesh_shape[1]}.tpaxis{tensor_parallel_axis}"
    )


def _tensorbin(stem: str, artifact: _Artifact) -> str:
    return f"{stem}_dtype_{artifact.dtype.name}_layout_{ttnn.TILE_LAYOUT.name}.tensorbin"


@dataclass(frozen=True)
class GDNWeights:
    input_projection: ttnn.Tensor
    output_projection: ttnn.Tensor
    decay_scale: ttnn.Tensor
    decay_bias: ttnn.Tensor
    norm: ttnn.Tensor
    convolution_taps: tuple[ttnn.Tensor, ...]
    tensor_parallel_size: int
    tensor_parallel_axis: int

    @classmethod
    def check_cache_complete(
        cls,
        cache_path: Path | None,
        cache_name_prefix: str,
        config: GDNConfig,
        mesh_shape: tuple[int, int],
        *,
        tensor_parallel_axis: int = 1,
    ) -> bool:
        """Return whether every dtype/layout/placement-specific GDN tensorbin exists for ``mesh_shape``."""
        if cache_path is None or not Path(cache_path).is_dir():
            return False
        checker = FastCacheChecker(Path(cache_path))
        mesh_shape = tuple(mesh_shape)
        return all(
            checker.pattern_exists(
                _tensorbin(_cache_stem(cache_name_prefix, a.name, config, mesh_shape, tensor_parallel_axis), a), "GDN"
            )
            for a in _artifacts(config)
        )

    @classmethod
    def build_ttnn_cache(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        cache_path: Path,
        cache_name_prefix: str,
        config: GDNConfig,
        mesh_shape: tuple[int, int],
        *,
        tensor_parallel_axis: int = 1,
    ) -> None:
        """Build all GDN tensorbins for a ``mesh_shape`` placement without a device (shape-only mesh mapper; a process
        that must not touch hardware sets ``TT_METAL_MOCK_CLUSTER_DESC_PATH``)."""
        mesh_shape = tuple(mesh_shape)
        tensor_parallel_size = _tensor_parallel_size(mesh_shape, config, tensor_parallel_axis)
        _materialize(
            prepare_gdn_host_weights(state_dict, config, tensor_parallel_size),
            device=None,
            config=config,
            cache_path=Path(cache_path),
            cache_name_prefix=cache_name_prefix,
            mesh_shape=mesh_shape,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_axis=tensor_parallel_axis,
        )


def load_gdn_weights(
    device: ttnn.Device | ttnn.MeshDevice,
    config: GDNConfig,
    state_dict: Mapping[str, torch.Tensor] | None,
    cache_path: Path | None = None,
    *,
    cache_name_prefix: str = "gdn",
    tensor_parallel_axis: int = 1,
) -> GDNWeights:
    """Prepare GDN weights and place them on ``device``; ``state_dict=None`` loads only from a complete cache and fails
    before touching the device when any tensorbin is missing."""
    mesh_shape = _mesh_shape(device)
    tensor_parallel_size = _tensor_parallel_size(mesh_shape, config, tensor_parallel_axis)
    cache_path = Path(cache_path) if cache_path is not None else None
    if state_dict is None:
        if not GDNWeights.check_cache_complete(
            cache_path, cache_name_prefix, config, mesh_shape, tensor_parallel_axis=tensor_parallel_axis
        ):
            raise FileNotFoundError(f"incomplete GDN TTNN cache for {cache_name_prefix!r} at {cache_path!r}")
        host_weights = None
    else:
        host_weights = prepare_gdn_host_weights(state_dict, config, tensor_parallel_size)
    tensors = _materialize(
        host_weights,
        device=device,
        config=config,
        cache_path=cache_path,
        cache_name_prefix=cache_name_prefix,
        mesh_shape=mesh_shape,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_axis=tensor_parallel_axis,
    )
    taps = tuple(tensors.pop(f"conv_tap_{tap}") for tap in range(config.conv_kernel_size))
    return GDNWeights(
        **tensors,
        convolution_taps=taps,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_axis=tensor_parallel_axis,
    )


def _materialize(
    host_weights: GDNHostWeights | None,
    *,
    device: ttnn.Device | ttnn.MeshDevice | None,
    config: GDNConfig,
    cache_path: Path | None,
    cache_name_prefix: str,
    mesh_shape: tuple[int, int],
    tensor_parallel_size: int,
    tensor_parallel_axis: int,
) -> dict[str, ttnn.Tensor]:
    """Write (and, with a device, place) every artifact; ``host_weights=None`` reads them from the cache instead."""
    if cache_path is not None:
        cache_path.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for artifact in _artifacts(config):
        stem = _cache_stem(cache_name_prefix, artifact.name, config, mesh_shape, tensor_parallel_axis)
        cache_file = cache_path / stem if cache_path is not None else None
        if host_weights is None:
            tensors[artifact.name] = ttnn.load_tensor(cache_path / _tensorbin(stem, artifact), device=device)
            continue
        tensors[artifact.name] = ttnn.as_tensor(
            host_weights.tensor(artifact.name).contiguous(),
            dtype=artifact.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=tensor_parallel_mesh_mapper(
                device,
                mesh_shape=mesh_shape,
                tensor_parallel_size=tensor_parallel_size,
                tensor_parallel_axis=tensor_parallel_axis,
                shard_dim=artifact.shard_dim,
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG if device is not None else None,
            cache_file_name=cache_file,
        )
    return tensors
