# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device weight loading for Kimi Delta Attention."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda.config import KDAConfig
from models.demos.deepseek_v3_d_p.reference.kda.weights import normalize_kda_state_dict
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import FastCacheChecker

# v3: decay_scale_flat and decay_bias_flat are FP32.
_CACHE_SCHEMA_VERSION = 3

# Stored at the caller's `projection_dtype`.
PROJECTION_ARTIFACTS = ("input_projection_head_major", "decay_output_projection", "output_projection")
# The decay gate's per-channel scale and additive term stay FP32: in BF16 they bias every token's sigmoid argument by up
# to A * 2^-7 per channel, a systematic gate error that a recurrence accumulates over each chunk instead of averaging
# out (K3 real text: mean dG/G +0.9% with a BF16 chain, +2e-3 with an FP32 offset passed as the matmul bias, ~1e-5 with
# a separate FP32 add; tt_metal_tracker-g1b.4.13). Every other weight is BF16.
_FP32_ARTIFACTS = frozenset(("decay_scale_flat", "decay_bias_flat"))


def _artifact_dtype(name: str, projection_dtype: ttnn.DataType) -> ttnn.DataType:
    if name in PROJECTION_ARTIFACTS:
        return projection_dtype
    return ttnn.float32 if name in _FP32_ARTIFACTS else ttnn.bfloat16


def _cache_file_suffix(name: str, projection_dtype: ttnn.DataType) -> str:
    return f"_dtype_{_artifact_dtype(name, projection_dtype).name}_layout_{ttnn.TILE_LAYOUT.name}.tensorbin"


def _parallel_geometry(device: ttnn.Device | ttnn.MeshDevice, tensor_parallel_axis: int) -> tuple[tuple[int, int], int]:
    if tensor_parallel_axis not in (0, 1):
        raise ValueError(f"tensor_parallel_axis must be 0 or 1, got {tensor_parallel_axis}")
    if isinstance(device, ttnn.MeshDevice):
        mesh_shape = tuple(device.shape)
        return mesh_shape, mesh_shape[tensor_parallel_axis]
    return (1, 1), 1


def _cache_artifact_names(config: KDAConfig) -> tuple[str, ...]:
    fixed = (
        "input_projection_head_major",
        "decay_output_projection",
        "output_projection",
        "decay_scale_flat",
        "decay_bias_flat",
        "norm",
    )
    return fixed + tuple(f"conv_tap_{tap}" for tap in range(config.conv_kernel_size))


def current_cache_globs(cache_name_prefix: str, config: KDAConfig, projection_dtype: ttnn.DataType) -> tuple[str, ...]:
    """One glob per KDA tensorbin as this code writes it: current schema version and per-artifact dtype.

    A cache written by other code (an older schema, BF16 decay terms, another projection dtype) misses at least one.
    """
    return tuple(
        f"{cache_name_prefix}.{name}.v{_CACHE_SCHEMA_VERSION}.*{_cache_file_suffix(name, projection_dtype)}"
        for name in _cache_artifact_names(config)
    )


def _cache_stem(
    cache_name_prefix: str,
    name: str,
    config: KDAConfig,
    mesh_shape: tuple[int, int],
    tensor_parallel_axis: int,
) -> str:
    config_payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    config_digest = hashlib.sha256(config_payload.encode("utf-8")).hexdigest()[:16]
    return (
        f"{cache_name_prefix}.{name}.v{_CACHE_SCHEMA_VERSION}.{config_digest}."
        f"mesh{mesh_shape[0]}x{mesh_shape[1]}.tpaxis{tensor_parallel_axis}"
    )


@dataclass(frozen=True)
class _KDAHostWeights:
    input_projection: torch.Tensor
    decay_output_projection: torch.Tensor
    output_projection: torch.Tensor
    decay_scale_flat: torch.Tensor
    decay_bias_flat: torch.Tensor
    norm: torch.Tensor
    convolution_taps: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class KDAWeights:
    input_projection: ttnn.Tensor
    decay_output_projection: ttnn.Tensor
    output_projection: ttnn.Tensor
    decay_scale_flat: ttnn.Tensor
    decay_bias_flat: ttnn.Tensor
    norm: ttnn.Tensor
    convolution_taps: tuple[ttnn.Tensor, ...]
    tensor_parallel_size: int
    tensor_parallel_axis: int

    @classmethod
    def check_cache_complete(
        cls,
        cache_path: Path | None,
        cache_name_prefix: str,
        config: KDAConfig,
        mesh_device: ttnn.Device | ttnn.MeshDevice,
        *,
        tensor_parallel_axis: int = 1,
        projection_dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> bool:
        """Return whether every dtype/layout/placement-specific KDA tensorbin exists."""
        if cache_path is None:
            return False
        cache_path = Path(cache_path)
        if not cache_path.is_dir():
            return False
        checker = FastCacheChecker(cache_path)
        mesh_shape, _ = _parallel_geometry(mesh_device, tensor_parallel_axis)
        for name in _cache_artifact_names(config):
            stem = _cache_stem(cache_name_prefix, name, config, mesh_shape, tensor_parallel_axis)
            pattern = f"{stem}{_cache_file_suffix(name, projection_dtype)}"
            if not checker.pattern_exists(pattern, "KDA"):
                return False
        return True

    @classmethod
    def build_ttnn_cache(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        cache_path: Path,
        cache_name_prefix: str,
        config: KDAConfig,
        mesh_device: ttnn.Device | ttnn.MeshDevice,
        *,
        tensor_parallel_axis: int = 1,
        projection_dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        """Build all KDA tensorbins without copying weights to device memory."""
        if not state_dict:
            raise ValueError("building the KDA TTNN cache requires a state_dict")
        mesh_shape, tensor_parallel_size = _validated_parallel_geometry(
            mesh_device,
            config,
            tensor_parallel_axis,
        )
        host_weights = _prepare_kda_host_weights(state_dict, config, tensor_parallel_size)
        _materialize_kda_weights(
            host_weights,
            device=mesh_device,
            config=config,
            tensor_cache_path=Path(cache_path),
            cache_name_prefix=cache_name_prefix,
            mesh_shape=mesh_shape,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_axis=tensor_parallel_axis,
            projection_dtype=projection_dtype,
            place_on_device=False,
        )

    @classmethod
    def from_cache(
        cls,
        cache_path: Path,
        cache_name_prefix: str,
        config: KDAConfig,
        mesh_device: ttnn.Device | ttnn.MeshDevice,
        *,
        tensor_parallel_axis: int = 1,
        projection_dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> "KDAWeights":
        """Construct device weights exclusively from a complete TTNN cache."""
        return load_kda_weights(
            mesh_device,
            config,
            None,
            cache_path,
            cache_name_prefix=cache_name_prefix,
            tensor_parallel_axis=tensor_parallel_axis,
            projection_dtype=projection_dtype,
        )


def _validated_parallel_geometry(
    device: ttnn.Device | ttnn.MeshDevice,
    config: KDAConfig,
    tensor_parallel_axis: int,
) -> tuple[tuple[int, int], int]:
    mesh_shape, tensor_parallel_size = _parallel_geometry(device, tensor_parallel_axis)
    if config.num_heads % tensor_parallel_size != 0:
        raise ValueError(
            f"num_heads {config.num_heads} must be divisible by tensor parallel size {tensor_parallel_size}"
        )
    return mesh_shape, tensor_parallel_size


def _group_projection_rows_by_tp_rank(
    weights: tuple[torch.Tensor, ...],
    tensor_parallel_size: int,
) -> torch.Tensor:
    """Place every projection's corresponding head slice next to the same TP rank."""
    grouped = []
    for device_index in range(tensor_parallel_size):
        rank_weights = []
        for weight in weights:
            shard_width = weight.shape[0] // tensor_parallel_size
            start = device_index * shard_width
            rank_weights.append(weight[start : start + shard_width])
        grouped.append(torch.cat(rank_weights, dim=0))
    return torch.cat(grouped, dim=0)


def _prepare_kda_host_weights(
    state_dict: Mapping[str, torch.Tensor],
    config: KDAConfig,
    tensor_parallel_size: int,
) -> _KDAHostWeights:
    state_dict = normalize_kda_state_dict(state_dict, config)
    common_input_weights = (
        state_dict["q_proj.weight"],
        state_dict["k_proj.weight"],
        state_dict["v_proj.weight"],
        state_dict["f_a_proj.weight"].repeat(tensor_parallel_size, 1),
    )
    if config.use_full_rank_gate:
        input_projection = _group_projection_rows_by_tp_rank(
            common_input_weights
            + (
                state_dict["g_proj.weight"],
                state_dict["b_proj.weight"],
            ),
            tensor_parallel_size,
        ).T
    else:
        output_gate_projection = state_dict["g_b_proj.weight"].reshape(
            config.num_heads,
            config.head_v_dim,
            config.head_v_dim,
        )
        output_gate_direct = torch.matmul(
            output_gate_projection,
            state_dict["g_a_proj.weight"],
        ).reshape(config.v_dim, config.hidden_size)
        input_projection = _group_projection_rows_by_tp_rank(
            common_input_weights
            + (
                output_gate_direct,
                state_dict["b_proj.weight"],
            ),
            tensor_parallel_size,
        ).T

    decay_scale = state_dict["A_log"].float().exp()
    if config.gate_lower_bound is None:
        decay_scale = -decay_scale
    decay_bias = state_dict["dt_bias"].float().reshape(1, 1, config.num_heads, config.head_k_dim)
    decay_scale_flat = decay_scale.expand(-1, -1, -1, config.head_k_dim).reshape(1, 1, config.q_dim)
    decay_bias_flat = decay_bias.reshape(1, 1, config.q_dim)

    convolution_taps = []
    for tap in range(config.conv_kernel_size):
        tap_weights = (
            state_dict["q_conv1d.weight"][:, 0, tap],
            state_dict["k_conv1d.weight"][:, 0, tap],
            state_dict["v_conv1d.weight"][:, 0, tap],
        )
        fused_tap = _group_projection_rows_by_tp_rank(tap_weights, tensor_parallel_size).reshape(1, 1, -1)
        convolution_taps.append(fused_tap)

    return _KDAHostWeights(
        input_projection=input_projection,
        decay_output_projection=state_dict["f_b_proj.weight"].T,
        output_projection=state_dict["o_proj.weight"].T,
        decay_scale_flat=decay_scale_flat,
        decay_bias_flat=decay_bias_flat,
        norm=state_dict["o_norm.weight"],
        convolution_taps=tuple(convolution_taps),
    )


def _mesh_mapper(
    device: ttnn.Device | ttnn.MeshDevice,
    *,
    mesh_shape: tuple[int, int],
    tensor_parallel_size: int,
    tensor_parallel_axis: int,
    shard_dim: int | None,
) -> ttnn.CppTensorToMesh | None:
    if tensor_parallel_size == 1:
        return None
    mesh_dims = [None, None]
    mesh_dims[tensor_parallel_axis] = shard_dim
    return ttnn.ShardTensor2dMesh(device, dims=tuple(mesh_dims), mesh_shape=mesh_shape)


def _materialize_kda_tensor(
    host_tensor: torch.Tensor | None,
    name: str,
    *,
    device: ttnn.Device | ttnn.MeshDevice,
    config: KDAConfig,
    tensor_cache_path: Path | None,
    cache_name_prefix: str,
    mesh_shape: tuple[int, int],
    tensor_parallel_size: int,
    tensor_parallel_axis: int,
    shard_dim: int | None = None,
    projection_dtype: ttnn.DataType,
    place_on_device: bool,
) -> ttnn.Tensor:
    cache_name = _cache_stem(cache_name_prefix, name, config, mesh_shape, tensor_parallel_axis)
    cache_file = tensor_cache_path / cache_name if tensor_cache_path is not None else None
    if host_tensor is None:
        if cache_file is None:
            raise ValueError("cache-only KDA weight loading requires tensor_cache_path")
        serialized_cache_file = Path(f"{cache_file}{_cache_file_suffix(name, projection_dtype)}")
        return ttnn.load_tensor(serialized_cache_file, device=device)

    return ttnn.as_tensor(
        host_tensor.contiguous(),
        dtype=_artifact_dtype(name, projection_dtype),
        layout=ttnn.TILE_LAYOUT,
        device=device if place_on_device else None,
        mesh_mapper=_mesh_mapper(
            device,
            mesh_shape=mesh_shape,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_axis=tensor_parallel_axis,
            shard_dim=shard_dim,
        ),
        memory_config=ttnn.DRAM_MEMORY_CONFIG if place_on_device else None,
        cache_file_name=cache_file,
    )


def _materialize_kda_weights(
    host_weights: _KDAHostWeights | None,
    *,
    device: ttnn.Device | ttnn.MeshDevice,
    config: KDAConfig,
    tensor_cache_path: Path | None,
    cache_name_prefix: str,
    mesh_shape: tuple[int, int],
    tensor_parallel_size: int,
    tensor_parallel_axis: int,
    projection_dtype: ttnn.DataType,
    place_on_device: bool,
) -> tuple[dict[str, ttnn.Tensor], tuple[ttnn.Tensor, ...]]:
    if tensor_cache_path is not None:
        tensor_cache_path.mkdir(parents=True, exist_ok=True)

    materialized = {
        "input_projection": _materialize_kda_tensor(
            None if host_weights is None else host_weights.input_projection,
            "input_projection_head_major",
            device=device,
            config=config,
            tensor_cache_path=tensor_cache_path,
            cache_name_prefix=cache_name_prefix,
            mesh_shape=mesh_shape,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_axis=tensor_parallel_axis,
            shard_dim=-1,
            projection_dtype=projection_dtype,
            place_on_device=place_on_device,
        ),
        "decay_output_projection": _materialize_kda_tensor(
            None if host_weights is None else host_weights.decay_output_projection,
            "decay_output_projection",
            device=device,
            config=config,
            tensor_cache_path=tensor_cache_path,
            cache_name_prefix=cache_name_prefix,
            mesh_shape=mesh_shape,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_axis=tensor_parallel_axis,
            shard_dim=-1,
            projection_dtype=projection_dtype,
            place_on_device=place_on_device,
        ),
        "output_projection": _materialize_kda_tensor(
            None if host_weights is None else host_weights.output_projection,
            "output_projection",
            device=device,
            config=config,
            tensor_cache_path=tensor_cache_path,
            cache_name_prefix=cache_name_prefix,
            mesh_shape=mesh_shape,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_axis=tensor_parallel_axis,
            shard_dim=-2,
            projection_dtype=projection_dtype,
            place_on_device=place_on_device,
        ),
        "decay_scale_flat": _materialize_kda_tensor(
            None if host_weights is None else host_weights.decay_scale_flat,
            "decay_scale_flat",
            device=device,
            config=config,
            tensor_cache_path=tensor_cache_path,
            cache_name_prefix=cache_name_prefix,
            mesh_shape=mesh_shape,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_axis=tensor_parallel_axis,
            shard_dim=-1,
            projection_dtype=projection_dtype,
            place_on_device=place_on_device,
        ),
        "decay_bias_flat": _materialize_kda_tensor(
            None if host_weights is None else host_weights.decay_bias_flat,
            "decay_bias_flat",
            device=device,
            config=config,
            tensor_cache_path=tensor_cache_path,
            cache_name_prefix=cache_name_prefix,
            mesh_shape=mesh_shape,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_axis=tensor_parallel_axis,
            shard_dim=-1,
            projection_dtype=projection_dtype,
            place_on_device=place_on_device,
        ),
        "norm": _materialize_kda_tensor(
            None if host_weights is None else host_weights.norm,
            "norm",
            device=device,
            config=config,
            tensor_cache_path=tensor_cache_path,
            cache_name_prefix=cache_name_prefix,
            mesh_shape=mesh_shape,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_axis=tensor_parallel_axis,
            projection_dtype=projection_dtype,
            place_on_device=place_on_device,
        ),
    }
    host_taps = [None] * config.conv_kernel_size if host_weights is None else host_weights.convolution_taps
    convolution_taps = tuple(
        _materialize_kda_tensor(
            tensor,
            f"conv_tap_{tap}",
            device=device,
            config=config,
            tensor_cache_path=tensor_cache_path,
            cache_name_prefix=cache_name_prefix,
            mesh_shape=mesh_shape,
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_axis=tensor_parallel_axis,
            shard_dim=-1,
            projection_dtype=projection_dtype,
            place_on_device=place_on_device,
        )
        for tap, tensor in enumerate(host_taps)
    )
    return materialized, convolution_taps


def _fold_decay_scale(
    projection: ttnn.Tensor, bias: ttnn.Tensor, scale: ttnn.Tensor
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Fold the bounded gate's per-head FP32 scale into the decay projection and bias on device.

    The bounded gate is lower_bound * sigmoid(scale * (x @ W + bias)) = lower_bound * sigmoid(x @ (scale W) + scale
    bias). The folded projection is rounded to BF16 once (it is a matmul weight); the folded offset stays FP32 and is
    applied by a separate FP32 add after the matmul (see ttKDA._compute_decay). Folding after the weights are loaded,
    rather than when they are prepared, keeps the cached tensors unscaled.
    """
    scale_row = ttnn.reshape(scale, (1, scale.shape[-1]))
    projection_fp32 = ttnn.typecast(projection, ttnn.float32)
    product = ttnn.multiply(projection_fp32, scale_row, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    folded_projection = ttnn.typecast(product, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    folded_bias = ttnn.multiply(bias, scale, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for tensor in (projection_fp32, product, projection, bias):
        ttnn.deallocate(tensor)
    return folded_projection, folded_bias


def load_kda_weights(
    device: ttnn.Device | ttnn.MeshDevice,
    config: KDAConfig,
    state_dict: Mapping[str, torch.Tensor] | None,
    tensor_cache_path: Path | None = None,
    *,
    cache_name_prefix: str = "kda",
    tensor_parallel_axis: int = 1,
    projection_dtype: ttnn.DataType = ttnn.bfloat16,
) -> KDAWeights:
    """Prepare KDA weights and materialize them on the target device."""
    if state_dict is not None and not state_dict:
        raise ValueError("state_dict must be non-empty when provided; use None for cache-only loading")
    mesh_shape, tensor_parallel_size = _validated_parallel_geometry(device, config, tensor_parallel_axis)
    cache_path = Path(tensor_cache_path) if tensor_cache_path is not None else None
    if state_dict is None:
        if not KDAWeights.check_cache_complete(
            cache_path,
            cache_name_prefix,
            config,
            device,
            tensor_parallel_axis=tensor_parallel_axis,
            projection_dtype=projection_dtype,
        ):
            raise FileNotFoundError(f"incomplete KDA TTNN cache for {cache_name_prefix!r} at {cache_path!r}")
        host_weights = None
    else:
        host_weights = _prepare_kda_host_weights(state_dict, config, tensor_parallel_size)

    materialized, convolution_taps = _materialize_kda_weights(
        host_weights,
        device=device,
        config=config,
        tensor_cache_path=cache_path,
        cache_name_prefix=cache_name_prefix,
        mesh_shape=mesh_shape,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_axis=tensor_parallel_axis,
        projection_dtype=projection_dtype,
        place_on_device=True,
    )
    if config.gate_lower_bound is not None:
        materialized["decay_output_projection"], materialized["decay_bias_flat"] = _fold_decay_scale(
            materialized["decay_output_projection"],
            materialized["decay_bias_flat"],
            materialized["decay_scale_flat"],
        )
    return KDAWeights(
        **materialized,
        convolution_taps=convolution_taps,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_axis=tensor_parallel_axis,
    )
