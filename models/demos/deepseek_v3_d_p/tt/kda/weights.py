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
from models.demos.deepseek_v3_d_p.tt.kda.weight_schema import normalize_kda_state_dict
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import FastCacheChecker

_CACHE_SCHEMA_VERSION = 2


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
            pattern = f"{stem}_dtype_{ttnn.bfloat16.name}_layout_{ttnn.TILE_LAYOUT.name}.tensorbin"
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
    ) -> None:
        """Build all KDA tensorbins without copying weights to device memory."""
        result = load_kda_weights(
            mesh_device,
            config,
            state_dict,
            cache_path,
            cache_name_prefix=cache_name_prefix,
            tensor_parallel_axis=tensor_parallel_axis,
            _load_to_device=False,
        )
        assert result is None

    @classmethod
    def from_cache(
        cls,
        cache_path: Path,
        cache_name_prefix: str,
        config: KDAConfig,
        mesh_device: ttnn.Device | ttnn.MeshDevice,
        *,
        tensor_parallel_axis: int = 1,
    ) -> "KDAWeights":
        """Construct device weights exclusively from a complete TTNN cache."""
        result = load_kda_weights(
            mesh_device,
            config,
            None,
            cache_path,
            cache_name_prefix=cache_name_prefix,
            tensor_parallel_axis=tensor_parallel_axis,
        )
        assert result is not None
        return result


def load_kda_weights(
    device: ttnn.Device | ttnn.MeshDevice,
    config: KDAConfig,
    state_dict: Mapping[str, torch.Tensor] | None,
    tensor_cache_path: Path | None = None,
    *,
    cache_name_prefix: str = "kda",
    tensor_parallel_axis: int = 1,
    _load_to_device: bool = True,
) -> KDAWeights | None:
    """Fuse compatible projections and place whole-head shards on device."""
    mesh_shape, tensor_parallel_size = _parallel_geometry(device, tensor_parallel_axis)
    if state_dict is not None and not state_dict:
        state_dict = None
    if config.num_heads % tensor_parallel_size != 0:
        raise ValueError(
            f"num_heads {config.num_heads} must be divisible by tensor parallel size {tensor_parallel_size}"
        )
    if not state_dict and not _load_to_device:
        raise ValueError("building the KDA TTNN cache requires a state_dict")
    if state_dict:
        state_dict = normalize_kda_state_dict(state_dict, config)
    elif _load_to_device and not KDAWeights.check_cache_complete(
        tensor_cache_path,
        cache_name_prefix,
        config,
        device,
        tensor_parallel_axis=tensor_parallel_axis,
    ):
        raise FileNotFoundError(f"incomplete KDA TTNN cache for {cache_name_prefix!r} at {tensor_cache_path!r}")
    if tensor_cache_path is not None:
        tensor_cache_path = Path(tensor_cache_path)
        tensor_cache_path.mkdir(parents=True, exist_ok=True)

    def device_tensor(
        tensor: torch.Tensor | None,
        name: str,
        *,
        dtype: ttnn.DataType = ttnn.bfloat16,
        shard_dim: int | None = None,
    ) -> ttnn.Tensor:
        cache_name = _cache_stem(cache_name_prefix, name, config, mesh_shape, tensor_parallel_axis)
        cache_file = tensor_cache_path / cache_name if tensor_cache_path is not None else None
        mesh_mapper = None
        if tensor_parallel_size > 1:
            mesh_dims = [None, None]
            mesh_dims[tensor_parallel_axis] = shard_dim
            mesh_mapper = ttnn.ShardTensor2dMesh(device, dims=tuple(mesh_dims), mesh_shape=mesh_shape)
        if state_dict is None:
            assert cache_file is not None
            serialized_cache_file = Path(f"{cache_file}_dtype_{dtype.name}_layout_{ttnn.TILE_LAYOUT.name}.tensorbin")
            return ttnn.load_tensor(serialized_cache_file, device=device)
        assert tensor is not None
        converted = ttnn.as_tensor(
            tensor.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device if _load_to_device else None,
            mesh_mapper=mesh_mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if _load_to_device else None,
            cache_file_name=cache_file,
        )
        return converted

    def group_output_shards(*weights: torch.Tensor) -> torch.Tensor:
        """Group every projection tensor corresponding head slice on the same device."""
        grouped = []
        for device_index in range(tensor_parallel_size):
            device_weights = []
            for weight in weights:
                shard_width = weight.shape[0] // tensor_parallel_size
                start = device_index * shard_width
                device_weights.append(weight[start : start + shard_width])
            grouped.append(torch.cat(device_weights, dim=0))
        return torch.cat(grouped, dim=0)

    if state_dict:
        common_input_weights = (
            state_dict["q_proj.weight"],
            state_dict["k_proj.weight"],
            state_dict["v_proj.weight"],
            state_dict["f_a_proj.weight"].repeat(tensor_parallel_size, 1),
        )
        if config.use_full_rank_gate:
            input_projection = group_output_shards(
                *common_input_weights,
                state_dict["g_proj.weight"],
                state_dict["b_proj.weight"],
            ).T
        else:
            output_gate_projection = state_dict["g_b_proj.weight"].reshape(
                config.num_heads, config.head_v_dim, config.head_v_dim
            )
            output_gate_direct = torch.matmul(
                output_gate_projection,
                state_dict["g_a_proj.weight"],
            ).reshape(config.v_dim, config.hidden_size)
            input_projection = group_output_shards(
                *common_input_weights,
                output_gate_direct,
                state_dict["b_proj.weight"],
            ).T

        decay_scale = state_dict["A_log"].float().exp()
        if config.gate_lower_bound is None:
            decay_scale = -decay_scale
        decay_bias = state_dict["dt_bias"].reshape(1, 1, config.num_heads, config.head_k_dim)
        decay_scale_flat = decay_scale.expand(-1, -1, -1, config.head_k_dim).reshape(1, 1, config.q_dim)
        decay_bias_flat = decay_bias.reshape(1, 1, config.q_dim)
        decay_output_projection = state_dict["f_b_proj.weight"].T
        output_projection = state_dict["o_proj.weight"].T
        norm = state_dict["o_norm.weight"]
        convolution_host_taps = []
        for tap in range(config.conv_kernel_size):
            tap_weights = (
                state_dict["q_conv1d.weight"][:, 0, tap],
                state_dict["k_conv1d.weight"][:, 0, tap],
                state_dict["v_conv1d.weight"][:, 0, tap],
            )
            fused_tap = group_output_shards(*tap_weights).reshape(1, 1, -1)
            convolution_host_taps.append(fused_tap)
    else:
        input_projection = None
        decay_output_projection = None
        output_projection = None
        decay_scale_flat = None
        decay_bias_flat = None
        norm = None
        convolution_host_taps = [None] * config.conv_kernel_size

    converted = {
        "input_projection": device_tensor(input_projection, "input_projection_head_major", shard_dim=-1),
        "decay_output_projection": device_tensor(decay_output_projection, "decay_output_projection", shard_dim=-1),
        "output_projection": device_tensor(output_projection, "output_projection", shard_dim=-2),
        "decay_scale_flat": device_tensor(decay_scale_flat, "decay_scale_flat", shard_dim=-1),
        "decay_bias_flat": device_tensor(decay_bias_flat, "decay_bias_flat", shard_dim=-1),
        "norm": device_tensor(norm, "norm"),
    }
    convolution_taps = tuple(
        device_tensor(tensor, f"conv_tap_{tap}", shard_dim=-1) for tap, tensor in enumerate(convolution_host_taps)
    )
    if not _load_to_device:
        return None
    return KDAWeights(
        **converted,
        convolution_taps=convolution_taps,
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_axis=tensor_parallel_axis,
    )
