# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tensor utility functions for TTTv2 modules.
"""

import json

import torch

import ttnn


# todo)) add a on-device pad_dim_to_size function?
def pad_dim_to_size(x: "torch.Tensor", dim: int, size: int) -> "torch.Tensor":
    """Pads the specified dimension of the input tensor with zeros."""
    if dim < 0:
        dim = x.dim() + dim
    current_size = x.size(dim)
    pad_size = size - current_size

    if pad_size < 0:
        raise ValueError(f"Target size {size} is smaller than current size {current_size} on dim {dim}")

    if pad_size == 0:
        return x

    pad = [0] * (2 * x.dim())
    pad_index = 2 * (x.dim() - dim - 1)
    pad[pad_index + 1] = pad_size

    return torch.nn.functional.pad(x, pad, mode="constant", value=0)


def memory_config_to_dict(memory_config: ttnn.MemoryConfig):
    # Convert to plain types for deterministic serialization.
    return {
        "memory_layout": str(memory_config.memory_layout),
        "buffer_type": str(memory_config.buffer_type),
        "shard_spec": str(memory_config.shard_spec),
        "is_sharded": bool(memory_config.is_sharded()),
        "interleaved": bool(memory_config.interleaved),
        "hash": int(memory_config.__hash__()),
    }


def compute_kernel_config_to_str(compute_kernel_config: ttnn.WormholeComputeKernelConfig):
    # Backward compat shim; prefer compute_kernel_config_to_dict + serialize_config.
    cfg = compute_kernel_config_to_dict(compute_kernel_config)
    return serialize_config(cfg)


def compute_kernel_config_to_dict(compute_kernel_config: ttnn.WormholeComputeKernelConfig):
    return {
        "math_fidelity": str(compute_kernel_config.math_fidelity),
        "math_approx_mode": str(compute_kernel_config.math_approx_mode),
        "fp32_dest_acc_en": bool(compute_kernel_config.fp32_dest_acc_en),
        "packer_l1_acc": bool(compute_kernel_config.packer_l1_acc),
        "dst_full_sync_en": bool(compute_kernel_config.dst_full_sync_en),
        "throttle_level": compute_kernel_config.throttle_level,
    }


def program_config_to_str(program_config: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig):
    # Backward compat shim; prefer program_config_to_dict + serialize_config.
    cfg = program_config_to_dict(program_config)
    return serialize_config(cfg)


def program_config_to_dict(program_config: ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig):
    return {
        "in0_block_w": program_config.in0_block_w,
        "per_core_M": program_config.per_core_M,
        "per_core_N": program_config.per_core_N,
        "fused_activation": str(program_config.fused_activation),
    }


def serialize_config(cfg_dict: dict, fmt: str = "json") -> str:
    if fmt == "json":
        return json.dumps(cfg_dict, sort_keys=True)
    if fmt == "yaml":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required for yaml serialization") from exc
        return yaml.safe_dump(cfg_dict, sort_keys=True)
    raise ValueError(f"Unsupported format: {fmt}")
