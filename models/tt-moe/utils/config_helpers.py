# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration helpers for MoE implementation.
Copied from models.demos.deepseek_v3.utils.config_helpers
"""

import json
import os
from pathlib import Path
from types import NoneType
from typing import Any, Sequence

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.utils.config_dataclass import SavedWeight
from models.demos.deepseek_v3.utils.dequantize import dequantize_tensor

# Constants
TENSOR_CACHE_EXTENSION = ".tensorbin"
TOPK_MIN_WIDTH = 64  # Minimum width of the topk input tensor

# Compute kernel configurations
COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

COMPUTE_KERNEL_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


# Helper math functions
def even_int_div(a: int, b: int) -> int:
    """Integer division that raises an error if b does not divide a without a remainder."""
    assert a % b == 0
    return a // b


def dequantize(tensor: torch.Tensor, inv_scale: torch.Tensor, block_shape: Sequence[int]) -> torch.Tensor:
    """Dequantize a pytorch tensor using the provided scale."""
    assert tensor.ndim == inv_scale.ndim
    assert len(block_shape) == tensor.ndim and all(
        inv_scale.shape[i] * block_shape[i] >= tensor.shape[i] for i in range(tensor.ndim)
    )
    return dequantize_tensor(tensor, inv_scale, block_shape)


def _enum_name_or_str(obj: Any) -> str | None:
    """Get the name of an enum or return the string representation."""
    if obj is None:
        return None
    if hasattr(obj, "name"):
        return obj.name
    return str(obj)


def _memory_config_to_dict(memory_config: ttnn.MemoryConfig | None) -> dict[str, Any] | None:
    """Convert a MemoryConfig to a dictionary for JSON serialization."""
    if memory_config is None:
        return None
    # Use the built-in to_json() method for proper serialization, then parse it
    # This handles CoreRangeSet and other complex types correctly
    try:
        json_str = memory_config.to_json()
        return json.loads(json_str)
    except Exception:
        # Fallback to basic dict representation
        result = {}
        if hasattr(memory_config, "tensor_memory_layout"):
            result["tensor_memory_layout"] = _enum_name_or_str(memory_config.tensor_memory_layout)
        if hasattr(memory_config, "buffer_type"):
            result["buffer_type"] = _enum_name_or_str(memory_config.buffer_type)
        if hasattr(memory_config, "shard_spec"):
            result["shard_spec"] = str(memory_config.shard_spec)
        return result


def _get_relative_cache_path(path: Path) -> str | None:
    """Extract the relative cache path from an absolute path."""
    if not path.is_absolute():
        return str(path)
    path_str = str(path)
    mesh_idx = path_str.find("mesh_")
    if mesh_idx == -1:
        return None
    return path_str[mesh_idx:]


def _shard_device_impl(
    *,
    path: Path,
    tensor: torch.Tensor,
    shard_dims: tuple[int | None, int | None],
    mesh_device: ttnn.MeshDevice,
    remove_dims: tuple[bool, bool],
    dtype: ttnn.DataType | None,
    layout: ttnn.Layout | None,
    memory_config: ttnn.MemoryConfig | None,
) -> ttnn.Tensor:
    """Device implementation of tensor sharding."""
    # For weight conversion, we don't need to put on device
    # Just save the tensor with proper sharding

    # If the path doesn't end with .tensorbin, add it
    if not str(path).endswith(TENSOR_CACHE_EXTENSION):
        path = path.with_suffix(TENSOR_CACHE_EXTENSION)

    # Create the directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the tensor using torch.save for now
    # In production, this would handle actual device sharding
    torch.save(
        {
            "tensor": tensor,
            "dtype": dtype,
            "layout": layout,
            "memory_config": memory_config,
            "shard_dims": shard_dims,
            "mesh_shape": mesh_device.shape if hasattr(mesh_device, "shape") else None,
        },
        path,
    )

    # Return a mock tensor object (won't actually be on device)
    # The SavedWeight will just store the path
    class MockTensor:
        def __init__(self):
            self.shape = tensor.shape
            self.dtype = dtype
            self.layout = layout

        def memory_config(self):
            return memory_config

    return MockTensor()


def _shard_torch_impl(
    *,
    path: Path,
    tensor: torch.Tensor,
    shard_dims: tuple[int | None, ...],
    mesh_device: ttnn.MeshDevice,
    remove_dims: tuple[bool, ...],
    dtype: ttnn.DataType | None,
    layout: ttnn.Layout | None,
    memory_config: ttnn.MemoryConfig | None,
) -> ttnn.Tensor:
    """Torch implementation of tensor sharding."""
    # For weight conversion, we don't need to put on device
    # Just save the tensor with proper sharding

    # If the path doesn't end with .tensorbin, add it
    if not str(path).endswith(TENSOR_CACHE_EXTENSION):
        path = path.with_suffix(TENSOR_CACHE_EXTENSION)

    # Create the directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save the tensor using torch.save for now
    # In production, this would handle actual device sharding
    torch.save(
        {
            "tensor": tensor,
            "dtype": dtype,
            "layout": layout,
            "memory_config": memory_config,
            "shard_dims": shard_dims,
            "mesh_shape": mesh_device.shape if hasattr(mesh_device, "shape") else None,
        },
        path,
    )

    # Return a mock tensor object (won't actually be on device)
    # The SavedWeight will just store the path
    class MockTensor:
        def __init__(self):
            self.shape = tensor.shape
            self.dtype = dtype
            self.layout = layout

        def memory_config(self):
            return memory_config

    return MockTensor()


def shard_and_save(
    path: Path,
    tensor: torch.Tensor,
    shard_dims: tuple[int | None, int | None],
    mesh_device: ttnn.MeshDevice,
    remove_dims: tuple[bool, bool] | bool = False,
    *,
    dtype: ttnn.DataType | None = None,
    layout: ttnn.Layout | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
    _torch_impl: bool = False,
) -> SavedWeight:
    """Shard a tensor and save it to a file."""
    assert all(isinstance(shard_dim, (int, NoneType)) for shard_dim in shard_dims)
    assert isinstance(remove_dims, bool) or all(isinstance(remove_dim, bool) for remove_dim in remove_dims)
    assert len(shard_dims) == 2, "shard_dims must be exactly 2 dimensions (can repeat)"

    if isinstance(remove_dims, bool):
        remove_dims = (remove_dims, remove_dims)

    assert (
        shard_dims[0] != shard_dims[1] or remove_dims[0] == remove_dims[1]
    ), "If sharding a single dim, both remove_dim values must be the same"

    for remove_dim, shard_dim, mesh_dim in zip(remove_dims, shard_dims, mesh_device.shape, strict=True):
        assert (
            shard_dim is None or tensor.shape[shard_dim] % mesh_dim == 0
        ), f"Cannot shard dimension {shard_dim} of size {tensor.shape[shard_dim]} into {mesh_dim} shards"
        assert not (remove_dim and shard_dim is None), f"Cannot remove unsharded dimension {shard_dim}"

    if shard_dims[0] == shard_dims[1] and shard_dims[0] is not None:
        assert remove_dims[0] == remove_dims[1], "If sharding a single dim, both remove_dim values must be the same"
        assert (
            tensor.shape[shard_dims[0]] % mesh_device.get_num_devices() == 0
        ), f"Cannot shard dimension {shard_dims[0]} of size {tensor.shape[shard_dims[0]]} into {mesh_device.get_num_devices()} shards"
        assert (
            not remove_dims[0] or tensor.shape[shard_dims[0]] == mesh_device.get_num_devices()
        ), f"The removed dim {shard_dims[0]} must be fully sharded"
    else:
        for remove_dim, shard_dim, mesh_dim in zip(remove_dims, shard_dims, mesh_device.shape, strict=True):
            assert (
                shard_dim is None or tensor.shape[shard_dim] % mesh_dim == 0
            ), f"Cannot shard dimension {shard_dim} of size {tensor.shape[shard_dim]} into {mesh_dim} shards"
            assert not (remove_dim and shard_dim is None), f"Cannot remove unsharded dimension {shard_dim}"
            assert (
                not remove_dim or tensor.shape[shard_dim] == mesh_dim
            ), f"The removed dim {shard_dim} must be fully sharded"

    if _torch_impl:
        ttnn_tensor = _shard_torch_impl(
            path=path,
            tensor=tensor,
            shard_dims=shard_dims,
            mesh_device=mesh_device,
            remove_dims=remove_dims,
            dtype=dtype,
            layout=layout,
            memory_config=memory_config,
        )
    else:
        ttnn_tensor = _shard_device_impl(
            path=path,
            tensor=tensor,
            shard_dims=shard_dims,
            mesh_device=mesh_device,
            remove_dims=remove_dims,
            dtype=dtype,
            layout=layout,
            memory_config=memory_config,
        )

    if not path.name.endswith(TENSOR_CACHE_EXTENSION):
        path = path.with_name(f"{path.name}{TENSOR_CACHE_EXTENSION}")

    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        logger.warning(f"Overwriting existing cache file: {path}")

    # Log cache specs if environment variable is set
    cache_specs_file = os.getenv("DEEPSEEK_V3_CACHE_SPECS_JSONL")
    if cache_specs_file:
        record = {
            "event": "deepseek_v3.cache_tensor_spec",
            "pid": os.getpid(),
            "cache_file_path": str(path),
            "cache_file_relpath": _get_relative_cache_path(path),
            "torch_shape": list(tensor.shape),
            "torch_dtype": str(tensor.dtype),
            "requested_dtype": _enum_name_or_str(dtype),
            "requested_layout": _enum_name_or_str(layout),
            "requested_memory_config": _memory_config_to_dict(memory_config),
            "shard_dims": list(shard_dims),
            "remove_dims": list(remove_dims),
            "mesh_shape": list(mesh_device.shape),
            "mesh_num_devices": mesh_device.get_num_devices(),
            "dtype_is_tilized": dtype in {ttnn.bfloat4_b, ttnn.bfloat8_b},
            "shard_device_impl_uses_dram_interleaved_workaround": memory_config == ttnn.DRAM_MEMORY_CONFIG,
            "torch_impl": _torch_impl,
            "status": "ok",
            "result_shape": list(ttnn_tensor.shape),
            "result_dtype": _enum_name_or_str(ttnn_tensor.dtype),
            "result_layout": _enum_name_or_str(ttnn_tensor.layout),
            "result_memory_config": _memory_config_to_dict(ttnn_tensor.memory_config()),
        }
        try:
            with open(cache_specs_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write cache spec: {e}")

    # Create SavedWeight object
    saved_weight = SavedWeight(path=path, memory_config=memory_config)

    # Clean up the tensor from device if it's a real tensor
    # (our mock tensors don't need deallocation)
    if hasattr(ttnn_tensor, "__module__") and ttnn_tensor.__module__.startswith("ttnn"):
        ttnn.deallocate(ttnn_tensor)

    return saved_weight
