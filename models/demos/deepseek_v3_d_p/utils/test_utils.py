# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import torch
from loguru import logger

import ttnn


def print_buffers(device, name, buffer_type):
    buffers = ttnn._ttnn.reports.get_buffers(device)
    filtered_buffers = [buf for buf in buffers if buf.buffer_type == buffer_type]
    for i, buf in enumerate(filtered_buffers):
        logger.warning(
            f"{buffer_type} [{name}] Buffer {i}: addr={buf.address}, "
            f"size={buf.max_size_per_bank}, layout={buf.buffer_layout}"
        )


def print_l1_buffers(device, name):
    print_buffers(device, name, ttnn.BufferType.L1)


def print_l1_small_buffers(device, name):
    print_buffers(device, name, ttnn.BufferType.L1_SMALL)


def adjust_shapes_for_testing(config, mesh_device):
    """Scale TP dimension for smaller meshes. sp_dim (per-device seq len) is always correct."""
    _, n_tp_devices = mesh_device.shape
    if n_tp_devices != 4:
        config.dim = config.dim // (4 // n_tp_devices)


def get_input_mem_config(config, mesh_shape):
    shard_height = (config.sp_dim + config.num_cores - 1) // config.num_cores
    shard_height = ((shard_height + 31) // 32) * 32
    shard_width = (config.dim + mesh_shape[1] - 1) // mesh_shape[1]
    return ttnn.create_sharded_memory_config(
        shape=(shard_height, shard_width),
        core_grid=config.core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def save_norm_output(
    norm_tensor: torch.Tensor,
    test_params: dict,
    output_dir: Optional[Union[str, Path]] = None,
):
    """
    Save final norm output to timestamped .pt file.

    Args:
        norm_tensor: Final norm output tensor
        test_params: Dict with all test parameters (mesh_shape, isl_total, num_layers, etc.)
        output_dir: Output directory (default: /tmp/norm_outputs or NORM_OUTPUT_DIR env var)
    """
    # Get output directory
    if output_dir is None:
        output_dir = Path(os.getenv("NORM_OUTPUT_DIR", "/tmp/norm_outputs"))
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp and params
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize string values
    input_source = test_params.get("input_source", "unknown").replace("/", "_").replace(".", "_")
    gate_mode = test_params.get("gate_fallback_mode")
    gate_str = gate_mode.value if hasattr(gate_mode, "value") else str(gate_mode)

    # Build filename
    mesh_shape = test_params["mesh_shape"]
    filename = (
        f"norm_{timestamp}_"
        f"mesh{mesh_shape[0]}x{mesh_shape[1]}_"
        f"isl{test_params['isl_total']}_"
        f"L{test_params['num_layers']}_"
        f"e{test_params['n_routed_experts']}_"
        f"cf{test_params['capacity_factor']}_"
        f"gate{gate_str}_"
        f"{'pretrained' if test_params.get('use_pretrained') else 'random'}_"
        f"{input_source}.pt"
    )

    save_path = output_dir / filename

    # Save tensor with metadata
    torch.save(
        {
            "norm_output": norm_tensor,
            "metadata": test_params,
        },
        save_path,
    )

    logger.info(f"✓ Saved final norm output to: {save_path}")
    logger.info(f"  Shape: {norm_tensor.shape}, Mean: {norm_tensor.mean():.6f}, Std: {norm_tensor.std():.6f}")

    return save_path
