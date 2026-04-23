# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Literal

import ttnn

PaddingType1D = int | tuple[int, int] | Literal["same"]
PaddingType2D = int | tuple[int, int] | tuple[int, int, int, int] | Literal["same"]


@dataclass(frozen=True)
class ConvConfiguration:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int
    padding: tuple[int, int]
    dilation: int
    groups: int
    activation: ttnn.UnaryWithParam | None = None
    weights_dtype: ttnn.DataType = ttnn.bfloat16
    dtype: ttnn.DataType = ttnn.bfloat16
    output_layout: ttnn.Layout = ttnn.TILE_LAYOUT
    deallocate_input: bool = False
    # enable_act_double_buffer: bool = False
    # enable_weights_double_buffer: bool = False
    # deallocate_activation: bool = True
    # reallocate_halo_output: bool = True
    # config_tensors_in_dram: bool = True


def _bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def _format_memory_view(view: ttnn._ttnn.device.MemoryView, label: str) -> str:
    total_bytes = view.total_bytes_per_bank * view.num_banks
    allocated_bytes = view.total_bytes_allocated_per_bank * view.num_banks
    free_bytes = view.total_bytes_free_per_bank * view.num_banks
    percent_used = (allocated_bytes / total_bytes * 100.0) if total_bytes else 0.0
    total_mib = _bytes_to_mib(total_bytes)
    allocated_mib = _bytes_to_mib(allocated_bytes)
    free_mib = _bytes_to_mib(free_bytes)
    allocated_per_bank_mib = _bytes_to_mib(view.total_bytes_allocated_per_bank)
    free_per_bank_mib = _bytes_to_mib(view.total_bytes_free_per_bank)
    largest_contig_mib = _bytes_to_mib(view.largest_contiguous_bytes_free_per_bank)
    per_bank_mib = _bytes_to_mib(view.total_bytes_per_bank)
    return (
        f"{label} usage: {allocated_mib:.2f} / {total_mib:.2f} MiB "
        f"({percent_used:.2f}%), free={free_mib:.2f} MiB, "
        f"largest_contiguous_free_per_bank={largest_contig_mib:.2f} MiB, "
        f"banks={view.num_banks}, per_bank={per_bank_mib:.2f} MiB, "
        f"allocated_per_bank={allocated_per_bank_mib:.2f} MiB, "
        f"free_per_bank={free_per_bank_mib:.2f} MiB"
    )


def dump_ttnn_meminfo(mesh_device: ttnn.MeshDevice, header: str = "") -> None:
    """Dump DRAM memory usage of the mesh device to the log."""
    dram_view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    label = f"DRAM ({header})" if header else "DRAM"
    print(f"dram info: {label}")
    print(_format_memory_view(dram_view, label))

    l1_view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1)
    label = f"L1 ({header})" if header else "L1"
    print(f"L1 info: {label}")
    print(_format_memory_view(l1_view, label))


def _normalize_conv2d_activation(activation: str | tuple[str, dict] | None) -> ttnn.UnaryWithParam | None:
    if activation is None:
        return None
    if isinstance(activation, tuple):
        activation_name, kwargs = activation
    else:
        activation_name = activation.strip().lower()

    if activation_name == "relu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
    if activation_name == "silu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)
    if activation_name == "gelu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU)
    if activation_name == "sigmoid":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.SIGMOID)
    if activation_name == "tanh":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.TANH)
    if activation_name == "leaky_relu":
        return ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, kwargs["negative_slope"])

    supported = "relu, silu, gelu, sigmoid, tanh, leaky_relu"
    raise ValueError(f"Unsupported conv activation '{activation}'. Supported activations: {supported}")


def input_shape_to_memory_config(
    input_shape, output_length, in_channels, kernel_size, device: ttnn.MeshDevice
) -> ttnn.MemoryConfig:
    batch_size, input_height, input_width, in_channels = input_shape
    memory_cost = batch_size * input_height * input_width * in_channels * 2  # assuming bfloat16, so 2 bytes per element
    if (output_length, in_channels, kernel_size) in dims_to_num_slices:
        return ttnn.DRAM_MEMORY_CONFIG

    # Keep tiny-channel inputs interleaved to avoid expensive/invalid sharding setups.

    if memory_cost > 64 * 1_400_000:  # if input is larger than 1.4MB, use DRAM to avoid L1 thrashing
        return ttnn.DRAM_MEMORY_CONFIG
    if in_channels < 16:
        return ttnn.DRAM_MEMORY_CONFIG

    nhw = batch_size * input_height * input_width

    # Use best sharding strategy based on NHW-to-C ratio:
    # - HEIGHT_SHARDED if NHW >> C
    # - WIDTH_SHARDED if C >> NHW
    # - BLOCK_SHARDED if NHW ~= C
    if nhw >= 4 * in_channels:
        strategy = ttnn.ShardStrategy.HEIGHT
    elif in_channels >= 4 * nhw:
        strategy = ttnn.ShardStrategy.WIDTH
    else:
        strategy = ttnn.ShardStrategy.BLOCK

    grid_size = device.compute_with_storage_grid_size()
    candidate_grids = [
        ttnn.CoreGrid(y=grid_size.y, x=grid_size.x),
        ttnn.CoreGrid(y=grid_size.y, x=1),
        ttnn.CoreGrid(y=1, x=grid_size.x),
        ttnn.CoreGrid(y=1, x=1),
    ]

    for core_grid in candidate_grids:
        try:
            return ttnn.create_sharded_memory_config_(
                input_shape,
                core_grid=core_grid,
                strategy=strategy,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            )
        except RuntimeError:
            continue

    return ttnn.DRAM_MEMORY_CONFIG


def get_shard_strategy_for_conv(input_shape):
    # handle 1d and 2d input cases
    nhw = input_shape[0] * input_shape[1] if len(input_shape) == 3 else input_shape[0] * input_shape[1] * input_shape[2]
    in_channels = input_shape[2] if len(input_shape) == 3 else input_shape[3]
    if nhw >= 4 * in_channels:
        return ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    elif in_channels >= 4 * nhw:
        return ttnn.TensorMemoryLayout.WIDTH_SHARDED
    else:
        return ttnn.TensorMemoryLayout.BLOCK_SHARDED


def resolve_padding_1d(
    padding: PaddingType1D,
    kernel_size: int,
    stride: int,
    dilation: int,
) -> tuple[int, int]:
    if isinstance(padding, str):
        if padding != "same":
            raise ValueError(f"Unsupported padding mode: {padding}")
        if stride != 1:
            raise ValueError("Only stride=1 is supported for 'same' padding")
        padding_needed = dilation * (kernel_size - 1)
        left_padding = padding_needed // 2
        right_padding = padding_needed - left_padding
        return (left_padding, right_padding)

    if isinstance(padding, tuple):
        if len(padding) != 2:
            raise ValueError(f"padding tuple must contain 2 values, got {len(padding)}")
        return (padding[0], padding[1])

    return (padding, padding)


def resolve_padding_2d(
    padding: PaddingType2D,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    dilation: int | tuple[int, int],
) -> tuple[int, int] | tuple[int, int, int, int]:
    kernel_h, kernel_w = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    stride_h, stride_w = stride if isinstance(stride, tuple) else (stride, stride)
    dilation_h, dilation_w = dilation if isinstance(dilation, tuple) else (dilation, dilation)

    if isinstance(padding, str):
        if padding != "same":
            raise ValueError(f"Unsupported padding mode: {padding}")
        if stride_h != 1 or stride_w != 1:
            raise ValueError("Only stride=1 is supported for 'same' padding")
        pad_h_needed = dilation_h * (kernel_h - 1)
        pad_w_needed = dilation_w * (kernel_w - 1)
        pad_top = pad_h_needed // 2
        pad_bottom = pad_h_needed - pad_top
        pad_left = pad_w_needed // 2
        pad_right = pad_w_needed - pad_left
        return (pad_top, pad_bottom, pad_left, pad_right)
    if isinstance(padding, int):
        return (padding, padding)
    return padding
