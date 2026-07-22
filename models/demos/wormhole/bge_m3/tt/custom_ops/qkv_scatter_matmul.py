# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixed-shape Python descriptor for BGE's S8192 QKV MinimalMatmul.

This is intentionally a parity scaffold: it mirrors the compiled
``minimal_matmul_factory_helper_common`` host factory while reusing its three
unmodified JIT kernel sources.  Real dataflow changes (direct Q/K/V scatter,
output-CB depth A/B) must not be added until this descriptor matches output,
device time, warm reuse, and trace replay.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import ttnn

_KERNEL_ROOT = "ttnn/cpp/ttnn/operations/experimental/minimal_matmul/device/kernels"
_IN0_KERNEL = f"{_KERNEL_ROOT}/dm_in0_sender.cpp"
_BGE_KERNEL_ROOT = "models/demos/wormhole/bge_m3/tt/custom_ops/fused_qkv_heads/kernels"
_IN1_OUT_KERNEL = f"{_BGE_KERNEL_ROOT}/writer_qkv_scatter.cpp"
_COMPUTE_KERNEL = f"{_BGE_KERNEL_ROOT}/compute_qkv_scatter.cpp"


# Matches minimal_matmul_program_factory.cpp CB IDs.
_CB_IN0 = 0
_CB_IN1 = 1
_CB_OUT = 2
_CB_INTERM = 3
_CB_BIAS = 4
_CB_OUT_K = 5
_CB_OUT_V = 6


_QKV_M_BLOCK = 16
_QKV_K_BLOCK = 8
_QKV_N_BLOCK = 4
_QKV_SUBBLOCK_H = 4
_QKV_SUBBLOCK_W = 2
_GRID_X = 8
_GRID_Y = 8


@dataclass(frozen=True)
class QKVScatterConfig:
    """Explicit block configuration for the fixed-shape QKV scatter kernel."""

    M_block_size: int = _QKV_M_BLOCK
    K_block_size: int = _QKV_K_BLOCK
    N_block_size: int = _QKV_N_BLOCK
    subblock_h: int = _QKV_SUBBLOCK_H
    subblock_w: int = _QKV_SUBBLOCK_W


def _tile_bytes(dtype: ttnn.DataType) -> int:
    return {
        ttnn.bfloat16: 2048,
        ttnn.bfloat8_b: 1088,
        ttnn.bfloat4_b: 576,
        ttnn.float32: 4096,
    }[dtype]


def _core_set(x0: int, y0: int, x1: int, y1: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def _physical(device, x: int, y: int) -> ttnn.CoreCoord:
    return device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))


def _axis_order(
    device,
    *,
    x: int,
    y: int,
    axis_length: int,
    axis: str,
    noc: ttnn.NOC,
) -> tuple[list[ttnn.CoreCoord], int]:
    """Mirror build_core_order_for_axis for the fixed transposed 8x8 grid."""
    assert axis in ("x", "y")
    values = [0]
    values.extend(range(1, axis_length) if noc == ttnn.NOC.NOC_0 else range(axis_length - 1, 0, -1))
    current = x if axis == "x" else y
    order = [_physical(device, value if axis == "x" else x, y if axis == "x" else value) for value in values]
    return order, values.index(current)


def _append_accessor_args(args: list[int], *tensors: ttnn.Tensor | None) -> None:
    for tensor in tensors:
        if tensor is not None:
            args.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())


def _cb(index: int, tiles: int, dtype: ttnn.DataType, cores: ttnn.CoreRangeSet) -> ttnn.CBDescriptor:
    tile_bytes = _tile_bytes(dtype)
    return ttnn.CBDescriptor(
        total_size=tiles * tile_bytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=tile_bytes)],
    )


def bge_qkv_scatter_matmul(
    input_tensor: ttnn.Tensor,
    weight_tensor: ttnn.Tensor,
    *,
    bias_tensor: ttnn.Tensor | None,
    memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
    config: QKVScatterConfig | None = None,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Run the exact B6/S8192 QKV matmul through a Python ProgramDescriptor.

    Required padded tile shape per DP shard: M=1536, K=32, N=96.  The output is
    ``[B, 1, S, 3072]`` with the same dtype/memory contract as MinimalMatmul.
    """
    input_shape = tuple(input_tensor.padded_shape)
    weight_shape = tuple(weight_tensor.padded_shape)
    if input_shape[-1] != 1024 or weight_shape[-2:] != (1024, 3072):
        raise ValueError(f"expected QKV K/N=(1024,3072), got input={input_shape}, weight={weight_shape}")
    if input_shape[-2] % 32 != 0:
        raise ValueError(f"M dimension must be tile aligned, got {input_shape[-2]}")

    device = input_tensor.device()
    output_shape = (*input_shape[:-1], 3072)
    head_shape = (input_shape[0], 16, input_shape[-2], 64)
    q_output = ttnn.allocate_tensor_on_device(ttnn.Shape(head_shape), dtype, ttnn.TILE_LAYOUT, device, memory_config)
    k_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(head_shape), ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device, memory_config
    )
    v_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(head_shape), ttnn.bfloat4_b, ttnn.TILE_LAYOUT, device, memory_config
    )

    # MinimalMatmul flattens every leading activation dimension into M.
    M = math.prod(input_shape[:-1]) // 32
    K = input_shape[-1] // 32
    N = output_shape[-1] // 32
    if (M, K, N) != (1536, 32, 96):
        raise ValueError(f"expected per-shard tile shape (1536,32,96), got {(M, K, N)}")

    config = config or QKVScatterConfig()
    M_block = config.M_block_size
    K_block = config.K_block_size
    N_block = config.N_block_size
    sub_h = config.subblock_h
    sub_w = config.subblock_w
    if min(M_block, K_block, N_block, sub_h, sub_w) <= 0:
        raise ValueError(f"QKV block sizes must be positive, got {config}")
    if M_block % sub_h or N_block % sub_w:
        raise ValueError(f"QKV subblock must divide the output block, got {config}")
    if sub_h * sub_w > 8:
        raise ValueError(f"QKV subblock exceeds the 8-tile DST capacity, got {config}")
    if M % M_block or K % K_block or N % N_block:
        raise ValueError(f"QKV block sizes must divide tile shape {(M, K, N)}, got {config}")
    in0_num_subblocks = M_block // sub_h
    in1_num_subblocks = N_block // sub_w
    in0_block_num_tiles = M_block * K_block
    in1_block_num_tiles = N_block * K_block
    out_block_num_tiles = M_block * N_block
    num_m_blocks = M // M_block
    num_n_blocks = N // N_block
    num_k_blocks = K // K_block
    batch_size = 1

    # M > N: the compiled factory transposes the logical multicast grid.
    transpose_core_grid = True
    in0_parallel_axis_cores = _GRID_X
    in1_parallel_axis_cores = _GRID_Y
    if num_m_blocks % in0_parallel_axis_cores or num_n_blocks % in1_parallel_axis_cores:
        raise ValueError("QKV block counts must divide uniformly over the fixed 8x8 grid")
    m_blocks_per_core = num_m_blocks // in0_parallel_axis_cores
    n_blocks_per_core = num_n_blocks // in1_parallel_axis_cores

    all_cores = _core_set(0, 0, _GRID_X - 1, _GRID_Y - 1)
    in0_sender_cores = _core_set(0, 0, _GRID_X - 1, 0)
    in0_receiver_cores = _core_set(0, 1, _GRID_X - 1, _GRID_Y - 1)
    in1_sender_cores = _core_set(0, 0, 0, _GRID_Y - 1)
    in1_receiver_cores = _core_set(1, 0, _GRID_X - 1, _GRID_Y - 1)

    input_dtype = input_tensor.dtype
    weight_dtype = weight_tensor.dtype
    bias_dtype = bias_tensor.dtype if bias_tensor is not None else dtype
    interm_dtype = ttnn.bfloat16
    out_dtype = dtype
    double_buffer_factor = 2

    cbs = [
        _cb(_CB_IN0, in0_block_num_tiles * double_buffer_factor, input_dtype, all_cores),
        _cb(_CB_IN1, in1_block_num_tiles * double_buffer_factor, weight_dtype, all_cores),
        _cb(_CB_OUT, out_block_num_tiles * double_buffer_factor, out_dtype, all_cores),
        _cb(_CB_INTERM, out_block_num_tiles, interm_dtype, all_cores),
        _cb(_CB_OUT_K, out_block_num_tiles * double_buffer_factor, ttnn.bfloat4_b, all_cores),
        _cb(_CB_OUT_V, out_block_num_tiles * double_buffer_factor, ttnn.bfloat4_b, all_cores),
    ]
    if bias_tensor is not None:
        cbs.append(_cb(_CB_BIAS, N_block, bias_dtype, all_cores))

    # IDs and initial values exactly match the compiled host factory.
    semaphores = [
        ttnn.SemaphoreDescriptor(id=0, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=1, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=2, core_ranges=all_cores, initial_value=1),
        ttnn.SemaphoreDescriptor(id=3, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=4, core_ranges=all_cores, initial_value=0),
        ttnn.SemaphoreDescriptor(id=5, core_ranges=all_cores, initial_value=1),
    ]

    in0_noc = ttnn.NOC.NOC_0
    in1_noc = ttnn.NOC.NOC_1
    kernel_defines = [("FUSE_BIAS", "1")] if bias_tensor is not None else []

    # Compile-time argument order mirrors minimal_matmul_factory_helper_common.
    in0_prefix = [
        M,
        M,
        K,
        K,
        N,
        N,
        M_block,
        K_block,
        N_block,
        m_blocks_per_core,
        n_blocks_per_core,
        _tile_bytes(input_dtype),
        _tile_bytes(out_dtype),
        _tile_bytes(bias_dtype),
        0,
        1,
        2,
        0,  # transposed grid: in1 writes output
    ]
    in0_sender_ct = [*in0_prefix, 1, 1, N, _tile_bytes(weight_dtype)]
    in0_receiver_ct = [*in0_prefix, 0, 1, N, _tile_bytes(weight_dtype)]
    _append_accessor_args(in0_sender_ct, input_tensor, q_output, bias_tensor)
    _append_accessor_args(in0_receiver_ct, input_tensor, q_output, bias_tensor)

    in1_prefix = [
        M,
        M,
        K,
        K,
        N,
        N,
        M_block,
        K_block,
        N_block,
        m_blocks_per_core,
        n_blocks_per_core,
        _tile_bytes(weight_dtype),
        _tile_bytes(out_dtype),
        _tile_bytes(bias_dtype),
        3,
        4,
        5,
        1,  # transposed grid: in1 writes output
    ]
    in1_sender_ct = [*in1_prefix, 1, 1, N]
    in1_receiver_ct = [*in1_prefix, 0, 1, N]
    in1_sender_ct.extend([32, 32, input_shape[-2] // 32, 16, 2, _tile_bytes(dtype), _tile_bytes(ttnn.bfloat4_b)])
    in1_receiver_ct.extend([32, 32, input_shape[-2] // 32, 16, 2, _tile_bytes(dtype), _tile_bytes(ttnn.bfloat4_b)])
    _append_accessor_args(in1_sender_ct, weight_tensor, q_output, k_output, v_output, bias_tensor)
    _append_accessor_args(in1_receiver_ct, weight_tensor, q_output, k_output, v_output, bias_tensor)

    compute_ct = [
        num_k_blocks,
        M_block,
        K_block,
        N_block,
        m_blocks_per_core,
        n_blocks_per_core,
        sub_h,
        sub_w,
        32,
        32,
    ]

    in0_rt: list[tuple[tuple[int, int], list[int]]] = []
    in1_rt: list[tuple[tuple[int, int], list[int]]] = []
    compute_rt: list[tuple[tuple[int, int], list[int]]] = []
    bias_addr = bias_tensor.buffer_address() if bias_tensor is not None else 0
    k_blocks_per_core = math.ceil(num_k_blocks / in1_parallel_axis_cores)
    max_defer_write_k_block = num_k_blocks - 1
    for x in range(_GRID_X):
        for y in range(_GRID_Y):
            # Transposed core grid: in0 multicasts down columns; in1 across rows.
            in0_order, in0_idx = _axis_order(
                device, x=x, y=y, axis_length=in1_parallel_axis_cores, axis="y", noc=in0_noc
            )
            in1_order, in1_idx = _axis_order(
                device, x=x, y=y, axis_length=in0_parallel_axis_cores, axis="x", noc=in1_noc
            )
            prev_in0 = in0_order[max(0, in0_idx - 1)]
            next_in0 = in0_order[min(len(in0_order) - 1, in0_idx + 1)]
            prev_in1 = in1_order[max(0, in1_idx - 1)]
            next_in1 = in1_order[min(len(in1_order) - 1, in1_idx + 1)]

            m_start = (M // in0_parallel_axis_cores) * x
            m_end = (M // in0_parallel_axis_cores) * (x + 1)
            n_start = (N // in1_parallel_axis_cores) * y
            n_end = (N // in1_parallel_axis_cores) * (y + 1)
            defer_write_k_block = min(y * k_blocks_per_core, num_k_blocks - 1)

            in0_rt.append(
                (
                    (x, y),
                    [
                        input_tensor.buffer_address(),
                        bias_addr,
                        0,
                        int(in0_idx == len(in0_order) - 1),
                        next_in0.x,
                        next_in0.y,
                        prev_in0.x,
                        prev_in0.y,
                        m_start,
                        m_end,
                        n_start,
                        n_end,
                        defer_write_k_block,
                        max_defer_write_k_block,
                        q_output.buffer_address(),
                    ],
                )
            )
            in1_rt.append(
                (
                    (x, y),
                    [
                        weight_tensor.buffer_address(),
                        bias_addr,
                        int(in1_idx == len(in1_order) - 1),
                        next_in1.x,
                        next_in1.y,
                        prev_in1.x,
                        prev_in1.y,
                        m_start,
                        m_end,
                        n_start,
                        n_end,
                        defer_write_k_block,
                        max_defer_write_k_block,
                        q_output.buffer_address(),
                        k_output.buffer_address(),
                        v_output.buffer_address(),
                    ],
                )
            )
            compute_rt.append(((x, y), [m_start, m_end, n_start, n_end]))

    in0_sender_rt = [entry for entry in in0_rt if entry[0][1] == 0]
    in0_receiver_rt = [entry for entry in in0_rt if entry[0][1] != 0]
    in1_sender_rt = [entry for entry in in1_rt if entry[0][0] == 0]
    in1_receiver_rt = [entry for entry in in1_rt if entry[0][0] != 0]

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=_IN0_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=in0_sender_cores,
            compile_time_args=in0_sender_ct,
            runtime_args=in0_sender_rt,
            defines=kernel_defines,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=in0_noc),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_IN0_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=in0_receiver_cores,
            compile_time_args=in0_receiver_ct,
            runtime_args=in0_receiver_rt,
            defines=kernel_defines,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=in0_noc),
        ),
        ttnn.KernelDescriptor(
            kernel_source=_IN1_OUT_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=in1_sender_cores,
            compile_time_args=in1_sender_ct,
            runtime_args=in1_sender_rt,
            defines=kernel_defines,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=in1_noc),
            compiler_include_paths=[_KERNEL_ROOT],
        ),
        ttnn.KernelDescriptor(
            kernel_source=_IN1_OUT_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=in1_receiver_cores,
            compile_time_args=in1_receiver_ct,
            runtime_args=in1_receiver_rt,
            defines=kernel_defines,
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=in1_noc),
            compiler_include_paths=[_KERNEL_ROOT],
        ),
        ttnn.KernelDescriptor(
            kernel_source=_COMPUTE_KERNEL,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            defines=kernel_defines,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        ),
    ]

    descriptor = ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=semaphores)
    io_tensors = [input_tensor, weight_tensor, q_output, k_output, v_output]
    if bias_tensor is not None:
        io_tensors.append(bias_tensor)
    ttnn.generic_op(io_tensors, descriptor)
    return q_output, k_output, v_output
