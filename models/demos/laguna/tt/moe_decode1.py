# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batch-1 decode routed experts as two generic_op programs (Laguna, per chip).

The stock batch-1 path runs the packed gate|up sparse_matmul, two slices, silu(gate)*up, the routing-weight
multiply, a zero-filled down sparse_matmul and an expert sum, all over the 64 local experts although only the
routed ones (0-10 per chip, ~2.5 on average) are real, and one token is a 32-row tile. Here:

  gate_up_swiglu: core (n-tile, slot group) reads row 0 of the activation once and, for each active expert in its
    slot group, its gate and up weight columns; DST computes silu(gate) * up * routing weight and writes the tile
    to [1, E, 32, I] at (expert, n-tile). Inactive experts' tiles are never written.
  down_sum: core n-tile accumulates every active expert's activation row times its down-weight column into one
    output tile of [1, 1, 1, H] (a zero tile when no local expert is active).

Active experts and routing weights are read on device from the sparsity row, so both programs are trace-safe.
"""

import os
from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
SLOTS = 4  # active experts per gate/up core
MAX_ACTIVE = 10  # Laguna top-10: at most 10 routed experts land on one chip


def _tile_bytes(dtype):
    return {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.bfloat4_b: 576}[dtype]


def _accessor_args(*tensors):
    args = []
    for t in tensors:
        args.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    return args


def _cb(grid, index, dtype, page, pages):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
    )


def _compute_config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.LoFi
    # fp32 DST: the gate/up dot products sum 96 tiles and the down sum runs over every active expert in one DST
    # tile (bf16 accumulation there cost ~.004 PCC on the AIME teacher-forced test); 2 DST tiles are enough.
    cfg.fp32_dest_acc_en = os.environ.get("TT_LAGUNA_MOE1_FP32_DST", "1") == "1"
    cfg.math_approx_mode = False
    return cfg


def gate_up_swiglu(x, w_gate_up, sparsity, memory_config=ttnn.L1_MEMORY_CONFIG):
    """x [1, 1, 1, H] bf16 TILE interleaved; w_gate_up [1, E, H, 2I] (gate | up per row); sparsity [1, 1, 1, E] bf16
    ROW_MAJOR routing weights. Returns [1, E, 32, I] bf16 (row 0 of each active expert's tiles is real)."""
    device = x.device()
    E = w_gate_up.shape[1]
    kt = x.padded_shape[-1] // TILE
    nt = w_gate_up.padded_shape[-1] // TILE // 2
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, E, TILE, nt * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config
    )
    slot_groups = -(-MAX_ACTIVE // SLOTS)
    grid_size = device.compute_with_storage_grid_size()
    num_cores = nt * slot_groups
    assert num_cores <= grid_size.x * grid_size.y, (num_cores, grid_size)
    grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, True)
    chunk = int(os.environ.get("TT_LAGUNA_MOE1_CHUNK", "32"))
    assert kt % chunk == 0, kt
    x_page, w_page, sp_page = _tile_bytes(x.dtype), _tile_bytes(w_gate_up.dtype), max(E * 2, 64)
    cbs = [
        _cb(grid, 0, x.dtype, x_page, kt),
        _cb(grid, 1, w_gate_up.dtype, w_page, 2 * chunk),
        _cb(grid, 2, ttnn.uint32, 64, 1),
        _cb(grid, 3, ttnn.bfloat16, sp_page, 1),
        _cb(grid, 4, ttnn.bfloat16, sp_page, 1),
        _cb(grid, 16, ttnn.bfloat16, 2048, 2),
    ]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "moe1_gu_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[kt, nt, E, chunk, x_page, w_page, sp_page, grid_size.x, slot_groups, SLOTS]
        + _accessor_args(x, w_gate_up, sparsity),
        common_runtime_args=[x.buffer_address(), w_gate_up.buffer_address(), sparsity.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "moe1_gu_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[nt, E, 2048, sp_page, grid_size.x, slot_groups, SLOTS] + _accessor_args(out, sparsity),
        common_runtime_args=[out.buffer_address(), sparsity.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "moe1_gu_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[kt, chunk, SLOTS],
        config=_compute_config(),
    )
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op([x, w_gate_up, sparsity, out], program)
    return out


def down_sum(glu, w_down, sparsity, memory_config=ttnn.L1_MEMORY_CONFIG):
    """glu [1, E, 32, I] (gate_up_swiglu output); w_down [1, E, I, H]; sparsity as above. Returns [1, 1, 1, H] bf16:
    the routing-weighted sum of the active experts' down projections."""
    device = glu.device()
    E = w_down.shape[1]
    kt = w_down.padded_shape[-2] // TILE
    nt = w_down.padded_shape[-1] // TILE
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 1, nt * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config
    )
    grid_size = device.compute_with_storage_grid_size()
    assert nt <= grid_size.x * grid_size.y, (nt, grid_size)
    grid = ttnn.num_cores_to_corerangeset(nt, grid_size, True)
    x_page, w_page, sp_page = _tile_bytes(glu.dtype), _tile_bytes(w_down.dtype), max(E * 2, 64)
    cbs = [
        _cb(grid, 0, glu.dtype, x_page, 2 * kt),
        _cb(grid, 1, w_down.dtype, w_page, 2 * kt),
        _cb(grid, 2, ttnn.uint32, 64, 1),
        _cb(grid, 3, ttnn.bfloat16, sp_page, 1),
        _cb(grid, 4, ttnn.bfloat16, sp_page, 1),
        _cb(grid, 5, ttnn.bfloat16, 2048, 1),
        _cb(grid, 16, ttnn.bfloat16, 2048, 1),
    ]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "moe1_down_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[kt, nt, E, x_page, w_page, sp_page, grid_size.x, MAX_ACTIVE]
        + _accessor_args(glu, w_down, sparsity),
        common_runtime_args=[glu.buffer_address(), w_down.buffer_address(), sparsity.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "moe1_down_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[2048, grid_size.x, E, sp_page] + _accessor_args(out, sparsity),
        common_runtime_args=[out.buffer_address(), sparsity.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "moe1_down_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[kt],
        config=_compute_config(),
    )
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op([glu, w_down, sparsity, out], program)
    return out


def swiglu32(gu, wv, sparsity, memory_config=ttnn.L1_MEMORY_CONFIG):
    """Batched decode routed SwiGLU over only the active experts: gu [1, E, 32, 2I] packed gate|up sparse_matmul
    output, wv [1, E, 32, 1] per-(expert, token) routing weights, sparsity [1, 1, 1, E] bf16 ROW_MAJOR union.
    Returns [1, E, 32, I] = silu(gate) * up * w for active experts (inactive experts' tiles unwritten; the down
    sparse_matmul skips them)."""
    device = gu.device()
    E = gu.shape[1]
    nt = gu.padded_shape[-1] // TILE // 2
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, E, gu.shape[2], nt * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config
    )
    grid_size = device.compute_with_storage_grid_size()
    num_cores = int(os.environ.get("TT_LAGUNA_SW32_CORES", grid_size.x * grid_size.y))
    grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, True)
    page, sp_page = 2048, max(E * 2, 64)
    cbs = [
        _cb(grid, 0, ttnn.bfloat16, page, 2),
        _cb(grid, 1, ttnn.bfloat16, page, 2),
        _cb(grid, 2, ttnn.bfloat16, page, 2),
        _cb(grid, 3, ttnn.uint32, 64, 1),
        _cb(grid, 4, ttnn.bfloat16, sp_page, 1),
        _cb(grid, 5, ttnn.bfloat16, sp_page, 1),
        _cb(grid, 6, ttnn.uint32, E * 4, 1),
        _cb(grid, 7, ttnn.uint32, E * 4, 1),
        _cb(grid, 16, ttnn.bfloat16, page, 2),
    ]
    ct = [nt, E, page, sp_page, grid_size.x, num_cores]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "swiglu32_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=ct + _accessor_args(gu, wv, sparsity),
        common_runtime_args=[gu.buffer_address(), wv.buffer_address(), sparsity.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "swiglu32_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=ct + _accessor_args(out, sparsity),
        common_runtime_args=[out.buffer_address(), sparsity.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = os.environ.get("TT_LAGUNA_SW32_FP32_DST", "0") == "1"  # silu/products in fp32 DST
    cfg.math_approx_mode = False
    compute = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "swiglu32_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[],
        config=cfg,
    )
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op([gu, wv, sparsity, out], program)
    return out
