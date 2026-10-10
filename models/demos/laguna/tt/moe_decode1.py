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


def _core_ranges(device, start, count):
    """CoreRangeSet of the row-major core indices [start, start + count)."""
    gx = device.compute_with_storage_grid_size().x
    ranges, i = set(), start
    while i < start + count:
        y, x = divmod(i, gx)
        n = min(gx - x, start + count - i)
        ranges.add(ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x + n - 1, y)))
        i += n
    return ttnn.CoreRangeSet(ranges)


def _gate_up_parts(x, w_gate_up, sparsity, out, grid, core_base, slot_groups, chunk):
    """Kernels and CBs of one gate_up_swiglu core set (cores core_base .. in row-major order)."""
    E = w_gate_up.shape[1]
    kt = x.padded_shape[-1] // TILE
    nt = w_gate_up.padded_shape[-1] // TILE // 2
    gx = x.device().compute_with_storage_grid_size().x
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
        compile_time_args=[kt, nt, E, chunk, x_page, w_page, sp_page, gx, slot_groups, SLOTS, core_base]
        + _accessor_args(x, w_gate_up, sparsity),
        common_runtime_args=[x.buffer_address(), w_gate_up.buffer_address(), sparsity.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "moe1_gu_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[nt, E, 2048, sp_page, gx, slot_groups, SLOTS, core_base] + _accessor_args(out, sparsity),
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
    return [reader, writer, compute], cbs


def gate_up_swiglu(x, w_gate_up, sparsity, memory_config=ttnn.L1_MEMORY_CONFIG, chunk=None, shared=None):
    """x [1, 1, 1, H] bf16 TILE interleaved; w_gate_up [1, E, H, 2I] (gate | up per row); sparsity [1, 1, 1, E] bf16
    ROW_MAJOR routing weights. Returns [1, E, 32, I] bf16 (row 0 of each active expert's tiles is real).
    shared = (w_sh [1, 1, H, 2 I_sh], active row, chunk): the shared expert as one always-active expert on the cores
    after the routed ones, in the same program; returns (routed, shared [1, 1, 32, I_sh])."""
    device = x.device()
    E = w_gate_up.shape[1]
    nt = w_gate_up.padded_shape[-1] // TILE // 2
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, E, TILE, nt * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config
    )
    slot_groups = -(-MAX_ACTIVE // SLOTS)
    grid_size = device.compute_with_storage_grid_size()
    num_cores = nt * slot_groups
    assert num_cores <= grid_size.x * grid_size.y, (num_cores, grid_size)
    grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, True)
    # weight tiles per read: 16 lets the compute start sooner than 32 (b1 decode 14.92 -> 14.78 ms/token; 8/12/24/48/96
    # measured 14.89/14.83/14.79/14.96/15.17)
    # (chunk: the caller's override, e.g. the shared expert's 8 active cores read best in 48-tile pieces)
    chunk = int(chunk or os.environ.get("TT_LAGUNA_MOE1_CHUNK", "16"))
    kernels, cbs = _gate_up_parts(x, w_gate_up, sparsity, out, grid, 0, slot_groups, chunk)
    io = [x, w_gate_up, sparsity, out]
    sh_out = None
    if shared is not None:
        w_sh, act, sh_chunk = shared
        nt_sh = w_sh.padded_shape[-1] // TILE // 2
        assert num_cores + nt_sh <= grid_size.x * grid_size.y, (num_cores, nt_sh)
        sh_out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, TILE, nt_sh * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config
        )
        k2, c2 = _gate_up_parts(x, w_sh, act, sh_out, _core_ranges(device, num_cores, nt_sh), num_cores, 1, sh_chunk)
        kernels, cbs = kernels + k2, cbs + c2
        io += [w_sh, act, sh_out]
    ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return out if shared is None else (out, sh_out)


def down_sum(glu, w_down, sparsity, memory_config=ttnn.L1_MEMORY_CONFIG, shared=None):
    """glu [1, E, 32, I] (gate_up_swiglu output); w_down [1, E, I, H]; sparsity as above. Returns [1, 1, 1, H] bf16:
    the routing-weighted sum of the active experts' down projections. shared = (sh_glu [1, 1, 32, I_sh], w_sh_down
    [1, 1, I_sh, H] column pages): the shared expert's down projection accumulated into the same output tiles."""
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
    sh_glu, w_sh = shared if shared is not None else (glu, w_down)
    kt_sh = w_sh.padded_shape[-2] // TILE if shared is not None else 0
    w_sh_page = _tile_bytes(w_sh.dtype)
    cbs = [
        _cb(grid, 0, glu.dtype, x_page, 2 * kt),
        _cb(grid, 1, w_down.dtype, w_page, 2 * kt),
        _cb(grid, 2, ttnn.uint32, 64, 1),
        _cb(grid, 3, ttnn.bfloat16, sp_page, 1),
        _cb(grid, 4, ttnn.bfloat16, sp_page, 1),
        _cb(grid, 5, ttnn.bfloat16, 2048, 1),
        _cb(grid, 16, ttnn.bfloat16, 2048, 1),
    ]
    if kt_sh:
        cbs.append(_cb(grid, 6, w_sh.dtype, w_sh_page, kt_sh))
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "moe1_down_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[kt, nt, E, x_page, w_page, sp_page, grid_size.x, MAX_ACTIVE, kt_sh, w_sh_page]
        + _accessor_args(glu, w_down, sparsity, sh_glu, w_sh),
        common_runtime_args=[glu.buffer_address(), w_down.buffer_address(), sparsity.buffer_address(),
                             sh_glu.buffer_address(), w_sh.buffer_address()],  # fmt: skip
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "moe1_down_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[2048, grid_size.x, E, sp_page, int(kt_sh > 0)] + _accessor_args(out, sparsity),
        common_runtime_args=[out.buffer_address(), sparsity.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "moe1_down_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[kt, kt_sh],
        config=_compute_config(),
    )
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    io = [glu, w_down, sparsity, out] if shared is None else [glu, w_down, sparsity, sh_glu, w_sh, out]
    ttnn.generic_op(io, program)
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


def expert_sum32(x, sparsity, memory_config=ttnn.L1_MEMORY_CONFIG):
    """Sum over the active experts of the down output x [1, E, 32, H] (inactive experts' tiles are never read, so the
    down sparse_matmul need not zero-fill them). Returns [1, 1, 32, H] bf16."""
    device = x.device()
    E = x.shape[1]
    nt = x.padded_shape[-1] // TILE
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, x.shape[2], nt * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config
    )
    grid_size = device.compute_with_storage_grid_size()
    assert nt <= grid_size.x * grid_size.y, (nt, grid_size)
    grid = ttnn.num_cores_to_corerangeset(nt, grid_size, True)
    page, sp_page = 2048, max(E * 2, 64)
    cbs = [
        _cb(grid, 0, ttnn.bfloat16, page, 2),
        _cb(grid, 1, ttnn.uint32, 64, 1),
        _cb(grid, 2, ttnn.bfloat16, sp_page, 1),
        _cb(grid, 3, ttnn.bfloat16, sp_page, 1),
        _cb(grid, 4, ttnn.bfloat16, page, 1),
        _cb(grid, 16, ttnn.bfloat16, page, 1),
    ]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "esum32_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[nt, E, page, sp_page, grid_size.x] + _accessor_args(x, sparsity),
        common_runtime_args=[x.buffer_address(), sparsity.buffer_address()],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "esum32_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[page, grid_size.x, E, sp_page] + _accessor_args(out, sparsity),
        common_runtime_args=[out.buffer_address(), sparsity.buffer_address()],
        config=ttnn.WriterConfigDescriptor(),
    )
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    compute = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "esum32_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[],
        config=cfg,
    )
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op([x, sparsity, out], program)
    return out


_GU32_X_MCAST = os.environ.get("TT_LAGUNA_GU32_X_MCAST", "1") == "1"
_DN32_ROW_READS = os.environ.get("TT_LAGUNA_DN32_ROW_READS", "1") == "1"


def _mcast_rects(device, grid):
    """(x0, y0, x1, y1, dests) per rectangle of ``grid`` in NOC coordinates for a multicast from logical core (0, 0)
    (which belongs to the first rectangle and is not counted there)."""
    rects = []
    for i, r in enumerate(grid.ranges()):
        a = device.worker_core_from_logical_core(r.start)
        b = device.worker_core_from_logical_core(r.end)
        n = (r.end.x - r.start.x + 1) * (r.end.y - r.start.y + 1)
        own = r.start.x == 0 and r.start.y == 0
        rects.append((a.x, a.y, b.x, b.y, n - 1 if own else n))
    return rects


def gate_up32(x, gu_cols, wv, sparsity, groups=3, chunk=48, memory_config=ttnn.L1_MEMORY_CONFIG):
    """32-token routed gate/up + SwiGLU + routing weight over column-page weights (see colpage.py).
    x [1, 1, 32, H] bf16 TILE interleaved; gu_cols: ColumnPages of the packed [E, H, 2I] weight (Ct = 2 * I/32);
    wv [1, E, 32, 1] per-(expert, token) weights; sparsity [1, 1, R, E] bf16 row-major: the union row (R = 1) or
    per-token routing rows (unioned in the kernels). wv None (R > 1 only): each expert's weight tile is built from
    the routing rows. Returns [1, E, 32, I]."""
    device = x.device()
    E, Kt = gu_cols.E, gu_cols.Kt
    nt = gu_cols.Ct // 2
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, E, x.shape[-2], nt * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config
    )
    gs = device.compute_with_storage_grid_size()
    cores = nt * groups
    assert cores <= gs.x * gs.y and Kt % chunk == 0, (cores, Kt, chunk)
    grid = ttnn.num_cores_to_corerangeset(cores, gs, True)
    page, wt, sp_page = 2048, 576, max(E * 2, 64)
    sp_rows = int(sparsity.shape[-2])
    assert wv is not None or sp_rows > 1
    cbs = [
        _cb(grid, 0, ttnn.bfloat16, page, Kt),
        _cb(grid, 1, ttnn.bfloat4_b, wt, 2 * chunk),
        _cb(grid, 2, ttnn.bfloat16, page, 2),
        _cb(grid, 3, ttnn.uint32, 64, 1),
        _cb(grid, 4, ttnn.bfloat16, sp_page, sp_rows + 1),
        _cb(grid, 5, ttnn.bfloat16, page, 1),
        _cb(grid, 6, ttnn.bfloat16, page, 1),
        _cb(grid, 7, ttnn.bfloat16, sp_page, sp_rows),
        _cb(grid, 16, ttnn.bfloat16, page, 2),
    ]
    rects = _mcast_rects(device, grid) if _GU32_X_MCAST else []
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "gu32_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[Kt, nt, E, chunk, page, wt, sp_page, gs.x, groups]
        + _accessor_args(x, gu_cols.buf, sparsity if wv is None else wv, sparsity),
        common_runtime_args=[x.buffer_address(), gu_cols.buf.buffer_address(), 0 if wv is None else wv.buffer_address(),
                             sparsity.buffer_address(), sp_rows, len(rects)] + [v for r in rects for v in r],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "gu32_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[nt, E, page, sp_page, gs.x, groups] + _accessor_args(out, sparsity),
        common_runtime_args=[out.buffer_address(), sparsity.buffer_address(), sp_rows],
        config=ttnn.WriterConfigDescriptor(),
    )
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.LoFi
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    compute = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "gu32_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[Kt, chunk],
        config=cfg,
    )
    ins = [x, gu_cols.buf, sparsity, out] if wv is None else [x, gu_cols.buf, wv, sparsity, out]
    sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=grid, initial_value=0)]
    ttnn.generic_op(ins, ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=sems, cbs=cbs))
    return out


def down32(glu, d_cols, sparsity, cols_per_core=8, expert_groups=8, memory_config=ttnn.L1_MEMORY_CONFIG):
    """32-token routed down projection + sum over active experts, column-page weights. glu [1, E, 32, I] (gate_up32
    output); d_cols: ColumnPages of [E, I, H]; sparsity as in gate_up32. Returns [1, 1, 32, H] bf16."""
    device = glu.device()
    E, Kt, Nh = d_cols.E, d_cols.Kt, d_cols.Ct
    # partial sums per expert group, reduced over dim 1 after the kernel
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, expert_groups, glu.shape[-2], Nh * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config
    )
    gs = device.compute_with_storage_grid_size()
    cores = Nh // cols_per_core * expert_groups
    assert Nh % cols_per_core == 0 and cores <= gs.x * gs.y
    grid = ttnn.num_cores_to_corerangeset(cores, gs, True)
    page, wt, sp_page = 2048, 576, max(E * 2, 64)
    sp_rows = int(sparsity.shape[-2])
    T = int(glu.shape[-2])
    x_rows = T if T <= 8 and _DN32_ROW_READS else 32  # <= 8: the packed-rows all-reduce reads rows < T only
    cbs = [
        _cb(grid, 0, ttnn.bfloat16, page, 2 * Kt),
        _cb(grid, 1, ttnn.bfloat4_b, wt, 2 * cols_per_core * Kt),
        _cb(grid, 2, ttnn.uint32, 64, 1),
        _cb(grid, 3, ttnn.bfloat16, sp_page, sp_rows),
        _cb(grid, 4, ttnn.bfloat16, sp_page, sp_rows),
        _cb(grid, 5, ttnn.bfloat16, page, 1),
        _cb(grid, 16, ttnn.bfloat16, page, cols_per_core),
    ]
    reader = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "dn32_reader.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[Kt, Nh, E, page, wt, sp_page, gs.x, cols_per_core, expert_groups]
        + _accessor_args(glu, d_cols.buf, sparsity),
        # up to 8 token rows (DFlash verify): read only those rows of each activation tile; the output rows past them
        # are not used (the packed-rows all-reduce reads rows < T)
        common_runtime_args=[glu.buffer_address(), d_cols.buf.buffer_address(), sparsity.buffer_address(), sp_rows,
                             x_rows],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "dn32_writer.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[page, gs.x, E, sp_page, cols_per_core, expert_groups, Nh] + _accessor_args(out, sparsity),
        common_runtime_args=[out.buffer_address(), sparsity.buffer_address(), sp_rows],
        config=ttnn.WriterConfigDescriptor(),
    )
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.LoFi
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    compute = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "dn32_compute.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[Kt, cols_per_core],
        config=cfg,
    )
    ttnn.generic_op([glu, d_cols.buf, sparsity, out], ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs))
    if expert_groups == 1:
        return out
    red = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, glu.shape[-2], Nh * TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, memory_config
    )
    return ttnn.experimental.fast_reduce_nc(out, dims=[1], output=red, memory_config=memory_config)
