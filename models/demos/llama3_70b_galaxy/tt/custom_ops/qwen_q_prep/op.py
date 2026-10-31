# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen long-prefill Q prep as one ``ttnn.generic_op`` program per device.

Replaces ``TtLlamaAttention._qwen_local_q_prep`` (llama_attention.py): slice x2 + concat, typecast
bf8 -> bf16, q_norm (RMSNorm), rotary_embedding_llama, typecast bf16 -> bf8, paged_fill_cache. The
result is meant to be bit-identical to that chain (README.md lists what is argued and what needs a
device run to confirm).

Contract (per device of a (rows, cols) mesh; column c owns ring slices c and 2 * cols - 1 - c):

    q          [1, H, S, 128] bf8   TILE, DRAM interleaved. Updated in place on the owned slices;
                                    the other slices are not touched. Returned.
    cos, sin   [1, 1, S_rot, 128] bf16 TILE interleaved, S_rot >= S (the full prefill rot mats;
                                    rows 0..S-1 are used, positions = row index).
    trans_mat  [1, 1, 32, 32] bf16 TILE interleaved (transformation_mats["prefill"]).
    gamma      [1, 1, 4, 32] bf16 ROW_MAJOR interleaved (RMSNorm.weight of q_norm).
    eps        q_norm.eps (the reader truncates it to bf16, as the stock RMSNorm reader does).
    S % (2 * cols * 32) == 0.

Program: the compute grid is split into two halves. Each N core (first half) reads its share of the
owned tile rows (bf8) and runs typecast + RMSNorm with the q_norm compute config (HiFi2, fp32 DEST);
it hands each bf16 row to its paired R core (second half) through an L1-to-L1 write. The R core runs
RoPE + typecast with the rotary_embedding_llama / typecast configs (HiFi4, 16-bit DEST, precise bf8
pack) and writes the bf8 row back to q at the same tile pages. Two core groups are needed because the
stock norm and the stock RoPE use different math fidelity and DEST width, and one compute kernel
has one of each.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import ttnn

KERNEL_ROOT = "models/demos/llama3_70b_galaxy/tt/custom_ops/qwen_q_prep/kernels"

HEAD_DIM = 128
WT = HEAD_DIM // 32

# Must match kernels/qprep_common.hpp
CB_QIN, CB_X, CB_SCALER, CB_EPS, CB_GAMMA = 0, 1, 2, 3, 4
CB_XMM2, CB_EX2, CB_EX2PE, CB_FUSION, CB_NOUT = 5, 6, 7, 8, 16
CB_NORMED = 9
CB_TRANS, CB_COS, CB_SIN, CB_ROT, CB_SIN_I, CB_COS_I, CB_QOUT = 10, 11, 12, 13, 14, 15, 17
SEM_CREDIT, SEM_READY = 0, 1

DEFAULT_HANDOFF_DEPTH = 2  # CB_NORMED slots (tile rows) per R core

_DESCRIPTOR_CACHE: dict = {}
# Keys carry buffer addresses (gamma differs per layer), so one sequence length needs ~2 entries per layer
# (Q and K) = ~128 for 64 layers. 2048 holds every prefill bucket without the clear() below thrashing.
_DESCRIPTOR_CACHE_MAX = 2048


def _tile_bytes(dtype) -> int:
    return {ttnn.bfloat16: 2048, ttnn.bfloat8_b: 1088, ttnn.float32: 4096}[dtype]


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def column_slices(col: int, cols: int) -> tuple[int, int]:
    """Ring SDPA slices owned by mesh column ``col`` (of 2 * cols slices)."""
    return col, 2 * cols - 1 - col


def pair_cores(grid_x: int, grid_y: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """(N core, R core) logical coords. Splits the grid in two halves along y (or x when y is odd)."""
    if grid_y % 2 == 0:
        half = grid_y // 2
        return [((x, y), (x, y + half)) for y in range(half) for x in range(grid_x)]
    if grid_x % 2 == 0:
        half = grid_x // 2
        return [((x, y), (x + half, y)) for y in range(grid_y) for x in range(half)]
    half = (grid_y - 1) // 2
    return [((x, y), (x, y + half)) for y in range(half) for x in range(grid_x)]


@dataclass(frozen=True)
class QPrepPlan:
    n_heads: int
    seq_len: int
    cols: int
    num_pairs: int

    @property
    def ht(self) -> int:
        return self.seq_len // 32

    @property
    def sl_t(self) -> int:
        """Tile rows per ring slice."""
        return self.seq_len // (2 * self.cols) // 32

    @property
    def rows(self) -> int:
        """Owned tile rows per device: 2 slices x sl_t x heads."""
        return self.n_heads * 2 * self.sl_t

    def pair_rows(self, p: int) -> tuple[int, int]:
        """(first row, row count) of pair p; the first rows % num_pairs pairs take one extra row."""
        base, extra = divmod(self.rows, self.num_pairs)
        return p * base + min(p, extra), base + (1 if p < extra else 0)

    def row_coords(self, row: int) -> tuple[int, int, int]:
        """(head, local slice j, tile row in slice) of a global owned-row index.

        Rows run head-fastest (j, tl, h), so a core's consecutive rows share one cos/sin row; this
        must match RowWalker::next in kernels/qprep_common.hpp.
        """
        j, rem = divmod(row, self.sl_t * self.n_heads)
        tl, h = divmod(rem, self.n_heads)
        return h, j, tl


def _validate(q, cos, sin, trans_mat, gamma, cols: int, split: bool = False) -> tuple[int, int]:
    shape = tuple(q.shape)
    if len(shape) != 4 or shape[0] != 1 or shape[3] != HEAD_DIM:
        raise ValueError(f"q must be [1, H, S, {HEAD_DIM}], got {shape}")
    n_heads, seq_len = shape[1], shape[2] * (2 * cols if split else 1)
    if q.dtype != ttnn.bfloat8_b or q.layout != ttnn.TILE_LAYOUT:
        raise ValueError(f"q must be bfloat8_b TILE, got {q.dtype} {q.layout}")
    mc = q.memory_config()
    if mc.buffer_type != ttnn.BufferType.DRAM or mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        raise ValueError(f"q must be DRAM interleaved, got {mc}")
    if seq_len % (2 * cols * 32):
        raise ValueError(f"S={seq_len} must be a multiple of 2 * cols * 32 = {2 * cols * 32}")
    for name, t in (("cos", cos), ("sin", sin)):
        s = tuple(t.shape)
        if len(s) != 4 or s[0] != 1 or s[1] != 1 or s[3] != HEAD_DIM or s[2] < seq_len or s[2] % 32:
            raise ValueError(f"{name} must be [1, 1, S_rot >= {seq_len}, {HEAD_DIM}], got {s}")
        if t.dtype != ttnn.bfloat16 or t.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"{name} must be bfloat16 TILE, got {t.dtype} {t.layout}")
    if tuple(trans_mat.shape) != (1, 1, 32, 32) or trans_mat.dtype != ttnn.bfloat16:
        raise ValueError(f"trans_mat must be bfloat16 [1, 1, 32, 32], got {tuple(trans_mat.shape)} {trans_mat.dtype}")
    if trans_mat.layout != ttnn.TILE_LAYOUT:
        raise ValueError("trans_mat must be TILE")
    if tuple(gamma.shape) != (1, 1, WT, 32) or gamma.dtype != ttnn.bfloat16 or gamma.layout != ttnn.ROW_MAJOR_LAYOUT:
        raise ValueError(
            f"gamma must be row-major bfloat16 [1, 1, {WT}, 32] (RMSNorm.weight), got "
            f"{tuple(gamma.shape)} {gamma.dtype} {gamma.layout}"
        )
    for name, t in (("cos", cos), ("sin", sin), ("trans_mat", trans_mat), ("gamma", gamma)):
        if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise ValueError(f"{name} must be interleaved")
    return n_heads, seq_len


def _core_range_set(coords) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in coords})


def _accessor(t) -> list[int]:
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _build_program(
    device, plan, pairs, col, q, cos, sin, trans_mat, gamma, eps, depth, src, src_layout, src_local=False, q1=None
) -> ttnn.ProgramDescriptor:
    n_coords = [n for n, _ in pairs]
    r_coords = [r for _, r in pairs]
    n_set = _core_range_set(n_coords)
    r_set = _core_range_set(r_coords)
    all_set = _core_range_set(n_coords + r_coords)

    def cb(idx, tiles, dtype, cores):
        size = _tile_bytes(dtype)
        return ttnn.CBDescriptor(
            total_size=tiles * size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=size)],
        )

    bf16, bf8, f32 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32
    cbs = [
        # N cores
        cb(CB_QIN, 2 * WT, bf8, n_set),
        cb(CB_X, WT, bf16, n_set),
        cb(CB_SCALER, 1, bf16, n_set),
        cb(CB_EPS, 1, bf16, n_set),
        cb(CB_GAMMA, WT, bf16, n_set),
        cb(CB_XMM2, WT, f32, n_set),
        cb(CB_EX2, 1, f32, n_set),
        cb(CB_EX2PE, 1, f32, n_set),
        cb(CB_FUSION, WT, f32, n_set),
        cb(CB_NOUT, 2 * WT, bf16, n_set),
        # both groups: one L1 address on every core, so the N core can address the R core's ring
        cb(CB_NORMED, depth * WT, bf16, all_set),
        # R cores
        cb(CB_TRANS, 1, bf16, r_set),
        cb(CB_COS, 2 * WT, bf16, r_set),
        cb(CB_SIN, 2 * WT, bf16, r_set),
        cb(CB_ROT, WT, bf16, r_set),
        cb(CB_SIN_I, WT, bf16, r_set),
        cb(CB_COS_I, WT, bf16, r_set),
        cb(CB_QOUT, 2 * WT, bf8, r_set),
    ]
    semaphores = [
        ttnn.SemaphoreDescriptor(id=SEM_CREDIT, core_ranges=all_set, initial_value=0),
        ttnn.SemaphoreDescriptor(id=SEM_READY, core_ranges=all_set, initial_value=0),
    ]

    s0, s1 = column_slices(col if plan.cols > 1 else 0, plan.cols)
    walk = [plan.sl_t, s0 * plan.sl_t, s1 * plan.sl_t, plan.ht, plan.n_heads]
    n_reader_rt, n_compute_rt, n_writer_rt = [], [], []
    r_reader_rt, r_compute_rt, r_writer_rt = [], [], []
    for p, (n_xy, r_xy) in enumerate(pairs):
        start, count = plan.pair_rows(p)
        h0, j0, tl0 = plan.row_coords(start) if count else (0, 0, 0)
        n_phys = device.worker_core_from_logical_core(ttnn.CoreCoord(*n_xy))
        r_phys = device.worker_core_from_logical_core(ttnn.CoreCoord(*r_xy))
        n_reader_rt.append(
            (
                n_xy,
                [src.buffer_address(), gamma.buffer_address(), count, h0, j0, tl0]
                + walk
                + list(src_layout)
                + [int(src_local)],
            )
        )
        n_compute_rt.append((n_xy, [count]))
        n_writer_rt.append((n_xy, [count, r_phys.x, r_phys.y, depth]))
        r_reader_rt.append(
            (
                r_xy,
                [cos.buffer_address(), sin.buffer_address(), trans_mat.buffer_address(), count, h0, j0, tl0]
                + walk
                + [n_phys.x, n_phys.y],
            )
        )
        r_compute_rt.append((r_xy, [count, h0, plan.n_heads]))
        r_writer_rt.append(
            (
                r_xy,
                [q.buffer_address(), count, h0, j0, tl0]
                + walk
                + [int(q1 is not None), (q1 if q1 is not None else q).buffer_address()],
            )
        )

    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_ROOT}/norm_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=n_set,
            compile_time_args=[_f32_bits(eps)] + _accessor(src) + _accessor(gamma),
            runtime_args=n_reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_ROOT}/norm_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=n_set,
            compile_time_args=[_f32_bits(1.0 / HEAD_DIM)],
            runtime_args=n_compute_rt,
            # RMSNorm.compute_kernel_config_hifi2 (models/common/rmsnorm.py)
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                dst_full_sync_en=False,
                bfp8_pack_precise=False,
            ),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_ROOT}/norm_handoff_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=n_set,
            compile_time_args=[],
            runtime_args=n_writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_ROOT}/rope_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=r_set,
            compile_time_args=_accessor(cos) + _accessor(sin) + _accessor(trans_mat),
            runtime_args=r_reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_ROOT}/rope_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=r_set,
            compile_time_args=[],
            runtime_args=r_compute_rt,
            # rotary_embedding_llama default (HiFi4, approx, 16-bit DEST); bfp8_pack_precise is the
            # bf16 -> bf8 typecast's setting (typecast.cpp) and only affects the bf8 output CB.
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
                bfp8_pack_precise=True,
            ),
        ),
        ttnn.KernelDescriptor(
            kernel_source=f"{KERNEL_ROOT}/rope_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=r_set,
            compile_time_args=_accessor(q) + _accessor(q1 if q1 is not None else q),
            runtime_args=r_writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=semaphores, cbs=cbs)


def qwen_q_prep(
    q: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    trans_mat: ttnn.Tensor,
    gamma: ttnn.Tensor,
    *,
    eps: float,
    grid: tuple[int, int] | None = None,
    handoff_depth: int = DEFAULT_HANDOFF_DEPTH,
    all_rows: bool = False,
    src: ttnn.Tensor | None = None,
    src_col: int = 0,
    src_local: bool = False,
    q_out1: ttnn.Tensor | None = None,
) -> ttnn.Tensor:
    """typecast -> q_norm -> RoPE -> typecast on this column's two ring slices of q, in place.

    all_rows=True processes every position on every device instead (the K path: ring SDPA reads all of K),
    as two "slices" of S/2 rows; gamma / eps are then k_norm's.

    src: read the input rows from this fused QKV buffer instead of q ([1, 1, S, W] bf8 TILE DRAM interleaved, the
    heads at columns src_col + 128 * h), as nlp_create_qkv_heads would copy them; q is then only written (its
    other rows are not touched). Without src, q is updated in place.

    src_local=True (with src, not all_rows): src holds only this column's two ring slices in local order
    ([1, 1, S / cols, W] = [slice c ; slice 2 * cols - 1 - c], the context-parallel prefill layout); the source row
    is the local one, while cos/sin and the q rows written stay at the natural positions.

    q_out1 (with src, not all_rows): split output. q and q_out1 are then [1, H, S / (2 * cols), 128] bf8 buffers that
    receive this column's local slice 0 (ring slice c) and local slice 1 (ring slice 2 * cols - 1 - c), each at its
    rows within the slice (what ttnn.slice of the natural q at the slice start would give), so no slice op is needed
    before a per-slice consumer. The computed values are those of the natural-q form.

    Returns ``q``. See the module docstring for the tensor contract. Raises ValueError on any
    shape, dtype or layout mismatch; there is no fallback to the stock ops.
    """
    device = q.device()
    mesh_rows, cols = list(device.shape)
    plan_cols = 1 if all_rows else cols
    split = q_out1 is not None
    if split:
        if src is None or all_rows:
            raise ValueError("q_out1 (split output) needs src and is not defined for all_rows")
        if tuple(q_out1.shape) != tuple(q.shape) or q_out1.dtype != q.dtype or q_out1.layout != q.layout:
            raise ValueError(f"q_out1 must match q: {tuple(q_out1.shape)} {q_out1.dtype} vs {tuple(q.shape)} {q.dtype}")
        q1mc = q_out1.memory_config()
        if q1mc.buffer_type != ttnn.BufferType.DRAM or q1mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise ValueError(f"q_out1 must be DRAM interleaved, got {q1mc}")
    n_heads, seq_len = _validate(q, cos, sin, trans_mat, gamma, plan_cols, split)
    if handoff_depth < 1:
        raise ValueError(f"handoff_depth must be >= 1, got {handoff_depth}")
    if grid is None:
        g = device.compute_with_storage_grid_size()
        grid = (int(g.x), int(g.y))
    if src_local and (src is None or all_rows):
        raise ValueError("src_local needs src and is not defined for all_rows")
    if src is None:
        src = q
        src_layout = (0, WT, (seq_len // 32) * WT)
    else:
        s = tuple(src.shape)
        src_rows = seq_len // plan_cols if src_local else seq_len
        if len(s) != 4 or s[0] != 1 or s[1] != 1 or s[2] != src_rows or s[3] % 32:
            raise ValueError(f"src must be [1, 1, {src_rows}, W], got {s}")
        if src.dtype != ttnn.bfloat8_b or src.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"src must be bfloat8_b TILE, got {src.dtype} {src.layout}")
        smc = src.memory_config()
        if smc.buffer_type != ttnn.BufferType.DRAM or smc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
            raise ValueError(f"src must be DRAM interleaved, got {smc}")
        if src_col % 32 or src_col + n_heads * HEAD_DIM > s[3]:
            raise ValueError(f"src_col {src_col} + {n_heads} heads does not fit in width {s[3]}")
        src_layout = (src_col // 32, s[3] // 32, WT)
    pairs = pair_cores(*grid)
    if not pairs:
        raise ValueError(f"grid {grid} has no core pairs")
    plan = QPrepPlan(n_heads=n_heads, seq_len=seq_len, cols=plan_cols, num_pairs=len(pairs))

    key = (
        tuple(device.shape),
        grid,
        all_rows,
        src_local,
        src_layout,
        src.buffer_address(),
        handoff_depth,
        n_heads,
        seq_len,
        _f32_bits(eps),
        q.buffer_address(),
        q_out1.buffer_address() if split else None,
        cos.buffer_address(),
        sin.buffer_address(),
        trans_mat.buffer_address(),
        gamma.buffer_address(),
    )
    mesh_program = _DESCRIPTOR_CACHE.get(key)
    if mesh_program is None:
        mesh_program = ttnn.MeshProgramDescriptor()
        for col in range(cols):
            program = _build_program(
                device,
                plan,
                pairs,
                col,
                q,
                cos,
                sin,
                trans_mat,
                gamma,
                eps,
                handoff_depth,
                src,
                src_layout,
                src_local,
                q_out1,
            )
            mesh_range = ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, col), ttnn.MeshCoordinate(mesh_rows - 1, col))
            mesh_program[mesh_range] = program
        if len(_DESCRIPTOR_CACHE) >= _DESCRIPTOR_CACHE_MAX:
            _DESCRIPTOR_CACHE.clear()
        _DESCRIPTOR_CACHE[key] = mesh_program

    # io_tensors: inputs, then the output last (q, updated in place).
    if src is q:
        return ttnn.generic_op([cos, sin, trans_mat, gamma, q], mesh_program)
    if split:
        ttnn.generic_op([src, cos, sin, trans_mat, gamma, q_out1, q], mesh_program)
        return q
    return ttnn.generic_op([src, cos, sin, trans_mat, gamma, q], mesh_program)
