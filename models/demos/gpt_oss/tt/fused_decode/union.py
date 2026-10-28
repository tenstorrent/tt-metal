# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Multi-row decode MoE over the union of the rows' routed experts (DFlash verify: M token rows of one user).

The multi-row router (LinearStream rows = M, out_mode 4) writes
  * union: UINT16 [1, 64] = [U, the U union expert ids ascending],
  * weights: BF16 [1, 32 M], row r's softmax weight of union expert i at entry 32 r + i (0 when row r did not route
    to it).
ExpertGateUpUnion streams the gate|up columns of the U union experts (the ExpertGateUpStream weight layout) for all M
rows at once ([M, 32] activation tiles), applies the SwiGLU and scales row r by w[r][i]; ExpertDownUnion streams the
down columns of the same experts and sums over them (K = U segments [w-scaled act_i | w[:, i] bias multipliers]) into
the row-tile partial sum the boundary all-reduces. Every union expert's weights are read once for the M rows.
"""

import torch

import ttnn

from .stream import NBIAS, TILE, TILE_BYTES, TINY_BYTES, _BankStreamOp, _f32_bits, _page_bytes, columns_per_bank

MAX_EXPERTS = 32


def union_buffers(mesh_device, rows):
    """(union, weights) router outputs (L1 interleaved, replicated)."""
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    uni = ttnn.from_torch(
        torch.zeros(1, 64, dtype=torch.int32),
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    w = ttnn.from_torch(
        torch.zeros(1, 32 * rows),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=rep,
    )
    return uni, w


def act_buffer(mesh_device, inter_pad, rows, slots=MAX_EXPERTS, memory_config=ttnn.L1_MEMORY_CONFIG):
    """[slots, I_pad / 32 [rows, 32] tiles] BF16 activation buffer, L1 interleaved, one page per tile (slot s, tile t =
    page s I_pad / 32 + t): every down core reads every tile of a slot, and a slot in one page would put all those
    reads on one core's bank."""
    return ttnn.from_torch(
        torch.zeros(slots * inter_pad // TILE, TILE * rows),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


_IDENTITY = {}


def bfp4_identity(mesh_device, dtype=ttnn.bfloat4_b):
    """The 32x32 identity as one tile of `dtype` (exact: entries 0 / 1), replicated in DRAM."""
    key = (id(mesh_device), dtype)
    cached = _IDENTITY.get(key)
    if cached is None or cached[0] is not mesh_device:
        t = ttnn.from_torch(
            torch.eye(TILE)[None, None],
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        cached = _IDENTITY[key] = (mesh_device, t)
    return cached[1]


class ExpertGateUpUnion(_BankStreamOp):
    """(x, union, weights, gate_up_stream, act) -> act. x: the boundary's normed hidden in the row-tile layout (M rows);
    act (act_buffer): slot i = expert union[i]'s SwiGLU rows scaled by w[r][i]."""

    def __init__(
        self,
        mesh_device,
        hidden,
        inter_pad,
        swiglu_limit,
        alpha,
        readers,
        rows,
        real_inter,
        dense_silu=False,
        max_experts=MAX_EXPERTS,
        weight_dtype=ttnn.bfloat4_b,
        rot=0,
        placed=None,
    ):
        """dense_silu: plain silu(g) * u (a dense MLP run as one union expert, e.g. the DFlash drafter).
        real_inter: the unpadded intermediate size; cores whose column pairs are all padding are dropped (their act
        tiles stay zero, as do the matching down weights). rot: the weights' bank rotation (_BankStreamOp); placed:
        explicit reader cores (_BankStreamOp)."""
        super().__init__(mesh_device, readers, rot=rot, placed=placed)
        self.dense_silu = dense_silu
        self.max_experts = max_experts
        self.rows = rows
        self.kx = hidden // TILE
        self.kt = self.kx + 1
        self.inter_pad = inter_pad
        self.P = inter_pad // TILE // self.banks
        assert self.P % readers == 0, f"{self.P} pairs per bank do not split over {readers} readers"
        self.pairs = self.P // readers
        self.limit, self.alpha = swiglu_limit, alpha
        self.weight_dtype = weight_dtype
        self.tile_bytes = TILE_BYTES[weight_dtype]
        self.col_bytes = self.kt * self.tile_bytes
        self.expert_stride = 2 * self.P * self.col_bytes
        self._keep(lambda bank, r: (self.logical_bank(bank) * self.P + r * self.pairs) * TILE < real_inter)

    def weight_rows(self, num_experts):
        return num_experts * 2 * self.P * self.kt * TILE

    def __call__(
        self,
        x,
        union,
        weights,
        w_stream,
        act,
        extra_defines,
        extra_writer_rt,
        extra_reader_rt=None,
        pred_p=0,
    ):
        """(program, io) for the fused program (dflash/moe_fused.py). pred_p > 0 (MOE_PRED): a CB (index 6) for the
        prefetched columns of pred_p experts."""
        M = self.rows
        cb_x, cb_w, cb_idx, cb_scr, cb_sc, cb_id, cb_act = 0, 1, 2, 3, 4, 5, 16
        tiny = TINY_BYTES * M
        cbs = [
            self._cb(cb_x, ttnn.bfloat16, tiny, self.kt, tiny=True, rows=M),
            self._cb(cb_w, self.weight_dtype, self.tile_bytes, 3 * self.kt),
            self._cb(cb_idx, ttnn.uint16, 256, 1),
            self._cb(cb_scr, ttnn.bfloat16, 128 + 64 * M + 32, 1),
            self._cb(cb_sc, ttnn.bfloat16, tiny, self.max_experts, tiny=True, rows=M),
            self._cb(cb_id, self.weight_dtype, self.tile_bytes, 1),
            self._cb(cb_act, ttnn.bfloat16, tiny, 4, tiny=True, rows=M),
        ]
        if pred_p:  # MOE_PRED: the predicted experts' prefetched columns
            cbs.insert(0, self._cb(6, self.weight_dtype, self.tile_bytes, pred_p * 2 * self.pairs * self.kt))
        ident = bfp4_identity(self.mesh_device, self.weight_dtype)
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [
                w_stream.buffer_address(),
                union.buffer_address(),
                bank,
                vc,
                r * 2 * self.pairs * self.col_bytes,
            ] + list(extra_reader_rt or [])
            writer_rt[core.x][core.y] = [
                x.buffer_address(),
                act.buffer_address(),
                self.logical_bank(bank) * self.P + r * self.pairs,
                weights.buffer_address(),
                union.buffer_address(),
                ident.buffer_address(),
            ] + list(extra_writer_rt or [])
        reader_ct = [cb_w, cb_idx, self.kt, 2 * self.pairs, self.expert_stride, self.tile_bytes]
        reader_ct += ttnn.TensorAccessorArgs(union).get_compile_time_args()
        act_page = self.inter_pad // TILE * tiny
        writer_ct = [cb_x, cb_act, self.kx, NBIAS, self.pairs, M, act_page, cb_sc, cb_id, cb_scr, self.tile_bytes]
        for t in (x, act, weights, union, ident):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_x, cb_w, cb_act, self.kt, self.pairs]
        compute_ct += [_f32_bits(v) for v in (self.limit, -self.limit, -3.0e38, self.alpha, 1.0 / self.alpha, 1.0)]
        compute_ct += [cb_sc, cb_id]
        program = self._program(
            "gate_up_union",
            cbs,
            reader_ct,
            reader_rt,
            writer_ct,
            writer_rt,
            compute_ct,
            "gate_up_union",
            defines=[("ROWS", str(M))] + ([("DENSE_SILU", "1")] if self.dense_silu else []) + list(extra_defines or []),
        )
        return program, [w_stream, x, union, weights, ident, act]


class ExpertDownUnion(_BankStreamOp):
    """(act, union, weights, down_stream, out) -> out: the row-tile partial sum (M rows) of
    sum_i w[r][i] * (act_i @ Wd_i + bd_i) over the union experts (the ExpertDownStream weight layout), built into a
    fused program (dflash/moe_fused.py). Each core reads its columns of one expert as one contiguous block and
    accumulates its (<= 4) output tiles across the experts in DST (kernels/stream_down_union_*)."""

    def __init__(
        self,
        mesh_device,
        hidden,
        inter_pad,
        readers,
        rows,
        max_experts=MAX_EXPERTS,
        weight_dtype=ttnn.bfloat4_b,
        num_buffers=3,
        rot=0,
        skip_pad=False,
        placed=None,
        bank_cols=None,
        seg_inter=None,
    ):
        """num_buffers: weight blocks in the reader's ring (2 when the blocks are large). rot: the weights' bank
        rotation (_BankStreamOp). skip_pad: drop the cores whose output columns are all past `hidden` (padding).
        bank_cols (the balanced layout, fused_decode/stream.py down_bank_columns / stream_down_bal_layout; `placed`
        only): (first output column, count) per DRAM bank, split over that bank's placed cores (<= 4 columns each, a
        per-core count: runtime args); seg_inter: the K tiles per segment before the bias tile (the unpadded
        intermediate's, not inter_pad's)."""
        self.balanced = bank_cols is not None
        if self.balanced:
            assert placed is not None
            per_bank = {}
            for c, b, _ in placed:
                per_bank.setdefault(b, []).append(c)
            placed, self.core_cols = [], {}
            for b, (c0, n) in enumerate(bank_cols):
                k = -(-n // 4)
                cores = per_bank[b][:k]
                assert len(cores) == k, f"bank {b}: {n} columns need {k} cores, {len(per_bank[b])} placed"
                q, extra = divmod(n, k)
                slot = 0
                for r, c in enumerate(cores):
                    m = q + (1 if r < extra else 0)
                    placed.append((c, b, r))
                    self.core_cols[(c.x, c.y)] = (slot, m, c0 + slot)
                    slot += m
            readers = None
        super().__init__(mesh_device, readers, rot=rot, placed=placed)
        self.rows = rows
        self.hidden = hidden
        self.inter_pad = inter_pad
        if self.balanced:
            self.seg_tiles = seg_inter + 1
            self.C = max(n for _, n in bank_cols)
            self.cols = 4
            self.readers = max(r for _, _, r, _ in self.assignments) + 1
        else:
            self.seg_tiles = inter_pad // TILE + 1
            self.C = columns_per_bank(hidden, self.banks)
            assert self.C % readers == 0, f"{self.C} columns per bank do not split over {readers} readers"
            self.cols = self.C // readers
        self.weight_dtype = weight_dtype
        self.tile_bytes = TILE_BYTES[weight_dtype]
        self.seg_bytes = self.seg_tiles * self.tile_bytes
        # a core's columns of one expert (contiguous) are read in packets of <= 16 KB
        self.page_bytes = _page_bytes(self.seg_tiles, self.tile_bytes)
        # (the balanced layout's segments split into large packets: 24 BFP4 tiles = one 13.5 KB packet at TP 4)
        assert not self.balanced or self.page_bytes >= 8192, "a balanced segment with no packet size near 16 KB"
        self.expert_stride = self.C * self.seg_bytes
        self.max_experts = max_experts
        self.num_buffers = num_buffers
        assert self.cols <= 4, "a core's output tiles stay in DST"
        if skip_pad:
            self._keep(lambda bank, r: (self.logical_bank(bank) * self.C + r * self.cols) * TILE < hidden)

    def weight_rows(self, num_experts):
        return num_experts * self.C * self.seg_tiles * TILE

    def __call__(
        self,
        act,
        union,
        weights,
        w_stream,
        out,
        extra_defines,
        extra_writer_rt,
        notify_args,
        extra_reader_rt=None,
        pred_p=0,
    ):
        """(program, io) for the fused program. notify_args: the writers' sender notification [1, x, y, semaphore id]
        (the fused program's boundary sender). pred_p > 0 (MOE_PRED): a CB (index 6) for the prefetched columns of
        pred_p experts."""
        M = self.rows
        assert act.shape[-1] == TILE * M, "act_buffer's one page per tile layout"
        cb_in0, cb_w, cb_idx, cb_scr, cb_out = 0, 1, 2, 3, 16
        tiny = TINY_BYTES * M
        block = self.cols * self.seg_tiles
        cbs = [
            # a ring of 4 activation slots
            self._cb(cb_in0, ttnn.bfloat16, tiny, min(4, self.max_experts) * self.seg_tiles, tiny=True, rows=M),
            self._cb(cb_w, self.weight_dtype, self.tile_bytes, self.num_buffers * block),
            self._cb(cb_idx, ttnn.uint16, 256, 1),
            self._cb(cb_scr, ttnn.bfloat16, 128 + 64 * M + 128, 1),
            self._cb(cb_out, ttnn.bfloat16, tiny, 4, tiny=True, rows=M),
        ]
        if pred_p:  # MOE_PRED: the predicted experts' prefetched blocks
            cbs.insert(0, self._cb(6, self.weight_dtype, self.tile_bytes, pred_p * block))
        reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            if self.balanced:  # (slot in the bank, columns, first output column) of this core
                slot, ncols, col0 = self.core_cols[(core.x, core.y)]
            else:
                slot, ncols, col0 = r * self.cols, self.cols, self.logical_bank(bank) * self.C + r * self.cols
            reader_rt[core.x][core.y] = [
                w_stream.buffer_address(),
                union.buffer_address(),
                bank,
                vc,
                slot * self.seg_bytes,
            ] + list(extra_reader_rt or [])
            writer_rt[core.x][core.y] = [
                act.buffer_address(),
                weights.buffer_address(),
                out.buffer_address(),
                col0,
                union.buffer_address(),
            ] + list(extra_writer_rt)
            if self.balanced:  # the column count, last (kernels: D_NCOLS_ARG / D_NCOLS_WARG; compute: arg 0)
                reader_rt[core.x][core.y] = list(reader_rt[core.x][core.y]) + [ncols]
                writer_rt[core.x][core.y] = list(writer_rt[core.x][core.y]) + [ncols]
                compute_rt[core.x][core.y] = [ncols]
        reader_ct = [cb_w, cb_idx, self.seg_tiles, self.cols, self.expert_stride, self.tile_bytes, self.page_bytes]
        reader_ct.append(self.num_buffers)
        reader_ct += ttnn.TensorAccessorArgs(union).get_compile_time_args()
        act_page = self.inter_pad // TILE * tiny
        writer_ct = [cb_in0, cb_out, cb_scr, self.seg_tiles, M, NBIAS, self.cols, self.hidden // TILE, act_page]
        for t in (act, weights, out, union):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct += notify_args
        compute_ct = [cb_in0, cb_w, cb_out, self.seg_tiles, self.cols]
        program = self._program(
            "down_union",
            cbs,
            reader_ct,
            reader_rt,
            writer_ct,
            writer_rt,
            compute_ct,
            "down_union",
            compute_rt=compute_rt if self.balanced else None,
            defines=list(extra_defines)
            + (
                [
                    ("D_NCOLS_ARG", str(5 + len(extra_reader_rt or []))),
                    ("D_NCOLS_WARG", str(5 + len(extra_writer_rt))),
                ]
                if self.balanced
                else []
            ),
        )
        return program, [w_stream, act, union, weights, out]
