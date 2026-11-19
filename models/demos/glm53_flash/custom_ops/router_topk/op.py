# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``RouterTopk``: the fused exact-fp32 MoE router top-k for decode (T <= 32 token rows) as ONE model-local
``ttnn.generic_op`` (kernels in ``kernels/``; contract in README.md).

HF noaux_tc with n_group = 1 (``Glm5NextTextMoE``): from the fp32 sigmoid scores ``s [1,1,T,E]`` (the fused-sigmoid
gate matmul output) and the fp32 bias ``[1,1,1,E]``:

    choice = s + bias ; idx = top-K(choice) ; w = s[idx] / sum(s[idx]) * route_scale

One token row per core (32 cores): the writer pulls the row's 288 scores and the biases into two fp32 tiles, the
compute kernel adds them on the SFPU (fp32 in dest: bit-exact with ``ttnn.add``), the writer ranks the biased
row with an insertion top-K on the fp32 bit patterns (exact ranking, ties lowest column first), normalizes the
unbiased scores in soft fp32 and writes the row's pieces of the requested outputs: the dense bf16 routing matrix
``[1,1,T,E]`` (weights at the picks, 0 elsewhere), ``w [1,1,T,K]`` (fp32 or bf16) and ``idx [1,1,T,K]`` (uint32 or
fp32), which are exactly the tensors ``tt/moe.py::TtRouter.forward`` hands to the expert paths. It replaces the
16-op composite after the gate matmul (add, FillPad, topk, FillPad, slice, gather, FillPad, sum, rdiv, mul,
typecast, 3 untilize, scatter, tilize).

Decode ledger C10 (2026-10-26): the op can also write the routing in the layouts the EP expert path consumes
directly, dropping the five glue ops between the router and ``all_to_all_dispatch`` (untilize x, untilize w, concat
[x | w], typecast idx, untilize idx):

* ``xw`` [T, 1, 1, H + K] bf16 ROW_MAJOR (L1): token row t = its untilized x row followed by its K bf16 routing
  weights (a reader kernel on NCRISC untilizes the row while BRISC ranks);
* ``idx`` [T, 1, 1, K] uint16 ROW_MAJOR (L1): the ids (``allocate_idx_rows``);
* ``idx`` [1, 1, T, 32] bf16 TILE with ``IDX_PAD`` beyond K (``allocate_idx_tiles``): the single-chip EP op's index
  tiles, next to the bf16 ``w`` tile;
* ``perm`` [1, 1, 1, E] uint16 ROW_MAJOR: the EP placement's global -> slot table applied to the ids
  (``custom_ops/expert_stream/plan.py``); the dense row keeps the global columns.
"""

from __future__ import annotations

import struct
from pathlib import Path

import torch

import ttnn

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
TILE = 32
NUM_CBS = 64  # unpack_to_dest_mode entries (>= the number of CBs the build knows)
FP32_TILE_BYTES = TILE * TILE * 4
IDX_PAD = 4096  # the EP index tiles' pad id (custom_ops/expert_stream/op.py IDX_PAD: no chip owns it, exact in bf16)
L1_ALIGN = 16


def f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def bf16_bits(v: float) -> int:
    return int(torch.tensor(float(v)).bfloat16().view(torch.int16).item()) & 0xFFFF


def rm_page_bytes(width_elems: int, elem_bytes: int = 2, align: int = L1_ALIGN) -> int:
    """The aligned page of a ROW_MAJOR interleaved L1 tensor row (one page per row)."""
    return -(-width_elems * elem_bytes // align) * align


class RouterTopk:
    """Builds and launches the fused router program. Outputs are pre-allocated by the caller (``allocate_*``)."""

    CB_S, CB_B, CB_CH, CB_SCR, CB_XS, CB_XROW = 0, 1, 2, 3, 4, 5
    SCRATCH_BYTES = 4096  # >= the writer's 1536 B staging area + the 576 B perm page + 64 B alignment slack
    W_MODES = {None: 0, ttnn.float32: 1, ttnn.bfloat16: 2}
    IDX_MODES = {None: 0, ttnn.uint32: 1, ttnn.float32: 2}
    IDX_MODE_ROWS = 3  # uint16 ROW_MAJOR pages
    IDX_MODE_TILES = 4  # bf16 TILE with IDX_PAD

    def __init__(self, device, E: int = 288, k: int = 8, scale: float = 2.5):
        assert E % TILE == 0 and k <= 16, (E, k)
        self.device = device
        self.E, self.k, self.scale = E, k, float(scale)
        grid = device.compute_with_storage_grid_size()
        gx, gy = int(grid.x), int(grid.y)
        assert gx * gy >= TILE, f"router_topk needs {TILE} cores, grid {gx}x{gy}"
        # rows 0..31 over the grid row-major (x fastest): full grid rows, then the remainder of the next row
        self.cores = [ttnn.CoreCoord(i % gx, i // gx) for i in range(TILE)]
        full, rem = divmod(TILE, gx)
        ranges = []
        if full:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1)))
        if rem:
            ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
        self.core_set = ttnn.CoreRangeSet(ranges)

    # ------------------------------------------------------------------ golden
    @staticmethod
    def golden(scores: torch.Tensor, bias: torch.Tensor, k: int = 8, scale: float = 2.5):
        """scores [T,E] fp32 (sigmoid), bias [E] fp32 -> (idx [T,k] int64, w [T,k] fp32, dense [T,E] bf16).
        Ranking: descending ``scores + bias`` in fp32, ties lowest column first (stable sort); weights
        ``scores[idx] / sum * scale`` (HF order); dense = the weights at their columns, rounded to bf16."""
        scores = scores.float()
        choice = scores + bias.float().reshape(1, -1)
        idx = torch.sort(choice, dim=-1, descending=True, stable=True).indices[:, :k]
        w = torch.gather(scores, 1, idx)
        w = w / w.sum(-1, keepdim=True) * scale
        dense = torch.zeros_like(scores).scatter(1, idx, w).bfloat16()
        return idx, w, dense

    @staticmethod
    def golden_rows(
        x: torch.Tensor,
        scores: torch.Tensor,
        bias: torch.Tensor,
        k: int = 8,
        scale: float = 2.5,
        perm: torch.Tensor | None = None,
    ):
        """The EP row layouts (host model of the X_MODE / IDX_MODE 3 / 4 writers): x [T,H] bf16, scores [T,E] ->
        (xw [T, H + K] bf16 = [x | w_bf16], idx_rows [T, K] int64 (through ``perm`` when given), idx_tiles [T, 32]
        bf16 = the ids as bf16 integers with IDX_PAD beyond K)."""
        idx, w, _ = RouterTopk.golden(scores, bias, k, scale)
        if perm is not None:
            idx = perm.to(torch.int64)[idx]
        T = x.shape[0]
        xw = torch.cat([x.bfloat16(), w.bfloat16()], dim=-1)
        tiles = torch.full((T, TILE), float(IDX_PAD))
        tiles[:, :k] = idx.float()
        return xw, idx, tiles.bfloat16()

    # ------------------------------------------------------------------ outputs
    def _alloc(self, shape, dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, layout, self.device, memory_config)

    def allocate_dense(self, T: int) -> ttnn.Tensor:
        return self._alloc([1, 1, T, self.E], ttnn.bfloat16)

    def allocate_w(self, T: int, dtype=ttnn.float32) -> ttnn.Tensor:
        assert dtype in (ttnn.float32, ttnn.bfloat16), dtype
        return self._alloc([1, 1, T, self.k], dtype)

    def allocate_idx(self, T: int, dtype=ttnn.float32) -> ttnn.Tensor:
        assert dtype in (ttnn.uint32, ttnn.float32), dtype
        return self._alloc([1, 1, T, self.k], dtype)

    def allocate_idx_rows(self, T: int) -> ttnn.Tensor:
        """[T, 1, 1, K] uint16 ROW_MAJOR in L1: the ``all_to_all_dispatch`` id rows (one 16 B page per token)."""
        return self._alloc([T, 1, 1, self.k], ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG)

    def allocate_idx_tiles(self, T: int) -> ttnn.Tensor:
        """[1, 1, T, 32] bf16 TILE: the single-chip EP op's index tiles (ids as bf16 integers, IDX_PAD beyond K)."""
        return self._alloc([1, 1, T, TILE], ttnn.bfloat16)

    def allocate_xw(self, T: int, H: int) -> ttnn.Tensor:
        """[T, 1, 1, H + K] bf16 ROW_MAJOR in L1: the ``all_to_all_dispatch`` token rows ``[x | w]``."""
        return self._alloc([T, 1, 1, H + self.k], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG)

    # ------------------------------------------------------------------ launch
    def __call__(
        self,
        scores: ttnn.Tensor,
        bias: ttnn.Tensor,
        T: int,
        *,
        dense: ttnn.Tensor | None = None,
        w: ttnn.Tensor | None = None,
        idx: ttnn.Tensor | None = None,
        x: ttnn.Tensor | None = None,
        xw: ttnn.Tensor | None = None,
        perm: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """scores [1,1,T<=32,E] fp32 TILE interleaved (padded to one tile row), bias [1,1,1,E] fp32 TILE; T valid
        rows (rows >= T get zeros). Writes the given outputs (at least one): dense [1,1,T,E] bf16, w [1,1,T,k]
        fp32 | bf16, idx [1,1,T,k] uint32 | fp32 TILE, or [T,1,1,k] uint16 ROW_MAJOR (``allocate_idx_rows``), or
        [1,1,T,32] bf16 TILE (``allocate_idx_tiles``); with ``x`` [1,1,T,H] bf16 TILE and ``xw`` (``allocate_xw``)
        the [x | w] row-major pages. ``perm`` [1,1,1,E] uint16 ROW_MAJOR: the ids go through the table. Returns the
        generic op's tensor (the last output)."""
        E, k = self.E, self.k
        assert 1 <= T <= TILE, T
        assert scores.dtype == ttnn.float32 and scores.layout == ttnn.TILE_LAYOUT, (scores.dtype, scores.layout)
        assert list(scores.padded_shape) == [1, 1, TILE, E], scores.padded_shape
        assert bias.dtype == ttnn.float32 and bias.layout == ttnn.TILE_LAYOUT, (bias.dtype, bias.layout)
        assert list(bias.padded_shape) == [1, 1, TILE, E], bias.padded_shape
        assert not scores.memory_config().is_sharded() and not bias.memory_config().is_sharded()
        outs = []
        if dense is not None:
            assert dense.dtype == ttnn.bfloat16 and list(dense.padded_shape) == [1, 1, TILE, E], dense.padded_shape
            outs.append(dense)
        w_mode = 0
        if w is not None:
            assert w.dtype in self.W_MODES and list(w.padded_shape) == [1, 1, TILE, TILE], (w.dtype, w.padded_shape)
            w_mode = self.W_MODES[w.dtype]
            outs.append(w)
        idx_mode = 0
        if idx is not None:
            if idx.layout == ttnn.ROW_MAJOR_LAYOUT:
                assert idx.dtype == ttnn.uint16 and list(idx.shape) == [T, 1, 1, k], (idx.dtype, idx.shape)
                assert (
                    idx.memory_config().buffer_type == ttnn.BufferType.L1
                ), "the id rows live in L1 (the dispatch's input)"
                idx_mode = self.IDX_MODE_ROWS
            elif idx.dtype == ttnn.bfloat16:
                assert list(idx.padded_shape) == [1, 1, TILE, TILE], idx.padded_shape
                idx_mode = self.IDX_MODE_TILES
            else:
                assert idx.dtype in self.IDX_MODES and list(idx.padded_shape) == [1, 1, TILE, TILE], idx.padded_shape
                idx_mode = self.IDX_MODES[idx.dtype]
            outs.append(idx)
        x_mode = 0
        HT = 0
        xw_page = 0
        if xw is not None:
            assert x is not None, "the [x | w] rows need x"
            assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT and not x.memory_config().is_sharded()
            H = x.shape[-1]
            assert H % TILE == 0 and list(x.padded_shape)[-2:] == [TILE, H], (x.padded_shape, H)
            assert xw.dtype == ttnn.bfloat16 and xw.layout == ttnn.ROW_MAJOR_LAYOUT, (xw.dtype, xw.layout)
            assert list(xw.shape) == [T, 1, 1, H + k], (xw.shape, T, H, k)
            assert (
                xw.memory_config().buffer_type == ttnn.BufferType.L1
            ), "the [x | w] rows live in L1 (the dispatch's input)"
            x_mode = 1
            HT = H // TILE
            xw_page = rm_page_bytes(H + k)
            outs.append(xw)
        perm_mode = 0
        if perm is not None:
            assert (
                perm.dtype == ttnn.uint16 and perm.layout == ttnn.ROW_MAJOR_LAYOUT and perm.shape[-1] == E
            ), perm.shape
            perm_mode = 1
        assert outs, "router_topk: at least one output"
        for t in outs:
            assert not t.memory_config().is_sharded()

        def cb(index, size, dtype=ttnn.float32, page=FP32_TILE_BYTES):
            return ttnn.CBDescriptor(
                total_size=size,
                core_ranges=self.core_set,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
            )

        cbs = [
            cb(self.CB_S, FP32_TILE_BYTES),
            cb(self.CB_B, FP32_TILE_BYTES),
            cb(self.CB_CH, FP32_TILE_BYTES),
            cb(self.CB_SCR, self.SCRATCH_BYTES, ttnn.uint32, self.SCRATCH_BYTES),
        ]
        if x_mode:
            xs_bytes = 2 * HT * 64
            cbs.append(cb(self.CB_XS, xs_bytes, ttnn.bfloat16, xs_bytes))
            cbs.append(cb(self.CB_XROW, xw_page, ttnn.bfloat16, xw_page))
        acc = lambda t: ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct = [
            self.CB_S,
            self.CB_B,
            self.CB_CH,
            self.CB_SCR,
            E,
            k,
            f32_bits(self.scale),
            w_mode,
            idx_mode,
            0 if dense is None else 1,
            x_mode,
            perm_mode,
            self.CB_XROW,
            HT,
            xw_page,
            bf16_bits(IDX_PAD),
        ]
        # absent outputs get the scores accessor as a placeholder (never dereferenced: their mode is 0)
        for t in (scores, bias, dense, w, idx, xw, perm):
            writer_ct += acc(scores if t is None else t)
        compute_ct = [self.CB_S, self.CB_B, self.CB_CH]
        writer_rt, compute_rt, reader_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for row, core in enumerate(self.cores):
            writer_rt[core.x][core.y] = [row]
            compute_rt[core.x][core.y] = []
            reader_rt[core.x][core.y] = [row]
        common = [
            scores.buffer_address(),
            bias.buffer_address(),
            0 if dense is None else dense.buffer_address(),
            0 if w is None else w.buffer_address(),
            0 if idx is None else idx.buffer_address(),
            T,
            0 if xw is None else xw.buffer_address(),
            0 if perm is None else perm.buffer_address(),
        ]
        # the score and bias tiles unpack straight into the fp32 dest (SrcA holds 19 bits): exact fp32 add
        compute_cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
        modes = [ttnn.UnpackToDestMode.Default] * NUM_CBS
        for index in (self.CB_S, self.CB_B):
            modes[index] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_cfg.unpack_to_dest_mode = modes
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "writer_router_topk.cpp"),
                core_ranges=self.core_set,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                common_runtime_args=common,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "compute_router_topk.cpp"),
                core_ranges=self.core_set,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_cfg,
            ),
        ]
        io = [scores, bias]
        if x_mode:
            reader_ct = [self.CB_XS, self.CB_XROW, HT, k] + acc(x)
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "reader_router_topk.cpp"),
                    core_ranges=self.core_set,
                    compile_time_args=reader_ct,
                    runtime_args=reader_rt,
                    common_runtime_args=[x.buffer_address(), T],
                    config=ttnn.ReaderConfigDescriptor(),
                )
            )
            io.append(x)
        if perm is not None:
            io.append(perm)
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        return ttnn.generic_op(io + outs, program)
