# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM-streaming decode matmuls of the fused gpt-oss decode layer (one token per device, TP shards).

At decode a projection reads its whole weight to produce one row. ttnn's matmuls carry that row in 32-row tiles
and read the weights tile by tile (the indexed sparse_matmul reaches about 220 GB/s on the 1x4 Blackhole mesh).
These ttnn.generic_op ops (kernels/stream_*.cpp) stream the weights per DRAM bank instead:

* Layout: a weight is stored width-sharded over the DRAM banks, bank b holding a fixed set of output columns,
  column-major (all K tiles of a column contiguous), so a worker core reads its columns as one contiguous byte range
  in large NOC packets (transaction-id pipelined, one column in flight ahead of compute). Expert weights are stored
  expert-major, so the columns of one routed expert are contiguous too.
* Bias: every column carries one extra K tile whose first NBIAS rows hold the BF16 bias as a sum of residual terms
  rounded to the weight format (q1 = q(b), q2 = q(b - q1), ...); the activation's extra 1x32 tile holds the matching
  multipliers (ones, or the routing score for the expert down projection), so the matmul adds the bias.
* Compute: worker cores next to each DRAM bank run custom_mm with 1x32 activation tiles (LoFi, FP32 accumulation).
  The activation is the flat normed hidden of the layer boundary (tt/decode_boundary.py: value h at byte 2 h, so its
  32-value groups are the 1x32 tiles, read in one piece), or for o_proj the SDPA output gathered head by head.

Ops: LinearStream (dense projection + bias: QKV straight into the head-split layout, o_proj straight into the flat
partial sum the boundary all-reduces, router + top-k + softmax), ExpertGateUpStream (routed gate|up + bias + SwiGLU,
scaled by the routing score) and ExpertDownStream (routed down + bias, summed over the routed experts, into the flat
partial sum).
"""

import struct
from pathlib import Path

import torch

import ttnn

from ..fused_decode import DECODE_BOUNDARY_PREFETCH_ALL

KERNEL_DIR = Path(__file__).parent / "kernels"
NBIAS = 4  # residual terms carrying a bias (BFP4: relative error ~ 8^-NBIAS of the 16-value group maximum)
TILE = ttnn.TILE_SIZE
BFP4_TILE_BYTES = 576
BFP8_TILE_BYTES = 1088
TILE_BYTES = {ttnn.bfloat4_b: BFP4_TILE_BYTES, ttnn.bfloat8_b: BFP8_TILE_BYTES, ttnn.bfloat16: 2048}
TINY_BYTES = 2 * TILE  # one 1x32 BF16 tile
MAX_PACKET_BYTES = 16384  # Blackhole NOC max burst


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _bfp4_round(tiles):
    """Round [n, 32, 32] tiles exactly as the device stores them in BFP4 (host conversion)."""
    t = ttnn.from_torch(tiles.float(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(t).float().reshape(tiles.shape)


def bias_tiles(bias):
    """[..., 32] BF16 bias rows -> [..., 32, 32] tiles whose first NBIAS rows sum (in BFP4) to the bias."""
    lead = bias.shape[:-1]
    b = bias.reshape(-1, TILE).float()
    out = torch.zeros(b.shape[0], TILE, TILE)
    r = b
    for i in range(NBIAS):
        t = torch.zeros(b.shape[0], TILE, TILE)
        t[:, 0] = r
        q = _bfp4_round(t)[:, 0]
        out[:, i] = q
        r = r - q
    return out.reshape(*lead, TILE, TILE)


def _columns_with_bias(w, bias):
    """w [E, K, N], bias [E, N] -> [E, n_tiles, k_tiles + 1, 32, 32] tile columns (bias tile last)."""
    E, K, N = w.shape
    n_tiles, k_tiles = N // TILE, K // TILE
    cols = w.reshape(E, k_tiles, TILE, n_tiles, TILE).permute(0, 3, 1, 2, 4)
    b = bias_tiles(bias.reshape(E, n_tiles, TILE)).to(cols.dtype)
    return torch.cat([cols, b.unsqueeze(2)], dim=2)


def _bank_columns(cols, banks):
    """[E, C * banks, k, 32, 32] -> [rows, banks * 32]: bank b's shard = its C columns as one column of tiles."""
    C = cols.shape[1] // banks
    return torch.cat([cols[:, b * C : (b + 1) * C].reshape(-1, TILE) for b in range(banks)], dim=-1)


def columns_per_bank(n, banks):
    return -(-n // TILE // banks)


def stream_gate_up_layout(gate_up, gate_up_bias, banks):
    """One TP shard of [E, hidden, 2 * I_pad] packed [gate | up] weights and [E, 2 * I_pad] biases -> [rows, banks * 32]:
    bank b's shard is a column of tiles ordered expert -> (gate j, up j) for the P pairs j in [b * P, (b + 1) * P)
    -> K tile (hidden / 32 weight tiles + 1 bias tile)."""
    E, H, W = gate_up.shape
    half = W // TILE // 2
    P = half // banks
    assert P * banks == half, f"{half} column pairs do not split over {banks} banks"
    cols = _columns_with_bias(gate_up, gate_up_bias)
    order = [c for bank in range(banks) for j in range(bank * P, (bank + 1) * P) for c in (j, half + j)]
    return _bank_columns(cols[:, order], banks)


def stream_down_layout(down, down_bias, banks):
    """One TP shard of [E, I_pad, hidden] down weights and [E, hidden] biases -> [rows, banks * 32]: bank b holds output
    columns [b * C, (b + 1) * C) (C = ceil(hidden / 32 / banks), zero padded), ordered expert -> column -> K tile
    (I_pad / 32 weight tiles + 1 bias tile)."""
    E, K, N = down.shape
    pad = columns_per_bank(N, banks) * banks * TILE - N
    cols = _columns_with_bias(torch.nn.functional.pad(down, (0, pad)), torch.nn.functional.pad(down_bias, (0, pad)))
    return _bank_columns(cols, banks)


def stream_linear_layout(w, bias, banks):
    """[K, N] weights and [N] bias -> [rows, banks * 32]: bank b holds output columns [b * C, (b + 1) * C)
    (C = ceil(N / 32 / banks), zero padded), ordered column -> K tile (K / 32 weight tiles + 1 bias tile)."""
    K, N = w.shape
    pad = columns_per_bank(N, banks) * banks * TILE - N
    cols = _columns_with_bias(
        torch.nn.functional.pad(w, (0, pad)).unsqueeze(0), torch.nn.functional.pad(bias, (0, pad)).unsqueeze(0)
    )
    return _bank_columns(cols, banks)


def linear_stream_rows(mesh_device, k, n):
    return columns_per_bank(n, mesh_device.dram_grid_size().x) * (k // TILE + 1) * TILE


def stream_weight_memory_config(mesh_device, rows):
    banks = mesh_device.dram_grid_size().x
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
            (rows, TILE),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )


def as_stream_tensor(mesh_device, layouts, rows, dtype, cache_file_name):
    """Per-device streamed layouts ([rows, banks * 32] each, one per mesh column / TP shard, or None to load from the
    cache) -> one DRAM width-sharded tensor per device. The cached file keeps the shard layout, so cache names carry
    the layout parameters."""
    t = torch.stack(layouts).unsqueeze(0) if layouts is not None else None
    return ttnn.as_tensor(
        t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, 1)),
        cache_file_name=cache_file_name,
        memory_config=stream_weight_memory_config(mesh_device, rows),
    )


def _page_bytes(tiles, tile_bytes):
    """Largest NOC packet holding whole tiles that divides a `tiles`-tile block."""
    return max(d for d in range(1, tiles + 1) if tiles % d == 0 and d * tile_bytes <= MAX_PACKET_BYTES) * tile_bytes


class _BankStreamOp:
    """Worker placement shared by the streaming ops: `readers` cores per DRAM bank (the first `active_banks` banks),
    starting at the bank's optimal NOC0 reader core and growing along its row, each with a NOC virtual channel
    distinct from earlier cores on the same row (as the DeepSeek DRAM-streaming matmul assigns them). The cores are
    coalesced into rectangles: dispatch writes kernel binaries and arguments per core range, and single-core ranges
    made multi-reader programs measurably slower to launch under trace."""

    def __init__(self, mesh_device, readers, active_banks=None):
        self.mesh_device = mesh_device
        self.banks = mesh_device.dram_grid_size().x
        self.readers = readers
        bank_workers = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
        assert len(bank_workers) == self.banks
        bank_workers = bank_workers[: active_banks or self.banks]
        grid = mesh_device.compute_with_storage_grid_size()
        used = {(c.x, c.y) for c in bank_workers}
        placed = []
        for bank, core in enumerate(bank_workers):
            cores = [core]
            dx = 1
            while len(cores) < readers:
                assert dx < grid.x, f"no room for {readers} readers next to bank {bank} worker {core}"
                for x in (core.x + dx, core.x - dx):
                    if 0 <= x < grid.x and (x, core.y) not in used and len(cores) < readers:
                        cores.append(ttnn.CoreCoord(x, core.y))
                        used.add((x, core.y))
                dx += 1
            placed += [(c, bank, r) for r, c in enumerate(cores)]
        vcs = []
        for i, (core, bank, r) in enumerate(placed):
            vc = (bank + r) & 0x3
            for j in range(i):
                if placed[j][0].y == core.y and vcs[j] == vc:
                    vc = (vc + 1) & 0x3
            vcs.append(vc)
        self.assignments = [(c, b, r, v) for (c, b, r), v in zip(placed, vcs)]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(self.assignments[0][0], self.assignments[0][0])])
        for c, *_ in self.assignments[1:]:
            self.cores = self.cores.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))

    def _cb(self, index, dtype, page, pages, tiny=False):
        kwargs = dict(buffer_index=index, data_format=dtype, page_size=page)
        if tiny:
            kwargs["tile"] = ttnn.TileDescriptor(ttnn.Tile([1, TILE]))
        return ttnn.CBDescriptor(
            total_size=page * pages, core_ranges=self.cores, format_descriptors=[ttnn.CBFormatDescriptor(**kwargs)]
        )

    def _program(self, name, cbs, reader_ct, reader_rt, writer_ct, writer_rt, compute_ct, compute="matmul"):
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"stream_{name}_reader.cpp"),
                core_ranges=self.cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"stream_{name}_writer.cpp"),
                core_ranges=self.cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / f"stream_{compute}_compute.cpp"),
                core_ranges=self.cores,
                compile_time_args=compute_ct,
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, math_approx_mode=False
                ),
            ),
        ]
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)


class ExpertGateUpStream(_BankStreamOp):
    """(x, ids, scores, weights, act) -> act. x: the flat BF16 normed hidden (tt/decode_boundary.py); ids / scores: UINT16 / BF16 row-major buffers whose first k entries are the routed experts and their
    softmax weights; act: [k, I_pad] BF16 row-major L1, receives w_e * SwiGLU(gate_e, up_e) per routed expert e."""

    def __init__(self, mesh_device, hidden, inter_pad, num_sel, swiglu_limit, alpha, readers):
        super().__init__(mesh_device, readers)
        self.kx = hidden // TILE
        self.kt = self.kx + 1
        self.num_sel = num_sel
        self.inter_pad = inter_pad
        self.P = inter_pad // TILE // self.banks
        assert self.P % readers == 0, f"{self.P} pairs per bank do not split over {readers} readers"
        self.pairs = self.P // readers
        self.limit, self.alpha = swiglu_limit, alpha
        self.col_bytes = self.kt * BFP4_TILE_BYTES
        self.expert_stride = 2 * self.P * self.col_bytes
        self.page_bytes = _page_bytes(self.kt, BFP4_TILE_BYTES)

    def weight_rows(self, num_experts):
        return num_experts * 2 * self.P * self.kt * TILE

    def __call__(self, x, ids, scores, weights, act):
        cb_x, cb_w, cb_idx, cb_scr, cb_act = 0, 1, 2, 3, 16
        cbs = [
            self._cb(cb_x, ttnn.bfloat16, TINY_BYTES, self.kt, tiny=True),
            self._cb(cb_w, ttnn.bfloat4_b, BFP4_TILE_BYTES, 3 * self.kt),
            self._cb(cb_idx, ttnn.uint16, 64, 1),
            self._cb(cb_scr, ttnn.bfloat16, 64, 1),
            self._cb(cb_act, ttnn.bfloat16, TINY_BYTES, 4, tiny=True),
        ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [
                weights.buffer_address(),
                ids.buffer_address(),
                bank,
                vc,
                r * 2 * self.pairs * self.col_bytes,
            ]
            writer_rt[core.x][core.y] = [
                x.buffer_address(),
                act.buffer_address(),
                bank * self.P + r * self.pairs,
                scores.buffer_address(),
            ]
        reader_ct = [cb_w, cb_idx, self.kt, 2 * self.pairs, self.num_sel, self.expert_stride, BFP4_TILE_BYTES]
        reader_ct += [self.page_bytes, 64] + ttnn.TensorAccessorArgs(ids).get_compile_time_args()
        writer_ct = [
            cb_x,
            cb_act,
            self.kx,
            NBIAS,
            self.pairs,
            self.num_sel,
            self.inter_pad * 2,
            TILE * TILE * 2,
            cb_scr,
        ]
        for t in (x, act, scores):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_x, cb_w, cb_act, self.kt, self.pairs, self.num_sel]
        compute_ct += [_f32_bits(v) for v in (self.limit, -self.limit, -3.0e38, self.alpha, 1.0 / self.alpha, 1.0)]
        program = self._program("gate_up", cbs, reader_ct, reader_rt, writer_ct, writer_rt, compute_ct, "gate_up")
        return ttnn.generic_op([weights, x, ids, scores, act], program)


class ExpertDownStream(_BankStreamOp):
    """(act, ids, scores, weights, out) -> out. act: [k, I_pad] BF16 row-major (score-weighted SwiGLU outputs);
    out: the flat BF16 partial sum (tt/decode_boundary.py: value h at byte 2 h) receives
    sum_e w_e * (act_e @ Wd_e + bd_e) (one matmul per output column over the K = k segments [w_e * act_e | w_e] of
    the routed experts)."""

    def __init__(self, mesh_device, hidden, inter_pad, num_sel, readers):
        super().__init__(mesh_device, readers)
        self.hidden = hidden
        self.num_sel = num_sel
        self.inter_pad = inter_pad
        self.seg_tiles = inter_pad // TILE + 1
        self.C = columns_per_bank(hidden, self.banks)
        assert self.C % readers == 0, f"{self.C} columns per bank do not split over {readers} readers"
        self.cols = self.C // readers
        self.seg_bytes = self.seg_tiles * BFP4_TILE_BYTES
        assert self.seg_bytes <= MAX_PACKET_BYTES, "one expert column segment must fit one NOC packet"
        self.expert_stride = self.C * self.seg_bytes

    def weight_rows(self, num_experts):
        return num_experts * self.C * self.seg_tiles * TILE

    def __call__(self, act, ids, scores, weights, out, send=None):
        """send = (boundary, site): fuse the boundary all-reduce's fabric send (decode_boundary.py sending_program)."""
        cb_in0, cb_w, cb_idx, cb_scr, cb_out = 0, 1, 2, 3, 16
        kt = self.num_sel * self.seg_tiles
        cbs = [
            self._cb(cb_in0, ttnn.bfloat16, TINY_BYTES, kt, tiny=True),
            self._cb(cb_w, ttnn.bfloat4_b, BFP4_TILE_BYTES, 3 * kt),
            self._cb(cb_idx, ttnn.uint16, 64, 1),
            self._cb(cb_scr, ttnn.bfloat16, 128, 1),
            self._cb(cb_out, ttnn.bfloat16, TINY_BYTES, 4, tiny=True),
        ]
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for core, bank, r, vc in self.assignments:
            reader_rt[core.x][core.y] = [
                weights.buffer_address(),
                ids.buffer_address(),
                bank,
                vc,
                r * self.cols * self.seg_bytes,
            ]
            writer_rt[core.x][core.y] = [
                act.buffer_address(),
                scores.buffer_address(),
                out.buffer_address(),
                bank * self.C + r * self.cols,
            ]
        reader_ct = [cb_w, cb_idx, self.seg_tiles, self.cols, self.num_sel, self.expert_stride, BFP4_TILE_BYTES]
        reader_ct += ttnn.TensorAccessorArgs(ids).get_compile_time_args()
        writer_ct = [cb_in0, cb_out, cb_scr, self.seg_tiles, self.num_sel, NBIAS, self.cols, self.hidden // TILE]
        writer_ct += [self.inter_pad * 2]
        for t in (act, scores, out):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct += send[0].notify_args() if send else [0, 0, 0, 0]
        compute_ct = [cb_in0, cb_w, cb_out, kt, self.cols]

        def build():
            return self._program("down", cbs, reader_ct, reader_rt, writer_ct, writer_rt, compute_ct)

        program = send[0].sending_program(build, out, send[1], self.cores) if send else build()
        return ttnn.generic_op([weights, act, ids, scores, out], program)


class LinearStream(_BankStreamOp):
    """Dense decode linear y = x @ W + b for one token, weights streamed per DRAM bank (stream_linear_layout).

    x: x_pages = 0: the flat BF16 normed hidden of the layer boundary (tt/decode_boundary.py); otherwise a BF16
    32x32-tile tensor (any TensorAccessor layout; a DRAM source is staged in L1 first) whose activation is row
    (j / x_pages) of tile page (j % x_pages) for j < k_tiles (x_pages = 2: the [heads, 64] SDPA output, head by head).
    prefetch_cols: cap on the weight columns buffered ahead (default: all of them when a boundary is fused in, else 3).
    out_mode:
      3: attention heads, out = (Q, K, V) BF16 [heads, head_tiles * 32] tile tensors, head h in row h,
         heads = (q_cols, k_cols) output tiles of Q and K;
      4: router top-k + softmax, out = (ids, scores) [1, 32] UINT16 / BF16 row-major buffers, heads = (k, n_experts);
      5: out = the flat BF16 partial sum the boundary all-reduces (output column n -> bytes [64 n, 64 n + 64)).
    """

    def __init__(
        self,
        mesh_device,
        k_tiles,
        n,
        weight_dtype,
        readers=1,
        x_pages=None,
        out_mode=3,
        out_page_bytes=2048,
        heads=(0, 0),
        head_tiles=2,
        prefetch_cols=None,
    ):
        n_tiles = -(-n // TILE)
        banks = mesh_device.dram_grid_size().x
        self.C = -(-n_tiles // banks)
        super().__init__(mesh_device, readers, active_banks=-(-n_tiles // self.C))
        assert self.C % readers == 0, f"{self.C} columns per bank do not split over {readers} readers"
        self.cols = self.C // readers
        self.n_tiles = n_tiles
        self.kx = k_tiles
        self.kt = k_tiles + 1
        self.x_pages = 0 if x_pages is None else x_pages
        assert out_mode in (3, 4, 5), f"out_mode {out_mode}"
        self.out_mode = out_mode
        self.out_page_bytes = out_page_bytes
        self.heads = heads
        self.head_tiles = head_tiles
        self.prefetch_cols = prefetch_cols
        self.weight_dtype = weight_dtype
        self.tile_bytes = TILE_BYTES[weight_dtype]
        self.col_bytes = self.kt * self.tile_bytes
        self.page_bytes = _page_bytes(self.kt, self.tile_bytes)

    def weight_rows(self):
        return self.C * self.kt * TILE

    def __call__(self, x, weights, out, send=None, exchange=None):
        """send = (boundary, site), out_mode 5: fuse the boundary all-reduce's fabric send (decode_boundary.py
        sending_program). x may be a decode_boundary.PendingBoundary: the boundary producing x then runs inside this
        op (DecodeBoundary.consumer_parts) while the weight readers stream ahead. exchange (out_mode 3): an object
        whose writer_args() / program() add the fused decode LM head's top-k candidate merge and exchange
        (decode_terminal.py TerminalExchange)."""
        pending = None
        if not isinstance(x, ttnn.Tensor):
            pending = x
            fused_kernels, fused_cbs, fused_semaphores, fused_io = pending.boundary.consumer_parts(pending, self.cores)
            x = pending.x
        if self.out_mode == 3:
            outs = tuple(out)
        elif self.out_mode == 4:
            outs = (out[0], out[1], out[1])
        else:
            outs = (out,) * 3
        cb_x, cb_w, cb_scr, cb_stage, cb_out = 0, 1, 3, 4, 16
        stage = self.x_pages if x.memory_config().buffer_type == ttnn.BufferType.DRAM else 0
        # A fused boundary's consumer can buffer all its weight columns: the readers stream them while the boundary
        # runs (fused_decode.DECODE_BOUNDARY_PREFETCH_ALL).
        w_cols = max(3, self.cols) if pending is not None and DECODE_BOUNDARY_PREFETCH_ALL else 3
        if self.prefetch_cols is not None:
            # Large columns (the LM head's ~97 KB): cap the ring at what L1 holds.
            w_cols = min(w_cols, max(3, self.prefetch_cols))
        cbs = [
            self._cb(cb_x, ttnn.bfloat16, TINY_BYTES, self.kt, tiny=True),
            self._cb(cb_w, self.weight_dtype, self.tile_bytes, w_cols * self.kt),
            self._cb(cb_scr, ttnn.bfloat16, 64, 1),
        ]
        if stage:
            cbs.append(self._cb(cb_stage, ttnn.bfloat16, 2048, stage))
        cbs.append(self._cb(cb_out, ttnn.bfloat16, TINY_BYTES, 4, tiny=True))
        reader_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, (core, bank, r, vc) in enumerate(self.assignments):
            reader_rt[core.x][core.y] = [weights.buffer_address(), bank, vc, r * self.cols * self.col_bytes]
            writer_rt[core.x][core.y] = [
                x.buffer_address(),
                outs[0].buffer_address(),
                bank * self.C + r * self.cols,
                outs[1].buffer_address(),
                outs[2].buffer_address(),
            ] + (exchange.writer_runtime_args(i) if exchange is not None else [])
        reader_ct = [cb_w, self.kt, self.cols, self.tile_bytes, self.page_bytes, w_cols]
        writer_ct = [cb_x, cb_out, cb_scr, self.kx, self.x_pages, NBIAS, self.cols, self.n_tiles, self.out_mode]
        writer_ct += [self.out_page_bytes, stage, cb_stage, self.heads[0], self.heads[1], self.head_tiles]
        for t in (x, *outs):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct += send[0].notify_args() if send else [0, 0, 0, 0]
        writer_ct += [1 if pending is not None else 0]
        writer_ct += exchange.writer_compile_args() if exchange is not None else [0, 0, 0, 0]
        compute_ct = [cb_x, cb_w, cb_out, self.kt, self.cols]

        def build():
            program = self._program("linear", cbs, reader_ct, reader_rt, writer_ct, writer_rt, compute_ct)
            if pending is not None:
                program.kernels = list(program.kernels) + fused_kernels
                program.cbs = list(program.cbs) + fused_cbs
                program.semaphores = fused_semaphores
            return program

        assert not (send and pending), "an op is either a boundary producer or its consumer"
        assert not (send and exchange), "the LM-head exchange and a boundary send do not combine"
        if exchange is not None:
            assert self.out_mode == 3
            program = exchange.program(build, self.cores)
        else:
            program = send[0].sending_program(build, out, send[1], self.cores) if send else build()
        unique = [t for i, t in enumerate(outs) if all(t is not o for o in outs[:i])]
        extra = [t for t in fused_io if all(t is not u for u in (x, *unique))] if pending is not None else []
        if exchange is not None:
            extra += [t for t in exchange.io if all(t is not u for u in (x, *unique, *extra))]
        ttnn.generic_op([weights, x, *unique, *extra], program)
        return out
