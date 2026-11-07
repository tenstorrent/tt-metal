# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-1 routed-expert decode ops streaming the ND-sharded expert tensors that prefill also reads (TP=1, one chip;
gpt-oss-120b at TP=4, whose 128 experts leave no room for decode-only copies).

On one Blackhole chip (P150) the BFP8 experts take ~20 GB of the 32 GB DRAM, so there is no room for decode-only
expert copies next to them: the experts are stored once, in the ND-sharded "balanced" layout below, and both the
prefill / batched-decode sparse matmuls (through their TensorAccessor) and these streaming ops read them.

Layout (nd_expert_memory_config): gate [1, E, H, I], up [1, E, H, I] and down [1, E, I, H] are ND-sharded in
[1, 1, K, 32] shards (one per expert and output tile column) dealt round-robin over the DRAM banks, so the K tiles of
one (expert, column) are one contiguous run of one bank, which kernels/nd_reader.cpp streams in large packets into
custom_mm in1 blocks (kernels/nd_compute.cpp). Every bank holds the same number of shards (no padding).

Bias: as in fused_decode/stream.py, an extra K tile per column (NBIAS BFP4-rounded residual terms) times an activation
tile of ones (gate|up) or of the routing score (down); the bias tiles live in a small decode-only [1, E, 32, N] tensor
sharded like the weights.

Ops (one token), one core per output tile column (gate|up at TP=4: per (column, routed expert)):
  NdExpertGateUp: (x, ids, scores, gate, up, bias, act) -> act [k, I] = w_e * SwiGLU(gate_e, up_e)
  NdExpertDown:   (act, ids, scores, down, bias, out) -> the flat partial sum (fused_decode/boundary.py)
"""

import struct
from pathlib import Path

import torch

import ttnn

from .boundary import BOUNDARY_CORE
from .stream import NBIAS, TILE_BYTES, bias_tiles

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = ttnn.TILE_SIZE
TINY_BYTES = 2 * TILE

# Weight rows per CB slot (90 = 5 x 18), slots in flight per stream, tiles per NOC packet (swept on one chip: 18 / 4 / 9
# stream gate|up at ~489 GB/s and down at ~448).
ND_K_BLOCK = 18
ND_DEPTH = 4
ND_PACKET_TILES = 9
# Split accumulation over the K blocks and full DST sync in the compute kernel.
ND_SPLIT_ACC = True
ND_FULL_SYNC = True
# Weight streams per core: 2 = the reader (NOC0) streams the even K blocks, the writer (NOC1) the odd ones.
ND_STREAMS = 2


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def nd_expert_memory_config(mesh_device, k, width=TILE):
    """[1, 1, k, width] DRAM shards dealt round-robin over the banks: shard (e, c) of an [1, E, k, N] tensor (N / width
    columns) lands in bank (e * N / width + c) % banks."""
    dram = mesh_device.dram_grid_size()
    return ttnn.MemoryConfig(
        ttnn.BufferType.DRAM,
        ttnn.NdShardSpec(
            ttnn.Shape([1, 1, k, width]),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))}),
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )


# Bias tiles packed ND_BIAS_PACK columns to a tile (TP > 1): column c's NBIAS residual terms in rows
# [NBIAS (c % 8), NBIAS (c % 8 + 1)) of packed tile c / 8, the activation's bias multipliers at those positions
# (kernels ND_BIAS_PACK). Same values, same BFP groups (one K row of 16 output values), 8x fewer bias bytes: the
# gpt-oss-120b BFP8 bias tiles would otherwise take 0.68 GB per device.
ND_BIAS_PACK = TILE // NBIAS


def nd_bias_tensor(mesh_device, bias, num_experts, cache_file_name=None, dtype=ttnn.bfloat8_b, tp=1, parts=1, pack=1):
    """[E, N] bias (torch, or None to load the cache) -> [1, E, 32, N] tensor ND-sharded in [1, 1, 32, 32] shards like
    the weights, whose tile (e, n) holds the NBIAS residual terms of column n's bias in its first rows. tp > 1: bias is
    [E, tp * N], device d's [E, N] side by side, and the tensor is sharded over the mesh columns. pack > 1: each of a
    device's `parts` equal column ranges (gate | up: 2) packed `pack` columns to a tile (ND_BIAS_PACK)."""
    t = None
    if bias is not None:
        E, N = bias.shape
        tiles = bias_tiles(bias.float().reshape(E, N // TILE, TILE))  # [E, n, 32, 32]
        if pack > 1:
            assert pack * NBIAS == TILE
            n_part = tiles.shape[1] // (tp * parts)
            q = -(-n_part // pack)
            out = torch.zeros(E, tp * parts * q, TILE, TILE)
            for i in range(tp * parts):
                src = torch.zeros(E, q * pack, NBIAS, TILE)
                src[:, :n_part] = tiles[:, i * n_part : (i + 1) * n_part, :NBIAS]
                out[:, i * q : (i + 1) * q] = src.reshape(E, q, TILE, TILE)  # row NBIAS j + k = column 8 t + j, term k
            tiles = out
        t = tiles.permute(0, 2, 1, 3).reshape(1, E, TILE, tiles.shape[1] * TILE)
    mapper = (
        ttnn.ReplicateTensorToMesh(mesh_device)
        if tp == 1
        else ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, 3))
    )
    return ttnn.as_tensor(
        t,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        mesh_mapper=mapper,
        cache_file_name=cache_file_name,
        memory_config=nd_expert_memory_config(mesh_device, TILE),
    )


def default_cores(mesh_device, n, exclude=()):
    """The first n cores of the grid in row-major order, skipping `exclude` ((x, y) pairs)."""
    grid = mesh_device.compute_with_storage_grid_size()
    ex = {tuple(c) for c in exclude}
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x) if (x, y) not in ex]
    assert n <= len(cores), f"{n} cores > grid {grid} minus {list(exclude)}"
    return cores[:n]


# A K run of at most this many tiles is streamed as one block (TP=4 down: 23 tiles, a prime: no smaller divisor).
ND_ONE_BLOCK_TILES = 24


def even_k_block(kt, cap=None):
    """The largest divisor of kt up to cap (default ND_K_BLOCK): weight rows per CB slot; the whole run when kt is
    at most ND_ONE_BLOCK_TILES."""
    if cap is None and kt <= ND_ONE_BLOCK_TILES:
        return kt
    cap = cap or ND_K_BLOCK
    return max(d for d in range(1, min(cap, kt) + 1) if kt % d == 0)


def nd_experts_per_core(mesh_device, n_cols, num_sel, exclude=1):
    """Routed experts per gate|up core: 1 (one core per (column, expert)) when the grid holds n_cols * num_sel cores
    (TP=4: 23 columns x 4 = 92), else all of them (TP=1: 90 columns)."""
    grid = mesh_device.compute_with_storage_grid_size()
    return 1 if n_cols * num_sel <= grid.x * grid.y - exclude else num_sel


class _NdOp:
    """groups: output-column shards (col0 tiles); experts_per_core: routed experts per core (gate|up splits the k
    experts over cores; down keeps all k on one core to sum them)."""

    def __init__(
        self,
        mesh_device,
        kt,
        kt_all,
        nt,
        bias_nt,
        n_out,
        g,
        num_sel,
        mode,
        up_off,
        experts_per_core,
        weight_dtype=ttnn.bfloat8_b,
        kb=None,
        depth=ND_DEPTH,
        pkt=ND_PACKET_TILES,
        streams=ND_STREAMS,
        exclude_cores=(BOUNDARY_CORE,),
        bias_pack=1,
    ):
        kb = kb or even_k_block(kt)
        # bias_pack > 1 (ND_BIAS_PACK): bias_nt / up_off in packed tiles
        self.bias_pack = bias_pack
        if bias_pack > 1:
            parts = 2 if mode == 1 else 1
            q = -(-nt // bias_pack) if mode == 1 else -(-bias_nt // bias_pack)
            bias_nt, up_off = parts * q, (q if mode == 1 else 0)
        self.mesh_device = mesh_device
        self.kt, self.kt_all, self.nt, self.bias_nt, self.n_out, self.g = kt, kt_all, nt, bias_nt, n_out, g
        self.num_sel, self.mode, self.up_off = num_sel, mode, up_off
        self.kb, self.depth, self.pkt, self.streams = kb, depth, pkt, streams
        assert kt % self.kb == 0
        n_groups = -(-n_out // g)
        self.groups = [i * g for i in range(n_groups)]
        assert num_sel % experts_per_core == 0
        self.epc = experts_per_core
        work = [(col0, e0) for col0 in self.groups for e0 in range(0, num_sel, experts_per_core)]
        # per-expert shards: the bank depends on the routed expert, so there is no static bank-adjacent placement
        cores = default_cores(mesh_device, len(work), exclude_cores)
        self.work = list(zip(cores, work))
        # Coalesced into rectangles: dispatch writes binaries and arguments per core range.
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[0])])
        for c in cores[1:]:
            self.cores = self.cores.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))
        assert weight_dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b), f"ND expert dtype {weight_dtype}"
        self.weight_dtype = weight_dtype
        self.tile_bytes = TILE_BYTES[weight_dtype]
        self.sets = 2 if mode == 1 else 1

    def _cb(self, index, dtype, page, pages, tiny=False):
        kwargs = dict(buffer_index=index, data_format=dtype, page_size=page)
        if tiny:
            kwargs["tile"] = ttnn.TileDescriptor(ttnn.Tile([1, TILE]))
        return ttnn.CBDescriptor(
            total_size=page * pages, core_ranges=self.cores, format_descriptors=[ttnn.CBFormatDescriptor(**kwargs)]
        )

    def bytes_per_call(self):
        """Weight + bias bytes streamed per call (all routed experts)."""
        cols = len(self.groups) * self.g
        return self.num_sel * self.sets * cols * (self.kt + 1) * self.tile_bytes

    def _run(self, inp, out, ids, scores, w, u, b, in_page, out_page, compute_extra, send=None):
        """send = (boundary, site): run the boundary all-reduce's fabric sender (boundary.DecodeBoundary
        sending_program; TP > 1 only)."""
        cb_in0, cb_w, cb_idx, cb_scr, cb_out, cb_w2, cb_idx2 = 0, 1, 2, 3, 16, 5, 6
        seg = self.kt + 1
        in0_tiles = seg * (1 if self.mode == 1 else self.num_sel)
        slot_tiles = self.kb * self.g
        out_tiles = self.g * (self.epc if self.mode == 1 else 1)
        cbs = [
            self._cb(cb_in0, ttnn.bfloat16, TINY_BYTES, in0_tiles, tiny=True),
            self._cb(cb_w, self.weight_dtype, self.tile_bytes, self.depth * slot_tiles),
            self._cb(cb_idx, ttnn.uint16, 64, 1),
            self._cb(cb_scr, ttnn.bfloat16, 64, 1),
            self._cb(cb_out, ttnn.bfloat16, TINY_BYTES, out_tiles, tiny=True),
        ]
        if self.streams == 2:
            cbs += [
                self._cb(cb_w2, self.weight_dtype, self.tile_bytes, self.depth * slot_tiles),
                self._cb(cb_idx2, ttnn.uint16, 64, 1),
            ]
        reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for i, (core, (col0, e0)) in enumerate(self.work):
            n_groups = 1
            reader_rt[core.x][core.y] = [
                w.buffer_address(),
                u.buffer_address(),
                b.buffer_address(),
                ids.buffer_address(),
                i & 3,
                n_groups,
                e0,
                col0,
            ]
            compute_rt[core.x][core.y] = [1]
            writer_rt[core.x][core.y] = [
                inp.buffer_address(),
                out.buffer_address(),
                scores.buffer_address(),
                n_groups,
                e0,
                w.buffer_address(),
                u.buffer_address(),
                b.buffer_address(),
                ids.buffer_address(),
                (i + 1) & 3,
                col0,
            ]
        nsel = self.epc if self.mode == 1 else self.num_sel
        stream_ct = [self.kt, self.kt_all, self.nt, self.bias_nt, self.kb, self.g, nsel, self.sets, self.up_off]
        reader_ct = [cb_w, cb_idx, self.tile_bytes] + stream_ct + [self.depth, self.pkt, self.streams]
        for t in (w, u, b, ids):
            reader_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        compute_ct = [cb_in0, cb_w, cb_out, self.kt, self.kb, self.g, nsel, self.mode] + compute_extra
        compute_ct += [int(ND_SPLIT_ACC), cb_w2, self.streams]
        writer_ct = [cb_in0, cb_out, cb_scr, self.kt, NBIAS, nsel, in_page, out_page, self.mode, self.g, self.n_out]
        for t in (inp, out, scores):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct += [self.streams, cb_w2, cb_idx2, self.tile_bytes, self.kt_all, self.nt, self.bias_nt, self.kb]
        writer_ct += [self.sets, self.up_off, self.depth, self.pkt]
        for t in (w, u, b, ids):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct += send[0].notify_args() if send else [0, 0, 0, 0]
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "nd_reader.cpp"),
                core_ranges=self.cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "nd_writer.cpp"),
                core_ranges=self.cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "nd_compute.cpp"),
                core_ranges=self.cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    fp32_dest_acc_en=True,
                    math_approx_mode=False,
                    dst_full_sync_en=ND_FULL_SYNC,
                ),
            ),
        ]

        if self.bias_pack > 1:
            for k in kernels:
                k.defines = list(k.defines or []) + [("ND_BIAS_PACK", str(self.bias_pack))]

        def build():
            return ttnn.ProgramDescriptor(kernels=list(kernels), semaphores=[], cbs=cbs)

        program = send[0].sending_program(build, out, send[1], self.cores) if send else build()
        io = [w, b, inp, ids, scores, out] + ([u] if u is not w else [])
        ttnn.generic_op(io, program)
        return out


class NdExpertGateUp(_NdOp):
    """(x, ids, scores, gate, up, bias, act) -> act. x: the flat BF16 normed hidden (fused_decode/boundary.py);
    ids / scores: the router's [1, 32] UINT16 / BF16 buffers; gate / up: the ND-sharded [1, E, H, I] expert tensors
    (I: this device's intermediate shard, whole tiles); bias: nd_bias_tensor of the packed [E, 2 I] = [gate | up]
    bias; act: [k, I] BF16 row-major L1."""

    def __init__(self, mesh_device, hidden, inter, num_sel, swiglu_limit, alpha, experts_per_core=None, **kw):
        g = 1
        assert inter % TILE == 0, f"intermediate shard {inter}: whole tiles"
        experts_per_core = experts_per_core or nd_experts_per_core(mesh_device, inter // TILE, num_sel)
        if experts_per_core == 1:
            # one (column, expert) per core (TP=4): 30-row slots in 15-tile packets (gpt-oss-120b: ~1% faster per
            # decode step than 18 / 9, in BFP4 and BFP8)
            kw.setdefault("kb", 30)
            kw.setdefault("pkt", 15)
        super().__init__(
            mesh_device,
            hidden // TILE,
            hidden // TILE,
            inter // TILE,
            2 * inter // TILE,
            inter // TILE,
            g,
            num_sel,
            1,
            inter // TILE,
            experts_per_core or num_sel,
            **kw,
        )
        self.inter = inter
        self.limit, self.alpha = swiglu_limit, alpha

    def __call__(self, x, ids, scores, gate, up, bias, act):
        extra = [_f32_bits(v) for v in (self.limit, -self.limit, -3.0e38, self.alpha, 1.0 / self.alpha, 1.0)]
        return self._run(x, act, ids, scores, gate, up, bias, 2048, self.inter * 2, extra)


class NdExpertDown(_NdOp):
    """(act, ids, scores, down, bias, out) -> out. act: [k, I] BF16 row-major (score-weighted SwiGLU outputs); down:
    the ND-sharded [1, E, I, H] tensor; bias: nd_bias_tensor of the [E, H] down bias (TP > 1: real on the first device,
    zero on the others); out: the flat BF16 partial sum."""

    def __init__(self, mesh_device, hidden, inter, num_sel, **kw):
        g = 1
        assert inter % TILE == 0, f"intermediate shard {inter}: whole tiles"
        super().__init__(
            mesh_device,
            inter // TILE,
            inter // TILE,
            hidden // TILE,
            hidden // TILE,
            hidden // TILE,
            g,
            num_sel,
            0,
            0,
            num_sel,
            **kw,
        )
        self.inter = inter

    def __call__(self, act, ids, scores, down, bias, out, send=None):
        return self._run(act, out, ids, scores, down, down, bias, self.inter * 2, 2048, [0] * 6, send=send)
