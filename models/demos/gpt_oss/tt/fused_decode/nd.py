# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-1 routed-expert decode ops at TP=1 (one P150) streaming the experts' own ND-sharded weights.

One chip has no DRAM for a decode-only copy of the BFP8 experts (20.3 GB), so at TP=1 the routed experts keep ONE
weight copy that prefill (ttnn.sparse_matmul, through its TensorAccessor) and these ops share: gate_proj / up_proj
[1, E, H, I] and down_proj [1, E, I, H], BFP8, ND-sharded in [1, 1, K, 32] shards (one shard per (expert, output tile
column), ROUND_ROBIN_1D over the DRAM banks: shard (e, c) of an N-tile-wide tensor is in bank (e * N + c) % banks).
The K rows of one (expert, column) are then one contiguous run of one bank, which kernels/nd_reader.cpp streams in
large packets into custom_mm in1 blocks (ct_dim = 1 column) for kernels/nd_compute.cpp. The shard placement depends
on the routed expert, so work items are not pinned next to their banks.

Bias: as in stream.py, an extra K tile per column (NBIAS BFP4-rounded residual terms, exact in BFP8) times an
activation tile of ones (gate|up) or of the routing score (down); the bias tiles live in a small decode-only [1, E, 32,
N] BFP8 tensor sharded like the weight (gate|up: the packed [gate | up] bias, N = 2 I). The bias block is read twice
(the activation's second extra tile is zero), keeping the per-call K even.

Ops (one token):
  NdExpertGateUp: (x, ids, scores, gate, up, bias, act) -> act [k, I] = w_e * SwiGLU(gate_e, up_e)
  NdExpertDown:   (act, ids, scores, down, bias, out) -> the flat partial sum (boundary.py)
"""

import struct
from pathlib import Path

import ttnn

from .config import ND_DEPTH, ND_FULL_SYNC, ND_K_BLOCK, ND_PACKET_TILES, ND_SPLIT_ACC, ND_STREAMS
from .stream import NBIAS, TILE_BYTES, bias_tiles

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = ttnn.TILE_SIZE
TINY_BYTES = 2 * TILE
ND_SHARD_WIDTH = TILE  # one output tile column per shard


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def nd_expert_memory_config(mesh_device, k, width=ND_SHARD_WIDTH):
    """DRAM memory config of a [1, E, k, N] expert weight in [1, 1, k, width] shards dealt round-robin over the banks."""
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


def nd_bias_tensor(mesh_device, bias, num_experts=None, cache_file_name=None):
    """[E, N] bias (torch, or None to load from the cache) -> [1, E, 32, N] BFP8 tensor, sharded like the weights,
    whose tile (e, n) holds the NBIAS residual terms of column n's bias in its first rows."""
    if bias is not None:
        E, N = bias.shape
        tiles = bias_tiles(bias.float().reshape(E, N // TILE, TILE))  # [E, n, 32, 32]
        bias = tiles.permute(0, 2, 1, 3).reshape(1, E, TILE, N)
    return ttnn.as_tensor(
        bias,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        cache_file_name=cache_file_name,
        memory_config=nd_expert_memory_config(mesh_device, TILE),
    )


def _cores(mesh_device, n):
    grid = mesh_device.compute_with_storage_grid_size()
    assert n <= grid.x * grid.y, f"{n} cores > grid {grid}"
    return [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(n)]


class _NdOp:
    """One work item per core: g output columns (one shard each) of experts_per_core routed experts (gate|up splits
    the k routed experts over cores; down keeps all k on one core to sum them).
    kt: K rows streamed; nt: weight columns; nt_b: bias columns (2 nt for the packed gate|up bias); up_off: column
    offset of the up bias; sets: column sets per expert (2 = gate and up)."""

    def __init__(self, mesh_device, kt, nt, nt_b, n_out, g, num_sel, mode, up_off, experts_per_core):
        self.mesh_device = mesh_device
        self.kt, self.nt, self.nt_b, self.n_out, self.g = kt, nt, nt_b, n_out, g
        self.num_sel, self.mode, self.up_off = num_sel, mode, up_off
        self.kb, self.depth, self.pkt, self.streams = ND_K_BLOCK, ND_DEPTH, ND_PACKET_TILES, ND_STREAMS
        assert kt % self.kb == 0, f"K block {self.kb} does not divide {kt}"
        n_groups = -(-n_out // g)
        self.groups = [i * g for i in range(n_groups)]
        assert num_sel % experts_per_core == 0
        self.epc = experts_per_core
        work = [(col0, e0) for col0 in self.groups for e0 in range(0, num_sel, experts_per_core)]
        cores = _cores(mesh_device, len(work))
        self.work = list(zip(cores, work))
        # One rectangle per row of cores: dispatch writes binaries and arguments per core range, and 90 single-core
        # ranges launch measurably slower under trace.
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(cores[0], cores[0])])
        for c in cores[1:]:
            self.cores = self.cores.merge(ttnn.CoreRangeSet([ttnn.CoreRange(c, c)]))
        self.tile_bytes = TILE_BYTES[ttnn.bfloat8_b]
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
        return self.num_sel * self.sets * cols * (self.kt + 2) * self.tile_bytes

    def _run(self, inp, out, ids, scores, w, u, b, in_page, out_page, compute_extra):
        cb_in0, cb_w, cb_idx, cb_scr, cb_out, cb_w2, cb_idx2 = 0, 1, 2, 3, 16, 5, 6
        seg = self.kt + 2
        in0_tiles = seg * (1 if self.mode == 1 else self.num_sel)
        slot_tiles = self.kb * self.g
        out_tiles = self.g * (self.epc if self.mode == 1 else 1)
        cbs = [
            self._cb(cb_in0, ttnn.bfloat16, TINY_BYTES, in0_tiles, tiny=True),
            self._cb(cb_w, ttnn.bfloat8_b, self.tile_bytes, self.depth * slot_tiles),
            self._cb(cb_idx, ttnn.uint16, 64, 1),
            self._cb(cb_scr, ttnn.bfloat16, 64, 1),
            self._cb(cb_out, ttnn.bfloat16, TINY_BYTES, out_tiles, tiny=True),
        ]
        if self.streams == 2:
            cbs += [
                self._cb(cb_w2, ttnn.bfloat8_b, self.tile_bytes, self.depth * slot_tiles),
                self._cb(cb_idx2, ttnn.uint16, 64, 1),
            ]
        reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        addrs = [w.buffer_address(), u.buffer_address(), b.buffer_address(), ids.buffer_address()]
        for i, (core, (col0, e0)) in enumerate(self.work):
            reader_rt[core.x][core.y] = addrs + [i & 3, 1, e0, col0]
            compute_rt[core.x][core.y] = [1]
            writer_rt[core.x][core.y] = [inp.buffer_address(), out.buffer_address(), scores.buffer_address(), 1, e0] + (
                addrs + [(i + 1) & 3, col0]
            )
        nsel = self.epc if self.mode == 1 else self.num_sel
        stream_ct = [self.tile_bytes, self.kt, self.nt, self.nt_b, self.kb, self.g, nsel, self.sets, self.up_off]
        stream_ct += [self.depth, self.pkt, self.streams]
        accessors = []
        for t in (w, u, b, ids):
            accessors += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        reader_ct = [cb_w, cb_idx] + stream_ct + accessors
        compute_ct = [cb_in0, cb_w, cb_out, self.kt, self.kb, self.g, nsel, self.mode] + compute_extra
        compute_ct += [int(ND_SPLIT_ACC), cb_w2, self.streams]
        writer_ct = [cb_in0, cb_out, cb_scr, self.kt, NBIAS, nsel, in_page, out_page, self.mode, self.g, self.n_out]
        for t in (inp, out, scores):
            writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        writer_ct += [cb_w2, cb_idx2] + stream_ct + accessors
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
        program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        io = [w, b, inp, ids, scores, out] if u is w else [w, u, b, inp, ids, scores, out]
        ttnn.generic_op(io, program)
        return out


class NdExpertGateUp(_NdOp):
    """(x, ids, scores, gate, up, bias, act) -> act. x: the flat BF16 normed hidden (boundary.py); ids / scores: the
    router's [1, 32] UINT16 / BF16 buffers; gate / up: the ND-sharded [1, E, H, I] BFP8 expert weights; bias:
    nd_bias_tensor of the packed [E, 2 I] gate|up bias; act: [k, I] BF16 row-major L1."""

    def __init__(self, mesh_device, hidden, inter, num_sel, swiglu_limit, alpha):
        nt = inter // TILE
        # One core per output tile column, running the k routed experts of its column (90 cores for gpt-oss-20b).
        super().__init__(mesh_device, hidden // TILE, nt, 2 * nt, nt, 1, num_sel, 1, nt, num_sel)
        self.inter = inter
        self.limit, self.alpha = swiglu_limit, alpha

    def __call__(self, x, ids, scores, gate, up, bias, act):
        extra = [_f32_bits(v) for v in (self.limit, -self.limit, -3.0e38, self.alpha, 1.0 / self.alpha, 1.0)]
        return self._run(x, act, ids, scores, gate, up, bias, 2048, self.inter * 2, extra)


class NdExpertDown(_NdOp):
    """(act, ids, scores, down, bias, out) -> out. act: [k, I] BF16 row-major (score-weighted SwiGLU outputs); down:
    the ND-sharded [1, E, I, H] BFP8 expert weight; bias: nd_bias_tensor of the [E, H] down bias; out: the flat BF16
    partial sum."""

    def __init__(self, mesh_device, hidden, inter, num_sel):
        nt = hidden // TILE
        super().__init__(mesh_device, inter // TILE, nt, nt, nt, 1, num_sel, 0, 0, num_sel)
        self.inter = inter

    def __call__(self, act, ids, scores, down, bias, out):
        return self._run(act, out, ids, scores, down, down, bias, self.inter * 2, 2048, [0] * 6)
