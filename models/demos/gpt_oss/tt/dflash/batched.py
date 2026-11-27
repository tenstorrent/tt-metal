# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Greedy DFlash for 32 users at once (batch 32) on the batched gpt-oss-20b decode model (batched_decode.py, dense MoE).

A round, for every user u: the drafter proposes 7 tokens after the user's last token; the target checks the 8-row
block [last token, 7 drafts] of every user in one 256-row verify pass and keeps the drafts up to the first mismatch plus
its own next token. Every pass uses the same row order: row 32 i + u is row i of user u's block, at position
start_u + i, so a 32-row step i holds one row of each user (the batched decode's layout: one K/V cache write per step).

Passes (device tensors throughout; one trace per round, the acceptance on device):
  draft: the noise block [last token, mask x 7] through the drafter's 8 layers (256-row matmuls; per-head Q / K norms
      and RoPE for all 256 rows in one program; block attention: each user's 8 rows attend its whole block and cached
      history, not causal), its LM head on rows 32 .. 255, a per-device argmax;
  verify: the block [last token, 7 drafts] through the target: QKV over all 256 rows, Q / K RoPE and the BFP8 K / V
      cache writes in two programs, block attention (causal within the block; windowed on the sliding-window layers),
      o_proj, the residual add + RMSNorm (+ router logits) in one program per add, the grouped MoE on the routed
      (row, expert) pairs only (grouped_moe.py), the target's LM head, a per-device argmax; the residuals of target
      layers 1, 6, 11, 16, 21 kept for the drafter;
  context: those features of all 256 rows into the drafter's K/V cache at positions start_u + i (rows past the
      accepted ones are rewritten by the next round's block before they are read).
"""

import math
import os
import struct
import time

import torch

import ttnn
from models.tt_transformers.tt.load_checkpoints import reverse_permute, reverse_permute_1d

from .draft import draft_snapshot, load_draft_config

TILE = ttnn.TILE_SIZE
U32 = 32  # rows per attention step: one row per user


def _row_grid(rows):
    """(core rows, 32-row tiles per core) of a pass's row-tiled layouts: one row tile per core row up to 8 row tiles
    (256 rows), then 8 core rows of rows / 256 row tiles each (512 / 1024-row passes)."""
    rt = rows // TILE
    gy = min(rt, 8)
    assert rt % gy == 0, rows
    return gy, rt // gy


def _residual_mc(rows, hidden):
    """A pass's residual: block sharded on 10 x gy cores, [mpc 32 rows, hidden / 10] each (_row_grid)."""
    gy, mpc = _row_grid(rows)
    return ttnn.create_sharded_memory_config(
        (mpc * TILE, hidden // 10), ttnn.CoreGrid(x=10, y=gy), ttnn.ShardStrategy.BLOCK,
        ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
    )  # fmt: skip


def _replicate(mesh, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=True):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh if device else None, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _copy(mesh, dst, t, dtype):
    """Write host tensor t into the persistent device tensor dst (a trace input)."""
    ttnn.copy_host_to_device_tensor(_replicate(mesh, t, dtype, device=False), dst)


def hold_prefill_l1(model):
    """The batched runners' resident L1 (persistent buffers, captured traces' outputs) stays allocated across prefills:
    the model's prefill ops fit their circular buffers below the lowest of it (Model.set_prefill_l1_end, as
    dflash/runner.py does for batch 1). Called after each allocation of resident L1 (it only moves down)."""
    if hasattr(model, "set_prefill_l1_end"):
        model.set_prefill_l1_end(model.mesh_device.lowest_occupied_compute_l1_address() or 0)


class _Parts(list):
    """An all-reduce output of a 512 / 1024-row pass: one width-sharded tensor per 256 rows (_TPAllReduce.PART)."""

    def deallocate(self, force=True):
        for t in self:
            t.deallocate(force)


class _TPAllReduce:
    """TP all-reduce of [rows, hidden] partial sums over the mesh row with all_reduce_async: persistent scratch and a
    global semaphore per slot, two slots used alternately (each call is an all-rank synchronization, so no device reuses
    a slot before every device is done with it). ttnn.all_reduce, run back to back with other ops, now and then left a
    wrong tile on one device (verify outputs differed between identical runs); this one is repeatable. Inputs go in as
    BF8 (the batched decode's all-reduce precision) on 90 cores, so the [rows, 4 x hidden] scratch fits L1."""

    CORES, LINKS = 90, 2
    PART = (
        256  # rows per all_reduce_async call: its link workers' CB holds 45 shards' rows (391 KB at 256 rows; a 512 /
    )
    # 1024-row pass runs one call per 256 rows, each part straight from the interleaved input's rows)
    _instances = {}

    def __init__(self, mesh, rows, hidden):
        from ..fused_decode.config import width_sharded_memory_config

        D = mesh.shape[1]
        self.mesh, self.hidden = mesh, hidden
        self.parts = max(1, rows // self.PART)
        rows = rows // self.parts
        self.memory_config = width_sharded_memory_config(mesh, hidden, self.CORES, rows=rows)
        grid = self.memory_config.shard_spec.grid.bounding_box()
        self.grid = ttnn.CoreCoord(grid.end.x - grid.start.x + 1, grid.end.y - grid.start.y + 1)
        self.shard = list(self.memory_config.shard_spec.shape)
        scratch = width_sharded_memory_config(mesh, hidden * D, self.CORES, rows=rows)
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        self.slots = [
            (
                ttnn.empty([1, 1, rows, hidden * D], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=scratch),
                ttnn.create_global_semaphore(mesh, cores, 0),
            )
            for _ in range(2)
        ]  # fmt: skip
        self.calls = 0

    @classmethod
    def get(cls, x):
        mesh, rows, hidden = x.device(), x.shape[-2], x.shape[-1]
        key = (id(mesh), rows, hidden)
        if key not in cls._instances:
            cls._instances[key] = cls(mesh, rows, hidden)
        return cls._instances[key]

    @classmethod
    def reset(cls):
        """Start every pass on slot 0 (each pass makes an even number of calls: the slot order is the same in every
        captured trace and replay)."""
        for inst in cls._instances.values():
            inst.calls = 0

    def __call__(self, x):
        outs = _Parts()
        for i in range(self.parts):
            if self.parts == 1:
                xs = ttnn.to_memory_config(x, self.memory_config, dtype=ttnn.bfloat8_b)
            else:
                xs = ttnn.interleaved_to_sharded_partial(
                    x, self.grid, self.shard, self.parts, i, ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.ShardOrientation.ROW_MAJOR, output_dtype=ttnn.bfloat8_b,
                )  # fmt: skip
            scratch, semaphore = self.slots[self.calls % 2]
            self.calls += 1
            outs.append(
                ttnn.experimental.all_reduce_async(
                    xs,
                    scratch,
                    cluster_axis=1,
                    mesh_device=self.mesh,
                    multi_device_global_semaphore=semaphore,
                    memory_config=self.memory_config,
                    dtype=ttnn.bfloat16,
                    topology=ttnn.Topology.Ring,
                    num_links=self.LINKS,
                )
            )
            xs.deallocate(True)
        return outs[0] if self.parts == 1 else outs


class _ResNorm:
    """The residual add and the RMSNorm in one program (fused_decode/kernels/resnorm_*.cpp): the all-reduce output
    (its [rows, hidden] width shards, _TPAllReduce.memory_config) + the residual (block sharded on GX x rows / 32 cores,
    [32, hidden / GX] each) -> the new residual (same layout) and its RMSNorm (L1 interleaved, BF16). Replaces the
    reshard of the all-reduce output to the residual, the add, the sharded norm and the norm output's copy to L1
    interleaved. Each core sums its rows' squares over its 9 column tiles; the row's 10 partials are exchanged
    (every core of a row gets all 10)."""

    GX = 10

    def __init__(self, mesh, rows, hidden, res_mc, ar_mc):
        self.mesh, self.rows, self.hidden, self.res_mc = mesh, rows, hidden, res_mc
        self.nt = hidden // TILE
        self.nb = self.nt // self.GX
        grid = ar_mc.shard_spec.grid.bounding_box()
        gw = grid.end.x - grid.start.x + 1
        assert ar_mc.shard_spec.shape[1] == TILE and gw * (grid.end.y - grid.start.y + 1) == self.nt
        ar_cores = [ttnn.CoreCoord(grid.start.x + c % gw, grid.start.y + c // gw) for c in range(self.nt)]
        self.gy, self.mpc = _row_grid(rows)  # (mpc row tiles per core: the kernels loop over them, RN_MPC)
        res_cores = [ttnn.CoreCoord(x, y) for y in range(self.gy) for x in range(self.GX)]

        def nocs(cores):
            w = []
            for c in cores:
                p = mesh.worker_core_from_logical_core(c)
                w.append((p.x << 8) | p.y)
            w += [0] * (len(w) % 2)
            return [w[i] | (w[i + 1] << 16) for i in range(0, len(w), 2)]

        self.ar_noc, self.res_noc = nocs(ar_cores), nocs(res_cores)
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.GX - 1, self.gy - 1))])
        self.gammas = {}

    def gamma(self, norm):
        """norm's weights (an RMSNorm module, or its weights tensor [1, 1, hidden / 32, 32]) as [32, hidden] BF16 tiles
        (every row the weights; DRAM), cached per module / tensor."""
        if id(norm) not in self.gammas:
            wt = norm.tt_weight if hasattr(norm, "tt_weight") else norm
            w = ttnn.to_torch(ttnn.get_device_tensors(wt)[0]).float().reshape(-1)
            assert w.numel() == self.hidden
            self.gammas[id(norm)] = ttnn.from_torch(
                w[None, None, None, :].expand(1, 1, TILE, self.hidden).contiguous(), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )  # fmt: skip
        return self.gammas[id(norm)]

    @staticmethod
    def _ck(f32_res, cb_r):
        ck = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                          math_approx_mode=False)  # fmt: skip
        if f32_res:  # (the FP32 residual straight to DST: its add on the SFPU keeps every bit)
            modes = [ttnn.UnpackToDestMode.Default] * 64
            modes[cb_r] = ttnn.UnpackToDestMode.UnpackToDestFp32
            ck.unpack_to_dest_mode = modes
        return ck

    def router_bias(self, r):
        """The router's bias as a [32, 32] BF16 tile with every row the bias (DRAM), cached per router."""
        key = ("rb", id(r))
        if key not in self.gammas:
            b = ttnn.to_torch(ttnn.get_device_tensors(r.bias)[0]).float().reshape(-1)[:TILE]
            self.gammas[key] = ttnn.from_torch(
                b[None, None, None, :].expand(1, 1, TILE, TILE).contiguous(), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )  # fmt: skip
        return self.gammas[key]

    def row_tiles(self, t):
        """A [1, hidden] (or [1, 1, 1, hidden]) bias as [32, hidden] BF16 tiles, every row the bias (DRAM), cached."""
        key = ("bias", id(t))
        if key not in self.gammas:
            b = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().reshape(-1)[: self.hidden]
            self.gammas[key] = ttnn.from_torch(
                b[None, None, None, :].expand(1, 1, TILE, self.hidden).contiguous(), dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )  # fmt: skip
        return self.gammas[key]

    def __call__(self, y, res, norm, out_bf16=True, x8=False, router=None, eps=None, bias=None, out_dtype=None):
        """y: the all-reduce output (width shards); res: the residual. Returns [res + y, then the outputs asked for:
        rmsnorm(res + y) in BF16 (out_bf16), in BFP8 (x8), and router (the MoE router module): its logits
        rmsnorm(res + y) W + b ([rows, 32] BF16; each core's 9 K tiles, the row's 10 partials summed in FP32)].
        An FP32 residual (the drafter's) is added in FP32 on the SFPU; bias: a [1, hidden] tensor added with y (row_tiles);
        out_dtype: the normed output's dtype (BF16 default, FP32)."""
        from ..fused_decode.boundary import KERNEL_DIR

        g = self.gamma(norm)
        shape = [1, 1, self.rows, self.hidden]
        mk = lambda shp, dt: ttnn.empty(shp, dtype=dt, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                                         memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
        f32_res = res.dtype == ttnn.float32
        out_dtype = out_dtype or ttnn.bfloat16
        eps = norm.eps if eps is None else eps
        res2 = ttnn.empty(shape, dtype=res.dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=self.res_mc)
        out = mk(shape, out_dtype) if out_bf16 else None
        out8 = mk(shape, ttnn.bfloat8_b) if x8 else None
        logits = mk([1, 1, self.rows, TILE], ttnn.bfloat16) if router is not None else None
        (cb_y, cb_r, cb_g, cb_sc, cb_all, cb_r2, cb_sq, cb_part, cb_rs, cb_t, cb_n, cb_n8, cb_nm, cb_w, cb_lp, cb_lall,
         cb_lb, cb_lo, cb_b) = range(19)  # fmt: skip
        f32 = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]
        cores = self.cores
        defines = ([("RN_OUT_BF16", "1")] if out_bf16 else []) + ([("RN_X8", "1")] if x8 else [])
        defines += ([("RN_F32", "1")] if f32_res else []) + ([("RN_BIAS", "1")] if bias is not None else [])
        if self.mpc > 1:
            defines.append(("RN_MPC", str(self.mpc)))
        reader_ct = [cb_y, cb_r, cb_g, cb_sc, cb_all, self.nb, self.GX, cb_w, cb_lb, cb_lall, cb_b]
        reader_ct += ttnn.TensorAccessorArgs(g).get_compile_time_args()
        ys = list(y) if isinstance(y, _Parts) else [y]
        reader_args = [ys[0].buffer_address(), g.buffer_address()]
        io = ys + [res, g, res2]
        if router is not None:
            defines.append(("RN_ROUTER", "1"))
            lb = self.router_bias(router)
            reader_ct += ttnn.TensorAccessorArgs(router.weight).get_compile_time_args()
            reader_ct += ttnn.TensorAccessorArgs(lb).get_compile_time_args()
            reader_args += [router.weight.buffer_address(), lb.buffer_address()]
            io += [router.weight, lb]
        if bias is not None:
            bt = self.row_tiles(bias)
            reader_ct += ttnn.TensorAccessorArgs(bt).get_compile_time_args()
            reader_args.append(bt.buffer_address())
            io.append(bt)
        n_tile = 4096 if out_dtype == ttnn.float32 else 2048
        writer_ct = [cb_part, cb_all, cb_n, self.nb, self.GX, self.nt, cb_n8, cb_lp, cb_lall, cb_lo, n_tile]
        for t in (out, out8, logits):
            if t is not None:
                writer_ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
                io.append(t)
        writer_args = [t.buffer_address() if t is not None else 0 for t in (out, out8, logits)] + self.res_noc
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "resnorm_reader.cpp"), core_ranges=cores, compile_time_args=reader_ct,
                runtime_args=[], common_runtime_args=reader_args + self.ar_noc + [t.buffer_address() for t in ys[1:]],
                defines=defines + ([("RN_YPARTS", str(len(ys))), ("RN_YPART_ARG", str(len(reader_args) + len(self.ar_noc)))]
                                   if len(ys) > 1 else []),
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "resnorm_writer.cpp"), core_ranges=cores, compile_time_args=writer_ct,
                runtime_args=[], common_runtime_args=writer_args, defines=defines,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "resnorm_compute.cpp"), core_ranges=cores,
                compile_time_args=[cb_y, cb_r, cb_g, cb_sc, cb_all, cb_r2, cb_sq, cb_part, cb_rs, cb_t, cb_n, self.nb,
                                   self.GX, f32(1.0 / self.hidden), f32(eps), cb_n8, cb_nm, cb_w, cb_lp, cb_lall,
                                   cb_lb, cb_lo, cb_b],
                runtime_args=[], defines=defines, config=self._ck(f32_res, cb_r),
            ),
        ]  # fmt: skip

        def cb(i, dtype, page, pages):
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores,
                                     format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype,
                                                                                 page_size=page)])  # fmt: skip

        bf, fp, b8 = ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b
        nb = self.nb
        cbs = [
            cb(cb_y, bf, 2048, nb),
            ttnn.cb_descriptor_from_sharded_tensor(cb_r, res),
            cb(cb_g, bf, 2048, nb),
            cb(cb_sc, bf, 2048, 1),
            cb(cb_all, fp, 4096, self.GX * self.mpc),  # (a slot per (row tile, core of the row))
            ttnn.cb_descriptor_from_sharded_tensor(cb_r2, res2),
            # (x^2 and x rs in one allocation: x^2 is consumed before x rs is written)
            ttnn.CBDescriptor(total_size=4096 * nb, core_ranges=cores, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=cb_sq, data_format=fp, page_size=4096),
                ttnn.CBFormatDescriptor(buffer_index=cb_t, data_format=fp, page_size=4096)]),
            cb(cb_part, fp, 4096, 1),
            cb(cb_rs, fp, 4096, 1),
        ]  # fmt: skip
        if out_bf16:
            cbs.append(cb(cb_n, out_dtype, n_tile, 2))
        if bias is not None:
            cbs.append(cb(cb_b, bf, 2048, nb))
        if x8:
            cbs.append(cb(cb_n8, b8, 1088, 2))
        if router is not None:
            cbs += [cb(cb_nm, bf, 2048, nb), cb(cb_w, bf, 2048, nb), cb(cb_lp, fp, 4096, 1),
                    cb(cb_lall, fp, 4096, self.GX * self.mpc), cb(cb_lb, bf, 2048, 1), cb(cb_lo, bf, 2048, 1)]  # fmt: skip
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, cores, 0) for i in range(2)]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return [res2] + [t for t in (out, out8, logits) if t is not None]


def _all_reduce(x):
    """TP all-reduce over the mesh row of a [1, 1, rows, hidden] partial (any layout): the BF16 sum in the all-reduce's
    width shards (_TPAllReduce.memory_config; _ResNorm reads them)."""
    return _TPAllReduce.get(x)(x)


def _mm_config_256(k, n, rows=256):
    """Program config for the drafter's and the verify's [256, k] x [k, n] projections (None: the auto config).
    Measured on one device: the auto config spreads these narrow outputs over few cores with all 8 row tiles each;
    2D over 10 x 8 cores (one row tile, n / 10 columns each) is 1.4-2.5x faster (drafter QKV 74 -> 37 us, gate / up
    86 -> 49, down 73 -> 51, o_proj 45 -> 33; verify QKV BF8 68 -> 27); the drafter's fc (k = 14400) is fastest 1D
    over 90 cores (487 -> 293 us). rows (32 .. 256): one row tile per core row (rows / 32 core rows)."""
    kt, nt, mt = k // TILE, n // TILE, rows // TILE
    gy = min(mt, 8)  # (more than 8 row tiles, e.g. a 512-row pass: several row tiles per core row)
    pm = -(-mt // gy)
    if k == 14400 and nt == 90 and rows == 256:
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 9), in0_block_w=10, out_subblock_h=1, out_subblock_w=1,
            out_block_h=8, out_block_w=1, per_core_M=8, per_core_N=1, fuse_batch=True, fused_activation=None,
            mcast_in0=True,
        )  # fmt: skip
    gx = 10
    if nt % 10:  # (the context pass's K / V projection, 12 tiles: 6 x 8 cores)
        gx = max(d for d in range(1, 12) if nt % d == 0)
        if gx < 4:
            return None
    pn = nt // gx
    bw = {90: 10, 60: 12, 32: 8}.get(kt, max(d for d in range(1, 17) if kt % d == 0))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=bw, out_subblock_h=1,
        out_subblock_w=max(d for d in range(1, 5) if pn % d == 0), out_block_h=pm, out_block_w=pn, per_core_M=pm,
        per_core_N=pn, transpose_mcast=False, fused_activation=None,
    )  # fmt: skip


_SUBBLOCKS = [
    (4, 2),
    (2, 4),
    (8, 1),
    (1, 8),
    (7, 1),
    (1, 7),
    (3, 2),
    (2, 3),
    (6, 1),
    (1, 6),
    (5, 1),
    (1, 5),
    (2, 2),
    (4, 1),
    (1, 4),
    (3, 1),
    (1, 3),
    (2, 1),
    (1, 2),
    (1, 1),
]


def _subblock(m, n, fp32):
    return next(((h, w) for h, w in _SUBBLOCKS if (not fp32 or h * w <= 4) and m % h == 0 and n % w == 0), (1, 1))


def _matmul_config(x, w, out_dram, fp32=False, grid=(11, 10)):
    """The program config ttnn.matmul picks for interleaved [M, K] x [K, N] when L1 is free (matmul_program_config.cpp
    create_simple_matmul_program_config with every CB size fitting). ttnn sizes its blocks to the L1 free at the call, so
    the auto config, and with it the K blocking and the rounding, changed with whatever else was in L1 (DFlash lanes in
    lockstep); a fixed config keeps every call's result independent of the L1 state."""
    M, K, N = x.shape[-2], x.shape[-1], w.shape[-1]
    Mt, Kt, Nt = M // TILE, K // TILE, N // TILE
    in0_bw = 2
    pcM, pcN = min(16, Mt), min(16, Nt)
    nby, nbx = -(-Mt // pcM), -(-Nt // pcN)
    if (nby > 1 or nbx > 1) and Kt % in0_bw:
        in0_bw = 1
    dram = lambda t: t.memory_config().buffer_type == ttnn.BufferType.DRAM
    all_dram = dram(x) and dram(w) and out_dram
    narrow = max(M, N) // min(M, N) > 8 or (all_dram and min(M, N) <= TILE)
    wide, tall = narrow and N > M, narrow and N <= M
    gx, gy = grid
    if wide and Mt > 8 and not all_dram:
        # (a 512 / 1024-row LM head: every core all the rows, output blocks of up to 8 row tiles: the output CB holds a block,
        # not all per_core_M rows)
        pm, pn = Mt, -(-(-(-N // (gx * gy))) // TILE)
        ob = max(d for d in range(1, 9) if Mt % d == 0)
        sh, sw = _subblock(ob, pn, fp32)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=2 if Kt % 2 == 0 else 1,
            out_subblock_h=sh, out_subblock_w=sw, out_block_h=ob, out_block_w=pn, per_core_M=pm, per_core_N=pn,
            fuse_batch=False, fused_activation=None, mcast_in0=True,
        )  # fmt: skip
    assert all_dram or (nbx * nby <= gx * gy and Kt % in0_bw == 0), "config outside the mirrored cases"
    cr = (nbx, nby) if not (nby == 1 and nbx == 1) and nby <= gy and nbx <= gx else (0, 0)
    if (cr[1] == 1 and not all_dram) or wide:
        pm, pn = Mt, -(-(-(-N // (gx * gy))) // TILE)
        sh, sw = _subblock(pm, pn, fp32)
        return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=2 if Kt % 2 == 0 else 1,
            out_subblock_h=sh, out_subblock_w=sw, out_block_h=pm, out_block_w=pn, per_core_M=pm, per_core_N=pn,
            fuse_batch=False, fused_activation=None, mcast_in0=True,
        )  # fmt: skip
    assert not ((cr[0] == 1 and not all_dram) or tall), "config outside the mirrored cases"
    if all_dram:
        in0_bw = Kt // gx if Kt % gx == 0 else 1
        pcM, pcN = -(-Mt // gy), -(-Nt // gx)
        sh, sw = _subblock(pcM, pcN, fp32)
    else:
        sh, sw = _subblock(pcM, pcN, fp32)
        if sh != 1 and sw != pcN:
            sh, sw = next(((h, w) for h, w in _SUBBLOCKS if not (fp32 and h * w > 4) and (h == 1 or w == pcN)
                           and pcM % h == 0 and pcN % w == 0), (1, 1))  # fmt: skip
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy), in0_block_w=in0_bw, out_subblock_h=sh,
        out_subblock_w=sw, out_block_h=pcM, out_block_w=pcN, per_core_M=pcM, per_core_N=pcN, transpose_mcast=False,
        fused_activation=None,
    )  # fmt: skip


def _even_lm_head(model):
    """The target's LM head [hidden, vocab] BF8, sharded evenly over the mesh columns (vocab / D columns per device). The
    model's copy pads the vocabulary to 65536 columns per device before sharding (ttnn.topk's power-of-two path), so
    every device multiplies 65536 columns (the last one holding 4480 real ones); the batch-32 LM heads (compute-bound at
    256 rows) and _Argmax need no padding. Built once from the model's copy (the BF8 values exactly: 50272 columns per
    device keep the 16-column exponent groups), cached under TT_CACHE_PATH; shared by the drafter and the verify."""
    if getattr(model, "_b32_lm_head_even", None) is not None:
        return model._b32_lm_head_even
    import glob

    mesh, D, V = model.mesh_device, model.mesh_device.shape[1], model.vocab_size
    per = V // D
    assert per * D == V and per % TILE == 0
    cache = os.environ.get("TT_CACHE_PATH")
    name = os.path.join(cache, "b32_lm_head_even.weight") if cache else None
    full = None
    if name is not None and not glob.glob(name + "_dtype_*") and not os.access(cache, os.W_OK):
        name = None  # (a read-only cache, as CI mounts the shared one: built in memory, not cached)
    if name is None or not glob.glob(name + "_dtype_*"):
        w = model.lm_head_weight
        full = ttnn.to_torch(w, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1))
        full = full.reshape(-1, full.shape[-1])[:, :V].reshape(1, 1, -1, V)
    model._b32_lm_head_even = ttnn.as_tensor(
        full, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1), cache_file_name=name,
    )  # fmt: skip
    return model._b32_lm_head_even


class _Argmax:
    """Row argmax over a vocabulary split over the mesh columns: each device's best (value, index) on device; the host
    combines the devices (read after the pass)."""

    NCH = 13  # vocabulary chunks per 32-row tile row (8 x 13 = 104 cores)

    def __init__(self, per_dev, vocab):
        self.per_dev, self.vocab = per_dev, vocab

    def device(self, logits, out_rows=None):
        """(values, indices) [1, 1, rows, 1] FLOAT32: each row's maximum over this device's columns and its first
        column index, by the per-row max / index kernels (fused_decode/kernels/b32_argmax_*: one pass over the logits,
        BFP8 or BF16 read as stored; ttnn.topk k = 1 took ~1.0 ms at 256 rows). out_rows: the outputs have out_rows
        rows, the logits being their last rows (the first ones left unwritten)."""
        mesh = logits.device()
        rows, width = logits.shape[-2], logits.shape[-1]
        RT, NT, NCH = rows // TILE, width // TILE, self.NCH
        out_rows = out_rows or rows
        r0 = out_rows // TILE - RT  # the output row tile of logits row tile 0
        f32 = ttnn.float32
        vals = ttnn.empty([1, 1, out_rows, 1], dtype=f32, layout=ttnn.TILE_LAYOUT, device=mesh,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        idx = ttnn.empty([1, 1, out_rows, 1], dtype=f32, layout=ttnn.TILE_LAYOUT, device=mesh,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        # one program per (up to) 8 row tiles x 13 chunks (104 cores; a 512 / 1024-row pass: 2 / 4 programs)
        for t0 in range(0, RT, 8):
            self._device_part(logits, vals, idx, t0, min(8, RT - t0), r0)
        return vals, idx

    def _device_part(self, logits, vals, idx, t0, nrt, r0):
        """Row tiles t0 .. t0 + nrt - 1 of logits into rows (t0 + r0) 32 .. of vals / idx."""
        from ..fused_decode.boundary import KERNEL_DIR

        mesh = logits.device()
        NT, NCH = logits.shape[-1] // TILE, self.NCH
        grid = mesh.compute_with_storage_grid_size()
        core_list = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][: nrt * NCH]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_list])
        f32 = ttnn.float32
        bounds = [NT * c // NCH for c in range(NCH + 1)]
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        noc = [mesh.worker_core_from_logical_core(c) for c in core_list]
        for i, c in enumerate(core_list):
            ch, rt = i // nrt, i % nrt
            col0, ncol = bounds[ch], bounds[ch + 1] - bounds[ch]
            rd[c.x][c.y] = [t0 + rt, col0, ncol]
            wr[c.x][c.y] = [t0 + rt + r0, ch, (noc[rt].x << 8) | noc[rt].y]
            cp[c.x][c.y] = [col0, ncol]
        B, SEM = 8, 0
        cb_in, cb_part, cb_recv = 0, 1, 2
        tile = 2048 if logits.dtype == ttnn.bfloat16 else 1088
        rct = [cb_in, NT, B] + ttnn.TensorAccessorArgs(logits).get_compile_time_args()
        wct = [cb_part, cb_recv, NCH, SEM]
        for t in (vals, idx):
            wct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "b32_argmax_reader.cpp"), core_ranges=cores, compile_time_args=rct,
                runtime_args=rd, common_runtime_args=[logits.buffer_address()],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "b32_argmax_writer.cpp"), core_ranges=cores, compile_time_args=wct,
                runtime_args=wr, common_runtime_args=[vals.buffer_address(), idx.buffer_address()],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "b32_argmax_compute.cpp"), core_ranges=cores,
                compile_time_args=[cb_in, cb_part], runtime_args=cp,
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                                    math_approx_mode=False),
            ),
        ]  # fmt: skip

        def cb(i, dtype, page, pages):
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores,
                                     format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype,
                                                                                 page_size=page)])  # fmt: skip

        recv = NCH * 256 + 256 + 2 * 4096
        cbs = [cb(cb_in, logits.dtype, tile, 2 * B), cb(cb_part, f32, 4096, 4), cb(cb_recv, f32, recv, 1)]
        sems = [ttnn.SemaphoreDescriptor(SEM, ttnn.CoreType.WORKER, cores, 0)]
        ttnn.generic_op([logits, vals, idx], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))

    def combine(self, out, mesh):
        """The devices' (value, index) combined on device: [1, 1, rows, 1] FLOAT32 global token ids, the same on every
        device (the host combine's choice: padded vocabulary columns excluded, ties to the first device)."""
        D = mesh.get_num_devices()
        if not hasattr(self, "offsets"):
            rep = ttnn.ReplicateTensorToMesh(mesh)
            put = lambda t: ttnn.from_torch(
                t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep
            )
            self.offsets = put((torch.arange(D) * self.per_dev).float().reshape(1, 1, 1, D))
        vals, idxs = out
        g = []
        for t in (vals, idxs):
            f = ttnn.typecast(t, ttnn.float32) if t.dtype != ttnn.float32 else ttnn.clone(t)
            g.append(
                ttnn.all_gather(
                    f, dim=3, cluster_axis=1, num_links=1, topology=ttnn.Topology.Linear,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )  # fmt: skip
            f.deallocate(True)
        v, gi = g
        gi = ttnn.add(gi, self.offsets, output_tensor=gi)
        pad = ttnn.ge(gi, float(self.vocab))
        v = ttnn.where(pad, -1e30, v, output_tensor=v)
        pad.deallocate(True)
        best = ttnn.max(v, dim=-1, keepdim=True)
        hit = ttnn.eq(v, best)
        v.deallocate(True)
        best.deallocate(True)
        cand = ttnn.where(hit, gi, 1e9)
        hit.deallocate(True)
        gi.deallocate(True)
        tok = ttnn.min(cand, dim=-1, keepdim=True)
        cand.deallocate(True)
        return tok

    @staticmethod
    def read(tok):
        """Host copy of combine()'s token ids (device 0's)."""
        return ttnn.to_torch(ttnn.get_device_tensors(tok)[0]).reshape(-1).long().tolist()

    def host(self, out):
        vals, idxs = out
        v = torch.stack([ttnn.to_torch(t).float().reshape(-1) for t in ttnn.get_device_tensors(vals)])
        i = torch.stack([ttnn.to_torch(t).long().reshape(-1) for t in ttnn.get_device_tensors(idxs)])
        gi = i + torch.arange(v.shape[0])[:, None] * self.per_dev
        v = torch.where(gi < self.vocab, v, torch.full_like(v, -1e30))
        best = v.argmax(0)  # the first device with the maximum
        return gi[best, torch.arange(v.shape[1])].tolist()


class BatchedKVWrite:
    """Every user's block rows (rotated K / V heads in the prefill layout [1, kv_heads, rows, 64], L1 interleaved) ->
    the users' paged BF16 (drafter) or BFP8_b (target) caches, one core per user (fused_decode/kernels/b32_kv_write.cpp: K on BRISC, V on NCRISC),
    one or several layers per program: replaces the 8 per-step head splits + paged_fused_update_cache calls."""

    def __init__(self, mesh, users, block, kv_heads, head_dim, block_size, head_split=False):
        """head_split: one core per (user, kv head) (the BFP8 path's row packing is RISC-bound: 35 us per layer on one
        core per user)."""
        self.mesh, self.users, self.block = mesh, users, block
        self.kv_heads, self.head_tiles, self.block_size = kv_heads, head_dim // TILE, block_size
        self.hpc = 1 if head_split else kv_heads  # kv heads per core
        grid = mesh.compute_with_storage_grid_size()
        n = users * kv_heads // self.hpc
        # more (user, head) units than cores (more than 32 users): consecutive units per core
        self.max_cores = min(grid.x * grid.y, 100)
        n = min(n, self.max_cores)
        self.core_list = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][:n]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.core_list])

    def _units(self, hpc):
        """Per core of core_list: (u0, u1, h_lo, h_hi): users [u0, u1) of kv heads [h_lo, h_hi)."""
        per = self.kv_heads // hpc
        n_units = self.users * per
        if n_units <= len(self.core_list):
            return [(i // per, i // per + 1, (i % per) * hpc, (i % per) * hpc + hpc) for i in range(n_units)]
        # every head per core, users split evenly
        n = len(self.core_list)
        return [(self.users * i // n, self.users * (i + 1) // n, 0, self.kv_heads) for i in range(n)]

    def __call__(self, ks, vs, caches, pos_ui, page_table, src_head_stride=0):
        """ks / vs: per layer [1, kv_heads, rows, hd]; caches: per layer (k_cache, v_cache). src_head_stride: ks / vs
        are one tensor each for all layers, layer l's heads from head l src_head_stride."""
        from ..fused_decode.boundary import KERNEL_DIR

        n = len(caches)
        k0, v0 = caches[0]
        if k0.dtype == ttnn.bfloat8_b and n == 1 and self.hpc == 1 and not src_head_stride:
            return self._call_pack(ks[0], vs[0], k0, v0, pos_ui, page_table)
        ct = [0, self.block, self.kv_heads, self.head_tiles, self.block_size, page_table.shape[-1] * 4, self.users]
        for t in (k0, v0, pos_ui, page_table, ks[0]):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        more = [a for l in range(1, n) for a in (caches[l][0].buffer_address(), caches[l][1].buffer_address(),
                                                  ks[l].buffer_address(), vs[l].buffer_address())]  # fmt: skip
        units = self._units(self.hpc)
        cores = self.core_list[: len(units)]
        for c, (u0, u1, h0, h1) in zip(cores, units):
            arg0 = u0 | (u1 << 10) | (h0 << 20) | (h1 << 26)
            rt[c.x][c.y] = [arg0, ks[0].buffer_address(), vs[0].buffer_address(), k0.buffer_address(),
                            v0.buffer_address(), pos_ui.buffer_address(), page_table.buffer_address()] + more  # fmt: skip
        defines = [("KV_LAYERS", str(n))] if n > 1 else []
        if src_head_stride:
            defines.append(("KV_SRC_HSTRIDE", str(src_head_stride)))
        bf8 = k0.dtype == ttnn.bfloat8_b  # the target's cache: rows packed to BFP8_b in the kernel
        if bf8:
            defines.append(("KV_BF8", "1"))
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "b32_kv_write.cpp"),
                core_ranges=core_set,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=(ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1)[sel],
                    noc=(ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)[sel],
                ),
                defines=[("KV_SEL", str(sel))] + defines,
            )
            for sel in (0, 1)
        ]
        tile_bytes = 1088 if bf8 else 2048
        stage = 2 * n * self.block * self.kv_heads * self.head_tiles * 64 if bf8 else 0
        scratch = 512 + n * 2 * 2 * self.kv_heads * self.head_tiles * tile_bytes + stage
        cb = ttnn.CBDescriptor(
            total_size=scratch,
            core_ranges=core_set,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=scratch)],
        )
        io = [pos_ui, page_table] + [t for l in range(n) for t in (ks[l], vs[l], caches[l][0], caches[l][1])]
        io = [t for i, t in enumerate(io) if all(t is not u for u in io[:i])]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[cb]))

    def joint(self, qk, v0, v1, cache, pos_ui, page_table, k_head0):
        """The pack path from nlp_create_qkv_heads(num_heads = q + kv heads, num_kv_heads = 1)'s split: K = heads
        k_head0 .. of qk (Q and K rotated by one RoPE op), V heads 0 / 1 = the one-head tensors v0 / v1."""
        assert self.kv_heads == 2 and self.hpc == 1
        self._call_pack(qk, v0, cache[0], cache[1], pos_ui, page_table, joint=(v1, k_head0))

    def _call_pack(self, k, v, k_cache, v_cache, pos_ui, page_table, joint=None):
        """The BFP8_b caches with the packing on the compute core (fused_decode/kernels/b32_kv_write_pack*.cpp: the
        RISC-V row packing took ~15 of b32_kv_write.cpp's ~20 us per layer). One layer, one core per (user, kv head)."""
        from ..fused_decode.boundary import KERNEL_DIR

        cb_scr, cb_old, cb_new, cb_mask, cb_out = 0, 1, 3, 5, 6  # (old / new / out: K at the index, V at index + 1)
        ct = [cb_scr, self.block, self.kv_heads, self.head_tiles, self.block_size, page_table.shape[-1] * 4,
              self.users, cb_old, cb_new, cb_mask, cb_out]  # fmt: skip
        for t in (k_cache, v_cache, pos_ui, page_table, k):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt, rtc = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        P = self.users * self.kv_heads  # (user, kv head) pairs p = u kv_heads + h, consecutive ranges per core
        nc = min(P, len(self.core_list))
        cores = self.core_list[:nc]
        for i, c in enumerate(cores):
            p0, p1 = P * i // nc, P * (i + 1) // nc
            rt[c.x][c.y] = [p0 | (p1 << 16), k.buffer_address(), v.buffer_address(), k_cache.buffer_address(),
                            v_cache.buffer_address(), pos_ui.buffer_address(), page_table.buffer_address()]  # fmt: skip
            if joint is not None:
                rt[c.x][c.y].append(joint[0].buffer_address())
            rtc[c.x][c.y] = [p1 - p0]
        core_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
        dm = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "b32_kv_write_pack.cpp"), core_ranges=core_set, compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=(ttnn.DataMovementProcessor.RISCV_0, ttnn.DataMovementProcessor.RISCV_1)[sel],
                    noc=(ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)[sel],
                ),
                defines=[("KV_SEL", str(sel))] + ([("KV_QK_JOINT", str(joint[1]))] if joint is not None else []),
            )
            for sel in (0, 1)
        ]  # fmt: skip
        comp = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "b32_kv_write_pack_compute.cpp"), core_ranges=core_set,
            compile_time_args=[cb_old, cb_new, cb_mask, cb_out, self.head_tiles], runtime_args=rtc,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                                math_approx_mode=False),
        )  # fmt: skip
        n = 2 * self.head_tiles  # tiles per kv at most (2 groups)

        def cb(i, dtype, page, pages):
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=core_set,
                                     format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype,
                                                                                 page_size=page)])  # fmt: skip

        bf8, bf16 = ttnn.bfloat8_b, ttnn.bfloat16
        cbs = [cb(cb_scr, bf16, 512, 1), cb(cb_old, bf8, 1088, n), cb(cb_old + 1, bf8, 1088, n), cb(cb_new, bf16, 2048, n),
               cb(cb_new + 1, bf16, 2048, n), cb(cb_mask, bf16, 2048, 2), cb(cb_out, bf8, 1088, n),
               cb(cb_out + 1, bf8, 1088, n)]  # fmt: skip
        io = [pos_ui, page_table, k, v, k_cache, v_cache] + ([joint[0]] if joint is not None else [])
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=dm + [comp], semaphores=[], cbs=cbs))


def _gu_joint(gT, uT, D, gx):
    """[K, D I] gate / up weights -> [K, 2 D I], per device d and core column j the gate columns of block j then the up
    ones (I / gx columns each)."""
    I = gT.shape[1] // D
    w = I // gx
    return torch.cat(
        [torch.cat([gT[:, d * I + j * w : d * I + (j + 1) * w], uT[:, d * I + j * w : d * I + (j + 1) * w]], dim=1)
         for d in range(D) for j in range(gx)], dim=1)  # fmt: skip


def _qkv_rope(mesh, trans, qkv, cos, sin, na, nb, rope_heads, headnorm=None):
    """A QKV projection [1, 1, rows, heads 64] -> its heads in three heads-layout tensors (heads 0 .. na - 1, the next
    nb, the rest), heads < rope_heads rotated (rotary_embedding_llama's prefill mode, the rotation tile trans), in one
    program (fused_decode/kernels/qkv_rope_*.cpp): one core per (32-row tile, 2 heads). headnorm = (gq, gk, nq, eps):
    the rotated heads RMS-normed per head first (heads < nq with the weight tiles gq [1, 1, 32, 64], the others gk), as
    ttnn.rms_norm on the heads layout."""
    from ..fused_decode.boundary import KERNEL_DIR

    rows, nt = qkv.shape[-2], qkv.shape[-1] // TILE
    heads, hd = nt // 2, 2 * TILE
    hpc = 2
    gx = heads // hpc
    gy, mpc = _row_grid(rows)  # (mpc row tiles per core: QR_MPC)
    assert gx * hpc == heads and gx <= 11
    mk = lambda n: ttnn.empty([1, n, rows, hd], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                              memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
    a, b, c = mk(na), mk(nb), mk(heads - na - nb)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    cb_x, cb_cos, cb_sin, cb_t, cb_out, cb_g, cb_sc, cb_sq, cb_rs, cb_tn, cb_nrm = range(11)
    rin = [qkv, cos, sin, trans] + ([headnorm[0], headnorm[1]] if headnorm else [])
    rct = [cb_x, cb_cos, cb_sin, cb_t, hpc, nt, cb_g, cb_sc, rope_heads]
    for t in rin:
        rct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    wct = [cb_out, hpc, na, rows // TILE, nb]
    for t in (a, b, c):
        wct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
    f32 = lambda v: struct.unpack("<I", struct.pack("<f", v))[0]
    defines = [("QR_HEADNORM", "1"), ("QR_NQ", str(headnorm[2]))] if headnorm else []
    if mpc > 1:
        defines.append(("QR_MPC", str(mpc)))
    dm = ttnn.DataMovementConfigDescriptor
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "qkv_rope_reader.cpp"), core_ranges=cores, compile_time_args=rct,
            runtime_args=[], common_runtime_args=[t.buffer_address() for t in rin], defines=defines,
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "qkv_rope_writer.cpp"), core_ranges=cores, compile_time_args=wct,
            runtime_args=[], common_runtime_args=[t.buffer_address() for t in (a, b, c)], defines=defines,
            config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "qkv_rope_compute.cpp"), core_ranges=cores,
            compile_time_args=[cb_x, cb_cos, cb_sin, cb_t, cb_out, hpc, rope_heads, cb_g, cb_sc, cb_sq, cb_rs, cb_tn,
                               cb_nrm, f32(1.0 / hd), f32(headnorm[3] if headnorm else 0.0)],
            runtime_args=[], defines=defines,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                                math_approx_mode=False),
        ),
    ]  # fmt: skip

    def cb(i, pages, dt=ttnn.bfloat16, page=2048):
        return ttnn.CBDescriptor(total_size=page * pages, core_ranges=cores,
                                 format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dt,
                                                                             page_size=page)])  # fmt: skip

    cbs = [cb(cb_x, 2 * hpc), cb(cb_cos, 2), cb(cb_sin, 2), cb(cb_t, 1), cb(cb_out, 2)]
    if headnorm:
        cbs += [cb(cb_g, 2 * hpc), cb(cb_sc, 1), cb(cb_sq, 2, ttnn.float32, 4096), cb(cb_rs, 1, ttnn.float32, 4096),
                cb(cb_tn, 2, ttnn.float32, 4096), cb(cb_nrm, 2)]  # fmt: skip
    ttnn.generic_op(rin + [a, b, c], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
    return a, b, c


class BatchedDraft:
    """The z-lab drafter (8 layers, TP like a dense target layer) for `users` sequences, each with its own paged BF16
    K/V cache; draft and context passes over all users' rows at once."""

    def __init__(self, mesh_device, target, max_positions, users, window=None, block_size=64, cache_users=None,
                 ring=None, block=None, share=None):  # fmt: skip
        """users: the rows' users (a pass: users x block rows); cache_users: the users with a K/V cache (lanes of `users`
        each: the page table rows of a lane's users are set before its passes, BatchedDFlashRunner); ring: each user's
        cache a ring of `ring` blocks (logical block b at physical block b mod ring; with the window the drafter reads
        only the last `window` positions, add_prompt_context fills only those). share: another BatchedDraft whose
        weights and K / V caches (with their prompt context) this one uses, with its own block and users per pass."""
        from safetensors.torch import load_file

        self.mesh_device = mesh_device
        self.target = target
        self.path = draft_snapshot()
        c = load_draft_config(self.path)
        D = mesh_device.shape[1]
        self.hidden, self.hd = c["hidden_size"], c["head_dim"]
        self.nq, self.nkv = c["num_attention_heads"] // D, c["num_key_value_heads"] // D
        self.layers = c["num_hidden_layers"]
        self.eps = c["rms_norm_eps"]
        # the draft block: the last token + block - 1 masks (the checkpoint's 8, or fewer: more users per pass)
        self.block = block or c["block_size"]
        self.mask_token_id = c["dflash_config"]["mask_token_id"]
        self.target_layer_ids = c["dflash_config"]["target_layer_ids"]
        self.ctx_in = len(self.target_layer_ids) * self.hidden
        self.window = window
        self.users = users
        self.rows = users * self.block
        if share is not None:
            for k in ("fc_full", "hidden_norm", "norm", "L", "kv_joint", "kv_joint_b", "knorm_joint", "block_size",
                      "num_blocks", "ring", "user_tables", "caches", "rope", "ck", "fc_ck", "act_mc", "mm_ck", "lm_head",
                      "logits_dt", "argmax"):  # fmt: skip
                setattr(self, k, getattr(share, k))
            self.page_table = _replicate(mesh_device, self.user_tables[:users].contiguous(), ttnn.int32)
            self._setup_passes(mesh_device, users, window)
            return
        w = load_file(str(self.path / "model.safetensors"))
        mesh = mesh_device
        rep = ttnn.ReplicateTensorToMesh(mesh)

        # BFP8 weights (with the HiFi2 matmuls, mm_ck): round 27.3 -> 26.6 ms at ISL 256, 28.1 -> 27.5 at 1024, the
        # acceptance within its run-to-run spread (4.92 / 4.98 -> 4.93 / 5.01 tokens per round)
        wdt = ttnn.bfloat8_b

        def put(t, mapper=rep, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=dtype,
                layout=layout,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        def norm_w(t):
            return put(t.float().reshape(1, 1, -1, TILE), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

        cols = ttnn.ShardTensorToMesh(mesh, dim=-1)
        rows = ttnn.ShardTensorToMesh(mesh, dim=-2)
        hd, nq, nkv = self.hd, self.nq, self.nkv

        def per_device(parts):
            """[q | k | v]-style row blocks: each device's slice of every part, side by side."""
            return torch.cat(
                [torch.cat([p[d * (p.shape[0] // D) : (d + 1) * (p.shape[0] // D)] for p in parts]) for d in range(D)]
            )

        # replicated: the features are replicated on every device; BFP8 (the context pass's 83 MB BF16 read: round
        # unchanged within 0.1 ms, acceptance in its spread)
        self.fc_full = put(w["fc.weight"].T, dtype=ttnn.bfloat8_b)
        self.hidden_norm = norm_w(w["hidden_norm.weight"])
        self.norm = norm_w(w["norm.weight"])
        self.L = []
        for i in range(self.layers):
            p, a, m = f"layers.{i}.", f"layers.{i}.self_attn.", f"layers.{i}.mlp."
            q = reverse_permute(w[a + "q_proj.weight"], D * nq, D * nq * hd, self.hidden)
            k = reverse_permute(w[a + "k_proj.weight"], D * nkv, D * nkv * hd, self.hidden)
            v = w[a + "v_proj.weight"]
            qb = reverse_permute(w[a + "q_proj.bias"][:, None], D * nq, D * nq * hd, 1)[:, 0]
            kb = reverse_permute(w[a + "k_proj.bias"][:, None], D * nkv, D * nkv * hd, 1)[:, 0]
            vb = w[a + "v_proj.bias"]
            g, u = w[m + "gate_proj.weight"], w[m + "up_proj.weight"]
            self.L.append(
                dict(
                    ln1=norm_w(w[p + "input_layernorm.weight"]),
                    ln2=norm_w(w[p + "post_attention_layernorm.weight"]),
                    qkv=put(per_device([q, k, v]).T, cols, dtype=wdt),
                    qkv_b=put(per_device([qb, kb, vb]).reshape(1, -1), cols),
                    k_norm=norm_w(reverse_permute_1d(w[a + "k_norm.weight"])),  # (the prompt context's K norm)
                    # (the per-head norm weights as [32, 64] tiles, every row the weights: _qkv_rope's headnorm)
                    q_norm_t=put(
                        reverse_permute_1d(w[a + "q_norm.weight"]).float()[None, :].expand(TILE, -1).contiguous()
                    ),
                    k_norm_t=put(
                        reverse_permute_1d(w[a + "k_norm.weight"]).float()[None, :].expand(TILE, -1).contiguous()
                    ),
                    o=put(w[a + "o_proj.weight"].T, rows, dtype=wdt),
                    o_b=put(w[a + "o_proj.bias"].reshape(1, -1)),
                    # gate and up as one matmul, its columns interleaved per core column of the 10 x 8 grid: core
                    # column j's 2 x 6 output tiles = gate tiles 6 j .. 6 j + 5, then the up ones of the same columns
                    gu=put(_gu_joint(g.T, u.T, D, 10), cols, dtype=wdt),
                    down=put(w[m + "down_proj.weight"].T, rows, dtype=wdt),
                )
            )
        # the context pass's joint K / V projection for every layer: [q stubs (nkv heads per layer) | k | v], each part
        # layer-major (nlp_create_qkv_heads with L nkv heads of each), one matmul + one heads op + one norm + one
        # RoPE for the 8 layers instead of 8 of each; the per-layer k_norm weights as a per-head multiplier
        parts = [[], [], []]
        bparts = [[], [], []]
        knw = []
        for i in range(self.layers):
            a = f"layers.{i}.self_attn."
            q = reverse_permute(w[a + "q_proj.weight"], D * nq, D * nq * hd, self.hidden)
            k = reverse_permute(w[a + "k_proj.weight"], D * nkv, D * nkv * hd, self.hidden)
            v = w[a + "v_proj.weight"]
            qb = reverse_permute(w[a + "q_proj.bias"][:, None], D * nq, D * nq * hd, 1)[:, 0]
            kb = reverse_permute(w[a + "k_proj.bias"][:, None], D * nkv, D * nkv * hd, 1)[:, 0]
            vb = w[a + "v_proj.bias"]
            qh, kh = nq * hd, nkv * hd
            for j, (t, tb) in enumerate(((q, qb), (k, kb), (v, vb))):
                rows_per = qh if j == 0 else kh
                parts[j].append([t[d * rows_per : d * rows_per + kh] for d in range(D)])
                bparts[j].append([tb[d * rows_per : d * rows_per + kh] for d in range(D)])
            knw.append(reverse_permute_1d(w[a + "k_norm.weight"]).float())
        joint = torch.cat([torch.cat([parts[j][i][d] for j in range(3) for i in range(self.layers)]) for d in range(D)])
        jb = torch.cat([torch.cat([bparts[j][i][d] for j in range(3) for i in range(self.layers)]) for d in range(D)])
        self.kv_joint = put(joint.T, cols, dtype=wdt)
        self.kv_joint_b = put(jb.reshape(1, -1), cols)
        kn = torch.stack([knw[i] for i in range(self.layers) for _ in range(nkv)]).reshape(1, self.layers * nkv, 1, hd)
        self.knorm_joint = put(kn, dtype=ttnn.bfloat16)
        self.block_size = block_size
        self.num_blocks = -(-(-(-(max_positions + self.block) // block_size)) // 8) * 8  # (32-byte page-table rows)
        cache_users = cache_users or users
        self.ring = ring
        if ring:
            # the window's positions plus the block (and a block of alignment) never wrap onto each other
            assert window and ring * block_size >= window + self.block + block_size
            phys = ring
            user_tables = (torch.arange(cache_users)[:, None] * ring + torch.arange(self.num_blocks)[None] % ring).int()
        else:
            phys = self.num_blocks
            user_tables = torch.arange(cache_users * phys, dtype=torch.int32).reshape(cache_users, phys)
        self.user_tables = user_tables
        self.page_table = _replicate(
            mesh, user_tables[:users].contiguous(), ttnn.int32
        )  # row u: user u (each 32-row step)
        heads_split = ttnn.ShardTensorToMesh(mesh, dim=1)
        self.caches = []
        for _ in range(self.layers):
            z = torch.zeros(cache_users * phys, D * nkv, block_size, hd)
            self.caches.append(
                tuple(
                    ttnn.from_torch(
                        z,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=heads_split,
                    )
                    for _ in range(2)
                )
            )
        from ..model import create_rope_setup

        self.rope = create_rope_setup(mesh, target.hf_config, max_local_batch_size=U32, datatype=ttnn.bfloat16)
        # all rows' Q / K norms and RoPE at once (prefill layout, per-row cos / sin); K / V written by BatchedKVWrite,
        # attention by the block-attention kernels
        self.ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        # the context pass's fc (compute-bound: 21 GFLOP per pass) at LoFi: round 23.0 -> 22.9 ms at ISL 256, acceptance
        # 5.02 / 5.01 tokens per round (in its spread)
        self.fc_ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        # the pass activations between ops in L1
        self.act_mc = ttnn.L1_MEMORY_CONFIG
        # (with BF16 weights HiFi2 / LoFi were no faster; with the BFP8 ones HiFi2 is: 26.8 -> 26.6 ms, LoFi as HiFi2)
        self.mm_ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        # the drafter's LM head on a BFP4 copy of the target's (unpadded, _even_lm_head) with BF8 rows: 0.89 -> 0.52 ms
        # per pass, live-like acceptance on a fixed text 3.07 (BF16 rows x BF8 weight: 3.05 / 3.10 in two runs)
        lm8 = _even_lm_head(target)
        self.lm_head = ttnn.typecast(lm8, ttnn.bfloat4_b)
        # BFP8 logits (the argmax reads them as stored; half the 26 MB BF16 write and read): round 23.3 -> 23.2 ms at
        # ISL 256, acceptance in its spread
        self.logits_dt = ttnn.bfloat8_b
        self.argmax = _Argmax(lm8.shape[-1], target.vocab_size)
        self._setup_passes(mesh, users, window)

    def _setup_passes(self, mesh, users, window):
        """The passes' block-dependent pieces (rows = users x block)."""
        nq, nkv, hd, block_size = self.nq, self.nkv, self.hd, self.block_size
        from ..batched_decode import sharded_norm_program_config
        from ..fused_decode.config import width_sharded_memory_config

        # the FP32 residual block sharded on 10 x 8 cores, one 32-row tile x 9 column tiles each, as the verify's: the
        # sharded norm reduces over 10 cores per row tile
        self.res_mc = _residual_mc(self.rows, self.hidden)
        self.norm_pc = sharded_norm_program_config(self.res_mc)
        # each residual add (+ the o_proj bias) and the norm after it in one program (_ResNorm, FP32 residual, FP32 norm
        # outputs as the sharded norm's); the all-reduces' outputs stay in their width shards
        self.rn = _ResNorm(mesh, self.rows, self.hidden, self.res_mc,
                           width_sharded_memory_config(mesh, self.hidden, _TPAllReduce.CORES, rows=self.rows))  # fmt: skip
        self.kv_write = BatchedKVWrite(mesh, users, self.block, nkv, hd, block_size)
        # (the context pass's 8-layer write: one core per (user, kv head))
        self.kv_write_ctx = BatchedKVWrite(mesh, users, self.block, nkv, hd, block_size, head_split=True)
        # attention: block attention (b1's kernels, uniform), as many users per program as the grid holds on 1 history
        # slice each (the window bounds every core's history to ~33 chunks; 16 x 1 vs 8 x 3: round 25.0 -> 24.8 ms at ISL
        # 256, 25.8 -> 25.4 at 1024; the whole grid vs 64 cores: 256 users' round at 8K 65.9 -> 65.3 ms)
        from .block_attention import BlockAttention

        grid = mesh.compute_with_storage_grid_size()
        per = min(users, grid.x * grid.y // BlockAttention.cores_needed(nkv, self.block, 1))
        self.ba = _UserBlockAttention(
            mesh, users, self.block, nq, nkv, hd, block_size, per, 1, uniform=True, window=window or 0
        )
        for L in self.L:  # (the weights' tiles made here, not during a trace capture)
            self.rn.gamma(L["ln1"])
            self.rn.gamma(L["ln2"])
            self.rn.row_tiles(L["o_b"])
        self.rn.gamma(self.norm)

    # -- pieces ----------------------------------------------------------------------------------------------------
    def _linear(self, x, w, bias=None, ck=None):
        pc = _mm_config_256(w.shape[-2], w.shape[-1], self.rows) if x.shape[-2] == self.rows else None
        # (a pass's 256-row activations in L1; the prompt context's chunks in DRAM)
        mc = self.act_mc if x.shape[-2] == self.rows else ttnn.DRAM_MEMORY_CONFIG
        if pc is None:  # (the auto config, fixed: _matmul_config)
            pc = _matmul_config(x, w, mc.buffer_type == ttnn.BufferType.DRAM, fp32=True)
        return ttnn.linear(
            x,
            w,
            bias=bias,
            program_config=pc,
            dtype=ttnn.bfloat16,
            compute_kernel_config=ck or self.mm_ck,
            memory_config=mc,
        )

    def _gate_up(self, x, L):
        """act = silu(x W_gate) * (x W_up) [rows, I] BF16 (L1 interleaved): one joint matmul into 10 x 8 block shards and
        one silu * up program on them (fused_decode/kernels/draft_silu_mul_*.cpp)."""
        from ..fused_decode.boundary import KERNEL_DIR

        K, N2 = L["gu"].shape[-2], L["gu"].shape[-1]
        gx = 10
        gy, mpc = _row_grid(self.rows)  # (mpc row tiles per core: SM_MPC)
        nh = N2 // 2 // TILE // gx
        sm_def = [("SM_MPC", str(mpc))] if mpc > 1 else []
        gu_mc = ttnn.create_sharded_memory_config((mpc * TILE, N2 // gx), ttnn.CoreGrid(x=gx, y=gy), ttnn.ShardStrategy.BLOCK,
                                                  ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)  # fmt: skip
        gu = ttnn.linear(x, L["gu"], program_config=_mm_config_256(K, N2, self.rows), dtype=ttnn.bfloat16,
                         compute_kernel_config=self.mm_ck, memory_config=gu_mc)  # fmt: skip
        act = ttnn.empty([1, 1, self.rows, N2 // 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
                         memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        cb_gu, cb_act = 0, 1
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "draft_silu_mul_reader.cpp"), core_ranges=cores,
                compile_time_args=[cb_gu, nh], runtime_args=[], defines=sm_def,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "draft_silu_mul_writer.cpp"), core_ranges=cores,
                compile_time_args=[cb_act, nh, N2 // 2 // TILE] + ttnn.TensorAccessorArgs(act).get_compile_time_args(),
                runtime_args=[], common_runtime_args=[act.buffer_address()], defines=sm_def,
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "draft_silu_mul_compute.cpp"), core_ranges=cores,
                compile_time_args=[cb_gu, cb_act, nh], runtime_args=[], defines=sm_def,
                config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True,
                                                    math_approx_mode=False),
            ),
        ]  # fmt: skip
        cbs = [
            ttnn.cb_descriptor_from_sharded_tensor(cb_gu, gu),
            ttnn.CBDescriptor(total_size=2048 * 2, core_ranges=cores,
                              format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_act, data_format=ttnn.bfloat16,
                                                                          page_size=2048)]),
        ]  # fmt: skip
        ttnn.generic_op([gu, act], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs))
        gu.deallocate(True)
        return act

    def rows_cos_sin(self, pos_rows):
        """Per-row cos / sin [1, 1, rows, hd] for positions pos_rows (device [1, rows] UINT32)."""
        out = []
        for m in (self.rope.cos_matrix, self.rope.sin_matrix):
            t = ttnn.embedding(pos_rows, m, layout=ttnn.TILE_LAYOUT)
            out.append(ttnn.reshape(t, [1, 1, self.rows, self.hd]))
        return out

    # -- prompt context ---------------------------------------------------------------------------------------------
    def add_prompt_context(self, features, user, n, chunk=2048):
        """The prompt's context K / V (prefill-style, paged_fill_cache of block-aligned chunks). features: the target
        features of the first n prompt positions, [1, 1, >= n rows, ctx_in] BF16 on the device (TargetCapture.on_device:
        rows past n, and the chunk's padding up to a block, are past the prompt: the rounds rewrite those positions
        before they are read)."""
        mesh = self.mesh_device
        pt_host = self.user_tables[user][None]
        qw, kvw = self.nq * self.hd, self.nkv * self.hd
        avail = features.shape[2]
        first = 0
        if self.ring:
            # only the positions a round can read (the last `window` before the first block), block aligned, in one
            # chunk (a chunk's blocks must not wrap onto each other)
            first = max(0, (n - self.window) // self.block_size * self.block_size)
            chunk = n - first + self.block_size
            assert chunk <= self.ring * self.block_size
        src = features.buffer_address()

        def free(t):  # (a slice of the whole tensor is the tensor itself: the caller's features stay allocated)
            if t.buffer_address() != src:
                t.deallocate(True)

        for s0 in range(first, n, chunk):
            rows = min(chunk, -(-(n - s0) // self.block_size) * self.block_size)
            tx = ttnn.slice(features, [0, 0, s0, 0], [1, 1, min(s0 + rows, avail), self.ctx_in])
            if s0 + rows > avail:
                txp = ttnn.pad(tx, [(0, 0), (0, 0), (0, s0 + rows - avail), (0, 0)], value=0.0)
                free(tx)
                tx = txp
            ctx = self._linear(tx, self.fc_full)
            free(tx)
            ctx = ttnn.rms_norm(ctx, epsilon=self.eps, weight=self.hidden_norm, compute_kernel_config=self.ck)
            cos = self.rope.cos_matrix_prefill[:, :, s0 : s0 + rows, :]
            sin = self.rope.sin_matrix_prefill[:, :, s0 : s0 + rows, :]
            trans = self.rope.get_both_trans_mats()["prefill"]
            b0 = s0 // self.block_size
            chunk_pt = _replicate(mesh, pt_host[:, b0 : b0 + rows // self.block_size].contiguous(), ttnn.int32)
            for i in range(self.layers):
                L = self.L[i]
                qkv = self._linear(ctx, L["qkv"], L["qkv_b"])
                for j, cache in enumerate(self.caches[i]):
                    t = ttnn.slice(qkv, (0, 0, 0, qw + j * kvw), (1, 1, rows, qw + (j + 1) * kvw))
                    t = ttnn.transpose(ttnn.reshape(t, [1, rows, self.nkv, self.hd]), 1, 2)
                    if j == 0:
                        t = ttnn.rms_norm(t, epsilon=self.eps, weight=L["k_norm"], compute_kernel_config=self.ck)
                        t = ttnn.experimental.rotary_embedding_llama(t, cos, sin, trans, is_decode_mode=False)
                    ttnn.experimental.paged_fill_cache(cache, t, chunk_pt, batch_idx=0)
                    t.deallocate(True)
                qkv.deallocate(True)
            chunk_pt.deallocate(True)
            ctx.deallocate(True)

    # -- passes -----------------------------------------------------------------------------------------------------
    def context(self, features, pos, rot):
        """features: device [1, 1, rows, 5 * hidden] (row 32 i + u); pos: per-step positions; rot: per-row (cos, sin)."""
        ctx = self._linear(features, self.fc_full, ck=self.fc_ck)
        # the hidden norm on the residual's shards (interleaved: 8 cores, 44 us)
        cs = ttnn.to_memory_config(ctx, self.res_mc)
        ctx.deallocate(True)
        n = ttnn.rms_norm(
            cs, epsilon=self.eps, weight=self.hidden_norm, program_config=self.norm_pc, memory_config=self.res_mc,
            compute_kernel_config=self.ck,
        )  # fmt: skip
        cs.deallocate(True)
        ctx2 = ttnn.to_memory_config(n, self.act_mc)
        n.deallocate(True)
        # every layer's K / V of the context rows in one projection (kv_joint), heads op, norm, RoPE and cache write
        L, nkv = self.layers, self.nkv
        qkv = self._linear(ctx2, self.kv_joint, self.kv_joint_b)
        ctx2.deallocate(True)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=L * nkv, num_kv_heads=L * nkv, transpose_k_heads=False, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        qkv.deallocate(True)
        q.deallocate(True)
        n = ttnn.rms_norm(k, epsilon=self.eps, compute_kernel_config=self.ck)
        k.deallocate(True)
        nw = ttnn.multiply(n, self.knorm_joint, memory_config=ttnn.L1_MEMORY_CONFIG)
        n.deallocate(True)
        trans = self.rope.get_both_trans_mats()["prefill"]
        k2 = ttnn.experimental.rotary_embedding_llama(nw, rot[0], rot[1], trans, is_decode_mode=False,
                                                      memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
        nw.deallocate(True)
        self.kv_write_ctx([k2] * L, [v] * L, self.caches, pos[self.block], self.page_table, src_head_stride=nkv)
        k2.deallocate(True)
        v.deallocate(True)

    def draft(self, tokens, pos, rot, bound):
        """tokens: device [1, rows] UINT32 noise block (row 32 i + u: user u's last token for i = 0, else the mask).
        Returns the per-device argmax of every row (rows 32 i + u, i >= 1: the draft of position start_u + i)."""
        _TPAllReduce.reset()
        emb = ttnn.embedding(tokens, self.target.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        # the FP32 residual stays block-sharded: the residual adds and the norms run on its shards (_ResNorm)
        h = ttnn.to_memory_config(ttnn.reshape(emb, [1, 1, self.rows, self.hidden]), self.res_mc, dtype=ttnn.float32)
        n = ttnn.rms_norm(
            h, epsilon=self.eps, weight=self.L[0]["ln1"], program_config=self.norm_pc, memory_config=self.res_mc,
            compute_kernel_config=self.ck,
        )  # fmt: skip
        x = ttnn.to_memory_config(n, self.act_mc)
        n.deallocate(True)
        rn, f32 = self.rn, ttnn.float32
        trans = self.rope.get_both_trans_mats()["prefill"]
        for li in range(self.layers):
            L = self.L[li]
            qkv = self._linear(x, L["qkv"], L["qkv_b"])
            x.deallocate(True)
            # heads split + per-head norms + RoPE in one program (_qkv_rope)
            q_rows, k, v = _qkv_rope(self.mesh_device, trans, qkv, *rot, self.nq, self.nkv, self.nq + self.nkv,
                                     headnorm=(L["q_norm_t"], L["k_norm_t"], self.nq, self.eps))  # fmt: skip
            qkv.deallocate(True)
            self.kv_write([k], [v], [self.caches[li]], pos[self.block], self.page_table)
            k.deallocate(True)
            v.deallocate(True)
            kc, vc = self.caches[li]
            heads = self.ba(q_rows, kc, vc, pos[self.block], self.page_table, None, self.hd**-0.5)
            q_rows.deallocate(True)
            partial = self._linear(heads, L["o"])
            heads.deallocate(True)
            o = _all_reduce(partial)
            partial.deallocate(True)
            h2, x = rn(o, h, L["ln2"], eps=self.eps, bias=L["o_b"], out_dtype=f32)
            o.deallocate(True)
            h.deallocate(True)
            act = self._gate_up(x, L)
            x.deallocate(True)
            partial = self._linear(act, L["down"])
            act.deallocate(True)
            mlp = _all_reduce(partial)
            partial.deallocate(True)
            last = li + 1 == self.layers
            if last:  # the final norm straight to BFP8 (the LM head's input)
                h, x8 = rn(mlp, h2, self.norm, out_bf16=False, x8=True, eps=self.eps)
            else:
                h, x = rn(mlp, h2, self.L[li + 1]["ln1"], eps=self.eps, out_dtype=f32)
            mlp.deallocate(True)
            h2.deallocate(True)
        h.deallocate(True)
        # rows 0 .. users - 1 (each user's last token, i = 0) draft nothing: the LM head on the other rows only
        x8r = ttnn.slice(x8, [0, 0, self.users, 0], [1, 1, self.rows, self.hidden])
        x8.deallocate(True)
        logits = ttnn.matmul(
            x8r, self.lm_head, dtype=self.logits_dt, program_config=_matmul_config(x8r, self.lm_head, True)
        )
        x8r.deallocate(True)
        out = self.argmax.device(logits, self.rows)
        logits.deallocate(True)
        return out


class _UserBlockAttention:
    """Block attention for the verify's full-attention layers (b1's dflash/block_attention.py kernels, one instance per
    user): each user's K / V history is read once for its 8 block rows (causal within the block, sinks), not once per
    row as 8 decode-SDPA steps do. `per_program` users run side by side in one program, each on its own
    4 x slices cores (kv head x packed Q tile x history slice); 32 / per_program programs per layer."""

    WAIT_SEM = 8  # the kernels' wait semaphore (count 0: Q and the block's K / V are written by earlier ops)

    def __init__(self, mesh, users, block, n_heads, n_kv, head_dim, block_size, per_program, slices, min_chunks=4, uniform=False, window=0, swin=0):  # fmt: skip
        """uniform (the drafter): every row of a block attends [start, start + 7]'s history up to start + 7 (no causal
        mask inside the block, no sinks), within the last `window` positions. swin (the verify's sliding-window layers):
        row r attends [pos_r + 1 - swin, pos_r] (causal, with the sinks)."""
        from .block_attention import BlockAttention

        self.mesh, self.users, self.block, self.P = mesh, users, block, per_program
        grid = mesh.compute_with_storage_grid_size()
        per = BlockAttention.cores_needed(n_kv, block, slices)
        cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
        assert per * per_program <= len(cores)
        # one kernel set for the program's users (a set per user: 3 x per_program kernel groups to dispatch, the
        # sliding-window program 57 us); the last program may hold fewer users: more history slices per user, up to
        # the cores of a full program (its users' histories split over the same number of cores)
        full = per * per_program
        self.programs = []
        for p0 in range(0, users, per_program):
            n = min(per_program, users - p0)
            s = slices
            while s < 8 and BlockAttention.cores_needed(n_kv, block, s + 1) * n <= full:
                s += 1
            self.programs.append((p0, n, s))
        self.inst = {
            (n, s): BlockAttention(
                mesh,
                block,
                n_heads,
                n_kv,
                head_dim,
                block_size,
                cores[: BlockAttention.cores_needed(n_kv, block, s) * n],
                s,
                q_rows=True,
                min_chunks=min_chunks,
                uniform=uniform,
                window=window,
                swin=swin,
                users=n,
            )
            for n, s in sorted({(n, s) for _, n, s in self.programs})
        }

    def __call__(self, q, k_cache, v_cache, pos_ui, page_table, sinks, scale, q_heads=None):
        """q: the rotated queries in the heads layout [1, heads, rows, hd] (the RoPE output; row 32 i + u; q_heads: its
        first q_heads heads are the queries, e.g. Q and K in one RoPE output); pos_ui: INT32 [rows] (8 u + i: start_u +
        i); page_table: INT32 [users, blocks]. Returns the output [1, 1, rows, heads * hd] (o_proj's input, L1)."""
        shape = [1, 1, q.shape[2], (q_heads or q.shape[1]) * q.shape[3]]
        out = ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh,
                         memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
        for p0, n, s in self.programs:
            inst = self.inst[(n, s)]
            kernels, cbs, tio = inst.parts(
                q, k_cache, v_cache, pos_ui, page_table, sinks, scale, wait=(self.WAIT_SEM, 0),
                user=(p0, self.users), out=out,
            )  # fmt: skip
            sems = inst.semaphores() + [ttnn.SemaphoreDescriptor(self.WAIT_SEM, ttnn.CoreType.WORKER, inst.cores, 0)]
            io = [t for t in tio if t is not out]
            ttnn.generic_op(io + [out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return out


class BatchedVerify:
    """The target's 256-row verify pass for 32 users on the batched decode model's layers (DecoderLayer.batched_decode):
    every op over all 256 rows at once, the residual block sharded on 10 x 8 cores."""

    def __init__(self, model, page_table, kv_cache, users, block, capture_layers, moe_copies=True):
        """moe_copies: the interleaved MoE weight copies GroupedMoE relays out (False: the grouped MoE is shared)."""
        self.model = model
        self.mesh = model.mesh_device
        self.users, self.block = users, block
        self.rows = users * block
        self.page_table = page_table
        self.kv_cache = kv_cache
        self.capture_layers = list(capture_layers)
        self.hidden = model.hf_config.hidden_size
        # every layer's MoE weights in DRAM interleaved (the grouped MoE relays them out at init, GroupedMoE)
        self.moe = []
        for layer in model.layers:
            assert layer.batched_decode, "BatchedVerify runs on the batched decode model (batch 32, dense MoE)"
            if not moe_copies:
                continue
            dw = layer.mlp.experts.dense_decode_weights
            n_act = dw.gate.shape[-1]
            self.moe.append(
                dict(
                    gate=ttnn.to_memory_config(dw.gate, ttnn.DRAM_MEMORY_CONFIG),
                    up=ttnn.to_memory_config(dw.up, ttnn.DRAM_MEMORY_CONFIG),
                    down=ttnn.to_memory_config(dw.down, ttnn.DRAM_MEMORY_CONFIG),
                    gate_bias=ttnn.slice(dw.gate_bias, [0, 0, 0, 0], [1, 1, 1, n_act]),
                    up_bias=ttnn.slice(dw.up_bias, [0, 0, 0, 0], [1, 1, 1, n_act]),
                    down_bias=dw.down_bias,
                )
            )
        self.lm_head = _even_lm_head(model)
        self.argmax = _Argmax(self.lm_head.shape[-1], model.vocab_size)
        # the grouped MoE (dflash/grouped_moe.py: only the routed (row, expert) pairs), set by the runner
        self.grouped = None
        from ..fused_decode.config import width_sharded_memory_config

        attn0 = model.layers[0].self_attn
        mc = model.layers[0].self_attn.mesh_config
        self.local_heads = mc.shard_size(attn0.config.num_heads)
        self.local_kv_heads = mc.shard_size(attn0.config.num_kv_heads)
        hd = self.head_dim = attn0.config.head_dim
        # the residual block sharded on 10 x 8 cores, one 32-row tile x 9 column tiles each: the sharded norm reduces
        # over 10 cores per row tile instead of 30 per 8 (standalone 14.6 -> 6.4 us per norm, 49 per pass)
        self.residual_mc = _residual_mc(self.rows, self.hidden)
        # each residual add and the norm after it in one program (_ResNorm); the post-attention one also computes the
        # router's logits (split K over its 10 cores of a row, FP32 sums) and the BFP8 copy for the grouped MoE's gather
        self.resnorm = _ResNorm(self.mesh, self.rows, self.hidden, self.residual_mc,
                                width_sharded_memory_config(self.mesh, self.hidden, _TPAllReduce.CORES, rows=self.rows))  # fmt: skip
        for l in model.layers:  # (the weights' tiles made here, not during a trace capture)
            self.resnorm.gamma(l.input_layernorm)
            self.resnorm.gamma(l.post_attention_layernorm)
            self.resnorm.router_bias(l.mlp.router)
        self.resnorm.gamma(model.norm)
        # the unpadded decode o_proj (batched_decode.BatchedOProj) is DRAM width-sharded: interleaved copies
        self.oproj_w = [
            ttnn.to_memory_config(layer.self_attn.batched_oproj.weight, ttnn.DRAM_MEMORY_CONFIG)
            for layer in model.layers
        ]
        # QKV / o_proj over 256 rows: the batched decode's compute config (HiFi2, BF16 accumulation in L1)
        self.qkv_ck = attn0.batched_qkv.compute_kernel_config
        # full-attention layers: block attention (each user's K / V read once for its 8 rows), 8 users per program on 3
        # history slices each (96 cores): 24 layers' attention at 16K context 47.9 -> 18.1 ms (16 x 1: 20.3, 4 x 6:
        # 20.6), at 300 11.3 -> 10.9 ms
        # the heads' sink tiles in L1 (every slice-0 core of a program reads them: one DRAM page each)
        self.sinks = [
            ttnn.to_memory_config(l.self_attn.weights.decode_sinks, ttnn.L1_MEMORY_CONFIG) for l in model.layers
        ]
        # (users x slices measured at ISL 256 / 4K / 16K: 8 x 3 18.9 / 21.4 / 26.4 ms per round, 12 x 2 18.5 / 21.0 /
        # 26.6, 16 x 1 18.4 / 21.2 / 28.7)
        from .block_attention import BlockAttention

        # (block 8: 8 users per program on 3 history slices each, 96 cores: 4 equal programs for 32 users (the whole
        # grid, 9 + 9 + 9 + 5 users: round 21.5 -> 21.8 ms at ISL 4K); otherwise as many users per program as the grid
        # holds: 3 slices at 32 users (plain step at 4K 10.36 -> 9.76 ms vs 96 cores), else 1 (64 users at ISL 16K
        # 23.3 -> 22.2 ms, 256 at 8K 43.4 -> 40.7); 2 or 3 slices for every user: slower, e.g. 64 users at 16K 24.2 /
        # 24.6 ms)
        slices = 3 if users <= 32 else 1  # (more users per pass, e.g. a 512 / 1024-row pass: 1 slice, the grid's width)
        grid = self.mesh.compute_with_storage_grid_size()
        budget = 96 if block == 8 and users <= 32 else grid.x * grid.y
        per = min(users, budget // BlockAttention.cores_needed(self.local_kv_heads, block, slices))
        self.block_attn = _UserBlockAttention(
            self.mesh, users, block, self.local_heads, self.local_kv_heads, hd, kv_cache[1][0].shape[2], per, slices
        )
        # sliding-window layers: the same block attention, windowed (row r attends [pos_r + 1 - window, pos_r]: at most
        # 6 chunks), as many users per program as the grid holds on 1 slice each (vs 64 cores: 256 users' plain step
        # 16.6 -> 16.3 ms), instead of one decode SDPA per 32-row step (8 SDPA calls + 8 Q slices + the concat per layer)
        windows = {l.self_attn.config.sliding_window for l in model.layers if l.self_attn.config.sliding_window}
        (window,) = windows
        self.swin_attn = _UserBlockAttention(
            self.mesh,
            users,
            block,
            self.local_heads,
            self.local_kv_heads,
            hd,
            kv_cache[0][0].shape[2],
            min(users, grid.x * grid.y // BlockAttention.cores_needed(self.local_kv_heads, block, 1)),
            1,
            swin=window,
        )
        # heads, RoPE and the K / V cache writes once per layer for all 256 rows (prefill-layout heads, per-row cos / sin,
        # the multi-user BFP8_b write kernel) instead of per 32-row step (interleaved_to_sharded_partial, decode heads,
        # fused-QK RoPE, paged_fused_update_cache: 4 x 8 ops per layer): round 42.2 -> 41.0 ms at ISL 256. The kernel
        # packs the exact BF16 rows (round half up, as the device packer); the per-step path stored ~10% of the values
        # one BFP8 step larger in magnitude (tests/_scratch_b32_verify.py test_kvwrite). trans_rows (the prefill RoPE
        # transformation matrix of the target's RoPE) is set by the runner.
        self.trans_rows = None
        # the activations between the attention's ops in L1
        self.attn_mc = ttnn.L1_MEMORY_CONFIG
        # the drafter's context features in BFP8 (the context fc multicasts them to its 90 cores: 275 -> 218 us;
        # acceptance in its spread)
        self.feat_dt = ttnn.bfloat8_b
        self.kv_write = BatchedKVWrite(self.mesh, users, block, self.local_kv_heads, hd, kv_cache[1][0].shape[2],
                                       head_split=True)  # fmt: skip

    def _attention(self, li, layer, x, pos, rot):
        """The layer's attention for all rows: QKV, the attention core (_attention_rows) and o_proj (+ all-reduce), each
        once over the 256 rows. x: the input norm's output (deallocated here); rot: every row's cos / sin [1, 1, rows,
        hd]. Returns the all-reduce output (its width shards: _ResNorm adds it to the residual)."""
        attn_mod = layer.self_attn
        w = attn_mod.weights
        amc = self.attn_mc
        qkv = ttnn.linear(
            x,
            w.wqkv,
            bias=w.wqkv_bias,
            program_config=_mm_config_256(w.wqkv.shape[-2], w.wqkv.shape[-1], self.rows),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.qkv_ck,
            memory_config=amc,
        )
        x.deallocate(True)
        heads = self._attention_rows(li, attn_mod, qkv, pos, rot)
        qkv.deallocate(True)
        # the bias inside the matmul (BF16 output; the all-reduce converts it to its BFP8 input shards)
        o = ttnn.linear(heads, self.oproj_w[li], bias=attn_mod.batched_oproj.bias,
                        program_config=_mm_config_256(heads.shape[-1], self.oproj_w[li].shape[-1], self.rows),
                        dtype=ttnn.bfloat16, compute_kernel_config=self.qkv_ck, memory_config=amc)  # fmt: skip
        heads.deallocate(True)
        out = _all_reduce(o)
        o.deallocate(True)
        return out

    def _attention_rows(self, li, attn_mod, qkv, pos, cos_sin):
        """The attention core for all 256 rows at once: the QKV heads split into Q and K in one heads-layout tensor
        [1, nh + nkv, rows, hd] (rotated per row) and V heads 0 / 1, by one program (_qkv_rope); K / V into the BFP8_b
        caches by one multi-user kernel; then block attention (windowed on the sliding-window layers). Returns the
        concatenated heads [1, 1, rows, heads * hd] (row 32 i + u): the block attention reads Q in the RoPE output's heads
        layout and writes o_proj's input layout itself (no transposes, no concat-heads op)."""
        cfg = attn_mod.config
        nh, nkv = self.local_heads, self.local_kv_heads
        k_cache, v_cache = self.kv_cache[li]
        cos, sin = cos_sin
        qk2, v0, v1 = _qkv_rope(self.mesh, self.trans_rows, qkv, cos, sin, nh + nkv, 1, nh + nkv)
        self.kv_write.joint(qk2, v0, v1, (k_cache, v_cache), pos[self.block], self.page_table, nh)
        v0.deallocate(True)
        v1.deallocate(True)
        ba = self.swin_attn if cfg.sliding_window else self.block_attn
        out = ba(qk2, k_cache, v_cache, pos[self.block], self.page_table, self.sinks[li], cfg.scaling, q_heads=nh)
        qk2.deallocate(True)
        return out

    def forward(self, tokens, pos, rot):
        """tokens: device [1, rows] UINT32 (row 32 i + u). pos[i]: step i's [32] INT32 positions (pos[block]: every
        row's, user-major); rot: every row's cos / sin [1, 1, rows, hd] (BatchedDraft.rows_cos_sin). Returns (per-device
        argmax of every row, device features [1, 1, rows, 5 hidden]). Each residual add and the norm after it run in one
        program (_ResNorm): the attention's and the MoE's all-reduce outputs stay in their width shards, the norms'
        outputs land in L1 interleaved."""
        _TPAllReduce.reset()
        model = self.model
        emb = ttnn.embedding(tokens, model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        x = ttnn.reshape(emb, [1, 1, self.rows, self.hidden])
        res = ttnn.to_memory_config(x, self.residual_mc, dtype=ttnn.bfloat16)
        x.deallocate(True)
        n0 = model.layers[0].input_layernorm.forward_sharded(res)
        xin = ttnn.to_memory_config(n0, self.attn_mc)
        n0.deallocate(True)
        feats = []
        L = len(model.layers)
        for li, layer in enumerate(model.layers):
            y = self._attention(li, layer, xin, pos, rot)  # (deallocates xin)
            # the MoE input in BFP8 and the router's logits from the same program; the routing + plan in one program
            # (GroupedMoE.route), the combine adds the down bias
            res2, x8, logits = self.resnorm(y, res, layer.post_attention_layernorm, out_bf16=False, x8=True,
                                            router=layer.mlp.router)  # fmt: skip
            y.deallocate(True)
            res.deallocate(True)
            self.grouped.route(logits, li)
            logits.deallocate(True)
            down = self.grouped.from_x8(li, x8)
            x8.deallocate(True)
            y = _all_reduce(down)
            down.deallocate(True)
            nxt = model.layers[li + 1].input_layernorm if li + 1 < L else model.norm
            res, xin = self.resnorm(y, res2, nxt)
            y.deallocate(True)
            res2.deallocate(True)
            if li in self.capture_layers:
                feats.append(ttnn.to_memory_config(res, ttnn.DRAM_MEMORY_CONFIG, dtype=self.feat_dt))
        features = ttnn.concat(feats, dim=-1) if feats else None
        for t in feats:
            t.deallocate(True)
        res.deallocate(True)
        normed8 = ttnn.typecast(xin, ttnn.bfloat8_b)
        xin.deallocate(True)
        logits8 = ttnn.matmul(  # as the batched decode's LM head
            normed8, self.lm_head, dtype=ttnn.bfloat8_b, program_config=_matmul_config(normed8, self.lm_head, True)
        )
        normed8.deallocate(True)
        out = self.argmax.device(logits8)  # (reads the BFP8 logits as stored)
        logits8.deallocate(True)
        return out, features


class _Lane:
    """One lane's persistent per-round state (BatchedDFlashRunner: 32 users per lane)."""


class BatchedDFlashRunner:
    """Greedy DFlash for `users` sequences (each prefilled into its own KV slot) on the batched decode model. More than
    32 users run as lanes of 32: every lane's round (draft, verify, context, acceptance: the batch-32 passes on the
    lane's users) in one trace, the weights and the passes' buffers shared."""

    ROWS = 256  # rows per pass: users per lane x block

    def __init__(self, generator, model, page_table, kv_cache, users=32, window=1024, eos_ids=(200002, 200012),
                 ring=None, block=None, share=None, rows_per_pass=None):  # fmt: skip
        """ring: the drafter's per-user cache as a ring of `ring` blocks (BatchedDraft; with more than one lane, 24 by
        default: the window's 1024 positions + a block). share: a runner over the same users (prefilled) whose drafter
        (weights, caches with the prompt context) and grouped MoE weights this one uses, with its own block."""
        from .capture import TargetCapture

        self.generator, self.model, self.kv_cache = generator, model, kv_cache
        mesh = model.mesh_device
        self.mesh = mesh
        self.total_users = users
        # rows per verify / draft pass: 256, or 512 from 128 users (one pass of 128 users x 4 rows / 256 x 2; measured at
        # ISL 256, 512 new tokens: 128 users 101.5 tok/s/user vs 89.4 with 2-row blocks in 256-row passes, 256 users
        # 55.7 vs 45.2 with two 256-row passes; at 64 users 8 rows in a 512-row pass 155 vs 167 with 4 rows in 256)
        self.ROWS = rows_per_pass or (512 if users >= 128 else self.ROWS)
        # block: the draft / verify block per user; a lane holds ROWS / block users. Default: the largest block that keeps
        # every user in one pass, at least 2 and at most the checkpoint's 8: 8 rows at 32 users, 4 at 64 and 128, 2 at 256
        # (256-row passes, ISL 256, 256 new tokens: 32 users 8 rows 247 vs 4 rows 207 tok/s/user; 64 users 4 rows 169 vs
        # 8 rows 124)
        if block is None:
            ckpt = load_draft_config(draft_snapshot())["block_size"]
            block = max(2, min(ckpt, self.ROWS // max(users, 1)))
            block = 1 << (block.bit_length() - 1)  # (a power of 2: the block attention's row counts)
        self.LANE = self.ROWS // block
        lanes = max(1, users // self.LANE)
        assert users == self.LANE * lanes or (lanes == 1 and users <= self.LANE)
        users = min(users, self.LANE)  # (a lane's users)
        self.users = users
        self.page_table_host = page_table
        self.V = model.vocab_size
        self.eos_ids = set(eos_ids)
        max_pos = page_table.shape[-1] * 64
        if ring is None and self.total_users > 32:  # (more users: the drafter's cache only holds its window)
            ring = 24
        if share is not None:
            ring = share.draft.ring
        self.draft = BatchedDraft(mesh, model, max_pos, users, window=window, cache_users=self.total_users, ring=ring,
                                  block=block, share=share.draft if share is not None else None)  # fmt: skip
        self.block = B = self.draft.block
        self.rows = users * B
        if share is not None:
            self.capture = share.capture
        else:
            self.capture = TargetCapture(model, self.draft.target_layer_ids)
            self.capture.on_device = True  # (the prompt features stay on the device: no host round trip per user)
        kv = kv_cache[0] if isinstance(kv_cache[0][0], list) else kv_cache
        self.page_table = _replicate(mesh, page_table[:users].contiguous(), ttnn.int32)
        self.verify = BatchedVerify(model, self.page_table, kv, users, B, self.draft.target_layer_ids,
                                    moe_copies=share is None)  # fmt: skip
        self.verify.trans_rows = self.draft.rope.get_both_trans_mats()["prefill"]  # (the target's RoPE)
        from .grouped_moe import GroupedMoE

        # the verify MoE on only the routed (row, expert) pairs
        if share is not None:
            self.verify.grouped = GroupedMoE(mesh, None, self.verify.hidden, rows=self.rows, share=share.verify.grouped)
            for t in self.verify.oproj_w:  # (the share's copies)
                t.deallocate(True)
            self.verify.oproj_w = share.verify.oproj_w
        else:
            self.verify.grouped = GroupedMoE(mesh, self.verify.moe, self.verify.hidden, rows=self.rows)
            for W in self.verify.moe:  # (the interleaved copies were only the relayout's source)
                for k in ("gate", "up", "down"):
                    W.pop(k).deallocate(True)
        # the plain batched decode's dense MoE streams the same relaid gate|up pages (experts/decode.py
        # _dense_gate_up_stream: gate, up and SwiGLU in one program)
        for li, layer in enumerate(model.layers):
            layer.mlp.experts.gu_relaid = (self.verify.grouped.layers[li]["gate_up"], self.verify.grouped.gu_split)
        U = users
        put = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
        self.lanes = []
        for l in range(lanes):
            lane = _Lane()
            u0 = l * U
            lane.u0 = u0
            lane.page_table = (
                self.page_table if l == 0 else _replicate(mesh, page_table[u0 : u0 + U].contiguous(), ttnn.int32)
            )
            lane.draft_page_table = (self.draft.page_table if l == 0 else
                                     _replicate(mesh, self.draft.user_tables[u0 : u0 + U].contiguous(), ttnn.int32))  # fmt: skip
            # persistent per-round inputs (the traces' inputs)
            z = torch.zeros(users, dtype=torch.int32)
            lane.d_tokens = _replicate(mesh, torch.zeros(1, self.rows, dtype=torch.int32), ttnn.uint32)
            lane.v_tokens = _replicate(mesh, torch.zeros(1, self.rows, dtype=torch.int32), ttnn.uint32)
            # per-step positions, then (entry B) every row's position user-major (8 u + i: start_u + i, block attention)
            lane.pos = [_replicate(mesh, z, ttnn.int32) for _ in range(B)] + [
                _replicate(mesh, torch.zeros(self.rows, dtype=torch.int32), ttnn.int32)
            ]
            lane.bound = _replicate(mesh, z, ttnn.int32)
            lane.pos_rows = _replicate(mesh, torch.zeros(1, self.rows, dtype=torch.int32), ttnn.uint32)
            # the per-round inputs travel as two packed host writes (int32: per-step positions, user-major positions,
            # the draft bound; uint32: the draft tokens, per-row positions), unpacked on device into the tensors above
            # at the start of a pass (one host-to-device copy each instead of 11)
            lane.unpack_i = [(t, [U]) for t in lane.pos[:B]] + [(lane.pos[B], [self.rows]), (lane.bound, [U])]
            lane.unpack_u = [(lane.d_tokens, [1, self.rows]), (lane.pos_rows, [1, self.rows])]
            n_i = sum(math.prod(shape) for _, shape in lane.unpack_i)
            n_u = sum(math.prod(shape) for _, shape in lane.unpack_u)
            lane.in_i = _replicate(mesh, torch.zeros(1, n_i, dtype=torch.int32), ttnn.int32)
            lane.in_u = _replicate(mesh, torch.zeros(1, n_u, dtype=torch.int32), ttnn.uint32)
            lane.limit = put(torch.zeros(1, 1, 1, U))  # each user's last start position (its token budget)
            lane.round_out = None
            self.lanes.append(lane)
        # the next round's inputs are computed on device at the end of a round (acceptance on device): every traced
        # round is queued before the previous round's result is read
        self.iota8 = put(torch.arange(B).float().reshape(1, 1, B, 1).expand(1, 1, B, U).contiguous())  # row i: i
        self.mask_rest = put(torch.full((1, 1, 1, self.rows - U), float(self.draft.mask_token_id)))
        self.first_step = ttnn.from_torch(
            (torch.arange(self.rows) < users).float().reshape(1, 1, 1, self.rows),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self._use(self.lanes[0])
        self.traces = None
        hold_prefill_l1(model)

    def _use(self, lane):
        """The passes read lane's inputs and page tables (the attributes they use are set to lane's tensors)."""
        for k in ("d_tokens", "v_tokens", "pos", "bound", "pos_rows", "unpack_i", "unpack_u", "in_i", "in_u", "limit"):
            setattr(self, k, getattr(lane, k))
        self.page_table = lane.page_table
        self.verify.page_table = lane.page_table
        self.draft.page_table = lane.draft_page_table
        self.lane = lane

    def prefill(self, prompts, u0=0):
        """prompts: users u0, u0 + 1, ...; returns their first tokens."""
        hold_prefill_l1(self.model)
        firsts = []
        for u, p in enumerate(prompts, start=u0):
            logits = self.generator.prefill_forward_text(
                torch.tensor([p]),
                # the generator takes the page table rows of the prompts it is given (row 0 for a one-prompt call):
                # this user's row (the whole table wrote every user's K/V into user 0's blocks)
                page_table=self.page_table_host[u : u + 1],
                kv_cache=self.kv_cache,
                prompt_lens=[len(p)],
                empty_slots=[u % 32],  # (the model's batch: 32 slots; the page table row picks the blocks)
                enable_trace=False,
                warmup_prefill=False,
            )
            firsts.append(int(logits.reshape(-1)[: self.V].float().argmax()))
            features = self.capture.prefill_features_device()
            self.draft.add_prompt_context(features, user=u, n=len(p))
            features.deallocate(True)
        return firsts

    # -- passes (eager, or captured once and replayed) ----------------------------------------------------------------
    def _rot_mats(self):
        return self.draft.rows_cos_sin(self.pos_rows)

    def _unpack(self):
        """The packed per-round inputs into their tensors (slices of the two packed rows)."""
        for pack, parts in ((self.in_i, self.unpack_i), (self.in_u, self.unpack_u)):
            off = 0
            for dst, shape in parts:
                n = math.prod(shape)
                part = ttnn.reshape(ttnn.slice(pack, [0, off], [1, off + n]), shape)
                ttnn.copy(part, dst)
                part.deallocate(True)
                off += n

    def _draft_pass(self):
        self._unpack()
        rot = self._rot_mats()
        out = self.draft.draft(self.d_tokens, self.pos, rot, self.bound)
        for t in rot:
            t.deallocate(True)
        return out

    def _verify_pass(self):
        self._unpack()  # (the step-by-step path may run verify without a draft pass)
        rot = self._rot_mats()  # every verify row's cos / sin: the target's layers and the drafter's context
        out, features = self.verify.forward(self.v_tokens, self.pos, rot)
        self.draft.context(features, self.pos, rot)
        features.deallocate(True)
        for t in rot:
            t.deallocate(True)
        return out

    def _round_pass(self):
        """A whole round on the device: draft, the devices' argmax combined, the verify block [last token, 7 drafts]
        built from it, verify + context. Returns the draft and verify token ids ([1, 1, rows, 1] FLOAT32)."""
        self._unpack()
        rot = self._rot_mats()  # (the draft block's positions are the verify block's: one cos / sin for the round)
        d_out = self.draft.draft(self.d_tokens, self.pos, rot, self.bound)
        d_tok = self.draft.argmax.combine(d_out, self.mesh)
        for t in d_out:
            t.deallocate(True)
        row = ttnn.transpose(d_tok, -2, -1)  # [1, 1, 1, rows]
        lasts = ttnn.typecast(ttnn.to_layout(self.d_tokens, ttnn.TILE_LAYOUT), ttnn.float32)  # rows 0-31: last tokens
        vt = ttnn.where(self.first_step, ttnn.reshape(lasts, [1, 1, 1, self.rows]), row)
        row.deallocate(True)
        lasts.deallocate(True)
        v_tokens = ttnn.reshape(ttnn.to_layout(ttnn.typecast(vt, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [1, self.rows])
        vt.deallocate(True)
        out, features = self.verify.forward(v_tokens, self.pos, rot)
        v_tokens.deallocate(True)
        self.draft.context(features, self.pos, rot)
        features.deallocate(True)
        for t in rot:
            t.deallocate(True)
        v_tok = self.verify.argmax.combine(out, self.mesh)
        for t in out:
            t.deallocate(True)
        self._next_inputs(d_tok, v_tok)
        return d_tok, v_tok

    def _next_inputs(self, d_tok, v_tok):
        """The acceptance on device: per user the drafts up to the first mismatch with the target (k), the next start
        start + k + 1 (capped at the user's budget: finished users repeat their block), the target's token at row k as
        the next last token; the next round's packed inputs written from them (FLOAT32 math: ids < 2^24 are exact)."""
        U, B = self.users, self.block
        d8, v8 = (ttnn.reshape(ttnn.transpose(t, -2, -1), [1, 1, B, U]) for t in (d_tok, v_tok))  # [row i, user u]
        m = ttnn.eq(ttnn.slice(d8, [0, 0, 1, 0], [1, 1, B, U]), ttnn.slice(v8, [0, 0, 0, 0], [1, 1, B - 1, U]))
        run = ttnn.slice(m, [0, 0, 0, 0], [1, 1, 1, U])
        k = ttnn.clone(run)
        for i in range(1, B - 1):  # leading matches: the running product of the row matches
            run = ttnn.multiply(run, ttnn.slice(m, [0, 0, i, 0], [1, 1, i + 1, U]), output_tensor=run)
            k = ttnn.add(k, run, output_tensor=k)
        run.deallocate(True)
        m.deallocate(True)
        last = ttnn.sum(ttnn.multiply(ttnn.eq(self.iota8, k), v8), dim=2, keepdim=True)  # the target's token at row k
        bound = ttnn.slice(self.in_i, [0, 2 * self.rows], [1, 2 * self.rows + U])  # start + B - 1 (int32)
        start = ttnn.typecast(ttnn.to_layout(ttnn.reshape(bound, [1, 1, 1, U]), ttnn.TILE_LAYOUT), ttnn.float32)
        start = ttnn.minimum(ttnn.add(start, k, output_tensor=start) - float(B - 2), self.limit)  # + k + 1
        steps = ttnn.add(self.iota8, start)  # [i, u]: start_u + i
        rows_i = ttnn.reshape(steps, [1, 1, 1, self.rows])
        rows_u = ttnn.reshape(ttnn.transpose(steps, -2, -1), [1, 1, 1, self.rows])
        vi = ttnn.concat([rows_i, rows_u, start + float(B - 1)], dim=-1)
        vu = ttnn.concat([last, self.mask_rest, rows_i], dim=-1)
        for src, dst, dtype in ((vi, self.in_i, ttnn.int32), (vu, self.in_u, ttnn.uint32)):
            t = ttnn.reshape(ttnn.to_layout(ttnn.typecast(src, dtype), ttnn.ROW_MAJOR_LAYOUT), list(dst.shape))
            ttnn.copy(t, dst)
            t.deallocate(True)

    def capture_traces(self):
        """One trace for the whole round (draft, verify, context and the acceptance: no host round trip between draft
        and verify; every lane's round, one after the other), replayed by generate."""
        mesh = self.mesh
        self.release_traces()
        for lane in self.lanes:  # (compile)
            self._use(lane)
            for t in self._round_pass():
                t.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for lane in self.lanes:
            self._use(lane)
            lane.round_out = self._round_pass()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        self._use(self.lanes[0])
        self.round_out = self.lanes[0].round_out
        self.traces = (tid,)
        hold_prefill_l1(self.model)  # (the passes' first run allocated more resident L1: all-reduce scratch, outputs)

    def release_traces(self):
        if self.traces is not None:
            ttnn.synchronize_device(self.mesh)
            for t in self.traces:
                ttnn.release_trace(self.mesh, t)
        self.traces = None

    def _set_inputs(self, starts, lasts):
        """starts / lasts: per user (all lanes' users, or one lane's)."""
        mesh, U, B = self.mesh, self.users, self.block
        lanes = self.lanes if len(starts) == self.total_users else [self.lane]
        for l, lane in enumerate(lanes):
            toks = torch.full((1, self.rows), self.draft.mask_token_id, dtype=torch.int32)
            toks[0, :U] = torch.tensor(lasts[l * U : (l + 1) * U], dtype=torch.int32)
            st = torch.tensor(starts[l * U : (l + 1) * U], dtype=torch.int32)
            steps = st[None] + torch.arange(B)[:, None]  # [step i, user u]: start_u + i
            vi = torch.cat([steps.reshape(-1), steps.T.reshape(-1), st + B - 1])
            vu = torch.cat([toks.reshape(-1), steps.reshape(-1)])
            _copy(mesh, lane.in_i, vi.reshape(1, -1), ttnn.int32)
            _copy(mesh, lane.in_u, vu.reshape(1, -1), ttnn.uint32)

    def _accept(self, ids, accept, done, dr, post, u0=0):
        """Each user's drafts up to the first mismatch with the target's tokens, plus the target's next token (the
        lane of users u0 .. u0 + 31)."""
        U, B = self.users, self.block
        for u in range(U):
            d = [dr[i * U + u] for i in range(1, B)]
            p = [post[i * U + u] for i in range(B)]
            k = 0
            while k < B - 1 and d[k] == p[k]:
                k += 1
            if not done(u0 + u):
                ids[u0 + u] += d[:k] + [p[k]]
                accept[u0 + u].append(k + 1)

    def _enqueue_round(self):
        """Queue one traced round and non-blocking reads of its token ids. Returns (host tensors, event)."""
        ttnn.execute_trace(self.mesh, self.traces[0], cq_id=0, blocking=False)
        host = [t.cpu(blocking=False) for lane in self.lanes for t in lane.round_out]
        return host, ttnn.record_event(self.mesh, 0)

    def _generate_queued(self, ids, lens, accept, done, times, max_new_tokens):
        """Traced rounds with the acceptance on device (b1's queued rounds): round r + 1 is queued before round r's
        result is read (the device writes its inputs), so the host's read and bookkeeping overlap the next round."""
        U, N = self.users, self.total_users
        for lane in self.lanes:
            us = range(lane.u0, lane.u0 + U)
            limit = torch.tensor([lens[u] + max_new_tokens for u in us], dtype=torch.float32).reshape(1, 1, 1, U)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(limit, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)),
                lane.limit,
            )  # fmt: skip
        t = time.perf_counter()
        pending = [self._enqueue_round()]
        while pending:
            # a round adds at most B tokens: queue the next one unless every user may finish with this one
            if len(pending) == 1 and any(len(ids[u]) - lens[u] + self.block < max_new_tokens for u in range(N)):
                pending.append(self._enqueue_round())
            host, event = pending.pop(0)
            ttnn.event_synchronize(event)
            for l, lane in enumerate(self.lanes):
                dr, post = (ttnn.to_torch(ttnn.get_device_tensors(h)[0]).reshape(-1).long().tolist()
                            for h in host[2 * l : 2 * l + 2])  # fmt: skip
                self._accept(ids, accept, done, dr, post, lane.u0)
            now = time.perf_counter()
            times.append((now - t, now - t, 0.0, 0.0))
            t = now
            if all(done(u) for u in range(N)):
                break
            if not pending:
                pending.append(self._enqueue_round())
        if pending:  # a round queued past the end: let it finish
            ttnn.synchronize_device(self.mesh)

    def generate(self, prompts, max_new_tokens, firsts):
        """Returns (per-user new tokens, per-user accepted counts, per-round seconds). A user's new tokens (the first one
        is the prefill's) are at most max_new_tokens, up to and including its first EOS, as in the batch-1 runner (a
        round appends its whole accepted block, which can pass either limit); the accepted counts keep every round."""
        mesh = self.mesh
        U = len(prompts)
        lens = [len(p) for p in prompts]
        ids = [list(p) + [f] for p, f in zip(prompts, firsts)]
        accept = [[] for _ in range(U)]
        times = []

        def done(u):
            return len(ids[u]) - lens[u] >= max_new_tokens or bool(set(ids[u][lens[u] :]) & self.eos_ids)

        while not all(done(u) for u in range(U)):
            t0 = time.perf_counter()
            starts = [len(ids[u]) - 1 for u in range(U)]
            lasts = [ids[u][-1] for u in range(U)]
            self._set_inputs(starts, lasts)
            t1 = time.perf_counter()
            if self.traces is not None:  # traced rounds, queued
                self._generate_queued(ids, lens, accept, done, times, max_new_tokens)
                break
            assert len(self.lanes) == 1, "more than one lane: traced rounds only"
            # eager: the draft pass, the verify block from its drafts on the host, verify + context
            d_out = self._draft_pass()
            dr = self.draft.argmax.host(d_out)
            t2 = time.perf_counter()
            for t in d_out:
                t.deallocate(True)
            vt = torch.tensor(lasts + dr[U:], dtype=torch.int32).reshape(1, self.rows)
            _copy(mesh, self.v_tokens, vt, ttnn.uint32)
            t3 = time.perf_counter()
            v_out = self._verify_pass()
            post = self.verify.argmax.host(v_out)
            t4 = time.perf_counter()
            for t in v_out:
                t.deallocate(True)
            self._accept(ids, accept, done, dr, post)
            t5 = time.perf_counter()
            times.append((t5 - t0, t2 - t1, t4 - t3, (t1 - t0) + (t3 - t2) + (t5 - t4)))  # round, draft, verify, host
        out = []
        for u in range(U):
            toks = ids[u][lens[u] : lens[u] + max_new_tokens]
            out.append(toks[: next((i + 1 for i, t in enumerate(toks) if t in self.eos_ids), len(toks))])
        return out, accept, times
