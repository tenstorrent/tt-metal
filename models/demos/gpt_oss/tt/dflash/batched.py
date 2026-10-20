# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Greedy DFlash for 32 users at once (batch 32) on the batched gpt-oss-20b decode model (batched_decode.py, dense MoE).

A round, for every user u: the drafter proposes 7 tokens after the user's last token; the target checks the 8-row
block [last token, 7 drafts] of every user in one 256-row verify pass and keeps the drafts up to the first mismatch plus
its own next token. Every pass uses the same row order: row 32 i + u is row i of user u's block, at position
start_u + i, so a 32-row step i holds one row of each user (the batched decode's layout: one K/V cache write per step).

Passes (device tensors throughout; the host writes the per-round inputs and reads two 256-row argmax results):
  draft: the noise block [last token, mask x 7] through the drafter's 8 layers (256-row matmuls; Q / K norms and RoPE
      for all 256 rows at once with per-row cos / sin; attention in 8 steps of 32 rows: K/V of every step written first,
      then each step attends to its user's whole block, not causal), its LM head, a per-device argmax;
  verify: the block [last token, 7 drafts] through the target: norm, QKV, o_proj and the all-reduce over all 256 rows;
      the attention core in 8 steps of 32 rows (step i is the batched decode attention at positions start_u + i, so it
      sees the K/V rows of steps 0 .. i: causal within the block; Q and K rotated by one op, K and V written by one op),
      the MoE on all 256 rows at once (router, every expert's gate / up / down as 256-row matmuls in the dense layout of
      the batched decode, each expert weight read once per pass, the routing weights zeroing unrouted experts), the
      target's LM head, a per-device argmax; the outputs of target layers 1, 6, 11, 16, 21 kept for the drafter;
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


def _replicate(mesh, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=True):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh if device else None, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _copy(mesh, dst, t, dtype):
    """Write host tensor t into the persistent device tensor dst (a trace input)."""
    ttnn.copy_host_to_device_tensor(_replicate(mesh, t, dtype, device=False), dst)


class _TPAllReduce:
    """TP all-reduce of [rows, hidden] partial sums over the mesh row with all_reduce_async: persistent scratch and a
    global semaphore per slot, two slots used alternately (each call is an all-rank synchronization, so no device reuses
    a slot before every device is done with it). ttnn.all_reduce, run back to back with other ops, now and then left a
    wrong tile on one device (verify outputs differed between identical runs); this one is repeatable. Inputs go in as
    BF8 (the batched decode's all-reduce precision) on 90 cores, so the [rows, 4 x hidden] scratch fits L1."""

    CORES, LINKS = 90, 2
    _instances = {}

    def __init__(self, mesh, rows, hidden):
        from ..fused_decode.config import width_sharded_memory_config

        D = mesh.shape[1]
        self.mesh = mesh
        self.memory_config = width_sharded_memory_config(mesh, hidden, self.CORES, rows=rows)
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

    def __call__(self, x, memory_config):
        xs = ttnn.to_memory_config(x, self.memory_config, dtype=ttnn.bfloat8_b)
        scratch, semaphore = self.slots[self.calls % 2]
        self.calls += 1
        out = ttnn.experimental.all_reduce_async(
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
        xs.deallocate(True)
        if memory_config is None:
            return out
        moved = ttnn.to_memory_config(out, memory_config)
        out.deallocate(True)
        return moved


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
        res_cores = [ttnn.CoreCoord(x, y) for y in range(rows // TILE) for x in range(self.GX)]

        def nocs(cores):
            w = []
            for c in cores:
                p = mesh.worker_core_from_logical_core(c)
                w.append((p.x << 8) | p.y)
            w += [0] * (len(w) % 2)
            return [w[i] | (w[i + 1] << 16) for i in range(0, len(w), 2)]

        self.ar_noc, self.res_noc = nocs(ar_cores), nocs(res_cores)
        self.cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(self.GX - 1, rows // TILE - 1))]
        )
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
        fid = getattr(ttnn.MathFidelity, os.environ.get("B32_RESNORM_FID", "HiFi4"))
        ck = ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=True, math_approx_mode=False)
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
        reader_ct = [cb_y, cb_r, cb_g, cb_sc, cb_all, self.nb, self.GX, cb_w, cb_lb, cb_lall, cb_b]
        reader_ct += ttnn.TensorAccessorArgs(g).get_compile_time_args()
        reader_args = [y.buffer_address(), g.buffer_address()]
        io = [y, res, g, res2]
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
                runtime_args=[], common_runtime_args=reader_args + self.ar_noc, defines=defines,
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
            cb(cb_all, fp, 4096, self.GX),
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
                    cb(cb_lall, fp, 4096, self.GX), cb(cb_lb, bf, 2048, 1), cb(cb_lo, bf, 2048, 1)]  # fmt: skip
        sems = [ttnn.SemaphoreDescriptor(i, ttnn.CoreType.WORKER, cores, 0) for i in range(2)]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return [res2] + [t for t in (out, out8, logits) if t is not None]


def _all_reduce(x, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """TP all-reduce over the mesh row of a [1, 1, rows, hidden] partial (any layout); the BF16 sum in memory_config."""
    return _TPAllReduce.get(x)(x, memory_config)


def _mm_config_256(k, n):
    """Program config for the drafter's and the verify's [256, k] x [k, n] projections (None: the auto config).
    Measured on one device: the auto config spreads these narrow outputs over few cores with all 8 row tiles each;
    2D over 10 x 8 cores (one row tile, n / 10 columns each) is 1.4-2.5x faster (drafter QKV 74 -> 37 us, gate / up
    86 -> 49, down 73 -> 51, o_proj 45 -> 33; verify QKV BF8 68 -> 27); the drafter's fc (k = 14400) is fastest 1D
    over 90 cores (487 -> 293 us)."""
    kt, nt = k // TILE, n // TILE
    if k == 14400 and nt == 90:
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
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, 8), in0_block_w=bw, out_subblock_h=1,
        out_subblock_w=max(d for d in range(1, 5) if pn % d == 0), out_block_h=1, out_block_w=pn, per_core_M=1,
        per_core_N=pn, transpose_mcast=False, fused_activation=None,
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

    def __init__(self, per_dev, vocab, custom=None):
        self.per_dev, self.vocab = per_dev, vocab
        # custom: the per-row max / index kernels (fused_decode/kernels/b32_argmax_*: one pass over the logits, BFP8 or
        # BF16 read as stored) instead of ttnn.topk k = 1 (its prep + large-indices passes: ~1.0 ms at 256 rows)
        self.custom = os.environ.get("B32_ARGMAX_TOPK", "0") != "1" if custom is None else custom

    def device(self, logits, out_rows=None):
        """out_rows (custom only): the outputs have out_rows rows, the logits being their last rows (the first ones
        left unwritten)."""
        if not self.custom:
            assert out_rows is None
            return ttnn.topk(logits, k=1, dim=-1, sorted=True)
        return self._device_custom(logits, out_rows)

    def _device_custom(self, logits, out_rows=None):
        """(values, indices) [1, 1, rows, 1] FLOAT32: each row's maximum over this device's columns and its first
        column index."""
        from ..fused_decode.boundary import KERNEL_DIR

        mesh = logits.device()
        rows, width = logits.shape[-2], logits.shape[-1]
        RT, NT, NCH = rows // TILE, width // TILE, self.NCH
        out_rows = out_rows or rows
        r0 = out_rows // TILE - RT  # the output row tile of logits row tile 0
        grid = mesh.compute_with_storage_grid_size()
        core_list = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][: RT * NCH]
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in core_list])
        f32 = ttnn.float32
        vals = ttnn.empty([1, 1, out_rows, 1], dtype=f32, layout=ttnn.TILE_LAYOUT, device=mesh,
                          memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        idx = ttnn.empty([1, 1, out_rows, 1], dtype=f32, layout=ttnn.TILE_LAYOUT, device=mesh,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
        bounds = [NT * c // NCH for c in range(NCH + 1)]
        rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        noc = [mesh.worker_core_from_logical_core(c) for c in core_list]
        for i, c in enumerate(core_list):
            ch, rt = i // RT, i % RT
            col0, ncol = bounds[ch], bounds[ch + 1] - bounds[ch]
            rd[c.x][c.y] = [rt, col0, ncol]
            wr[c.x][c.y] = [rt + r0, ch, (noc[rt].x << 8) | noc[rt].y]
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
        return vals, idx

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
        self.core_list = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)][:n]
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in self.core_list])

    def __call__(self, ks, vs, caches, pos_ui, page_table, src_head_stride=0):
        """ks / vs: per layer [1, kv_heads, rows, hd]; caches: per layer (k_cache, v_cache). src_head_stride: ks / vs
        are one tensor each for all layers, layer l's heads from head l src_head_stride."""
        from ..fused_decode.boundary import KERNEL_DIR

        n = len(caches)
        k0, v0 = caches[0]
        if (k0.dtype == ttnn.bfloat8_b and n == 1 and self.hpc == 1 and not src_head_stride
                and os.environ.get("B32_KVW_PACK", "1") == "1"):  # fmt: skip
            return self._call_pack(ks[0], vs[0], k0, v0, pos_ui, page_table)
        ct = [0, self.block, self.kv_heads, self.head_tiles, self.block_size, page_table.shape[-1] * 4, self.users]
        for t in (k0, v0, pos_ui, page_table, ks[0]):
            ct += ttnn.TensorAccessorArgs(t).get_compile_time_args()
        rt = ttnn.RuntimeArgs()
        more = [a for l in range(1, n) for a in (caches[l][0].buffer_address(), caches[l][1].buffer_address(),
                                                  ks[l].buffer_address(), vs[l].buffer_address())]  # fmt: skip
        for i, c in enumerate(self.core_list):
            per = self.kv_heads // self.hpc
            u, h0 = i // per, (i % per) * self.hpc
            arg0 = u | (h0 << 8) | ((h0 + self.hpc) << 16)
            rt[c.x][c.y] = [arg0, ks[0].buffer_address(), vs[0].buffer_address(), k0.buffer_address(),
                            v0.buffer_address(), pos_ui.buffer_address(), page_table.buffer_address()] + more  # fmt: skip
        defines = [("KV_LAYERS", str(n))] if n > 1 else []
        if src_head_stride:
            defines.append(("KV_SRC_HSTRIDE", str(src_head_stride)))
        bf8 = k0.dtype == ttnn.bfloat8_b  # the target's cache: rows packed to BFP8_b in the kernel
        if bf8:
            defines.append(("KV_BF8", "1"))
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "b32_kv_write.cpp"),
                core_ranges=self.cores,
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
            core_ranges=self.cores,
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
        rt = ttnn.RuntimeArgs()
        for i, c in enumerate(self.core_list):
            u, h = i // self.kv_heads, i % self.kv_heads
            rt[c.x][c.y] = [u | (h << 8), k.buffer_address(), v.buffer_address(), k_cache.buffer_address(),
                            v_cache.buffer_address(), pos_ui.buffer_address(), page_table.buffer_address()]  # fmt: skip
            if joint is not None:
                rt[c.x][c.y].append(joint[0].buffer_address())
        dm = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "b32_kv_write_pack.cpp"), core_ranges=self.cores, compile_time_args=ct,
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
            kernel_source=str(KERNEL_DIR / "b32_kv_write_pack_compute.cpp"), core_ranges=self.cores,
            compile_time_args=[cb_old, cb_new, cb_mask, cb_out, self.head_tiles], runtime_args=[],
            config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                fp32_dest_acc_en=os.environ.get("B32_KVW_FP32", "1") == "1",
                                                math_approx_mode=False),
        )  # fmt: skip
        n = 2 * self.head_tiles  # tiles per kv at most (2 groups)

        def cb(i, dtype, page, pages):
            return ttnn.CBDescriptor(total_size=page * pages, core_ranges=self.cores,
                                     format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype,
                                                                                 page_size=page)])  # fmt: skip

        bf8, bf16 = ttnn.bfloat8_b, ttnn.bfloat16
        cbs = [cb(cb_scr, bf16, 512, 1), cb(cb_old, bf8, 1088, n), cb(cb_old + 1, bf8, 1088, n), cb(cb_new, bf16, 2048, n),
               cb(cb_new + 1, bf16, 2048, n), cb(cb_mask, bf16, 2048, 2), cb(cb_out, bf8, 1088, n),
               cb(cb_out + 1, bf8, 1088, n)]  # fmt: skip
        io = [pos_ui, page_table, k, v, k_cache, v_cache] + ([joint[0]] if joint is not None else [])
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=dm + [comp], semaphores=[], cbs=cbs))


_DRAFT_GU = os.environ.get("B32_DRAFT_GU", "1") == "1"


def _gu_joint(gT, uT, D, gx):
    """[K, D I] gate / up weights -> [K, 2 D I], per device d and core column j the gate columns of block j then the up
    ones (I / gx columns each)."""
    I = gT.shape[1] // D
    w = I // gx
    return torch.cat(
        [torch.cat([gT[:, d * I + j * w : d * I + (j + 1) * w], uT[:, d * I + j * w : d * I + (j + 1) * w]], dim=1)
         for d in range(D) for j in range(gx)], dim=1)  # fmt: skip


class BatchedDraft:
    """The z-lab drafter (8 layers, TP like a dense target layer) for `users` sequences, each with its own paged BF16
    K/V cache; draft and context passes over all users' rows at once."""

    def __init__(self, mesh_device, target, max_positions, users, window=None, block_size=64):
        from safetensors.torch import load_file

        self.mesh_device = mesh_device
        self.target = target
        self.path = draft_snapshot()
        c = load_draft_config(self.path)
        D = mesh_device.shape[1]
        self.hidden, self.hd = c["hidden_size"], c["head_dim"]
        self.nq, self.nkv = c["num_attention_heads"] // D, c["num_key_value_heads"] // D
        self.inter = c["intermediate_size"] // D
        self.layers = c["num_hidden_layers"]
        self.eps = c["rms_norm_eps"]
        self.block = c["block_size"]
        self.mask_token_id = c["dflash_config"]["mask_token_id"]
        self.target_layer_ids = c["dflash_config"]["target_layer_ids"]
        self.ctx_in = len(self.target_layer_ids) * self.hidden
        self.window = window
        self.users = users
        self.rows = users * self.block
        w = load_file(str(self.path / "model.safetensors"))
        mesh = mesh_device
        rep = ttnn.ReplicateTensorToMesh(mesh)

        # BFP8 weights (with the HiFi2 matmuls, mm_ck): round 27.3 -> 26.6 ms at ISL 256, 28.1 -> 27.5 at 1024, the
        # acceptance within its run-to-run spread (4.92 / 4.98 -> 4.93 / 5.01 tokens per round). (At round 49.85 ms
        # BF8 alone had been no faster.) B32_DRAFT_WDT=bf16: BF16 weights
        wdt = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}[os.environ.get("B32_DRAFT_WDT", "bf8")]

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
        # unchanged within 0.1 ms, acceptance in its spread; B32_DRAFT_FC_WDT=bf16: BF16)
        fc_dt = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}[os.environ.get("B32_DRAFT_FC_WDT", "bf8")]
        self.fc_full = put(w["fc.weight"].T, dtype=fc_dt)
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

            def kv_only(qp, kp, vp):
                """[q | k | v] per device with only nkv of its q heads (the context pass needs K / V only: a 3.3x
                narrower projection; the q stub keeps nlp_create_qkv_heads' layout)"""
                qh, kh = nq * hd, nkv * hd
                return torch.cat(
                    [torch.cat([qp[d * qh : d * qh + kh], kp[d * kh : (d + 1) * kh], vp[d * kh : (d + 1) * kh]])
                     for d in range(D)]
                )  # fmt: skip

            self.L.append(
                dict(
                    ln1=norm_w(w[p + "input_layernorm.weight"]),
                    ln2=norm_w(w[p + "post_attention_layernorm.weight"]),
                    qkv=put(per_device([q, k, v]).T, cols, dtype=wdt),
                    qkv_b=put(per_device([qb, kb, vb]).reshape(1, -1), cols),
                    kv=put(kv_only(q, k, v).T, cols, dtype=wdt),
                    kv_b=put(kv_only(qb, kb, vb).reshape(1, -1), cols),
                    q_norm=norm_w(reverse_permute_1d(w[a + "q_norm.weight"])),
                    k_norm=norm_w(reverse_permute_1d(w[a + "k_norm.weight"])),
                    o=put(w[a + "o_proj.weight"].T, rows, dtype=wdt),
                    o_b=put(w[a + "o_proj.bias"].reshape(1, -1)),
                    g=put(g.T, cols, dtype=wdt),  # gate and up as two matmuls: no slices of a joint output
                    u=put(u.T, cols, dtype=wdt),
                    # (B32_DRAFT_GU: one joint matmul, its columns interleaved per core column of the 10 x 8 grid: core
                    # column j's 2 x 6 output tiles = gate tiles 6 j .. 6 j + 5, then the up ones of the same columns)
                    gu=put(_gu_joint(g.T, u.T, D, 10), cols, dtype=wdt) if _DRAFT_GU else None,
                    down=put(w[m + "down_proj.weight"].T, rows, dtype=wdt),
                )
            )
        # the context pass's joint K / V projection for every layer: [q stubs (nkv heads per layer) | k | v], each part
        # layer-major (nlp_create_qkv_heads with L nkv heads of each), one matmul + one heads op + one norm + one
        # RoPE for the 8 layers instead of 8 of each; the per-layer k_norm weights as a per-head multiplier
        if os.environ.get("B32_CTX_JOINT", "1") == "1":
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
            joint = torch.cat(
                [torch.cat([parts[j][i][d] for j in range(3) for i in range(self.layers)]) for d in range(D)]
            )
            jb = torch.cat(
                [torch.cat([bparts[j][i][d] for j in range(3) for i in range(self.layers)]) for d in range(D)]
            )
            self.kv_joint = put(joint.T, cols, dtype=wdt)
            self.kv_joint_b = put(jb.reshape(1, -1), cols)
            kn = torch.stack([knw[i] for i in range(self.layers) for _ in range(nkv)]).reshape(
                1, self.layers * nkv, 1, hd
            )
            self.knorm_joint = put(kn, dtype=ttnn.bfloat16)
        else:
            self.kv_joint = None
        self.block_size = block_size
        self.num_blocks = -(-(-(-(max_positions + self.block) // block_size)) // 8) * 8  # (32-byte page-table rows)
        user_tables = torch.arange(users * self.num_blocks, dtype=torch.int32).reshape(users, self.num_blocks)
        self.user_tables = user_tables
        self.page_table = _replicate(mesh, user_tables, ttnn.int32)  # row u: user u (each 32-row step)
        from ..batched_decode import sharded_norm_program_config
        from ..fused_decode.config import width_sharded_memory_config

        self.res_mc = width_sharded_memory_config(mesh, self.hidden, _TPAllReduce.CORES, rows=self.rows)
        # (B32_DRAFT_RES_BLOCK: block sharded on 10 x 8 cores, one 32-row tile x 9 column tiles each, as the verify's
        # residual: the sharded norm reduces over 10 cores per row tile; the all-reduces' outputs resharded to it)
        self.res_block = os.environ.get("B32_DRAFT_RES_BLOCK", "1") == "1"
        if self.res_block:
            self.res_mc = ttnn.create_sharded_memory_config(
                (TILE, self.hidden // 10), ttnn.CoreGrid(x=10, y=self.rows // TILE), ttnn.ShardStrategy.BLOCK,
                ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
            )  # fmt: skip
        self.norm_pc = sharded_norm_program_config(self.res_mc)
        # (B32_DRAFT_RESNORM, with the block residual: each residual add (+ the o_proj bias) and the norm after it in
        # one program (_ResNorm, FP32 residual, FP32 norm outputs as the sharded norm's); the all-reduces' outputs stay in
        # their width shards)
        self.rn = None
        if self.res_block and os.environ.get("B32_DRAFT_RESNORM", "1") == "1":
            self.rn = _ResNorm(mesh, self.rows, self.hidden, self.res_mc,
                               width_sharded_memory_config(mesh, self.hidden, _TPAllReduce.CORES, rows=self.rows))  # fmt: skip
        self.kv_write = BatchedKVWrite(mesh, users, self.block, nkv, hd, block_size)
        # (the context pass's 8-layer write: one core per (user, kv head))
        self.kv_write_ctx = BatchedKVWrite(mesh, users, self.block, nkv, hd, block_size, head_split=True)
        # attention: block attention (b1's kernels, uniform), 16 users per program on 4 cores each
        # (16 users x 1 history slice: the window bounds every core's history to ~33 chunks; vs 8 x 3: round 25.0 -> 24.8
        # ms at ISL 256, 25.8 -> 25.4 at 1024)
        self.ba = _UserBlockAttention(
            mesh, users, self.block, nq, nkv, hd, block_size, 16, 1, uniform=True, window=window or 0
        )
        heads_split = ttnn.ShardTensorToMesh(mesh, dim=1)
        self.caches = []
        for _ in range(self.layers):
            z = torch.zeros(users * self.num_blocks, D * nkv, block_size, hd)
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
        # 5.02 / 5.01 tokens per round (in its spread); B32_CTX_FC_FID=HiFi2: the other matmuls' fidelity
        self.fc_ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("B32_CTX_FC_FID", "LoFi")), math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )  # fmt: skip
        # the pass activations between ops in L1 (B32_DRAFT_L1)
        self.act_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("B32_DRAFT_L1", "1") == "1" else ttnn.DRAM_MEMORY_CONFIG
        # (with BF16 weights HiFi2 / LoFi were no faster; with the BFP8 ones HiFi2 is: 26.8 -> 26.6 ms, LoFi as HiFi2)
        self.mm_ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("B32_DRAFT_MMFID", "HiFi2")), math_approx_mode=False,
            fp32_dest_acc_en=True, packer_l1_acc=True,
        )  # fmt: skip
        # the drafter's LM head on a BFP4 copy of the target's with BF8 rows: 0.89 -> 0.52 ms per pass, live-like
        # acceptance on a fixed text 3.07 (BF16 rows x BF8 weight: 3.05 / 3.10 in two runs); the unpadded copy
        # (_even_lm_head)
        lm8 = _even_lm_head(target) if os.environ.get("B32_LM_EVEN", "1") == "1" else target.lm_head_weight
        self.lm_head = ttnn.typecast(lm8, ttnn.bfloat4_b)
        # BFP8 logits (the argmax reads them as stored; half the 26 MB BF16 write and read): round 23.3 -> 23.2 ms at
        # ISL 256, acceptance in its spread (B32_DRAFT_LOGITS=bf16: BF16)
        self.logits_dt = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}[os.environ.get("B32_DRAFT_LOGITS", "bf8")]
        self.argmax = _Argmax(lm8.shape[-1], target.vocab_size)
        if self.rn is not None:  # (the weights' tiles made here, not during a trace capture)
            for L in self.L:
                self.rn.gamma(L["ln1"])
                self.rn.gamma(L["ln2"])
                self.rn.row_tiles(L["o_b"])
            self.rn.gamma(self.norm)

    # -- pieces ----------------------------------------------------------------------------------------------------
    def _linear(self, x, w, bias=None, ck=None):
        pc = _mm_config_256(w.shape[-2], w.shape[-1]) if x.shape[-2] == self.rows else None
        # (a pass's 256-row activations in L1; the prompt context's chunks in DRAM)
        mc = self.act_mc if x.shape[-2] == self.rows else ttnn.DRAM_MEMORY_CONFIG
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
        """act = silu(x W_gate) * (x W_up) [rows, I] BF16 (L1 interleaved): with B32_DRAFT_GU one joint matmul into
        10 x 8 block shards and one silu * up program on them (fused_decode/kernels/draft_silu_mul_*.cpp); else two
        matmuls and a multiply."""
        if L.get("gu") is None:
            g, u = self._linear(x, L["g"]), self._linear(x, L["u"])
            act = ttnn.multiply(g, u, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=self.act_mc)
            g.deallocate(True)
            u.deallocate(True)
            return act
        from ..fused_decode.boundary import KERNEL_DIR

        K, N2 = L["gu"].shape[-2], L["gu"].shape[-1]
        gx, gy = 10, self.rows // TILE
        nh = N2 // 2 // TILE // gx
        gu_mc = ttnn.create_sharded_memory_config((TILE, N2 // gx), ttnn.CoreGrid(x=gx, y=gy), ttnn.ShardStrategy.BLOCK,
                                                  ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)  # fmt: skip
        gu = ttnn.linear(x, L["gu"], program_config=_mm_config_256(K, N2), dtype=ttnn.bfloat16,
                         compute_kernel_config=self.mm_ck, memory_config=gu_mc)  # fmt: skip
        act = ttnn.empty([1, 1, self.rows, N2 // 2], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh_device,
                         memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        cb_gu, cb_act = 0, 1
        dm = ttnn.DataMovementConfigDescriptor
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "draft_silu_mul_reader.cpp"), core_ranges=cores,
                compile_time_args=[cb_gu, nh], runtime_args=[],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "draft_silu_mul_writer.cpp"), core_ranges=cores,
                compile_time_args=[cb_act, nh, N2 // 2 // TILE] + ttnn.TensorAccessorArgs(act).get_compile_time_args(),
                runtime_args=[], common_runtime_args=[act.buffer_address()],
                config=dm(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "draft_silu_mul_compute.cpp"), core_ranges=cores,
                compile_time_args=[cb_gu, cb_act, nh], runtime_args=[],
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

    def _heads_qk(self, qkv, L, cos, sin, with_q, nq=None):
        """[1, 1, rows, qkv] projection -> (Q or None, K, V) heads in the prefill layout [1, heads, rows, hd] (L1),
        K (and Q) head-normed and rotated, every row at its own position. nq: the projection's q heads (the context
        pass's K / V projection carries an nkv-head q stub)."""
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=nq or self.nq, num_kv_heads=self.nkv, transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )  # fmt: skip
        trans = self.rope.get_both_trans_mats()["prefill"]

        def norm_rope(t, w):
            n = ttnn.rms_norm(t, epsilon=self.eps, weight=w, compute_kernel_config=self.ck)
            t.deallocate(True)
            out = ttnn.experimental.rotary_embedding_llama(n, cos, sin, trans, is_decode_mode=False,
                                                           memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
            n.deallocate(True)
            return out

        k = norm_rope(k, L["k_norm"])
        if with_q:
            q = norm_rope(q, L["q_norm"])
        else:
            q.deallocate(True)
            q = None
        return q, k, v

    # -- prompt context ---------------------------------------------------------------------------------------------
    def add_prompt_context(self, features, user, chunk=2048):
        """The prompt's context K / V (prefill-style, paged_fill_cache of block-aligned chunks)."""
        n = features.shape[0]
        mesh = self.mesh_device
        pt_host = self.user_tables[user][None]
        qw, kvw = self.nq * self.hd, self.nkv * self.hd
        for s0 in range(0, n, chunk):
            rows = min(chunk, -(-(n - s0) // self.block_size) * self.block_size)
            x = torch.zeros(1, 1, rows, self.ctx_in)
            x[0, 0, : min(rows, n - s0)] = features[s0 : s0 + rows].float()
            tx = ttnn.from_torch(
                x,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            ctx = self._linear(tx, self.fc_full)
            tx.deallocate(True)
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
        if os.environ.get("B32_CTX_NORM_SHARDED", "1") == "1":
            # the hidden norm on the residual's 90 width shards (interleaved: 8 cores, 44 us)
            cs = ttnn.to_memory_config(ctx, self.res_mc)
            ctx.deallocate(True)
            n = ttnn.rms_norm(
                cs, epsilon=self.eps, weight=self.hidden_norm, program_config=self.norm_pc, memory_config=self.res_mc,
                compute_kernel_config=self.ck,
            )  # fmt: skip
            cs.deallocate(True)
            ctx2 = ttnn.to_memory_config(n, self.act_mc)
            n.deallocate(True)
        else:
            ctx2 = ttnn.rms_norm(ctx, epsilon=self.eps, weight=self.hidden_norm, compute_kernel_config=self.ck)
            ctx.deallocate(True)
        ks, vs = [], []  # every layer's K / V, then one write program for the 8 layers
        if self.kv_joint is not None:
            self._context_joint(ctx2, pos, rot)
            return
        kv_only = os.environ.get("B32_CTX_KV_ONLY", "1") == "1"
        for li in range(self.layers):
            L = self.L[li]
            if kv_only:  # K / V (and an nkv-head q stub) only
                qkv = self._linear(ctx2, L["kv"], L["kv_b"])
                _, k, v = self._heads_qk(qkv, L, *rot, with_q=False, nq=self.nkv)
            else:
                qkv = self._linear(ctx2, L["qkv"], L["qkv_b"])
                _, k, v = self._heads_qk(qkv, L, *rot, with_q=False)
            qkv.deallocate(True)
            ks.append(k)
            vs.append(v)
        ctx2.deallocate(True)
        self.kv_write(ks, vs, self.caches, pos[self.block], self.page_table)
        for t in ks + vs:
            t.deallocate(True)

    def _context_joint(self, ctx2, pos, rot):
        """Every layer's K / V of the context rows in one projection (kv_joint), heads op, norm, RoPE and cache write."""
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

    def _draft_resnorm(self, h, x, pos, rot):
        """draft()'s layers and LM head with _ResNorm: h the FP32 residual, x layer 0's input norm."""
        rn, f32 = self.rn, ttnn.float32
        for li in range(self.layers):
            L = self.L[li]
            qkv = self._linear(x, L["qkv"], L["qkv_b"])
            x.deallocate(True)
            q_rows, k, v = self._heads_qk(qkv, L, *rot, with_q=True)
            qkv.deallocate(True)
            self.kv_write([k], [v], [self.caches[li]], pos[self.block], self.page_table)
            k.deallocate(True)
            v.deallocate(True)
            kc, vc = self.caches[li]
            heads = self.ba(q_rows, kc, vc, pos[self.block], self.page_table, None, self.hd**-0.5, heads_io=True)
            q_rows.deallocate(True)
            partial = self._linear(heads, L["o"])
            heads.deallocate(True)
            o = _all_reduce(partial, None)
            partial.deallocate(True)
            h2, x = rn(o, h, L["ln2"], eps=self.eps, bias=L["o_b"], out_dtype=f32)
            o.deallocate(True)
            h.deallocate(True)
            act = self._gate_up(x, L)
            x.deallocate(True)
            partial = self._linear(act, L["down"])
            act.deallocate(True)
            mlp = _all_reduce(partial, None)
            partial.deallocate(True)
            last = li + 1 == self.layers
            if last:  # the final norm straight to BFP8 (the LM head's input)
                h, x8 = rn(mlp, h2, self.norm, out_bf16=False, x8=True, eps=self.eps)
            else:
                h, x = rn(mlp, h2, self.L[li + 1]["ln1"], eps=self.eps, out_dtype=f32)
            mlp.deallocate(True)
            h2.deallocate(True)
        h.deallocate(True)
        skip = self.argmax.custom and os.environ.get("B32_DRAFT_SKIP_ROW0", "1") == "1"
        if skip:  # rows 0 .. 31 (each user's last token, i = 0) draft nothing: the LM head on rows 32 .. 255 only
            x8r = ttnn.slice(x8, [0, 0, U32, 0], [1, 1, self.rows, self.hidden])
            x8.deallocate(True)
            x8 = x8r
        logits = ttnn.matmul(x8, self.lm_head, dtype=self.logits_dt)
        x8.deallocate(True)
        out = self.argmax.device(logits, self.rows if skip else None)
        logits.deallocate(True)
        return out

    def draft(self, tokens, pos, rot, bound):
        """tokens: device [1, rows] UINT32 noise block (row 32 i + u: user u's last token for i = 0, else the mask).
        Returns the per-device argmax of every row (rows 32 i + u, i >= 1: the draft of position start_u + i)."""
        _TPAllReduce.reset()
        emb = ttnn.embedding(tokens, self.target.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        # the FP32 residual stays width-sharded on the all-reduce's 90 cores: the residual adds and the norms run on
        # the shards (no DRAM round trips; the DRAM norm ran on 8 cores)
        h = ttnn.to_memory_config(ttnn.reshape(emb, [1, 1, self.rows, self.hidden]), self.res_mc, dtype=ttnn.float32)

        def norm(h, w):
            n = ttnn.rms_norm(
                h, epsilon=self.eps, weight=w, program_config=self.norm_pc, memory_config=self.res_mc,
                compute_kernel_config=self.ck,
            )  # fmt: skip
            x = ttnn.to_memory_config(n, self.act_mc)
            n.deallocate(True)
            return x

        if self.rn is not None:
            return self._draft_resnorm(h, norm(h, self.L[0]["ln1"]), pos, rot)
        for li in range(self.layers):
            L = self.L[li]
            x = norm(h, L["ln1"])
            qkv = self._linear(x, L["qkv"], L["qkv_b"])
            x.deallocate(True)
            q_rows, k, v = self._heads_qk(qkv, L, *rot, with_q=True)
            qkv.deallocate(True)
            self.kv_write([k], [v], [self.caches[li]], pos[self.block], self.page_table)  # every row's K/V first
            k.deallocate(True)
            v.deallocate(True)
            kc, vc = self.caches[li]
            # block attention (b1's kernels, uniform): a user's 8 rows see the same positions (start .. start + 7, not
            # causal), every user's K/V read once per layer; Q read in the heads layout, the output written in
            # o_proj's (no transposes, no concat-heads op)
            heads = self.ba(q_rows, kc, vc, pos[self.block], self.page_table, None, self.hd**-0.5, heads_io=True)
            q_rows.deallocate(True)
            partial = self._linear(heads, L["o"])
            heads.deallocate(True)
            o = _all_reduce(partial, self.res_mc if self.res_block else None)  # (the residual's shards)
            partial.deallocate(True)
            o = ttnn.add(o, L["o_b"], output_tensor=o)
            h2 = ttnn.add(h, o, memory_config=self.res_mc, dtype=ttnn.float32)
            h.deallocate(True)
            o.deallocate(True)
            h = h2
            x = norm(h, L["ln2"])
            act = self._gate_up(x, L)
            x.deallocate(True)
            partial = self._linear(act, L["down"])
            act.deallocate(True)
            mlp = _all_reduce(partial, self.res_mc if self.res_block else None)
            partial.deallocate(True)
            h2 = ttnn.add(h, mlp, memory_config=self.res_mc, dtype=ttnn.float32)
            h.deallocate(True)
            mlp.deallocate(True)
            h = h2
        x = norm(h, self.norm)
        h.deallocate(True)
        x8 = ttnn.typecast(x, ttnn.bfloat8_b)
        x.deallocate(True)
        skip = self.argmax.custom and os.environ.get("B32_DRAFT_SKIP_ROW0", "1") == "1"
        if skip:  # rows 0 .. 31 (each user's last token, i = 0) draft nothing: the LM head on rows 32 .. 255 only
            x8r = ttnn.slice(x8, [0, 0, U32, 0], [1, 1, self.rows, self.hidden])
            x8.deallocate(True)
            x8 = x8r
        logits = ttnn.matmul(x8, self.lm_head, dtype=self.logits_dt)
        x8.deallocate(True)
        out = self.argmax.device(logits, self.rows if skip else None)
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
        # sliding-window program 57 us); the last program may hold fewer users
        self.programs = [(p0, min(per_program, users - p0)) for p0 in range(0, users, per_program)]
        self.inst = {
            n: BlockAttention(
                mesh,
                block,
                n_heads,
                n_kv,
                head_dim,
                block_size,
                cores[: per * n],
                slices,
                q_rows=True,
                min_chunks=min_chunks,
                uniform=uniform,
                window=window,
                swin=swin,
                users=n,
            )
            for n in sorted({n for _, n in self.programs})
        }

    def __call__(self, q, k_cache, v_cache, pos_ui, page_table, sinks, scale, heads_io=False, q_heads=None):
        """q: [1, rows, heads, hd] (row 32 i + u, rotated); pos_ui: INT32 [rows] (8 u + i: start_u + i); page_table:
        INT32 [users, blocks]. Returns the attention output in q's layout. heads_io: q in the heads layout [1, heads,
        rows, hd] (the RoPE output) and the output [1, 1, rows, heads * hd] (o_proj's input) in L1."""
        if heads_io:  # (q_heads: q's first heads are the query heads, e.g. Q and K in one RoPE output)
            shape, mc = [1, 1, q.shape[2], (q_heads or q.shape[1]) * q.shape[3]], ttnn.L1_MEMORY_CONFIG
        else:
            shape, mc = list(q.shape), ttnn.DRAM_MEMORY_CONFIG
        out = ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.mesh, memory_config=mc)
        for p0, n in self.programs:
            inst = self.inst[n]
            kernels, cbs, tio = inst.parts(
                q, k_cache, v_cache, pos_ui, page_table, sinks, scale, wait=(self.WAIT_SEM, 0),
                user=(p0, self.users), out=out, heads_io=heads_io,
            )  # fmt: skip
            sems = inst.semaphores() + [ttnn.SemaphoreDescriptor(self.WAIT_SEM, ttnn.CoreType.WORKER, inst.cores, 0)]
            io = [t for t in tio if t is not out]
            ttnn.generic_op(io + [out], ttnn.ProgramDescriptor(kernels=kernels, semaphores=sems, cbs=cbs))
        return out


class BatchedVerify:
    """The target's 256-row verify pass for 32 users on the batched decode model's layers (DecoderLayer.batched_decode):
    attention in 8 batched-decode steps, the MoE on all rows at once, norms on 32-row residual shards as the decode."""

    def __init__(self, model, page_table, kv_cache, users, block, capture_layers):
        self.model = model
        self.mesh = model.mesh_device
        self.users, self.block = users, block
        self.rows = users * block
        self.page_table = page_table
        self.kv_cache = kv_cache
        self.capture_layers = list(capture_layers)
        self.hidden = model.hf_config.hidden_size
        cfg = model.hf_config
        self.top_k, self.num_experts = cfg.num_experts_per_tok, cfg.num_local_experts
        self.swiglu_limit = cfg.swiglu_limit
        self.alpha = 1.702  # ExpertConfig.alpha
        self.moe = []
        for layer in model.layers:
            assert layer.batched_decode, "BatchedVerify runs on the batched decode model (batch 32, dense MoE)"
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
                    expand=dw.expand,
                )
            )
        self.ck = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        # down [rows, E * I_local] x [E * I_local, hidden]: a long K and a narrow N, which the auto-configured matmul
        # spreads poorly (0.75 ms per layer); 2D over 10 x 8 cores, 16-tile K blocks: 0.27 ms
        n_tiles, m_tiles = self.hidden // TILE, self.rows // TILE
        self.down_pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(10, m_tiles),
            in0_block_w=16,
            out_subblock_h=1,
            out_subblock_w=3,
            out_block_h=1,
            out_block_w=n_tiles // 10,
            per_core_M=1,
            per_core_N=n_tiles // 10,
            transpose_mcast=False,
            fused_activation=None,
        )
        self.down_ck = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,  # FP32 sums over the 23040-long K (4.5x smaller error, same time)
            packer_l1_acc=True,
        )
        # gate / up [rows, hidden] x [hidden, E * I_local] and the routing-weight expand [rows, E] x [E, E * I_local]:
        # 1D over 72 cores (all 8 row tiles x 10 output tiles each; gate / up in 2-tile K blocks, the auto config's
        # sums), outputs width-sharded in L1 ([rows, 320] per core), SwiGLU and the routing multiply there: MoE
        # 1.16 -> 1.00 ms per layer, identical output
        n_act = self.moe[0]["gate"].shape[-1]
        moe_cores = ttnn.CoreCoord(9, 8)
        per_n = n_act // TILE // (moe_cores.x * moe_cores.y)

        def moe_pc(in0_block_w):
            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=moe_cores,
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=5,
                out_block_h=m_tiles,
                out_block_w=per_n,
                per_core_M=m_tiles,
                per_core_N=per_n,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )

        self.gate_pc, self.expand_pc = moe_pc(2), moe_pc(1)
        self.moe_mc = ttnn.create_sharded_memory_config(
            (self.rows, per_n * TILE),
            ttnn.CoreGrid(y=moe_cores.y, x=moe_cores.x),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # router: the whole K in one block per output tile (TopKRouter.dense_decode's accumulation order: the same
        # logits), one core per 32-row tile
        k_tiles = self.hidden // TILE
        self.router_pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.rows // TILE, 1),
            in0_block_w=k_tiles,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=-(-self.num_experts // TILE),
            per_core_M=1,
            per_core_N=-(-self.num_experts // TILE),
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=False,
        )
        self.router_ck = ttnn.init_device_compute_kernel_config(
            self.mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.lm_head = _even_lm_head(model) if os.environ.get("B32_LM_EVEN", "1") == "1" else model.lm_head_weight
        self.argmax = _Argmax(self.lm_head.shape[-1], model.vocab_size)
        # grouped MoE (dflash/grouped_moe.py): only the routed (row, expert) pairs; None: the dense MoE
        self.grouped = None
        self.route_fused = os.environ.get("B32_GMOE_ROUTE", "1") == "1"
        from ..fused_decode.config import RESIDUAL_CORES, sdpa_decode_program_config, width_sharded_memory_config

        attn0 = model.layers[0].self_attn
        mc = model.layers[0].self_attn.mesh_config
        self.local_heads = mc.shard_size(attn0.config.num_heads)
        self.local_kv_heads = mc.shard_size(attn0.config.num_kv_heads)
        hd = self.head_dim = attn0.config.head_dim
        self.residual_mc = width_sharded_memory_config(self.mesh, self.hidden, RESIDUAL_CORES, rows=self.rows)
        if os.environ.get("B32_RES_BLOCK", "1") == "1":
            # block sharded on 10 x 8 cores, one 32-row tile x 9 column tiles each: the sharded norm reduces over 10
            # cores per row tile instead of 30 per 8 (standalone 14.6 -> 6.4 us per norm, 49 per pass)
            self.residual_mc = ttnn.create_sharded_memory_config(
                (TILE, self.hidden // 10), ttnn.CoreGrid(x=10, y=self.rows // TILE), ttnn.ShardStrategy.BLOCK,
                ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True,
            )  # fmt: skip
        # each residual add and the norm after it in one program (_ResNorm, B32_RESNORM; needs the block residual)
        self.resnorm = None
        if os.environ.get("B32_RESNORM", "1") == "1" and os.environ.get("B32_RES_BLOCK", "1") == "1":
            self.resnorm = _ResNorm(self.mesh, self.rows, self.hidden, self.residual_mc,
                                    width_sharded_memory_config(self.mesh, self.hidden, _TPAllReduce.CORES, rows=self.rows))  # fmt: skip
            for l in model.layers:  # (the weights' tiles made here, not during a trace capture)
                self.resnorm.gamma(l.input_layernorm)
                self.resnorm.gamma(l.post_attention_layernorm)
                self.resnorm.router_bias(l.mlp.router)
            self.resnorm.gamma(model.norm)
        # (B32_RESNORM_ROUTER: the post-attention _ResNorm also computes the router's logits (split K over its 10 cores of
        # a row, FP32 sums) and the BFP8 copy for the grouped MoE's gather: no router matmul, no typecast)
        self.resnorm_router = (self.resnorm is not None and self.route_fused
                               and os.environ.get("B32_RESNORM_ROUTER", "1") == "1")  # fmt: skip
        # sliding-window layers: one decode SDPA per 32-row step (the decode's 8 x 8 config, Q read from DRAM)
        self.sliding_pc = sdpa_decode_program_config(True)
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
        # (16 x 1 after the one-kernel-set change: round -0.4 ms at ISL 256 .. 4K, +2.4 ms at 16K)
        self.block_attn = _UserBlockAttention(
            self.mesh, users, block, self.local_heads, self.local_kv_heads, hd, kv_cache[1][0].shape[2],
            int(os.environ.get("B32_BA_P", "8")), int(os.environ.get("B32_BA_S", "3")),
        )  # fmt: skip
        # sliding-window layers: the same block attention, windowed (row r attends [pos_r + 1 - window, pos_r]: at most
        # 6 chunks), 16 users per program on 1 slice each (64 cores), instead of one decode SDPA per 32-row step (8
        # SDPA calls + 8 Q slices + the concat per layer). B32_SWIN=0: the per-step decode SDPA.
        windows = {l.self_attn.config.sliding_window for l in model.layers if l.self_attn.config.sliding_window}
        self.swin_attn = None
        if os.environ.get("B32_SWIN", "1") == "1" and windows:
            (window,) = windows
            self.swin_attn = _UserBlockAttention(
                self.mesh, users, block, self.local_heads, self.local_kv_heads, hd, kv_cache[0][0].shape[2],
                int(os.environ.get("B32_SWIN_P", "16")), int(os.environ.get("B32_SWIN_S", "1")), swin=window,
            )  # fmt: skip
        # heads, RoPE and the K / V cache writes once per layer for all 256 rows (prefill-layout heads, per-row cos / sin,
        # the multi-user BFP8_b write kernel) instead of per 32-row step (interleaved_to_sharded_partial, decode heads,
        # fused-QK RoPE, paged_fused_update_cache: 4 x 8 ops per layer): round 42.2 -> 41.0 ms at ISL 256. The kernel
        # packs the exact BF16 rows (round half up, as the device packer); the per-step path stored ~10% of the values
        # one BFP8 step larger in magnitude (tests/_scratch_b32_verify.py test_kvwrite). trans_rows (the prefill RoPE
        # transformation matrix of the target's RoPE) is set by the runner.
        self.trans_rows = None
        # the MoE input (router, typecast for the gather) in L1 (B32_MOE_IN_L1)
        self.moe_in_mc = (
            ttnn.L1_MEMORY_CONFIG if os.environ.get("B32_MOE_IN_L1", "1") == "1" else ttnn.DRAM_MEMORY_CONFIG
        )
        self.attn_mc = ttnn.L1_MEMORY_CONFIG if os.environ.get("B32_ATTN_L1", "1") == "1" else ttnn.DRAM_MEMORY_CONFIG
        # Q and K rotated by one RoPE op (B32_QK_JOINT)
        self.qk_joint = os.environ.get("B32_QK_JOINT", "1") == "1" and os.environ.get("B32_KVW_PACK", "1") == "1"
        # the drafter's context features in BFP8 (the context fc multicasts them to its 90 cores: 275 -> 218 us;
        # acceptance in its spread; B32_FEAT_DT=bf16: BF16)
        self.feat_dt = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b}[os.environ.get("B32_FEAT_DT", "bf8")]
        self.kv_write = BatchedKVWrite(self.mesh, users, block, self.local_kv_heads, hd, kv_cache[1][0].shape[2],
                                       head_split=True)  # fmt: skip

    def _router(self, layer, x, with_ids=False):
        """Dense [rows, experts] routing weights: each row's softmax over its top-k logits, zeros elsewhere (as
        TopKRouter.dense_decode, for any number of rows); with_ids: also the top-k expert ids (UINT16)."""
        r = layer.mlp.router
        logits = ttnn.linear(
            x,
            r.weight,
            bias=r.bias,
            program_config=self.router_pc,
            compute_kernel_config=self.router_ck,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        # (with_ids: top-(k + 1) ids, for the grouped MoE, which routes the (k + 1)-th expert on a BF16 tie as the dense
        # weights below do)
        values, indices = ttnn.topk(logits, k=self.top_k + int(with_ids), dim=-1, sorted=True)
        if not with_ids:
            indices.deallocate(True)
        if with_ids:  # the k-th largest logit
            v4 = ttnn.slice(values, [0, 0, 0, 0], [1, 1, values.shape[-2], self.top_k])
            values.deallocate(True)
            values = v4
        kth = ttnn.min(values, dim=-1, keepdim=True)
        values.deallocate(True)
        top = ttnn.max(logits, dim=-1, keepdim=True)
        selected = ttnn.ge(logits, kth)
        kth.deallocate(True)
        e = ttnn.exp(ttnn.subtract(logits, top, output_tensor=logits), output_tensor=logits)
        top.deallocate(True)
        e = ttnn.multiply(e, selected, output_tensor=e)
        selected.deallocate(True)
        total = ttnn.sum(e, dim=-1, keepdim=True)
        w = ttnn.divide(e, total)
        total.deallocate(True)
        e.deallocate(True)
        return (w, indices) if with_ids else w

    def _moe(self, li, layer, x, memory_config=ttnn.DRAM_MEMORY_CONFIG):
        """The MoE (this device's TP share, _moe_dense or _moe_grouped) and the TP all-reduce (into memory_config). x:
        [1, 1, rows, hidden] normed (DRAM)."""
        down = (self._moe_grouped if self.grouped is not None else self._moe_dense)(li, layer, x)
        out = _all_reduce(down, memory_config)
        down.deallocate(True)
        return out

    def _moe_grouped(self, li, layer, x):
        """Only the routed (row, expert) pairs (dflash/grouped_moe.py), plus the routing-weighted down bias."""
        W = self.moe[li]
        if self.route_fused:  # the router matmul, then routing + plan in one program (GroupedMoE.route)
            r = layer.mlp.router
            logits = ttnn.linear(x, r.weight, bias=r.bias, program_config=self.router_pc,
                                 compute_kernel_config=self.router_ck, memory_config=ttnn.L1_MEMORY_CONFIG,
                                 dtype=ttnn.bfloat16)  # fmt: skip
            w = self.grouped.route(logits, li)
            logits.deallocate(True)
            down = self.grouped(li, x)
            if self.grouped.bias_slot:  # (the combine added the down bias)
                w.deallocate(True)
                return down
        else:
            w, ids = self._router(layer, x, with_ids=True)
            down = self.grouped(li, x, ids, w)
            ids.deallocate(True)
        bias = ttnn.matmul(w, W["down_bias"], dtype=ttnn.bfloat16)
        w.deallocate(True)
        down = ttnn.add(down, bias, output_tensor=down)
        bias.deallocate(True)
        return down

    def _moe_dense(self, li, layer, x):
        """All experts for all rows ([rows, E * I_local] matmuls), unrouted experts zeroed by the routing weights."""
        UWP, U = ttnn.UnaryWithParam, ttnn.UnaryOpType
        # gate / up / expand / SwiGLU in L1 interleaved (the elementwise ops run faster there than on the matmuls'
        # width shards: SwiGLU 228 -> 160 us; BF8 gate / up outputs broke the verify: agreement 0.31)
        mc = ttnn.L1_MEMORY_CONFIG
        gu_dt = ttnn.bfloat16
        W = self.moe[li]
        w = self._router(layer, x)
        x8 = ttnn.typecast(x, ttnn.bfloat8_b)
        # the biases inside the matmuls (with the FP32 down below: the fixed-text agreement of the separate adds)
        gate, up = (
            ttnn.linear(
                x8,
                W[n],
                bias=W[n + "_bias"],
                program_config=self.gate_pc,
                memory_config=mc,
                dtype=gu_dt,
                compute_kernel_config=self.ck,
            )
            for n in ("gate", "up")
        )
        x8.deallocate(True)
        w_cols = ttnn.matmul(w, W["expand"], program_config=self.expand_pc, memory_config=mc, dtype=ttnn.bfloat16)
        limit, alpha = self.swiglu_limit, self.alpha
        act = ttnn.multiply(
            up,
            gate,
            input_tensor_a_activations=[UWP(U.CLAMP_TSS, -limit, limit), UWP(U.ADD_UNARY_SFPU, 1.0)],
            input_tensor_b_activations=[
                UWP(U.CLAMP_TSS, -3.0e38, limit),
                UWP(U.MUL_UNARY_SFPU, alpha),
                UWP(U.SILU),
                UWP(U.MUL_UNARY_SFPU, 1.0 / alpha),
            ],
            memory_config=mc,
        )
        gate.deallocate(True)
        up.deallocate(True)
        # BF8 into the 2D down (DRAM): half the activation reads; the down's sums stay FP32
        act8 = ttnn.multiply(act, w_cols, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        act.deallocate(True)
        w_cols.deallocate(True)
        down = ttnn.matmul(
            act8, W["down"], program_config=self.down_pc, dtype=ttnn.bfloat16, compute_kernel_config=self.down_ck
        )
        act8.deallocate(True)
        bias = ttnn.matmul(w, W["down_bias"], dtype=ttnn.bfloat16)
        w.deallocate(True)
        down = ttnn.add(down, bias, output_tensor=down)
        bias.deallocate(True)
        return down

    def _attention(self, li, layer, res, pos, rot, x=None):
        """The layer's attention for all rows: input norm, QKV, the attention core (_attention_rows) and o_proj (+
        all-reduce), each once over the 256 rows. res: [rows, hidden] width-sharded residual; rot: every row's cos / sin
        [1, 1, rows, hd]. Returns res + attention (same layout)."""
        attn_mod, model = layer.self_attn, self.model
        w, cfg = attn_mod.weights, attn_mod.config
        nh, nkv, hd = self.local_heads, self.local_kv_heads, cfg.head_dim
        # (the activations between the attention's ops in L1, B32_ATTN_L1)
        amc = self.attn_mc
        if x is None:
            normed = layer.input_layernorm.forward_sharded(res)
            x = ttnn.to_memory_config(normed, amc)
            normed.deallocate(True)
        qkv = ttnn.linear(
            x,
            w.wqkv,
            bias=w.wqkv_bias,
            program_config=_mm_config_256(w.wqkv.shape[-2], w.wqkv.shape[-1]),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.qkv_ck,
            memory_config=amc,
        )
        x.deallocate(True)
        heads = self._attention_rows(li, attn_mod, qkv, pos, rot)
        qkv.deallocate(True)
        op = attn_mod.batched_oproj
        if os.environ.get("B32_OPROJ_BIAS_MM", "1") == "1" and self.resnorm is not None:
            # the bias inside the matmul (BF16 output; the all-reduce converts it to its BFP8 input shards): one op
            # instead of the matmul + the bias add with the BF8 output
            o = ttnn.linear(heads, self.oproj_w[li], bias=op.bias,
                            program_config=_mm_config_256(heads.shape[-1], self.oproj_w[li].shape[-1]),
                            dtype=ttnn.bfloat16, compute_kernel_config=self.qkv_ck, memory_config=amc)  # fmt: skip
            heads.deallocate(True)
            out = _all_reduce(o, None)
            o.deallocate(True)
            return out
        o = ttnn.matmul(
            heads,
            self.oproj_w[li],
            program_config=_mm_config_256(heads.shape[-1], self.oproj_w[li].shape[-1]),
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.qkv_ck,
            memory_config=amc,
        )
        heads.deallocate(True)
        # the decode's BF8 partial sum (the bias inside the matmul with a BF8 output: top-1 vs HF 0.08)
        # (written straight into the all-reduce's input shards: no conversion op before it)
        partial = ttnn.add(o, op.bias, dtype=ttnn.bfloat8_b, memory_config=_TPAllReduce.get(o).memory_config)
        o.deallocate(True)
        if self.resnorm is not None:  # (the all-reduce output as is: _ResNorm adds it to the residual)
            out = _all_reduce(partial, None)
            partial.deallocate(True)
            return out
        summed_s = _all_reduce(partial, self.residual_mc)
        partial.deallocate(True)
        out = ttnn.add(res, summed_s, memory_config=self.residual_mc, dtype=ttnn.bfloat16)
        summed_s.deallocate(True)
        return out

    def _attention_rows(self, li, attn_mod, qkv, pos, cos_sin):
        """The attention core for all 256 rows at once: QKV -> prefill-layout heads [1, heads, rows, hd], Q and K rotated
        per row, K / V into the BFP8_b caches by one multi-user kernel, then block attention (full layers; windowed on
        the sliding-window layers; or with B32_SWIN=0 one sliding-window decode SDPA per 32-row step on Q's step slice).
        Returns the concatenated heads [1, 1, rows, heads * hd] (row 32 i + u). The block attention reads Q in the RoPE
        output's heads layout and writes o_proj's input layout itself (no transposes, no concat-heads op)."""
        w, cfg = attn_mod.weights, attn_mod.config
        nh, nkv, hd = self.local_heads, self.local_kv_heads, cfg.head_dim
        k_cache, v_cache = self.kv_cache[li]
        block_attn = not cfg.sliding_window
        cos, sin = cos_sin
        if self.qk_joint and (block_attn or self.swin_attn is not None):
            # Q and K in one tensor (nlp_create_qkv_heads with nh + nkv "q" heads and one "kv" head: the QKV columns
            # [q | k | v] split as [q k | v0 | v1]), one RoPE op for both; the cache write reads K from heads nh .. and
            # V from the two one-head tensors, the block attention Q from heads 0 .. nh - 1
            qk, v0, v1 = ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=nh + nkv, num_kv_heads=1, transpose_k_heads=False, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            qk2 = ttnn.experimental.rotary_embedding_llama(qk, cos, sin, self.trans_rows, is_decode_mode=False,
                                                           memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
            qk.deallocate(True)
            self.kv_write.joint(qk2, v0, v1, (k_cache, v_cache), pos[self.block], self.page_table, nh)
            v0.deallocate(True)
            v1.deallocate(True)
            ba = self.block_attn if block_attn else self.swin_attn
            out = ba(qk2, k_cache, v_cache, pos[self.block], self.page_table, self.sinks[li], cfg.scaling,
                     heads_io=True, q_heads=nh)  # fmt: skip
            qk2.deallocate(True)
            return out
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv, num_heads=nh, num_kv_heads=nkv, transpose_k_heads=False, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        # (in L1, as q: the block attention reads its rows 32 bytes at a time)
        q2 = ttnn.experimental.rotary_embedding_llama(q, cos, sin, self.trans_rows, is_decode_mode=False,
                                                      memory_config=ttnn.L1_MEMORY_CONFIG)  # fmt: skip
        q.deallocate(True)
        k2 = ttnn.experimental.rotary_embedding_llama(k, cos, sin, self.trans_rows, is_decode_mode=False)
        k.deallocate(True)
        self.kv_write([k2], [v], [(k_cache, v_cache)], pos[self.block], self.page_table)
        k2.deallocate(True)
        v.deallocate(True)
        if block_attn or self.swin_attn is not None:
            ba = self.block_attn if block_attn else self.swin_attn
            out = ba(q2, k_cache, v_cache, pos[self.block], self.page_table, self.sinks[li], cfg.scaling, heads_io=True)
            q2.deallocate(True)
            return out
        q_rows = ttnn.permute(q2, (0, 2, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        q2.deallocate(True)
        outs = []
        for i in range(self.block):
            qi = ttnn.slice(q_rows, [0, i * U32, 0, 0], [1, (i + 1) * U32, nh, hd])
            outs.append(
                ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    qi,
                    k_cache,
                    v_cache,
                    cur_pos_tensor=pos[i],
                    sliding_window_size=cfg.sliding_window,
                    attention_sink=w.decode_sinks,
                    page_table_tensor=self.page_table,
                    scale=cfg.scaling,
                    program_config=self.sliding_pc,
                    compute_kernel_config=attn_mod.program_config.get_compute_kernel_config(),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            qi.deallocate(True)
        q_rows.deallocate(True)
        out = ttnn.concat(outs, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in outs:
            t.deallocate(True)
        # [1, rows, heads, hd] (row 32 i + u) -> [1, heads, rows, hd] -> [1, 1, rows, heads * hd]
        perm = ttnn.permute(out, (0, 2, 1, 3))
        out.deallocate(True)
        heads = ttnn.experimental.nlp_concat_heads(perm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        perm.deallocate(True)
        return heads

    def _sharded_rows(self, x, norm, residual_mc):
        """Norm each 32-row tile of x on its width-sharded residual shards (the batched decode's norm)."""
        parts = []
        for i in range(self.block):
            xi = ttnn.slice(x, [0, 0, i * U32, 0], [1, 1, (i + 1) * U32, self.hidden])
            res = ttnn.to_memory_config(xi, residual_mc, dtype=ttnn.bfloat16)
            xi.deallocate(True)
            n = norm.forward_sharded(res)
            res.deallocate(True)
            parts.append(ttnn.to_memory_config(n, ttnn.DRAM_MEMORY_CONFIG))
            n.deallocate(True)
        out = ttnn.concat(parts, dim=2)
        for t in parts:
            t.deallocate(True)
        return out

    def _forward_resnorm(self, res, pos, rot):
        """forward() with each residual add and the norm after it in one program (_ResNorm): the attention's and the
        MoE's all-reduce outputs stay in their width shards, the norms' outputs land in L1 interleaved."""
        model = self.model
        n0 = model.layers[0].input_layernorm.forward_sharded(res)
        xin = ttnn.to_memory_config(n0, self.attn_mc)
        n0.deallocate(True)
        feats = []
        L = len(model.layers)
        for li, layer in enumerate(model.layers):
            y = self._attention(li, layer, res, pos, rot, x=xin)  # (deallocates xin)
            if self.resnorm_router and self.grouped is not None and self.grouped.bias_slot:
                # the MoE input in BFP8 and the router's logits from the same program (the combine adds the down bias)
                res2, x8, logits = self.resnorm(y, res, layer.post_attention_layernorm, out_bf16=False, x8=True,
                                                router=layer.mlp.router)  # fmt: skip
                y.deallocate(True)
                res.deallocate(True)
                w = self.grouped.route(logits, li)
                logits.deallocate(True)
                down = self.grouped.from_x8(li, x8)
                x8.deallocate(True)
                w.deallocate(True)
                y = _all_reduce(down, None)
                down.deallocate(True)
            else:
                res2, normed = self.resnorm(y, res, layer.post_attention_layernorm)
                y.deallocate(True)
                res.deallocate(True)
                y = self._moe(li, layer, normed, None)
                normed.deallocate(True)
            nxt = model.layers[li + 1].input_layernorm if li + 1 < L else model.norm
            res, xin = self.resnorm(y, res2, nxt)
            y.deallocate(True)
            res2.deallocate(True)
            if li in self.capture_layers:
                feats.append(ttnn.to_memory_config(res, ttnn.DRAM_MEMORY_CONFIG, dtype=self.feat_dt))
        features = ttnn.concat(feats, dim=-1)
        for t in feats:
            t.deallocate(True)
        res.deallocate(True)
        normed8 = ttnn.typecast(xin, ttnn.bfloat8_b)
        xin.deallocate(True)
        logits8 = ttnn.matmul(normed8, self.lm_head, dtype=ttnn.bfloat8_b)  # as the batched decode's LM head
        normed8.deallocate(True)
        if self.argmax.custom:
            out = self.argmax.device(logits8)
            logits8.deallocate(True)
            return out, features
        logits = ttnn.typecast(logits8, ttnn.bfloat16)
        logits8.deallocate(True)
        out = self.argmax.device(logits)
        logits.deallocate(True)
        return out, features

    def forward(self, tokens, pos, rot):
        """tokens: device [1, rows] UINT32 (row 32 i + u). pos[i]: step i's [32] INT32 positions (pos[block]: every
        row's, user-major); rot: every row's cos / sin [1, 1, rows, hd] (BatchedDraft.rows_cos_sin). Returns (per-device argmax of every row, device features [1, 1, rows, 5 hidden]).
        """
        _TPAllReduce.reset()
        model = self.model
        emb = ttnn.embedding(tokens, model.embedding_weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        x = ttnn.reshape(emb, [1, 1, self.rows, self.hidden])
        feats = []
        res = ttnn.to_memory_config(x, self.residual_mc, dtype=ttnn.bfloat16)
        x.deallocate(True)
        if self.resnorm is not None:
            return self._forward_resnorm(res, pos, rot)
        for li, layer in enumerate(model.layers):
            res2 = self._attention(li, layer, res, pos, rot)
            res.deallocate(True)
            n2 = layer.post_attention_layernorm.forward_sharded(res2)
            normed = ttnn.to_memory_config(n2, self.moe_in_mc)
            n2.deallocate(True)
            # the MoE's all-reduce lands on the residual shards: the residual add stays sharded (as the attention's)
            moe = self._moe(li, layer, normed, self.residual_mc)
            normed.deallocate(True)
            res = ttnn.add(res2, moe, memory_config=self.residual_mc, dtype=ttnn.bfloat16)
            res2.deallocate(True)
            moe.deallocate(True)
            if li in self.capture_layers:
                feats.append(ttnn.to_memory_config(res, ttnn.DRAM_MEMORY_CONFIG, dtype=self.feat_dt))
        features = ttnn.concat(feats, dim=-1)
        for t in feats:
            t.deallocate(True)
        normed_s = model.norm.forward_sharded(res)
        res.deallocate(True)
        normed = ttnn.to_memory_config(normed_s, self.attn_mc)
        normed_s.deallocate(True)
        normed8 = ttnn.typecast(normed, ttnn.bfloat8_b)  # (to_memory_config to the same layout returns its input)
        normed.deallocate(True)
        logits8 = ttnn.matmul(normed8, self.lm_head, dtype=ttnn.bfloat8_b)  # as the batched decode's LM head
        normed8.deallocate(True)
        if self.argmax.custom:  # (reads the BFP8 logits as stored)
            out = self.argmax.device(logits8)
            logits8.deallocate(True)
            return out, features
        logits = ttnn.typecast(logits8, ttnn.bfloat16)
        logits8.deallocate(True)
        out = self.argmax.device(logits)
        logits.deallocate(True)
        return out, features


class BatchedDFlashRunner:
    """Greedy DFlash for `users` sequences (each prefilled into its own KV slot) on the batched decode model."""

    def __init__(self, generator, model, page_table, kv_cache, users=32, window=1024, eos_ids=(200002, 200012)):
        from .capture import TargetCapture

        self.generator, self.model, self.kv_cache = generator, model, kv_cache
        mesh = model.mesh_device
        self.mesh = mesh
        self.users = users
        self.page_table_host = page_table
        self.page_table = _replicate(mesh, page_table, ttnn.int32)
        self.V = model.vocab_size
        self.eos_ids = set(eos_ids)
        max_pos = page_table.shape[-1] * 64
        self.draft = BatchedDraft(mesh, model, max_pos, users, window=window)
        self.block = B = self.draft.block
        self.rows = users * B
        self.capture = TargetCapture(model, self.draft.target_layer_ids)
        kv = kv_cache[0] if isinstance(kv_cache[0][0], list) else kv_cache
        self.verify = BatchedVerify(model, self.page_table, kv, users, B, self.draft.target_layer_ids)
        self.verify.trans_rows = self.draft.rope.get_both_trans_mats()["prefill"]  # (the target's RoPE)
        if os.environ.get("B32_DENSE_MOE", "0") != "1":
            from .grouped_moe import GroupedMoE

            # only the routed (row, expert) pairs (the dense verify MoE computes all 32 experts for all 256 rows)
            self.verify.grouped = GroupedMoE(mesh, self.verify.moe, self.verify.hidden)
        # persistent per-round inputs (the traces' inputs)
        z = torch.zeros(users, dtype=torch.int32)
        self.d_tokens = _replicate(mesh, torch.zeros(1, self.rows, dtype=torch.int32), ttnn.uint32)
        self.v_tokens = _replicate(mesh, torch.zeros(1, self.rows, dtype=torch.int32), ttnn.uint32)
        # per-step positions, then (entry B) every row's position user-major (8 u + i: start_u + i, block attention)
        self.pos = [_replicate(mesh, z, ttnn.int32) for _ in range(B)] + [
            _replicate(mesh, torch.zeros(self.rows, dtype=torch.int32), ttnn.int32)
        ]
        self.bound = _replicate(mesh, z, ttnn.int32)
        self.pos_rows = _replicate(mesh, torch.zeros(1, self.rows, dtype=torch.int32), ttnn.uint32)
        # the per-round inputs travel as two packed host writes (int32: per-step positions, user-major positions, the
        # draft bound; uint32: the draft tokens, per-row positions), unpacked on device into the tensors above at the
        # start of a pass (one host-to-device copy each instead of 11)
        U = users
        self.unpack_i = [(t, [U]) for t in self.pos[:B]] + [(self.pos[B], [self.rows]), (self.bound, [U])]
        self.unpack_u = [(self.d_tokens, [1, self.rows]), (self.pos_rows, [1, self.rows])]
        n_i = sum(math.prod(shape) for _, shape in self.unpack_i)
        n_u = sum(math.prod(shape) for _, shape in self.unpack_u)
        self.in_i = _replicate(mesh, torch.zeros(1, n_i, dtype=torch.int32), ttnn.int32)
        self.in_u = _replicate(mesh, torch.zeros(1, n_u, dtype=torch.int32), ttnn.uint32)
        self.one_trace = True
        # the next round's inputs computed on device at the end of a round (acceptance on device): every round is
        # queued before the previous round's result is read
        self.queue_rounds = True
        put = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
        self.iota8 = put(torch.arange(B).float().reshape(1, 1, B, 1).expand(1, 1, B, U).contiguous())  # row i: i
        self.mask_rest = put(torch.full((1, 1, 1, self.rows - U), float(self.draft.mask_token_id)))
        self.limit = put(torch.zeros(1, 1, 1, U))  # each user's last start position (its token budget)
        self.first_step = ttnn.from_torch(
            (torch.arange(self.rows) < users).float().reshape(1, 1, 1, self.rows),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.traces = None

    def prefill(self, prompts):
        firsts = []
        for u, p in enumerate(prompts):
            logits = self.generator.prefill_forward_text(
                torch.tensor([p]),
                # the generator takes the page table rows of the prompts it is given (row 0 for a one-prompt call):
                # this user's row (the whole table wrote every user's K/V into user 0's blocks)
                page_table=self.page_table_host[u : u + 1],
                kv_cache=self.kv_cache,
                prompt_lens=[len(p)],
                empty_slots=[u],
                enable_trace=False,
                warmup_prefill=False,
            )
            firsts.append(int(logits.reshape(-1)[: self.V].float().argmax()))
            self.draft.add_prompt_context(self.capture.prefill_features(len(p)), user=u)
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
        if self.queue_rounds:
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
        """One trace for the draft pass, one for verify + context (the step-by-step passes), and one for the whole
        round (generate; one trace per round: no host round trip between draft and verify, b1 row 41)."""
        mesh = self.mesh
        self.release_traces()
        for t in list(self._draft_pass()) + list(self._verify_pass()) + list(self._round_pass()):
            t.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.draft_out = self._draft_pass()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        tid2 = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.verify_out = self._verify_pass()
        ttnn.end_trace_capture(mesh, tid2, cq_id=0)
        tid3 = ttnn.begin_trace_capture(mesh, cq_id=0)
        self.round_out = self._round_pass()
        ttnn.end_trace_capture(mesh, tid3, cq_id=0)
        self.traces = (tid, tid2, tid3)

    def release_traces(self):
        if self.traces is not None:
            ttnn.synchronize_device(self.mesh)
            for t in self.traces:
                ttnn.release_trace(self.mesh, t)
        self.traces = None

    def _set_inputs(self, starts, lasts):
        mesh, U, B = self.mesh, self.users, self.block
        toks = torch.full((1, self.rows), self.draft.mask_token_id, dtype=torch.int32)
        toks[0, :U] = torch.tensor(lasts, dtype=torch.int32)
        st = torch.tensor(starts, dtype=torch.int32)
        steps = st[None] + torch.arange(B)[:, None]  # [step i, user u]: start_u + i
        vi = torch.cat([steps.reshape(-1), steps.T.reshape(-1), st + B - 1])
        vu = torch.cat([toks.reshape(-1), steps.reshape(-1)])
        _copy(mesh, self.in_i, vi.reshape(1, -1), ttnn.int32)
        _copy(mesh, self.in_u, vu.reshape(1, -1), ttnn.uint32)

    def _run(self, which):
        if self.traces is None:
            return (self._draft_pass if which == 0 else self._verify_pass)(), True
        ttnn.execute_trace(self.mesh, self.traces[which], cq_id=0, blocking=False)
        return (self.draft_out if which == 0 else self.verify_out), False

    def _accept(self, ids, accept, done, dr, post):
        """Each user's drafts up to the first mismatch with the target's tokens, plus the target's next token."""
        U, B = self.users, self.block
        for u in range(U):
            d = [dr[i * U + u] for i in range(1, B)]
            p = [post[i * U + u] for i in range(B)]
            k = 0
            while k < B - 1 and d[k] == p[k]:
                k += 1
            if not done(u):
                ids[u] += d[:k] + [p[k]]
                accept[u].append(k + 1)

    def _enqueue_round(self):
        """Queue one traced round and non-blocking reads of its token ids. Returns (host tensors, event)."""
        ttnn.execute_trace(self.mesh, self.traces[2], cq_id=0, blocking=False)
        host = [t.cpu(blocking=False) for t in self.round_out]
        return host, ttnn.record_event(self.mesh, 0)

    def _generate_queued(self, ids, lens, accept, done, times, max_new_tokens):
        """Traced rounds with the acceptance on device (b1's queued rounds): round r + 1 is queued before round r's
        result is read (the device writes its inputs), so the host's read and bookkeeping overlap the next round."""
        U = self.users
        limit = torch.tensor([lens[u] + max_new_tokens for u in range(U)], dtype=torch.float32).reshape(1, 1, 1, U)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(limit, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)),
            self.limit,
        )  # fmt: skip
        t = time.perf_counter()
        pending = [self._enqueue_round()]
        while pending:
            # a round adds at most B tokens: queue the next one unless every user may finish with this one
            if len(pending) == 1 and any(len(ids[u]) - lens[u] + self.block < max_new_tokens for u in range(U)):
                pending.append(self._enqueue_round())
            host, event = pending.pop(0)
            ttnn.event_synchronize(event)
            dr, post = (ttnn.to_torch(ttnn.get_device_tensors(h)[0]).reshape(-1).long().tolist() for h in host)
            self._accept(ids, accept, done, dr, post)
            now = time.perf_counter()
            times.append((now - t, now - t, 0.0, 0.0))
            t = now
            if all(done(u) for u in range(U)):
                break
            if not pending:
                pending.append(self._enqueue_round())
        if pending:  # a round queued past the end: let it finish
            ttnn.synchronize_device(self.mesh)

    def generate(self, prompts, max_new_tokens, firsts):
        """Returns (per-user new tokens, per-user accepted counts, per-round seconds)."""
        mesh = self.mesh
        U, B = self.users, self.block
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
            if self.traces is not None and self.one_trace and self.queue_rounds:
                self._generate_queued(ids, lens, accept, done, times, max_new_tokens)
                break
            if self.traces is not None and self.one_trace:  # one trace per round
                ttnn.execute_trace(mesh, self.traces[2], cq_id=0, blocking=False)
                dr, post = (_Argmax.read(t) for t in self.round_out)
                t2 = t3 = t4 = time.perf_counter()
                self._accept(ids, accept, done, dr, post)
                t5 = time.perf_counter()
                times.append((t5 - t0, t2 - t1, 0.0, (t1 - t0) + (t5 - t4)))  # round, device (draft + verify), -, host
                continue
            d_out, owned = self._run(0)
            dr = self.draft.argmax.host(d_out)
            t2 = time.perf_counter()
            if owned:
                for t in d_out:
                    t.deallocate(True)
            vt = torch.tensor(lasts + dr[U:], dtype=torch.int32).reshape(1, self.rows)
            _copy(mesh, self.v_tokens, vt, ttnn.uint32)
            t3 = time.perf_counter()
            v_out, owned = self._run(1)
            post = self.verify.argmax.host(v_out)
            t4 = time.perf_counter()
            if owned:
                for t in v_out:
                    t.deallocate(True)
            self._accept(ids, accept, done, dr, post)
            t5 = time.perf_counter()
            times.append((t5 - t0, t2 - t1, t4 - t3, (t1 - t0) + (t3 - t2) + (t5 - t4)))  # round, draft, verify, host
        return [ids[u][lens[u] : lens[u] + max_new_tokens] for u in range(U)], accept, times
