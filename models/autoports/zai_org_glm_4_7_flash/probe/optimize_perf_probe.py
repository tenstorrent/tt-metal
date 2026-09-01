# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated-op program-config/dtype sweeps for the optimized-decoder stage.

Every dominant decode matmul role from the fused-decoder tt-perf-report gets
a candidate family sweep (default / explicit 1D mcast wide-grid / DRAM-sharded
/ batched-DRAM-sharded), timed as traced warmed replays on device 0, with a
one-shot PCC check vs torch. Results print as a table and dump to JSON.

    python .../optimize_perf_probe.py [--roles absorbed,flat,norm,router,sparse]
"""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn

TILE = 32
RESULTS = []


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if a.std() == 0 or b.std() == 0:
        return float(torch.equal(a, b))
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


class Bench:
    def __init__(self, dev):
        self.dev = dev

    def run(self, role, name, fn, ref=None, iters=64, warm=3, calls=8):
        """fn() -> output tensor. Times iters trace replays of `calls` fn()s."""
        tid = None
        try:
            out = fn()  # compile + correctness run
            p = None
            if ref is not None:
                got = ttnn.to_torch(out).float()
                p = pcc(ref, got[tuple(slice(0, s) for s in ref.shape)] if got.shape != ref.shape else got)
            ttnn.deallocate(out)
            tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
            outs = [fn() for _ in range(calls)]
            ttnn.end_trace_capture(self.dev, tid, cq_id=0)
            for _ in range(warm):
                ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.dev)
            t0 = time.perf_counter()
            for _ in range(iters):
                ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(self.dev)
            t1 = time.perf_counter()
            us = (t1 - t0) / (iters * calls) * 1e6
            ttnn.release_trace(self.dev, tid)
            tid = None
            for o in outs:
                ttnn.deallocate(o)
            row = {"role": role, "name": name, "us": round(us, 2), "pcc": None if p is None else round(p, 6)}
            print(f"{role:24s} {name:52s} {us:9.2f} us  pcc={p if p is None else f'{p:.6f}'}", flush=True)
        except Exception as e:
            if tid is not None:  # close/release a trace left open by a mid-capture failure
                for closer in (
                    lambda: ttnn.end_trace_capture(self.dev, tid, cq_id=0),
                    lambda: ttnn.release_trace(self.dev, tid),
                ):
                    try:
                        closer()
                    except Exception:
                        pass
            row = {"role": role, "name": name, "us": None, "error": f"{type(e).__name__}: {str(e)[:220]}"}
            print(f"{role:24s} {name:52s} FAIL {row['error']}", flush=True)
        RESULTS.append(row)
        return row


def dram_ws_weight_cfg(dev, k, n):
    """Width-sharded DRAM weight memory config across all banks (llama idiom)."""
    banks = dev.dram_grid_size().x
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    padded = -(-n // (TILE * banks)) * (TILE * banks)
    spec = ttnn.ShardSpec(dram_grid, (k, padded // banks), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def l1_ws_cfg(dev, rows, width, num_cores):
    grid = dev.compute_with_storage_grid_size()
    return ttnn.create_sharded_memory_config(
        shape=(rows, width // num_cores),
        core_grid=ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def ck(dev, fidelity, fp32_acc):
    return ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=fp32_acc, packer_l1_acc=True
    )


def optimal_worker_grid(dev):
    cores = dev.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    return cores, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


# ---------------------------------------------------------------------- roles


def role_absorbed(dev, b):
    """w_uk (b20 32x192x512) and w_uv (b20 32x512x256) batched matmuls."""
    torch.manual_seed(0)
    hifi2 = ck(dev, ttnn.MathFidelity.HiFi2, False)
    hifi4 = ck(dev, ttnn.MathFidelity.HiFi4, True)
    lofi = ck(dev, ttnn.MathFidelity.LoFi, False)
    banks = dev.dram_grid_size().x
    workers, worker_grid = optimal_worker_grid(dev)

    for tag, nh, k, n in (("w_uk", 20, 192, 512), ("w_uv", 20, 512, 256)):
        x = torch.randn(1, nh, 32, k)
        w = torch.randn(1, nh, k, n) * 0.02
        ref = (x @ w).float()
        xt = ttnn.from_torch(x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        for wdt, wname in ((ttnn.bfloat16, "bf16"), (ttnn.bfloat8_b, "bf8")):
            wt = ttnn.from_torch(w, device=dev, dtype=wdt, layout=ttnn.TILE_LAYOUT)
            b.run(tag, f"default_{wname}_hifi4fp32", lambda: ttnn.matmul(xt, wt, compute_kernel_config=hifi4), ref)
            b.run(tag, f"default_{wname}_hifi2", lambda: ttnn.matmul(xt, wt, compute_kernel_config=hifi2), ref)
            # explicit MatmulMultiCoreReuse: one core per head
            for gx, gy in ((5, 4), (10, 2)):
                pc = ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                    in0_block_w=k // TILE,
                    out_subblock_h=1,
                    out_subblock_w=min(4, n // TILE),
                    per_core_M=1,
                    per_core_N=n // TILE,
                )
                b.run(
                    tag,
                    f"reuse_{gx}x{gy}_{wname}_hifi2",
                    lambda pc=pc, wt=wt: ttnn.matmul(xt, wt, program_config=pc, compute_kernel_config=hifi2),
                    ref,
                )
            ttnn.deallocate(wt)

        # batched DRAM-sharded: pad heads to banks multiple (20 -> 24)
        nh_pad = -(-nh // banks) * banks
        bpb = nh_pad // banks  # batches per bank/core
        x_pad = torch.zeros(1, nh_pad, 32, k)
        x_pad[:, :nh] = x
        w_pad = torch.zeros(1, nh_pad, k, n)
        w_pad[:, :nh] = w
        in0_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(worker_grid, (bpb * 32, k), ttnn.ShardOrientation.ROW_MAJOR),
        )
        out_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(worker_grid, (bpb * 32, n), ttnn.ShardOrientation.ROW_MAJOR),
        )
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        w_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (bpb * k, n), ttnn.ShardOrientation.ROW_MAJOR),
        )
        ref_pad = (x_pad @ w_pad).float()
        xt_sh = ttnn.from_torch(x_pad, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=in0_mem)
        for wdt, wname in ((ttnn.bfloat16, "bf16"), (ttnn.bfloat8_b, "bf8")):
            wt_sh = ttnn.from_torch(w_pad, device=dev, dtype=wdt, layout=ttnn.TILE_LAYOUT, memory_config=w_mem)
            pc = ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
                in0_block_w=k // TILE, per_core_M=1, per_core_N=n // TILE, fused_activation=None
            )
            for ckc, ckname in ((hifi2, "hifi2"), (lofi, "lofi")):
                b.run(
                    tag,
                    f"batched_dram_pad{nh_pad}_{wname}_{ckname}",
                    lambda pc=pc, wt_sh=wt_sh, ckc=ckc: ttnn.matmul(
                        xt_sh, wt_sh, program_config=pc, memory_config=out_mem, compute_kernel_config=ckc
                    ),
                    ref_pad,
                )
            ttnn.deallocate(wt_sh)
        ttnn.deallocate(xt_sh)
        ttnn.deallocate(xt)


def _flat_candidates(dev, b, tag, k, n, act_rows=32, silu=False, w_dtypes=("bf16", "bf8"), extra_1d_grids=()):
    """Sweep default / 1D-mcast wide / DRAM-sharded for a flat [32,k]x[k,n] matmul."""
    torch.manual_seed(1)
    hifi2 = ck(dev, ttnn.MathFidelity.HiFi2, False)
    hifi4 = ck(dev, ttnn.MathFidelity.HiFi4, True)
    lofi = ck(dev, ttnn.MathFidelity.LoFi, False)
    banks = dev.dram_grid_size().x
    grid = dev.compute_with_storage_grid_size()
    x = torch.randn(1, 1, act_rows, k)
    w = torch.randn(k, n) * 0.02
    ref = (x @ w).float()
    if silu:
        ref = torch.nn.functional.silu(ref)
    dt = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}
    xt = ttnn.from_torch(x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    act = "silu" if silu else None
    kt, nt = k // TILE, n // TILE

    for wname in w_dtypes:
        wt = ttnn.from_torch(w, device=dev, dtype=dt[wname], layout=ttnn.TILE_LAYOUT)
        b.run(
            tag,
            f"default_{wname}_hifi4fp32",
            lambda wt=wt: ttnn.linear(xt, wt, compute_kernel_config=hifi4, activation=act),
            ref,
        )
        # 1D mcast wide grids (qwen36-blackhole idiom; weights interleaved)
        seen_1d = set()
        for target in (22, 33, 44, 64, 88, 110):
            per_core_n = -(-nt // target)
            blocks = -(-nt // per_core_n)
            if (blocks, per_core_n) in seen_1d:
                continue
            seen_1d.add((blocks, per_core_n))
            try:
                cols, rows = rect_grid(blocks)
            except ValueError:
                continue
            cores = blocks
            in0_bw = max(d for d in range(1, min(kt, 48) + 1) if kt % d == 0 and d * TILE * TILE * 2 < 200000)
            pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(cols, rows),
                in0_block_w=in0_bw,
                out_subblock_h=1,
                out_subblock_w=min(per_core_n, 4),
                out_block_h=1,
                out_block_w=per_core_n,
                per_core_M=1,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=ttnn.UnaryOpType.SILU if silu else None,
                mcast_in0=True,
            )
            for ckc, ckname in ((hifi2, "hifi2"), (lofi, "lofi")):
                if ckname == "lofi" and wname == "bf16":
                    continue
                b.run(
                    tag,
                    f"1d_{cols}x{rows}_bw{in0_bw}_{wname}_{ckname}",
                    lambda pc=pc, wt=wt, ckc=ckc: ttnn.linear(xt, wt, program_config=pc, compute_kernel_config=ckc),
                    ref,
                )
        ttnn.deallocate(wt)

        # DRAM-sharded: in0 width-sharded L1 over cores dividing kt
        n_pad = -(-n // (TILE * banks)) * (TILE * banks)
        w_padded = torch.zeros(k, n_pad)
        w_padded[:, :n] = w
        wt_ds = ttnn.from_torch(
            w_padded, device=dev, dtype=dt[wname], layout=ttnn.TILE_LAYOUT, memory_config=dram_ws_weight_cfg(dev, k, n)
        )
        ref_pad = (x @ w_padded).float()
        if silu:
            ref_pad = torch.nn.functional.silu(ref_pad)
        npt = n_pad // TILE
        for cores in (8, 16, 24, 32, 40, 64):
            if kt % cores != 0:
                continue
            shard_kt = kt // cores
            per_core_n = -(-npt // cores)
            if per_core_n * cores != npt:
                continue
            for bw in sorted({d for d in (2, 4, 5, 6, 8, 10, 16, 20, 40) if shard_kt % d == 0} | {shard_kt}):
                if bw > 40:
                    continue
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=bw,
                    per_core_M=1,
                    per_core_N=per_core_n,
                    fused_activation=ttnn.UnaryOpType.SILU if silu else None,
                )
                xt_sh = ttnn.to_memory_config(xt, l1_ws_cfg(dev, 32, k, cores))
                for ckc, ckname in ((hifi2, "hifi2"), (lofi, "lofi")):
                    if ckname == "lofi" and wname == "bf16":
                        continue
                    b.run(
                        tag,
                        f"dram_{cores}c_bw{bw}_{wname}_{ckname}",
                        lambda pc=pc, xt_sh=xt_sh, wt_ds=wt_ds, ckc=ckc: ttnn.linear(
                            xt_sh,
                            wt_ds,
                            program_config=pc,
                            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                            compute_kernel_config=ckc,
                        ),
                        ref_pad,
                    )
                ttnn.deallocate(xt_sh)
        ttnn.deallocate(wt_ds)
    ttnn.deallocate(xt)


def role_flat(dev, b):
    _flat_candidates(dev, b, "wqkv_a[2048->1344p1536]", 2048, 1344)
    _flat_candidates(dev, b, "wq_b[768->5120]", 768, 5120)
    _flat_candidates(dev, b, "wo[5120->2048]", 5120, 2048)
    _flat_candidates(dev, b, "shared_gate[2048->1536]", 2048, 1536, silu=True)
    _flat_candidates(dev, b, "shared_down[1536->2048]", 1536, 2048)
    _flat_candidates(dev, b, "dense_gate[2048->10240]", 2048, 10240, silu=True)
    _flat_candidates(dev, b, "dense_down[10240->2048]", 10240, 2048)


def rect_grid(cores):
    """Rectangular core grid (cols, rows) fitting 11x10 for an exact core count."""
    for cols in range(min(cores, 11), 0, -1):
        if cores % cols == 0 and cores // cols <= 10:
            return cols, cores // cols
    raise ValueError(cores)


def l1_ws_rect_cfg(dev, rows, width, num_cores):
    cols, grows = rect_grid(num_cores)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, grows - 1))})
    return ttnn.create_sharded_memory_config(
        shape=(rows, width // num_cores),
        core_grid=core_grid,
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def role_norm(dev, b):
    torch.manual_seed(2)
    hifi4 = ck(dev, ttnn.MathFidelity.HiFi4, True)
    for width in (2048, 768, 512):
        x = torch.randn(1, 1, 32, width)
        g = torch.randn(width).abs()
        ref = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * g).float()
        xt = ttnn.from_torch(x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        g_rm = ttnn.from_torch(
            g.reshape(1, 1, width // TILE, TILE), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        b.run(
            f"rms[{width}]",
            "default_1core",
            lambda: ttnn.rms_norm(xt, epsilon=1e-5, weight=g_rm, compute_kernel_config=hifi4),
            ref,
        )
        wt_tiles = width // TILE
        for cores in (4, 8, 16, 24, 32, 64):
            if wt_tiles % cores != 0:
                continue
            bw = wt_tiles // cores
            mem = l1_ws_rect_cfg(dev, 32, width, cores)
            cols, rows = rect_grid(cores)
            pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(cols, rows),
                subblock_w=max(d for d in (1, 2, 4) if bw % d == 0),
                block_h=1,
                block_w=bw,
                inplace=False,
            )
            xt_sh = ttnn.to_memory_config(xt, mem)
            b.run(
                f"rms[{width}]",
                f"sharded_{cores}c",
                lambda pc=pc, xt_sh=xt_sh, mem=mem: ttnn.rms_norm(
                    xt_sh, epsilon=1e-5, weight=g_rm, program_config=pc, compute_kernel_config=hifi4, memory_config=mem
                ),
                ref,
            )
            ttnn.deallocate(xt_sh)
        ttnn.deallocate(xt)


def role_router(dev, b):
    torch.manual_seed(3)
    hifi4f = ck(dev, ttnn.MathFidelity.HiFi4, True)
    hifi2f = ck(dev, ttnn.MathFidelity.HiFi2, True)
    x = torch.randn(1, 1, 32, 2048)
    w = torch.randn(2048, 64) * 0.02
    ref = (x @ w).float()
    xt = ttnn.from_torch(x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    wt = ttnn.from_torch(w, device=dev, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    b.run(
        "router[2048->64]",
        "default_fp32w_hifi4fp32",
        lambda: ttnn.linear(xt, wt, compute_kernel_config=hifi4f, dtype=ttnn.float32),
        ref,
    )
    b.run(
        "router[2048->64]",
        "default_fp32w_hifi2fp32",
        lambda: ttnn.linear(xt, wt, compute_kernel_config=hifi2f, dtype=ttnn.float32),
        ref,
    )
    for cores, bw in ((2, 8), (2, 16), (2, 32)):
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(cores, 1),
            in0_block_w=bw,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        b.run(
            "router[2048->64]",
            f"1d_{cores}c_bw{bw}_fp32w_hifi4fp32",
            lambda pc=pc: ttnn.linear(xt, wt, program_config=pc, compute_kernel_config=hifi4f, dtype=ttnn.float32),
            ref,
        )
    ttnn.deallocate(wt)
    ttnn.deallocate(xt)


def role_sparse(dev, b):
    torch.manual_seed(4)
    hifi2 = ck(dev, ttnn.MathFidelity.HiFi2, True)
    hifi2nf = ck(dev, ttnn.MathFidelity.HiFi2, False)
    lofi = ck(dev, ttnn.MathFidelity.LoFi, False)
    E, H, I, k = 64, 2048, 1536, 4
    x = torch.randn(1, 1, 32, H)
    gate_up = torch.randn(1, E, H, 2 * I) * 0.02
    down = torch.randn(1, E, I, H) * 0.02
    idx = torch.tensor([3, 17, 42, 63], dtype=torch.int32)
    xt = ttnn.from_torch(x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ones = ttnn.from_torch(torch.ones(1, 1, 1, E), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    idx_rm = ttnn.from_torch(
        idx.reshape(1, 1, 1, k).to(torch.int16), device=dev, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    ref_gu = torch.stack([x[0, 0] @ gate_up[0, e] for e in idx.tolist()]).unsqueeze(0).float()
    h_in = torch.randn(1, k, 32, I)
    ref_dn = torch.stack([h_in[0, i] @ down[0, e] for i, e in enumerate(idx.tolist())]).unsqueeze(0).float()
    ht = ttnn.from_torch(h_in, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def sparse_pc(nt, kt, pcn, in0_bw, osw=1):
        """Grid sized exactly to ceil(nt/pcn) blocks (rectangularity contract)."""
        blocks = -(-nt // pcn)
        cx, cy = rect_grid(blocks)
        if kt % in0_bw != 0:
            in0_bw = max(d for d in range(1, in0_bw + 1) if kt % d == 0)
        if pcn % osw != 0:
            osw = max(d for d in range(1, osw + 1) if pcn % d == 0)
        return (
            ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
                in0_block_w=in0_bw,
                out_subblock_h=1,
                out_subblock_w=osw,
                out_block_h=1,
                out_block_w=osw,
                per_core_M=1,
                per_core_N=pcn,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            ),
            f"{cx}x{cy}",
        )

    gu_nt, gu_kt = (2 * I) // TILE, H // TILE  # 96, 64
    dn_nt, dn_kt = H // TILE, I // TILE  # 64, 48
    for wdt, wname in ((ttnn.bfloat8_b, "bf8"), (ttnn.bfloat4_b, "bf4")):
        gut = ttnn.from_torch(gate_up, device=dev, dtype=wdt, layout=ttnn.TILE_LAYOUT)
        dnt = ttnn.from_torch(down, device=dev, dtype=wdt, layout=ttnn.TILE_LAYOUT)
        for pcn in (2, 3, 4, 6, 12):
            for bw in (8, 16, 32, 64):
                if gu_kt % bw != 0:
                    continue
                for osw in (1, 2):
                    if pcn % osw != 0:
                        continue
                    for ckc, ckn in ((hifi2, "hifi2f32"), (hifi2nf, "hifi2"), (lofi, "lofi")):
                        if ckn == "hifi2f32" and not (wname == "bf8" and pcn == 3 and bw == 8 and osw == 1):
                            continue  # current-baseline config only
                        pc, gname = sparse_pc(gu_nt, gu_kt, pcn, bw, osw)
                        for memc, memn in ((ttnn.DRAM_MEMORY_CONFIG, "dram"), (ttnn.L1_MEMORY_CONFIG, "L1")):
                            if memn == "L1" and not (osw == 1 and ckn != "hifi2f32"):
                                continue
                            b.run(
                                "sparse_gate_up",
                                f"gu_{gname}_pcn{pcn}_bw{bw}_osw{osw}_{wname}_{ckn}_{memn}",
                                lambda pc=pc, gut=gut, ckc=ckc, memc=memc: ttnn.sparse_matmul(
                                    xt,
                                    gut,
                                    sparsity=ones,
                                    indices=idx_rm,
                                    is_input_b_sparse=True,
                                    program_config=pc,
                                    memory_config=memc,
                                    compute_kernel_config=ckc,
                                    dtype=ttnn.bfloat16,
                                ),
                                None if memn == "L1" else ref_gu.reshape(1, 1, 1, k, 32, 2 * I),
                                iters=16,
                            )
        for pcn in (1, 2, 4, 8):
            for bw in (6, 8, 12, 16, 24, 48):
                if dn_kt % bw != 0:
                    continue
                for osw in (1, 2):
                    if pcn % osw != 0:
                        continue
                    for ckc, ckn in ((hifi2, "hifi2f32"), (hifi2nf, "hifi2"), (lofi, "lofi")):
                        if ckn == "hifi2f32" and not (wname == "bf8" and pcn == 2 and bw == 6 and osw == 1):
                            continue
                        pc, gname = sparse_pc(dn_nt, dn_kt, pcn, bw, osw)
                        for memc, memn in ((ttnn.DRAM_MEMORY_CONFIG, "dram"), (ttnn.L1_MEMORY_CONFIG, "L1")):
                            if memn == "L1" and not (osw == 1 and ckn != "hifi2f32"):
                                continue
                            b.run(
                                "sparse_down",
                                f"dn_{gname}_pcn{pcn}_bw{bw}_osw{osw}_{wname}_{ckn}_{memn}",
                                lambda pc=pc, dnt=dnt, ckc=ckc, memc=memc: ttnn.sparse_matmul(
                                    ht,
                                    dnt,
                                    sparsity=ones,
                                    indices=idx_rm,
                                    is_input_a_sparse=True,
                                    is_input_b_sparse=True,
                                    program_config=pc,
                                    memory_config=memc,
                                    compute_kernel_config=ckc,
                                    dtype=ttnn.bfloat16,
                                ),
                                None if memn == "L1" else ref_dn,
                                iters=16,
                            )
        ttnn.deallocate(gut)
        ttnn.deallocate(dnt)
    ttnn.deallocate(xt)
    ttnn.deallocate(ht)


def role_qpath(dev, b):
    """Fold-W_UK-v2 candidates: per-head batched matmuls under explicit reuse
    configs + bf8, vs the current flat wq_b + untilize/reshape/tilize + slices
    + transpose + w_uk chain. Also gather-dtype candidates for the router."""
    torch.manual_seed(5)
    hifi2 = ck(dev, ttnn.MathFidelity.HiFi2, False)
    lofi = ck(dev, ttnn.MathFidelity.LoFi, False)
    nh, qlr, dnope, drope, dkv = 20, 768, 192, 64, 512
    q_normed = torch.randn(1, 1, 32, qlr)
    wq_b = torch.randn(qlr, nh * (dnope + drope)) * 0.02
    w_uk = torch.randn(1, nh, dnope, dkv) * 0.02
    # fold: W_fold[h] = wq_b_nope[h] @ w_uk[h]  -> [1, nh, qlr, dkv]
    wq_b_heads = wq_b.reshape(qlr, nh, dnope + drope).permute(1, 0, 2)  # [nh, qlr, 256]
    w_fold = torch.stack([wq_b_heads[h, :, :dnope] @ w_uk[0, h] for h in range(nh)]).unsqueeze(0)  # [1,nh,qlr,dkv]
    w_rope_heads = wq_b_heads[:, :, dnope:].unsqueeze(0).contiguous()  # [1, nh, qlr, drope]
    ref_lat = (q_normed @ w_fold).float()  # [1, nh, 32, dkv]
    ref_rope = (q_normed @ w_rope_heads).float()

    xt = ttnn.from_torch(q_normed, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    xt_rep = ttnn.from_torch(
        q_normed.repeat(1, nh, 1, 1), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )  # pre-repeated LHS for the equal-batch reuse config
    for wdt, wname in ((ttnn.bfloat8_b, "bf8"),):
        wf = ttnn.from_torch(w_fold, device=dev, dtype=wdt, layout=ttnn.TILE_LAYOUT)
        wr = ttnn.from_torch(w_rope_heads, device=dev, dtype=wdt, layout=ttnn.TILE_LAYOUT)
        pc_lat = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 4),
            in0_block_w=qlr // TILE,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=dkv // TILE,
        )
        pc_rope = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 4),
            in0_block_w=qlr // TILE,
            out_subblock_h=1,
            out_subblock_w=2,
            per_core_M=1,
            per_core_N=drope // TILE,
        )
        # 1D non-mcast in0-reuse path (the only legal broadcast-LHS config)
        pc_lat_1d = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 4),
            in0_block_w=qlr // TILE,
            out_subblock_h=1,
            out_subblock_w=4,
            out_block_h=1,
            out_block_w=dkv // TILE,
            per_core_M=1,
            per_core_N=dkv // TILE,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=False,
        )
        pc_rope_1d = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 4),
            in0_block_w=qlr // TILE,
            out_subblock_h=1,
            out_subblock_w=2,
            out_block_h=1,
            out_block_w=drope // TILE,
            per_core_M=1,
            per_core_N=drope // TILE,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=False,
        )
        for ckc, ckn in ((hifi2, "hifi2"), (lofi, "lofi")):
            b.run(
                "q_fold_lat",
                f"bcast1d_5x4_{wname}_{ckn}",
                lambda ckc=ckc: ttnn.matmul(xt, wf, program_config=pc_lat_1d, compute_kernel_config=ckc),
                ref_lat,
            )
            b.run(
                "q_fold_rope",
                f"bcast1d_5x4_{wname}_{ckn}",
                lambda ckc=ckc: ttnn.matmul(xt, wr, program_config=pc_rope_1d, compute_kernel_config=ckc),
                ref_rope,
            )
            b.run(
                "q_fold_lat",
                f"repeat+reuse5x4_{wname}_{ckn}",
                lambda ckc=ckc: ttnn.matmul(
                    ttnn.repeat(xt, ttnn.Shape((1, nh, 1, 1))), wf, program_config=pc_lat, compute_kernel_config=ckc
                ),
                ref_lat,
            )
            b.run(
                "q_fold_lat",
                f"prerepeated_reuse5x4_{wname}_{ckn}",
                lambda ckc=ckc: ttnn.matmul(xt_rep, wf, program_config=pc_lat, compute_kernel_config=ckc),
                ref_lat,
            )
    # gather dtype candidates (router chain)
    scores = torch.rand(1, 1, 32, 64)
    idx = torch.randint(0, 64, (1, 1, 32, 4))
    ref_g = torch.gather(scores, 3, idx).float()
    it = ttnn.from_torch(idx.to(torch.int16), device=dev, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT)
    for sdt, sname in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
        st = ttnn.from_torch(scores, device=dev, dtype=sdt, layout=ttnn.TILE_LAYOUT)
        b.run("router_gather", f"tile_{sname}", lambda st=st: ttnn.gather(st, dim=3, index=it), ref_g)
    # embedding-based compact lookup for B=1: table [E,1] from scores row 0
    s1 = scores[:, :, :1, :]  # [1,1,1,64]
    ref_e = torch.gather(s1.reshape(1, 64), 1, idx[0, 0, :1].to(torch.int64)).reshape(1, 4).float()
    st1 = ttnn.from_torch(
        s1, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    idx1 = ttnn.from_torch(
        idx[:1, 0, :1].to(torch.int32).reshape(1, 4),
        device=dev,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    def emb_pick():
        t = ttnn.to_layout(st1, ttnn.ROW_MAJOR_LAYOUT)  # [1,1,1,64]
        t = ttnn.reshape(t, (64, 1))
        picked = ttnn.embedding(idx1, t)  # [1, 4, 1]
        return ttnn.to_layout(ttnn.reshape(picked, (1, 1, 1, 4)), ttnn.TILE_LAYOUT)

    b.run("router_gather", "b1_embedding_pick_chain", emb_pick, ref_e.reshape(1, 1, 1, 4))


def role_sparse_prefill(dev, b):
    """Prefill-shaped sparse expert matmuls (G=32 groups of 32 tokens, all-ones
    sparsity = worst case): grid/in0_block_w sweep at bf4."""
    torch.manual_seed(8)
    hifi2f = ck(dev, ttnn.MathFidelity.HiFi2, True)
    lofi = ck(dev, ttnn.MathFidelity.LoFi, False)
    E, H, I, G = 64, 2048, 1536, 32
    x = torch.randn(1, G, 32, H) * 0.5
    gate_up = torch.randn(1, E, H, 2 * I) * 0.02
    down = torch.randn(1, E, I, H) * 0.02
    xt = ttnn.from_torch(x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ones = ttnn.from_torch(torch.ones(1, G, 1, E), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ones_e = ttnn.from_torch(torch.ones(1, 1, 1, E), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    h_in = torch.randn(1, E, 1024, I) * 0.5
    ht = ttnn.from_torch(h_in, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def pc_of(nt, kt, pcn, bw, osw, m=32):
        blocks = -(-nt // pcn)
        cols, rows = rect_grid(blocks)
        return (
            ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=ttnn.CoreCoord(cols, rows),
                in0_block_w=bw,
                out_subblock_h=1,
                out_subblock_w=osw,
                out_block_h=1,
                out_block_w=osw,
                per_core_M=m // TILE,
                per_core_N=pcn,
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            ),
            f"{cols}x{rows}",
        )

    for wdt, wname in ((ttnn.bfloat4_b, "bf4"),):
        gut = ttnn.from_torch(gate_up, device=dev, dtype=wdt, layout=ttnn.TILE_LAYOUT)
        dnt = ttnn.from_torch(down, device=dev, dtype=wdt, layout=ttnn.TILE_LAYOUT)
        for pcn, bw, osw, ckc, ckn in (
            (3, 8, 1, hifi2f, "hifi2f32"),  # current baseline
            (3, 16, 1, hifi2f, "hifi2f32"),
            (3, 32, 1, hifi2f, "hifi2f32"),
            (3, 32, 3, hifi2f, "hifi2f32"),
            (2, 32, 2, hifi2f, "hifi2f32"),
            (2, 64, 2, hifi2f, "hifi2f32"),
            (3, 32, 3, lofi, "lofi"),
            (2, 64, 2, lofi, "lofi"),
        ):
            pc, g = pc_of(96, 64, pcn, bw, osw)
            b.run(
                "sparse_gu_prefill",
                f"{g}_pcn{pcn}_bw{bw}_osw{osw}_{wname}_{ckn}",
                lambda pc=pc, ckc=ckc: ttnn.sparse_matmul(
                    xt,
                    gut,
                    sparsity=ones,
                    nnz=G * E,
                    program_config=pc,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=ckc,
                    dtype=ttnn.bfloat16,
                ),
                None,
                iters=4,
                calls=1,
            )
        for pcn, bw, osw, ckc, ckn in (
            (2, 6, 1, hifi2f, "hifi2f32"),  # current baseline (s=1024 chunk)
            (2, 24, 2, hifi2f, "hifi2f32"),
            (1, 24, 1, hifi2f, "hifi2f32"),
            (1, 48, 1, hifi2f, "hifi2f32"),
            (1, 48, 1, lofi, "lofi"),
        ):
            pc, g = pc_of(64, 48, pcn, bw, osw, m=1024)
            b.run(
                "sparse_dn_prefill",
                f"{g}_pcn{pcn}_bw{bw}_osw{osw}_{wname}_{ckn}",
                lambda pc=pc, ckc=ckc: ttnn.sparse_matmul(
                    ht,
                    dnt,
                    sparsity=ones_e,
                    nnz=E,
                    is_input_a_sparse=True,
                    program_config=pc,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=ckc,
                    dtype=ttnn.bfloat16,
                ),
                None,
                iters=4,
                calls=1,
            )
        ttnn.deallocate(gut)
        ttnn.deallocate(dnt)


def role_sparse_osw_bug(dev, b):
    """Minimal repro for the sparse_matmul multi-group corruption: in the
    NON-indexed sparsity-walk mode, out_subblock_w>1 (out_block_w=osw)
    corrupts multi-group outputs (PCC ~0.82-0.87 vs 0.9939 at osw=1); the
    indexed/gather mode is immune (identical PCC osw1 vs osw2). Candidate
    ttnn issue; do not use osw>1 on non-indexed multi-group sparse matmuls."""
    torch.manual_seed(9)
    hifi2f = ck(dev, ttnn.MathFidelity.HiFi2, True)
    E, H, I, G = 64, 2048, 1536, 4
    x = torch.randn(1, G, 32, H) * 0.5
    w = torch.randn(1, E, H, 2 * I) * 0.02
    xt = ttnn.from_torch(x, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ones = ttnn.from_torch(torch.ones(1, G, 1, E), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    wt = ttnn.from_torch(w, device=dev, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)
    ref = torch.stack([x[0] @ w[0, e] for e in range(E)], dim=1).unsqueeze(0).float()  # [1,G,E,32,N]
    for pcn, bw, osw in ((3, 8, 1), (2, 32, 1), (2, 32, 2), (3, 32, 3), (2, 8, 2)):
        blocks = -(-(2 * I // TILE) // pcn)
        cols, rows = rect_grid(blocks)
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(cols, rows),
            in0_block_w=bw,
            out_subblock_h=1,
            out_subblock_w=osw,
            out_block_h=1,
            out_block_w=osw,
            per_core_M=1,
            per_core_N=pcn,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        try:
            out = ttnn.sparse_matmul(
                xt,
                wt,
                sparsity=ones,
                nnz=G * E,
                program_config=pc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=hifi2f,
                dtype=ttnn.bfloat16,
            )
            got = ttnn.to_torch(out).float().reshape(1, G, E, 32, 2 * I)
            p = pcc(ref, got)
            verdict = "OK" if p > 0.99 else "CORRUPT"
            print(f"sparse_osw_bug pcn{pcn}_bw{bw}_osw{osw} {cols}x{rows}: PCC={p:.6f} {verdict}", flush=True)
            RESULTS.append({"role": "sparse_osw_bug", "name": f"pcn{pcn}_bw{bw}_osw{osw}", "pcc": round(p, 6)})
            ttnn.deallocate(out)
        except Exception as e:
            print(f"sparse_osw_bug pcn{pcn}_bw{bw}_osw{osw}: FAIL {str(e)[:120]}", flush=True)


def role_wopath(dev, b):
    """Attention output path candidates for [1,nh,B,*] inputs:
    (b) batched per-head wo + reduce vs (c) host-folded w_uv@wo + reduce."""
    torch.manual_seed(7)
    hifi2 = ck(dev, ttnn.MathFidelity.HiFi2, False)
    lofi = ck(dev, ttnn.MathFidelity.LoFi, False)
    nh, dkv, dv, hid = 20, 512, 256, 2048
    attn = torch.randn(1, nh, 32, dkv) * 0.3
    w_uv = torch.randn(1, nh, dkv, dv) * 0.02
    wo = torch.randn(nh * dv, hid) * 0.02
    wo_heads = wo.reshape(nh, dv, hid).unsqueeze(0)
    fold = (w_uv[0] @ wo_heads[0]).unsqueeze(0)  # [1, nh, dkv, hid]
    v_ref = attn @ w_uv
    ref_b = (v_ref @ wo_heads).sum(1, keepdim=True).float()

    at = ttnn.from_torch(attn, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    vt = ttnn.from_torch(v_ref, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    for wdt, wname in ((ttnn.bfloat8_b, "bf8"),):
        woh = ttnn.from_torch(wo_heads, device=dev, dtype=wdt, layout=ttnn.TILE_LAYOUT)
        foldt = ttnn.from_torch(fold, device=dev, dtype=wdt, layout=ttnn.TILE_LAYOUT)
        pc_b = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 4),
            in0_block_w=dv // TILE,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=hid // TILE,
        )
        pc_c = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(5, 4),
            in0_block_w=dkv // TILE,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=hid // TILE,
        )
        for ckc, ckn in ((hifi2, "hifi2"), (lofi, "lofi")):
            b.run(
                "wo_batched+sum",
                f"reuse5x4_{wname}_{ckn}",
                lambda ckc=ckc: ttnn.experimental.fast_reduce_nc(
                    ttnn.matmul(vt, woh, program_config=pc_b, compute_kernel_config=ckc), dims=[1]
                ),
                ref_b,
                iters=32,
            )
            b.run(
                "wuv_wo_fold+sum",
                f"reuse5x4_{wname}_{ckn}",
                lambda ckc=ckc: ttnn.experimental.fast_reduce_nc(
                    ttnn.matmul(at, foldt, program_config=pc_c, compute_kernel_config=ckc), dims=[1]
                ),
                ref_b,
                iters=32,
            )
        ttnn.deallocate(woh)
        ttnn.deallocate(foldt)
    ttnn.deallocate(at)
    ttnn.deallocate(vt)


def role_sdpa(dev, b):
    """paged MLA flash decode at ctx 1024: bf16 vs bf8 latent cache, config sweep."""
    torch.manual_seed(6)
    nh, dkv, dpe = 20, 512, 64
    kvpe_dim = dkv + dpe
    block, nblocks = 64, 16 * 4  # 4096 ctx worth of blocks
    cache_t = torch.randn(nblocks, 1, block, kvpe_dim) * 0.5
    pt_t = torch.randperm(nblocks, dtype=torch.int32).reshape(1, nblocks)
    q_t = torch.randn(1, 1, nh, kvpe_dim) * 0.5
    pos = 1023
    grid = dev.compute_with_storage_grid_size()
    pt = ttnn.from_torch(pt_t, device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cur = ttnn.from_torch(torch.tensor([pos], dtype=torch.int32), device=dev)
    q = ttnn.from_torch(q_t, device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ck_flash = ck(dev, ttnn.MathFidelity.HiFi4, False)
    ref = None
    for cdt, cname in ((ttnn.bfloat16, "bf16"), (ttnn.bfloat8_b, "bf8")):
        cache = ttnn.from_torch(cache_t, device=dev, dtype=cdt, layout=ttnn.TILE_LAYOUT)
        for kchunk in (128, 256, 512):
            for mcphb in (8, 16):
                pc = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=grid,
                    q_chunk_size=0,
                    k_chunk_size=kchunk,
                    exp_approx_mode=False,
                    max_cores_per_head_batch=mcphb,
                )
                r = b.run(
                    "sdpa_mla_1024",
                    f"{cname}_k{kchunk}_mc{mcphb}",
                    lambda pc=pc, cache=cache: ttnn.transformer.paged_flash_multi_latent_attention_decode(
                        q,
                        cache,
                        head_dim_v=dkv,
                        page_table_tensor=pt,
                        cur_pos_tensor=cur,
                        scale=(192 + 64) ** -0.5,
                        program_config=pc,
                        compute_kernel_config=ck_flash,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    ),
                    ref,
                )
        ttnn.deallocate(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default="absorbed,flat,norm,router,sparse")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=0)
    b = Bench(dev)
    try:
        for role in args.roles.split(","):
            {
                "absorbed": role_absorbed,
                "flat": role_flat,
                "norm": role_norm,
                "router": role_router,
                "sparse": role_sparse,
                "qpath": role_qpath,
                "wopath": role_wopath,
                "sparse_prefill": role_sparse_prefill,
                "sparse_osw_bug": role_sparse_osw_bug,
                "sdpa": role_sdpa,
            }[role](dev, b)
    finally:
        ttnn.close_device(dev)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(RESULTS, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
