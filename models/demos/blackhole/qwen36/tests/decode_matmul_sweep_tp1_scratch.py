# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SCRATCH (lane C): TP=1 decode matmul grid sweep on one Blackhole die (mesh (1,1)).

For each decode matmul shape of Qwen3.8-27B at TP=1 (weights interleaved DRAM, activation [1,1,32,K] bf16, M=1 row
valid), time the CURRENT program config and a set of candidates with a captured trace of REPS back-to-back matmuls
(replay timing = device time incl. in-trace dispatch), and check whether each candidate's output is BIT-IDENTICAL to
the current config's output (fp32 dest accumulation -> a grid/blocking change must not change the numbers).

  SWEEP_SHAPES=gate_up,down,qkvzab,gdn_out,attn_qkv,attn_wo,lm_head  SWEEP_REPS=20  python <this>
Prints SWEEP_RESULT json lines; appends to $SWEEP_OUT when set.
"""

import json
import math
import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc

REPS = int(os.environ.get("SWEEP_REPS", "20"))
OUT = os.environ.get("SWEEP_OUT", "")
SHAPES = os.environ.get("SWEEP_SHAPES", "gate_up,down,qkvzab,gdn_out,attn_qkv,attn_wo,lm_head").split(",")
GRID_W = 11  # BH P150 worker grid width

# name -> (K, N, weight dtype, current progcfg factory(grid_w), fused_activation)
DIM, HID = 5120, 17408
SHAPE_DEFS = {
    "gate_up": (
        DIM,
        HID,
        ttnn.bfloat4_b,
        lambda: tpc.create_matmul_1d_decode_progcfg(
            1, DIM, HID, num_cores=77, fused_activation=ttnn.UnaryOpType.SILU, grid_w=GRID_W
        ),
        ttnn.UnaryOpType.SILU,
    ),
    "down": (
        HID,
        DIM,
        ttnn.bfloat8_b,
        lambda: tpc.create_matmul_1d_decode_progcfg(1, HID, DIM, num_cores=44, grid_w=GRID_W),
        None,
    ),
    "qkvzab": (
        DIM,
        16480,
        ttnn.bfloat8_b,
        lambda: tpc.create_matmul_1d_decode_progcfg(1, DIM, 16480, num_cores=44, grid_w=GRID_W),
        None,
    ),
    "gdn_out": (
        6144,
        DIM,
        ttnn.bfloat8_b,
        lambda: tpc.create_matmul_1d_decode_progcfg(1, 6144, DIM, num_cores=44, grid_w=GRID_W),
        None,
    ),
    "attn_qkv": (
        DIM,
        14336,
        ttnn.bfloat8_b,
        lambda: tpc.create_matmul_1d_decode_progcfg(1, DIM, 14336, num_cores=56),
        None,
    ),
    "attn_wo": (
        8192,
        DIM,
        ttnn.bfloat8_b,
        lambda: tpc.create_matmul_1d_decode_progcfg(1, 8192, DIM, num_cores=44, grid_w=GRID_W),
        None,
    ),
    "lm_head": (DIM, 248320, ttnn.bfloat8_b, lambda: None, None),
}


def emit(**kw):
    line = "SWEEP_RESULT " + json.dumps(kw)
    print(line, flush=True)
    if OUT:
        with open(OUT, "a") as f:
            f.write(json.dumps(kw) + "\n")


def candidates(name, K, N, act):
    """(label, progcfg) candidates: 1D mcast grids of various sizes/shapes (+ per_core_N balance), and the
    ttnn auto config."""
    out = []
    n_tiles = N // 32
    for cores in (32, 44, 55, 64, 66, 77, 88, 99, 110):
        for gw in (GRID_W, 8):
            if cores > 110 or (gw == 8 and cores > 80):
                continue
            if math.ceil(cores / gw) > 10:
                continue
            pcN = math.ceil(n_tiles / cores)
            eff = n_tiles / (pcN * cores)
            out.append(
                (
                    f"1d_c{cores}_w{gw}_pcN{pcN}_eff{eff:.2f}",
                    tpc.create_matmul_1d_decode_progcfg(1, K, N, num_cores=cores, fused_activation=act, grid_w=gw),
                )
            )
    if name == "lm_head":
        out = []  # the generic 1D grids overflow L1 (per_core_N 71-243 tiles x in0_block_w 8); use small K blocks
        k_tiles = K // 32
        for cores in (110, 100, 88, 80):
            gw = GRID_W if cores in (110, 88) else 10
            pcN = math.ceil(n_tiles / cores)
            for ibw in (1, 2, 4):
                for sub_w in (1, 2, 4):
                    if pcN % sub_w:
                        continue
                    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(gw, math.ceil(cores / gw)),
                        in0_block_w=ibw,
                        out_subblock_h=1,
                        out_subblock_w=sub_w,
                        per_core_M=1,
                        per_core_N=pcN,
                        fuse_batch=True,
                        fused_activation=None,
                        mcast_in0=True,
                    )
                    out.append((f"1d_c{cores}_ibw{ibw}_sw{sub_w}_pcN{pcN}", pc))
    out.append(("auto", None))
    return out


def dram_sharded_candidates(K, N):
    """(label, num_cores) for the DRAM-WIDTH_SHARDED weight kernel: the worker count that reads the 8 DRAM
    shards; None = tp_common's default (_find_grid(k_tiles) ~32)."""
    out = [("dram_default", None)]
    k_tiles = K // 32
    for cores in (8, 16, 24, 32, 40, 48, 56, 64, 72, 80):
        if k_tiles % cores == 0:
            out.append((f"dram_c{cores}", cores))
    return out


def run_one(mesh, name, K, N, wdtype, cur_factory, act):
    torch.manual_seed(0)
    w_host = torch.randn(K, N, dtype=torch.bfloat16) * 0.02
    x_host = torch.zeros(1, 1, 32, K, dtype=torch.bfloat16)
    x_host[0, 0, 0] = torch.randn(K, dtype=torch.bfloat16)
    w = ttnn.from_torch(
        w_host, dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    x = ttnn.from_torch(
        x_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    wbytes = K * N * {ttnn.bfloat4_b: 0.5625, ttnn.bfloat8_b: 1.0625, ttnn.bfloat16: 2.0}[wdtype]
    out_mc = ttnn.DRAM_MEMORY_CONFIG if name == "lm_head" else ttnn.L1_MEMORY_CONFIG

    def mm(pc):
        kw = {"program_config": pc} if pc is not None else {}
        if pc is None and act is not None:
            kw["activation"] = "silu"
        return ttnn.linear(x, w, compute_kernel_config=ckc, memory_config=out_mc, **kw)

    ref = None
    cur_pc = cur_factory()
    cands = [("current", cur_pc)] + candidates(name, K, N, act)
    for label, pc in cands:
        try:
            o = mm(pc)  # compile
            ttnn.synchronize_device(mesh)
            o_host = ttnn.to_torch(o)
            ttnn.deallocate(o)
            if ref is None:
                ref = o_host
                exact = True
            else:
                exact = bool(torch.equal(ref, o_host))
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                outs = [mm(pc) for _ in range(REPS)]
            finally:
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            best = 1e9
            for _ in range(3):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                best = min(best, (time.perf_counter() - t0) / REPS)
            ttnn.release_trace(mesh, tid)
            for t in outs:
                ttnn.deallocate(t)
            us = best * 1e6
            gbs = wbytes / best / 1e9
            logger.info(f"[SWEEP] {name} {label}: {us:.1f} us  {gbs:.0f} GB/s  exact={exact}")
            emit(shape=name, K=K, N=N, dtype=str(wdtype), label=label, us=round(us, 1), gbs=round(gbs), exact=exact)
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:200]
            logger.warning(f"[SWEEP] {name} {label}: FAILED {msg}")
            emit(shape=name, K=K, N=N, dtype=str(wdtype), label=label, error=msg)
            ttnn.synchronize_device(mesh)
    ttnn.deallocate(w)
    ttnn.deallocate(x)
    if os.environ.get("SWEEP_DRAM", "1") != "1":
        return
    # DRAM-sharded weight path (tpc.sharded_decode_matmul): weight WIDTH_SHARDED over the 8 DRAM banks, activation
    # L1 WIDTH_SHARDED (act_shard config), output L1 width-sharded then -> L1 interleaved (as the model does).
    try:
        w_mc = tpc.create_dram_sharded_mem_config(K, N)
        w = ttnn.from_torch(w_host, dtype=wdtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=w_mc)
        act_cfg = tpc.create_activation_shard_config(K)
        x = ttnn.from_torch(x_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=act_cfg)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[SWEEP] {name} dram-sharded setup FAILED {str(e).splitlines()[0][:200]}")
        emit(shape=name, K=K, N=N, dtype=str(wdtype), label="dram_setup", error=str(e).splitlines()[0][:200])
        return

    def mm_ds(pc):
        o = ttnn.linear(
            x, w, compute_kernel_config=ckc, program_config=pc, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        )
        o2 = ttnn.to_memory_config(o, out_mc)
        ttnn.deallocate(o)
        return o2

    for label, cores in dram_sharded_candidates(K, N):
        try:
            pc = tpc.create_dram_sharded_matmul_program_config(1, K, N, num_cores=cores)
            if act is not None:
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=pc.in0_block_w, per_core_M=pc.per_core_M, per_core_N=pc.per_core_N, fused_activation=act
                )
            o = mm_ds(pc)
            ttnn.synchronize_device(mesh)
            o_host = ttnn.to_torch(o)
            ttnn.deallocate(o)
            exact = bool(torch.equal(ref, o_host)) if ref is not None else None
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                outs = [mm_ds(pc) for _ in range(REPS)]
            finally:
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            best = 1e9
            for _ in range(3):
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                best = min(best, (time.perf_counter() - t0) / REPS)
            ttnn.release_trace(mesh, tid)
            for t in outs:
                ttnn.deallocate(t)
            us = best * 1e6
            gbs = wbytes / best / 1e9
            logger.info(
                f"[SWEEP] {name} {label} (in0_block_w {pc.in0_block_w} per_core_N {pc.per_core_N}): {us:.1f} us  {gbs:.0f} GB/s  exact={exact}"
            )
            emit(
                shape=name,
                K=K,
                N=N,
                dtype=str(wdtype),
                label=label,
                us=round(us, 1),
                gbs=round(gbs),
                exact=exact,
                in0_block_w=int(pc.in0_block_w),
                per_core_N=int(pc.per_core_N),
            )
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:200]
            logger.warning(f"[SWEEP] {name} {label}: FAILED {msg}")
            emit(shape=name, K=K, N=N, dtype=str(wdtype), label=label, error=msg)
            ttnn.synchronize_device(mesh)
    ttnn.deallocate(w)
    ttnn.deallocate(x)


def main():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576, trace_region_size=268435456)
    mesh.enable_program_cache()
    try:
        for name in SHAPES:
            K, N, wdtype, fac, act = SHAPE_DEFS[name]
            run_one(mesh, name, K, N, wdtype, fac, act)
    finally:
        ttnn.synchronize_device(mesh)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
