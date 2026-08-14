# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stage-04 micro-probe: the two replicated blocks the decode profile is capped by.

The stage-03 decode layer (``../multichip_decoder/ops_perf_multichip_decode.csv.gz``,
device 0, rows 134-197, 414.66 us) spends

    row 134   LayerNorm   20.081 us   on **1 core**
    row 159   LayerNorm   20.127 us   on **1 core**
    row 160   Matmul      24.916 us   on **4 cores**   (the router projection)

i.e. 65.1 us -- 15.7% of the layer -- in three ops that between them use six
cores of 110. All three are *replicated* work, so they are also 65.1 us of the
129.09 us that caps decode at 3.97x.

This probe prices the alternatives standalone, by trace slope (median-of-30
blocking replay of a 33-op trace minus a 1-op trace, over 32), which removes the
host-dispatch floor. Shapes are the shipped decode shapes: [1,1,32,2048] bf16
activation (one logical row padded to a tile), [2048,128] bf16 router weight,
fp32 logits out.

    python norm_router_probe.py

Prints ``P|`` lines only.
"""
import statistics
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")

REPS = 32
HIDDEN = 2048
EXPERTS = 128
ROWS = 32
EPS = 1e-6


def bank_row(n):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))})


def grid_set(cores):
    """A rectangular CoreRangeSet holding ``cores`` cores, one row per 8."""
    if cores <= 8:
        return bank_row(cores), (cores, 1)
    rows = cores // 8
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, rows - 1))}), (8, rows)


def width_sharded(dim, cores):
    crs, _ = grid_set(cores)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, [ROWS, dim // cores], ttnn.ShardOrientation.ROW_MAJOR),
    )


def norm_pc(dim, cores):
    _, (gx, gy) = grid_set(cores)
    block_w = dim // cores // 32
    subblock_w = next(w for w in (4, 3, 2, 1) if block_w % w == 0)
    return ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=[gx, gy],
        subblock_w=subblock_w,
        block_h=1,
        block_w=block_w,
        inplace=False,
    )


mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=60_000_000, l1_small_size=32768)


def slope(fn):
    """us per op, trace slope over REPS."""
    out = []

    def build(n):
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(n):
            r = fn()
            out.append(r)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for _ in range(5):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        s = []
        for _ in range(30):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            s.append((time.perf_counter() - t0) * 1e6)
        ttnn.release_trace(mesh, tid)
        return statistics.median(s)

    fn()  # program-cache warm
    ttnn.synchronize_device(mesh)
    long = build(REPS + 1)
    short = build(1)
    return (long - short) / REPS


def rep(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mc=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        t, dtype=dtype, layout=layout, device=mesh, memory_config=mc, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


try:
    torch.manual_seed(0)
    x_t = torch.randn(1, 1, 1, HIDDEN) * 0.5
    w_t = torch.ones(1, 1, 1, HIDDEN)
    r_t = torch.randn(HIDDEN, EXPERTS) * 0.02

    x = rep(x_t)
    w_tile = rep(w_t)
    w_rm = rep(w_t.reshape(1, 1, HIDDEN // 32, 32), layout=ttnn.ROW_MAJOR_LAYOUT)
    wr = rep(r_t.reshape(1, 1, HIDDEN, EXPERTS))

    # ---- reference: today's interleaved rms_norm ----------------------------
    ref_out = ttnn.rms_norm(x, weight=w_tile, epsilon=EPS)
    ref = ttnn.to_torch(ref_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0:1].float()
    print(
        f"P|norm interleaved (shipped)            {slope(lambda: ttnn.rms_norm(x, weight=w_tile, epsilon=EPS)):8.2f} us",
        flush=True,
    )

    for cores in (8, 16, 32, 64):
        try:
            mc = width_sharded(HIDDEN, cores)
            pc = norm_pc(HIDDEN, cores)
            xs = ttnn.to_memory_config(x, mc)

            def leg(xs=xs, mc=mc, pc=pc):
                return ttnn.rms_norm(xs, weight=w_rm, epsilon=EPS, program_config=pc, memory_config=mc)

            o = leg()
            got = ttnn.to_torch(ttnn.sharded_to_interleaved(o), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[
                0:1
            ].float()
            d = (got - ref).abs().max().item()
            t_norm = slope(leg)
            t_i2s = slope(lambda mc=mc: ttnn.to_memory_config(x, mc))
            t_s2i = slope(lambda o=o: ttnn.sharded_to_interleaved(o))
            print(
                f"P|norm sharded {cores:2d} cores               {t_norm:8.2f} us  "
                f"(+i2s {t_i2s:.2f} +s2i {t_s2i:.2f} = {t_norm + t_i2s + t_s2i:.2f})  max|diff| {d:.3e}",
                flush=True,
            )
        except Exception as exc:
            print(f"P|norm sharded {cores:2d} cores               FAILED {str(exc)[:120]}", flush=True)

    # ---- router matmul ------------------------------------------------------
    normed = ref_out
    ref_log = ttnn.to_torch(
        ttnn.linear(normed, wr, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
    )[0:1].float()
    print(
        f"P|router matmul interleaved (shipped)   "
        f"{slope(lambda: ttnn.linear(normed, wr, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)):8.2f} us",
        flush=True,
    )

    # (b) width-sharded in0, interleaved weight, plain matmul
    for cores in (4, 8):
        try:
            in_mc = width_sharded(HIDDEN, cores)
            ns = ttnn.to_memory_config(normed, in_mc)
            t = slope(lambda ns=ns: ttnn.linear(ns, wr, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG))
            got = ttnn.to_torch(
                ttnn.linear(ns, wr, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0),
            )[0:1].float()
            print(
                f"P|router matmul in0 L1 wsh {cores:2d} cores    {t:8.2f} us  max|diff| "
                f"{(got - ref_log).abs().max().item():.3e}",
                flush=True,
            )
        except Exception as exc:
            print(f"P|router matmul in0 L1 wsh {cores:2d} cores    FAILED {str(exc)[:120]}", flush=True)

    # (c) DRAM-sharded, N padded 128 -> 256 so both dims are bank-divisible
    for out_dtype, tag in ((ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")):
        for wdt, wtag in ((ttnn.bfloat16, "bf16 w"), (ttnn.bfloat8_b, "bfp8 w")):
            try:
                NPAD = 256
                r_pad = torch.zeros(HIDDEN, NPAD)
                r_pad[:, :EXPERTS] = r_t
                wr_ds = ttnn.from_torch(
                    r_pad.reshape(1, 1, HIDDEN, NPAD),
                    dtype=wdt,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.DRAM,
                        ttnn.ShardSpec(bank_row(8), [HIDDEN, NPAD // 8], ttnn.ShardOrientation.ROW_MAJOR),
                    ),
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=HIDDEN // 8 // 32, per_core_M=1, per_core_N=NPAD // 8 // 32, fused_activation=None
                )
                in_mc = width_sharded(HIDDEN, 8)
                out_mc = width_sharded(NPAD, 8)
                ns = ttnn.to_memory_config(normed, in_mc)

                def leg(ns=ns, wr_ds=wr_ds, pc=pc, out_mc=out_mc, out_dtype=out_dtype):
                    return ttnn.linear(ns, wr_ds, program_config=pc, memory_config=out_mc, dtype=out_dtype)

                o = leg()
                got = ttnn.to_torch(ttnn.sharded_to_interleaved(o), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[
                    0:1, :, :, :EXPERTS
                ].float()
                t = slope(leg)
                t_s2i = slope(lambda o=o: ttnn.sharded_to_interleaved(o))
                print(
                    f"P|router matmul DRAM-sharded N=256 {tag} {wtag}  {t:8.2f} us (+s2i {t_s2i:.2f})  "
                    f"max|diff| {(got - ref_log).abs().max().item():.3e}",
                    flush=True,
                )
            except Exception as exc:
                print(f"P|router matmul DRAM-sharded {tag} {wtag} FAILED {str(exc)[:150]}", flush=True)

    # ---- topk, for the record ----------------------------------------------
    logits = ttnn.linear(normed, wr, dtype=ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for k, srt in ((8, True), (8, False)):
        try:
            t = slope(lambda k=k, srt=srt: ttnn.topk(logits, k=k, dim=-1, largest=True, sorted=srt))
            print(f"P|topk k={k} sorted={srt}                {t:8.2f} us", flush=True)
        except Exception as exc:
            print(f"P|topk k={k} sorted={srt} FAILED {str(exc)[:120]}", flush=True)
    lg16 = ttnn.typecast(logits, ttnn.bfloat16)
    try:
        print(
            f"P|topk bf16 input                       {slope(lambda: ttnn.topk(lg16, k=8, dim=-1, largest=True, sorted=True)):8.2f} us",
            flush=True,
        )
    except Exception as exc:
        print(f"P|topk bf16 FAILED {str(exc)[:120]}", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
print("P|done")
