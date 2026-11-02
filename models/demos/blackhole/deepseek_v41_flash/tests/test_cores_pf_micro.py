# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip micro tests of the few-core prefill ops (cores-for-prefill study, dsv4-logs/cores/CORES_PF.md).
Sections (CMU_SECTIONS csv): norm (collapse rms_norm [1,1,128,5120] on 4 cores -> block-sharded norm), sum (mHC ttnn.sum over dim 1 of [4,64,32,32] fp32 vs fast_reduce_nc),
ropew (slice + slice_write in place instead of slice + concat in rope_pe), shared (shared_big linears M=128: 1D mcast configs on more cores)."""
import os
import time

import pytest
import torch

import ttnn

REPS = int(os.environ.get("CMU_REPS", "20"))


def bench(fn, dev):
    o = fn()
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(REPS):
        o = fn()
        ttnn.synchronize_device(dev) if False else None
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / REPS * 1e6, o


def pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def up(t, dev, dt=ttnn.bfloat16):
    return ttnn.from_torch(t, device=dev, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_cores_pf_micro(device):
    dev = device
    secs = os.environ.get("CMU_SECTIONS", "norm,sum,ropew,shared").split(",")
    g = torch.Generator().manual_seed(0)
    if "norm" in secs:
        C, N, eps = 5120, 128, 1e-6
        x = torch.randn(1, 1, N, C, generator=g)
        w = torch.rand(1, 1, 1, C, generator=g) + 0.5
        tx = up(x, dev)
        tw = up(w, dev, ttnn.float32)
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        us, ref = bench(lambda: ttnn.rms_norm(tx, epsilon=eps, weight=tw, compute_kernel_config=ckc), dev)
        refc = ttnn.to_torch(ref).float()
        print(f"CMU norm baseline interleaved {us:.1f} us", flush=True)
        wrm = ttnn.from_torch(
            w.reshape(1, 1, C // 32, 32),
            device=dev,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for gx, gy, orient in (
            (10, 4, "ROW"),
            (8, 4, "ROW"),
            (5, 4, "ROW"),
            (4, 10, "COL"),
            (4, 8, "COL"),
            (4, 5, "COL"),
        ):
            try:
                ncols = gx if orient == "ROW" else gy  # cores along the width
                bw = C // 32 // ncols
                mc = ttnn.create_sharded_memory_config(
                    shape=(32, C // ncols),
                    core_grid=ttnn.CoreGrid(x=gx, y=gy),
                    strategy=ttnn.ShardStrategy.BLOCK,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR if orient == "ROW" else ttnn.ShardOrientation.COL_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                sbw = max(d for d in range(1, 5) if bw % d == 0)
                pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=[gx, gy], subblock_w=sbw, block_h=1, block_w=bw, inplace=False
                )
                for wname, wt in (("tile32", tw), ("rm16", wrm)):

                    def fn():
                        xs = ttnn.to_memory_config(tx, mc)
                        o = ttnn.rms_norm(xs, epsilon=eps, weight=wt, program_config=pc, compute_kernel_config=ckc)
                        ttnn.deallocate(xs)
                        o2 = ttnn.sharded_to_interleaved(o, ttnn.DRAM_MEMORY_CONFIG)
                        ttnn.deallocate(o)
                        return o2

                    try:
                        us, o = bench(fn, dev)
                        d = (ttnn.to_torch(o).float() - refc).abs().max().item()
                        print(
                            f"CMU norm sharded {gx}x{gy} {orient} w={wname} {us:.1f} us (I2S+norm+S2I) maxdiff {d:.4f} pcc {pcc(ttnn.to_torch(o), refc):.6f}",
                            flush=True,
                        )
                    except Exception as e:  # noqa
                        print(f"CMU norm sharded {gx}x{gy} {orient} w={wname} FAIL {str(e)[:90]}", flush=True)
            except Exception as e:  # noqa
                print(f"CMU norm cfg {gx}x{gy} {orient} FAIL {str(e)[:90]}", flush=True)
    if "sum" in secs:
        p = torch.randn(4, 64, 32, 32, generator=g)
        tp = up(p, dev, ttnn.float32)
        us, ref = bench(lambda: ttnn.sum(tp, dim=1, keepdim=False), dev)
        refc = ttnn.to_torch(ref).float()
        print(f"CMU sum baseline ttnn.sum {us:.1f} us", flush=True)
        for nm, fn in (
            (
                "fast_reduce_nc",
                lambda: ttnn.experimental.fast_reduce_nc(tp, dims=[1], output=None, compute_kernel_config=None),
            ),
        ):
            try:
                us, o = bench(fn, dev)
                oo = ttnn.to_torch(o).float().reshape(refc.shape)
                print(f"CMU sum {nm} {us:.1f} us maxdiff {(oo - refc).abs().max().item():.5f}", flush=True)
            except Exception as e:  # noqa
                print(f"CMU sum {nm} FAIL {str(e)[:120]}", flush=True)
    if "ropew" in secs:
        for H in (8, 1):
            x = torch.randn(2, H, 512, 512, generator=g)
            c = torch.randn(2, 1, 512, 64, generator=g)
            s = torch.randn(2, 1, 512, 64, generator=g)
            Pm = torch.zeros(64, 64)
            for i in range(32):
                Pm[2 * i + 1, 2 * i] = -1.0
                Pm[2 * i, 2 * i + 1] = 1.0
            tx, tc, ts, tP = up(x, dev), up(c, dev), up(s, dev), up(Pm.reshape(1, 1, 64, 64), dev)
            ckc = ttnn.init_device_compute_kernel_config(
                dev.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )

            def rope_ref():
                sh = list(tx.shape)
                lo = 448
                xp = ttnn.slice(tx, [0, 0, 0, lo], sh)
                y = ttnn.add(ttnn.multiply(xp, tc), ttnn.multiply(ttnn.matmul(xp, tP, compute_kernel_config=ckc), ts))
                return ttnn.concat([ttnn.slice(tx, [0, 0, 0, 0], sh[:3] + [lo]), y], dim=3)

            def rope_w():
                sh = list(tx.shape)
                lo = 448
                xp = ttnn.slice(tx, [0, 0, 0, lo], sh)
                y = ttnn.add(ttnn.multiply(xp, tc), ttnn.multiply(ttnn.matmul(xp, tP, compute_kernel_config=ckc), ts))
                ttnn.experimental.slice_write(y, tx, [0, 0, 0, lo], sh, [1, 1, 1, 1])
                return tx

            us, ref = bench(rope_ref, dev)
            refc = ttnn.to_torch(ref).float()
            print(f"CMU ropew H={H} baseline slice+concat {us:.1f} us", flush=True)
            tx0 = tx
            # the in-place variant overwrites tx: compare after ONE application on a fresh copy
            tx = up(x, dev)
            try:
                us, _ = bench(rope_w, dev)
                tx2 = up(x, dev)
                tx = tx2
                rope_w()
                d = (ttnn.to_torch(tx2).float() - refc).abs().max().item()
                print(f"CMU ropew H={H} slice_write in place {us:.1f} us maxdiff(one application) {d:.5f}", flush=True)
            except Exception as e:  # noqa
                print(f"CMU ropew H={H} slice_write FAIL {str(e)[:150]}", flush=True)
    if "shared" in secs:
        M, K, F = 128, 5120, 2304
        x = torch.randn(1, 1, M, K, generator=g) * 0.5
        w01 = torch.randn(1, 1, K, 2 * F, generator=g) * 0.03
        w2 = torch.randn(1, 1, F, K, generator=g) * 0.03
        tx = up(x, dev)
        t01 = up(w01, dev, ttnn.bfloat8_b)
        t2 = up(w2, dev, ttnn.bfloat8_b)
        ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        us0, r1 = bench(
            lambda: ttnn.linear(
                tx,
                t01,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ckc,
                core_grid=ttnn.CoreGrid(y=8, x=8),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            dev,
        )
        r1c = ttnn.to_torch(r1).float()
        act = up(torch.randn(1, 1, M, F, generator=g), dev)
        us1, r2 = bench(
            lambda: ttnn.linear(
                act,
                t2,
                dtype=ttnn.float32,
                compute_kernel_config=ckc,
                core_grid=ttnn.CoreGrid(y=8, x=8),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
            dev,
        )
        r2c = ttnn.to_torch(r2).float()
        print(f"CMU shared baseline w01 {us0:.1f} us, w2 {us1:.1f} us", flush=True)
        for nm, a, wt, dt, refc, Kd, Nd in (
            ("w01", tx, t01, ttnn.bfloat16, r1c, K, 2 * F),
            ("w2", act, t2, ttnn.float32, r2c, F, K),
        ):
            Kt, Nt = Kd // 32, Nd // 32
            for pcn in (1, 2, 3, 4):
                n = -(-Nt // pcn)
                if n > 120:
                    continue
                for gxm in (12, 10, 8):
                    gy = -(-n // gxm)
                    if gy > 10:
                        continue
                    for ibw in (4, 8, 16):
                        if Kt % ibw:
                            continue
                        for sbh, sbw in ((1, min(pcn, 4)), (2, 2), (4, 1)):
                            if pcn % sbw or 4 % sbh or sbh * sbw > 4:
                                continue
                            try:
                                pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                                    compute_with_storage_grid_size=ttnn.CoreCoord(gxm, gy),
                                    in0_block_w=ibw,
                                    out_subblock_h=sbh,
                                    out_subblock_w=sbw,
                                    per_core_M=4,
                                    per_core_N=pcn,
                                    fuse_batch=True,
                                    fused_activation=None,
                                    mcast_in0=True,
                                )
                                us, o = bench(
                                    lambda: ttnn.linear(
                                        a,
                                        wt,
                                        dtype=dt,
                                        compute_kernel_config=ckc,
                                        program_config=pc,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                    ),
                                    dev,
                                )
                                d = (ttnn.to_torch(o).float() - refc).abs().max().item()
                                print(
                                    f"CMU shared {nm} {us:7.1f} us cores {n} grid {gxm}x{gy} pcn {pcn} ibw {ibw} sub {sbh}x{sbw} maxdiff {d:.4f}",
                                    flush=True,
                                )
                            except Exception as e:  # noqa
                                pass
