# DP4 x TP8 residual add + RMSNorm over full 8192-wide rows (R = M/8 rows per chip), one Blackhole chip: today's
# add + 5-op norm (mul, ones-column matmul, add eps, rsqrt, mul) vs the distributed-norm pair used on one chip
# (rms_norm_pre_all_gather -> per-row sum of squares, rms_norm_post_all_gather -> scale), with and without the residual
# add folded into the pre op. Times and PCC vs torch.
#   pytest models/demos/llama3_70b_galaxy/tests/test_dp4_norm_fast.py -s        env: NF_R (512,2048)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H = 8192
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def _time(device, fn, n=20):
    for z in fn():
        ttnn.deallocate(z)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(n):
        for z in fn():
            ttnn.deallocate(z)
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / n * 1e3


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_dp4_norm_fast(device):
    for R in [int(r) for r in (os.environ.get("NF_R") or "512,2048").split(",")]:
        torch.manual_seed(0)
        x = torch.randn(1, 1, R, H)
        x[..., 17] *= 100
        r = torch.randn(1, 1, R, H)
        s = x + r
        gold = s / torch.sqrt(s.pow(2).mean(-1, keepdim=True) + 1e-5)
        tx = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM)
        tr = ttnn.from_torch(r, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM)
        ones = ttnn.from_torch(
            torch.full((1, 1, H, 1), 1.0 / H), device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
        )
        ck = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )
        eps = 1e-5

        def today():
            s1 = ttnn.add(tx, tr, dtype=ttnn.bfloat16, memory_config=DRAM)
            sq = ttnn.mul(s1, s1, dtype=ttnn.bfloat16, memory_config=DRAM)
            ms = ttnn.matmul(sq, ones, compute_kernel_config=ck, dtype=ttnn.bfloat16, memory_config=DRAM)
            ttnn.deallocate(sq)
            me = ttnn.add(ms, eps, memory_config=DRAM)
            ttnn.deallocate(ms)
            rr = ttnn.rsqrt(me, memory_config=DRAM)
            ttnn.deallocate(me)
            o = ttnn.mul(s1, rr, dtype=ttnn.bfloat8_b, memory_config=DRAM)
            ttnn.deallocate(rr)
            return [s1, o]

        def dist():
            s1 = ttnn.add(tx, tr, dtype=ttnn.bfloat16, memory_config=DRAM)
            st = ttnn.rms_norm_pre_all_gather(s1, dtype=ttnn.bfloat16, compute_kernel_config=ck, memory_config=DRAM)
            o = ttnn.rms_norm_post_all_gather(
                s1, st, epsilon=eps, dtype=ttnn.bfloat8_b, memory_config=DRAM, compute_kernel_config=ck
            )
            ttnn.deallocate(st)
            return [s1, o]

        def dist_res():
            # residual folded into the stats pass; the scale pass still needs the sum, so it is formed once here too
            st = ttnn.rms_norm_pre_all_gather(
                tx, residual_input_tensor=tr, dtype=ttnn.bfloat16, compute_kernel_config=ck, memory_config=DRAM
            )
            s1 = ttnn.add(tx, tr, dtype=ttnn.bfloat16, memory_config=DRAM)
            o = ttnn.rms_norm_post_all_gather(
                s1, st, epsilon=eps, dtype=ttnn.bfloat8_b, memory_config=DRAM, compute_kernel_config=ck
            )
            ttnn.deallocate(st)
            return [s1, o]

        def rms(welford):
            cfg = ttnn.LayerNormDefaultProgramConfig(use_welford=welford)

            def f():
                s1 = ttnn.add(tx, tr, dtype=ttnn.bfloat16, memory_config=DRAM)
                n = ttnn.rms_norm(s1, epsilon=eps, memory_config=DRAM, program_config=cfg, compute_kernel_config=ck)
                o = ttnn.typecast(n, ttnn.bfloat8_b)
                ttnn.deallocate(n)
                return [s1, o]

            return f

        def rms_bf8():
            s1 = ttnn.add(tx, tr, dtype=ttnn.bfloat16, memory_config=DRAM)
            o = ttnn.rms_norm(s1, epsilon=eps, memory_config=DRAM, compute_kernel_config=ck, dtype=ttnn.bfloat8_b)
            return [s1, o]

        def rms_res():
            # add folded into the norm; the residual sum is still needed downstream, so it is formed separately
            s1 = ttnn.add(tx, tr, dtype=ttnn.bfloat16, memory_config=DRAM)
            n = ttnn.rms_norm(tx, epsilon=eps, residual_input_tensor=tr, memory_config=DRAM, compute_kernel_config=ck)
            o = ttnn.typecast(n, ttnn.bfloat8_b)
            ttnn.deallocate(n)
            return [s1, o]

        def add_only():
            return [ttnn.add(tx, tr, dtype=ttnn.bfloat16, memory_config=DRAM)]

        res = {}
        for name, fn in (
            ("add only", add_only),
            ("today add+5op", today),
            ("add+rms_norm+typecast", rms(False)),
            ("add+rms_norm(dtype bf8)", rms_bf8),
            ("add | rms_norm(res)+typecast", rms_res),
        ):
            try:
                ms = _time(device, fn)
                outs = fn()
                p = _pcc(gold, ttnn.to_torch(outs[-1])) if len(outs) > 1 else float("nan")
                for z in outs:
                    ttnn.deallocate(z)
                res[name] = ms
                logger.info(f"NORMFAST R={R} {name}: {ms:.3f} ms pcc {p:.5f}")
            except Exception as e:
                logger.warning(f"NORMFAST R={R} {name} ERROR {str(e)[:300]}")
        logger.info(f"NORMFAST R={R} SUMMARY " + " ".join(f"[{k}]={v:.3f}" for k, v in res.items()))
