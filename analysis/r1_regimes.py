# SPDX-License-Identifier: Apache-2.0
"""R1 regime re-measurement harness: cross, windowed and dense-mask prefill, env-driven shapes.

Forms copied exactly from analysis/p2_sweep.py (the July calibration harness): bfp8_b Q/K/V, nh = nkv (no GQA),
q_chunk = k_chunk = R1_QCHUNK (128), exp_approx_mode True in the program config, full grid; compute config HiFi2,
fp32_dest_acc_en False, packer_l1_acc False, math_approx_mode False for cross and mask, True for window (p2 forms).
Mask: bfloat4_b additive mask, bernoulli(density) * -1e9 (p2_mask form).
Env: R1_MODE in {cross, window, mask}; cross: R1_SQ, R1_SK; window: R1_S, R1_W; mask: R1_S, R1_DENSITY;
common: R1_NH (16), R1_D (128), R1_QCHUNK (128), R1_ITERS (3).
"""
import os

import pytest


def _common(device):
    import ttnn

    nh = int(os.environ.get("R1_NH", "16"))
    d = int(os.environ.get("R1_D", "128"))
    qc = int(os.environ.get("R1_QCHUNK", "128"))
    iters = int(os.environ.get("R1_ITERS", "3"))
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                                q_chunk_size=qc, k_chunk_size=qc, exp_approx_mode=True)
    return nh, d, qc, iters, pc


def test_r1_regime(device):
    import torch
    import ttnn
    from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand

    mode = os.environ.get("R1_MODE", "cross")
    nh, d, qc, iters, pc = _common(device)
    kw = dict(dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    if mode == "cross":
        sq = int(os.environ["R1_SQ"]); sk = int(os.environ["R1_SK"])
        ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                              fp32_dest_acc_en=False, packer_l1_acc=False)
        Q = fa_rand(1, nh, sq, d); K = fa_rand(1, nh, sk, d); V = fa_rand(1, nh, sk, d)
        tt_Q, tt_K, tt_V = (ttnn.from_torch(t, **kw) for t in (Q, K, V))
        print(f"\n[r1_cross] Sq={sq} Sk={sk} nh={nh} d={d} qc={qc} iters={iters}", flush=True)
        for it in range(iters):
            o = ttnn.transformer.scaled_dot_product_attention(tt_Q, tt_K, tt_V, is_causal=False, program_config=pc, compute_kernel_config=ck)
            ttnn.synchronize_device(device); o.deallocate(); print(f"[r1_cross] iter {it} done", flush=True)
        for t in (tt_Q, tt_K, tt_V):
            t.deallocate()
    elif mode == "window":
        s = int(os.environ["R1_S"]); w = int(os.environ["R1_W"])
        ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
                                              fp32_dest_acc_en=False, packer_l1_acc=False)
        Q = fa_rand(1, nh, s, d); K = fa_rand(1, nh, s, d); V = fa_rand(1, nh, s, d)
        tt_Q, tt_K, tt_V = (ttnn.from_torch(t, **kw) for t in (Q, K, V))
        print(f"\n[r1_window] S={s} W={w} nh={nh} d={d} qc={qc} iters={iters}", flush=True)
        for it in range(iters):
            o = ttnn.transformer.scaled_dot_product_attention(tt_Q, tt_K, tt_V, is_causal=True, sliding_window_size=w,
                                                              program_config=pc, compute_kernel_config=ck)
            ttnn.synchronize_device(device); o.deallocate(); print(f"[r1_window] iter {it} done", flush=True)
        for t in (tt_Q, tt_K, tt_V):
            t.deallocate()
    elif mode == "mask":
        s = int(os.environ["R1_S"]); density = float(os.environ.get("R1_DENSITY", "0.25"))
        ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False,
                                              fp32_dest_acc_en=False, packer_l1_acc=False)
        Q = fa_rand(1, nh, s, d); K = fa_rand(1, nh, s, d); V = fa_rand(1, nh, s, d)
        mask = (torch.bernoulli(torch.full((1, 1, s, s), density)) * -1e9)
        tt_Q, tt_K, tt_V = (ttnn.from_torch(t, **kw) for t in (Q, K, V))
        tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
        print(f"\n[r1_mask] S={s} density={density} nh={nh} d={d} qc={qc} iters={iters}", flush=True)
        for it in range(iters):
            o = ttnn.transformer.scaled_dot_product_attention(tt_Q, tt_K, tt_V, is_causal=False, attn_mask=tt_mask,
                                                              program_config=pc, compute_kernel_config=ck)
            ttnn.synchronize_device(device); o.deallocate(); print(f"[r1_mask] iter {it} done", flush=True)
        for t in (tt_Q, tt_K, tt_V, tt_mask):
            t.deallocate()
    else:
        pytest.skip(f"unknown R1_MODE {mode}")
