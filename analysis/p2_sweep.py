# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-only P2 sweeps for perf-counter capture (TEN-4716 P2 validation).

SLIDING-WINDOW prefill (is_causal + sliding_window_size, a causal band = Gemma/Mistral SWA) and
MASKED prefill (is_causal=False + dense attn_mask). Captures per-engine counters (run under tracy
multipass) so predict()'s saturating-window K_eff and mask-add pass can be validated against
measured MATH. No torch golden.

Env: P2_MODE=window|mask, P2_NH, P2_D, P2_SEQ (csv), P2_WIN (csv, window sizes), P2_QCHUNK, P2_ITERS.
"""
from __future__ import annotations
import os
import pytest


def _configs():
    seqs = [int(x) for x in os.environ.get("P2_SEQ", "8192,16384").split(",")]
    wins = [int(x) for x in os.environ.get("P2_WIN", "1024,4096").split(",")]
    return [(s, w) for s in seqs for w in wins if w <= s]


@pytest.mark.parametrize("s,w", _configs())
def test_p2_window(device, s, w):
    import torch
    import ttnn
    from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand

    if os.environ.get("P2_MODE", "window") != "window":
        pytest.skip("P2_MODE != window")
    nh = int(os.environ.get("P2_NH", "16"))
    d = int(os.environ.get("P2_D", "128"))
    qc = int(os.environ.get("P2_QCHUNK", "128"))
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=qc,
        k_chunk_size=qc,
        exp_approx_mode=True,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    Q = fa_rand(1, nh, s, d)
    K = fa_rand(1, nh, s, d)
    V = fa_rand(1, nh, s, d)
    tt_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    print(f"\n[p2_window] S={s} W={w} nh={nh} d={d}", flush=True)
    for _ in range(int(os.environ.get("P2_ITERS", "2"))):
        o = ttnn.transformer.scaled_dot_product_attention(
            tt_Q, tt_K, tt_V, is_causal=True, sliding_window_size=w, program_config=pc, compute_kernel_config=ck
        )
        ttnn.synchronize_device(device)
        o.deallocate()
    tt_Q.deallocate()
    tt_K.deallocate()
    tt_V.deallocate()
    print(f"[p2_window] OK S={s} W={w}", flush=True)


_MASK_CFGS = [
    (int(s), float(dn))
    for s in os.environ.get("P2_SEQ", "8192,16384").split(",")
    for dn in os.environ.get("P2_DENSITIES", "0.25").split(",")
]


@pytest.mark.parametrize("s,density", _MASK_CFGS)
def test_p2_mask(device, s, density):
    """Non-causal prefill with a dense additive mask at a given masked-fraction density -> lets us
    see whether measured MATH is density-dependent (mask-aware skipping) or a fixed masked-path cost."""
    import torch, ttnn
    from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand

    if os.environ.get("P2_MODE") != "mask":
        pytest.skip("P2_MODE != mask")
    nh = int(os.environ.get("P2_NH", "16"))
    d = int(os.environ.get("P2_D", "128"))
    qc = int(os.environ.get("P2_QCHUNK", "128"))
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=qc,
        k_chunk_size=qc,
        exp_approx_mode=True,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    Q = fa_rand(1, nh, s, d)
    K = fa_rand(1, nh, s, d)
    V = fa_rand(1, nh, s, d)
    mask = torch.bernoulli(torch.full((1, 1, s, s), density)) * -1e9
    tt_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)
    print(f"\n[p2_mask] S={s} density={density} nh={nh}", flush=True)
    for _ in range(int(os.environ.get("P2_ITERS", "2"))):
        o = ttnn.transformer.scaled_dot_product_attention(
            tt_Q, tt_K, tt_V, is_causal=False, attn_mask=tt_mask, program_config=pc, compute_kernel_config=ck
        )
        ttnn.synchronize_device(device)
        o.deallocate()
    for t in (tt_Q, tt_K, tt_V, tt_mask):
        t.deallocate()
    print(f"[p2_mask] OK S={s} density={density}", flush=True)


@pytest.mark.parametrize("sq,sk", [(1024, 4096), (2048, 8192)])
def test_p2_cross(device, sq, sk):
    """Non-causal cross-attention (kv_seq != q_seq, no mask) -> validates the kv_seq generalization."""
    import ttnn
    from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand

    if os.environ.get("P2_MODE") != "cross":
        pytest.skip("P2_MODE != cross")
    nh = int(os.environ.get("P2_NH", "16"))
    d = int(os.environ.get("P2_D", "128"))
    qc = int(os.environ.get("P2_QCHUNK", "128"))
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=qc,
        k_chunk_size=qc,
        exp_approx_mode=True,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    Q = fa_rand(1, nh, sq, d)
    K = fa_rand(1, nh, sk, d)
    V = fa_rand(1, nh, sk, d)
    tt_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    print(f"\n[p2_cross] Sq={sq} Sk={sk} nh={nh}", flush=True)
    for _ in range(int(os.environ.get("P2_ITERS", "2"))):
        o = ttnn.transformer.scaled_dot_product_attention(
            tt_Q, tt_K, tt_V, is_causal=False, program_config=pc, compute_kernel_config=ck
        )
        ttnn.synchronize_device(device)
        o.deallocate()
    for t in (tt_Q, tt_K, tt_V):
        t.deallocate()
    print(f"[p2_cross] OK Sq={sq} Sk={sk}", flush=True)
