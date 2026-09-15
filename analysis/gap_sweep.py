# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Op-only sweeps to close the remaining single-chip SDPA validation gaps (TEN-4716):
head_dim variety (vision 72/96/160/256), batch>1 prefill, and attention sinks. Captures per-engine
counters (under tracy multipass) to validate/fix predict(). No torch golden.

Env: GAP_MODE=headdim|batch|sink ; GAP_NH, GAP_S, GAP_ITERS.
"""
from __future__ import annotations
import os
import pytest
import functools


def _pc_ck(device, qc=128):
    import ttnn

    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=qc,
        k_chunk_size=qc,
        exp_approx_mode=True,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    return pc, ck


@pytest.mark.parametrize("d", [int(x) for x in os.environ.get("GAP_HEADDIMS", "64,72,96,160,256").split(",")])
def test_gap_headdim(device, d):
    import ttnn
    from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand

    if os.environ.get("GAP_MODE") != "headdim":
        pytest.skip("GAP_MODE != headdim")
    nh = int(os.environ.get("GAP_NH", "16"))
    s = int(os.environ.get("GAP_S", "4096"))
    pc, ck = _pc_ck(device)
    Q = fa_rand(1, nh, s, d)
    K = fa_rand(1, nh, s, d)
    V = fa_rand(1, nh, s, d)
    f = functools.partial(ttnn.from_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tq, tk, tv = f(Q), f(K), f(V)
    print(f"\n[gap_headdim] S={s} d={d} nh={nh}", flush=True)
    for _ in range(int(os.environ.get("GAP_ITERS", "2"))):
        o = ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=True, program_config=pc, compute_kernel_config=ck
        )
        ttnn.synchronize_device(device)
        o.deallocate()
    for t in (tq, tk, tv):
        t.deallocate()
    print(f"[gap_headdim] OK d={d}", flush=True)


@pytest.mark.parametrize("b", [int(x) for x in os.environ.get("GAP_BATCHES", "1,2,4").split(",")])
def test_gap_batch(device, b):
    import ttnn
    from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand

    if os.environ.get("GAP_MODE") != "batch":
        pytest.skip("GAP_MODE != batch")
    nh = int(os.environ.get("GAP_NH", "16"))
    s = int(os.environ.get("GAP_S", "2048"))
    d = 128
    pc, ck = _pc_ck(device)
    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nh, s, d)
    V = fa_rand(b, nh, s, d)
    f = functools.partial(ttnn.from_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tq, tk, tv = f(Q), f(K), f(V)
    print(f"\n[gap_batch] b={b} S={s} nh={nh}", flush=True)
    for _ in range(int(os.environ.get("GAP_ITERS", "2"))):
        o = ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=True, program_config=pc, compute_kernel_config=ck
        )
        ttnn.synchronize_device(device)
        o.deallocate()
    for t in (tq, tk, tv):
        t.deallocate()
    print(f"[gap_batch] OK b={b}", flush=True)


@pytest.mark.parametrize("sink", [0, 1])
def test_gap_sink(device, sink):
    import torch, ttnn
    from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import fa_rand

    if os.environ.get("GAP_MODE") != "sink":
        pytest.skip("GAP_MODE != sink")
    nh = int(os.environ.get("GAP_NH", "16"))
    s = int(os.environ.get("GAP_S", "4096"))
    d = 128
    pc, ck = _pc_ck(device)
    Q = fa_rand(1, nh, s, d)
    K = fa_rand(1, nh, s, d)
    V = fa_rand(1, nh, s, d)
    f = functools.partial(ttnn.from_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tq, tk, tv = f(Q), f(K), f(V)
    tsink = None
    if sink:
        sink_t = torch.randn(1, nh, 1, 1)
        tsink = ttnn.from_torch(sink_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(f"\n[gap_sink] sink={sink} S={s} nh={nh}", flush=True)
    for _ in range(int(os.environ.get("GAP_ITERS", "2"))):
        o = ttnn.transformer.scaled_dot_product_attention(
            tq, tk, tv, is_causal=True, attention_sink=tsink, program_config=pc, compute_kernel_config=ck
        )
        ttnn.synchronize_device(device)
        o.deallocate()
    for t in (tq, tk, tv):
        t.deallocate()
    if tsink is not None:
        tsink.deallocate()
    print(f"[gap_sink] OK sink={sink}", flush=True)
