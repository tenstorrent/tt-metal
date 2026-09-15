# SPDX-License-Identifier: Apache-2.0
"""Op-only prefill WALL-CLOCK sweep (TEN-4716 #2: floor->wall-clock gap). Host-timed device latency
of the prefill SDPA op (no counters) so we can model the gap between the compute floor (MATH) and
actual wall-clock (front-end/dispatch/data-movement idle). Env: PF_SEQ, PF_NH, PF_NKV, PF_ITERS."""
import os, time, pytest


@pytest.mark.parametrize("s", [int(x) for x in os.environ.get("PF_SEQ", "2048,4096,8192,16384").split(",")])
def test_prefill_latency(device, s):
    import ttnn
    from tests.ttnn.unit_tests.operations.sdpa.test_sdpa_prefill import fa_rand

    nh = int(os.environ.get("PF_NH", "32"))
    nkv = int(os.environ.get("PF_NKV", "8"))
    d = 128
    causal = os.environ.get("PF_CAUSAL", "1") == "1"
    iters = int(os.environ.get("PF_ITERS", "50"))
    pc = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=True,
    )
    ck = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False
    )
    import functools

    f = functools.partial(ttnn.from_torch, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    q, k, v = f(fa_rand(1, nh, s, d)), f(fa_rand(1, nkv, s, d)), f(fa_rand(1, nkv, s, d))
    for _ in range(3):
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=causal, program_config=pc, compute_kernel_config=ck
        )
        o.deallocate()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        o = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=causal, program_config=pc, compute_kernel_config=ck
        )
        o.deallocate()
    ttnn.synchronize_device(device)
    lat_us = (time.perf_counter() - t0) / iters * 1e6
    print(f"[prefill_lat] S={s} nh={nh} nkv={nkv} wallclock_us={lat_us:.1f}", flush=True)
    for t in (q, k, v):
        t.deallocate()
