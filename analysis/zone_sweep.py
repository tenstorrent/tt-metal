# SPDX-License-Identifier: Apache-2.0
"""SDPA zone-decomposition harness (revamp T0.1 / T2.1 / T2.2).

Runs the single-chip SDPA prefill op SDPA_ITERS times (default 3) in ONE process so the
device profiler CSV carries one run host ID per invocation. Mirrors analysis/sdpa_sweep.py
`_run_sdpa_op_only` exactly (streaming compute path: fp32_dest_acc_en=False, HiFi2, bfp8_b,
exp_approx, full grid) with k_chunk pinned explicitly via SDPA_KCHUNK.

Env knobs (all optional): SDPA_SEQ (default 4096), SDPA_NH 32, SDPA_NKV 8, SDPA_HEAD_DIM 128,
SDPA_QCHUNK 128, SDPA_KCHUNK (default = q_chunk), SDPA_CAUSAL 1, SDPA_DTYPE bfp8_b,
SDPA_FIDELITY HiFi2, SDPA_EXP_APPROX 1, SDPA_ITERS 3.

Run:
  python -m tracy -m pytest analysis/zone_sweep.py::test_zone_sweep -s
"""
import os

import pytest


def test_zone_sweep(device):
    import ttnn
    from tests.ttnn.unit_tests.operations.sdpa.test_sdpa_prefill import fa_rand

    s = int(os.environ.get("SDPA_SEQ", "4096"))
    nh = int(os.environ.get("SDPA_NH", "32"))
    nkv = int(os.environ.get("SDPA_NKV", "8"))
    d = int(os.environ.get("SDPA_HEAD_DIM", "128"))
    qc = int(os.environ.get("SDPA_QCHUNK", "128"))
    kc = int(os.environ.get("SDPA_KCHUNK", str(qc)))
    iters = int(os.environ.get("SDPA_ITERS", "3"))
    causal = os.environ.get("SDPA_CAUSAL", "1") == "1"
    exp_approx = os.environ.get("SDPA_EXP_APPROX", "1") == "1"
    dmap = {"bfp8_b": ttnn.bfloat8_b, "bfloat16": ttnn.bfloat16}
    dtype = dmap[os.environ.get("SDPA_DTYPE", "bfp8_b")]
    kv_dtype = dmap[os.environ.get("SDPA_KV_DTYPE", os.environ.get("SDPA_DTYPE", "bfp8_b"))]  # K/V dtype override (DRAM-law runs)
    fidelity = {
        "LoFi": ttnn.MathFidelity.LoFi,
        "HiFi2": ttnn.MathFidelity.HiFi2,
        "HiFi3": ttnn.MathFidelity.HiFi3,
        "HiFi4": ttnn.MathFidelity.HiFi4,
    }[os.environ.get("SDPA_FIDELITY", "HiFi2")]
    grid = device.compute_with_storage_grid_size()
    if os.environ.get("SDPA_GRID"):  # e.g. 8x8: smaller compute grid (DRAM-law runs)
        gx, gy = [int(v) for v in os.environ["SDPA_GRID"].split("x")]
        grid = ttnn.CoreCoord(gx, gy)
    print(
        f"\n[zone_sweep] S={s} nh={nh} nkv={nkv} d={d} q_chunk={qc} k_chunk={kc} causal={causal} "
        f"dtype={os.environ.get('SDPA_DTYPE', 'bfp8_b')} kv_dtype={os.environ.get('SDPA_KV_DTYPE', '')} fidelity={os.environ.get('SDPA_FIDELITY', 'HiFi2')} "
        f"exp_approx={exp_approx} grid={grid.x}x{grid.y} iters={iters}",
        flush=True,
    )
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=qc,
        k_chunk_size=kc,
        exp_approx_mode=exp_approx,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    Q = fa_rand(1, nh, s, d)
    K = fa_rand(1, nkv, s, d)
    V = fa_rand(1, nkv, s, d)
    kw = dict(dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, device=device, pad_value=0.0)
    tt_Q = ttnn.from_torch(Q, **kw)
    kwkv = dict(kw, dtype=kv_dtype)
    tt_K = ttnn.from_torch(K, **kwkv)
    tt_V = ttnn.from_torch(V, **kwkv)
    for it in range(iters):
        tt_back = ttnn.transformer.scaled_dot_product_attention(
            tt_Q, tt_K, tt_V, is_causal=causal, program_config=program_config, compute_kernel_config=compute_kernel_config
        )
        ttnn.synchronize_device(device)
        _ = tt_back.shape
        tt_back.deallocate()
        print(f"[zone_sweep] iter {it} done", flush=True)
    tt_Q.deallocate()
    tt_K.deallocate()
    tt_V.deallocate()
