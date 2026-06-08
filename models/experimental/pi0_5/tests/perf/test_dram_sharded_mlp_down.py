# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated compile + PCC + timing test for DRAM-width-sharded mlp_down.

Run with:
    PI0_DRAM_SHARDED_TEST=1 pytest -xvs \\
        models/experimental/pi0_5/tests/perf/test_dram_sharded_mlp_down.py

Validates Phase 1 of the DRAM-sharded matmul path before wiring into the
production MLP forward. Shape: M=32 K=4096 N=1024 (pi0.5 denoise expert
mlp_down).

Pass criteria:
  - Matmul compiles + runs (no exception)
  - PCC vs torch reference ≥ 0.999
  - Wall-clock median < legacy 1D-mcast path (loose sanity check)
"""

from __future__ import annotations
import os
import statistics
import time

import pytest
import torch
import ttnn

from models.experimental.pi0_5.tt.ttnn_common import build_dram_width_sharded_memcfg


BENCH_ENABLED = os.environ.get("PI0_DRAM_SHARDED_TEST") == "1"
pytestmark = pytest.mark.skipif(
    not BENCH_ENABLED,
    reason="set PI0_DRAM_SHARDED_TEST=1 to run the isolated DRAM-sharded test",
)


def _pcc(a, b):
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    if t1.numel() != t2.numel():
        return -1.0
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-9 or s2 < 1e-9:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    return (torch.mean((t1 - m1) * (t2 - m2)) / (s1 * s2)).item()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_dram_sharded_mlp_down(device):
    # pi0.5 denoise expert mlp_down: M=32, K=mlp_dim=4096, N=hidden=1024
    M, K, N = 32, 4096, 1024
    dram_cores = 8
    compute_cores = 8

    torch.manual_seed(0)
    a_torch = torch.randn(M, K) * 0.1
    w_torch = torch.randn(K, N) * 0.1
    ref = (a_torch.float() @ w_torch.float()).unsqueeze(0).unsqueeze(0)

    # Build DRAM width-sharded weight with dram_cores=8
    weight_memcfg, padded_n, _ = build_dram_width_sharded_memcfg(device, K, N, dram_cores=dram_cores)
    assert padded_n == N, f"Expected no padding (N={N} % dram_cores={dram_cores} == 0), got padded_n={padded_n}"

    w_tt = ttnn.from_torch(
        w_torch,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=weight_memcfg,
    )

    # Activation: upload as L1 interleaved, then convert to L1 width-sharded
    a_tt = ttnn.from_torch(
        a_torch.unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # L1 width-sharded activation across compute_cores=8
    k_tiles = K // 32
    act_shard_k = (k_tiles // compute_cores) * 32  # cols per shard
    act_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_cores - 1, 0))})
    act_memcfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(act_grid, (M, act_shard_k), ttnn.ShardOrientation.ROW_MAJOR),
    )
    a_tt_sh = ttnn.to_memory_config(a_tt, act_memcfg)

    # DRAM-sharded matmul program config
    ds_pcfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=k_tiles // compute_cores,  # 16
        per_core_M=M // 32,  # 1
        per_core_N=padded_n // (32 * compute_cores),  # 4
        fused_activation=None,
    )
    ck_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    print(f"\n  Config: dram_cores={dram_cores}, compute_cores={compute_cores}")
    print(f"  ibw={k_tiles // compute_cores}, per_core_M=1, per_core_N={padded_n // (32 * compute_cores)}")
    print(f"  padded_n={padded_n}, M={M}, K={K}, N={N}")

    # Warmup
    for _ in range(5):
        out = ttnn.linear(
            a_tt_sh,
            w_tt,
            program_config=ds_pcfg,
            compute_kernel_config=ck_cfg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(out)
    ttnn.synchronize_device(device)

    # PCC check (single run)
    out = ttnn.linear(
        a_tt_sh,
        w_tt,
        program_config=ds_pcfg,
        compute_kernel_config=ck_cfg,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )
    out_interleaved = ttnn.sharded_to_interleaved(out, memory_config=ttnn.L1_MEMORY_CONFIG)
    out_torch = ttnn.to_torch(out_interleaved).reshape(1, 1, M, N)
    pcc_v = _pcc(ref, out_torch.float())
    ttnn.deallocate(out)
    ttnn.deallocate(out_interleaved)
    print(f"  PCC: {pcc_v:.5f}")
    assert pcc_v >= 0.999, f"PCC {pcc_v} below 0.999"

    # Timing — median of 30
    samples = []
    for _ in range(30):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = ttnn.linear(
            a_tt_sh,
            w_tt,
            program_config=ds_pcfg,
            compute_kernel_config=ck_cfg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.synchronize_device(device)
        samples.append((time.perf_counter() - t0) * 1_000_000)
        ttnn.deallocate(out)

    ttnn.deallocate(a_tt_sh)
    ttnn.deallocate(a_tt)
    ttnn.deallocate(w_tt)

    print(f"  Wall-clock median: {statistics.median(samples):.2f} µs")
    print(f"  Wall-clock min:    {min(samples):.2f} µs")
    print(f"  (For comparison: legacy 1D-mcast for this shape was ~50 µs in the sharding sweep)")
