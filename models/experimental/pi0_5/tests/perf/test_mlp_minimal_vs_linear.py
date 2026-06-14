# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MLP matmul sweep: ttnn.experimental.minimal_matmul vs traditional ttnn.linear.

Automates the manual A/B I ran for the pi0.5 MLP matmuls. Answers the team's
question: for the SigLIP and VLM-prefill MLP shapes, which kernel wins
(minimal_matmul streaming-K vs the standard 2D/BS linear), and does it hold on
1 chip vs 4 chips.

Covers both MLP matmuls of each model:
  SigLIP MLP  (M=768 @ bs=3):  fc1 1152->4608 (+GELU),  fc2 4608->1152
  VLM prefill (M=768):         gate/up 2048->16384 (+GELU), down 16384->2048

For each (shape × device × kernel × config) it reports device-synced latency
(µs) and PCC vs the torch reference, then prints a ranked sweep table per shape.

Background (what the manual runs found — this test re-verifies it):
  * minimal_matmul is a streaming-K *CB-budget escape hatch*, not a faster
    matmul. It wins when 2D-mcast can't fit its CB (large N); it loses on
    shapes that fit a 2D/BS linear. See PERF_PLAYBOOKS/05_MLP.md sec 4.
  * SigLIP MLP end-to-end prefers the BS linear (reshard-avoidance); VLM MLP
    runs interleaved so minimal_matmul fits naturally.

Run (opt-in, like the other perf sweeps):
    PI0_MLP_SWEEP=1 pytest -xvs \\
      models/experimental/pi0_5/tests/perf/test_mlp_minimal_vs_linear.py

Pin one shape / device:
    PI0_MLP_SWEEP_SHAPE=siglip_fc1   PI0_MLP_SWEEP_DEVICE=4chip
"""

from __future__ import annotations

import os
import statistics
import time
from typing import List, Tuple

import pytest
import torch
import ttnn

BENCH_ENABLED = os.environ.get("PI0_MLP_SWEEP", "").lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not BENCH_ENABLED, reason="set PI0_MLP_SWEEP=1 to run the MLP minimal-vs-linear sweep")

NUM_WARMUP = int(os.environ.get("PI0_MLP_SWEEP_WARMUP", "10"))
NUM_ITER = int(os.environ.get("PI0_MLP_SWEEP_ITER", "40"))
PCC_GATE = float(os.environ.get("PI0_MLP_SWEEP_PCC", "0.99"))

# --- shapes: (label, M, K, N, activation) -------------------------------------
# M=768 = bs=3 * seq=256 for SigLIP; VLM prefix at bs=2 single-pass is also 768.
SHAPES = {
    "siglip_fc1": (768, 1152, 4608, "gelu"),
    "siglip_fc2": (768, 4608, 1152, None),
    "vlm_gate_up": (768, 2048, 16384, "gelu"),
    "vlm_down": (768, 16384, 2048, None),
}

# --- minimal_matmul block configs to sweep (the range I explored manually) ----
# (M_block, K_block, N_block, subblock_h, subblock_w)
MINIMAL_CONFIGS = [
    (8, 8, 8, 4, 2),  # balanced (the manual default)
    (8, 8, 8, 8, 1),  # tall subblock (unlocked by fp32_dest=False) — playbook FF1
    (8, 4, 8, 4, 2),  # smaller K stream
    (16, 8, 8, 4, 2),  # bigger M block
]

# --- traditional linear configs to sweep --------------------------------------
# (grid_x, grid_y, dst_budget) — 2D mcast linear; in0_block_w auto from K.
LINEAR_CONFIGS = [
    (12, 10, 4),  # full BH grid
    (8, 8, 4),  # smaller grid, bigger per-core
    (12, 8, 4),
]

MATH_FIDELITY = ttnn.MathFidelity.LoFi  # MLP matmuls run LoFi (bf8 weights)


def _from_torch(t, device, dtype, mem_cfg, layout=ttnn.TILE_LAYOUT):
    """from_torch that replicates across a multi-device mesh when needed."""
    mapper = None
    try:
        if device.get_num_devices() > 1:
            mapper = ttnn.ReplicateTensorToMesh(device)
    except Exception:
        pass
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=mem_cfg, mesh_mapper=mapper)


def _to_torch_first(out, device, M, N):
    """to_torch returning a single device's logical tensor (replicated mesh =>
    every shard is identical, so device 0 is representative)."""
    try:
        if device.get_num_devices() > 1:
            full = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
            return full.reshape(-1, M, N)[0]
    except Exception:
        pass
    return ttnn.to_torch(out).reshape(M, N)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1, t2 = a.flatten().float(), b.flatten().float()
    if t1.numel() != t2.numel():
        return -1.0
    m1, m2 = t1.mean(), t2.mean()
    s1, s2 = t1.std(), t2.std()
    if s1 < 1e-9 or s2 < 1e-9:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    return (((t1 - m1) * (t2 - m2)).mean() / (s1 * s2)).item()


def _gelu(x):
    return torch.nn.functional.gelu(x)


def _time(fn, sync) -> Tuple[float, float]:
    for _ in range(NUM_WARMUP):
        out = fn()
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
    ttnn.synchronize_device(sync)
    samples: List[float] = []
    for _ in range(NUM_ITER):
        ttnn.synchronize_device(sync)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(sync)
        samples.append((time.perf_counter() - t0) * 1e6)  # µs
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
    return statistics.mean(samples), min(samples)


def _ck(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=MATH_FIDELITY,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _build_linear_pcfg(M, K, N, gx, gy, dst_budget, activation):
    m_t, k_t, n_t = M // 32, K // 32, N // 32
    per_core_M = (m_t + gy - 1) // gy
    per_core_N = (n_t + gx - 1) // gx
    in0_block_w = 1
    for cand in (8, 4, 2, 1):
        if k_t % cand == 0:
            in0_block_w = cand
            break
    out_sw = min(per_core_N, dst_budget)
    while out_sw > 1 and per_core_N % out_sw != 0:
        out_sw -= 1
    out_sh = max(1, dst_budget // out_sw)
    out_sh = min(per_core_M, out_sh)
    while out_sh > 1 and per_core_M % out_sh != 0:
        out_sh -= 1
    fused = ttnn.UnaryOpType.GELU if activation == "gelu" else None
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=out_sh,
        out_subblock_w=out_sw,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=fused,
    )


def _sweep_shape(device, sync, label, M, K, N, activation, max_grid):
    gx_max, gy_max = max_grid
    torch.manual_seed(0)
    a = torch.randn(1, 1, M, K)
    w = torch.randn(K, N)
    ref = a.reshape(M, K) @ w
    if activation == "gelu":
        ref = _gelu(ref)

    a_tt = _from_torch(a, device, ttnn.bfloat8_b, ttnn.L1_MEMORY_CONFIG)
    w_tt = _from_torch(w, device, ttnn.bfloat8_b, ttnn.DRAM_MEMORY_CONFIG)
    ck = _ck(device)

    rows = []  # (kernel, cfg_str, avg_us, min_us, pcc, status)

    # --- minimal_matmul sweep ---
    fa = (ttnn.UnaryOpType.GELU, True) if activation == "gelu" else None
    for mb, kb, nb, sh, sw in MINIMAL_CONFIGS:
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=ttnn.CoreCoord(gx_max, gy_max),
        )
        cfgs = f"M{mb}K{kb}N{nb}/sb{sh}x{sw}"

        def _run():
            return ttnn.experimental.minimal_matmul(
                a_tt,
                w_tt,
                fused_activation=fa,
                config=cfg,
                compute_kernel_config=ck,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )

        try:
            out = _run()
            pcc = _pcc(ref, _to_torch_first(out, device, M, N))
            ttnn.deallocate(out)
            avg, mn = _time(_run, sync)
            rows.append(("minimal", cfgs, avg, mn, pcc, "ok" if pcc >= PCC_GATE else "LOWPCC"))
        except Exception as e:
            rows.append(("minimal", cfgs, float("inf"), float("inf"), 0.0, f"ERR:{repr(e)[:32]}"))

    # --- traditional linear sweep ---
    for gx, gy, dst in LINEAR_CONFIGS:
        if gx > gx_max or gy > gy_max:
            continue
        cfgs = f"grid{gx}x{gy}/dst{dst}"
        try:
            pcfg = _build_linear_pcfg(M, K, N, gx, gy, dst, activation)

            def _run():
                return ttnn.linear(
                    a_tt,
                    w_tt,
                    program_config=pcfg,
                    compute_kernel_config=ck,
                    dtype=ttnn.bfloat8_b,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )

            out = _run()
            pcc = _pcc(ref, _to_torch_first(out, device, M, N))
            ttnn.deallocate(out)
            avg, mn = _time(_run, sync)
            rows.append(("linear", cfgs, avg, mn, pcc, "ok" if pcc >= PCC_GATE else "LOWPCC"))
        except Exception as e:
            rows.append(("linear", cfgs, float("inf"), float("inf"), 0.0, f"ERR:{repr(e)[:32]}"))

    ttnn.deallocate(a_tt)
    ttnn.deallocate(w_tt)

    # --- display: ranked by min latency, valid configs first ---
    print(f"\n  shape {label}: M={M} K={K} N={N} act={activation or '-'}  (PCC gate {PCC_GATE})")
    print(f"    {'kernel':8s} {'config':18s} {'avg_us':>9s} {'min_us':>9s} {'pcc':>9s}  status")
    for kernel, cfgs, avg, mn, pcc, status in sorted(rows, key=lambda r: r[3]):
        avg_s = f"{avg:9.2f}" if avg != float("inf") else "      n/a"
        mn_s = f"{mn:9.2f}" if mn != float("inf") else "      n/a"
        print(f"    {kernel:8s} {cfgs:18s} {avg_s} {mn_s} {pcc:9.5f}  {status}")
    valid = [r for r in rows if r[4] >= PCC_GATE and r[3] != float("inf")]
    if valid:
        best = min(valid, key=lambda r: r[3])
        print(f"    -> winner: {best[0]} {best[1]}  ({best[3]:.2f} µs, PCC {best[4]:.5f})")
    return rows


# device param: single chip (1x1) and 4-chip (1x4 — the SigLIP vision submesh)
DEVICE_CASES = {
    "single": (ttnn.MeshShape(1, 1), (12, 10)),
    "4chip": (ttnn.MeshShape(1, 4), (12, 10)),
}


@pytest.mark.parametrize("device_case", list(DEVICE_CASES.keys()))
def test_mlp_minimal_vs_linear(device_case):
    pin_dev = os.environ.get("PI0_MLP_SWEEP_DEVICE", "").strip()
    if pin_dev and pin_dev != device_case:
        pytest.skip(f"pinned to {pin_dev}")
    mesh_shape, max_grid = DEVICE_CASES[device_case]

    pin_shape = os.environ.get("PI0_MLP_SWEEP_SHAPE", "").strip()
    shapes = {pin_shape: SHAPES[pin_shape]} if pin_shape in SHAPES else SHAPES

    device = ttnn.open_mesh_device(mesh_shape=mesh_shape, l1_small_size=24576)
    try:
        print(f"\n=== MLP minimal_matmul vs linear  device={device_case} mesh ===")
        all_rows = []
        for label, (M, K, N, act) in shapes.items():
            rows = _sweep_shape(device, device, label, M, K, N, act, max_grid)
            all_rows.append((label, rows))

        # every shape must have at least one config that passes PCC
        for label, rows in all_rows:
            ok = [r for r in rows if r[4] >= PCC_GATE and r[3] != float("inf")]
            assert ok, f"{label} on {device_case}: no config passed PCC gate {PCC_GATE}"
    finally:
        ttnn.close_mesh_device(device)
