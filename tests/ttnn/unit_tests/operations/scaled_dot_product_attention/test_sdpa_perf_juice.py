# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-juicing unit tests for the generic SDPA op.  See SDPA_PERF_HANDOFF.md.

Four grid-filling, *favorable* shapes whose only job is to let you push the
kernel as fast as it will go on a small, friendly subset — generality and
precision elsewhere are explicitly NOT a concern here.

Contract for each case:
  * bfloat16 inputs, self-attention (MHA), mask=none, auto scale.
  * default compute config (fp32 dest-acc is NOT required).
  * must fill the whole 8x8 worker grid (asserted: cores_used == 64).
  * must stay above PCC 0.99 vs a torch fp32 reference (the only correctness bar).

Each case prints — and appends to generated/sdpa_juice_perf.txt — the measured
on-device kernel time, achieved TFLOPs, LoFi-peak utilization, and cores used,
so you can watch utilization climb as you optimize.

Run (see device output with -s):
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/scaled_dot_product_attention/test_sdpa_perf_juice.py
"""

import os

# Device profiler MUST be enabled before the device is opened (RuntimeOptions
# caches these at first device init). CPP_POST_PROCESS is what computes the
# "DEVICE KERNEL DURATION [ns]" analysis we read back.
for _k, _v in {
    "TT_METAL_DEVICE_PROFILER": "1",
    "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
    "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
    "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES": "1",
}.items():
    os.environ.setdefault(_k, _v)

import math
from pathlib import Path

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention as sdpa


# --- LoFi matmul peak --------------------------------------------------------
# 8*16*16 = 2048 MAC/cycle/core -> *2 = 4096 FLOP/cycle/core.
# Wormhole worker grid = 8x8 = 64 cores; AI clock ~0.9855 GHz (from device sync).
GRID_CORES = 64
PEAK_TFLOPS = 4096 * GRID_CORES * 0.9855e9 / 1e12  # ~258 TFLOPs LoFi
TILE = 32
PCC_THRESHOLD = 0.99
_PERF_LOG = Path("generated/sdpa_juice_perf.txt")

# (id, B, H_q, H_kv, S, head_dim) — self-attention (S_q == S_kv), MHA (H_kv == H_q),
# mask=none. Every case is chosen so B*H*ceil((S/32)/c_q) >= 64 (fills the grid).
CASES = [
    ("favorable_h8_s4096_d64", 1, 8, 8, 4096, 64),
    ("favorable_h8_s8192_d128", 1, 8, 8, 8192, 128),
    ("favorable_h16_s8192_d128", 1, 16, 16, 8192, 128),
    ("very_favorable_h8_s16384_d512", 1, 8, 8, 16384, 512),
]


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _ref_sdpa(q, k, v, scale):
    # q,k,v: [B, H, S, d] in fp32. Full (no mask) softmax attention.
    scores = (q @ k.transpose(-2, -1)) * scale
    return torch.softmax(scores, dim=-1) @ v


def _cores_used(B, H, S, d):
    """Mirror the op's work distribution (program_descriptor): work unit =
    (b, h, q_chunk); c_q = clamp(16 / (d/32), 1, 4) Q tile-rows per chunk."""
    Sq_t = S // TILE
    Dt = d // TILE
    c = max(1, min(4, 16 // Dt))
    c_q = min(c, Sq_t)
    Nq = -(-Sq_t // c_q)
    return min(GRID_CORES, B * H * Nq)


def _measure_device_kernel_ns(device):
    """Sum DEVICE KERNEL DURATION [ns] over the programs dispatched since the
    last ReadDeviceProfiler (i.e. the op). ReadDeviceProfiler finishes the CQ
    internally, so no separate synchronize is needed."""
    try:
        ttnn.ReadDeviceProfiler(device)
        per_chip = ttnn.get_latest_programs_perf_data() or {}
    except Exception:
        return None
    key = "DEVICE KERNEL DURATION [ns]"
    total, found = 0.0, False
    for programs in per_chip.values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", {}) or {}).get(key)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


@pytest.mark.parametrize("name,B,Hq,Hkv,S,d", CASES, ids=[c[0] for c in CASES])
def test_sdpa_perf(device, name, B, Hq, Hkv, S, d):
    torch.manual_seed(0)
    q = torch.randn(B, Hq, S, d, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, S, d, dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, S, d, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(d)

    tq = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Bracket the op: flush prep dispatches, run, read the op's window.
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    out = sdpa(tq, tk, tv)  # mask=none, scale=auto, default compute config (no fp32 dest)
    ttnn.synchronize_device(device)
    ns = _measure_device_kernel_ns(device)

    ref = _ref_sdpa(q.float(), k.float(), v.float(), scale)
    got = ttnn.to_torch(out).float()
    pcc = _pcc(got, ref)

    flops = 4 * B * Hq * (S**2) * d  # two matmuls: QK^T + P@V, mask=none
    tflops = (flops / (ns * 1e-9) / 1e12) if ns else float("nan")
    util = (tflops / PEAK_TFLOPS * 100) if ns else float("nan")
    cores = _cores_used(B, Hq, S, d)

    line = (
        f"[{name}] cores={cores}/{GRID_CORES} "
        f"device_kernel_ns={ns:.0f} achieved={tflops:.2f}TFLOPs "
        f"util={util:.2f}% pcc={pcc:.5f}"
    )
    print("\n" + line)
    _PERF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _PERF_LOG.open("a") as f:
        f.write(line + "\n")

    # Grid-fill is a hard requirement for these cases (see handoff).
    assert cores == GRID_CORES, f"{name}: only {cores}/{GRID_CORES} cores used — does not fill the grid"
    # The ONLY correctness bar. Break precision freely as long as this holds.
    assert pcc >= PCC_THRESHOLD, f"{name}: pcc {pcc:.5f} < {PCC_THRESHOLD}"
