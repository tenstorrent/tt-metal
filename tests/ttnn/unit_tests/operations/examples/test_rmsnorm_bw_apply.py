# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness + device-kernel timing: all-SFPU eltwise_chain generic op vs the host composite
from PR #54829 (`apply_backward`) for the distributed-RMSNorm backward apply step.

Both variants start from the same gathered statistics (E[x^2] and E[x*g] as [N,C,H,1] fp32
columns) and produce dx. Correctness is the only pass/fail; perf is measured (sum of
DEVICE KERNEL DURATION [ns] over every program launched by the variant) and reported.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics

import pytest
import torch
import ttnn
from loguru import logger

from ttnn.operations.examples.rmsnorm_bw_apply import rmsnorm_bw_apply

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
EPS = 1e-5


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found, n = 0.0, False, 0
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
                n += 1
    return (total, n) if found else (None, 0)


def _dev(t, device):
    return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def _make_case(device, shape, seed=7):
    torch.manual_seed(seed)
    n, c, h, w = shape
    x64 = torch.randn(shape, dtype=torch.float64)
    dy64 = torch.randn(shape, dtype=torch.float64)
    gamma64 = torch.rand(1, 1, 1, w, dtype=torch.float64) + 0.5
    mean_x2_64 = (x64 * x64).mean(dim=-1, keepdim=True)
    rms64 = torch.sqrt(mean_x2_64 + EPS)
    g64 = gamma64 * dy64 / rms64
    scale64 = (x64 * g64).mean(dim=-1, keepdim=True)  # E[x*g]
    dx64 = g64 - x64 * scale64 / (rms64 * rms64)

    f32 = lambda t: t.to(torch.float32)
    tensors = dict(
        x=_dev(f32(x64), device),
        dy=_dev(f32(dy64), device),
        gamma=_dev(f32(gamma64), device),
        mean_x2=_dev(f32(mean_x2_64), device),
        scale=_dev(f32(scale64), device),
    )
    return tensors, dx64


# --- variants --------------------------------------------------------------------------------
def run_chain(t):
    inv_rms = ttnn.rsqrt(ttnn.add(t["mean_x2"], EPS))
    d = ttnn.multiply(t["scale"], ttnn.square(inv_rms))
    return rmsnorm_bw_apply(t["x"], t["dy"], t["gamma"], inv_rms, d)


def run_pr_dx(t):
    """Verbatim structure of PR #54829 apply_backward, dx part (inputs already fp32 so the
    typecasts are no-ops and are omitted)."""
    rms = ttnn.sqrt(ttnn.add(t["mean_x2"], EPS))
    g = ttnn.divide(ttnn.multiply(t["gamma"], t["dy"]), rms)
    return ttnn.subtract(g, ttnn.multiply(t["x"], ttnn.divide(t["scale"], ttnn.square(rms))))


def run_pr_full(t):
    """PR apply_backward including weight grad, for context."""
    rms = ttnn.sqrt(ttnn.add(t["mean_x2"], EPS))
    g = ttnn.divide(ttnn.multiply(t["gamma"], t["dy"]), rms)
    dx = ttnn.subtract(g, ttnn.multiply(t["x"], ttnn.divide(t["scale"], ttnn.square(rms))))
    dgamma = ttnn.sum(ttnn.multiply(t["dy"], ttnn.divide(t["x"], rms)), dim=[0, 1, 2], keepdim=True)
    return dx, dgamma


SHAPES = [(1, 1, 512, 1024), (1, 1, 8192, 512), (1, 8, 1024, 1024)]

# Full-size fp32 tensors moved through DRAM per variant (reads + writes). The small column
# stats / gamma are ignored (<0.2% of traffic).
#   chain  : read x, read dy, write dx                                      -> 3
#   pr_dx  : multiply(gamma,dy) 2, divide(.,rms) 2, multiply(x,col) 2, subtract 3  -> 9
#   pr_full: pr_dx + divide(x,rms) 2 + multiply(dy,.) 3 + sum reads 1            -> 15
TRAFFIC_TENSORS = {"chain": 3, "pr_dx": 9, "pr_full": 15}
USEFUL_TENSORS = 3


@pytest.mark.parametrize("shape", SHAPES, ids=["x".join(map(str, s)) for s in SHAPES])
def test_rmsnorm_bw_apply(device, shape):
    t, dx64 = _make_case(device, shape)

    got = {"chain": run_chain(t), "pr_dx": run_pr_dx(t)}
    errs = {}
    for name, out in got.items():
        out_t = ttnn.to_torch(out).to(torch.float64)
        err = (out_t - dx64).abs().max().item()
        rel = err / dx64.abs().max().item()
        errs[name] = (err, rel)
        assert torch.allclose(out_t, dx64, rtol=1e-4, atol=1e-4), f"{name}: max abs err {err}"
    chain_vs_pr = (ttnn.to_torch(got["chain"]) - ttnn.to_torch(got["pr_dx"])).abs().max().item()

    # --- timing ---
    trials = int(os.environ.get("RB_TRIALS", "5"))
    runners = {"chain": lambda: run_chain(t), "pr_dx": lambda: run_pr_dx(t), "pr_full": lambda: run_pr_full(t)}
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # discard warmup (includes kernel compiles)
    samples = {k: [] for k in runners}
    nprog = {}
    for trial in range(trials + 1):
        for name, run in runners.items():
            run()
            ttnn.synchronize_device(device)
            ns, n = _read_kernel_ns(device)
            assert ns is not None, f"no profiler data for {name}"
            nprog[name] = n
            if trial:
                samples[name].append(ns)

    med = {k: statistics.median(v) for k, v in samples.items()}
    logger.info(f"shape {shape}")
    logger.info(
        f"  max|err| vs fp64 golden: chain={errs['chain'][0]:.3e} (rel {errs['chain'][1]:.2e})  "
        f"pr_dx={errs['pr_dx'][0]:.3e} (rel {errs['pr_dx'][1]:.2e})  chain-vs-pr={chain_vs_pr:.3e}"
    )
    for k in runners:
        logger.info(
            f"  {k:8s} programs={nprog[k]:2d}  device kernel ns median={med[k]:12.0f}  "
            f"std={statistics.pstdev(samples[k]):10.0f}  ({med[k]/1e3:9.1f} us)"
        )
    logger.info(
        f"  speedup chain vs pr_dx: {med['pr_dx']/med['chain']:.2f}x   vs pr_full: {med['pr_full']/med['chain']:.2f}x"
    )
    n_elem = 1
    for s_ in shape:
        n_elem *= s_
    tensor_bytes = n_elem * 4
    for k in runners:
        useful = USEFUL_TENSORS * tensor_bytes / med[k]  # bytes/ns == GB/s
        actual = TRAFFIC_TENSORS[k] * tensor_bytes / med[k]
        logger.info(
            f"  {k:8s} DRAM BW: effective (3 tensors) {useful:7.1f} GB/s   "
            f"actual traffic ({TRAFFIC_TENSORS[k]:2d} tensors) {actual:7.1f} GB/s"
        )
