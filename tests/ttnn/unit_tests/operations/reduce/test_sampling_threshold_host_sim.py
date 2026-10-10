# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-side validation of the sampling random-threshold bias model (issue #59732).

Complements tests/ttnn/unit_tests/operations/reduce/test_sampling_distribution.py
which is hardware-only (skipped on ttsim). This file validates the SAME reference
and statistical pass/fail criteria WITHOUT hardware so the bounds can be checked
independently, as requested in the bounty guidance:

  "Before finalizing the test thresholds, validate the reference probabilities
   and statistical bounds independently on the host."

Reference distribution: Binomial(n, p0) for two-candidate tail; multinomial
with Wilson 99% / chi-square 99% (df=3, crit 11.34) for multi-candidate tail
and homogeneity — identical criteria to the device tests.

Run:
  pytest tests/ttnn/unit_tests/operations/reduce/test_sampling_threshold_host_sim.py -v

The bias model reproduces the exact writer-side behaviour:
  - BF16 path: threshold quantized to BF16 grid (via f32->bf16->f32 round-trip);
    near cum~0.99 the BF16 step is ~0.008 and the old 255/256 cap truncates
    the tail, causing small-p candidates to be under-sampled or zero.
  - Fixed path: Float32 lattice (hi*256+lo)/65536 (upstream #59759) or Float32
    tile (PR luoyu2475/1939566682). Both give >=16-bit resolution so the tail
    is sampled correctly; host simulation uses the finer Float32 grid.
"""

import math
import struct

import pytest
import torch


def bf16_to_f32_bits(bf16_u16: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bf16_u16 << 16))[0]


def f32_to_bf16_u16(v: float) -> int:
    u = struct.unpack("<I", struct.pack("<f", v))[0]
    return (u >> 16) & 0xFFFF


def bf16_quantize(v: float) -> float:
    return bf16_to_f32_bits(f32_to_bf16_u16(v))


def simulate_sample_from_probs(probs: torch.Tensor, rand_uniform_fp32: float, *, bf16_threshold: bool) -> int:
    thr = bf16_quantize(rand_uniform_fp32) if bf16_threshold else rand_uniform_fp32
    cum = 0.0
    for i, p in enumerate(probs.tolist()):
        cum += float(p)
        if cum > thr:
            return i
    return len(probs) - 1


def estimate_counts(probs: torch.Tensor, n: int, seed: int, *, bf16_threshold: bool):
    gen = torch.Generator().manual_seed(seed)
    counts = torch.zeros(len(probs), dtype=torch.int64)
    for _ in range(n):
        u = torch.rand((), generator=gen).item()
        idx = simulate_sample_from_probs(probs, u, bf16_threshold=bf16_threshold)
        counts[idx] += 1
    return counts


def wilson_interval(k: int, n: int, z: float = 2.576) -> tuple:
    """99% Wilson interval (z=2.576)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


class TestHostSimulatedThresholdPrecision:
    """
    Host-only: documents BF16 bias and that the fixed threshold removes it.
    Reference: Binomial(n, p) with explicit p0. Pass/fail at 99% level.
    Also validates that upstream #59759's 16-bit lattice (hi*256+lo)/65536
    would similarly remove the bias (host models it as Float32, strictly finer).
    """

    def test_two_candidate_small_prob_bf16_is_biased(self):
        """
        Two candidates: p_small = 0.01.
        BF16 near 0.99 step ~0.004-0.008 so effective tail ~0.008 not 0.01 (~20% bias).
        n=100k makes bias >4 sigma and test robust.
        Device fix: threshold with 16-bit+ resolution gives ~0.01.
        """
        probs = torch.tensor([0.99, 0.01])
        n = 100000
        counts_bf16 = estimate_counts(probs, n, seed=0, bf16_threshold=True)
        counts_fixed = estimate_counts(probs, n, seed=0, bf16_threshold=False)
        p0 = 0.01
        mean = n * p0
        std = math.sqrt(n * p0 * (1 - p0))
        bf16_tail = int(counts_bf16[1])
        fixed_tail = int(counts_fixed[1])
        bf16_z = abs(bf16_tail - mean) / std
        fixed_z = abs(fixed_tail - mean) / std
        assert bf16_z > 4.0, (
            f"BF16 simulation not biased: tail={bf16_tail} mean~{mean:.0f} z={bf16_z:.2f}."
        )
        assert fixed_z < 3.5, f"Fixed tail z={fixed_z:.2f} tail={fixed_tail} outside 3.5 sigma"

    def test_two_candidate_fixed_within_binomial_tolerance(self):
        """Fixed simulation for p=0.02 tail within 99% Wilson/Binomial tolerance. n=20000."""
        probs = torch.tensor([0.98, 0.02])
        n = 20000
        counts = estimate_counts(probs, n, seed=1, bf16_threshold=False)
        k = int(counts[1])
        lo, hi = wilson_interval(k, n, z=2.576)
        assert lo <= 0.02 <= hi, f"Wilson 99% [{lo:.4f},{hi:.4f}] misses p0=0.02 (k={k}, n={n})"

    def test_multi_candidate_tail_sampled_and_homogeneous(self):
        """
        Eight candidates: [0.50,0.30,0.10,0.05,0.02,0.015,0.01,0.005]
        - Tail p=0.005 must be sampled and Wilson 99% contain p0
        - 4 equal 0.25: chi-square df=3 crit 11.34 at 99%
        """
        probs = torch.tensor([0.50, 0.30, 0.10, 0.05, 0.02, 0.015, 0.01, 0.005])
        assert abs(probs.sum().item() - 1.0) < 1e-6
        n = 30000
        counts = estimate_counts(probs, n, seed=42, bf16_threshold=False)
        tail_k = int(counts[-1])
        assert tail_k > 0, "Tail p=0.005 never sampled (biased threshold)"
        lo, hi = wilson_interval(tail_k, n, z=2.576)
        assert lo <= 0.005 <= hi, f"Tail Wilson 99% [{lo:.5f},{hi:.5f}] misses p0=0.005 (k={tail_k})"

        probs_eq = torch.tensor([0.25, 0.25, 0.25, 0.25])
        counts_eq = estimate_counts(probs_eq, 8000, seed=7, bf16_threshold=False)
        expected = 8000 / 4
        chi2 = float(((counts_eq.float() - expected) ** 2 / expected).sum().item())
        assert chi2 < 11.34, f"Homogeneity failed: chi2={chi2:.2f} > 11.34 (counts {counts_eq.tolist()})"
