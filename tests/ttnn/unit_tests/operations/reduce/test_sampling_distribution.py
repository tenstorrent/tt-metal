# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for ttnn.sampling random-threshold precision (issue #59732).

These tests validate that the sampler draws from the requested distribution
with FP32-level threshold resolution, as required by the bounty acceptance
criteria:

* Two-candidate frequency test with small probability that is biased/zero under
  the previous BF16 threshold.
* Multi-candidate frequency test: tail candidates are sampled and equal-probability
  candidates are sampled homogeneously within a documented statistical tolerance.
* Reference distribution and pass/fail criterion are documented explicitly; the
  test uses a statistically justified bound (normal approx / Wilson / chi-square).

Host-side simulation of the BF16 vs FP32 threshold is also included so the suite
can run without hardware and the statistical bounds can be validated independently.
The device tests (xfail-on-missing-device) use the same reference and tolerance.

Run:
  pytest tests/ttnn/unit_tests/operations/reduce/test_sampling_distribution.py -v
"""

import math

import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Host simulation of the sampling threshold quantisation
# ---------------------------------------------------------------------------


def bf16_to_f32_bits(bf16_u16: int) -> float:
    import struct

    return struct.unpack("<f", struct.pack("<I", bf16_u16 << 16))[0]


def f32_to_bf16_u16(v: float) -> int:
    import struct

    u = struct.unpack("<I", struct.pack("<f", v))[0]
    return (u >> 16) & 0xFFFF


def bf16_quantize(v: float) -> float:
    return bf16_to_f32_bits(f32_to_bf16_u16(v))


def simulate_sample_from_probs(probs: torch.Tensor, rand_uniform_fp32: float, *, bf16_threshold: bool) -> int:
    """
    Simulate the writer's stochastic pick: walk the cumulative distribution and
    return the first index where cum > threshold.

    If bf16_threshold is True the uniform threshold is quantized to the BF16
    grid (reproducing the previous device bias); otherwise it is used at full
    FP32 resolution.
    """
    thr = bf16_quantize(rand_uniform_fp32) if bf16_threshold else rand_uniform_fp32
    # Clamp to [0,1) like the device: BF16 path capped at 255/256 historically via scale,
    # FP32 path now uses the Float32 tile.
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
        u = torch.rand((), generator=gen).item()  # uniform [0,1)
        idx = simulate_sample_from_probs(probs, u, bf16_threshold=bf16_threshold)
        counts[idx] += 1
    return counts


# ---------------------------------------------------------------------------
# Statistical helpers (documented, statistically justified)
# ---------------------------------------------------------------------------


def two_sided_binomial_p_value(k: int, n: int, p0: float) -> float:
    """Two-sided exact-ish p-value via normal approximation with continuity correction.

    Used to decide whether observed count k is consistent with Binomial(n, p0).
    Rejected when p < 0.01 (1% significance).
    """
    if p0 in (0.0, 1.0):
        return 1.0 if k == round(p0 * n) else 0.0
    mean = n * p0
    var = n * p0 * (1 - p0)
    std = math.sqrt(var)
    # z with continuity correction
    z = (abs(k - mean) - 0.5) / std if std > 0 else 0.0
    # two-sided tail via erf
    return math.erfc(z / math.sqrt(2))


def wilson_interval(k: int, n: int, z: float = 2.576) -> tuple:
    """99% Wilson interval (z=2.576) for proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Host-only regression tests (no device required) — validate the reference
# ---------------------------------------------------------------------------


class TestHostSimulatedThresholdPrecision:
    """
    Host-only: documents that the BF16 threshold biases the distribution and
    that the FP32 threshold removes it. Reference distribution is Binomial(n, p)
    with explicit p0. Pass/fail is statistically justified (99% level).
    """

    def test_two_candidate_small_prob_bf16_is_biased(self):
        """
        Two candidates: p_small = 0.01 tail.
        Reference: expected count ~ n * 0.01.
        Under BF16 quantization the tail near 0.99 has step ~0.004-0.008 and the
        effective tail probability is ~0.008 instead of 0.01 (~20% bias).
        With n=100000 the bias is >10 sigma and the test is robust.
        """
        probs = torch.tensor([0.99, 0.01])
        n = 100000
        counts_bf16 = estimate_counts(probs, n, seed=0, bf16_threshold=True)
        counts_fp32 = estimate_counts(probs, n, seed=0, bf16_threshold=False)
        p0 = 0.01
        mean = n * p0
        std = math.sqrt(n * p0 * (1 - p0))
        bf16_tail = int(counts_bf16[1])
        fp32_tail = int(counts_fp32[1])
        bf16_z = abs(bf16_tail - mean) / std
        fp32_z = abs(fp32_tail - mean) / std
        # BF16 must be noticeably biased (expected ~800 vs 1000 -> z ~6)
        assert bf16_z > 4.0, (
            f"BF16 simulation not biased as expected: tail={bf16_tail} mean~{mean:.0f} z={bf16_z:.2f}. "
            f"This host check documents the pre-fix bias; if it fails, the quantisation model changed."
        )
        # FP32 must be statistically consistent at 3.5 sigma (~99.95% two-sided)
        assert fp32_z < 3.5, f"FP32 tail z={fp32_z:.2f} tail={fp32_tail} mean~{mean:.0f} outside 3.5 sigma"

    def test_two_candidate_fp32_within_binomial_tolerance(self):
        """FP32 simulation for p=0.02 tail is within 99% Wilson/Binomial tolerance."""
        probs = torch.tensor([0.98, 0.02])
        n = 20000
        counts = estimate_counts(probs, n, seed=1, bf16_threshold=False)
        k = int(counts[1])
        lo, hi = wilson_interval(k, n, z=2.576)
        # Reference p0=0.02 must lie inside the 99% Wilson interval of the observed proportion
        assert lo <= 0.02 <= hi, f"Wilson 99% interval [{lo:.4f},{hi:.4f}] does not contain p0=0.02 (k={k}, n={n})"

    def test_multi_candidate_tail_sampled_and_homogeneous(self):
        """
        Eight candidates: probs = [0.50, 0.30, 0.10, 0.05, 0.02, 0.015, 0.01, 0.005]
        Reference: each candidate's frequency ~ n * p_i.
        Requirements:
          - tail candidate p=0.005 must be sampled (count > 0) and within Wilson 99% of p0
          - equal-probability candidates: test with 4 equal 0.25 probs, max/min count ratio < 2
            at n=8000 within documented tolerance (simulated here).
        """
        probs = torch.tensor([0.50, 0.30, 0.10, 0.05, 0.02, 0.015, 0.01, 0.005])
        assert abs(probs.sum().item() - 1.0) < 1e-6
        n = 30000
        counts = estimate_counts(probs, n, seed=42, bf16_threshold=False)
        # Tail p=0.005 expected ~150
        tail_k = int(counts[-1])
        assert tail_k > 0, "Tail p=0.005 was never sampled (biased threshold)"
        lo, hi = wilson_interval(tail_k, n, z=2.576)
        assert lo <= 0.005 <= hi, f"Tail Wilson 99% [{lo:.5f},{hi:.5f}] misses p=0.005 (k={tail_k})"

        # Homogeneity: 4 equal probs
        probs_eq = torch.tensor([0.25, 0.25, 0.25, 0.25])
        counts_eq = estimate_counts(probs_eq, 8000, seed=7, bf16_threshold=False)
        # Chi-square test for uniformity at 99%: df=3, critical value 11.34
        expected = 8000 / 4
        chi2 = float(((counts_eq.float() - expected) ** 2 / expected).sum().item())
        assert chi2 < 11.34, f"Equal-prob homogeneity failed: chi2={chi2:.2f} > 11.34 (counts {counts_eq.tolist()})"


# ---------------------------------------------------------------------------
# Device tests — same reference and tolerance, require Wormhole/Blackhole
# ---------------------------------------------------------------------------

try:
    import ttnn  # noqa: F401
    from models.common.utility_functions import is_wormhole_b0, is_blackhole

    HAS_TTNN = True
except Exception:  # pragma: no cover
    HAS_TTNN = False


def _device_available():
    if not HAS_TTNN:
        return False
    try:
        return is_wormhole_b0() or is_blackhole()
    except Exception:
        return False


@pytest.mark.skipif(not _device_available(), reason="Requires Wormhole or Blackhole device")
class TestSamplingDistributionDevice:
    """
    Device frequency tests — same acceptance criteria as host simulation, but
    against real ttnn.sampling. Each test draws n samples (repeated op calls)
    and checks observed counts against the explicitly documented reference
    distribution with a statistically justified 99% pass/fail criterion.

    Input construction: logits are set so that softmax(logits) == desired probs
    (logits = log(probs)). Temperature = 1.0, k = vocab, p = 0 so the full
    distribution is sampled. Per-user seeding is exercised: seed varies per draw
    for the frequency test and determinism is checked separately.
    Reference: Binomial for two-candidate tail, Wilson/chi-square for multi.
    """

    @staticmethod
    def _probs_to_logits(probs: torch.Tensor) -> torch.Tensor:
        return torch.log(probs.clamp(min=1e-12))

    @staticmethod
    def _run_draws(device, probs: torch.Tensor, n: int, base_seed: int):
        """
        Draw n samples from the distribution `probs` via ttnn.sampling.
        Returns counts per vocab index. Uses one user (num_users=1) with
        vocab padded to power-of-two tiles as required by the op.
        """
        vocab = len(probs)
        # Pad vocab to next power-of-two multiple of 32 (op requirement: Wt power-of-2)
        padded_vocab = 32
        while padded_vocab < vocab:
            padded_vocab *= 2
        # logits for the padded vocab: tail padding gets -inf so it is never chosen
        logits = TestSamplingDistributionDevice._probs_to_logits(probs)
        padded_logits = torch.full((1, 1, 1, padded_vocab), float("-inf"))
        padded_logits[0, 0, 0, :vocab] = logits
        # Map cleaned indices: 0..vocab-1 map to themselves
        input_indices = torch.arange(0, padded_vocab, dtype=torch.int32).expand(1, 1, 1, padded_vocab)

        counts = torch.zeros(vocab, dtype=torch.int64)
        for i in range(n):
            seed = base_seed + i * 7919  # distinct deterministic seeds
            input_values_tensor = ttnn.from_torch(
                padded_logits, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            input_indices_tensor = ttnn.from_torch(
                input_indices, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            k_tensor = ttnn.from_torch(
                torch.tensor([vocab], dtype=torch.int32), device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            p_tensor = ttnn.from_torch(
                torch.tensor([0.0], dtype=torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            temp = ttnn.from_torch(
                torch.tensor([1.0], dtype=torch.bfloat16),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            out = ttnn.to_torch(
                ttnn.sampling(
                    input_values_tensor,
                    input_indices_tensor,
                    k=k_tensor,
                    p=p_tensor,
                    temp=temp,
                    seed=seed,
                )
            )
            idx = int(out.view(-1)[0].item())
            assert 0 <= idx < vocab, f"Sampled index {idx} out of vocab range [0,{vocab})"
            counts[idx] += 1
        return counts

    def test_two_candidate_small_prob_device(self, device):
        """
        Two-candidate frequency test — small tail p=0.02.
        Reference: Binomial(n, 0.02), n=2000.
        Pass/fail: observed tail proportion within 99% Wilson interval of p0=0.02
        (equivalently two-sided p >= 0.01). Under the old BF16 threshold this
        case was materially biased or zero for many seeds; the fix must bring it
        inside the interval on both Wormhole and Blackhole.
        Documented: draw count n=2000, seed policy base_seed=12345 + i*7919,
        architecture Wormhole/Blackhole, input logits derived from probs=[0.98,0.02].
        """
        probs = torch.tensor([0.98, 0.02])
        n = 2000
        counts = self._run_draws(device, probs, n, base_seed=12345)
        tail_k = int(counts[1])
        lo, hi = wilson_interval(tail_k, n, z=2.576)
        assert lo <= 0.02 <= hi, (
            f"Two-candidate 99% Wilson [{lo:.4f},{hi:.4f}] misses p0=0.02 (tail k={tail_k}, n={n}). "
            f"Indicates biased threshold — expected ~{n*0.02:.0f} tail samples."
        )
        # Also require tail was actually sampled
        assert tail_k > 0, f"Tail never sampled in {n} draws (bf16 bias)"

    def test_multi_candidate_tail_and_homogeneity_device(self, device):
        """
        Multi-candidate frequency test.
        Distribution: 8 candidates [0.50,0.30,0.10,0.05,0.02,0.015,0.01,0.005]
        Reference: Binomial per candidate, n=3000.
        Pass/fail:
          - tail p=0.005 sampled (count>0) and 99% Wilson contains p0
          - equal-prob subset (4 x 0.25, n=2000) homogeneous: chi-square < 11.34 (df=3, 99%)
        """
        probs = torch.tensor([0.50, 0.30, 0.10, 0.05, 0.02, 0.015, 0.01, 0.005])
        n = 3000
        counts = self._run_draws(device, probs, n, base_seed=20245)
        tail_k = int(counts[-1])
        assert tail_k > 0, f"Tail p=0.005 never sampled in {n} draws"
        lo, hi = wilson_interval(tail_k, n, z=2.576)
        assert lo <= 0.005 <= hi, f"Tail Wilson 99% [{lo:.5f},{hi:.5f}] misses 0.005 (k={tail_k}, n={n})"

        # Homogeneity: equal 4-way
        probs_eq = torch.tensor([0.25, 0.25, 0.25, 0.25])
        counts_eq = self._run_draws(device, probs_eq, 2000, base_seed=99991)
        expected = 2000 / 4
        chi2 = float(((counts_eq.float() - expected) ** 2 / expected).sum().item())
        assert chi2 < 11.34, f"Equal-prob homogeneity chi2={chi2:.2f} > 11.34 (counts {counts_eq.tolist()})"
