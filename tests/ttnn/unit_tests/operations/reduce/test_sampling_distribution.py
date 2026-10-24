# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Distribution-level check of ttnn.sampling.

The device threshold is k / 2^24 (three BF16 integer digits; see writer_interleaved.cpp), which
is the FP32 ulp on [0.5, 1). A single packed BF16 uniform is a k/256 grid near 1 and is capped
at 255/256, so a token whose mass sits in (255/256, 1] is never drawn.

Draw count: N calls (default 3000, override with SAMPLING_DRAWS) times USERS (32) independent
per-user streams. Seed policy: ttnn.manual_seed once, seeds = arange(1..USERS) * a fixed odd
constant, then every ttnn.sampling call takes the next value of that stream. The op `seed`
argument is not used here; it is a compile-time arg and would rebuild the kernel per call.
Same assertions on Wormhole and Blackhole.

Reference, per user: quantize logits to BF16 (what the device receives), restrict to the top-k,
softmax in FP32 of those BF16 probabilities, apply the nucleus (drop the smallest tokens while
their cumulative mass is <= 1 - p, always keep the largest), renormalize. Pass/fail is the
multinomial sampling error of that reference, spelled out on each test.
"""

import os

import pytest
import torch
import torch.nn.functional as F

import ttnn

# The statistics need ~10^5 device draws per test (USERS x N); on the ttsim simulator that is
# a timeout, not a measurement, so the module is hardware-only (same gate as test_reduction.py).
pytestmark = pytest.mark.skipif(
    bool(os.environ.get("TT_METAL_SIMULATOR")),
    reason="distribution-level sampling checks need ~10^5 draws; too slow on the ttsim simulator",
)

USERS, W = 32, 128  # production candidate row: max_top_k 32 x 4 devices
K, P = 20, 0.95  # the gpqa eval's sampling parameters, temperature 1
N = int(os.environ.get("SAMPLING_DRAWS", "3000"))


def _rows():
    torch.manual_seed(0)
    rows = []
    for u in range(USERS):
        if u == 0:  # one dominant token (0.97): nucleus is a single token
            r = torch.full((W,), -8.0)
            r[0] = 0.0
        elif u == 1:  # near-flat over 20 tokens (a strict order, so the nucleus edge is not a tie)
            r = torch.full((W,), -20.0)
            r[:20] = -0.01 * torch.arange(20, dtype=torch.float32)
        elif u == 2:  # long thin tail: many ~0.3% tokens (strictly ordered so top-k is unambiguous)
            tail = [0.003 * (1 - 0.001 * i) for i in range(100)]
            r = torch.log(torch.tensor([0.4, 0.2, 0.1] + tail + [1e-5] * 25))
        else:  # zipf-like with exponent 0.6..2.6
            a = 0.6 + 2.0 * (u - 3) / (USERS - 4)
            r = -a * torch.log(torch.arange(1, W + 1, dtype=torch.float32))
        rows.append(r)
    return torch.stack(rows).view(1, 1, USERS, W)


def _reference(values):
    """vLLM / HF semantics (v1 TopKTopPSampler.apply_top_k_top_p): sort, mask everything outside
    the top-k to -inf, softmax over what is left (so the nucleus is computed on the top-k
    renormalised mass), drop the smallest tokens whose cumulative mass is <= 1 - p, keep at least
    the largest, renormalise."""
    logits = values.float()[0, 0]  # [USERS, W]
    ref = torch.zeros_like(logits)
    for u in range(USERS):
        row = logits[u].clone()
        topk_v, topk_i = torch.topk(row, K)
        masked = torch.full_like(row, float("-inf"))
        masked[topk_i] = topk_v
        asc_v, asc_i = torch.sort(masked, descending=False)
        # The kernel accumulates BF16 probabilities; mirror that so the nucleus edge agrees.
        probs = torch.softmax(asc_v, dim=-1).to(torch.bfloat16).float()
        drop = torch.cumsum(probs, dim=-1) <= (1 - P)
        drop[-1] = False
        keep_i = asc_i[~drop]
        kept = probs[~drop]
        ref[u, keep_i] = kept / kept.sum()
    return ref


def test_sampling_matches_reference_distribution(device):
    """Multi-candidate rows (dominant, near-flat, long ~0.3% tail, zipf) against `_reference`.

    Pass when, for every user, total variation is at most max(0.03, 3 sigma of the multinomial
    TV) and the reference mass of never-sampled tokens with probability >= 0.002 — excluding the
    single nucleus-edge token — is at most 0.003. The 0.03 floor is the approximate-exp softmax
    residual. A threshold stuck on the BF16 1/256 grid leaves those tail tokens unsampled, so
    `never` exceeds 0.003 even when the TV floor would not.
    """
    values = _rows()
    values_bf16 = values.to(torch.bfloat16)
    ref = _reference(values_bf16)
    vt = ttnn.from_torch(values_bf16, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    it = ttnn.from_torch(
        torch.arange(W, dtype=torch.int32).expand(1, 1, USERS, W),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kt = ttnn.from_torch(
        torch.full((USERS,), K, dtype=torch.int32), device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    pt = ttnn.from_torch(torch.full((USERS,), P), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt = ttnn.from_torch(torch.ones(USERS), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    counts = torch.zeros(USERS, W)
    # Serving path: seed the per-user RNG streams once (ttnn.manual_seed), then every sampling
    # call draws the next number of each user's stream. An int ``seed`` on the op is a
    # compile-time arg and would rebuild the kernels on every new value.
    seeds = ttnn.from_torch(
        torch.arange(1, USERS + 1, dtype=torch.int32) * 7919,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    user_ids = ttnn.from_torch(
        torch.arange(USERS, dtype=torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    ttnn.manual_seed(seeds=seeds, user_ids=user_ids)
    for s in range(N):
        out = ttnn.to_torch(ttnn.sampling(vt, it, k=kt, p=pt, temp=tt)).reshape(-1)[:USERS].long()
        counts[torch.arange(USERS), out] += 1
    emp = counts / N
    tv = 0.5 * (emp - ref).abs().sum(-1)  # per user
    noise = (ref * (1 - ref)).sum(-1).sqrt() / N**0.5 * 3.0  # ~3 sigma of the multinomial estimate
    print(
        f"\n{'user':>4} {'kept':>4} {'TV':>6} {'~noise':>6} {'never-sampled mass (ref>=0.2%)':>30} {'tail(ref<1%) ratio emp/ref':>26}"
    )
    bad = []
    for u in range(USERS):
        kept = int((ref[u] > 0).sum())
        # The single smallest kept token sits on the nucleus edge, where the kernel's own BF16
        # softmax and the reference can disagree by one token; it is excluded from the
        # never-sampled check (its mass is bounded by the edge token's probability).
        edge = ref[u][ref[u] > 0].min()
        never = ref[u][(ref[u] >= 0.002) & (emp[u] == 0) & (ref[u] > edge)].sum().item()
        tail = ref[u] < 0.01
        ratio = (emp[u][tail].sum() / ref[u][tail].sum()).item() if ref[u][tail].sum() > 0 else float("nan")
        print(f"{u:>4} {kept:>4} {tv[u]:>6.3f} {noise[u]:>6.3f} {never:>30.4f} {ratio:>26.2f}")
        if tv[u] > max(0.03, noise[u]) or never > 0.003:
            bad.append(u)
    assert not bad, f"users {bad} deviate from the reference distribution (see table)"


def _draw(device, values, k, p, n):
    vt = ttnn.from_torch(values.to(torch.bfloat16), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    it = ttnn.from_torch(
        torch.arange(W, dtype=torch.int32).expand(1, 1, USERS, W),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    kt = ttnn.from_torch(
        torch.full((USERS,), k, dtype=torch.int32), device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    pt = ttnn.from_torch(torch.full((USERS,), p), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt = ttnn.from_torch(torch.ones(USERS), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    seeds = ttnn.from_torch(
        torch.arange(1, USERS + 1, dtype=torch.int32) * 104729,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    user_ids = ttnn.from_torch(
        torch.arange(USERS, dtype=torch.int32), device=device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    ttnn.manual_seed(seeds=seeds, user_ids=user_ids)
    counts = torch.zeros(USERS, W)
    for _ in range(n):
        out = ttnn.to_torch(ttnn.sampling(vt, it, k=kt, p=pt, temp=tt)).reshape(-1)[:USERS].long()
        counts[torch.arange(USERS), out] += 1
    return counts / n


def test_sampling_threshold_uniformity(device):
    """Equal-probability rows, no nucleus cut: the per-rank frequency is the random threshold's
    histogram over equal-width bins of [0, 1). A uniform threshold gives 1/m everywhere; each
    rank is asserted within 4 sigma of the multinomial estimate over USERS * N draws."""
    draws = USERS * N
    for m in (4, 10, 20):
        values = torch.full((1, 1, USERS, W), -30.0)
        values[..., :m] = 0.0
        emp = _draw(device, values, k=m, p=1.0, n=N)
        hist = emp[:, :m].mean(0)  # averaged over the 32 users' streams
        sigma = ((1 / m) * (1 - 1 / m) / draws) ** 0.5
        print(f"\nm={m} (expected {1/m:.4f} +- {4*sigma:.4f}): " + " ".join(f"{h:.3f}" for h in hist))
        assert torch.all(
            (hist - 1 / m).abs() <= 4 * sigma
        ), f"m={m}: rank frequencies {hist.tolist()} deviate from 1/{m}"
        assert emp[:, m:].sum() == 0, f"m={m}: a rank beyond the kept set was drawn"
    # nucleus p=0.95 over 20 near-equal tokens keeps the smallest set whose mass exceeds p (19 tokens).
    values = torch.full((1, 1, USERS, W), -30.0)
    values[..., :20] = -0.01 * torch.arange(20, dtype=torch.float32)
    emp = _draw(device, values, k=20, p=0.95, n=N)
    hist = emp[:, :19].mean(0)
    # expected: softmax of the 20 near-equal logits, nucleus keeps 19, renormalised
    probs = torch.softmax(values[0, 0, 0, :20].to(torch.bfloat16).float(), dim=-1)
    expected = probs[:19] / probs[:19].sum()
    sigma = (expected * (1 - expected) / draws).sqrt()
    print("m=20 with p=0.95: " + " ".join(f"{h:.3f}" for h in emp[:, :20].mean(0)))
    assert torch.all(
        (hist - expected).abs() <= 5 * sigma
    ), f"p=0.95: kept-rank frequencies {hist.tolist()} deviate from {expected.tolist()}"
    assert emp[:, 19:].sum() == 0, "p=0.95: a token outside the nucleus was drawn"


def test_sampling_low_digit_reaches_the_tail(device):
    """Two-candidate frequency check at a probability the BF16 threshold cannot represent.

    Every user is the same two-token row. gap = 6.25 is exact in BF16, and
    P(second) = 1 / (1 + exp(6.25)) ≈ 0.00193, which lies entirely inside (255/256, 1].
    A threshold capped at 255/256 never lands there, so the second token's count is 0.
    The reference is the softmax of the BF16 logits. Over draws = USERS * N the count is
    Binomial(draws, p); the test requires it within 4 standard deviations of draws * p
    (and therefore not zero, and not the 2x rate of a low digit that copies the high one).
    """
    gap = 6.25  # bf16-exact; P(second) = 1 / (1 + e^6.25) = 0.00193
    values = torch.full((1, 1, USERS, W), -30.0)
    values[..., 0] = 0.0
    values[..., 1] = -gap
    emp = _draw(device, values, k=2, p=1.0, n=N)
    expected = torch.softmax(values[0, 0, 0, :2].to(torch.bfloat16).float(), dim=-1)[1].item()
    draws = USERS * N
    hits = emp[:, 1].sum().item() * N
    sigma = (draws * expected * (1 - expected)) ** 0.5
    print(f"\nlow digit: P(second) expected {expected:.5f} -> {draws * expected:.0f} hits, device {hits:.0f}")
    assert hits > 0, "the tail token was never drawn: the threshold's low digit is dead (hi / 256 lattice)"
    assert (
        abs(hits - draws * expected) <= 4 * sigma
    ), f"{hits:.0f} hits vs {draws * expected:.0f} expected: the low digit is not independent of the high one"
    assert emp[:, 2:].sum() == 0


def test_sampling_softmax_gap(device):
    """Two-token rows with logit gap d: P(second) must be 1/(1+e^d). Reads out the kernel's
    softmax accuracy and the threshold resolution for small probabilities (the tail of a real
    distribution). Asserted as a ratio band where N gives enough expected hits (gap <= 6:
    >= 7 expected hits per row group at N=3000); larger gaps are printed as diagnostics."""
    gaps = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    values = torch.full((1, 1, USERS, W), -30.0)
    for u in range(USERS):
        values[0, 0, u, 0] = 0.0
        values[0, 0, u, 1] = -gaps[u % len(gaps)]
    emp = _draw(device, values, k=2, p=1.0, n=N)
    print()
    bad = []
    for i, d in enumerate(gaps):
        rows = [u for u in range(USERS) if u % len(gaps) == i]
        got = emp[rows, 1].mean().item()
        exp = 1 / (1 + torch.exp(torch.tensor(float(d)))).item()
        ratio = got / exp if exp else float("nan")
        print(f"gap {d:>4}: P(second) expected {exp:.4f}  device {got:.4f}  ratio {ratio:.2f}")
        if d <= 6 and not (0.5 <= ratio <= 1.5):
            bad.append((d, ratio))
    assert not bad, (
        f"P(second) off by more than 50% at gaps {bad} "
        "(a BF16 threshold capped at 255/256 is 0.67x at gap 4, 0.20x at gap 5, and 0 at gap 6)"
    )


def _grid_hits(numerator_bits, mass):
    """How many points of the uniform grid k / 2^numerator_bits, k = 0 .. 2^numerator_bits - 1,
    fall in the last `mass` of the unit interval. The sampler picks the second of two tokens
    when the threshold is >= 1 - mass (comparison is cum_sum > threshold)."""
    span = 1 << numerator_bits
    # k >= ceil((1 - mass) * span). Python floats are float64; span <= 2^24 is exact, and the
    # masses used here are either dyadic or a softmax probability whose product is nowhere near
    # a rounding boundary relative to 1 ulp of the grid.
    scaled = (1.0 - mass) * span
    first = int(scaled) if scaled.is_integer() else int(scaled) + 1
    return span - first


def test_host_threshold_grid_matches_the_reference():
    """No device. Fixes the pass/fail numbers used by the device tests.

    Reference for the two-candidate row in test_sampling_low_digit_reaches_the_tail:
    logits [0, -6.25] quantized to BF16 (6.25 is exact), then softmax.
    P(second) sits entirely above 255/256, so a packed-BF16 threshold draws that token 0 times.
    The same mass on the k/2^24 grid used by the kernel is within one grid cell of P(second).

    Equal-probability candidates (test_sampling_threshold_uniformity) split the same grid into
    bins whose counts differ by at most one cell, which is far inside the 4-sigma multinomial
    band the device test applies to USERS * N draws.

    A 16-bit grid still misses a token of mass 2^-17; the 24-bit grid assigns it exactly 2^-17.
    """
    logits = torch.tensor([0.0, -6.25])
    p_second = torch.softmax(logits.to(torch.bfloat16).float(), dim=-1)[1].item()
    # 255/256 is the largest BF16 value strictly below 1 that the old packed uniform could emit.
    assert 1.0 - p_second > 255.0 / 256.0
    assert _grid_hits(8, p_second) == 0

    hits_24 = _grid_hits(24, p_second)
    # One cell is 2^-24. The grid measure and the true probability differ by less than that.
    assert abs(hits_24 / (1 << 24) - p_second) < (1.0 / (1 << 24))

    # Mass 2^-17 starts past the last k/2^16 point (65535/65536 = 1 - 2^-16) and contains
    # exactly 128 points of the k/2^24 grid: 128 / 2^24 = 2^-17.
    assert _grid_hits(16, 2**-17) == 0
    assert _grid_hits(24, 2**-17) == 128

    # Device binomial band at the default draw count. A zero count is many sigma out, which is
    # why test_sampling_low_digit_reaches_the_tail fails on the BF16 threshold and passes here.
    draws = USERS * N
    mean = draws * p_second
    sigma = (draws * p_second * (1.0 - p_second)) ** 0.5
    assert mean > 30  # absence is not a plausible binomial fluctuation
    assert 0 < mean - 4 * sigma  # the device lower bound is above the BF16 count of 0

    # m equal bins. 2^24 is not always divisible by m; bin counts still differ by at most 1.
    for m in (4, 10, 20):
        span = 1 << 24
        counts = [((j + 1) * span) // m - (j * span) // m for j in range(m)]
        assert max(counts) - min(counts) <= 1
        expected = 1.0 / m
        device_sigma = (expected * (1.0 - expected) / draws) ** 0.5
        # The whole grid imbalance, spread over `draws` samples, is far below 4 sigma.
        imbalance = (max(counts) - min(counts)) / span
        assert imbalance < 4 * device_sigma
