# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Distribution-level check of ttnn.sampling: with fixed candidate logits, many seeded draws
must reproduce the reference distribution (softmax -> top-k -> top-p (nucleus incl. the
crossing token) -> renormalise). Reports per-user total variation and the realised
probability of tail tokens, which is where a quantised random threshold would show."""

import os

import pytest
import torch
import torch.nn.functional as F

import ttnn

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
    histogram over equal-width bins of [0, 1). A uniform threshold gives 1/m everywhere."""
    for m in (4, 10, 20):
        values = torch.full((1, 1, USERS, W), -30.0)
        values[..., :m] = 0.0
        emp = _draw(device, values, k=m, p=1.0, n=N)
        hist = emp[:, :m].mean(0)  # averaged over the 32 users' streams
        print(
            f"\nm={m} (expected {1/m:.4f} each): "
            + " ".join(f"{h:.3f}" for h in hist)
            + f"  | max rank drawn: {int((emp[:, :m].sum(0) > 0).nonzero().max())}"
        )
    values = torch.full((1, 1, USERS, W), -30.0)
    values[..., :20] = 0.0
    emp = _draw(device, values, k=20, p=0.95, n=N)
    print(f"m=20 with p=0.95: " + " ".join(f"{h:.3f}" for h in emp[:, :20].mean(0)))


def test_sampling_softmax_gap(device):
    """Two-token rows with logit gap d: P(second) must be 1/(1+e^d). Reads out the kernel's
    softmax accuracy for small probabilities (the tail of a real distribution)."""
    gaps = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    values = torch.full((1, 1, USERS, W), -30.0)
    for u in range(USERS):
        values[0, 0, u, 0] = 0.0
        values[0, 0, u, 1] = -gaps[u % len(gaps)]
    emp = _draw(device, values, k=2, p=1.0, n=N)
    print()
    for i, d in enumerate(gaps):
        rows = [u for u in range(USERS) if u % len(gaps) == i]
        got = emp[rows, 1].mean().item()
        exp = 1 / (1 + torch.exp(torch.tensor(float(d)))).item()
        print(
            f"gap {d:>4}: P(second) expected {exp:.4f}  device {got:.4f}  ratio {got/exp if exp else float('nan'):.2f}"
        )
