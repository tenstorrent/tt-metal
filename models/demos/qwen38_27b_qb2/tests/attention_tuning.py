# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context attention experiment contract, independent of device imports."""

import math
import statistics

CHUNKS = (32, 64, 128, 256, 512)
CASES = ((8192, 1), (8192, 16), (55000, 1), (131072, 1), (131072, 8), (262016, 1), (262016, 4))


def geometry(length, batch):
    if batch not in (1, 4, 8, 16) or not 1 <= length <= 262016:
        raise ValueError("Unsupported attention sweep geometry")
    native_capacity = (length + 127 + 31) // 32 * 32
    aligned_capacity = (length + 127 + 511) // 512 * 512
    native_chunk = 128
    while native_capacity % native_chunk:
        native_chunk //= 2
    return dict(
        input_tokens=length,
        batch=batch,
        native_capacity=native_capacity,
        aligned_capacity=aligned_capacity,
        native_chunk=native_chunk,
        extra_pool_tokens=(aligned_capacity - native_capacity) * batch,
        pool_tokens=aligned_capacity * batch,
        # Different causal bounds expose accidental reuse of another user's
        # bound. Each user's future cache rows have a large sentinel value.
        positions=[length - 1 - 37 * user for user in range(batch)],
    )


def reference(q, key, value, page_table, positions):
    """FP32 causal reference from the actual quantized, shuffled device cache.

    One KV head is shared by six Q heads. Gather one user's pages at a time;
    never materialize a six-fold repeated 256K KV tensor on the host.
    """
    import torch

    output = torch.empty_like(q, dtype=torch.float32)
    width = q.shape[-1]
    for user, position in enumerate(positions):
        count = position + 1
        pages = page_table[user, : (count + 31) // 32].long()
        keys = key[pages, 0].reshape(-1, width)[:count].float()
        values = value[pages, 0].reshape(-1, width)[:count].float()
        scores = q[0, user].float() @ keys.T / math.sqrt(width)
        output[0, user] = scores.softmax(dim=-1) @ values
    return output


def accuracy(actual, expected):
    import torch

    actual, expected = actual.float(), expected.float()
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        return dict(passed=False, reason="Shape mismatch or nonfinite output")
    # Require every user to pass, so one corrupted slot cannot hide in a large
    # batch-wide correlation. Report absolute error alongside correlation.
    correlations = []
    for user in range(actual.shape[1]):
        a, b = actual[0, user].flatten(), expected[0, user].flatten()
        correlations.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
    error = actual - expected
    relative_rms = float(error.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12))
    per_user_rms = [
        float(error[0, user].square().mean().sqrt() / expected[0, user].square().mean().sqrt().clamp_min(1e-12))
        for user in range(actual.shape[1])
    ]
    return dict(
        passed=all(math.isfinite(pcc) and pcc >= 0.999 for pcc in correlations)
        and all(math.isfinite(rms) and rms <= 0.02 for rms in per_user_rms),
        pcc_per_user=correlations,
        relative_rms=relative_rms,
        relative_rms_per_user=per_user_rms,
        max_abs_error=float(error.abs().max()),
    )


def select_candidate(candidates, baseline_chunk, baseline_repeat_us):
    baseline = next(row for row in candidates if row["chunk"] == baseline_chunk)
    if not baseline["accuracy_passed"]:
        raise ValueError("Baseline attention failed its numerical reference")
    drift = abs(statistics.median(baseline_repeat_us) / baseline["median_traced_call_us"] - 1)
    valid = [row for row in candidates if row["accuracy_passed"]]
    fastest = min(valid, key=lambda row: row["median_traced_call_us"])
    return dict(
        candidate_chunk=fastest["chunk"],
        speedup_vs_native_chunk=baseline["median_traced_call_us"] / fastest["median_traced_call_us"],
        baseline_repeat_drift_fraction=drift,
        timing_comparison_qualified=drift <= 0.03,
        promoted_to_model=False,
        scope="Synthetic attention only; full-layer timing and model accuracy still required",
    )
