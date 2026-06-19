# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Denoise-trajectory comparison harness (#47468).

The PCC harness must validate the diffusion *decisions* — per-step Gumbel-max
argmax agreement, the multi-step trajectory, and the committed tokens — not just
final logits, because bfp8 small-probability drift can flip accept/renoise
(plan.md §7). This module compares a candidate ``DenoiseTrajectory`` (from the
device loop / HF reference, once available) against the pure-torch reference
oracle (``reference/denoise_loop.py``).

It is intentionally ttnn-free and CPU-only so it is importable everywhere. Raw
device *tensor* PCC (e.g. final logits / per-layer activations) is checked
separately with ``tests/ttnn/utils_for_testing.assert_with_pcc`` — the canonical
util the plan references — inside the device tests that own those tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two 1-D sequences; 1.0 if both are constant-equal."""
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:  # one (or both) constant: defined as perfect iff identical
        return 1.0 if torch.equal(a, b) else 0.0
    return float((a @ b) / denom)


@dataclass
class TrajectoryComparison:
    steps_match: bool
    halted_match: bool
    per_step_argmax_agreement: List[float]  # fraction of positions matching, per step
    min_argmax_agreement: float
    committed_match: float  # fraction of committed tokens matching
    entropy_trajectory_pcc: float  # PCC of the per-step mean-entropy sequence
    passed: bool


def compare_trajectories(
    ref: DenoiseTrajectory,
    cand: DenoiseTrajectory,
    *,
    min_argmax_agreement: float = 1.0,
    committed_match_threshold: float = 1.0,
    entropy_pcc_threshold: float = 0.99,
) -> TrajectoryComparison:
    """Compare a candidate denoise trajectory against the reference oracle."""
    steps_match = ref.num_steps == cand.num_steps
    halted_match = ref.halted == cand.halted

    n = min(len(ref.per_step), len(cand.per_step))
    agreement = [float((ref.per_step[i].argmax == cand.per_step[i].argmax).float().mean()) for i in range(n)]
    min_agreement = min(agreement) if agreement else 0.0

    committed_match = float((ref.committed == cand.committed).float().mean())

    ent_ref = torch.tensor([r.entropy_mean for r in ref.per_step[:n]])
    ent_cand = torch.tensor([r.entropy_mean for r in cand.per_step[:n]])
    entropy_pcc = _pearson(ent_ref, ent_cand)

    passed = (
        steps_match
        and halted_match
        and min_agreement >= min_argmax_agreement
        and committed_match >= committed_match_threshold
        and entropy_pcc >= entropy_pcc_threshold
    )
    return TrajectoryComparison(
        steps_match=steps_match,
        halted_match=halted_match,
        per_step_argmax_agreement=agreement,
        min_argmax_agreement=min_agreement,
        committed_match=committed_match,
        entropy_trajectory_pcc=entropy_pcc,
        passed=passed,
    )


def assert_trajectory_matches(ref: DenoiseTrajectory, cand: DenoiseTrajectory, **kwargs) -> TrajectoryComparison:
    """Assert a candidate trajectory matches the reference; returns the comparison."""
    comparison = compare_trajectories(ref, cand, **kwargs)
    assert comparison.passed, f"denoise trajectory mismatch: {comparison}"
    return comparison
