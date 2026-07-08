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

from models.common.utility_functions import comp_pcc
from models.experimental.diffusion_gemma.reference.denoise_loop import DenoiseTrajectory


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Repo-standard PCC wrapper; magnitude drift is gated separately by max |Δ|."""
    _, pcc = comp_pcc(a.flatten().float(), b.flatten().float(), pcc=0.0)
    return float(pcc)


@dataclass
class TrajectoryComparison:
    """Decision-level trajectory diff (#47468).

    The harness reports every per-step decision class — not just final logits —
    because bfp8 small-probability drift can flip accept/renoise even when
    logit-PCC is high.
    """

    steps_match: bool
    halted_match: bool
    # Per-step decision diffs:
    per_step_argmax_agreement: List[float]  # clean argmax — the commit value
    per_step_sampled_agreement: List[float]  # Gumbel-max sampled token ids (under injected noise)
    per_step_accept_iou: List[float]  # accept mask IoU (intersection over union; 1.0 == identical)
    per_step_canvas_agreement: List[float]  # renoised canvas (the input to the next step)
    per_step_entropy_pcc: List[float]  # per-token entropy values, PCC per step
    per_step_entropy_max_abs: List[float]  # per-token entropy max |Δ| per step (PCC is affine-blind)
    # Aggregates:
    min_argmax_agreement: float
    min_sampled_agreement: float
    min_accept_iou: float
    min_canvas_agreement: float
    min_entropy_pcc: float
    max_entropy_abs_err: float  # worst per-token entropy |Δ| across steps
    committed_match: float
    entropy_trajectory_pcc: float  # PCC of the per-step *mean* entropy sequence (coarse summary)
    passed: bool


def _iou(a: torch.Tensor, b: torch.Tensor) -> float:
    """Bool-mask IoU. Returns 1.0 when both masks are empty (perfect "no accepts")."""
    a = a.bool()
    b = b.bool()
    union = (a | b).sum().item()
    if union == 0:
        return 1.0
    return float((a & b).sum().item()) / float(union)


def compare_trajectories(
    ref: DenoiseTrajectory,
    cand: DenoiseTrajectory,
    *,
    min_argmax_agreement: float = 1.0,
    min_sampled_agreement: float = 1.0,
    min_accept_iou: float = 1.0,
    min_canvas_agreement: float = 1.0,
    min_per_step_entropy_pcc: float = 0.99,
    max_entropy_abs_err_threshold: float = 1.0e-3,
    committed_match_threshold: float = 1.0,
    entropy_pcc_threshold: float = 0.99,
) -> TrajectoryComparison:
    """Compare a candidate denoise trajectory against the reference oracle.

    Each per-step decision class is diffed separately (#47468):
    clean argmax, Gumbel-sampled ids, accept mask, renoised canvas, per-token
    entropy. Token-for-token agreement requires the candidate to be driven with
    the reference run's exact injected noise (R5).

    Entropy is gated on BOTH PCC and an **absolute** tolerance
    (``max_entropy_abs_err_threshold``): PCC is invariant to a constant offset/scale,
    so a systematic entropy error (e.g. wrong log base, missing temperature) would
    pass PCC≈1 but is caught by the abs gate. Loosen the abs threshold for a
    bf16/bfp8 device-vs-torch comparison (default 1e-3 suits the deterministic
    injected-noise / self-compare case).
    """
    steps_match = ref.num_steps == cand.num_steps
    halted_match = ref.halted == cand.halted

    n = min(len(ref.per_step), len(cand.per_step))
    rec = lambda i: (ref.per_step[i], cand.per_step[i])  # noqa: E731

    argmax_agreement = [float((rec(i)[0].argmax == rec(i)[1].argmax).float().mean()) for i in range(n)]
    sampled_agreement = [float((rec(i)[0].sampled == rec(i)[1].sampled).float().mean()) for i in range(n)]
    accept_iou = [_iou(rec(i)[0].accept_mask, rec(i)[1].accept_mask) for i in range(n)]
    canvas_agreement = [float((rec(i)[0].canvas == rec(i)[1].canvas).float().mean()) for i in range(n)]
    entropy_pcc_per_step = [_pearson(rec(i)[0].entropy, rec(i)[1].entropy) for i in range(n)]
    entropy_abs_per_step = [float((rec(i)[0].entropy - rec(i)[1].entropy).abs().max()) for i in range(n)]

    committed_match = float((ref.committed == cand.committed).float().mean())

    ent_ref = torch.tensor([r.entropy_mean for r in ref.per_step[:n]])
    ent_cand = torch.tensor([r.entropy_mean for r in cand.per_step[:n]])
    entropy_traj_pcc = _pearson(ent_ref, ent_cand)

    min_arg = min(argmax_agreement) if argmax_agreement else 0.0
    min_smp = min(sampled_agreement) if sampled_agreement else 0.0
    min_iou = min(accept_iou) if accept_iou else 0.0
    min_can = min(canvas_agreement) if canvas_agreement else 0.0
    min_epcc = min(entropy_pcc_per_step) if entropy_pcc_per_step else 0.0
    max_eabs = max(entropy_abs_per_step) if entropy_abs_per_step else 0.0

    passed = (
        steps_match
        and halted_match
        and min_arg >= min_argmax_agreement
        and min_smp >= min_sampled_agreement
        and min_iou >= min_accept_iou
        and min_can >= min_canvas_agreement
        and min_epcc >= min_per_step_entropy_pcc
        and max_eabs <= max_entropy_abs_err_threshold
        and committed_match >= committed_match_threshold
        and entropy_traj_pcc >= entropy_pcc_threshold
    )
    return TrajectoryComparison(
        steps_match=steps_match,
        halted_match=halted_match,
        per_step_argmax_agreement=argmax_agreement,
        per_step_sampled_agreement=sampled_agreement,
        per_step_accept_iou=accept_iou,
        per_step_canvas_agreement=canvas_agreement,
        per_step_entropy_pcc=entropy_pcc_per_step,
        per_step_entropy_max_abs=entropy_abs_per_step,
        min_argmax_agreement=min_arg,
        min_sampled_agreement=min_smp,
        min_accept_iou=min_iou,
        min_canvas_agreement=min_can,
        min_entropy_pcc=min_epcc,
        max_entropy_abs_err=max_eabs,
        committed_match=committed_match,
        entropy_trajectory_pcc=entropy_traj_pcc,
        passed=passed,
    )


def assert_trajectory_matches(ref: DenoiseTrajectory, cand: DenoiseTrajectory, **kwargs) -> TrajectoryComparison:
    """Assert a candidate trajectory matches the reference; returns the comparison."""
    comparison = compare_trajectories(ref, cand, **kwargs)
    assert comparison.passed, f"denoise trajectory mismatch: {comparison}"
    return comparison
