# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Accuracy limits from the noise floor, never a fixed bar (F76).

Owner rule (DeepSeek-V4.1 run1): no fixed accuracy thresholds such as PCC 0.99 or a 1 % cap; every limit derives from
the noise floor of the reference's dtypes on the CPU op. A fixed bar either lets real errors through or, when the
reference's own dtypes cannot meet it, makes agents raise the device precision above the reference (exact paths, fp32
where the reference uses bf16) to clear it.

The noise floor of a step is its precision model (component_checks.precision_model): the CPU step on bf16 inputs with
every float intermediate rounded to bf16 and a bf16 output. A limit is calib (component_calib; component_calib_low for
tests.low_precision_kinds) x the precision model's miss on the same comparison (vs the CPU step on the same inputs, or
vs the golden), never below what one bf16 rounding of the output shows (an exact step) or one element of an integer
output. For a correlation the miss is 1 - pcc ~ rel^2 / 2, so calib applies squared.

A gate metric written ``>= limit`` in tasks.yaml is compared with the ``<metric>_limit`` the test records
(core/gate.check_metrics).
"""

from __future__ import annotations

import torch

from models.demos.common.bringup.core import metrics


def value(mode: str, got: torch.Tensor, want: torch.Tensor) -> float:
    """The comparison of harness.compare, without recording: pcc, match (equal elements), topk_overlap (per-row set
    overlap, order-free)."""
    if got.numel() != want.numel():
        return 0.0
    got = got.reshape(want.shape)
    if mode == "pcc":
        return metrics.pcc(got.float(), want.float())
    if mode == "match":
        return (got == want).float().mean().item()
    if mode == "topk_overlap":
        w, g = want.reshape(-1, want.shape[-1]), got.reshape(-1, got.shape[-1])
        return (
            torch.tensor([len(set(a.tolist()) & set(b.tolist())) / max(len(set(b.tolist())), 1) for a, b in zip(g, w)])
            .mean()
            .item()
        )
    raise ValueError(f"compare mode {mode!r}")


def gate_limit(mode: str, model_v: float, res_v: float, k: float, n: int) -> float:
    """The lower limit of a gate metric: 1 - calib x the miss of the precision model (model_v) or of one bf16 rounding
    of the output (res_v), whichever is larger; calib squared for pcc; an integer comparison never asks for fewer
    misses than one element of n."""
    miss = max(1.0 - model_v, 1.0 - res_v, 0.0 if mode == "pcc" else 1.0 / max(n, 1))
    return max(0.0, 1.0 - (k * k if mode == "pcc" else k) * miss)


def bf16_res(mode: str, ref: torch.Tensor) -> float:
    """The metric of ``ref`` rounded to bf16 vs ``ref``: the closest a correct bf16 output can get (1 for integers)."""
    if not ref.is_floating_point():
        return 1.0
    return value(mode, ref.float().to(torch.bfloat16).float(), ref.float())
