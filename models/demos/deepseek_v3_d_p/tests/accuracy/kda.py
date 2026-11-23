# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA accuracy against a reference: the carried state at a segment end, absolute and relative.

The KDA output is a continuous product, scored by ``windowed.errors``. The state is scored here, once per
segment end, since it carries the context: its error growth is the stateful family's growth signal.
"""

from __future__ import annotations

import math

import torch


def _abs_rel(got: torch.Tensor, want: torch.Tensor) -> tuple[float, float, float]:
    """(``‖got − want‖ / ‖want‖``, RMS error, RMS of ``want``) over all elements, in FP64."""
    err, ref = got.double() - want.double(), want.double()
    n = ref.numel()
    e2, r2 = float(err.square().sum()), float(ref.square().sum())
    return (math.sqrt(e2 / r2) if r2 else (0.0 if e2 == 0 else math.inf)), math.sqrt(e2 / n), math.sqrt(r2 / n)


def state_errors(got: dict[str, torch.Tensor], want: dict[str, torch.Tensor]) -> dict:
    """A KDA state ``got`` (``recurrent`` ``[..., heads, k, v]`` and ``{q,k,v}_convolution``) against ``want``:

    * ``recurrent_rel``, ``recurrent_rms_err``, ``recurrent_rms_ref`` over the whole recurrent state, and
      ``recurrent_head_rel_max`` / ``_median`` over heads;
    * ``conv_rel``: the worst of the q, k, v convolution states' relative errors, with that one's ``conv_channel``,
      ``conv_rms_err`` and ``conv_rms_ref`` (so ``conv_rel = conv_rms_err / conv_rms_ref``)."""
    rel, err, ref = _abs_rel(got["recurrent"], want["recurrent"])
    g, w = got["recurrent"].double().flatten(-2), want["recurrent"].double().flatten(-2)
    head = (g - w).norm(dim=-1) / w.norm(dim=-1)
    conv = max(((*_abs_rel(got[f"{c}_convolution"], want[f"{c}_convolution"]), c) for c in "qkv"), key=lambda t: t[0])
    return {
        "recurrent_rel": rel,
        "recurrent_rms_err": err,
        "recurrent_rms_ref": ref,
        "recurrent_head_rel_max": float(head.max()),
        "recurrent_head_rel_median": float(head.median()),
        "conv_rel": conv[0],
        "conv_rms_err": conv[1],
        "conv_rms_ref": conv[2],
        "conv_channel": conv[3],
    }
