# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA state error on small hand-computed values. CPU only."""

import math

import pytest
import torch

from models.demos.common.prefill.long_context_accuracy.kda.scorer import state_errors


def test_state_errors_hand_case() -> None:
    """Two heads of 1x2 recurrent state: head 0 exact (3, 4), head 1 (0, 2) read as (0, 3): ‖e‖ 1 over ‖y‖ √29, RMS
    error √(1/4), RMS √(29/4), head rels 0 and 1/2. Conv: q exact, k off by 1 in (2, 0), v off by 1 in (4, 0)."""
    want = {"recurrent": torch.tensor([[[3.0, 4.0]], [[0.0, 2.0]]])}
    got = {"recurrent": torch.tensor([[[3.0, 4.0]], [[0.0, 3.0]]])}
    for c, (w, g) in {
        "q": ([1.0, 0.0], [1.0, 0.0]),
        "k": ([2.0, 0.0], [3.0, 0.0]),
        "v": ([4.0, 0.0], [5.0, 0.0]),
    }.items():
        want[f"{c}_convolution"], got[f"{c}_convolution"] = torch.tensor(w), torch.tensor(g)
    e = state_errors(got, want)
    assert e["recurrent_rel"] == pytest.approx(1 / math.sqrt(29))
    assert e["recurrent_rms_err"] == pytest.approx(0.5) and e["recurrent_rms_ref"] == pytest.approx(math.sqrt(29 / 4))
    assert e["recurrent_head_rel_max"] == pytest.approx(0.5) and e["recurrent_head_rel_median"] == 0.0
    assert e["conv_channel"] == "k" and e["conv_rel"] == pytest.approx(0.5)
    assert e["conv_rms_err"] == pytest.approx(math.sqrt(0.5)) and e["conv_rms_ref"] == pytest.approx(math.sqrt(2.0))
