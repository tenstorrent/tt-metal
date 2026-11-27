# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSA selection error (selection masses, error split) on small hand-computed values. CPU only."""

import pytest
import torch

from models.demos.common.prefill.long_context_accuracy.dsa.selection_error import error_split, selection_masses


def test_selection_masses_hand_case() -> None:
    """Reference keys 5, 6, 7 (best first) with weights .6 .3 .1 in one head; the device picks 5, 6, 9 with weights
    .5 .3 .2: key 7 missed (mass .1, the bottom key too), key 9 an intruder (mass .2)."""
    m = selection_masses(
        torch.tensor([[0.6, 0.3, 0.1]]),
        torch.tensor([[0.5, 0.3, 0.2]]),
        torch.tensor([5, 6, 7]),
        torch.tensor([5, 6, 9]),
    )
    assert (m["missed"], m["intruders"]) == (1, 1)
    assert m["missed_mass"].tolist() == pytest.approx([0.1]) and m["bottom_mass"].tolist() == pytest.approx([0.1])
    assert m["intruder_mass"].tolist() == pytest.approx([0.2]) and m["missed_key_max"] == pytest.approx(0.1)
    none = selection_masses(torch.tensor([[1.0]]), torch.tensor([[1.0]]), torch.tensor([5]), torch.tensor([5]))
    assert none["missed_key_max"] == 0.0 and none["missed_mass"].tolist() == [0.0]


def test_error_split_hand_case() -> None:
    """want 0, the reference over the device's selection (1, 0), the device (1, 1): selection part (1, 0) and rest
    (0, 1) are orthogonal and share the squared error 2 equally."""
    s = error_split(torch.zeros(2), torch.tensor([1.0, 0.0]), torch.tensor([1.0, 1.0]))
    assert s == {"selection_rest_cos": 0.0, "selection_share": 0.5, "rest_share": 0.5}
    # (3, 0) then (3, 4): orthogonal, shares 9/25 and 16/25; (3, 0) then (6, 0): parallel (cos 1), 9/36 each and the
    # cross term the other half
    zero, sel = torch.zeros(2, dtype=torch.float64), torch.tensor([3.0, 0.0], dtype=torch.float64)
    assert error_split(zero, sel, torch.tensor([3.0, 4.0], dtype=torch.float64)) == pytest.approx(
        {"selection_rest_cos": 0.0, "selection_share": 9 / 25, "rest_share": 16 / 25}
    )
    assert error_split(zero, sel, torch.tensor([6.0, 0.0], dtype=torch.float64)) == pytest.approx(
        {"selection_rest_cos": 1.0, "selection_share": 0.25, "rest_share": 0.25}
    )
