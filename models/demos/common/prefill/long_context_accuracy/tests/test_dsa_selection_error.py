# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSA selection error (selection masses) on small hand-computed values. CPU only."""

import pytest
import torch

from models.demos.common.prefill.long_context_accuracy.dsa.selection_error import selection_masses


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
