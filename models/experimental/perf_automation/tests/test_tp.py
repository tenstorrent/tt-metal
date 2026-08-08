# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP gating + sizing (Increment 1). Pure-function unit tests: numbers in -> decision out, no device.
Gate = TP only when a model does NOT fit on one chip; size = smallest legal TP that fits, DP = rest."""
import pytest

from agent.tp import decide_tp, fits_on_one_chip, legal_tp_degrees, tp_regime

CAP = 100


def test_fits_on_one_chip():
    assert fits_on_one_chip(50, CAP)
    assert not fits_on_one_chip(90, CAP)


def test_tp_regime_off_on_single_chip():
    assert tp_regime(1, 200, CAP) is False


def test_tp_regime_off_when_model_fits():
    assert tp_regime(4, 50, CAP) is False


def test_tp_regime_on_when_too_big_and_mesh_present():
    assert tp_regime(4, 200, CAP) is True


def test_legal_degrees_respect_mesh_and_head_divisibility():
    assert legal_tp_degrees(4, 16, 2048) == [2, 4]
    assert legal_tp_degrees(4, 6, 1536) == [2]


@pytest.mark.parametrize(
    "weight,total,heads,hidden,expected",
    [
        (50, 4, 16, 2048, {"tp": 1, "dp": 4}),
        (120, 4, 16, 2048, {"tp": 2, "dp": 2}),
        (200, 4, 16, 2048, {"tp": 4, "dp": 1}),
        (320, 4, 16, 2048, {"tp": 4, "dp": 1}),
    ],
)
def test_decide_tp_picks_smallest_legal_fit(weight, total, heads, hidden, expected):
    assert decide_tp(weight, CAP, total, heads, hidden) == expected


def test_decide_tp_errors_when_no_legal_degree_fits():
    result = decide_tp(640, CAP, 4, 16, 2048)
    assert "error" in result and "tp" not in result


def test_decide_tp_errors_when_capacity_needs_indivisible_degree():
    result = decide_tp(320, CAP, 4, 6, 1536)
    assert "error" in result
