# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-only checks of vision_helpers, on which the vision device tests rest: check_close, which gives each of them its
verdict, and parametrize_vision_grids, which picks the grids they run. No device.

The 1-card leg selects these tests by name (test_vision_), as they have no mesh id.

Run:
    pytest models/demos/minimax_m3/tests/unit/test_vision_helpers.py
"""
from __future__ import annotations

import torch

from .vision_helpers import (
    ATTENTION_CI_GRIDS,
    CI_GRIDS,
    MAX_REL_L2_AT_GRID,
    VISION_GRIDS,
    check_close,
    max_rel_l2,
    parametrize_vision_grids,
)

MAX_REL_L2 = 0.004  # a bound of the size of the tightest ones in vision_helpers.MAX_REL_L2


def _with_last_value(x: torch.Tensor, value: float) -> torch.Tensor:
    out = x.clone()
    out.view(-1)[-1] = value
    return out


def test_vision_check_close():
    """check_close passes an output within its bounds, with no failure details on its line, and fails an output
    scaled by 1.01 or by 0.5, which have a PCC of 1 and pass on a loose relL2 bound, and outputs with one NaN or one
    Inf value. A failure's line gives the bounds, the gain and the NaN and Inf counts."""
    generator = torch.Generator().manual_seed(0)
    ref = torch.randn(256, 64, generator=generator)
    good = ref + 1e-3 * torch.randn(256, 64, generator=generator)  # relative L2 error about 0.001

    line, passed = check_close("good", ref, good, 0.9999, MAX_REL_L2)
    assert passed and line.startswith("[good] ") and line.endswith("relL2=0.0010"), line
    for name, out, details in (
        ("gain", 1.01 * ref, "gain 1.0100, 0 NaN and 0 Inf in 16384 values"),
        ("half", 0.5 * ref, "gain 0.5000, 0 NaN and 0 Inf in 16384 values"),
        ("nan", _with_last_value(good, float("nan")), "1 NaN and 0 Inf in 16384 values"),
        ("inf", _with_last_value(good, float("inf")), "0 NaN and 1 Inf in 16384 values"),
    ):
        line, passed = check_close(name, ref, out, 0.9999, MAX_REL_L2)
        assert not passed, line
        assert f"needs PCC >= 0.9999 and relL2 <= {MAX_REL_L2}" in line and line.endswith(details), line
        if name in ("gain", "half"):
            assert check_close(name, ref, out, 0.9999, 1.0)[1], f"{name} fails on PCC"


def test_vision_parametrize_rejects_unknown_grid(expect_error):
    """parametrize_vision_grids takes the CI grids, which are all in VISION_GRIDS, and raises ValueError for a grid
    id that VISION_GRIDS lacks, whose tests would otherwise drop out of the leg without a sign."""
    assert set(ATTENTION_CI_GRIDS) <= VISION_GRIDS.keys()
    for ci_grids in (CI_GRIDS, ATTENTION_CI_GRIDS):
        parametrize_vision_grids(ci_grids)
    with expect_error(ValueError, r"grid ids \['h40w45'\] of ci_grids are not among the \d+ grids"):
        parametrize_vision_grids(("h16w16", "h40w45"))


def test_vision_max_rel_l2_at_grid():
    """Each per-grid bound names a grid of VISION_GRIDS and a check, loosens that check's bound, and only at its grid."""
    for (check, grid), bound in MAX_REL_L2_AT_GRID.items():
        others = [thw for name, thw in VISION_GRIDS.items() if (check, name) not in MAX_REL_L2_AT_GRID]
        default = max_rel_l2(check, others[0])  # KeyError for a check that MAX_REL_L2 lacks
        assert max_rel_l2(check, VISION_GRIDS[grid]) == bound > default, (check, grid)
        assert all(max_rel_l2(check, thw) == default for thw in others), (check, grid)
