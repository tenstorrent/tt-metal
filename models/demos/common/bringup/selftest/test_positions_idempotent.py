# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""run_positions raises target.seq in memory; a second run in the same process must test the same positions
(Xing X.3: the precompile pass then the real pass doubled them, 0..204800 became 0..819200)."""

from models.demos.common.bringup.core.spec import Spec
from models.demos.common.bringup.testing import positions as P

SPEC = "models/demos/xing40_a4b_d_p/bringup/spec.yaml"


def test_positions_stay_the_same_after_the_target_is_raised():
    s = Spec.load(SPEC)
    if s.get("perf.positions"):
        s.data["perf"].pop("positions")
    first = P.positions(s)
    chunk = int(s.get("target.chunk"))
    s.data["target"]["seq"] = max(P._target_seq(s), max(first) + chunk)  # what run_positions does
    assert P.positions(s) == first
    assert first[-1] == 4 * (int(s.data["target"]["_seq_before_positions"]) - chunk)
