# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The ill-conditioned positions and frames the PCC gates skip stay few, in range, and documented.

A regenerated fixture that excluded a large share of a prompt would quietly hollow out the gates, so
the share is bounded here. The fixture comes from make_conditioning_fixture.py (bring-up tooling, see
the README), which runs a CPU proxy of the device's precision and flags the positions whose PCC
deficit far exceeds the median.

Run:
    pytest -svv models/experimental/voxtral_tts/tests/test_conditioning_fixture.py
"""

from models.experimental.voxtral_tts.tests.reference_helpers import (
    case_ids,
    conditioning_fixture,
    fixture_cases,
    ill_conditioned_frames,
    ill_conditioned_positions,
    real_frames_long,
)

MAX_SHARE = 0.05  # of a prompt's positions, or of its decode frames
HORIZON = 64  # the decode frames the fixture covers (test_backbone_decode_pcc.py's sweep)


def test_every_case_is_covered_and_the_rule_is_recorded():
    fx = conditioning_fixture()
    ids = {str(c) for c in case_ids()}
    assert set(fx["prefill"]) == ids and set(fx["decode"]) == ids, "regenerate the fixture for every case"
    assert "rule" in fx and "make_conditioning_fixture.py" in fx["about"]


def test_prefill_exclusions_are_few_in_range_and_spare_the_last_position():
    for ci, case in enumerate(fixture_cases()):
        P = len(case["ids"])
        ill = ill_conditioned_positions(ci)
        assert all(0 <= p < P for p in ill), f"case {ci}: position out of range in {sorted(ill)}"
        assert len(ill) <= max(1, MAX_SHARE * P), f"case {ci}: {len(ill)} of {P} positions excluded"
        assert P - 1 not in ill, f"case {ci}: the last position is what the flow model consumes"


def test_decode_exclusions_are_few_and_in_range():
    for ci in case_ids():
        T = min(HORIZON, real_frames_long(ci).shape[0])
        ill = ill_conditioned_frames(ci)
        assert all(0 <= t < T for t in ill), f"case {ci}: frame out of range in {sorted(ill)}"
        assert len(ill) <= max(1, MAX_SHARE * T), f"case {ci}: {len(ill)} of {T} frames excluded"
