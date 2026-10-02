# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Device-free: every Tuned value in the selector is registered, and each one with a fit equals what its fit
computes from the committed calibration data."""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tuned import REGISTRY, source_fields  # noqa: E402

ARCH = "wormhole_b0"  # the architecture whose values are the defaults in Params


def test_every_tuned_field_is_registered():
    registered = {(t.policy, t.field) for t in REGISTRY}
    missing = set(source_fields()) - registered
    assert not missing, f"Tuned fields without a calibration entry in tuned.py: {sorted(missing)}"
    stale = registered - set(source_fields())
    assert not stale, f"calibration entries for fields that no longer exist: {sorted(stale)}"


@pytest.mark.parametrize("entry", [t for t in REGISTRY if t.fit], ids=lambda t: f"{t.policy}.{t.field}")
def test_tuned_value_matches_its_fit(entry):
    data = HERE / "data" / ARCH / entry.data
    if not data.exists():
        pytest.skip(f"no {ARCH} calibration data at {data}")
    result = entry.fit(data)
    in_source = source_fields()[(entry.policy, entry.field)]
    assert in_source == pytest.approx(result["value"]), (
        f"{entry.policy}::Tuned::{entry.field} is {in_source} in the source, but its fit on {data.name} gives "
        f"{result['value']} (same choices for {result['range']}); rerun the fit and update the value"
    )
