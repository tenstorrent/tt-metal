# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU: every broken output fails the gates.json floors on paper and ragged, in both precisions; the golden passes."""
import json

import pytest

from models.experimental.kairos_forecast.tests.perf.broken_outputs import DEFAULT_GATES, verdicts


@pytest.mark.parametrize("case", ["paper", "ragged"])
@pytest.mark.parametrize("precision", ["accuracy", "performance"])
def test_broken_outputs_fail(case, precision):
    v = verdicts(case, precision, json.loads(DEFAULT_GATES.read_text()))
    for label, r in v.items():
        print(case, precision, label, "fails:", r["fails"])
    assert not v["golden"]["fails"]
    for label in ("collapsed", "band_0.88", "band_1.12", "no_negate"):
        assert v[label]["fails"], (case, precision, label)
