# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU: on every suite / bench case and both precisions, the golden passes every floor and every broken output
(broken_outputs.py) is caught by a floor or a drift check, against the fixed suite and bench records. Run it after
those records exist (it reads iter_00/suite.jsonl and bench.json); KAIROS_BROKEN_JSON writes the full table."""
import json
import os

import pytest

from models.experimental.kairos_forecast.tests.perf.broken_outputs import DEFAULT_GATES, FIXED, problems, verdicts
from models.experimental.kairos_forecast.tests.perf.check_gates import load_records
from models.experimental.kairos_forecast.tests.perf.suite import GIFT_CASES


@pytest.mark.parametrize("case", GIFT_CASES)
def test_broken_outputs_caught(case):
    gates = json.loads(DEFAULT_GATES.read_text())
    res = verdicts([case], gates, {k: load_records(p) for k, p in FIXED.items()})
    for cp, sets in res.items():
        for label, lab in sets["bench"].items():
            print(cp, label, lab)
    assert not problems(res)
