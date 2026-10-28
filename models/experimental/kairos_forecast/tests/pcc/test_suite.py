# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-model suite on real weights: generate() on every bench-golden case, both precisions, scored as gate records.
Records are appended to $KAIROS_RECORDS (JSON lines) for check_gates; the in-test asserts are the contract floors."""
import json
import os

import pytest

from models.experimental.kairos_forecast.reference import dataflow as DF
from models.experimental.kairos_forecast.tests.helpers import FLOORS, FLOORS_PERFORMANCE
from models.experimental.kairos_forecast.tests.perf.check_gates import session_stamp
from models.experimental.kairos_forecast.tests.perf.suite import run_case
from models.experimental.kairos_forecast.tt.common import Precision
from models.experimental.kairos_forecast.tt.model import TtKairos

STAMP = session_stamp()  # v1.23: taken once, at collection (the session start); every record carries it

CASES = os.environ.get("KAIROS_CASES", "single,single_plain,single_short,ragged,single_long_horizon,long_horizon,paper")


@pytest.fixture(scope="module")
def W():
    return DF.load_weights()


@pytest.mark.parametrize("prec", ["accuracy", "performance"])
@pytest.mark.parametrize("case", CASES.split(","))
def test_suite(device, W, case, prec):
    m = TtKairos(device, W, getattr(Precision, prec)())
    recs, _ = run_case(m, case, prec, collapsed=case == "paper")
    floors = FLOORS if prec == "accuracy" else FLOORS_PERFORMANCE
    stamp = STAMP
    for r in recs:
        r.update(stamp)
        keys = ("pcc_normalized", "pcc_normalized_full", "pcc_final", "pcc_spread", "pcc_row_min", "spread_ratio_p01",
                "spread_ratio_p99", "abs_err_max", "route_mismatch_rows", "bias_max_abs_diff", "min_top3_margin")
        print("RECORD", r["name"], prec, " ".join(f"{k}={r[k]:.6g}" for k in keys if k in r), flush=True)
        if os.environ.get("KAIROS_RECORDS"):
            with open(os.environ["KAIROS_RECORDS"], "a") as f:
                f.write(json.dumps(r) + "\n")
        assert r["route_mismatch_rows"] == 0 and r["bias_max_abs_diff"] <= FLOORS["bias_max_abs_diff"], r["name"]
        assert r["pcc_normalized"] >= floors["pcc_normalized"], (r["name"], r["pcc_normalized"])
        assert r["pcc_final"] >= floors["pcc_final"], (r["name"], r["pcc_final"])
        if "collapsed" in r:
            c = r["collapsed"]
            print("COLLAPSED", r["name"], " ".join(f"{k}={v:.6g}" for k, v in c.items() if isinstance(v, float)))
