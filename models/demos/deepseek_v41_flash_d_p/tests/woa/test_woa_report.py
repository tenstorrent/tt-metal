# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WA.5 (OWNER 2026-11-27, CPU only, no device): the wo_a bank-local report exists with its four parts, and its
estimate for wq_b / wo_b is the one tests/woa/woa_estimate.py computes from the checkpoint's shapes and the measured
numbers. The report's text lives in tests/woa/wa5_report.md; this test checks it and writes it to
results/WA.5_report.md (a results/<task>_*.md file is the gate's own output, F292: the gate deletes it before it runs).

Records woa_report_parts (parts found, of 4), woa_est_ms_wq_b / woa_est_ms_wo_b (best estimate of the device ms SAVED
a 8192 chunk, layers 0-20, 20 calls), their high ends and the 5120-chunk values.

    scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/woa/test_woa_report.py
"""

from pathlib import Path

import pytest

from models.demos.common.bringup.core.metrics import record
from models.demos.deepseek_v41_flash_d_p.tests.woa import woa_estimate as est
from models.demos.deepseek_v41_flash_d_p.tests.woa import woa_roofline as rl

SOURCE = Path(__file__).resolve().parent / "wa5_report.md"
REPORT = rl.RESULTS / "WA.5_report.md"
KNOWN = rl.MODEL_DIR / "bringup" / "known_issues.md"

PARTS = {
    "## 1. Result": ("151.1", "130.3", "131.7", "114.2", "69.6", "3.058", "2.656", "5120", "Ladder deltas"),
    "## 2. What closed the gap": ("rect 11", "NoC distance", "in flight", "compute-bound", "bank conflicts"),
    "## 3. Would the same layout help wq_b and wo_b? ESTIMATE ONLY": ("DS41-306", "PB.MM256", "P.81", "Uncertainty"),
    "## 4. Known issue": ("DS41-",),
}


@pytest.fixture(scope="module")
def res():
    return est.run()


def test_report_parts_and_estimate(res):
    assert SOURCE.exists(), SOURCE
    text = SOURCE.read_text()
    found = 0
    for head, words in PARTS.items():
        assert head in text, f"report lacks {head!r}"
        i = text.index(head)
        j = text.find("\n## ", i + 1)
        body = text[i : j if j > 0 else len(text)]
        missing = [w for w in words if w not in body]
        assert not missing, (head, missing)
        found += 1
    # the estimate table in the report is the estimator's
    assert est.table(res) in text, "part 3's table is not woa_estimate.table()"
    wq, wo = res[("wq_b", 256)], res[("wo_b", 256)]
    for v in (wq["gain_ms_chunk"], wo["gain_ms_chunk"]):
        assert f"{v:.2f} ms" in text
    # the known-issues entry named in part 4 exists
    known = KNOWN.read_text()
    assert "[DS41-396]" in known and "DS41-396" in text
    for r in res.values():
        assert 0 <= r["gain_ms_chunk"] <= r["gain_hi_ms_chunk"] + 1e-9
        assert r["realistic_us"] < r["today_us"]
        assert r["cores"] <= rl.load()[0].cap  # rule 18: matmul cores <= MAX_WORKER_CORES
    REPORT.write_text(text)
    print()
    print(est.table(res))
    record("woa_report_parts", found)
    record("woa_est_ms_wq_b", round(wq["gain_ms_chunk"], 3))
    record("woa_est_ms_wo_b", round(wo["gain_ms_chunk"], 3))
    record("woa_est_hi_ms_wq_b", round(wq["gain_hi_ms_chunk"], 3))
    record("woa_est_hi_ms_wo_b", round(wo["gain_hi_ms_chunk"], 3))
    record("woa_est_ms_wq_b_c5120", round(res[("wq_b", 160)]["gain_ms_chunk"], 3))
    record("woa_est_ms_wo_b_c5120", round(res[("wo_b", 160)]["gain_ms_chunk"], 3))
