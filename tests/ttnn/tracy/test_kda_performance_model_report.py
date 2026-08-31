# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tracy-facing literal goldens for KDA operation performance models."""

import pandas as pd
import pytest

from models.common.utility_functions import run_for_blackhole
from tracy.process_model_log import post_process_ops_log, run_device_profiler

_WORKLOAD = "pytest tests/ttnn/tracy/kda_performance_model_workload.py -q"
_REPORT_NAME = "KdaPerformanceModels"
_PM_COLUMNS = ("PM COMPUTE [ns]", "PM BANDWIDTH [ns]", "PM IDEAL [ns]")
_EXPECTED_MODELS = (
    ("SigmoidGatedRmsNormOperation", None, (94, 1936, 1936)),
    ("QkvCausalConv1dSiluOperation", None, (99, 1554, 1554)),
    ("ReduceAffineTransformsOperation", None, (6, 240, 240)),
    ("AffineExclusiveScanOperation", None, (4, 352, 352)),
    ("PrepareChunkRecurrenceOperation", None, (8, 160, 160)),
    ("RecurrentChunkScanOperation", "RecurrentChunkScanMode::RECURRENT", (6, 152, 152)),
    ("RecurrentChunkScanOperation", "RecurrentChunkScanMode::SUMMARY", (7, 144, 144)),
)

pytestmark = run_for_blackhole()


@pytest.fixture(scope="module")
def kda_report() -> pd.DataFrame:
    run_device_profiler(_WORKLOAD, _REPORT_NAME)
    return post_process_ops_log(_REPORT_NAME)


@pytest.mark.parametrize(("op_code", "attribute", "expected"), _EXPECTED_MODELS)
def test_kda_performance_model_report(
    kda_report: pd.DataFrame,
    op_code: str,
    attribute: str | None,
    expected: tuple[int, int, int],
) -> None:
    rows = kda_report[kda_report["OP CODE"] == op_code]
    if attribute is not None:
        rows = rows[rows["ATTRIBUTES"].astype(str).str.contains(attribute, regex=False)]

    assert len(rows) == 1, f"expected one {op_code} row matching {attribute!r}, found {len(rows)}"
    received = tuple(int(rows.iloc[0][column]) for column in _PM_COLUMNS)
    assert received == expected
