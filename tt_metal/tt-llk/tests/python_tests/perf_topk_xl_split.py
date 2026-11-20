# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

# Import under a non-test name. pytest collects every test* binding in this module.
import test_topk_xl_split as topk_split
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES

pytestmark = [skip_for_wormhole, skip_for_quasar]


def _recorded_key(case):
    """Kernel config the perf CSV records. The input mode is stimulus only."""
    k, chunks, tail, rows, _mode, fp32, seg, ascending = case
    return (k, chunks, tail, rows, fp32, seg, ascending)


_seen = set()
PERF_CASES = []
PERF_CASE_IDS = []
for _case, _case_id in zip(topk_split.CASES, topk_split.CASE_IDS):
    _key = _recorded_key(_case)
    if _key in _seen:
        continue
    _seen.add(_key)
    PERF_CASES.append(_case)
    PERF_CASE_IDS.append(_case_id)


@pytest.mark.perf
@pytest.mark.parametrize("case", PERF_CASES, ids=PERF_CASE_IDS)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_topk_xl_split(perf_report, case, run_types, loop_factor, is_perf):
    topk_split.test_topk_xl_split(
        case,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
