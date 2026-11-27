# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context accuracy of every registered case (``cases.ALL_CASES``) at every context length: the case's Block on
the device, teacher-fed from its FP64 golden chunk by chunk, scored against it and gated by the case's bars
(``common/prefill/long_context_accuracy/run.run_case``, which also logs progress and the result line).

Each case opens its own placement (mesh shape and fabric) and skips without its checkpoint. Test ids are
``<case id>-<context length in rows>``.
"""

from __future__ import annotations

import pytest

from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.long_context_accuracy.run import run_case
from models.demos.deepseek_v3_d_p.tests.long_context_accuracy.cases import ALL_CASES

# The longest case (DSA layer 3 at 1M rows) took ~13 min before the restructure; pytest.ini's 300 s default is short.
pytestmark = [
    pytest.mark.skipif(not is_blackhole(), reason="Kimi-K3 and K3.1 run on Blackhole only"),
    pytest.mark.timeout(1800),
]


def _params() -> list:
    params = []
    for case in ALL_CASES:
        mesh_shape, device_params = case.placement.values
        params += [
            pytest.param(case, rows, mesh_shape, device_params, marks=case.placement.marks, id=f"{case.id}-{rows}")
            for rows in case.context_lengths
        ]
    return params


@pytest.mark.parametrize(
    "case, context_len, mesh_device, device_params", _params(), indirect=["mesh_device", "device_params"]
)
def test_long_context_accuracy(case, context_len, mesh_device, device_params) -> None:
    result = run_case(case, context_len, mesh_device, case.checkpoint())
    assert result.passed, result.violations
