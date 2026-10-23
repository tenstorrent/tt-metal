# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The shared unary SFPU config must emit the kernel's build.h contract.

``test_sfpu_report_accuracy.py`` and the functional/perf drivers all go through
``build_eltwise_unary_sfpu_config``. A header that still carries the old
NUM_BLOCKS contract fails to compile ``eltwise_unary_sfpu_test.cpp``, and the
accuracy report swallows that (``measure`` runs with ``check=False``).
"""

from __future__ import annotations

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    MathOperation,
    PerfRunType,
)
from helpers.perf.test_schemas import PERF_TEST_SCHEMAS_QSR
from test_eltwise_unary_sfpu import (
    build_eltwise_unary_sfpu_config,
    eltwise_unary_sfpu,
)

# Template constexprs, plus the runtime-struct fields the kernel reads as params.*.
# NUM_BLOCKS / NUM_TILES_IN_BLOCK are the old contract and must not come back.
_REQUIRED = (
    "constexpr auto dest_sync = ckernel::DstSync::SyncHalf;",
    "constexpr auto PERF_RUN_TYPE = PerfRunType::L1_TO_L1;",
    "constexpr int ITERATIONS =",
    "constexpr bool STABLE_SORT =",
    "constexpr bool FUSED_SORT =",
    "std::uint32_t LOOP_FACTOR;",
    "std::uint32_t num_faces;",
    "bool UNPACK_TRANSPOSE_FACES;",
    "bool UNPACK_TRANSPOSE_WITHIN_FACE;",
)


def test_quasar_typecast_schema_starts_at_version_one():
    assert PERF_TEST_SCHEMAS_QSR["perf_eltwise_unary_typecast_quasar"]["version"] == 1


def _sweep_marks():
    import test_eltwise_unary_sfpu as unary

    return unary.test_eltwise_unary_sfpu.pytestmark


def test_sweep_dest_sync_id_is_filterable():
    marks = [
        mark for mark in _sweep_marks() if mark.args and mark.args[0] == "dest_sync"
    ]
    assert len(marks) == 1
    assert marks[0].kwargs["ids"] == ["dest_sync:Half"]


def test_unary_sfpu_build_header_matches_the_shared_kernel():
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    src = torch.zeros(32 * 32, dtype=torch.bfloat16)
    configuration = build_eltwise_unary_sfpu_config(
        formats,
        DestAccumulation.No,
        ApproximationMode.No,
        MathOperation.Abs,
        [32, 32],
        src,
        src,
        1,
        1,
    )
    header = configuration.generate_build_header()
    missing = [snippet for snippet in _REQUIRED if snippet not in header]
    assert not missing, missing
    assert "NUM_BLOCKS" not in header
    assert "NUM_TILES_IN_BLOCK" not in header


def test_collect_result_rejects_a_run_that_overwrites_l1():
    with pytest.raises(ValueError, match="collect_result"):  # allow-pytest.raises: host-only test  # fmt: skip
        eltwise_unary_sfpu(
            formats=InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
            dest_acc=DestAccumulation.No,
            approx_mode=ApproximationMode.No,
            mathop=MathOperation.Abs,
            input_dimensions=[32, 32],
            is_perf=True,
            perf_report=object(),
            collect_result=True,
        )


def test_return_golden_on_a_perf_run_needs_the_l1_result():
    with pytest.raises(ValueError, match="collect_result"):  # allow-pytest.raises: host-only test  # fmt: skip
        eltwise_unary_sfpu(
            formats=InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
            dest_acc=DestAccumulation.No,
            approx_mode=ApproximationMode.No,
            mathop=MathOperation.Abs,
            input_dimensions=[32, 32],
            is_perf=True,
            perf_report=object(),
            run_types=[PerfRunType.L1_TO_L1],
            return_golden=True,
        )
