# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free test of PerfConfig report generation (#51244).

Drives ``PerfConfig.build_report_frame`` — the pure report-assembly seam of
``run()`` — with synthetic per-run-type results and parameters, so the report is
produced with **no chip**. Then it checks that the produced column set conforms
to the shared output schema (``perf_wide_schema.OUTPUT_SCHEMA``).

This is the exact validation the static gate only approximates: the columns come
from the real assembly code path, so it catches the optional-None and
dynamic-param cases that static reading cannot. Needs the LLK env (pandas,
ttexalens importable) but no hardware.
"""

from types import SimpleNamespace

import pandas as pd
from helpers.llk_params import ApproximationMode, DestAccumulation, PerfRunType
from helpers.perf import PerfConfig
from helpers.perf_schema import MEAN, STD, assert_unique_columns, stat_column
from helpers.perf_wide_schema import OUTPUT_SCHEMA
from helpers.test_variant_parameters import APPROX_MODE, LOOP_FACTOR, TILE_COUNT

_SCHEMA_NAMES = {c.name for c in OUTPUT_SCHEMA}


def _fake_formats():
    """A stand-in for a formats_config entry: only the attributes the assembly reads."""
    fmt = SimpleNamespace(
        unpack_A_src="Float16_b",
        unpack_B_src="Float16_b",
        unpack_A_dst="Float16_b",
        unpack_B_dst="Float16_b",
        output_format="Float16_b",
    )
    return [fmt]


def _timing_result(run_type: PerfRunType) -> pd.DataFrame:
    """What get_stats would emit for one run type: marker + mean/std timing columns."""
    rt = run_type.name
    return pd.DataFrame(
        {
            "marker": ["INIT", "TILE_LOOP"],
            stat_column(rt, MEAN): [10.0, 20.0],
            stat_column(rt, STD): [1.0, 2.0],
        }
    )


def _build(formats_config):
    results = [
        _timing_result(PerfRunType.L1_TO_L1),
        _timing_result(PerfRunType.MATH_ISOLATE),
    ]
    code_sizes = {PerfRunType.L1_TO_L1: 4096, PerfRunType.MATH_ISOLATE: 2048}
    templates = [APPROX_MODE(ApproximationMode.No)]
    runtimes = [TILE_COUNT(tile_cnt=4), LOOP_FACTOR(loop_factor=1)]
    return PerfConfig.build_report_frame(
        results,
        code_sizes,
        formats_config,
        False,  # unpack_to_dest
        DestAccumulation.No,  # dest_acc
        templates,
        runtimes,
    )


def test_report_columns_conform_to_output_schema():
    combined = _build(_fake_formats())

    # Every produced column must be a known output-schema column.
    unknown = sorted(set(combined.columns) - _SCHEMA_NAMES)
    assert not unknown, (
        f"PerfConfig produced columns not in perf_wide_schema.OUTPUT_SCHEMA: "
        f"{unknown}. Either the report changed (add them to OUTPUT_SCHEMA as "
        f"nullable) or a column is malformed."
    )

    # No duplicate headers, and the expected columns are present.
    assert_unique_columns(combined.columns, context="hw-free report")
    for expected in [
        "marker",
        "tile_cnt",
        "loop_factor",
        "approx_mode",
        "formats.input_A",
        "unpack_to_dest",
        "dest_acc",
        "TEXT_SIZE(L1_TO_L1)",
        stat_column("L1_TO_L1", MEAN),
        stat_column("MATH_ISOLATE", STD),
    ]:
        assert expected in combined.columns, f"missing expected column {expected!r}"


def test_report_without_formats_still_conforms():
    # formats_config=None → no format columns; the rest must still be schema-clean.
    combined = _build(None)
    unknown = sorted(set(combined.columns) - _SCHEMA_NAMES)
    assert not unknown, f"columns not in OUTPUT_SCHEMA: {unknown}"
    assert not any(c.startswith("formats.") for c in combined.columns)


def test_single_row_per_config():
    # One test configuration -> exactly one sweep row cross-joined onto the markers.
    combined = _build(_fake_formats())
    assert len(combined) == 2  # INIT + TILE_LOOP markers
