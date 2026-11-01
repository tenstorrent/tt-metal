# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Host only: the signposted-region filter of run_model_device_perf_test_with_merge.
import pandas as pd
import pytest

from models.demos.deepseek_v3_d_p.utils import perf_utils
from models.demos.deepseek_v3_d_p.utils.perf_utils import ops_in_signposted_region

_REGION = ("MLA_START", "MLA_END")


def _ops_csv(rows):
    return pd.DataFrame(rows, columns=["OP TYPE", "OP CODE"])


def _device(code):
    return ("tt_dnn_device", code)


def _signpost(code):
    return ("signpost", code)


def test_one_region_keeps_only_its_ops():
    df = _ops_csv(
        [
            _device("Tilize"),
            _signpost("MLA_START"),
            _device("Matmul"),
            _device("RingJointSDPA"),
            _signpost("MLA_END"),
            _device("Typecast"),
        ]
    )
    assert ops_in_signposted_region(df, _REGION)["OP CODE"].tolist() == ["Matmul", "RingJointSDPA"]


@pytest.mark.parametrize(
    "rows",
    [
        # Two forwards in one profiling run, e.g. an unpinned -k axis: they would be summed.
        [
            _signpost("MLA_START"),
            _device("Matmul"),
            _signpost("MLA_END"),
            _signpost("MLA_START"),
            _device("Matmul"),
            _signpost("MLA_END"),
        ],
        # Nested regions.
        [_signpost("MLA_START"), _signpost("MLA_START"), _device("Matmul"), _signpost("MLA_END"), _signpost("MLA_END")],
        # No region.
        [_device("Matmul")],
        # A start without a stop.
        [_signpost("MLA_START"), _device("Matmul")],
    ],
    ids=["two_regions", "nested", "none", "unclosed"],
)
def test_anything_but_one_region_fails(rows, expect_error):
    with expect_error(pytest.fail.Exception, "expected one region per profiling run"):
        ops_in_signposted_region(_ops_csv(rows), _REGION)


@pytest.mark.parametrize(
    "rows",
    [
        [_signpost("MLA_START"), _signpost("MLA_END"), _device("Matmul")],
        # The stop before the start.
        [_signpost("MLA_END"), _device("Matmul"), _signpost("MLA_START")],
    ],
    ids=["empty", "stop_first"],
)
def test_no_ops_inside_fails(rows, expect_error):
    with expect_error(pytest.fail.Exception, "matched no device rows"):
        ops_in_signposted_region(_ops_csv(rows), _REGION)


def _wrapper_kwargs(**extra):
    return dict(
        command="pytest x",
        expected_device_perf_ns_per_iteration=None,
        subdir="sub",
        model_name="m",
        between_signposts=_REGION,
        **extra,
    )


def test_wrapper_rejects_several_iterations_before_profiling(monkeypatch, expect_error):
    """Only the last profiling run's CSV is merged, so a signposted measurement over several iterations is refused
    before anything runs."""
    monkeypatch.setattr(perf_utils, "run_device_perf", lambda *a, **k: pytest.fail("profiled"))
    with expect_error(pytest.fail.Exception, "needs num_iterations=1"):
        perf_utils.run_model_device_perf_test_with_merge(**_wrapper_kwargs(num_iterations=3))


def test_wrapper_rejects_a_run_with_several_regions(monkeypatch, tmp_path, expect_error):
    """One profiling run whose CSV holds three forwards (an unpinned -k axis) fails instead of summing them."""
    csv = tmp_path / "ops.csv"
    rows = [r for _ in range(3) for r in (_signpost("MLA_START"), _device("Matmul"), _signpost("MLA_END"))]
    _ops_csv(rows).assign(**{"DEVICE KERNEL DURATION [ns]": 1000.0}).to_csv(csv, index=False)
    monkeypatch.setattr(perf_utils, "run_device_perf", lambda *a, **k: {})
    monkeypatch.setattr(perf_utils, "get_latest_ops_log_filename", lambda subdir: str(csv))

    def merge_reached(df):
        raise AssertionError(f"merged {len(df)} rows from several regions")

    monkeypatch.setattr(perf_utils, "merge_device_rows", merge_reached)
    with expect_error(pytest.fail.Exception, "expected one region per profiling run"):
        perf_utils.run_model_device_perf_test_with_merge(**_wrapper_kwargs(num_iterations=1))


def test_wrapper_gates_the_merged_total_inside_the_region(monkeypatch, tmp_path):
    """A pinned single-region run on two devices: the checked value is the merged device time inside the region
    (150 ns, each op once), not the per-device rows summed (300 ns) nor the profiler's unfiltered total (5300 ns, which
    also counts the weight-load tilize outside the region)."""
    csv = tmp_path / "ops.csv"
    rows = [_device("Tilize"), _signpost("MLA_START")] + [_device("Matmul"), _device("RingJointSDPA")] * 2
    rows.append(_signpost("MLA_END"))
    _ops_csv(rows).assign(**{"DEVICE KERNEL DURATION [ns]": [5000.0, None, 50.0, 100.0, 50.0, 100.0, None]}).to_csv(
        csv, index=False
    )
    monkeypatch.setattr(perf_utils, "run_device_perf", lambda *a, **k: {"AVG DEVICE KERNEL DURATION [ns]": 5300.0})
    monkeypatch.setattr(perf_utils, "get_latest_ops_log_filename", lambda subdir: str(csv))

    def merge(df):  # the two devices' copies of each op become one row
        assert df["OP CODE"].tolist() == ["Matmul", "RingJointSDPA"] * 2, "merge saw rows outside the region"
        return df.iloc[:2]

    monkeypatch.setattr(perf_utils, "merge_device_rows", merge)
    gated, reported = [], []
    monkeypatch.setattr(perf_utils, "check_device_perf", lambda results, **k: gated.append(dict(results)) or {})
    monkeypatch.setattr(perf_utils, "prep_device_perf_report", lambda **k: reported.append(k["post_processed_results"]))
    perf_utils.run_model_device_perf_test_with_merge(
        **{**_wrapper_kwargs(), "expected_device_perf_ns_per_iteration": 150}
    )
    assert gated == [{"AVG DEVICE KERNEL DURATION [ns]": 150.0}]
    assert reported[0]["AVG DEVICE KERNEL DURATION [ns]"] == 150.0
