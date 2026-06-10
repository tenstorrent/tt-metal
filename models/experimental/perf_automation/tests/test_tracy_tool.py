"""M3 · tracy_tool pipeline tests (PLAN section 7.4).

Stage 1 (RUN) is mocked; stages 2-3 (REFINE via real tt-perf-report, TAG+BUCKET)
run for real against a small fixture cut from the real 105-col CSV.
"""

import shutil
from pathlib import Path

import pytest

from agent.tracy_tool import (
    build_buckets,
    median,
    normalize_dispatch,
    parse_lever_state,
    refine,
    tracy_tool,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ops_perf_sample.csv"

# A real ATTRIBUTES value cut from the sample matmul row.
ATTRS = (
    "{'bcast_batch': 'true'; 'compute_kernel_config': "
    "'ComputeKernelConfig(math_fidelity=LoFi;math_approx_mode=0;"
    "fp32_dest_acc_en=0;packer_l1_acc=1;dst_full_sync_en=0;"
    "throttle_level=ThrottleLevel::NO_THROTTLE)'; 'output_dtype': 'DataType::BFLOAT8_B'}"
)


def _refine_fixture(tmp_path) -> Path:
    report = tmp_path / "report.csv"
    refine(FIXTURE, report, start_signpost="start", end_signpost="stop")
    return report


def test_tracy_parse_real_schema(tmp_path):
    report = _refine_fixture(tmp_path)
    buckets = build_buckets(report, FIXTURE)
    by_id = {b["id"]: b for b in buckets}

    # Group by OP_CLASS_MAP: Matmul + MinimalMatmul collapse into one matmul bucket.
    assert by_id["matmul"]["count"] == 2
    assert by_id["reduction"]["count"] == 1  # LayerNorm
    assert by_id["eltwise"]["count"] == 1  # BinaryNg
    assert by_id["embedding"]["count"] == 1
    assert by_id["attention"]["count"] == 1  # SDPA
    assert by_id["other"]["count"] == 1  # GenericOp (TBD)

    # Tags are drawn from the closed section-4.1 vocabulary.
    mm = by_id["matmul"]["tags"]
    assert mm["op_class"] == "matmul"
    assert mm["bound"] in {"dram", "flop", "both", "slow", "host"}
    assert mm["fidelity"] in {"lofi", "hifi2", "hifi3", "hifi4", "na"}
    assert mm["grid"] in {"tiny", "partial", "full"}
    assert mm["memory"] in {"dram_interleaved", "l1_interleaved", "sharded"}
    # pct sums to ~100 across buckets.
    assert abs(sum(b["pct"] for b in buckets) - 100.0) < 1e-6


def test_attributes_join(tmp_path):
    # Unit: lever_state parsed straight from a real ATTRIBUTES string.
    lev = parse_lever_state(ATTRS)
    assert lev["math_fidelity"] == "LoFi"
    assert lev["fp32_dest_acc_en"] == "0"
    assert lev["packer_l1_acc"] == "1"
    assert lev["math_approx_mode"] == "0"

    # Integration: the matmul bucket carries a joined lever_state from raw CSV.
    report = _refine_fixture(tmp_path)
    buckets = build_buckets(report, FIXTURE)
    mm = next(b for b in buckets if b["id"] == "matmul")
    assert "math_fidelity" in mm["lever_state"]


def test_tracy_median_of_n():
    assert median([10.0, 12.0, 11.0]) == 11.0
    assert median([10.0, 12.0]) == 11.0
    with pytest.raises(ValueError):
        median([])


def test_o2o_uses_median_never_sum():
    # One huge outlier gap: the SUM would blow past the 6.5us floor, but the
    # MEDIAN stays small -> dispatch is `ok` (medians only, section 4.1).
    gaps = [100.0, 100.0, 100.0, 1_000_000_000.0]
    assert normalize_dispatch(gaps) == "ok"
    # All gaps above the floor -> gappy.
    assert normalize_dispatch([7000.0, 8000.0, 9000.0]) == "gappy"


def test_tracy_tool_orchestrates_runs_and_median(tmp_path):
    calls = {"n": 0}

    def fake_run(pcc_path, batch_size, seq_len, profiles_dir, i):
        # Stage-1 mock: stage the fixture as this iteration's raw CSV.
        raw = Path(profiles_dir) / f"raw_{i}.csv"
        shutil.copyfile(FIXTURE, raw)
        calls["n"] += 1
        return raw, [20.0, 18.0, 22.0][i]

    out = tracy_tool(
        pcc_path="models/x/test_e2e.py",
        batch_size=32,
        seq_len=384,
        runs=3,
        profiles_dir=tmp_path / "profiles",
        start_signpost="start",
        end_signpost="stop",
        run_profiled=fake_run,
    )
    assert calls["n"] == 3
    assert out["wall_ms"] == 20.0  # median(20,18,22)
    assert out["buckets"]
    assert Path(out["artifacts"]["report_csv"]).is_file()
    assert "matmul" in out["stack_report"]


def test_tracy_tool_requires_stage1(tmp_path):
    with pytest.raises(ValueError):
        tracy_tool("p", 1, 1, 1, tmp_path)


def test_device_time_units_are_physically_plausible(tmp_path):
    """Reality-pinned regression for the us-vs-ns units bug: on the fixture cut
    from a REAL capture, per-call device time must land in 0.1us..10ms. The bug
    (treating tt-perf-report's microseconds as nanoseconds) put it at ~0.05us."""
    report = _refine_fixture(tmp_path)
    buckets = build_buckets(report, FIXTURE, available_cores=64)
    mm = next(b for b in buckets if b["id"] == "matmul")
    us_per_call = mm["device_ms"] * 1000.0 / mm["count"]
    assert 0.1 < us_per_call < 10_000, f"implausible {us_per_call} us/call"
