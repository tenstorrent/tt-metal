# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the comparison logic: no device, no ttnn."""

import os
import sys

import pytest

# The repo runs pytest with --import-mode=importlib, which does not put this directory on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calls  # noqa: E402
import compare  # noqa: E402
import ops  # noqa: E402
import profile_csv  # noqa: E402
import stats  # noqa: E402


def make_run(values, impl="default", digest="d"):
    return {
        "calls": {
            cid: {"results": {impl: {"status": "ok", "best_us": v, "digest": digest}}} for cid, v in values.items()
        }
    }


def test_change_sign():
    assert stats.change_pct(100.0, 90.0) == pytest.approx(-10.0)
    assert stats.change_pct(100.0, 110.0) == pytest.approx(10.0)


def test_verdict_needs_size_and_agreement():
    assert stats.verdict([10, 11, 9, 12, 10], noise=2.0) == "slower"
    assert stats.verdict([-10, -11, -9, -12, -10], noise=2.0) == "faster"
    assert stats.verdict([1, 0.5, 1.5, 1, 1], noise=2.0) == "no change"
    assert stats.verdict([10, 11, -9, -12, 10], noise=2.0) == "no change"
    assert stats.verdict([10, 11, 9, 12, -10], noise=2.0) == "slower"
    assert stats.verdict([], noise=2.0) == "no data"


def test_noise_has_a_floor():
    assert stats.noise_pct({"a": [0.1, 0.0]}) == stats.MIN_NOISE_PCT
    assert stats.noise_pct({"a": [5.0, 6.0], "b": [-8.0, -7.0]}) == pytest.approx(7.5)


def test_geomean():
    assert stats.geomean_change_pct([10.0, -10.0]) == pytest.approx((((1.1 * 0.9) ** 0.5) - 1) * 100)
    assert stats.geomean_change_pct([]) is None


def test_call_file_validation_and_ids():
    doc = {
        "schema": calls.SCHEMA,
        "op": "repeat",
        "calls": [
            {"shape": [1, 1, 32, 32], "dtype": "bfloat16", "layout": "tile", "args": {"repeat_dims": [1, 1, 2, 2]}}
        ],
    }
    parsed = calls.parse_calls(doc)
    assert parsed["calls"][0]["memory_config"] == "DRAM"
    assert parsed["calls"][0]["id"].startswith("1x1x32x32|bfloat16|tile|DRAM|")
    for bad in ({"schema": "x"}, {**doc, "calls": []}, {**doc, "calls": doc["calls"] * 2}):
        with pytest.raises(calls.CallsError):  # allow-pytest.raises: host-only test, runs without the repo conftest
            calls.parse_calls(bad)
    with pytest.raises(calls.CallsError):  # allow-pytest.raises: host-only test, runs without the repo conftest
        calls.parse_calls({**doc, "calls": [{**doc["calls"][0], "dtype": "fp64"}]})


def test_control_indices_spread():
    assert calls.control_indices(10, 4) == [0, 3, 6, 9]
    assert calls.control_indices(2, 4) == [0, 1]
    assert calls.control_indices(5, 0) == []


def test_forced_entry_argument_filter():
    accepts = {"input_tensor", "repeat_dims", "memory_config"}
    assert ops.split_call_args({"repeat_dims": [1], "optional_output_tensor": None}, accepts) == {"repeat_dims": [1]}
    assert ops.split_call_args({"use_multicore": True}, {"memory_config"}) == {}
    with pytest.raises(ops.Refused):  # allow-pytest.raises: host-only test, runs without the repo conftest
        ops.split_call_args({"use_multicore": False}, {"memory_config"})
    assert ops.accepted_kwargs(type("F", (), {"__doc__": "f(a: int, *, b: x.Y = None) -> Tensor"})) == {"a", "b"}


def test_csv_reduction(tmp_path):
    path = tmp_path / "ops_perf_results_x.csv"
    rows = [
        ("BEGIN native~3#0", ""),
        ("OpA", "100"),
        ("OpB", "50"),
        ("END native~3#0", ""),
        ("BEGIN native~3#1", ""),
        ("OpA", "90"),
        ("OpB", "50"),
        ("END native~3#1", ""),
        ("OpOutside", "999"),
    ]
    path.write_text("OP CODE,DEVICE KERNEL DURATION [ns]\n" + "".join(f"{a},{b}\n" for a, b in rows))
    assert profile_csv.parse_csv(str(path)) == {("native", 3): {"ns": 140.0, "ops": ["OpA", "OpB"]}}
    assert profile_csv.find_csv(str(tmp_path)) == str(path)


def results_for(after_default, after_native, before=100.0, digest_native="d", control=None, device_ns=None):
    cid = "c"
    doc = {
        "calls": [{"id": cid, "shape": [1], "dtype": "bfloat16", "layout": "tile", "memory_config": "DRAM", "args": {}}]
    }
    rounds = 5

    def run(label_values, impl_values):
        out = []
        for r in range(rounds):
            entry = {"calls": {cid: {"results": {}}}}
            for impl, (v, d) in impl_values.items():
                entry["calls"][cid]["results"][impl] = {"status": "ok", "best_us": v, "digest": d}
            out.append(entry)
        return out

    raw = {
        "time": {
            "before": run(None, {"default": (before, "d")}),
            "after_default": run(None, {"default": (after_default, "d")}),
            "after": run(
                None,
                {
                    "default": (after_default, "d"),
                    "native": (after_native, digest_native),
                    "default_b": (after_default if control is None else control, "d"),
                },
            ),
            "ctl_before": run(None, {"default": (100.0, "d")}),
            "ctl_after": run(None, {"default": (100.0, "d")}),
        },
        "device": [
            {(impl, 0): {"ns": ns, "ops": [impl]} for impl, ns in (device_ns or {}).items()} for _ in range(rounds)
        ],
        "control_ids": [cid],
    }
    return compare.build(doc, raw, {"op": "t"})


def test_build_flags_slower_and_beaten_default():
    device = {"default": 1200.0, "native": 800.0, "codegen": 1200.0}
    row = results_for(after_default=120.0, after_native=80.0, device_ns=device)["calls"][0]
    assert row["q1"]["verdict"] == "slower"
    assert row["fastest_option_used"] is False
    assert row["beaten_by"] == ["native"]


def test_q2_ignores_host_gap_when_device_time_matches():
    # The default's entry overhead makes it slower on host time against any forced entry.
    device = {"default": 1000.0, "native": 1000.0, "codegen": 1000.0}
    row = results_for(after_default=120.0, after_native=80.0, device_ns=device)["calls"][0]
    assert row["q2_host"]["native"]["verdict"] == "faster"
    assert row["fastest_option_used"] is True
    assert row["beaten_by"] == []


def test_q2_without_device_time_is_not_comparable():
    row = results_for(after_default=100.0, after_native=100.0)["calls"][0]
    assert row["q2"]["native"] == {"comparable": False, "reason": "no device time"}
    assert row["fastest_option_used"] is None


def test_build_output_mismatch_is_not_comparable():
    result = results_for(after_default=100.0, after_native=50.0, digest_native="other")
    row = result["calls"][0]
    assert row["q2"]["native"] == {"comparable": False, "reason": "output differs"}
    assert row["q2"]["codegen"]["comparable"] is False
    assert row["fastest_option_used"] is None
    assert result["summary"]["q1"]["no change"] == 1


def test_default_uses_from_device_ops():
    device = {
        "default": {"ops": ["B", "A"]},
        "native": {"ops": ["A", "B"]},
        "codegen": {"ops": ["CodegenOp"]},
    }
    assert compare._default_uses(device) == "native"
    device["default"]["ops"] = ["X"]
    assert compare._default_uses(device) == "other: X"


def test_q2_control_counts_false_flags():
    clean = results_for(after_default=100.0, after_native=100.0)
    assert clean["calls"][0]["q2_control"]["verdict"] == "no change"
    assert clean["summary"]["q2_control"]["no change"] == 1
    flagged = results_for(after_default=100.0, after_native=100.0, control=80.0)
    assert flagged["calls"][0]["q2_control"]["verdict"] == "faster"
    assert flagged["summary"]["q2_control"]["faster"] == 1


def test_noisy_control_is_marked_unreliable():
    assert results_for(after_default=100.0, after_native=100.0)["noise"]["reliable"] is True
    assert stats.noise_pct({"c": [-12.0, -11.0, -10.0, 0.0, 0.0]}) > stats.MAX_NOISE_PCT
