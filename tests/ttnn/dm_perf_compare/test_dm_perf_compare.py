# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests of the comparison logic: no device, no ttnn."""

import importlib
import os
import sys

import pytest

# The repo runs pytest with --import-mode=importlib, which does not put this directory on sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import calls  # noqa: E402
import compare  # noqa: E402
import ops  # noqa: E402
import profile_csv  # noqa: E402
import render  # noqa: E402
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


def results_for(
    after_default,
    after_native,
    before=100.0,
    digest_native="d",
    control=None,
    device_ns=None,
    device_before_ns=None,
    host_ctl=100.0,
):
    """Host times are scalars per run; device values may be a scalar or one value per round."""
    cid = "c"
    doc = {
        "calls": [{"id": cid, "shape": [1], "dtype": "bfloat16", "layout": "tile", "memory_config": "DRAM", "args": {}}]
    }
    rounds = 5

    def run(impl_values):
        out = []
        for r in range(rounds):
            entry = {"calls": {cid: {"results": {}}}}
            for impl, (v, d) in impl_values.items():
                entry["calls"][cid]["results"][impl] = {"status": "ok", "best_us": v, "digest": d}
            out.append(entry)
        return out

    def device(values):
        per_round = {impl: v if isinstance(v, list) else [v] * rounds for impl, v in (values or {}).items()}
        return [{(impl, 0): {"ns": ns[r], "ops": [impl]} for impl, ns in per_round.items()} for r in range(rounds)]

    raw = {
        "time": {
            "before": run({"default": (before, "d")}),
            "after_default": run({"default": (after_default, "d")}),
            "after": run(
                {
                    "default": (after_default, "d"),
                    "native": (after_native, digest_native),
                    "default_b": (after_default if control is None else control, "d"),
                }
            ),
            "ctl_before": run({"default": (100.0, "d")}),
            "ctl_after": run({"default": (host_ctl, "d")}),
        },
        "device": {"before": device(device_before_ns), "after": device(device_ns)},
    }
    return compare.build(doc, raw, {"op": "t"})


def test_build_flags_slower_and_beaten_default():
    device = {"default": 1200.0, "native": 800.0, "codegen": 1200.0}
    row = results_for(after_default=120.0, after_native=80.0, device_ns=device, device_before_ns={"default": 1000.0})[
        "calls"
    ][0]
    assert row["q1"]["verdict"] == "slower"
    assert row["fastest_option_used"] is False
    assert row["beaten_by"] == ["native"]


def test_q1_ignores_host_change_when_device_time_matches():
    same = {"default": 1000.0}
    result = results_for(after_default=150.0, after_native=150.0, device_ns=same, device_before_ns=same)
    row = result["calls"][0]
    assert row["q1"]["verdict"] == "no change"
    assert row["q1_host"]["moved"] is True
    assert result["summary"]["q1_host_moved"] == 1
    # The same host change under a run whose control moved more is below the resolution.
    noisy = results_for(after_default=120.0, after_native=120.0, device_ns=same, device_before_ns=same, host_ctl=140.0)
    assert noisy["calls"][0]["q1_host"]["moved"] is False


def test_q1_without_before_device_time_is_not_comparable():
    row = results_for(after_default=100.0, after_native=100.0, device_ns={"default": 1000.0})["calls"][0]
    assert row["q1"] == {"comparable": False, "reason": "no device time"}


def test_q2_uses_the_device_band_not_the_host_resolution():
    device = {"default": 1000.0, "native": 800.0, "codegen": 1000.0}
    row = results_for(
        after_default=100.0, after_native=100.0, host_ctl=170.0, device_ns=device, device_before_ns={"default": 1000.0}
    )["calls"][0]
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
    result = results_for(
        after_default=100.0,
        after_native=50.0,
        digest_native="other",
        device_ns={"default": 1000.0},
        device_before_ns={"default": 1000.0},
    )
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


def test_disagreeing_device_rounds_are_marked_unreliable():
    steady = {"default": 1000.0, "native": 1000.0, "codegen": 1000.0}
    result = results_for(after_default=100.0, after_native=100.0, device_ns=steady, device_before_ns=steady)
    assert result["noise"]["reliable"] is True
    assert result["noise"]["device_band_pct"] == stats.DEVICE_MIN_PCT
    jumpy = {"default": [1000.0, 1100.0, 1000.0, 1100.0, 1000.0]}
    result = results_for(after_default=100.0, after_native=100.0, device_ns=jumpy, device_before_ns=steady)
    # Adjacent changes are +10, -9.09, +10, -9.09: the median of their absolute values.
    assert result["noise"]["device_band_pct"] == pytest.approx(9.545, abs=0.01)
    assert result["noise"]["reliable"] is False


def test_host_resolution_comes_from_the_control_over_all_calls():
    same = {"default": 1000.0}
    result = results_for(after_default=100.0, after_native=100.0, host_ctl=130.0, device_ns=same, device_before_ns=same)
    assert result["noise"]["host_resolution_pct"] == pytest.approx(30.0)


def test_committed_sweep_samples_parse():
    sweeps = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "sweeps")
    names = sorted(os.listdir(sweeps))
    assert names
    for name in names:
        doc = calls.load_calls(os.path.join(sweeps, name))
        assert doc["op"] == name[: -len(".json")]


def test_loaded_from_tree_detects_a_side_running_the_other_build():
    run = importlib.import_module("run")
    mine = {"loaded": ["/a/build/lib/libtt_metal.so", "/a/ttnn/ttnn/_ttnn.so"]}
    other = {"loaded": ["/b/build/lib/libtt_metal.so", "/a/ttnn/ttnn/_ttnn.so"]}
    assert run.loaded_from_tree("/a", [mine, None])
    assert not run.loaded_from_tree("/a", [mine, other])
    assert not run.loaded_from_tree("/a", [{}])


def render_one(**kwargs):
    doc = results_for(**kwargs)
    doc["op"] = "t"
    return doc


def test_render_lists_slower_and_beaten_calls_first():
    device = {"default": 1200.0, "native": 800.0, "codegen": 1200.0}
    doc = render_one(after_default=120.0, after_native=80.0, device_ns=device, device_before_ns={"default": 1000.0})
    text = render.render([doc])
    assert text.startswith(render.MARKER)
    assert "**Needs a look**" in text
    assert "**slower** 20.0% slower" in text
    assert "**native beats the default** by 33.3%" in text


def test_render_clean_run_has_nothing_to_look_at():
    same = {"default": 1000.0, "native": 1000.0, "codegen": 1000.0}
    text = render.render([render_one(after_default=100.0, after_native=100.0, device_ns=same, device_before_ns=same)])
    assert "Nothing needs a look" in text
    assert "0 slower, 0 faster, 1 no change" in text


def test_render_warns_when_device_rounds_disagree():
    ns = {"default": [1000.0, 1100.0, 1000.0, 1100.0, 1000.0], "native": 1000.0}
    text = render.render(
        [render_one(after_default=100.0, after_native=100.0, device_ns=ns, device_before_ns={"default": 1000.0})]
    )
    assert "device rounds disagreed" in text


def test_render_without_device_time_shows_a_reason_not_zeros():
    text = render.render([render_one(after_default=100.0, after_native=100.0)])
    assert "n/a (no device time)" in text
    assert "Needs a look" not in text


def test_render_sheds_detail_to_fit_the_limit():
    device = {"default": 1200.0, "native": 800.0, "codegen": 1200.0}
    doc = render_one(after_default=120.0, after_native=80.0, device_ns=device, device_before_ns={"default": 1000.0})
    full = render.render([doc])
    assert "<details>" in full and "```mermaid" in full
    no_detail = render.render([doc], limit=len(full) - 1)
    assert "<details>" not in no_detail and "```mermaid" in no_detail
    assert len(render.render([doc], limit=800)) <= 800
