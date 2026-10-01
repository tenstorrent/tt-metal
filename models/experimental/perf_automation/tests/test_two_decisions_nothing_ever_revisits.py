# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Two choices this tool makes once and never looks at again.

FUSION is decided statically, before any measurement, and it is paid for with a dispatch and a
memory round-trip it avoids. A traced run has already removed the dispatch for everyone -- the
program replays from a capture -- so under trace the fused op is defending a smaller advantage than
the one it was chosen for, while still doing two jobs in one kernel. Nothing re-opened that.

A HAND-WRITTEN KERNEL only ever beat the stock ttnn op OF ITS DAY. The ladder pushes one way --
knobs, structural, then author tt-lang or C++ -- and once a kernel wins it holds its place forever,
because asking the opposite question is not a rung. ttnn ships faster ops over time; a kernel written
against an older one can become the slow path with nothing in the tool able to notice.

The join is what these tests are mostly about, because the join is what the last three structural
gates got wrong: their logic was pinned to the millimetre and the wiring between capture and gate was
never exercised, so all three read an empty work queue and fired zero times. So: the fusion signal is
followed from the raw ATTRIBUTES a real capture produced, through top_ops, through residual_report,
into the gate -- rather than handing the gate a dict shaped the way it likes.
"""

import importlib.util
import json
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))

from agent import roofline as R  # noqa: E402
from agent.tracy_tool import build_buckets, parse_fused_ops, refine  # noqa: E402

_SRC = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
FIXTURE = Path(__file__).parent / "fixtures" / "ops_perf_sample.csv"
HW = {"dram_bw_gbps": 512.0, "worker_cores": 110, "mesh_chips": 1, "peak_tflops_per_core": {"lofi": 4.0, "hifi4": 1.0}}

# Cut from the sample rows. The matmul declares its fusion INSIDE program_config; the norm declares
# it at the top level; the eltwise spells "none" a third way again. All three are unfused here, which
# is the case that must not read as fused.
ATTRS_MATMUL = (
    "{'bcast_batch': 'true'; 'program_config': "
    "'MatmulMultiCoreReuseMultiCast1DProgramConfig(compute_with_storage_grid_size=11-10;in0_block_w=1;"
    "fuse_batch=0;fused_activation=std::nullopt;mcast_in0=1;untilize_out=0)'; "
    "'output_dtype': 'DataType::BFLOAT16'}"
)
ATTRS_NORM = "{'eps': '1e-05'; 'fused_activation': 'std::nullopt'; 'norm_type': 'LayerNormType::LAYERNORM'}"
ATTRS_ELTWISE = "{'lhs_activations': '{}'; 'post_activations': '{}'; 'rhs_activations': '{}'; 'scalar': '32'}"
# The same rows with something actually fused in.
ATTRS_MATMUL_FUSED = ATTRS_MATMUL.replace("fused_activation=std::nullopt", "fused_activation=UnaryWithParam(GELU)")
ATTRS_ELTWISE_FUSED = ATTRS_ELTWISE.replace("'post_activations': '{}'", "'post_activations': '{RELU}'")


def _mcp():
    spec = importlib.util.spec_from_file_location("_pm_revisit", _PA / "cc_optimize" / "perf_mcp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _wins_by_flag(m, monkeypatch):
    """Make `is_win` mean the flag on the row, so a test can state which attempts were wins."""

    class _L:
        @staticmethod
        def is_win(a):
            return bool(a.get("beat_baseline"))

    monkeypatch.setattr(m, "_ledger", lambda: _L())
    monkeypatch.setattr(m, "_load_attempts", lambda: [])


# ------------------------------------------------- 10: reading the fusion off the capture


def test_the_three_ways_a_real_capture_says_nothing_is_fused():
    """std::nullopt, {} and a program_config nested one level down. All mean unfused."""
    assert parse_fused_ops(ATTRS_MATMUL) == ()
    assert parse_fused_ops(ATTRS_NORM) == ()
    assert parse_fused_ops(ATTRS_ELTWISE) == ()
    assert parse_fused_ops("") == ()


def test_a_fused_op_is_read_out_of_the_same_field():
    assert parse_fused_ops(ATTRS_MATMUL_FUSED) == ("UnaryWithParam(GELU)",)
    assert parse_fused_ops(ATTRS_ELTWISE_FUSED) == ("{RELU}",)


def test_it_is_not_keyed_on_a_list_of_attribute_names():
    """The four fusion keys in one real capture are already four spellings; the next is not knowable.

    Matched on the shape of the key so an unseen one still reads, which is the difference between
    this and a table that is correct until ttnn adds a field.
    """
    assert parse_fused_ops("{'gelu_activations': 'X'}") == ("X",)
    assert parse_fused_ops("{'some_future_activation': 'Y'}") == ("Y",)
    # ...without turning every attribute into a fusion.
    assert parse_fused_ops("{'math_approx_mode': '0'; 'output_dtype': 'DataType::BFLOAT16'}") == ()


def test_the_signal_survives_the_whole_capture_path(tmp_path):
    """From the raw CSV a profiler wrote, through refine and bucketing, onto the op rows.

    The step the last three gates were missing. A parser that works on a string proves nothing if
    the field never reaches the profile, so this asserts on the real fixture rather than on a dict
    built to suit.
    """
    report = tmp_path / "report.csv"
    refine(FIXTURE, report, start_signpost="start", end_signpost="stop")
    rows = [o for b in build_buckets(report, FIXTURE) for o in b["top_ops"]]
    assert rows, "fixture produced no op rows"
    assert all("fused" in o for o in rows), "the fused field does not reach top_ops"
    # Every op in this capture is unfused, and each one says so differently.
    assert all(o["fused"] == [] for o in rows)


def test_the_roofline_carries_it_to_the_work_queue():
    """open_ops is what the gate reads; a field dropped here never reaches it."""
    prof = _profile_fused()
    ops = R.residual_report(prof, HW).get("open_ops") or []
    assert ops, "no open ops to carry anything"
    assert any(o.get("fused") for o in ops), "fused did not survive residual_report"


# ------------------------------------------------- 10: the gate


def _op(name, ms, count=30, fused=None, bound="flop"):
    return {
        "op_code": "MatmulDeviceOperation %s" % name,
        "shape": "512x3840 @ 3840x15360",
        "device_ms": ms,
        "count": count,
        "bytes": 1e9,
        "cores": 8,
        "fidelity": "lofi",
        "grid": "partial",
        "memory": "dram_interleaved",
        "fused": list(fused or []),
    }


def _profile_fused(decode_status="traced"):
    tops = [
        _op("cheap", 3.0),  # sets the self-calibrated dispatch floor; see the fold-gate fixture
        _op("A", 50.0, fused=["UnaryWithParam(GELU)"]),
        _op("B", 48.0),
        _op("C", 46.0),
    ]
    return {
        "device_ms": sum(t["device_ms"] for t in tops),
        "decode_status": decode_status,
        "buckets": [{"id": "matmul", "top_ops": tops}],
    }


def _view(prof):
    """The dict termination_check assembles, built here the one way it is built there."""
    return {**prof, "open_ops": R.residual_report(prof, HW).get("open_ops") or []}


def test_a_fused_op_under_trace_is_asked_about(monkeypatch):
    m = _mcp()
    _wins_by_flag(m, monkeypatch)
    b = m._split_gate(_view(_profile_fused()), [])
    assert b and b["next_rung"] == "structural-split"
    assert "split" in b["reason"].lower()


def test_an_untraced_run_is_left_alone(monkeypatch):
    """The premise is trace. Without it the fusion is still buying the dispatch it was chosen for,
    and asking to split it is asking for a measured regression -- the mistake the decode gate's
    comment records, where an encoder-only model was ordered to add a KV-cache."""
    m = _mcp()
    _wins_by_flag(m, monkeypatch)
    for status in ("off", "repeat_prefill", ""):
        assert m._split_gate(_view(_profile_fused(status)), []) is None, status


def test_an_op_that_fused_nothing_is_not_a_candidate(monkeypatch):
    m = _mcp()
    _wins_by_flag(m, monkeypatch)
    prof = _profile_fused()
    for t in prof["buckets"][0]["top_ops"]:
        t["fused"] = []
    assert m._split_gate(_view(prof), []) is None


def test_a_dispatch_bound_op_is_not_asked_to_add_a_dispatch(monkeypatch):
    """Splitting it adds a launch to an op whose floor is already the launch cost. Known without
    measuring, so it is not worth a round."""
    m = _mcp()
    _wins_by_flag(m, monkeypatch)
    view = _view(_profile_fused())
    for o in view["open_ops"]:
        o["bound_by"] = "dispatch"
    assert m._split_gate(view, []) is None


def test_it_clears_on_a_measured_result_including_a_negative_one(monkeypatch):
    """`none: <evidence>` is an outcome this gate invites by name, so it has to be able to run to
    the cap -- an unsplittable op must not re-emit forever."""
    m = _mcp()
    _wins_by_flag(m, monkeypatch)
    view = _view(_profile_fused())
    assert m._split_gate(view, [{"kernel_kind": "split", "beat_baseline": True}]) is None
    tried = [{"kernel_kind": "split", "beat_baseline": False}] * m._gate_cap("PERF_MCP_MAX_SPLIT_ATTEMPTS")
    assert m._split_gate(view, tried) is None


# ------------------------------------------------- 11: the kernel nobody re-checks


def _kernel(version, kind="tt-lang", win=True):
    return {
        "op_signature": "MatmulDeviceOperation A",
        "kernel_kind": kind,
        "beat_baseline": win,
        "ttnn_version": version,
    }


def test_a_kernel_graded_by_another_ttnn_is_re_opened(monkeypatch):
    m = _mcp()
    _wins_by_flag(m, monkeypatch)
    monkeypatch.setattr(m, "_ttnn_version", lambda: "0.66.0")
    b = m._stock_gate(_view(_profile_fused()), [_kernel("0.65.1")])
    assert b and b["next_rung"] == "structural-stock"
    assert "0.65.1" in b["reason"] and "0.66.0" in b["reason"]


def test_the_same_ttnn_asks_nothing(monkeypatch):
    """Within one run the comparison has already happened: the baseline a kernel must beat IS the
    stock op. Firing here would order a re-measurement of something just measured."""
    m = _mcp()
    _wins_by_flag(m, monkeypatch)
    monkeypatch.setattr(m, "_ttnn_version", lambda: "0.65.1")
    assert m._stock_gate(_view(_profile_fused()), [_kernel("0.65.1")]) is None


def test_an_unstamped_attempt_reads_as_unknown_not_as_stale(monkeypatch):
    """Every attempt recorded before the stamp existed has no version. Treating that as a mismatch
    would open every historical kernel at once on the first run after this shipped."""
    m = _mcp()
    _wins_by_flag(m, monkeypatch)
    monkeypatch.setattr(m, "_ttnn_version", lambda: "0.66.0")
    assert m._stock_gate(_view(_profile_fused()), [_kernel(None)]) is None
    assert m._stock_gate(_view(_profile_fused()), [_kernel("")]) is None


def test_a_row_older_than_the_stamp_gets_one_so_the_next_upgrade_can_find_it(monkeypatch, tmp_path):
    """Otherwise the gate only ever applies to kernels written from now on -- and the ones most
    likely to have gone stale are the oldest. Voxtral's banked ArgMax is exactly such a row."""
    m = _mcp()
    live = tmp_path / "log.json"
    live.write_text(json.dumps([{"op_signature": "X", "kernel_kind": "tt-lang", "beat_baseline": True}]))
    monkeypatch.setattr(m, "_KERNEL_LOG_PATH", live)
    monkeypatch.setattr(m, "_ttnn_version", lambda: "0.66.0")
    m._TTNN_BACKFILLED = False
    m._backfill_ttnn_version()
    assert json.loads(live.read_text())[0]["ttnn_version"] == "0.66.0"


def test_the_backfill_writes_nothing_when_there_is_nothing_to_fill(monkeypatch, tmp_path):
    """The common case is every row already stamped; it must not rewrite the log on every round."""
    m = _mcp()
    live = tmp_path / "log.json"
    live.write_text(json.dumps([{"op_signature": "X", "ttnn_version": "0.66.0"}]))
    before = live.stat().st_mtime_ns
    monkeypatch.setattr(m, "_KERNEL_LOG_PATH", live)
    monkeypatch.setattr(m, "_ttnn_version", lambda: "0.66.0")
    m._TTNN_BACKFILLED = False
    m._backfill_ttnn_version()
    assert live.stat().st_mtime_ns == before


def test_the_backfill_does_not_move_archived_rows_into_the_live_log(monkeypatch, tmp_path):
    """_load_attempts merges the archive with the live rows. Saving that union back would land
    archived rows in the live log, where the resume filter rewrites against this run's baseline."""
    m = _mcp()
    live, cum = tmp_path / "log.json", tmp_path / "log.json.cumulative"
    live.write_text(json.dumps([{"op_signature": "LIVE"}]))
    cum.write_text(json.dumps([{"op_signature": "ARCHIVED"}]))
    monkeypatch.setattr(m, "_KERNEL_LOG_PATH", live)
    monkeypatch.setattr(m, "_ttnn_version", lambda: "0.66.0")
    m._TTNN_BACKFILLED = False
    m._backfill_ttnn_version()
    assert [r["op_signature"] for r in json.loads(live.read_text())] == ["LIVE"]
    assert [r["op_signature"] for r in json.loads(cum.read_text())] == ["ARCHIVED"]
    assert json.loads(cum.read_text())[0]["ttnn_version"] == "0.66.0"


def test_a_process_that_cannot_name_its_ttnn_asks_nothing(monkeypatch):
    m = _mcp()
    _wins_by_flag(m, monkeypatch)
    monkeypatch.setattr(m, "_ttnn_version", lambda: "")
    assert m._stock_gate(_view(_profile_fused()), [_kernel("0.65.1")]) is None


def test_only_a_kernel_that_actually_won_holds_a_place_worth_re_checking(monkeypatch):
    """A losing attempt left no kernel in the model, so there is nothing stale to replace."""
    m = _mcp()
    _wins_by_flag(m, monkeypatch)
    monkeypatch.setattr(m, "_ttnn_version", lambda: "0.66.0")
    assert m._stock_gate(_view(_profile_fused()), [_kernel("0.65.1", win=False)]) is None
    # ...and a knob turned under an older ttnn is not a hand-written interior either.
    assert m._stock_gate(_view(_profile_fused()), [_kernel("0.65.1", kind="grid")]) is None


def test_it_clears_on_a_measured_comparison(monkeypatch):
    m = _mcp()
    _wins_by_flag(m, monkeypatch)
    monkeypatch.setattr(m, "_ttnn_version", lambda: "0.66.0")
    view, stale = _view(_profile_fused()), _kernel("0.65.1")
    assert m._stock_gate(view, [stale, {"kernel_kind": "stock", "beat_baseline": True}]) is None
    tried = [{"kernel_kind": "stock", "beat_baseline": False}] * m._gate_cap("PERF_MCP_MAX_STOCK_ATTEMPTS")
    assert m._stock_gate(view, [stale] + tried) is None


def test_the_version_is_stamped_where_the_gate_reads_it():
    """The gate compares a field on an attempt; something has to write it, and there is one writer.

    The random-weights caveat and the loader verdict were both written into fields nothing read;
    this is the same failure inverted -- a reader with no writer -- and it fails silently as "no
    kernel is ever stale".
    """
    assert '"ttnn_version": _ttnn_version()' in _SRC


def test_the_version_helper_does_not_re_answer_a_solved_question():
    """agent/ttlang.py already reports the running ttnn, and ensure_ttl depends on that answer."""
    assert "from agent.ttlang import ttnn_version" in _SRC
    assert _SRC.count("getattr(ttnn,") == 0


# ------------------------------------------------- both: reachable and clearable


def test_both_gates_are_wired_into_the_stop_gate():
    """...and handed the merged view, not the bare profile -- the exact bug the fold, order and
    conv gates shipped with."""
    for gate in ("_split_gate", "_stock_gate"):
        assert "%s(_gate_prof, attempts)" % gate in _SRC, gate
        assert "%s(prof, attempts)" % gate not in _SRC, gate


def test_the_report_names_these_levers_instead_of_filing_them_under_other():
    """The summary keeps its own set of structural kinds, and a kind missing from it does not fail
    -- it renders in the anonymous `other` column, which is the confusion that column was split out
    to end. So the omission looks like a vague report rather than a wrong one, and the four gate
    kinds that already fell in that hole are why the set is now read from the lever table."""
    spec = importlib.util.spec_from_file_location("_sum_revisit", _PA / "cc_optimize" / "summary.py")
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    m = _mcp()
    for kinds, _cap_env in (m._SPLIT_LEVER, m._STOCK_LEVER):
        for k in kinds:
            assert S._level_of(k) == "structural", k


def test_every_kind_these_gates_name_is_one_the_recorder_accepts(monkeypatch):
    """A gate that fires and cannot be cleared re-emits the same target forever. Both levers go in
    the one table, so this holds by construction -- asserted anyway, because the table is what a
    future gate will be tempted to bypass."""
    m = _mcp()
    for lever in (m._SPLIT_LEVER, m._STOCK_LEVER):
        kinds, cap_env = lever
        assert lever in m._GATE_LEVERS
        for k in kinds:
            # _KNOB_KINDS is local to record_kernel_attempt and is built as {...} | _GATE_KINDS, so
            # membership here IS membership there. That union is pinned separately, in the sibling
            # test for the three gates this pair follows.
            assert k in m._GATE_KINDS, k  # recordable without a custom-kernel marker
            assert m._rung_allowance("op", k, [])[1] >= m._gate_cap(cap_env), k
