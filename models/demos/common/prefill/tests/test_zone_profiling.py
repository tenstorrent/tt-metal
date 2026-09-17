# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free tests for the shared zone profiler (models/demos/common/prefill/profiling).

Everything here runs on host. The pieces pinned are the ones that would otherwise produce a *wrong
report* rather than an error: zone gating and the signpost wire format, attribution of ops to zones,
truncation / profiler-overflow detection, leaf detection, and the per-layer aggregation that the
compute/communication/memory split and the full-model projection are built on. The per-model contracts
(layer tags, comm/memory keys, env-var names) are tested next to each model.

    pytest models/demos/common/prefill/tests/test_zone_profiling.py
"""

import csv
import json

import pytest

import ttnn
from models.demos.common.prefill.profiling import parse_zone_perf as P
from models.demos.common.prefill.profiling import visualize_zones as V
from models.demos.common.prefill.profiling.spec import LayerClass, ZoneSpec
from models.demos.common.prefill.profiling.zones import (
    COARSE,
    FINE,
    MEDIUM,
    SIGNPOST_FORMAT,
    ZoneProfiler,
    parse_level,
    signpost,
)

SPEC = ZoneSpec(
    model_name="Test",
    signpost_prefix="T_ZONE",
    env_prefix="T_PROFILE",
    host_zone_scope="test_model",
    layer_classes=(LayerClass("a", "A layer", 3), LayerClass("b", "B layer", 5)),
    comm_keys=("allgather", "dispatch"),
    mem_keys=("kv_write",),
)
ROOT = SPEC.root_zone
MS = 1_000_000  # ns


def sig(name):
    return {"OP CODE": name, "OP TYPE": "signpost"}


def start(name):
    return sig(f"{SPEC.zone_start} {name}")


def end(name):
    return sig(f"{SPEC.zone_end} {name}")


def op(ns, dev=0, code="Matmul", fw=None, op_type=P.DEVICE_OP_TYPE, **extra):
    row = {"OP CODE": code, "OP TYPE": op_type, "DEVICE ID": dev, P.DURATION_COL: ns}
    if fw is not None:
        row[P.FW_DURATION_COL] = fw
    row.update(extra)
    return row


def layer(idx, cls, body):
    return [start(f"layer{idx:02d}_{cls}"), *body, end(f"layer{idx:02d}_{cls}")]


def chunk(*layers):
    rows = [start(ROOT)]
    for lyr in layers:
        rows += lyr
    rows.append(end(ROOT))
    return rows


def feed(rows, **flags):
    acc = P.ZoneAccumulator(SPEC)
    for k, v in flags.items():
        setattr(acc, k, v)
    for r in rows:
        acc.feed(r, {})
    return acc


def aggregate(rows, **flags):
    acc = feed(rows, **flags)
    return acc, P.aggregate_by_class(acc, P.summarize(acc))


@pytest.fixture
def messages(monkeypatch):
    """Record what the zone helper hands to ttnn.tracy_message / start_tracy_zone / stop_tracy_zone."""
    log = []
    monkeypatch.setattr(ttnn, "tracy_message", lambda msg, color=0: log.append(("msg", msg)))
    monkeypatch.setattr(ttnn, "start_tracy_zone", lambda scope, name, line, color=0: log.append(("zone+", scope, name)))
    monkeypatch.setattr(ttnn, "stop_tracy_zone", lambda name="", color=0: log.append(("zone-", name)))
    return log


class TestSpec:
    def test_derived_names(self):
        assert SPEC.zone_start == "T_ZONE_START"
        assert SPEC.zone_end == "T_ZONE_END"
        assert (SPEC.zones_env, SPEC.level_env, SPEC.host_zones_env) == (
            "T_PROFILE_ZONES",
            "T_PROFILE_LEVEL",
            "T_PROFILE_HOST_ZONES",
        )
        assert SPEC.class_keys == ("a", "b")
        assert SPEC.full_model_layers == 8

    def test_categorization(self):
        assert SPEC.cat("attn/kv_write") == "memory"
        assert SPEC.cat("attn/kv_write/k") == "memory"
        assert SPEC.cat("mlp/dispatch") == "comm"
        assert SPEC.cat("attn/allgather") == "comm"
        assert SPEC.cat("mlp/dispatch/(self)") == "compute"  # a self bucket is glue, not the collective
        assert SPEC.cat("attn/sdpa") == "compute"

    def test_duplicate_class_keys_are_rejected(self, expect_error):
        with expect_error(AssertionError, "duplicate layer class keys"):
            ZoneSpec("X", "X_ZONE", "X_PROFILE", "x", (LayerClass("a", "A", 1), LayerClass("a", "A2", 1)), (), ())


class TestZoneGating:
    def test_disabled_is_a_shared_no_op(self, messages):
        prof = ZoneProfiler(SPEC, enabled=False, level=FINE)
        cm = prof.zone("attn", COARSE)
        assert cm is prof.zone("mlp", FINE), "the disabled path must hand out one stateless singleton"
        with cm:
            pass
        assert messages == []

    def test_disabled_path_never_touches_ttnn(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("tracy_message called with zones disabled")

        monkeypatch.setattr(ttnn, "tracy_message", boom)
        with ZoneProfiler(SPEC, enabled=False).zone("attn"):
            pass

    def test_zone_above_level_is_suppressed(self, messages):
        prof = ZoneProfiler(SPEC, enabled=True, level=MEDIUM)
        with prof.zone("input_norm", FINE):
            pass
        assert messages == []

    def test_enabled_zone_emits_paired_signposts_and_host_zone(self, messages):
        prof = ZoneProfiler(SPEC, enabled=True, level=MEDIUM, host_zones=True)
        with prof.zone("attn", COARSE):
            assert messages == [("msg", "`TT_SIGNPOST: T_ZONE_START attn`"), ("zone+", "test_model", "attn")]
        assert messages[2:] == [("zone-", "attn"), ("msg", "`TT_SIGNPOST: T_ZONE_END attn`")]

    def test_host_zones_can_be_turned_off(self, messages):
        with ZoneProfiler(SPEC, enabled=True, host_zones=False).zone("attn"):
            pass
        assert [m[0] for m in messages] == ["msg", "msg"]

    def test_nested_zones_close_in_order(self, messages):
        prof = ZoneProfiler(SPEC, enabled=True, host_zones=False)
        with prof.zone("attn"):
            with prof.zone("qkv_proj"):
                pass
        assert [m[1].split()[-2:] for m in messages] == [
            ["T_ZONE_START", "attn`"],
            ["T_ZONE_START", "qkv_proj`"],
            ["T_ZONE_END", "qkv_proj`"],
            ["T_ZONE_END", "attn`"],
        ]

    def test_end_marker_survives_an_exception_in_the_body(self, messages, expect_error):
        prof = ZoneProfiler(SPEC, enabled=True, host_zones=False)
        with expect_error(RuntimeError, "op failed"):
            with prof.zone("attn"):
                raise RuntimeError("op failed")
        assert messages[-1] == ("msg", "`TT_SIGNPOST: T_ZONE_END attn`")

    def test_configured_from_the_spec_env_vars(self, monkeypatch):
        monkeypatch.setenv(SPEC.zones_env, "1")
        monkeypatch.setenv(SPEC.level_env, "3")
        monkeypatch.setenv(SPEC.host_zones_env, "0")
        prof = ZoneProfiler(SPEC)
        assert (prof.enabled, prof.level, prof.host_zones) == (True, 3, False)
        monkeypatch.delenv(SPEC.zones_env)
        assert ZoneProfiler(SPEC).enabled is False

    def test_bad_level_falls_back_instead_of_breaking_imports(self, monkeypatch):
        assert parse_level("fine", var="X") == MEDIUM
        assert parse_level("", var="X") == MEDIUM
        assert parse_level(None, var="X") == MEDIUM
        assert parse_level("3", var="X") == FINE
        monkeypatch.setenv(SPEC.level_env, "coarse")
        assert ZoneProfiler(SPEC).level == MEDIUM


class TestSignpostRoundTrip:
    """The signpost string is a contract with tools/tracy/process_ops_logs.py, which turns the message
    into the CSV's OP CODE — and with the parser, which reads that OP CODE back."""

    @staticmethod
    def _op_code_from_message(msg):
        # tools/tracy/process_ops_logs.py: the backticks are the message CSV's quotechar; the row's
        # OP CODE is `data.split(": ")[-1].split("\n")[0]`.
        assert msg.startswith("`") and msg.endswith("`"), "backticks are the message CSV quotechar"
        data = msg.strip("`")
        assert "TT_SIGNPOST" in data, "TT_SIGNPOST is what marks the row as a signpost"
        return data.split(": ")[-1].split("\n")[0]

    def test_same_bytes_as_tracy_signpost(self, messages):
        tracy = pytest.importorskip("tracy")
        header = f"{SPEC.zone_start} attn"
        signpost(header)
        tracy.signpost(header)
        assert messages[0] == messages[1], "the zone helper must emit exactly what tracy.signpost emits"

    def test_message_parses_back_into_the_zone(self, messages):
        prof = ZoneProfiler(SPEC, enabled=True, host_zones=False)
        with prof.zone("attn"):
            pass
        rows = [sig(self._op_code_from_message(m[1])) for m in messages]
        acc = feed([start(ROOT), *rows, end(ROOT)])
        assert acc.root_closed and acc.unmatched_ends == 0
        acc = feed([start(ROOT), rows[0], op(MS)])
        assert acc.path == f"{ROOT}/attn", "the START row must open exactly the zone that was entered"

    def test_format_is_the_documented_one(self):
        assert SIGNPOST_FORMAT.format(header="H") == "`TT_SIGNPOST: H`"


class TestAccumulator:
    """Attribution: ops belong to the innermost open zone and every enclosing one, and only the
    profiled chunk is reported."""

    def test_ops_outside_the_root_zone_are_ignored(self):
        # Warmup and cache-prefix ops share the CSV with the profiled chunk; they must not be counted.
        acc = feed([op(MS), start("layer00_a"), op(MS), end("layer00_a")])
        assert acc.rows_in_root == 0 and not acc.stats

    def test_op_is_charged_to_every_enclosing_zone(self):
        acc = feed(chunk(layer(3, "b", [start("attn"), op(2 * MS), end("attn")])))
        for path in (ROOT, f"{ROOT}/layer03_b", f"{ROOT}/layer03_b/attn"):
            assert acc.stats[(path, 0)]["ns"] == 2 * MS, path
        assert acc.rows_in_root == 1 and acc.devices == [0]

    def test_complete_capture_has_no_warnings(self):
        acc = feed(chunk(layer(0, "a", [op(MS)])))
        assert acc.root_opened and acc.root_closed and acc.warnings() == []

    def test_missing_root_zone_is_reported(self):
        acc = feed([op(MS)])
        assert not acc.root_opened
        assert any(f"no `{ROOT}` zone" in w for w in acc.warnings())

    def test_truncated_capture_is_reported(self):
        # The normal tail-truncation case: the CSV ends after a START, no END ever comes.
        acc = feed([start(ROOT), start("layer00_a"), start("attn"), op(MS)])
        assert acc.root_opened and not acc.root_closed and acc.stack == [ROOT, "layer00_a", "attn"]
        assert any("TRUNCATED" in w and "attn" in w for w in acc.warnings())

    def test_unmatched_end_marker_is_counted_not_fatal(self):
        acc = feed([end("never_opened")])
        assert acc.unmatched_ends == 1
        assert any("unmatched" in w for w in acc.warnings())

    def test_dropped_start_unwinds_to_the_matching_frame(self):
        acc = feed([start(ROOT), start("layer00_a"), start("attn"), end("layer00_a"), op(MS), end(ROOT)])
        assert acc.unmatched_ends == 1 and acc.root_closed
        assert acc.stats[(ROOT, 0)]["ns"] == MS and (f"{ROOT}/layer00_a", 0) not in acc.stats

    def test_device_op_without_duration_is_counted_as_profiler_overflow(self):
        acc = feed(chunk(layer(0, "a", [op(None), op(MS)])))
        assert acc.ops_no_device_data == 1 and acc.rows_in_root == 1
        assert any(P.DURATION_COL in w and "overflow" in w for w in acc.warnings())

    def test_non_device_ops_are_flagged_as_host_work_not_overflow(self):
        acc = feed(chunk(layer(0, "a", [op(None, code="Fallback", op_type="python_fallback")])))
        assert acc.host_ops["a:(layer total)"]["Fallback [python_fallback]"]["count"] == 1
        assert acc.ops_no_device_data == 0

    def test_buffer_transfers_are_recorded_per_zone(self):
        col = "HWCommandQueue_write_buffer_TT_HOST_FUNC [ns]"
        acc = feed(chunk(layer(0, "a", [start("attn"), op(MS, **{col: 5000.0}), end("attn")])))
        assert acc.movement["a:attn"][P.HOST_MOVEMENT_COLS[col]] == 5000.0

    def test_root_kernel_and_firmware_sums_per_device(self):
        acc = feed(chunk(layer(0, "a", [op(MS, dev=0, fw=2 * MS), op(3 * MS, dev=1, fw=4 * MS)])), fw_col_present=True)
        assert dict(acc.root_kernel_ns) == {0: MS, 1: 3 * MS}
        assert dict(acc.root_fw_ns) == {0: 2 * MS, 1: 4 * MS}


class TestLayerTagParsing:
    def test_any_class_suffix_is_accepted(self):
        assert P.layer_class(f"{ROOT}/layer00_a/attn/qkv_proj") == ("a", "attn/qkv_proj", 0)
        assert P.layer_class(f"{ROOT}/layer12_sparse/mlp") == ("sparse", "mlp", 12)
        assert P.layer_class(f"{ROOT}/layer07_b") == ("b", "", 7)

    def test_non_layer_zones(self):
        assert P.layer_class(ROOT) is None
        assert P.layer_class(f"{ROOT}/embedding") is None
        assert P.relative_path(f"{ROOT}/embedding") == f"{ROOT}/embedding"

    def test_relative_path_collapses_layer_index(self):
        assert P.relative_path(f"{ROOT}/layer04_a/attn/sdpa") == "a:attn/sdpa"
        assert P.relative_path(f"{ROOT}/layer05_b") == f"b:{P.LAYER_TOTAL}"


class TestAggregation:
    """aggregate_by_class feeds the compute/comm/memory split and the projection."""

    def test_each_layer_is_read_on_one_device(self):
        # dev 0: attn 5 + mlp 1 = 6 ms; dev 1: attn 1 + mlp 6 = 7 ms. Summing per-zone maxima would give
        # attn 5 + mlp 6 = 11 ms, more than any chip actually spent; the layer must be read on dev 1.
        _, by = aggregate(
            chunk(
                layer(
                    0,
                    "a",
                    [
                        start("attn"),
                        op(5 * MS, dev=0),
                        op(1 * MS, dev=1),
                        end("attn"),
                        start("mlp"),
                        op(1 * MS, dev=0),
                        op(6 * MS, dev=1),
                        end("mlp"),
                    ],
                )
            )
        )
        a = by["a"]
        assert a[P.LAYER_TOTAL]["ms_per_layer"] == 7.0
        assert (a["attn"]["ms_per_layer"], a["mlp"]["ms_per_layer"]) == (1.0, 6.0)
        assert a["attn"]["ms_per_layer"] + a["mlp"]["ms_per_layer"] == a[P.LAYER_TOTAL]["ms_per_layer"]

    def test_parent_exclusive_ops_land_in_a_self_bucket(self):
        # Norm and residual run directly in the layer zone (their FINE zones suppressed); one glue op
        # runs directly in attn next to its captured child. None of that time may vanish.
        _, by = aggregate(
            chunk(
                layer(
                    0,
                    "a",
                    [
                        op(1 * MS, code="Norm"),
                        start("attn"),
                        start("qkv_proj"),
                        op(2 * MS),
                        end("qkv_proj"),
                        op(MS // 2, code="Glue"),
                        end("attn"),
                        op(1 * MS, code="Residual"),
                    ],
                )
            )
        )
        a = by["a"]
        assert a[P.LAYER_TOTAL]["ms_per_layer"] == 4.5
        assert a["attn"]["ms_per_layer"] == 2.5  # inclusive
        assert a[f"attn/{P.SELF}"]["ms_per_layer"] == 0.5 and a[f"attn/{P.SELF}"]["ops_per_layer"] == 1
        assert a[P.SELF]["ms_per_layer"] == 2.0 and a[P.SELF]["ops_per_layer"] == 2
        leaves = set(a) - V.parent_rels(a)
        assert leaves == {"attn/qkv_proj", f"attn/{P.SELF}", P.SELF}
        assert sum(a[k]["ms_per_layer"] for k in leaves) == a[P.LAYER_TOTAL]["ms_per_layer"]

    def test_no_self_bucket_when_children_cover_the_parent(self):
        _, by = aggregate(chunk(layer(0, "a", [start("attn"), op(2 * MS), end("attn")])))
        assert set(by["a"]) == {P.LAYER_TOTAL, "attn"}

    def test_averages_over_the_sampled_layers_of_a_class(self):
        _, by = aggregate(
            chunk(
                layer(0, "a", [start("attn"), op(2 * MS), end("attn")]),
                layer(1, "b", [start("attn"), op(9 * MS), end("attn")]),
                layer(2, "a", [start("attn"), op(4 * MS), end("attn")]),
            )
        )
        assert by["a"]["attn"] == pytest.approx(
            {**by["a"]["attn"], "ms_total": 6.0, "ms_per_layer": 3.0, "layers": 2, "ops_per_layer": 1}
        )
        assert by["b"]["attn"]["ms_per_layer"] == 9.0 and by["b"]["attn"]["layers"] == 1

    def test_layer_without_a_total_is_skipped(self):
        # A layer whose own START was lost has children but no layer total to anchor a device on.
        acc = feed([start(ROOT), start("attn"), op(MS), end("attn"), end("layer00_a"), end(ROOT)])
        assert P.aggregate_by_class(acc, P.summarize(acc)) == {}

    def test_summary_keeps_the_per_zone_spread(self):
        acc = feed(chunk(layer(0, "a", [start("attn"), op(1 * MS, dev=0), op(3 * MS, dev=1), end("attn")])))
        s = P.summarize(acc)[f"{ROOT}/layer00_a/attn"]
        assert (s["ms_max"], s["ms_min"], s["skew_ms"], s["worst_device"], s["num_devices"]) == (3.0, 1.0, 2.0, 1, 2)


class TestLeafDetection:
    """A zone is a parent only when a descendant is present in THIS capture, so the exclusion set must
    come from the capture, never from a static list — a static list silently drops leaf time."""

    LEVEL2 = {P.LAYER_TOTAL, "attn", "mlp", "attn/ring_joint_sdpa", "attn/kv_write", "mlp/experts_mm", "mlp/dispatch"}
    LEVEL3 = LEVEL2 | {"attn/kv_write/k", "attn/kv_write/v", "mlp/routing_setup"}

    def test_zone_is_a_leaf_when_its_children_are_suppressed(self):
        assert "attn/kv_write" not in V.parent_rels(self.LEVEL2)

    def test_zone_is_a_parent_when_its_children_are_captured(self):
        assert "attn/kv_write" in V.parent_rels(self.LEVEL3)

    def test_real_parents_are_always_excluded(self):
        for level in (self.LEVEL2, self.LEVEL3):
            assert {"attn", "mlp", P.LAYER_TOTAL} <= V.parent_rels(level)

    def test_leaves_and_parents_partition_the_zones(self):
        for level in (self.LEVEL2, self.LEVEL3):
            parents = V.parent_rels(level)
            leaves = level - parents
            assert parents | leaves == level and not (parents & leaves)


class TestAccounting:
    def _acc(self, fw):
        return feed(
            chunk(
                layer(0, "a", [op(2 * MS, dev=0, fw=3 * MS if fw else None), op(MS, dev=1, fw=MS if fw else None)]),
                layer(1, "b", [op(1 * MS, dev=0, fw=int(1.5 * MS) if fw else None)]),
            ),
            fw_col_present=fw,
        )

    def test_projection_scales_each_class_by_its_full_model_count(self):
        acc = self._acc(fw=True)
        acct = V.accounting(acc, P.aggregate_by_class(acc, P.summarize(acc)), SPEC)
        assert acct["device"] == 0 and acct["num_devices"] == 2 and acct["fw_measured"]
        assert (acct["kernel_ms"], acct["busy_ms"], acct["fw_multiplier"]) == (3.0, 4.5, 1.5)
        assert acct["proj_kernel_ms"] == 3 * 2.0 + 5 * 1.0  # 3 'a' layers x 2 ms + 5 'b' layers x 1 ms
        assert acct["proj_busy_ms"] == pytest.approx(11.0 * 1.5)
        assert acct["proj_layers"] == 8 and acct["proj_breakdown"] == "3 a + 5 b"
        assert acct["sampled"] == {"a": 1, "b": 1}

    def test_without_firmware_column_busy_equals_kernel(self):
        acc = self._acc(fw=False)
        acct = V.accounting(acc, P.aggregate_by_class(acc, P.summarize(acc)), SPEC)
        assert not acct["fw_measured"] and acct["busy_ms"] == acct["kernel_ms"] and acct["fw_multiplier"] == 1.0

    def test_no_root_gives_no_accounting(self):
        assert V.accounting(feed([op(MS)]), {}, SPEC) is None


def _write_csv(path, rows):
    cols = list(dict.fromkeys([*P.BASE_COLS, *(c for r in rows for c in r)]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({**{"CORE COUNT": 1}, **r})


class TestCli:
    ROWS = chunk(
        layer(0, "a", [op(2 * MS, fw=3 * MS), start("attn"), op(MS, fw=MS), end("attn")]),
        layer(1, "b", [start("mlp"), start("dispatch"), op(4 * MS, fw=5 * MS), end("dispatch"), end("mlp")]),
    )

    def test_parse_cli_reports_and_writes_json(self, tmp_path):
        path, out = tmp_path / "ops.csv", tmp_path / "zones.json"
        _write_csv(path, self.ROWS)
        assert P.main(SPEC, [str(path), "--json", str(out), "--top", "2"]) == 0
        data = json.loads(out.read_text())
        assert data["warnings"] == []
        assert data["by_class"]["b"]["mlp/dispatch"]["ms_per_layer"] == 4.0
        assert data["by_class"]["a"][P.SELF]["ms_per_layer"] == 2.0

    def test_parse_csv_reads_the_file_once_and_flags_columns(self, tmp_path):
        path = tmp_path / "ops.csv"
        _write_csv(path, self.ROWS)
        acc, nrows = P.parse_csv(path, SPEC, collect_timeline=True)
        assert nrows == len(self.ROWS) and acc.fw_col_present and not acc.movement_cols_present
        assert len(acc.timeline) == 3

    def test_parse_cli_fails_on_a_csv_without_a_root_zone(self, tmp_path):
        path = tmp_path / "ops.csv"
        _write_csv(path, [op(MS)])
        assert P.main(SPEC, [str(path)]) == 1

    def test_parse_cli_rejects_a_csv_missing_base_columns(self, tmp_path, expect_error):
        path = tmp_path / "ops.csv"
        path.write_text("OP CODE,OP TYPE\nMatmul,tt_dnn_device\n")
        with expect_error(SystemExit, "missing expected column"):
            P.main(SPEC, [str(path)])

    def test_visualize_cli_renders_text_and_html(self, tmp_path, capsys):
        path, html = tmp_path / "ops.csv", tmp_path / "report.html"
        _write_csv(path, self.ROWS)
        assert V.main(SPEC, [str(path), "-o", str(html)]) == 0
        text = capsys.readouterr().out
        assert "A LAYER" in text and "B LAYER" in text and "PROJECTION to 8 layers" in text
        page = html.read_text()
        assert "Test prefill zone profile" in page and '"numDevices":1' in page and '"warnings":[]' in page

    def test_html_escapes_strings_that_came_from_the_command_line(self, tmp_path):
        # The CSV's basename lands in the page's <script> payload; a hostile name must not close the block.
        path = tmp_path / "evil<script>alert(1)<x&y>.csv"
        _write_csv(path, self.ROWS)
        acc, summary, byclass = V.collect(path, SPEC)
        page = V.build_html(byclass, acc, V.accounting(acc, byclass, SPEC), path, SPEC)
        assert "<script>alert" not in page
        assert "evil\\u003cscript\\u003ealert(1)\\u003cx\\u0026y\\u003e.csv" in page

    def test_visualize_cli_carries_truncation_into_the_report(self, tmp_path, capsys):
        path, html = tmp_path / "ops.csv", tmp_path / "report.html"
        _write_csv(path, [start(ROOT), start("layer00_a"), op(MS)])  # cut off mid-layer
        assert V.main(SPEC, [str(path), "-o", str(html)]) == 0
        assert "TRUNCATED" in capsys.readouterr().out and "TRUNCATED" in html.read_text()

    def test_visualize_cli_exits_without_a_root_zone(self, tmp_path, expect_error):
        path = tmp_path / "ops.csv"
        _write_csv(path, [op(MS)])
        with expect_error(SystemExit, f"no `{ROOT}` zone"):
            V.main(SPEC, [str(path), "-o", str(tmp_path / "r.html")])
