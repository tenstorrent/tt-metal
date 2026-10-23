# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The regression harness without a device: the rules, the verdicts (todo and info softening, per-key maps, the
idle condition), the metric definitions on constructed artifacts, seeding from agreeing and disagreeing runs, the
runner on a command job and on a server job (a fake launcher with a fake HTTP server), and the committed pins and
baselines."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from models.demos.blackhole.qwen38_flash_next.tools.ci import q38_ci as ci

MODEL_DIR = Path(ci.MODEL_DIR)
DEFAULTS = {"default_tolerance": 0.15}


def verdict(rule, observed, expected, *, gated=True, status="active", **fields):
    return ci.compare(
        "J/cfg/m", {"rule": rule, "status": status, **fields}, observed, expected, gated=gated, defaults=DEFAULTS
    )


# -- rules -------------------------------------------------------------------------------------------------------------


def test_band_is_two_sided_and_names_the_stale_side():
    assert verdict("band", 51.0, 50.0, tolerance=0.02)["status"] == "pass"
    slow = verdict("band", 52.0, 50.0, tolerance=0.02, better="lower")
    fast = verdict("band", 48.0, 50.0, tolerance=0.02, better="lower")
    assert slow["status"] == "fail" and slow["note"].startswith("regression")
    assert fast["status"] == "fail" and fast["note"].startswith("stale target")
    assert verdict("band", 21.0, 19.5, tolerance=0.05, better="higher")["note"].startswith("stale target")
    assert verdict("band", 60.0, 50.0)["tolerance"] == 0.15  # the default band


def test_floor_ceiling_exact_at_most_and_not_earlier():
    assert verdict("floor", 0.985, 0.99, slack=0.01)["status"] == "pass"
    assert verdict("floor", 0.979, 0.99, slack=0.01)["status"] == "fail"
    assert verdict("floor", 470_000_000, 478_202_944, slack_relative=0.01)["status"] == "fail"
    assert verdict("floor", 474_000_000, 478_202_944, slack_relative=0.01)["status"] == "pass"
    assert verdict("ceiling", 0.108, 0.10, tolerance=0.10)["status"] == "pass"
    assert verdict("ceiling", 0.108, 0.10, tolerance=0.10, warn_tolerance=0.05)["status"] == "warn"
    assert verdict("ceiling", 0.111, 0.10, tolerance=0.10)["status"] == "fail"
    assert verdict("exact", 96, 96)["status"] == "pass" and verdict("exact", 95, 96)["status"] == "fail"
    assert verdict("at_most", 3, 3)["status"] == "pass" and verdict("at_most", 4, 3)["status"] == "fail"
    # null is the largest index: no divergence is never earlier; a divergence against a null baseline is
    assert verdict("not_earlier", None, 8)["status"] == "pass"
    assert verdict("not_earlier", 9, 8)["status"] == "pass"
    assert verdict("not_earlier", 7, 8)["status"] == "fail"
    assert verdict("not_earlier", 90, None)["status"] == "fail"
    assert verdict("not_earlier", None, None)["status"] == "pass"


def test_flips_count_pass_to_fail_per_group_and_respect_the_item_list():
    expected = {"gsm8k/a": True, "gsm8k/b": True, "gsm8k/c": False, "he/x": True, "he/y": True}
    observed = {"gsm8k/a": False, "gsm8k/b": False, "gsm8k/c": True, "he/x": True, "he/y": False}
    result = verdict("flips", observed, expected, max_flips=2)
    assert result["status"] == "pass"
    assert result["details"]["groups"]["gsm8k"] == {
        "items": 3,
        "flips": 2,
        "changed": 3,
        "flipped": ["gsm8k/a", "gsm8k/b"],
    }
    assert verdict("flips", observed, expected, max_flips=1)["status"] == "fail"
    canary = verdict("flips", observed, expected, max_flips=0, items=["he/x", "gsm8k/c"])
    assert canary["status"] == "pass" and canary["details"]["groups"]["he"]["items"] == 1
    missing = verdict("flips", {"gsm8k/a": True}, expected, max_flips=5)
    assert missing["status"] == "fail" and "he/x" in missing["details"]["missing"]


# -- verdict softening, per-key maps, the idle condition ---------------------------------------------------------------


def test_todo_and_info_pins_warn_instead_of_failing_and_unseeded_pins_are_todo():
    assert verdict("exact", 1, 2, status="todo")["status"] == "warn"
    assert verdict("exact", 1, 2, severity="info")["status"] == "warn"
    assert verdict("exact", ci.MISSING, 2)["status"] == "missing"
    assert verdict("exact", ci.MISSING, 2, status="todo")["status"] == "warn"
    assert verdict("band", 50.0, None)["status"] == "todo"
    assert verdict("not_earlier", {"a": 1}, None, baseline=True)["status"] == "todo"


def test_per_key_maps_compare_every_baseline_key_and_report_unpinned_ones():
    expected = {"json": None, "chat": 8, "story": 6}
    result = verdict("not_earlier", {"json": None, "chat": 10, "story": 6, "new": 3}, expected)
    assert result["status"] == "pass" and result["note"] == "3/3 keys pass; unpinned keys ['new']"
    result = verdict("not_earlier", {"json": 40, "chat": 8}, expected)
    assert result["status"] == "fail"
    assert result["details"]["json"]["status"] == "fail" and result["details"]["story"]["status"] == "pass"
    assert verdict("floor", {"a": 0.97}, {"a": 0.98, "b": 0.9}, slack=0.02)["status"] == "missing"
    assert verdict("floor", {"a": 0.95}, {"a": 0.98, "b": 0.9}, slack=0.02)["status"] == "fail"
    assert verdict("floor", 0.95, {"a": 0.98}, slack=0.02)["status"] == "missing"


def test_device_timed_pins_are_not_gated_on_a_loaded_host_unless_a_wide_band_is_given():
    assert verdict("band", 55.0, 50.0, tolerance=0.02, idle_only=True, gated=False)["status"] == "not_gated"
    wide = verdict("band", 55.0, 50.0, tolerance=0.02, loaded_tolerance=0.15, idle_only=True, gated=False)
    assert wide["status"] == "pass" and wide["tolerance"] == 0.15 and "loaded host" in wide["note"]
    assert verdict("band", 55.0, 50.0, tolerance=0.02, idle_only=True, gated=True)["status"] == "fail"


# -- metric definitions ------------------------------------------------------------------------------------------------


def acceptance_document(indices, *, handoff=None):
    prompts = [
        {
            "prompt": name,
            "prompt_tokens": 40,
            "compared_tokens": 96,
            "divergence_index": index,
            "matched_tokens": 96 if index is None else index,
            "device_token_ids": list(range(96)) if index is None else list(range(index)) + [7] * (96 - index),
            "cpu_token_ids": list(range(96)),
        }
        for name, index in indices.items()
    ]
    return {
        "schema": "qwen38-chat-server-acceptance/v1",
        "continuation": 96,
        "gate_prompt": "json",
        "gate_pass": indices["json"] is None,
        "prompts": prompts,
        "handoff": handoff,
    }


def test_acceptance_metric_reads_the_replay_and_the_probes(tmp_path):
    indices = {"json": None, "chat": 8, "story": 6}
    ci._write_json(tmp_path / "acceptance.json", acceptance_document(indices, handoff={"pass": True, "orders": {}}))
    ci._write_json(
        tmp_path / "probes.json",
        {"echo": {"verbatim": True, "text": ci.ECHO_SENTENCE}, "completion": {"requested": 4, "completed": 3}},
    )
    observed, items = ci.extract_acceptance(tmp_path, {})
    assert items == ["chat", "json", "story"]
    assert observed["gate_pass"] is True and observed["gate_compared_tokens"] == 96
    assert observed["divergence_index"] == indices and observed["handoff_pass"] is True
    assert observed["echo_verbatim"] is True and observed["requests_incomplete"] == 1
    assert observed["matched_tokens_total"] == 96 + 8 + 6
    assert len(set(observed["device_token_ids_sha256"].values())) == 3
    (tmp_path / "probes.json").unlink()
    observed, _ = ci.extract_acceptance(tmp_path, {})
    assert "echo_verbatim" not in observed and "requests_incomplete" not in observed


def test_acceptance_metric_refuses_another_schema(expect_error, tmp_path):
    ci._write_json(tmp_path / "acceptance.json", {"schema": "other/v1"})
    with expect_error(ci.CIError, match="schema"):
        ci.extract_acceptance(tmp_path, {})


def timing_document(median_ms=50.0, delta=0):
    return {
        "schema": "qwen38-resident-target-only-b1-decode-timing/v2",
        "runtime": {"sha256": "ea" * 32},
        "source": {"head": "ab" * 20},
        "resident_target_full48_single_trace_chain": {
            "capture_ms": 4800.0,
            "program_cache": {"after_capture": 282, "after_replay": 282 + delta, "before_capture": 282, "delta": delta},
            "timing": {"median_ms": median_ms, "p90_ms": median_ms + 0.2, "measured_token_count": 32},
        },
    }


def write_perf_artifacts(job_dir, *, median_ms=50.0, ledger_rows=None):
    ci._write_json(job_dir / "timing.json", timing_document(median_ms))
    ci._write_json(
        job_dir / "server-result.json",
        {
            "runtime": {"sha256": "ea" * 32},
            "source": {"head": "cd" * 20},
            "chain": {
                "capture_ms": 4900.0,
                "allocated_context": 32768,
                "dram_after_captures": {"free_bytes_per_bank": 478_202_944},
            },
        },
    )
    ci._write_json(job_dir / "READY", {"pid": 1, "host": "127.0.0.1", "port": 8001, "startup_seconds": 290.5})
    rows = ledger_rows or [
        {
            "phase": "chat-request",
            "reset": True,
            "prefill_tokens": 100,
            "prefill_ms_per_prompt_token": 5.0,
            "completion_tokens": 200,
            "tokens_per_second": 19.5,
            "host_vmrss_kib": 4_000_000,
        },
        {
            "phase": "chat-request",
            "reset": True,
            "prefill_tokens": 32,
            "prefill_ms_per_prompt_token": 9.0,
            "completion_tokens": 8,
            "tokens_per_second": 15.0,
            "host_vmrss_kib": 4_000_100,
        },
        {
            "phase": "chat-request",
            "reset": False,
            "prefill_tokens": 300,
            "prefill_ms_per_prompt_token": 4.0,
            "completion_tokens": 300,
            "tokens_per_second": 19.3,
            "host_vmrss_kib": 4_000_200,
        },
        {
            "phase": "chat-request",
            "reset": True,
            "prefill_tokens": 128,
            "prefill_ms_per_prompt_token": 5.2,
            "completion_tokens": 64,
            "tokens_per_second": 19.7,
            "mtp": {"k": 4},
            "host_vmrss_kib": 4_000_300,
        },
    ]
    (job_dir / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    ci._write_json(
        job_dir / "probes.json",
        {
            "health_at_ready": {"program_cache_entries": 271, "host_vmrss_kib": 3_900_000},
            "ttft": [
                {"requested_prompt_tokens": 128, "prompt_tokens": 131, "seconds": 0.61},
                {"requested_prompt_tokens": 1024, "prompt_tokens": 1030, "seconds": 3.8},
            ],
        },
    )


def test_perf_metric_gathers_timing_ledger_ready_and_probes(tmp_path):
    write_perf_artifacts(tmp_path)
    observed, items = ci.extract_perf(tmp_path, {})
    assert observed["decode_ms_per_token"] == 50.0 and observed["program_cache_delta"] == 0
    assert observed["program_cache_after_capture"] == 282 and observed["timing_capture_ms"] == 4800.0
    assert observed["capture_ms"] == 4900.0 and observed["free_bytes_per_bank"] == 478_202_944
    assert observed["startup_seconds"] == 290.5
    # fresh requests of 64+ prompt tokens only (rows 1 and 4); decode over 32+ completion tokens without MTP (rows 1, 3)
    assert observed["prefill_ms_per_prompt_token"] == pytest.approx(5.1)
    assert observed["decode_tokens_per_second"] == pytest.approx(19.4)
    assert observed["ledger_requests"] == 4 and observed["host_vmrss_kib_last"] == 4_000_300
    assert observed["ttft_seconds/128"] == 0.61 and observed["ttft_prompt_tokens/1024"] == 1030
    assert observed["program_cache_entries_at_ready"] == 271
    assert items == ["single-trace-timing", "ledger", "ttft/128", "ttft/1024"]
    assert ci._runtime_identity(tmp_path) == {
        "sha256": "ea" * 32,
        "archive_sha256": None,
        "tt_metal_sha": None,
        "source_head": "cd" * 20,
    }
    # the server's newer result nests the extension and the bundle
    ci._write_json(
        tmp_path / "server-result.json",
        {
            "runtime": {
                "extension": {"sha256": "ab" * 32},
                "bundle": {"archive_sha256": "cd" * 32, "binary_base_tt_metal_sha": "f0" * 20},
            },
            "source": {"head": "ef" * 20},
            "chain": {},
        },
    )
    assert ci._runtime_identity(tmp_path) == {
        "sha256": "ab" * 32,
        "archive_sha256": "cd" * 32,
        "tt_metal_sha": "f0" * 20,
        "source_head": "ef" * 20,
    }
    ci._write_json(tmp_path / "server-result.json", {"runtime": {"bundle": {"archive_sha256": "cd" * 32}}, "chain": {}})
    merged = ci._runtime_identity(tmp_path)  # the server's bundle plus the timing runner's extension sha
    assert (
        merged["sha256"] == "ea" * 32 and merged["archive_sha256"] == "cd" * 32 and merged["source_head"] == "ab" * 20
    )
    (tmp_path / "server-result.json").unlink()
    assert ci._runtime_identity(tmp_path)["sha256"] == "ea" * 32
    ci._write_json(tmp_path / "timing.json", {"runtime": "none"})
    assert ci._runtime_identity(tmp_path) is None


def test_perf_metric_tolerates_absent_files(tmp_path):
    ci._write_json(tmp_path / "timing.json", timing_document(49.9, delta=1))
    observed, items = ci.extract_perf(tmp_path, {})
    assert observed == {
        "decode_ms_per_token": 49.9,
        "decode_p90_ms_per_token": pytest.approx(50.1),
        "decode_measured_tokens": 32,
        "program_cache_delta": 1,
        "program_cache_after_capture": 282,
        "timing_capture_ms": 4800.0,
    }
    assert items == ["single-trace-timing"]


def write_samples(directory, task, rows):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"samples_{task}_2026-09-04T02-14-19.575940.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def gsm8k_rows():
    rows = []
    for item, strict, flexible in (("g1", 0.0, 1.0), ("g2", 0.0, 0.0), ("g3", 1.0, 1.0)):
        for filter_name, value in (("strict-match", strict), ("flexible-extract", flexible)):
            rows.append(
                {
                    "doc_id": len(rows),
                    "doc": {"item_id": item},
                    "filter": filter_name,
                    "metrics": ["exact_match"],
                    "exact_match": value,
                    "filtered_resps": [f"{item}-{filter_name}"],
                }
            )
    return rows


def test_eval_metric_scores_each_task_by_its_filter_and_metric(expect_error, tmp_path):
    write_samples(tmp_path / "Qwen__x", "qwen38_gsm8k", gsm8k_rows())
    write_samples(
        tmp_path / "Qwen__x",
        "qwen38_humaneval",
        [
            {
                "doc_id": 0,
                "doc": {"item_id": "HumanEval/0"},
                "filter": "create_test",
                "metrics": ["pass@1"],
                "pass@1": 1.0,
                "filtered_resps": [["def f(): pass"]],
            },
            {
                "doc_id": 1,
                "doc": {},
                "filter": "create_test",
                "metrics": ["pass@1"],
                "pass@1": 0.0,
                "filtered_resps": [[""]],
            },
        ],
    )
    write_samples(
        tmp_path / "Qwen__x",
        "other_task",
        [{"doc_id": 5, "doc": {}, "filter": "none", "metrics": ["acc"], "acc": True, "filtered_resps": ["a"]}],
    )
    observed, items = ci.extract_eval(tmp_path, {})
    assert observed["item_pass"] == {
        "qwen38_gsm8k/g1": True,
        "qwen38_gsm8k/g2": False,
        "qwen38_gsm8k/g3": True,
        "qwen38_humaneval/HumanEval/0": True,
        "qwen38_humaneval/doc1": False,
        "other_task/doc5": True,
    }
    assert observed["accuracy"] == {"other_task": 1.0, "qwen38_gsm8k": 0.6667, "qwen38_humaneval": 0.5}
    assert observed["items_per_task"] == {"other_task": 1, "qwen38_gsm8k": 3, "qwen38_humaneval": 2}
    assert items == sorted(observed["item_pass"])
    # the strict filter's answers differ from the flexible ones: the scored filter's sha is recorded
    strict, _ = ci.extract_eval(tmp_path, {"tasks": {"qwen38_gsm8k": ["strict-match", "exact_match"]}})
    assert strict["item_pass"]["qwen38_gsm8k/g1"] is False
    assert strict["item_answer_sha256"]["qwen38_gsm8k/g1"] != observed["item_answer_sha256"]["qwen38_gsm8k/g1"]
    with expect_error(ci.CIError, match="samples"):
        ci.extract_eval(tmp_path / "empty", {})


def score_document(items):
    rows = [
        {
            "item_id": item_id,
            "positions": 300,
            "top1": top1,
            "top5": 0.99,
            "a_in_b_top32": 1.0,
            "clear_positions": 250,
            "clear_top1": clear,
            "kl_mean": kl,
            "first_divergence": first,
        }
        for item_id, (top1, clear, kl, first) in items.items()
    ]
    return {
        "schema": "qwen38-reference-score/v1",
        "clear_margin_logits": 0.125,
        "corpus": {
            "items": len(rows),
            "positions": 300 * len(rows),
            "top1": 0.95,
            "top5": 0.99,
            "a_in_b_top32": 1.0,
            "clear_positions": 250 * len(rows),
            "clear_top1": 0.995,
            "kl_mean": 0.02,
            "teacher_logprob_gap_mean": 0.01,
        },
        "parts": {"acceptance": {"top1": 0.95}},
        "items": rows,
    }


def test_corpus_agreement_metric_reads_the_scorer_output(tmp_path):
    ci._write_json(
        tmp_path / "score.json",
        score_document({"acceptance-json": (0.97, 1.0, 0.01, None), "acceptance-chat": (0.93, 0.99, 0.03, 12)}),
    )
    observed, items = ci.extract_corpus_agreement(tmp_path, {})
    assert items == ["acceptance-chat", "acceptance-json"]
    assert (
        observed["corpus_top1"] == 0.95
        and observed["corpus_clear_top1"] == 0.995
        and observed["corpus_kl_mean"] == 0.02
    )
    assert observed["top1"] == {"acceptance-json": 0.97, "acceptance-chat": 0.93}
    assert observed["first_divergence"] == {"acceptance-json": None, "acceptance-chat": 12}
    assert observed["part_top1"] == {"acceptance": 0.95}


# -- validation and seeding --------------------------------------------------------------------------------------------


def pins_document(**pins):
    return {
        "schema": ci.PINS_SCHEMA,
        "default_tolerance": 0.15,
        "idle_loadavg_1min": 16,
        "configurations": {"cfg": {"description": "test"}},
        "pins": pins,
    }


def result_document(job, observed, *, gated=True, run="2026-09-06/000000Z-a-abc", end="2026-09-06T00:00:00Z"):
    return {
        "schema": ci.RESULT_SCHEMA,
        "run": run,
        "job": job,
        "metric": "perf",
        "configuration": "cfg",
        "host": "h",
        "lane": "a",
        "head": "ab" * 20,
        "gated": gated,
        "observed": observed,
        "utc": {"start": end, "end": end},
        "error": None,
    }


def test_validate_matches_pins_by_job_and_configuration_and_sets_the_result_status(tmp_path):
    pins = pins_document(
        **{
            "D1/cfg/decode_ms_per_token": {"rule": "band", "status": "active", "target": 50.0, "tolerance": 0.02},
            "D1/cfg/program_cache_delta": {"rule": "exact", "status": "active", "target": 0},
            "D1/cfg/startup_seconds": {"rule": "band", "status": "todo", "target": None},
            "D1/cfg/handoff_pass": {"rule": "exact", "status": "active", "target": True, "optional": True},
            "D1/other/decode_ms_per_token": {"rule": "band", "status": "active", "target": 1.0},
            "A3/cfg/divergence_index": {"rule": "not_earlier", "status": "active", "baseline": True},
        }
    )
    pins["configurations"]["other"] = {}
    result = ci.validate(
        result_document("D1", {"decode_ms_per_token": 50.5, "program_cache_delta": 0, "startup_seconds": 300.0}),
        pins,
        tmp_path,
    )
    assert [v["pin"] for v in result["verdicts"]] == [
        "D1/cfg/decode_ms_per_token",
        "D1/cfg/program_cache_delta",
        "D1/cfg/startup_seconds",
    ]
    assert [v["status"] for v in result["verdicts"]] == ["pass", "pass", "todo"] and result["status"] == "pass"
    result = ci.validate(result_document("D1", {"decode_ms_per_token": 50.5, "program_cache_delta": 1}), pins, tmp_path)
    assert result["status"] == "fail" and [v["status"] for v in result["verdicts"]] == ["pass", "fail", "warn"]
    assert result["verdicts"][2]["note"] == "todo pin, warning only: no observation"
    # an unseeded baseline pin is todo; a seeded one gates
    result = ci.validate(result_document("A3", {"divergence_index": {"json": None, "chat": 8}}), pins, tmp_path)
    assert result["verdicts"][0]["status"] == "todo"
    ci._write_json(
        ci.baseline_path("A3/cfg/divergence_index", tmp_path),
        {"schema": ci.BASELINE_SCHEMA, "pin": "A3/cfg/divergence_index", "expected": {"json": None, "chat": 8}},
    )
    result = ci.validate(result_document("A3", {"divergence_index": {"json": None, "chat": 7}}), pins, tmp_path)
    assert result["status"] == "fail" and result["verdicts"][0]["details"]["chat"]["status"] == "fail"
    assert "chat: observed 7 expected 8" in ci.format_verdicts(result)
    errored = ci.validate({**result_document("D1", {}), "error": "CIError: boom"}, pins, tmp_path)
    assert errored["status"] == "error"
    # a job of another id validates against the pin family it names
    named = ci.validate(
        {**result_document("D1-second", {"decode_ms_per_token": 50.5, "program_cache_delta": 0}), "pin_job": "D1"},
        pins,
        tmp_path,
    )
    assert [v["pin"] for v in named["verdicts"]][:2] == ["D1/cfg/decode_ms_per_token", "D1/cfg/program_cache_delta"]


def test_load_pins_rejects_unknown_rules_statuses_and_configurations(expect_error, tmp_path):
    for pins in (
        pins_document(**{"D1/cfg/x": {"rule": "median", "status": "todo"}}),
        pins_document(**{"D1/cfg/x": {"rule": "exact", "status": "proposed"}}),
        pins_document(**{"D1/nope/x": {"rule": "exact", "status": "todo"}}),
        pins_document(**{"D1/x": {"rule": "exact", "status": "todo"}}),
    ):
        ci._write_json(tmp_path / "pins.json", pins)
        with expect_error(ci.CIError):
            ci.load_pins(tmp_path / "pins.json")
    ci._write_json(tmp_path / "pins.json", {**pins_document(), "schema": "other"})
    with expect_error(ci.CIError, match="schema"):
        ci.load_pins(tmp_path / "pins.json")


def three_runs(values, *, gated=(True, True, True), job="D1", metric="decode_ms_per_token"):
    return [
        result_document(
            job, {metric: value}, gated=g, run=f"2026-09-0{i + 1}/000000Z-a-abc", end=f"2026-09-0{i + 1}T00:00:00Z"
        )
        for i, (value, g) in enumerate(zip(values, gated))
    ]


def test_seed_promotes_a_band_pin_from_three_agreeing_idle_runs_to_their_median(tmp_path):
    pins = pins_document(
        **{"D1/cfg/decode_ms_per_token": {"rule": "band", "status": "todo", "tolerance": 0.02, "target": 50.0}}
    )
    rows = ci.seed(pins, three_runs([50.1, 49.9, 50.0]), baselines_dir=tmp_path, write=True)
    assert rows == [
        {
            "pin": "D1/cfg/decode_ms_per_token",
            "runs": 3,
            "promoted": True,
            "values": [50.1, 49.9, 50.0],
            "agree": True,
            "seeds": ["2026-09-01/000000Z-a-abc", "2026-09-02/000000Z-a-abc", "2026-09-03/000000Z-a-abc"],
            "target": 50.0,
        }
    ]
    pin = pins["pins"]["D1/cfg/decode_ms_per_token"]
    assert pin["status"] == "active" and pin["target"] == 50.0 and len(pin["seeded"]["runs"]) == 3


def test_seed_needs_three_gated_runs_that_agree_and_only_the_last_three_count(tmp_path):
    pins = pins_document(**{"D1/cfg/decode_ms_per_token": {"rule": "band", "status": "todo", "tolerance": 0.02}})
    assert (
        ci.seed(pins, three_runs([50.0, 50.0, 50.0], gated=(True, False, True)), baselines_dir=tmp_path)[0]["note"]
        == "2 gated runs, 3 needed"
    )
    assert (
        ci.seed(pins, three_runs([50.0, 52.0, 50.0]), baselines_dir=tmp_path)[0]["note"]
        == "the runs disagree beyond the tolerance"
    )
    assert pins["pins"]["D1/cfg/decode_ms_per_token"]["status"] == "todo"
    four = three_runs([40.0, 50.0, 50.2]) + three_runs([49.8], gated=(True,))
    four[-1]["run"], four[-1]["utc"]["end"] = "2026-09-04/000000Z-a-abc", "2026-09-04T00:00:00Z"
    rows = ci.seed(pins, four, baselines_dir=tmp_path, write=True)
    assert rows[0]["promoted"] and rows[0]["values"] == [50.0, 50.2, 49.8] and rows[0]["target"] == 50.0
    # a dry run writes nothing
    dry = pins_document(**{"D1/cfg/decode_ms_per_token": {"rule": "band", "status": "todo", "tolerance": 0.02}})
    assert ci.seed(dry, three_runs([50.0, 50.0, 50.0]), baselines_dir=tmp_path)[0]["promoted"]
    assert dry["pins"]["D1/cfg/decode_ms_per_token"]["status"] == "todo"


def test_seed_keeps_a_preset_floor_takes_the_minimum_otherwise_and_writes_baselines(tmp_path):
    pins = pins_document(
        **{
            "A1/cfg/corpus_clear_top1": {"rule": "floor", "status": "todo", "target": 0.99, "slack": 0.0},
            "A1/cfg/corpus_top1": {"rule": "floor", "status": "todo", "slack": 0.01},
            "A1/cfg/top1": {"rule": "floor", "status": "todo", "slack": 0.02, "baseline": True},
            "A3/cfg/divergence_index": {"rule": "not_earlier", "status": "todo", "baseline": True},
        }
    )
    runs = []
    for i, (clear, top1, items, indices) in enumerate(
        (
            (0.995, 0.951, {"a": 0.97, "b": 0.93}, {"json": None, "chat": 8}),
            (0.992, 0.949, {"a": 0.96, "b": 0.94}, {"json": None, "chat": 8}),
            (0.998, 0.955, {"a": 0.97, "b": 0.93}, {"json": None, "chat": 8}),
        )
    ):
        runs.append(
            result_document(
                "A1",
                {"corpus_clear_top1": clear, "corpus_top1": top1, "top1": items},
                run=f"r{i}",
                end=f"2026-09-0{i + 1}T00:00:00Z",
            )
        )
        runs.append(
            result_document("A3", {"divergence_index": indices}, run=f"s{i}", end=f"2026-09-0{i + 1}T00:00:00Z")
        )
    rows = {row["pin"]: row for row in ci.seed(pins, runs, baselines_dir=tmp_path, write=True)}
    assert (
        rows["A1/cfg/corpus_clear_top1"]["target"] == 0.99
        and pins["pins"]["A1/cfg/corpus_clear_top1"]["status"] == "active"
    )
    assert rows["A1/cfg/corpus_top1"]["target"] == 0.949
    assert rows["A1/cfg/top1"]["target"] == {"a": 0.96, "b": 0.93}
    assert ci.load_baseline("A1/cfg/top1", tmp_path)["expected"] == {"a": 0.96, "b": 0.93}
    baseline = ci.load_baseline("A3/cfg/divergence_index", tmp_path)
    assert baseline["expected"] == {"json": None, "chat": 8} and baseline["source"]["runs"] == ["s0", "s1", "s2"]
    assert (
        pins["pins"]["A3/cfg/divergence_index"]["status"] == "active"
        and "target" not in pins["pins"]["A3/cfg/divergence_index"]
    )
    # a preset floor the runs miss is not promoted; disagreeing indices are not promoted
    pins = pins_document(
        **{
            "A1/cfg/corpus_clear_top1": {"rule": "floor", "status": "todo", "target": 0.999},
            "A3/cfg/divergence_index": {"rule": "not_earlier", "status": "todo", "baseline": True},
        }
    )
    runs[-1]["observed"]["divergence_index"] = {"json": None, "chat": 9}
    rows = {row["pin"]: row for row in ci.seed(pins, runs, baselines_dir=tmp_path)}
    assert not rows["A1/cfg/corpus_clear_top1"]["promoted"] and not rows["A3/cfg/divergence_index"]["promoted"]


# -- the runner --------------------------------------------------------------------------------------------------------


def jobs_document(worktree, jobs, **fields):
    return {
        "schema": ci.JOBS_SCHEMA,
        "host": "test-host",
        "lane": "a",
        "worktree": str(worktree),
        "configuration": "cfg",
        "head": "ab" * 20,
        "tree": "cd" * 20,
        "clean": True,
        "jobs": jobs,
        **fields,
    }


def test_runner_executes_command_jobs_collects_artifacts_and_writes_results(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    score = score_document({"acceptance-json": (0.97, 1.0, 0.01, None)})
    writer = tmp_path / "write_score.py"
    writer.write_text(
        textwrap.dedent(
            f"""
            import json, os, sys
            score = json.loads({json.dumps(json.dumps(score))})
            json.dump(score, open(os.path.join(os.environ["Q38_CI_JOB_DIR"], "score.json"), "w"))
            os.makedirs({str(evidence)!r} + "/run-1", exist_ok=True)
            timing = json.loads({json.dumps(json.dumps(timing_document(50.3)))})
            json.dump(timing, open({str(evidence)!r} + "/run-1/result.json", "w"))
            print("ok")
            """
        )
    )
    jobs = jobs_document(
        tmp_path,
        [
            {"id": "A1", "metric": "corpus_agreement", "argv": [sys.executable, str(writer)]},
            {
                "id": "D1",
                "metric": "perf",
                "argv": [sys.executable, str(writer)],
                "artifacts": {"timing.json": f"{evidence}/run-*/result.json", "absent": f"{evidence}/nothing-*"},
            },
            {"id": "D2", "metric": "perf", "artifacts": {"timing.json": "$Q38_CI_RUN_DIR/D1/timing.json"}},
            {"id": "X", "metric": "perf", "argv": [sys.executable, "-c", "import sys; sys.exit(3)"]},
        ],
    )
    pins = pins_document(
        **{
            "A1/cfg/corpus_top1": {"rule": "floor", "status": "active", "target": 0.9},
            "D1/cfg/decode_ms_per_token": {
                "rule": "band",
                "status": "active",
                "target": 50.0,
                "tolerance": 0.02,
                "idle_only": True,
                "loaded_tolerance": 0.15,
            },
            "D2/cfg/program_cache_delta": {"rule": "exact", "status": "active", "target": 0},
            "X/cfg/decode_ms_per_token": {"rule": "band", "status": "todo"},
        }
    )
    pins["idle_loadavg_1min"] = 1e9
    summary = ci.Runner(jobs, tmp_path / "results", pins, baselines_dir=tmp_path).run()
    run_dir = tmp_path / "results" / summary["run"]
    assert summary["run"].endswith("-a-abababababab") and summary["head"] == "ab" * 20
    assert summary["jobs"]["A1"]["status"] == "pass" and summary["jobs"]["D1"]["status"] == "pass"
    assert summary["jobs"]["D2"] == {"status": "pass", "verdicts": {"D2/cfg/program_cache_delta": "pass"}}
    assert summary["jobs"]["X"]["status"] == "error" and summary["status"] == "fail"
    d1 = json.loads((run_dir / "D1" / "result.json").read_text())
    assert d1["artifacts"]["timing.json"] == str(evidence / "run-1" / "result.json") and "absent" not in d1["artifacts"]
    assert d1["observed"]["decode_ms_per_token"] == 50.3 and d1["runtime"]["sha256"] == "ea" * 32
    assert d1["command"]["exit_code"] == 0 and d1["gated"] is True and d1["item_ids"] == ["single-trace-timing"]
    x = json.loads((run_dir / "X" / "result.json").read_text())
    assert x["error"] == "CIError: command exited 3 (see command.log)" and x["status"] == "error"
    events = [json.loads(line)["event"] for line in (run_dir / "progress.log").read_text().splitlines()]
    assert (
        events[:3] == ["run_start", "job_start", "job_end"]
        and events.count("artifact_missing") == 1
        and events[-1] == "run_end"
    )
    assert (run_dir / "summary.txt").read_text().startswith("A1 cfg corpus_agreement  status pass")
    assert (run_dir / "jobs.json").exists() and (run_dir / "D1" / "command.log").read_text() == "ok\n"
    results = ci.load_run_results([run_dir])
    assert [r["job"] for r in results] == ["A1", "D1", "D2", "X"]
    # --only restricts the jobs; a second run gets its own directory
    second = ci.Runner(jobs, tmp_path / "results", pins, baselines_dir=tmp_path).run(only=["A1"])
    assert list(second["jobs"]) == ["A1"] and second["run"] != summary["run"]


def test_runner_marks_a_loaded_host_not_gated(tmp_path):
    ci._write_json(tmp_path / "score.json", score_document({"a": (0.9, 1.0, 0.0, None)}))
    jobs = jobs_document(
        tmp_path, [{"id": "D1", "metric": "perf", "artifacts": {"timing.json": str(tmp_path / "t.json")}}]
    )
    ci._write_json(tmp_path / "t.json", timing_document(50.0))
    pins = pins_document(
        **{
            "D1/cfg/decode_ms_per_token": {
                "rule": "band",
                "status": "active",
                "target": 50.0,
                "tolerance": 0.02,
                "idle_only": True,
            }
        }
    )
    pins["idle_loadavg_1min"] = -1
    summary = ci.Runner(jobs, tmp_path / "results", pins, baselines_dir=tmp_path).run()
    result = json.loads((tmp_path / "results" / summary["run"] / "D1" / "result.json").read_text())
    assert result["gated"] is False and result["verdicts"][0]["status"] == "not_gated" and summary["status"] == "pass"


FAKE_LAUNCHER = r"""
import json, os, signal, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

evidence = sys.argv[1]
os.makedirs(evidence, exist_ok=True)
acceptance = json.loads(sys.argv[2])
json.dump(acceptance, open(os.path.join(evidence, "acceptance.json"), "w"))
json.dump({"chain": {"capture_ms": 4900.0, "program_cache_entries": 271}, "runtime": {"sha256": "ea" * 32}, "source": {"head": "ab" * 20}}, open(os.path.join(evidence, "server-result.json"), "w"))
open(os.path.join(evidence, "requests.jsonl"), "w").write(json.dumps({"phase": "chat-request", "reset": True, "prefill_tokens": 100, "prefill_ms_per_prompt_token": 5.0, "completion_tokens": 100, "tokens_per_second": 19.6}) + "\n")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, document):
        body = json.dumps(document).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ready", "program_cache_entries": 271, "host_vmrss_kib": 4000000})
        else:
            self._send({"data": [{"id": "fake/model"}]})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert request["enable_thinking"] is False and request["temperature"] == 0
        content = request["messages"][0]["content"]
        prompt_tokens = len(content.split())
        if content.startswith("Repeat the following sentence"):
            text = "The quick brown fox jumps over the lazy dog."
        else:
            text = "fine"
        self._send({"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": prompt_tokens}})


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
json.dump({"pid": os.getpid(), "host": "127.0.0.1", "port": server.server_port, "startup_seconds": 1.5}, open(os.path.join(evidence, "READY"), "w"))
stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *a: stop.set())
while not stop.is_set():
    time.sleep(0.1)
server.shutdown()
open(os.path.join(evidence, "STOPPED"), "w").write("stopped\n")
"""


def test_runner_drives_a_server_job_through_ready_probes_and_stop(tmp_path):
    launcher = tmp_path / "launcher.py"
    launcher.write_text(FAKE_LAUNCHER)
    evidence = tmp_path / "evidence" / "q38-chat-server-x-abababababab-1"
    acceptance = acceptance_document({"json": None, "chat": 8, "story": 6})
    argv = [sys.executable, str(launcher), str(evidence), json.dumps(acceptance)]
    jobs = jobs_document(
        tmp_path,
        [
            {
                "id": "A3",
                "metric": "acceptance",
                "kind": "server",
                "argv": argv,
                "ready": str(tmp_path / "evidence" / "q38-chat-server-x-*" / "READY"),
                "ready_timeout_seconds": 60,
                "probes": {"echo": True, "completion_requests": 2, "ttft_prompt_tokens": [128, 1024]},
                "artifacts": {
                    name: str(tmp_path / "evidence" / "q38-chat-server-x-*" / name)
                    for name in ("acceptance.json", "server-result.json", "requests.jsonl", "READY", "STOPPED")
                },
            },
            {
                "id": "D1",
                "metric": "perf",
                "artifacts": {
                    name: f"$Q38_CI_RUN_DIR/A3/{name}"
                    for name in ("server-result.json", "requests.jsonl", "READY", "probes.json")
                },
            },
        ],
    )
    pins = pins_document(
        **{
            "A3/cfg/gate_pass": {"rule": "exact", "status": "active", "target": True},
            "A3/cfg/echo_verbatim": {"rule": "exact", "status": "active", "target": True},
            "A3/cfg/requests_incomplete": {"rule": "exact", "status": "active", "target": 0},
            "A3/cfg/divergence_index": {"rule": "not_earlier", "status": "active", "baseline": True},
            "D1/cfg/startup_seconds": {"rule": "band", "status": "active", "target": 1.5, "tolerance": 0.2},
            "D1/cfg/program_cache_entries_at_ready": {"rule": "exact", "status": "active", "target": 271},
            "D1/cfg/ttft_seconds/128": {"rule": "band", "status": "todo"},
        }
    )
    ci._write_json(
        ci.baseline_path("A3/cfg/divergence_index", tmp_path),
        {
            "schema": ci.BASELINE_SCHEMA,
            "pin": "A3/cfg/divergence_index",
            "expected": {"json": None, "chat": 8, "story": 9},
        },
    )
    pins["idle_loadavg_1min"] = 1e9
    summary = ci.Runner(jobs, tmp_path / "results", pins, baselines_dir=tmp_path).run()
    run_dir = tmp_path / "results" / summary["run"]
    a3 = json.loads((run_dir / "A3" / "result.json").read_text())
    assert a3["error"] is None, a3["error"]
    assert a3["command"]["exit_code"] == 0 and (evidence / "STOPPED").exists() and (run_dir / "A3" / "STOPPED").exists()
    probes = json.loads((run_dir / "A3" / "probes.json").read_text())
    assert probes["model"] == "fake/model" and probes["echo"]["verbatim"] is True
    assert probes["completion"] == {"requested": 2, "completed": 2, "finish_reasons": ["stop", "stop"]}
    assert [p["requested_prompt_tokens"] for p in probes["ttft"]] == [128, 1024] and probes["ttft"][1][
        "prompt_tokens"
    ] > 900
    verdicts = {v["pin"]: v["status"] for v in a3["verdicts"]}
    assert verdicts == {
        "A3/cfg/gate_pass": "pass",
        "A3/cfg/echo_verbatim": "pass",
        "A3/cfg/requests_incomplete": "pass",
        "A3/cfg/divergence_index": "fail",
    }
    assert a3["observed"]["requests_incomplete"] == 0 and a3["runtime"]["sha256"] == "ea" * 32
    d1 = json.loads((run_dir / "D1" / "result.json").read_text())
    assert d1["observed"]["startup_seconds"] == 1.5 and d1["observed"]["program_cache_entries_at_ready"] == 271
    assert d1["observed"]["decode_tokens_per_second"] == 19.6 and d1["observed"]["ttft_prompt_tokens/128"] > 100
    assert {v["pin"]: v["status"] for v in d1["verdicts"]} == {
        "D1/cfg/startup_seconds": "pass",
        "D1/cfg/program_cache_entries_at_ready": "pass",
        "D1/cfg/ttft_seconds/128": "todo",
    }
    events = [json.loads(line) for line in (run_dir / "progress.log").read_text().splitlines()]
    assert [e["event"] for e in events if e["event"] in ("server_ready", "server_stop")] == [
        "server_ready",
        "server_stop",
    ]
    assert summary["status"] == "fail"


def test_runner_reports_a_launcher_that_exits_before_ready(tmp_path):
    jobs = jobs_document(
        tmp_path,
        [
            {
                "id": "A3",
                "metric": "acceptance",
                "kind": "server",
                "argv": [sys.executable, "-c", "print('no')"],
                "ready": str(tmp_path / "never" / "READY"),
            }
        ],
    )
    summary = ci.Runner(jobs, tmp_path / "results", pins_document(), baselines_dir=tmp_path).run()
    result = json.loads((tmp_path / "results" / summary["run"] / "A3" / "result.json").read_text())
    assert (
        result["error"] == "CIError: the launcher exited 0 before READY (see launcher.log)"
        and result["status"] == "error"
    )


def test_load_jobs_checks_the_schema_metrics_kinds_and_ids(expect_error, tmp_path):
    good = jobs_document(tmp_path, [{"id": "A", "metric": "perf"}])
    ci._write_json(tmp_path / "jobs.json", good)
    assert ci.load_jobs(tmp_path / "jobs.json")["lane"] == "a"
    ci._write_json(
        tmp_path / "jobs.json",
        {**good, "jobs": [{"id": "A", "metric": "perf", "artifacts": {"probes.json": "/x/A3/probes.json"}}]},
    )
    assert ci.load_jobs(tmp_path / "jobs.json")["jobs"][0]["artifacts"] == {"probes.json": "/x/A3/probes.json"}
    for bad in (
        {**good, "schema": "x"},
        {**good, "jobs": [{"id": "A", "metric": "nope"}]},
        {**good, "jobs": [{"id": "A", "metric": "perf", "kind": "daemon"}]},
        {**good, "jobs": [{"id": "A", "metric": "perf", "kind": "server"}]},
        {**good, "jobs": [{"id": "A", "metric": "perf", "artifacts": {"result.json": "/x/*/result.json"}}]},
        {
            **good,
            "jobs": [
                {
                    "id": "A",
                    "metric": "perf",
                    "kind": "server",
                    "ready": "/x/READY",
                    "artifacts": {"probes.json": "/x/p"},
                }
            ],
        },
        {**good, "jobs": [{"id": "A", "metric": "perf"}, {"id": "A", "metric": "perf"}]},
        {k: v for k, v in good.items() if k != "lane"},
    ):
        ci._write_json(tmp_path / "jobs.json", bad)
        with expect_error(ci.CIError):
            ci.load_jobs(tmp_path / "jobs.json")


def test_main_validate_seed_and_report(tmp_path, capsys):
    pins = pins_document(**{"D1/cfg/decode_ms_per_token": {"rule": "band", "status": "todo", "tolerance": 0.02}})
    ci._write_json(tmp_path / "pins.json", pins)
    run_dirs = []
    for i, value in enumerate((50.0, 50.1, 49.9)):
        run_dir = tmp_path / "results" / f"2026-09-0{i + 1}" / "000000Z-a-abc"
        ci._write_json(
            run_dir / "D1" / "result.json",
            {
                **result_document("D1", {"decode_ms_per_token": value}, run=f"r{i}", end=f"2026-09-0{i + 1}T00:00:00Z"),
                "verdicts": [],
            },
        )
        run_dirs.append(str(run_dir))
    assert (
        ci.main(
            [
                "--pins",
                str(tmp_path / "pins.json"),
                "--baselines",
                str(tmp_path),
                "validate",
                "--result",
                f"{run_dirs[0]}/D1/result.json",
            ]
        )
        == 0
    )
    assert "todo      D1/cfg/decode_ms_per_token: observed 50.0 expected None" in capsys.readouterr().out
    assert (
        ci.main(
            [
                "--pins",
                str(tmp_path / "pins.json"),
                "--baselines",
                str(tmp_path),
                "seed",
                "--runs",
                *run_dirs,
                "--write",
            ]
        )
        == 0
    )
    assert "1 pins promoted" in capsys.readouterr().out
    promoted = ci.load_pins(tmp_path / "pins.json")["pins"]["D1/cfg/decode_ms_per_token"]
    assert promoted["status"] == "active" and promoted["target"] == 50.0
    assert (
        ci.main(["--pins", str(tmp_path / "pins.json"), "--baselines", str(tmp_path), "report", "--run", run_dirs[0]])
        == 0
    )
    assert "D1 cfg perf" in capsys.readouterr().out


# -- the committed pins and baselines ----------------------------------------------------------------------------------


def test_committed_pins_load_and_cover_the_four_metric_families():
    pins = ci.load_pins()
    jobs = {ci.split_pin_id(pin_id)[0] for pin_id in pins["pins"]}
    assert {"A1", "A2", "A3", "C1", "C2", "D1"} <= jobs
    assert pins["default_tolerance"] == 0.15 and pins["idle_loadavg_1min"] == 16 and pins["seed_runs"] == 3
    assert list(pins["pins"]) == sorted(pins["pins"]) and list(pins["configurations"]) == sorted(pins["configurations"])
    active = {pin_id for pin_id, pin in pins["pins"].items() if pin["status"] == "active"}
    # the hard rules of today gate from the start; every number stays todo until seeded
    assert active == {
        pin_id
        for pin_id in pins["pins"]
        if pin_id.endswith(
            (
                "/gate_pass",
                "/gate_compared_tokens",
                "/echo_verbatim",
                "/requests_incomplete",
                "/program_cache_delta",
                "/handoff_pass",
                "/equals_single_stream",  # the lanes' exactness gate: every lane stream equals the single-stream replay
                "/score_in_top5",  # the image class bound: every device token inside the CPU top-5 under teacher forcing
                "/score_max_gap",  # ... and no CPU log-prob gap above 2.0 nats
            )
        )
    }
    for pin_id, pin in pins["pins"].items():
        if pin["status"] == "todo" and pin.get("target") is not None and not pin.get("baseline"):
            assert pin["rule"] in ("floor", "band", "ceiling") and "note" in pin, pin_id  # a proposal says its source
        if pin["rule"] == "band":
            assert pin.get("better") in ("lower", "higher"), pin_id
        if pin["rule"] == "flips":
            assert pin.get("baseline") and pin["max_flips"] >= 2, pin_id
        if pin.get("idle_only"):
            assert pin["rule"] == "band" and pin["loaded_tolerance"] > pin["tolerance"], pin_id


def test_committed_baselines_belong_to_baseline_pins_and_carry_their_source():
    pins = ci.load_pins()
    files = sorted(ci.BASELINES_DIR.glob("*.json"))
    assert files, "no baselines"
    for path in files:
        document = json.loads(path.read_text())
        pin_id = document["pin"]
        assert pins["pins"][pin_id].get("baseline"), pin_id
        assert path == ci.baseline_path(pin_id) and document["schema"] == ci.BASELINE_SCHEMA
        assert document["status"] in ("proposal", "seeded") and document["source"]["head"], pin_id
        assert ci.load_baseline(pin_id) == document
    chunked = ci.load_baseline("A3/chunked-32k/divergence_index")["expected"]
    forced = ci.load_baseline("A3/forced-32k/divergence_index")["expected"]
    assert (
        set(chunked)
        == set(forced)
        == {
            "json",
            "chat",
            "code",
            "fact",
            "list",
            "math",
            "multilingual",
            "prose",
            "refactor",
            "sky",
            "story",
            "summary",
        }
    )
    assert chunked["json"] is None and forced["json"] is None
    assert (
        chunked
        == {  # the 2026-09-27 replay: gr_recip_last (the GR read's front with the rsqrt applied last) by default
            "json": None,
            "chat": 56,
            "code": None,
            "fact": 15,
            "list": 56,
            "math": 56,
            "multilingual": 9,
            "prose": 13,
            "refactor": 24,
            "sky": 19,
            "story": 6,
            "summary": 75,
        }
    )
    mtp4_table = {
        m: ci.load_baseline(f"A3/mtp4-32k/{m}")["expected"] for m in ("divergence_index", "device_token_ids_sha256")
    }
    mtp4 = mtp4_table["divergence_index"]
    assert set(mtp4) == set(chunked) and mtp4["json"] is None
    # the same replay through the MTP pass loop with the verify rows' GDN body on the fold (gdn_rows_scan, bitwise
    # the 1-row step) beside the MoE rows on two rings and gr_recip_last (2026-09-28): the stream leaves the reference
    # where plain decode does on 11 of 12 and LATER on math (61 against plain decode's 56: the re-associated norm moved
    # plain decode's near-tie at 56 while the MTP path's own logits held the reference's token there); before
    # gr_recip_last the fold read plain decode's index on all 12 (chat 2, math 61, summary 75 on 2026-09-26)
    assert mtp4 == {**chunked, "math": 61}
    # the slab tables (--prefill-slab 2048 with and without --mtp 4) carry a thirteenth row, the slab record: the one
    # acceptance record long enough to run a 2048-row slab (tools/acceptance/greedy-prompts/prompt-document-greedy.json,
    # replayed only under --prefill-slab).  The twelve study records stay under 2048 tokens (no slab fires), so on them
    # the drafting slab table is the --mtp 4 table by construction, streams included; the slab record's row is the only
    # pin exercising a slab body under MTP, tolerance-class against the chunked forms (docs/NUMERICS.md)
    for metric in ("divergence_index", "device_token_ids_sha256"):
        slab_mtp4 = ci.load_baseline(f"A3/slab2048-mtp4-32k/{metric}")
        slab = ci.load_baseline(f"A3/slab2048-32k/{metric}")
        assert set(slab_mtp4["expected"]) == set(slab["expected"]) == set(chunked) | {"document"}, metric
        assert {k: v for k, v in slab_mtp4["expected"].items() if k != "document"} == mtp4_table[metric], metric
        assert "the only pin exercising a slab body under MTP" in slab_mtp4["source"]["evidence"], metric
        assert "slab record" in slab["source"]["evidence"], metric
    items = ci.load_baseline("C2/greedy-nothink/item_pass")["expected"]
    per_task = {}
    for key, flag in items.items():
        per_task.setdefault(key.split("/", 1)[0], []).append(flag)
    assert {task: len(flags) for task, flags in per_task.items()} == {
        "qwen38_gsm8k": 100,
        "qwen38_humaneval": 50,
        "qwen38_ifeval": 100,
    }
    assert {task: round(sum(flags) / len(flags), 2) for task, flags in per_task.items()} == {
        "qwen38_gsm8k": 0.9,
        "qwen38_humaneval": 0.92,
        "qwen38_ifeval": 0.72,
    }
    # the eval baseline validates against itself with no flips, and fails when three items flip
    observed = dict(items)
    assert verdict("flips", observed, items, max_flips=2)["status"] == "pass"
    for key in [k for k, v in items.items() if v and k.startswith("qwen38_gsm8k/")][:3]:
        observed[key] = False
    assert verdict("flips", observed, items, max_flips=2)["status"] == "fail"


def test_the_harness_is_stdlib_only():
    source = (MODEL_DIR / "tools" / "ci" / "q38_ci.py").read_text()
    assert "import torch" not in source and "import ttnn" not in source and "import yaml" not in source
