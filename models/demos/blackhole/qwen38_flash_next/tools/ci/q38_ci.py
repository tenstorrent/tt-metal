# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The regression harness of this model directory: pins, baselines, one result file per job, verdicts, seeding.

    python -m ...tools.ci.q38_ci run --jobs JOBS.json --out RESULTS_ROOT      # execute a job list, validate each job
    python -m ...tools.ci.q38_ci validate --result RESULT.json               # re-validate one job result
    python -m ...tools.ci.q38_ci seed --runs RUN_DIR... [--pin ID...] --write  # promote todo pins from agreeing runs
    python -m ...tools.ci.q38_ci report --run RUN_DIR                       # the observed-vs-expected table

``pins.json`` (next to this file) holds every gated number: a pin is ``<job>/<configuration>/<metric>`` with a rule
(``band`` = symmetric relative band, two-sided like tt-metal's model targets: too fast is a stale target; ``floor`` =
target minus slack; ``ceiling`` = target times 1 + tolerance; ``not_earlier`` for divergence indices, null being the
largest; ``exact``; ``at_most``; ``flips`` for per-item eval answers), a status (``todo`` warns only; ``active``
gates) and its seeds.  A pin with ``baseline: true`` compares a per-key map (per prompt, item or task) against
``baselines/<pin id with - for />.json``.  A job may name the pin family it belongs to (``pin_job``, default
its id) so one run can carry a family's columns as separate jobs (A2 over two oracle columns).  ``todo`` pins are promoted by ``seed`` once three idle runs agree within
the pin's tolerance: the target is the median (``band``, ``ceiling``), the minimum (``floor``) or the common value.

A job result (``qwen38-ci-result/v1``) records the host, lane, head, runtime identity, the load average, the item
ids, every observed value and one verdict per pin with observed and expected side by side.  Device-timed pins are
``not_gated`` when the 1-minute load average is above the idle threshold (a ``loaded_tolerance`` on the pin widens
the band instead).  The metric definitions read the artifacts the existing tools write: the corpus scorer's
``score.json`` (A1), the server's ``acceptance.json`` plus the runner's probes (A3), the timing runner's result,
the server's ledger and the probes (D1), lm-eval ``samples_*.jsonl`` files (C1/C2).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import signal
import statistics
import string
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

CI_DIR = Path(__file__).resolve().parent
MODEL_DIR = CI_DIR.parents[1]
PINS_PATH = CI_DIR / "pins.json"
BASELINES_DIR = CI_DIR / "baselines"

PINS_SCHEMA = "qwen38-ci-pins/v1"
BASELINE_SCHEMA = "qwen38-ci-baseline/v1"
JOBS_SCHEMA = "qwen38-ci-jobs/v1"
RESULT_SCHEMA = "qwen38-ci-result/v1"
SUMMARY_SCHEMA = "qwen38-ci-summary/v1"
RULES = ("band", "floor", "ceiling", "not_earlier", "exact", "at_most", "flips")
PIN_STATUSES = ("todo", "active")
VERDICTS = ("pass", "fail", "warn", "missing", "not_gated", "todo")
DEFAULT_TOLERANCE = 0.15
DEFAULT_IDLE_LOADAVG = 16.0
SEED_RUNS = 3
RESERVED_JOB_FILES = {"result.json", "verdicts.txt", "command.log", "launcher.log"}  # plus probes.json on server jobs

ECHO_SENTENCE = "The quick brown fox jumps over the lazy dog."
ECHO_PROMPT = (
    f"Repeat the following sentence exactly, with no extra words, no quotes, and no commentary: {ECHO_SENTENCE}"
)
TTFT_FILLER = "The quick brown fox jumps over the lazy dog. "  # about ten tokens
COMPLETION_PROMPTS = ("Name three primary colours.", "What is 17 + 25?", "Write one sentence about the sea.", "Say hi.")
NO_THINKING = {"enable_thinking": False, "reasoning_effort": "low"}

EVAL_TASK_SCORING = {  # task -> (lm-eval filter, per-item metric)
    "qwen38_gsm8k": ("flexible-extract", "exact_match"),
    "qwen38_humaneval": ("create_test", "pass@1"),
    "qwen38_ifeval": ("none", "prompt_level_strict_acc"),
}


class CIError(RuntimeError):
    pass


MISSING = object()  # the metric is absent from a result (distinct from an observed None, a divergence index)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def _median(values: Sequence[float]) -> float | None:
    return None if not values else statistics.median(values)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -- pins and baselines ----------------------------------------------------------------------------------------------


def load_pins(path: Path = PINS_PATH) -> dict[str, Any]:
    document = _read_json(path)
    if document.get("schema") != PINS_SCHEMA:
        raise CIError(f"{path}: schema {document.get('schema')!r}, expected {PINS_SCHEMA!r}")
    for pin_id, pin in document["pins"].items():
        job, configuration, _metric = split_pin_id(pin_id)
        if pin["rule"] not in RULES:
            raise CIError(f"pin {pin_id}: rule {pin['rule']!r} not in {RULES}")
        if pin["status"] not in PIN_STATUSES:
            raise CIError(f"pin {pin_id}: status {pin['status']!r} not in {PIN_STATUSES}")
        if configuration not in document["configurations"]:
            raise CIError(f"pin {pin_id}: configuration {configuration!r} is not defined")
    return document


def split_pin_id(pin_id: str) -> tuple[str, str, str]:
    """``<job>/<configuration>/<metric>``; the metric may itself carry a ``/`` (``ttft_seconds/128``)."""

    parts = pin_id.split("/", 2)
    if len(parts) != 3 or not all(parts):
        raise CIError(f"pin id {pin_id!r} is not <job>/<configuration>/<metric>")
    return parts[0], parts[1], parts[2]


def baseline_path(pin_id: str, baselines_dir: Path = BASELINES_DIR) -> Path:
    return baselines_dir / (pin_id.replace("/", "-") + ".json")


def load_baseline(pin_id: str, baselines_dir: Path = BASELINES_DIR) -> dict[str, Any] | None:
    path = baseline_path(pin_id, baselines_dir)
    if not path.exists():
        return None
    document = _read_json(path)
    if document.get("schema") != BASELINE_SCHEMA or document.get("pin") != pin_id:
        raise CIError(f"{path}: not the {BASELINE_SCHEMA} baseline of {pin_id}")
    return document


def expected_value(pin_id: str, pin: dict[str, Any], baselines_dir: Path = BASELINES_DIR) -> Any:
    """The pin's target, or the baseline file's ``expected`` map for baseline-backed pins (None when unseeded)."""

    if pin.get("baseline"):
        baseline = load_baseline(pin_id, baselines_dir)
        return None if baseline is None else baseline["expected"]
    return pin.get("target")


# -- rules -----------------------------------------------------------------------------------------------------------


def _compare_scalar(rule: str, observed: Any, expected: Any, pin: dict[str, Any]) -> tuple[bool, str]:
    """One observation against one expectation; the note names the band or the reason."""

    if rule == "band":
        tolerance = pin["_tolerance"]
        low, high = expected * (1 - tolerance), expected * (1 + tolerance)
        if low <= observed <= high:
            return True, f"within [{low:.6g}, {high:.6g}]"
        better_high = pin.get("better", "lower") == "higher"
        stale = (observed > high) == better_high
        return False, ("stale target: better than the band" if stale else "regression") + f" [{low:.6g}, {high:.6g}]"
    if rule == "floor":
        slack = pin.get("slack", 0.0) + expected * pin.get("slack_relative", 0.0)
        return observed >= expected - slack, f"floor {expected - slack:.6g}"
    if rule == "ceiling":
        high = expected * (1 + pin["_tolerance"])
        if observed <= high:
            warn_high = expected * (1 + pin["warn_tolerance"]) if "warn_tolerance" in pin else None
            if warn_high is not None and observed > warn_high:
                return True, f"warning: above {warn_high:.6g} (ceiling {high:.6g})"
            return True, f"ceiling {high:.6g}"
        return False, f"ceiling {high:.6g}"
    if rule == "not_earlier":
        infinity = float("inf")
        ok = (infinity if observed is None else observed) >= (infinity if expected is None else expected)
        return ok, "not earlier than the baseline" if ok else "earlier than the baseline"
    if rule == "exact":
        return observed == expected, "equal" if observed == expected else "differs"
    if rule == "at_most":
        return observed <= expected, f"at most {expected}"
    raise CIError(f"rule {rule!r} is not scalar")


def _compare_flips(observed: dict[str, Any], expected: dict[str, Any], pin: dict[str, Any]) -> dict[str, Any]:
    """Per group (the key's prefix before ``/``): items the baseline passes and the run fails; changed items."""

    items = pin.get("items") or sorted(expected)
    groups: dict[str, dict[str, Any]] = {}
    missing = []
    for key in items:
        if key not in expected:
            continue
        group = groups.setdefault(key.split("/", 1)[0], {"items": 0, "flips": 0, "changed": 0, "flipped": []})
        group["items"] += 1
        if key not in observed:
            missing.append(key)
            continue
        if bool(observed[key]) != bool(expected[key]):
            group["changed"] += 1
            if expected[key] and not observed[key]:
                group["flips"] += 1
                group["flipped"].append(key)
    max_flips = pin["max_flips"]
    failed = {name: group["flips"] for name, group in groups.items() if group["flips"] > max_flips}
    return {
        "groups": groups,
        "missing": missing,
        "ok": not failed and not missing,
        "note": f"pass->fail flips per group at most {max_flips}" + (f"; over: {failed}" if failed else ""),
    }


def compare(pin_id: str, pin: dict[str, Any], observed: Any, expected: Any, *, gated: bool, defaults: dict) -> dict:
    """The verdict of one pin: observed and expected side by side, a status from ``VERDICTS`` and a note."""

    pin = {**pin, "_tolerance": pin.get("tolerance", defaults.get("default_tolerance", DEFAULT_TOLERANCE))}
    verdict = {
        "pin": pin_id,
        "rule": pin["rule"],
        "pin_status": pin["status"],
        "observed": observed,
        "expected": expected,
    }
    if pin["rule"] in ("band", "ceiling"):
        verdict["tolerance"] = pin["_tolerance"]
    if observed is MISSING or (observed is None and pin["rule"] != "not_earlier"):
        verdict.update(observed=None, status="missing", note="no observation")
        return _soften(verdict, pin)
    if expected is None and (pin["rule"] != "not_earlier" or pin.get("baseline")):
        verdict.update(status="todo", note="no target yet: seed it")
        return verdict
    if isinstance(expected, dict) and not isinstance(observed, dict):
        verdict.update(status="missing", note=f"the observation is not a per-key map: {_short(observed)}")
        return _soften(verdict, pin)
    if pin.get("idle_only") and not gated:
        if pin["rule"] == "band" and "loaded_tolerance" in pin:
            pin["_tolerance"] = pin["loaded_tolerance"]
            verdict["tolerance"] = pin["_tolerance"]
            verdict["note_load"] = "loaded host: the wide band applies"
        else:
            verdict.update(status="not_gated", note="loaded host: device-timed pin not gated")
            return verdict
    if pin["rule"] == "flips":
        detail = _compare_flips(observed or {}, expected, pin)
        verdict.update(details=detail, status="pass" if detail["ok"] else "fail", note=detail["note"])
    elif isinstance(expected, dict):
        details = {}
        for key, value in sorted(expected.items()):
            if key not in observed and pin["rule"] != "not_earlier":
                details[key] = {"observed": None, "expected": value, "status": "missing"}
                continue
            ok, note = _compare_scalar(pin["rule"], observed.get(key), value, pin)
            details[key] = {
                "observed": observed.get(key),
                "expected": value,
                "status": "pass" if ok else "fail",
                "note": note,
            }
        statuses = {detail["status"] for detail in details.values()}
        unpinned = sorted(set(observed) - set(expected))
        verdict.update(
            details=details,
            status="fail" if "fail" in statuses else "missing" if "missing" in statuses else "pass",
            note=f"{sum(d['status'] == 'pass' for d in details.values())}/{len(details)} keys pass"
            + (f"; unpinned keys {unpinned}" if unpinned else ""),
        )
    else:
        ok, note = _compare_scalar(pin["rule"], observed, expected, pin)
        verdict.update(status="pass" if ok else "fail", note=note)
        if ok and note.startswith("warning"):
            verdict["status"] = "warn"
    if "note_load" in verdict:
        verdict["note"] = f"{verdict['note']}; {verdict.pop('note_load')}"
    return _soften(verdict, pin)


def _soften(verdict: dict[str, Any], pin: dict[str, Any]) -> dict[str, Any]:
    """A todo pin and an info-severity pin never fail: their misses are warnings."""

    if verdict["status"] in ("fail", "missing"):
        if pin["status"] == "todo":
            verdict["status"], verdict["note"] = "warn", f"todo pin, warning only: {verdict['note']}"
        elif pin.get("severity") == "info":
            verdict["status"], verdict["note"] = "warn", f"informational: {verdict['note']}"
    return verdict


# -- metric definitions ----------------------------------------------------------------------------------------------


def extract_corpus_agreement(job_dir: Path, _job: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """A1: the corpus scorer's ``score.json`` (a = the HF reference, b = the column under test)."""

    score = _read_json(job_dir / "score.json")
    if score.get("schema") != "qwen38-reference-score/v1":
        raise CIError(f"{job_dir / 'score.json'}: schema {score.get('schema')!r}")
    corpus = score["corpus"]
    observed = {
        "corpus_top1": corpus["top1"],
        "corpus_top5": corpus["top5"],
        "corpus_clear_top1": corpus["clear_top1"],
        "corpus_kl_mean": corpus["kl_mean"],
        "corpus_positions": corpus["positions"],
        "top1": {row["item_id"]: row["top1"] for row in score["items"]},
        "clear_top1": {row["item_id"]: row["clear_top1"] for row in score["items"]},
        "first_divergence": {row["item_id"]: row["first_divergence"] for row in score["items"]},
        "part_top1": {part: row["top1"] for part, row in score["parts"].items()},
    }
    return observed, sorted(observed["top1"])


def extract_acceptance(job_dir: Path, job: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """A3: the server's startup replay (``acceptance.json``) and the runner's probes (``probes.json``).  A ``lanes``
    configuration (``--lanes B``) reads the lanes' replay instead (``acceptance-lanes.json``: the records through the
    lane scheduler, B at a time) and adds ``equals_single_stream`` (every lane stream equal to the single-stream replay
    made in the same process)."""

    lanes = str(job.get("configuration", "")).startswith("lanes")
    name = "acceptance-lanes.json" if lanes else "acceptance.json"
    acceptance = _read_json(job_dir / name)
    schema = "qwen38-chat-server-acceptance-lanes/v1" if lanes else "qwen38-chat-server-acceptance/v1"
    if acceptance.get("schema") != schema:
        raise CIError(f"{job_dir / name}: schema {acceptance.get('schema')!r}")
    prompts = acceptance["prompts"]
    gate = next((row for row in prompts if row["prompt"] == acceptance["gate_prompt"]), None)
    observed: dict[str, Any] = {
        "gate_pass": acceptance["gate_pass"],
        "gate_compared_tokens": None if gate is None else gate["compared_tokens"],
        "divergence_index": {row["prompt"]: row["divergence_index"] for row in prompts},
        "device_token_ids_sha256": {
            row["prompt"]: _sha256_text(json.dumps(row["device_token_ids"])) for row in prompts
        },
        "matched_tokens_total": sum(row["matched_tokens"] for row in prompts),
        "handoff_pass": None if acceptance.get("handoff") is None else acceptance["handoff"]["pass"],
    }
    if lanes:
        observed["equals_single_stream"] = acceptance["equals_single_stream"]
    probes_path = job_dir / "probes.json"
    if probes_path.exists():
        probes = _read_json(probes_path)
        echo = probes.get("echo")
        if echo is not None:
            observed["echo_verbatim"] = echo["verbatim"]
        completion = probes.get("completion")
        if completion is not None:
            observed["requests_incomplete"] = completion["requested"] - completion["completed"]
    return observed, sorted(observed["divergence_index"])


def _ledger_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("phase", "chat-request") == "chat-request":
                rows.append(row)
    return rows


def extract_perf(job_dir: Path, _job: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """D1: the timing runner's ``timing.json``, the server's ``server-result.json``, ``READY``, ``requests.jsonl`` and the
    runner's probes; every file is optional, the pins over absent files come out ``missing``."""

    observed: dict[str, Any] = {}
    items: list[str] = []
    timing_path = job_dir / "timing.json"
    if timing_path.exists():
        chain = _read_json(timing_path)["resident_target_full48_single_trace_chain"]
        timing = chain["timing"]
        observed.update(
            decode_ms_per_token=timing["median_ms"],
            decode_p90_ms_per_token=timing["p90_ms"],
            decode_measured_tokens=timing["measured_token_count"],
            program_cache_delta=chain["program_cache"]["delta"],
            program_cache_after_capture=chain["program_cache"]["after_capture"],
            timing_capture_ms=chain["capture_ms"],
        )
        items.append("single-trace-timing")
    result_path = job_dir / "server-result.json"
    if result_path.exists():
        chain = _read_json(result_path).get("chain") or {}
        if chain:
            observed.update(
                capture_ms=chain.get("capture_ms"),
                free_bytes_per_bank=(chain.get("dram_after_captures") or {}).get("free_bytes_per_bank"),
                allocated_context=chain.get("allocated_context"),
            )
    ready_path = job_dir / "READY"
    if ready_path.exists():
        observed["startup_seconds"] = _read_json(ready_path)["startup_seconds"]
    ledger_path = job_dir / "requests.jsonl"
    if ledger_path.exists():
        rows = _ledger_rows(ledger_path)
        prefill = [
            row["prefill_ms_per_prompt_token"]
            for row in rows
            if row.get("reset") and (row.get("prefill_tokens") or 0) >= 64 and row.get("prefill_ms_per_prompt_token")
        ]
        decode = [
            row["tokens_per_second"]
            for row in rows
            if row.get("tokens_per_second") and (row.get("completion_tokens") or 0) >= 32 and not row.get("mtp")
        ]
        observed.update(
            ledger_requests=len(rows),
            prefill_ms_per_prompt_token=_median(prefill),
            decode_tokens_per_second=_median(decode),
            host_vmrss_kib_last=rows[-1].get("host_vmrss_kib") if rows else None,
        )
        items.append("ledger")
    probes_path = job_dir / "probes.json"
    if probes_path.exists():
        probes = _read_json(probes_path)
        for probe in probes.get("ttft", []):
            observed[f"ttft_seconds/{probe['requested_prompt_tokens']}"] = probe["seconds"]
            observed[f"ttft_prompt_tokens/{probe['requested_prompt_tokens']}"] = probe["prompt_tokens"]
            items.append(f"ttft/{probe['requested_prompt_tokens']}")
        health = probes.get("health_at_ready")
        if health is not None:
            observed["program_cache_entries_at_ready"] = health.get("program_cache_entries")
            observed["host_vmrss_kib_at_ready"] = health.get("host_vmrss_kib")
    return observed, items


def eval_samples(paths: Sequence[Path], scoring: dict[str, Sequence[str]]) -> dict[str, dict[str, Any]]:
    """lm-eval ``samples_<task>_<stamp>.jsonl`` lines -> per ``<task>/<item>`` the pass flag and the answer's sha."""

    items: dict[str, dict[str, Any]] = {}
    for path in paths:
        task = path.name[len("samples_") :].rsplit("_", 1)[0]
        wanted_filter, metric = scoring.get(task, (None, None))
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if wanted_filter is not None and row.get("filter", wanted_filter) != wanted_filter:
                continue
            metric_name = metric or next(name for name in row["metrics"] if name in row)
            item = row["doc"].get("item_id", f"doc{row['doc_id']}")
            items[f"{task}/{item}"] = {
                "pass": bool(row[metric_name]),
                "answer_sha256": _sha256_text(json.dumps(row["filtered_resps"], sort_keys=True)),
                "filter": row.get("filter"),
                "metric": metric_name,
            }
    return items


def extract_eval(job_dir: Path, job: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """C1/C2: every lm-eval samples file under the job directory; ``job["tasks"]`` may override the scoring."""

    scoring = {**EVAL_TASK_SCORING, **{k: tuple(v) for k, v in (job.get("tasks") or {}).items()}}
    paths = sorted(job_dir.rglob("samples_*.jsonl"))
    if not paths:
        raise CIError(f"{job_dir}: no lm-eval samples_*.jsonl files")
    items = eval_samples(paths, scoring)
    by_task: dict[str, list[bool]] = {}
    for key, item in items.items():
        by_task.setdefault(key.split("/", 1)[0], []).append(item["pass"])
    observed = {
        "item_pass": {key: item["pass"] for key, item in items.items()},
        "item_answer_sha256": {key: item["answer_sha256"] for key, item in items.items()},
        "accuracy": {task: round(sum(flags) / len(flags), 4) for task, flags in sorted(by_task.items())},
        "items_per_task": {task: len(flags) for task, flags in sorted(by_task.items())},
    }
    return observed, sorted(items)


METRICS: dict[str, Callable[[Path, dict[str, Any]], tuple[dict[str, Any], list[str]]]] = {
    "corpus_agreement": extract_corpus_agreement,
    "acceptance": extract_acceptance,
    "perf": extract_perf,
    "eval": extract_eval,
}


# -- validation ------------------------------------------------------------------------------------------------------


def validate(result: dict[str, Any], pins: dict[str, Any], baselines_dir: Path = BASELINES_DIR) -> dict[str, Any]:
    """Every pin of the result's job and configuration against its observed values; the result's status:
    ``fail`` when an active pin fails or is missing, ``warn`` when only todo/info pins miss, else ``pass``."""

    verdicts = []
    for pin_id, pin in sorted(pins["pins"].items()):
        job, configuration, metric = split_pin_id(pin_id)
        if job != result.get("pin_job", result["job"]) or configuration != result["configuration"]:
            continue
        observed = result["observed"].get(metric, MISSING)
        if observed is MISSING and pin.get("optional"):
            continue
        verdicts.append(
            compare(
                pin_id, pin, observed, expected_value(pin_id, pin, baselines_dir), gated=result["gated"], defaults=pins
            )
        )
    statuses = [verdict["status"] for verdict in verdicts]
    result["verdicts"] = verdicts
    result["status"] = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    if result.get("error"):
        result["status"] = "error"
    return result


def format_verdicts(result: dict[str, Any]) -> str:
    lines = [
        f"{result['job']} {result['configuration']} {result['metric']}  status {result.get('status')}  "
        f"host {result['host']} lane {result['lane']} head {str(result['head'])[:12]} gated {result['gated']}"
    ]
    if result.get("error"):
        lines.append(f"  error: {result['error']}")
    for verdict in result.get("verdicts", []):
        if isinstance(verdict.get("details"), dict) and "groups" not in verdict["details"]:
            lines.append(f"  {verdict['status']:9} {verdict['pin']}: {verdict['note']}")
            for key, detail in verdict["details"].items():
                if detail["status"] != "pass":
                    lines.append(f"            {key}: observed {detail['observed']} expected {detail['expected']}")
        else:
            lines.append(
                f"  {verdict['status']:9} {verdict['pin']}: observed {_short(verdict['observed'])} "
                f"expected {_short(verdict['expected'])} ({verdict['note']})"
            )
    return "\n".join(lines)


def _short(value: Any) -> str:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return text if len(text) <= 80 else text[:77] + "..."


# -- seeding ---------------------------------------------------------------------------------------------------------


def load_run_results(run_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    results = []
    for run_dir in run_dirs:
        for path in sorted(run_dir.glob("*/result.json")):
            document = _read_json(path)
            if document.get("schema") == RESULT_SCHEMA:
                document["_path"] = str(path)
                results.append(document)
    return sorted(results, key=lambda r: r["utc"]["end"])


def _agree(rule: str, values: list[Any], pin: dict[str, Any], tolerance: float) -> tuple[bool, Any]:
    """Whether the runs' values agree within the pin's tolerance, and the target they seed: the median (band,
    ceiling), the minimum or the preset target (floor), the common value otherwise; per key for per-key maps."""

    if rule in ("band", "ceiling", "floor") and all(isinstance(v, dict) for v in values):
        if any(set(v) != set(values[0]) for v in values):
            return False, None
        targets = {}
        for key in values[0]:
            agree, target = _agree(rule, [v[key] for v in values], pin, tolerance)
            if not agree:
                return False, None
            targets[key] = target
        return True, targets
    if rule in ("band", "ceiling"):
        median = statistics.median(values)
        spread = (max(values) - min(values)) / abs(median) if median else float("inf")
        return spread <= tolerance, median
    if rule == "floor":
        if pin.get("target") is not None:
            return all(v >= pin["target"] - pin.get("slack", 0.0) for v in values), pin["target"]
        slack = pin.get("slack", 0.0) + min(values) * pin.get("slack_relative", 0.0)
        return max(values) - min(values) <= slack, min(values)
    same = all(json.dumps(v, sort_keys=True) == json.dumps(values[0], sort_keys=True) for v in values)
    return same, values[0]


def seed(
    pins: dict[str, Any],
    results: Sequence[dict[str, Any]],
    *,
    pin_ids: Sequence[str] = (),
    runs: int = SEED_RUNS,
    baselines_dir: Path = BASELINES_DIR,
    write: bool = False,
) -> list[dict[str, Any]]:
    """Promote todo pins whose last ``runs`` gated results agree; returns one report row per considered pin."""

    report = []
    for pin_id, pin in sorted(pins["pins"].items()):
        if pin_ids and pin_id not in pin_ids:
            continue
        if pin["status"] != "todo" and not pin_ids:
            continue
        job, configuration, metric = split_pin_id(pin_id)
        rows = [
            r
            for r in results
            if r.get("pin_job", r["job"]) == job
            and r["configuration"] == configuration
            and r["gated"]
            and metric in r["observed"]
            and r["observed"][metric] is not None
            and not r.get("error")
        ]
        row = {"pin": pin_id, "runs": len(rows), "promoted": False}
        if len(rows) < runs:
            row["note"] = f"{len(rows)} gated runs, {runs} needed"
            report.append(row)
            continue
        last = rows[-runs:]
        values = [r["observed"][metric] for r in last]
        tolerance = pin.get("tolerance", pins.get("default_tolerance", DEFAULT_TOLERANCE))
        agree, target = _agree(pin["rule"], values, pin, tolerance)
        row.update(values=values, agree=agree, seeds=[r["run"] for r in last])
        if not agree:
            row["note"] = "the runs disagree beyond the tolerance"
            report.append(row)
            continue
        row["promoted"] = True
        row["target"] = target
        if write:
            pin["status"] = "active"
            pin["seeded"] = {"runs": [r["run"] for r in last], "utc": utc_now()}
            if pin.get("baseline"):
                _write_json(
                    baseline_path(pin_id, baselines_dir),
                    {
                        "schema": BASELINE_SCHEMA,
                        "pin": pin_id,
                        "status": "seeded",
                        "source": {"runs": [r["run"] for r in last], "head": last[-1]["head"], "utc": utc_now()},
                        "expected": target,
                    },
                )
            else:
                pin["target"] = target
        report.append(row)
    return report


# -- the runner ------------------------------------------------------------------------------------------------------


def load_jobs(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    if document.get("schema") != JOBS_SCHEMA:
        raise CIError(f"{path}: schema {document.get('schema')!r}, expected {JOBS_SCHEMA!r}")
    for key in ("lane", "worktree", "configuration", "jobs"):
        if key not in document:
            raise CIError(f"{path}: no {key!r}")
    ids = [job["id"] for job in document["jobs"]]
    if len(set(ids)) != len(ids):
        raise CIError(f"{path}: duplicate job ids {ids}")
    for job in document["jobs"]:
        if job["metric"] not in METRICS:
            raise CIError(f"job {job['id']}: metric {job['metric']!r} not in {sorted(METRICS)}")
        if job.get("kind", "command") not in ("command", "server"):
            raise CIError(f"job {job['id']}: kind {job.get('kind')!r}")
        if not isinstance(job.get("pin_job", job["id"]), str) or not job.get("pin_job", job["id"]):
            raise CIError(f"job {job['id']}: pin_job must name a pin family")
        if job.get("kind", "command") == "server" and "ready" not in job:
            raise CIError(f"job {job['id']}: a server job needs the READY glob")
        reserved = RESERVED_JOB_FILES & set(job.get("artifacts") or {})
        if job.get("kind", "command") == "server" and "probes.json" in (job.get("artifacts") or {}):
            reserved.add("probes.json")
        if reserved:
            raise CIError(f"job {job['id']}: artifact names {sorted(reserved)} are the runner's own files")
    return document


def _git(worktree: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(worktree), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _loadavg() -> list[float]:
    try:
        return [round(v, 2) for v in os.getloadavg()]
    except OSError:
        return []


def _newest(pattern: str, not_before: float) -> Path | None:
    candidates = [Path(p) for p in glob.glob(pattern) if os.path.getmtime(p) >= not_before]
    return max(candidates, key=os.path.getmtime) if candidates else None


class Runner:
    """Executes a job list into ``out_root/<date>/<stamp>-<lane>-<head12>/`` and validates every job."""

    def __init__(
        self, jobs: dict[str, Any], out_root: Path, pins: dict[str, Any], *, baselines_dir: Path = BASELINES_DIR
    ):
        self.jobs = jobs
        self.pins = pins
        self.baselines_dir = baselines_dir
        self.worktree = Path(jobs["worktree"])
        self.head = jobs.get("head") or _git(self.worktree, "rev-parse", "HEAD")
        self.tree = jobs.get("tree") or _git(self.worktree, "rev-parse", "HEAD^{tree}")
        self.clean = jobs.get("clean")
        if self.clean is None:
            self.clean = _git(self.worktree, "status", "--porcelain", "--untracked-files=no") == ""
        self.host = jobs.get("host") or platform.node()
        started = datetime.now(timezone.utc)
        stem = f"{started:%Y-%m-%d}/{started:%H%M%SZ}-{jobs['lane']}-{self.head[:12]}"
        for attempt in range(1, 100):  # two runs within one second get distinct directories
            self.run_id = stem if attempt == 1 else f"{stem}-{attempt}"
            self.run_dir = out_root / self.run_id
            try:
                self.run_dir.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                continue
        else:
            raise CIError(f"no free run directory under {out_root / stem}")
        self.progress = self.run_dir / "progress.log"
        self.idle_loadavg = float(pins.get("idle_loadavg_1min", DEFAULT_IDLE_LOADAVG))

    def mark(self, event: str, **values: Any) -> None:
        line = json.dumps({"utc": utc_now(), "event": event, **values}, sort_keys=True)
        with self.progress.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def run(self, only: Sequence[str] = ()) -> dict[str, Any]:
        os.environ["Q38_CI_RUN_DIR"] = str(self.run_dir)
        _write_json(self.run_dir / "jobs.json", self.jobs)
        self.mark("run_start", run=self.run_id, host=self.host, head=self.head, tree=self.tree, clean=self.clean)
        results = []
        for job in self.jobs["jobs"]:
            if only and job["id"] not in only:
                continue
            results.append(self.run_job(job))
        summary = {
            "schema": SUMMARY_SCHEMA,
            "run": self.run_id,
            "host": self.host,
            "lane": self.jobs["lane"],
            "head": self.head,
            "tree": self.tree,
            "clean": self.clean,
            "configuration": self.jobs["configuration"],
            "jobs": {
                r["job"]: {"status": r["status"], "verdicts": {v["pin"]: v["status"] for v in r["verdicts"]}}
                for r in results
            },
            "status": (
                "fail"
                if any(r["status"] in ("fail", "error") for r in results)
                else "warn"
                if any(r["status"] == "warn" for r in results)
                else "pass"
            ),
            "utc": utc_now(),
        }
        _write_json(self.run_dir / "summary.json", summary)
        (self.run_dir / "summary.txt").write_text(
            "\n\n".join(format_verdicts(r) for r in results) + "\n", encoding="utf-8"
        )
        self.mark("run_end", status=summary["status"], jobs={k: v["status"] for k, v in summary["jobs"].items()})
        return summary

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        job_dir = self.run_dir / job["id"]
        job_dir.mkdir()
        started = time.time()
        result: dict[str, Any] = {
            "schema": RESULT_SCHEMA,
            "run": self.run_id,
            "job": job["id"],
            "pin_job": job.get("pin_job", job["id"]),  # the pin family: two columns of one job in one run
            "metric": job["metric"],
            "configuration": job.get("configuration", self.jobs["configuration"]),
            "host": self.host,
            "lane": self.jobs["lane"],
            "head": self.head,
            "tree": self.tree,
            "clean": self.clean,
            "runtime": None,
            "utc": {"start": utc_now(), "end": None},
            "loadavg": {"start": _loadavg(), "end": None},
            "gated": True,
            "command": {"argv": job.get("argv"), "exit_code": None, "seconds": None},
            "artifacts": {},
            "item_ids": [],
            "observed": {},
            "verdicts": [],
            "error": None,
            "status": None,
        }
        self.mark("job_start", job=job["id"], kind=job.get("kind", "command"), loadavg=result["loadavg"]["start"])
        try:
            if job.get("argv"):
                env = {
                    **os.environ,
                    **(job.get("env") or {}),
                    "Q38_CI_JOB_DIR": str(job_dir),
                    "Q38_CI_RUN_DIR": str(self.run_dir),
                }
                # ``$Q38_CI_JOB_DIR`` and ``$Q38_CI_RUN_DIR`` in the arguments name the job's own directory
                job = {**job, "argv": [string.Template(argument).safe_substitute(env) for argument in job["argv"]]}
                result["command"]["argv"] = job["argv"]
                if job.get("kind", "command") == "server":
                    self._run_server(job, job_dir, env, result)
                else:
                    self._run_command(job, job_dir, env, result)
            self._collect_artifacts(job, job_dir, started, result)
            result["observed"], result["item_ids"] = METRICS[job["metric"]](job_dir, job)
            result["runtime"] = _runtime_identity(job_dir)
        except (CIError, OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
            result["error"] = f"{type(error).__name__}: {error}"
            self.mark("job_error", job=job["id"], error=result["error"])
        result["command"]["seconds"] = round(time.time() - started, 1)
        result["utc"]["end"] = utc_now()
        result["loadavg"]["end"] = _loadavg()
        loads = [v[0] for v in (result["loadavg"]["start"], result["loadavg"]["end"]) if v]
        result["gated"] = bool(loads) and max(loads) <= self.idle_loadavg
        validate(result, self.pins, self.baselines_dir)
        _write_json(job_dir / "result.json", result)
        (job_dir / "verdicts.txt").write_text(format_verdicts(result) + "\n", encoding="utf-8")
        self.mark(
            "job_end",
            job=job["id"],
            status=result["status"],
            gated=result["gated"],
            seconds=result["command"]["seconds"],
        )
        return result

    def _run_command(self, job: dict[str, Any], job_dir: Path, env: dict[str, str], result: dict[str, Any]) -> None:
        with (job_dir / "command.log").open("wb") as log:
            process = subprocess.Popen(
                job["argv"],
                cwd=job.get("cwd") or str(self.worktree),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                result["command"]["exit_code"] = process.wait(timeout=job.get("timeout_seconds", 3600))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                result["command"]["exit_code"] = process.wait(timeout=60)
                raise CIError(f"command timed out after {job.get('timeout_seconds', 3600)} s")
        if result["command"]["exit_code"] != 0 and not job.get("ignore_exit_code"):
            raise CIError(f"command exited {result['command']['exit_code']} (see command.log)")

    def _run_server(self, job: dict[str, Any], job_dir: Path, env: dict[str, str], result: dict[str, Any]) -> None:
        """Start the launcher, wait for its READY marker, probe the server over HTTP, TERM the server pid."""

        started = time.time()
        with (job_dir / "launcher.log").open("wb") as log:
            process = subprocess.Popen(
                job["argv"],
                cwd=job.get("cwd") or str(self.worktree),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            ready_path = None
            deadline = started + job.get("ready_timeout_seconds", 1800)
            while ready_path is None:
                ready_path = _newest(job["ready"], started)
                if ready_path is None:
                    if process.poll() is not None:
                        result["command"]["exit_code"] = process.returncode
                        raise CIError(f"the launcher exited {process.returncode} before READY (see launcher.log)")
                    if time.time() > deadline:
                        os.killpg(process.pid, signal.SIGTERM)
                        raise CIError(f"no READY marker within {job.get('ready_timeout_seconds', 1800)} s")
                    time.sleep(5)
            ready = _read_json(ready_path)
            self.mark(
                "server_ready",
                job=job["id"],
                ready=str(ready_path),
                pid=ready["pid"],
                seconds=round(time.time() - started, 1),
            )
            base_url = f"http://{ready['host']}:{ready['port']}"
            probes = {"error": None}
            try:
                probes.update(
                    run_probes(
                        base_url, job.get("probes") or {}, log=lambda **kw: self.mark("probe", job=job["id"], **kw)
                    )
                )
            except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
                probes["error"] = f"{type(error).__name__}: {error}"
                self.mark("probe_error", job=job["id"], error=probes["error"])
            _write_json(job_dir / "probes.json", probes)
            self.mark("server_stop", job=job["id"], pid=ready["pid"])
            try:
                os.kill(ready["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                result["command"]["exit_code"] = process.wait(timeout=job.get("stop_timeout_seconds", 300))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                result["command"]["exit_code"] = process.wait(timeout=60)
                raise CIError("the launcher did not exit after TERM; killed")
            if probes["error"]:
                raise CIError(f"probe failed: {probes['error']}")

    def _collect_artifacts(self, job: dict[str, Any], job_dir: Path, started: float, result: dict[str, Any]) -> None:
        """Copy every named artifact into the job dir: a glob, ``$Q38_CI_RUN_DIR`` expanded (an earlier job's files),
        the newest match; written since the job started when the job ran a command over the host's evidence."""

        for name, pattern in (job.get("artifacts") or {}).items():
            expanded = os.path.expandvars(pattern)
            source = _newest(expanded, started if job.get("argv") and not expanded.startswith(str(self.run_dir)) else 0)
            if source is None:
                self.mark("artifact_missing", job=job["id"], artifact=name, pattern=pattern)
                continue
            shutil.copyfile(source, job_dir / name)
            result["artifacts"][name] = str(source)


def _runtime_identity(job_dir: Path) -> dict[str, Any] | None:
    """The runtime the artifacts were made with, merged over the server's and the timing runner's results: the
    extension's sha (``runtime.sha256`` or ``runtime.extension.sha256``), the bundle's archive sha and tt-metal base."""

    identity: dict[str, Any] = {"sha256": None, "archive_sha256": None, "tt_metal_sha": None, "source_head": None}
    for name in ("server-result.json", "timing.json"):
        path = job_dir / name
        if not path.exists():
            continue
        document = _read_json(path)
        runtime = document.get("runtime") if isinstance(document.get("runtime"), dict) else {}
        extension = runtime.get("extension") if isinstance(runtime.get("extension"), dict) else {}
        bundle = runtime.get("bundle") if isinstance(runtime.get("bundle"), dict) else {}
        found = {
            "sha256": runtime.get("sha256") or extension.get("sha256"),
            "archive_sha256": bundle.get("archive_sha256"),
            "tt_metal_sha": bundle.get("binary_base_tt_metal_sha"),
            "source_head": (document.get("source") or {}).get("head"),
        }
        identity = {key: identity[key] if identity[key] is not None else value for key, value in found.items()}
    return identity if identity["sha256"] or identity["archive_sha256"] else None


# -- HTTP probes -----------------------------------------------------------------------------------------------------


def _http_json(base_url: str, path: str, body: dict[str, Any] | None = None, timeout: float = 600.0) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def run_probes(base_url: str, spec: dict[str, Any], *, log: Callable[..., None] = lambda **kw: None) -> dict[str, Any]:
    """The client-side checks of a ready server: /health at ready, the verbatim echo (tt-metal's coherence guard),
    N short completions that must all finish, and TTFT at the requested prompt lengths (thinking off, one token)."""

    probes: dict[str, Any] = {"health_at_ready": _http_json(base_url, "/health")}
    model = _http_json(base_url, "/v1/models")["data"][0]["id"]
    probes["model"] = model

    def chat(content: str, max_tokens: int) -> tuple[dict[str, Any], float]:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
            **NO_THINKING,
        }
        begin = time.perf_counter()
        reply = _http_json(base_url, "/v1/chat/completions", body)
        return reply, time.perf_counter() - begin

    if spec.get("echo", True):
        reply, seconds = chat(ECHO_PROMPT, 32)
        text = reply["choices"][0]["message"].get("content") or ""
        probes["echo"] = {"verbatim": ECHO_SENTENCE in text, "text": text, "seconds": round(seconds, 3)}
        log(probe="echo", verbatim=probes["echo"]["verbatim"])
    requests = int(spec.get("completion_requests", 0))
    if requests:
        completed = 0
        finishes = []
        for index in range(requests):
            reply, _seconds = chat(COMPLETION_PROMPTS[index % len(COMPLETION_PROMPTS)], 48)
            finish = reply["choices"][0].get("finish_reason")
            finishes.append(finish)
            completed += finish in ("stop", "length")
        probes["completion"] = {"requested": requests, "completed": completed, "finish_reasons": finishes}
        log(probe="completion", completed=completed, requested=requests)
    ttft = []
    for prompt_tokens in spec.get("ttft_prompt_tokens", []):
        content = TTFT_FILLER * max(1, prompt_tokens // 10) + "Reply with one word."
        reply, seconds = chat(content, 1)
        usage = reply.get("usage") or {}
        ttft.append(
            {
                "requested_prompt_tokens": prompt_tokens,
                "prompt_tokens": usage.get("prompt_tokens"),
                "seconds": round(seconds, 4),
            }
        )
        log(
            probe="ttft",
            requested_prompt_tokens=prompt_tokens,
            prompt_tokens=usage.get("prompt_tokens"),
            seconds=round(seconds, 3),
        )
    probes["ttft"] = ttft
    return probes


# -- main ------------------------------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pins", type=Path, default=PINS_PATH)
    parser.add_argument("--baselines", type=Path, default=BASELINES_DIR)
    modes = parser.add_subparsers(dest="mode", required=True)
    run = modes.add_parser("run", help="execute a job list and validate every job")
    run.add_argument("--jobs", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True, help="results root: <out>/<date>/<stamp>-<lane>-<head12>/")
    run.add_argument("--only", nargs="*", default=[], help="job ids to run")
    validate_ = modes.add_parser("validate", help="re-validate a job result against the pins")
    validate_.add_argument("--result", type=Path, required=True)
    seed_ = modes.add_parser("seed", help="promote todo pins whose last runs agree")
    seed_.add_argument("--runs", type=Path, nargs="+", required=True, help="run directories (with <job>/result.json)")
    seed_.add_argument("--pin", nargs="*", default=[], help="pin ids (default: every todo pin)")
    seed_.add_argument("--agreeing-runs", type=int, default=SEED_RUNS)
    seed_.add_argument("--write", action="store_true", help="write pins.json and the baselines; a dry run otherwise")
    report = modes.add_parser("report", help="print a run's verdicts")
    report.add_argument("--run", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    pins = load_pins(args.pins)
    if args.mode == "run":
        summary = Runner(load_jobs(args.jobs), args.out, pins, baselines_dir=args.baselines).run(args.only)
        print(f"run {summary['run']}: {summary['status']}")
        return 0 if summary["status"] in ("pass", "warn") else 1
    if args.mode == "validate":
        result = validate(_read_json(args.result), pins, args.baselines)
        print(format_verdicts(result))
        return 0 if result["status"] in ("pass", "warn") else 1
    if args.mode == "seed":
        rows = seed(
            pins,
            load_run_results(args.runs),
            pin_ids=args.pin,
            runs=args.agreeing_runs,
            baselines_dir=args.baselines,
            write=args.write,
        )
        for row in rows:
            print(json.dumps(row, sort_keys=True))
        if args.write:
            _write_json(args.pins, pins)
            print(f"{args.pins}: {sum(row['promoted'] for row in rows)} pins promoted")
        return 0
    if args.mode == "report":
        for result in load_run_results([args.run]):
            print(format_verdicts(result))
            print()
        return 0
    raise CIError(args.mode)


if __name__ == "__main__":
    sys.exit(main())
