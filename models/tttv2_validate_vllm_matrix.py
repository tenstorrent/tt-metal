#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Validate canonical TTTv2 vLLM smoke, benchmark, or quality evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

FAILURE_PATTERNS = (
    re.compile(r"\bERROR\b", re.IGNORECASE),
    re.compile(r"\bCRITICAL\b", re.IGNORECASE),
    re.compile(r"Traceback"),
    re.compile(r"RuntimeError"),
    re.compile(r"EngineDead", re.IGNORECASE),
    re.compile(r"index_cpu", re.IGNORECASE),
    re.compile(r"sampled[- ]token[^\n]*(?:shape|dtype)", re.IGNORECASE),
    re.compile(r"invalid[- ]token", re.IGNORECASE),
    re.compile(r"SIGKILL", re.IGNORECASE),
    re.compile(r"\bOOM\b", re.IGNORECASE),
    re.compile(r"OutOfMemory", re.IGNORECASE),
    re.compile(r"\bKilled\b", re.IGNORECASE),
    # TTTv2 adaptors may otherwise turn an unusable declared cache into a
    # successful-looking run backed by a job-local /tmp cache.
    re.compile(
        r"(?:(?:fall(?:ing)? back|fallback)[^\n]*(?:job-local tensor cache|/tmp/tttv2_model_cache)|using job-local tensor cache)",
        re.IGNORECASE,
    ),
)
TRACE_PATTERNS = {"prefill": re.compile(r"Captured prefill trace"), "decode": re.compile(r"Captured decode trace")}
PROGRAM_PATTERNS = {
    "decode_compiles": re.compile(r"Compiled decode"),
    "sampling_compiles": re.compile(r"Compiled on-device sampling"),
}
BENCHMARK_METRICS = {
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
}
THROUGHPUT_METRICS = {"request_throughput", "output_throughput", "total_token_throughput"}
LATENCY_METRICS = {"median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms"}
CONTEXT_CLIENT_PROGRAM = r"""import json,pathlib,sys,urllib.request
spec=json.loads(sys.argv[1]); host=sys.argv[2]; port=sys.argv[3]; model=sys.argv[4]; output=pathlib.Path(sys.argv[5])
def call(prompt):
    payload={"model":model,"prompt":prompt,"max_tokens":spec["output_tokens"],"temperature":0,"ignore_eos":True}
    request=urllib.request.Request(f"http://{host}:{port}/v1/completions",data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(request,timeout=1800) as response:return {"request":payload,"response":json.loads(response.read())}
count=spec["input_tokens"]
if spec["kind"]=="cached_prefill":
    common=spec["common_prefix_tokens"]; first=[42]*common+[43]*(count-common); second=[42]*common+[44]*(count-common)
    calls=[call(first),call(second)]
else:calls=[call([42]*count)]
output.write_text(json.dumps({"schema_version":1,"subcase":spec,"calls":calls},indent=2)+"\n")"""


class Validation:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)

    def equal(self, actual: Any, expected: Any, message: str) -> None:
        if actual != expected:
            self.errors.append(f"{message}: expected {expected!r}, got {actual!r}")


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_exact_fields(
    value: Any,
    required: set[str],
    optional: set[str],
    name: str,
    validation: Validation,
) -> None:
    """Reject missing and unknown fields in an execution-affecting object."""
    validation.require(isinstance(value, dict), f"{name} must be an object")
    if not isinstance(value, dict):
        return
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required - optional)
    validation.require(not missing, f"{name} is missing required fields: {missing}")
    validation.require(not unknown, f"{name} has unknown fields: {unknown}")


def validate_process_spec(spec: Any, name: str, validation: Validation) -> None:
    validation.require(isinstance(spec, dict), f"{name} must be an object")
    if not isinstance(spec, dict):
        return
    validation.require(set(spec) == {"kind", "cwd", "argv", "env"}, f"{name} fields must be exactly kind/cwd/argv/env")
    validation.equal(spec.get("kind"), "process", f"{name}.kind")
    validation.require(
        isinstance(spec.get("cwd"), str) and Path(spec["cwd"]).is_absolute(), f"{name}.cwd must be absolute"
    )
    argv = spec.get("argv")
    validation.require(
        isinstance(argv, list) and bool(argv) and all(isinstance(v, str) and v for v in argv),
        f"{name}.argv must be a non-empty string list",
    )
    env = spec.get("env")
    validation.require(
        isinstance(env, dict) and all(isinstance(k, str) and k and isinstance(v, str) for k, v in env.items()),
        f"{name}.env must be a complete string map",
    )


def validate_performance_thresholds(value: Any, row_id: str, validation: Validation) -> None:
    validation.require(
        isinstance(value, dict) and bool(value), f"{row_id}: performance_thresholds must be a non-empty object"
    )
    if not isinstance(value, dict):
        return
    unknown = sorted(set(value) - BENCHMARK_METRICS)
    validation.require(not unknown, f"{row_id}: performance_thresholds has unknown metrics: {unknown}")
    for metric, threshold in value.items():
        validate_exact_fields(
            threshold, {"direction", "value"}, set(), f"{row_id}: performance_thresholds.{metric}", validation
        )
        if not isinstance(threshold, dict):
            continue
        direction = threshold.get("direction")
        limit = threshold.get("value")
        validation.require(direction in ("minimum", "maximum"), f"{row_id}: {metric} direction must be minimum/maximum")
        validation.require(
            isinstance(limit, (int, float)) and not isinstance(limit, bool) and math.isfinite(limit) and limit > 0,
            f"{row_id}: {metric} threshold must be finite and positive",
        )
        if metric in THROUGHPUT_METRICS:
            validation.equal(direction, "minimum", f"{row_id}: {metric} threshold direction")
        elif metric in LATENCY_METRICS:
            validation.equal(direction, "maximum", f"{row_id}: {metric} threshold direction")
    validation.require(
        any(metric in value for metric in THROUGHPUT_METRICS),
        f"{row_id}: performance_thresholds must include a throughput floor",
    )
    validation.require(
        any(metric in value for metric in LATENCY_METRICS),
        f"{row_id}: performance_thresholds must include a latency ceiling",
    )


def validate_expectations_document(expectations: dict[str, Any], expectations_path: Path | None = None) -> Validation:
    """Validate every execution-affecting field before a hardware action."""
    validation = Validation()
    validate_exact_fields(
        expectations,
        {
            "schema_version",
            "model_id",
            "model",
            "canonical_row_ids",
            "rows",
            "smoke",
            "quality",
            "common",
            "hf_cache",
            "execution",
            "provenance",
            "architecture",
            "generator",
        },
        set(),
        "expectations",
        validation,
    )
    rows = expectations.get("rows")
    canonical = expectations.get("canonical_row_ids")
    validation.equal(expectations.get("schema_version"), 2, "expectations schema_version")
    validation.require(
        isinstance(rows, list) and bool(rows) and all(isinstance(r, dict) for r in rows),
        "rows must be a non-empty object list",
    )
    ids = [row.get("id") for row in rows] if isinstance(rows, list) and all(isinstance(r, dict) for r in rows) else []
    validation.require(
        all(isinstance(i, str) and i for i in ids) and len(ids) == len(set(ids)),
        "row ids must be unique non-empty strings",
    )
    validation.equal(canonical, ids, "canonical_row_ids must exactly match rows order")
    validation.require(
        isinstance(expectations.get("model"), str) and bool(expectations.get("model")),
        "model must be a non-empty string",
    )
    validation.require(
        isinstance(expectations.get("model_id"), str) and bool(expectations.get("model_id")),
        "model_id must be a non-empty string",
    )
    validation.require(
        isinstance(expectations.get("architecture"), str) and bool(expectations.get("architecture")),
        "architecture must be a non-empty string",
    )
    validation.require(
        isinstance(expectations.get("generator"), str) and bool(expectations.get("generator")),
        "generator must be a non-empty string",
    )

    execution = expectations.get("execution")
    validate_exact_fields(
        execution,
        {"python", "vllm_dir", "server_script", "prompt", "validator"},
        set(),
        "execution",
        validation,
    )
    if isinstance(execution, dict):
        for key in ("python", "vllm_dir", "server_script", "prompt"):
            validation.require(
                isinstance(execution.get(key), str) and bool(execution.get(key)), f"execution.{key} must be non-empty"
            )
        for key in ("python", "vllm_dir"):
            validation.require(
                isinstance(execution.get(key), str) and Path(execution.get(key, "x")).is_absolute(),
                f"execution.{key} must be absolute",
            )
        validator = execution.get("validator")
        validate_exact_fields(validator, {"path", "sha256"}, set(), "execution.validator", validation)
        if isinstance(validator, dict):
            validation.require(
                isinstance(validator.get("path"), str) and Path(validator.get("path", "x")).is_absolute(),
                "execution.validator.path must be absolute",
            )
            validation.require(
                is_sha256(validator.get("sha256")), "execution.validator.sha256 must be lowercase SHA-256"
            )
            if (
                expectations_path is not None
                and isinstance(validator.get("path"), str)
                and is_sha256(validator.get("sha256"))
            ):
                try:
                    validation.equal(
                        sha256_file(Path(validator["path"])), validator["sha256"], "execution.validator live hash"
                    )
                except OSError as error:
                    validation.errors.append(f"cannot hash execution.validator.path: {error}")

    provenance = expectations.get("provenance")
    validate_exact_fields(
        provenance,
        {"required_capabilities", "characterization", "performance_floor"},
        set(),
        "provenance",
        validation,
    )
    if isinstance(provenance, dict):
        for name in ("required_capabilities", "characterization", "performance_floor"):
            item = provenance.get(name)
            validate_exact_fields(item, {"path", "sha256"}, set(), f"provenance.{name}", validation)
            if isinstance(item, dict):
                validation.require(
                    isinstance(item.get("path"), str) and Path(item.get("path", "x")).is_absolute(),
                    f"provenance.{name}.path must be absolute",
                )
                validation.require(is_sha256(item.get("sha256")), f"provenance.{name}.sha256 must be lowercase SHA-256")
                if (
                    expectations_path is not None
                    and isinstance(item.get("path"), str)
                    and is_sha256(item.get("sha256"))
                ):
                    try:
                        validation.equal(
                            sha256_file(Path(item["path"])), item["sha256"], f"provenance.{name} live hash"
                        )
                    except OSError as error:
                        validation.errors.append(f"cannot hash provenance.{name}.path: {error}")

    quality = expectations.get("quality")
    validate_exact_fields(
        quality,
        {"quality_tokens", "require_pair_review", "semantic_term_groups"},
        {"pair_review_file"},
        "quality",
        validation,
    )
    if isinstance(quality, dict):
        validation.require(
            type(quality.get("require_pair_review")) is bool, "quality.require_pair_review must be boolean"
        )
        validation.equal(quality.get("require_pair_review"), True, "quality.require_pair_review")
        groups = quality.get("semantic_term_groups")
        validation.require(
            isinstance(groups, list)
            and bool(groups)
            and all(
                isinstance(group, list)
                and bool(group)
                and all(isinstance(term, str) and bool(term.strip()) for term in group)
                for group in groups
            ),
            "quality.semantic_term_groups must be non-empty lists of non-empty strings",
        )
        validation.require(
            isinstance(quality.get("quality_tokens"), int)
            and not isinstance(quality.get("quality_tokens"), bool)
            and quality.get("quality_tokens", 0) > 0,
            "quality.quality_tokens must be a positive integer",
        )
        if "pair_review_file" in quality:
            pair_review_file = quality.get("pair_review_file")
            validation.require(
                isinstance(pair_review_file, str)
                and bool(pair_review_file)
                and Path(pair_review_file).name == pair_review_file,
                "quality.pair_review_file must be a non-empty basename",
            )

    smoke = expectations.get("smoke")
    validate_exact_fields(smoke, {"row_id", "tokens"}, set(), "smoke", validation)
    validation.require(isinstance(smoke, dict) and smoke.get("row_id") in ids, "smoke.row_id must name a canonical row")
    if isinstance(smoke, dict):
        validation.require(
            isinstance(smoke.get("tokens"), int)
            and not isinstance(smoke.get("tokens"), bool)
            and smoke.get("tokens", 0) > 0,
            "smoke.tokens must be a positive integer",
        )
        if smoke.get("row_id") in ids:
            validation.equal(
                rows[ids.index(smoke["row_id"])].get("manifest", {}).get("trace_mode"),
                "decode_only",
                "smoke row trace_mode",
            )

    hf_cache = expectations.get("hf_cache")
    validate_exact_fields(
        hf_cache,
        {"hf_home", "snapshot", "ref_path", "revision", "verified_files"},
        {"ref", "ref_revision"},
        "hf_cache",
        validation,
    )
    if isinstance(hf_cache, dict):
        for key in ("hf_home", "snapshot", "ref_path", "revision", "verified_files"):
            validation.require(key in hf_cache, f"hf_cache.{key} is required")
        for key in ("hf_home", "snapshot", "ref_path"):
            validation.require(
                isinstance(hf_cache.get(key), str) and Path(hf_cache.get(key, "x")).is_absolute(),
                f"hf_cache.{key} must be absolute",
            )
        validation.require(
            isinstance(hf_cache.get("revision"), str) and bool(hf_cache.get("revision")),
            "hf_cache.revision must be non-empty",
        )
        validation.require(
            isinstance(hf_cache.get("verified_files"), list)
            and bool(hf_cache.get("verified_files"))
            and all(isinstance(v, str) and v for v in hf_cache.get("verified_files", [])),
            "hf_cache.verified_files must be a non-empty string list",
        )
        for key in ("ref", "ref_revision"):
            if key in hf_cache:
                validation.require(
                    isinstance(hf_cache.get(key), str) and bool(hf_cache.get(key)), f"hf_cache.{key} must be non-empty"
                )

    common = expectations.get("common")
    validate_exact_fields(
        common,
        {
            "backend",
            "endpoint",
            "input_tokens",
            "output_tokens",
            "num_prompts",
            "max_concurrency",
            "request_rate",
            "temperature",
        },
        set(),
        "common",
        validation,
    )
    if isinstance(common, dict):
        for key in ("backend", "endpoint"):
            validation.require(
                isinstance(common.get(key), str) and bool(common.get(key)), f"common.{key} must be non-empty"
            )
        for key in ("input_tokens", "output_tokens", "num_prompts", "max_concurrency"):
            validation.require(
                isinstance(common.get(key), int) and not isinstance(common.get(key), bool) and common.get(key, 0) > 0,
                f"common.{key} must be a positive integer",
            )
        validation.require(
            isinstance(common.get("temperature"), (int, float))
            and not isinstance(common.get("temperature"), bool)
            and math.isfinite(common["temperature"])
            and common["temperature"] >= 0,
            "common.temperature must be finite and non-negative",
        )
        validation.require(
            common.get("request_rate") == "inf"
            or (
                isinstance(common.get("request_rate"), (int, float))
                and not isinstance(common.get("request_rate"), bool)
                and math.isfinite(common["request_rate"])
                and common["request_rate"] > 0
            ),
            "common.request_rate must be 'inf' or finite positive numeric",
        )

    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id", "<unknown>")
        validate_exact_fields(
            row,
            {"id", "manifest", "expected_traces", "expected_program_logs", "performance_thresholds"},
            set(),
            str(row_id),
            validation,
        )
        manifest = row.get("manifest")
        validate_exact_fields(
            manifest,
            {
                "model",
                "platform",
                "dp",
                "trace_mode",
                "trace_region_size",
                "sample_on_device_mode",
                "family_var",
                "family_version",
                "revision",
                "tokenizer_revision",
                "max_model_len",
                "max_num_seqs_per_rank",
                "async_scheduling",
                "prefix_caching",
                "cache_root",
                "visible_devices",
                "context_subcases",
            },
            {"fabric_config"},
            f"{row_id}: manifest",
            validation,
        )
        if not isinstance(manifest, dict):
            continue
        for key in (
            "model",
            "platform",
            "trace_mode",
            "sample_on_device_mode",
            "family_var",
            "family_version",
            "revision",
            "tokenizer_revision",
        ):
            validation.require(
                isinstance(manifest.get(key), str) and bool(manifest.get(key)),
                f"{row_id}: manifest.{key} must be non-empty",
            )
        family_var = manifest.get("family_var")
        reserved_env = {
            "PATH",
            "HOME",
            "LD_LIBRARY_PATH",
            "TT_METAL_HOME",
            "PYTHONNOUSERSITE",
            "MESH_DEVICE",
            "HF_MODEL",
            "HF_HOME",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "TOKENIZERS_PARALLELISM",
            "TT_CACHE_PATH",
            "TT_VISIBLE_DEVICES",
        }
        validation.require(
            isinstance(family_var, str)
            and re.fullmatch(r"[A-Z][A-Z0-9_]*", family_var or "") is not None
            and family_var not in reserved_env
            and not family_var.startswith("DISABLE_"),
            f"{row_id}: family_var is unsafe or reserved",
        )
        validation.equal(manifest.get("model"), expectations.get("model"), f"{row_id}: manifest.model")
        validation.require(
            manifest.get("trace_mode") in ("decode_only", "all"), f"{row_id}: trace_mode must be decode_only/all"
        )
        for key in ("trace_region_size", "max_model_len", "max_num_seqs_per_rank"):
            validation.require(
                isinstance(manifest.get(key), int)
                and not isinstance(manifest.get(key), bool)
                and manifest.get(key, 0) > 0,
                f"{row_id}: manifest.{key} must be a positive integer",
            )
        for key in ("async_scheduling", "prefix_caching"):
            validation.require(type(manifest.get(key)) is bool, f"{row_id}: manifest.{key} must be boolean")
        expected_traces = row.get("expected_traces")
        validation.require(
            isinstance(expected_traces, dict)
            and set(expected_traces) == {"prefill", "decode"}
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in expected_traces.values()),
            f"{row_id}: expected_traces must exactly contain non-negative prefill/decode counts",
        )
        expected_programs = row.get("expected_program_logs")
        validation.require(
            isinstance(expected_programs, dict)
            and set(expected_programs) == {"decode_compiles", "sampling_compiles"}
            and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in expected_programs.values()),
            f"{row_id}: expected_program_logs must exactly contain positive compile counts",
        )
        validate_performance_thresholds(row.get("performance_thresholds"), str(row_id), validation)
        cache_root = manifest.get("cache_root")
        validation.require(
            isinstance(cache_root, str) and Path(cache_root or "x").is_absolute(),
            f"{row_id}: cache_root must be absolute",
        )
        dp = manifest.get("dp")
        validation.require(
            isinstance(dp, int) and not isinstance(dp, bool) and dp > 0, f"{row_id}: dp must be a positive integer"
        )
        visible = manifest.get("visible_devices")
        validation.require(
            isinstance(visible, list)
            and bool(visible)
            and all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in visible)
            and len(visible) == len(set(visible)),
            f"{row_id}: visible_devices must be a non-empty unique integer list",
        )
        subcases = manifest.get("context_subcases")
        validation.require(isinstance(subcases, list), f"{row_id}: context_subcases must be a list")
        if isinstance(subcases, list):
            kinds = [s.get("kind") for s in subcases if isinstance(s, dict)]
            required = {"long_prefill", "chunked_prefill"} | (
                {"cached_prefill"} if manifest.get("prefix_caching") else set()
            )
            validation.require(
                len(kinds) == len(subcases) and len(kinds) == len(set(kinds)),
                f"{row_id}: context subcase kinds must be unique",
            )
            validation.require(set(kinds) == required, f"{row_id}: context subcases must be exactly {sorted(required)}")
            for index, subcase in enumerate(subcases):
                if not isinstance(subcase, dict):
                    validation.errors.append(f"{row_id}: context_subcases[{index}] must be an object")
                    continue
                kind = subcase.get("kind")
                required_subcase_fields = {"kind", "input_tokens", "output_tokens"}
                if kind == "chunked_prefill":
                    required_subcase_fields.add("expected_min_chunks")
                elif kind == "cached_prefill":
                    required_subcase_fields.update({"common_prefix_tokens", "expected_min_cache_hits"})
                validate_exact_fields(
                    subcase,
                    required_subcase_fields,
                    set(),
                    f"{row_id}: context_subcases[{index}]",
                    validation,
                )
                for key in ("input_tokens", "output_tokens"):
                    value = subcase.get(key)
                    validation.require(
                        isinstance(value, int) and not isinstance(value, bool) and value > 0,
                        f"{row_id}: {subcase.get('kind')}.{key} must be positive",
                    )
                if isinstance(subcase.get("input_tokens"), int) and isinstance(subcase.get("output_tokens"), int):
                    validation.require(
                        subcase["input_tokens"] + subcase["output_tokens"] <= manifest.get("max_model_len", 0),
                        f"{row_id}: {subcase.get('kind')} exceeds max_model_len",
                    )
                    if subcase.get("kind") == "long_prefill":
                        validation.equal(
                            subcase["input_tokens"] + subcase["output_tokens"],
                            manifest.get("max_model_len"),
                            f"{row_id}: long_prefill must exercise the exact max_model_len bucket",
                        )
                if subcase.get("kind") == "chunked_prefill":
                    validation.require(
                        isinstance(subcase.get("expected_min_chunks"), int)
                        and subcase.get("expected_min_chunks", 0) > 1,
                        f"{row_id}: chunked_prefill.expected_min_chunks must exceed one",
                    )
                if subcase.get("kind") == "cached_prefill":
                    validation.require(
                        isinstance(subcase.get("common_prefix_tokens"), int)
                        and subcase.get("common_prefix_tokens", 0) > 0,
                        f"{row_id}: cached_prefill.common_prefix_tokens must be positive",
                    )
                    validation.require(
                        isinstance(subcase.get("input_tokens"), int)
                        and subcase.get("common_prefix_tokens", 0) < subcase.get("input_tokens", 0),
                        f"{row_id}: cached_prefill common prefix must be shorter than input",
                    )
                    validation.require(
                        isinstance(subcase.get("expected_min_cache_hits"), int)
                        and subcase.get("expected_min_cache_hits", 0) > 0,
                        f"{row_id}: cached_prefill.expected_min_cache_hits must be positive",
                    )
    return validation


def load_json(path: Path, validation: Validation) -> Any | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        validation.errors.append(f"missing file: {path}")
    except (OSError, json.JSONDecodeError) as error:
        validation.errors.append(f"cannot read JSON {path}: {error}")
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_exit_code(path: Path, validation: Validation, row_id: str) -> int | None:
    try:
        return int(path.read_text().strip())
    except FileNotFoundError:
        validation.errors.append(f"{row_id}: missing exit-code file: {path.name}")
    except (OSError, ValueError) as error:
        validation.errors.append(f"{row_id}: invalid exit-code file {path.name}: {error}")
    return None


def scan_logs(paths: tuple[Path, ...], validation: Validation, row_id: str) -> None:
    for path in paths:
        try:
            text = path.read_text(errors="replace")
        except FileNotFoundError:
            validation.errors.append(f"{row_id}: missing log {path.name}")
            continue
        except OSError as error:
            validation.errors.append(f"{row_id}: cannot read {path.name}: {error}")
            continue
        lines = [
            line
            for line in text.splitlines()
            if not (
                ("| warning  |" in line and "hard error in a future release" in line)
                or (
                    " WARNING " in line
                    and "Encountered invalid prefix detokenization error" in line
                    and "resetting decode stream" in line
                )
            )
        ]
        filtered = "\n".join(lines)
        for pattern in FAILURE_PATTERNS:
            count = len(pattern.findall(filtered))
            if count:
                validation.errors.append(f"{row_id}: {path.name} matched forbidden {pattern.pattern!r} {count} time(s)")


def expected_tt_config(manifest: dict[str, Any]) -> str:
    config: dict[str, Any] = {
        "trace_mode": manifest["trace_mode"],
        "trace_region_size": manifest["trace_region_size"],
        "sample_on_device_mode": manifest["sample_on_device_mode"],
    }
    if manifest.get("fabric_config") is not None:
        config["fabric_config"] = manifest["fabric_config"]
    return json.dumps({"tt": config}, separators=(",", ":"))


def expected_server_argv(expectations: dict[str, Any], manifest: dict[str, Any], contract: dict[str, Any]) -> list[str]:
    execution = expectations["execution"]
    argv = [
        execution["python"],
        execution["server_script"],
        "--model",
        manifest["model"],
        "--revision",
        manifest["revision"],
        "--tokenizer-revision",
        manifest["tokenizer_revision"],
        "--host",
        contract["host"],
        "--port",
        str(contract["port"]),
        "--max-model-len",
        str(manifest["max_model_len"]),
        "--data-parallel-size",
        str(manifest["dp"]),
        "--max_num_seqs",
        str(manifest["max_num_seqs_per_rank"]),
        "--additional-config",
        expected_tt_config(manifest),
    ]
    argv.append("--async-scheduling" if manifest["async_scheduling"] else "--no-async-scheduling")
    argv.append("--enable-prefix-caching" if manifest["prefix_caching"] else "--no-enable-prefix-caching")
    return argv


def expected_server_env(expectations: dict[str, Any], manifest: dict[str, Any]) -> dict[str, str]:
    env = {
        "MESH_DEVICE": manifest["platform"],
        "HF_MODEL": manifest["model"],
        manifest["family_var"]: manifest["family_version"],
        "HF_HOME": expectations["hf_cache"]["hf_home"],
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if manifest.get("cache_root"):
        env["TT_CACHE_PATH"] = manifest["cache_root"]
    env["TT_VISIBLE_DEVICES"] = ",".join(str(value) for value in manifest["visible_devices"])
    return env


def expected_benchmark_argv(expectations: dict[str, Any], contract: dict[str, Any], case_dir: Path) -> list[str]:
    common = expectations["common"]
    return [
        expectations["execution"]["python"],
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        common["backend"],
        "--endpoint",
        common["endpoint"],
        "--model",
        expectations["model"],
        "--tokenizer",
        expectations["hf_cache"]["snapshot"],
        "--host",
        contract["host"],
        "--port",
        str(contract["port"]),
        "--dataset-name",
        "random",
        "--random-input-len",
        str(common["input_tokens"]),
        "--random-output-len",
        str(common["output_tokens"]),
        "--num-prompts",
        str(common["num_prompts"]),
        "--max-concurrency",
        str(common["max_concurrency"]),
        "--request-rate",
        str(common["request_rate"]),
        "--temperature",
        str(common["temperature"]),
        "--ignore-eos",
        "--save-result",
        "--save-detailed",
        "--result-filename",
        str(case_dir / "result.json"),
    ]


def validate_root_contract(
    root: Path,
    expectations_path: Path,
    expectations: dict[str, Any],
    tier: str,
    subset_evidence: bool,
    validation: Validation,
) -> dict[str, Any] | None:
    contract = load_json(root / "run_contract.json", validation)
    if not isinstance(contract, dict):
        validation.errors.append("run_contract.json must contain an object")
        return None
    validate_exact_fields(
        contract,
        {
            "schema_version",
            "tier",
            "model_id",
            "architecture",
            "generator",
            "canonical_expectations_sha256",
            "canonical_row_ids",
            "selected_row_ids",
            "acceptance_scope",
            "host",
            "port",
            "tt_device_recovery_mode",
            "base_env",
            "tested_code_sha",
            "vllm_sha",
            "repositories",
            "tools",
            "inputs",
            "resolved_cache_roots",
            "launch_sha256_by_row",
        },
        {"aggregate_sources", "row_sources"},
        "run_contract",
        validation,
    )
    canonical = expectations.get("canonical_row_ids")
    rows = expectations.get("rows")
    actual_ids = [row.get("id") if isinstance(row, dict) else None for row in rows] if isinstance(rows, list) else None
    validation.equal(expectations.get("schema_version"), 2, "expectations schema_version")
    validation.require(isinstance(canonical, list) and bool(canonical), "canonical_row_ids must be a non-empty list")
    validation.equal(actual_ids, canonical, "canonical_row_ids must exactly match rows order")
    validation.equal(contract.get("schema_version"), 2, "run_contract.schema_version")
    validation.equal(contract.get("tier"), tier, "run_contract.tier")
    validation.equal(contract.get("model_id"), expectations.get("model_id"), "run_contract.model_id")
    validation.equal(contract.get("architecture"), expectations.get("architecture"), "run_contract.architecture")
    validation.equal(contract.get("generator"), expectations.get("generator"), "run_contract.generator")
    validation.equal(
        contract.get("canonical_expectations_sha256"),
        sha256_file(expectations_path),
        "run_contract canonical expectations hash",
    )
    persisted_expectations = root / "canonical_expectations.json"
    try:
        validation.equal(
            sha256_file(persisted_expectations), sha256_file(expectations_path), "persisted canonical expectations hash"
        )
    except FileNotFoundError:
        validation.errors.append(f"missing file: {persisted_expectations}")
    expected_scope = "subset" if subset_evidence else ("smoke" if tier == "smoke" else "complete")
    validation.equal(contract.get("acceptance_scope"), expected_scope, "run_contract.acceptance_scope")
    if subset_evidence:
        selected = contract.get("selected_row_ids")
        validation.require(isinstance(selected, list) and bool(selected), "subset selected_row_ids must be non-empty")
        if isinstance(selected, list):
            validation.require(
                selected == [row for row in canonical if row in set(selected)],
                "subset selected_row_ids must be a canonical ordered subset",
            )
            validation.require(
                len(selected) < len(canonical), "subset evidence cannot contain the full canonical row set"
            )
            expected_selected = selected
        else:
            expected_selected = []
    else:
        expected_selected = [expectations["smoke"]["row_id"]] if tier == "smoke" else canonical
    validation.equal(contract.get("selected_row_ids"), expected_selected, "run_contract.selected_row_ids")
    validation.equal(contract.get("canonical_row_ids"), canonical, "run_contract.canonical_row_ids")
    validation.equal(
        contract.get("tt_device_recovery_mode"),
        "reset",
        "run_contract.tt_device_recovery_mode (acceptance requires reset)",
    )
    tools = contract.get("tools")
    repositories = contract.get("repositories")
    inputs = contract.get("inputs")
    validation.require(
        isinstance(tools, dict) and set(tools) == {"runner", "validator"},
        "run_contract.tools must exactly cover runner/validator",
    )
    if isinstance(tools, dict):
        expected_validator = expectations["execution"]["validator"]
        validation.equal(tools.get("validator", {}).get("path"), expected_validator["path"], "canonical validator path")
        validation.equal(
            tools.get("validator", {}).get("sha256"), expected_validator["sha256"], "canonical validator hash"
        )
        for name in ("runner", "validator"):
            item = tools.get(name)
            validate_exact_fields(item, {"path", "sha256"}, set(), f"run_contract.tools.{name}", validation)
            validation.require(
                isinstance(item, dict) and isinstance(item.get("path"), str) and is_sha256(item.get("sha256")),
                f"run_contract.tools.{name} is invalid",
            )
            if isinstance(item, dict) and isinstance(item.get("path"), str) and is_sha256(item.get("sha256")):
                try:
                    validation.equal(sha256_file(Path(item["path"])), item["sha256"], f"live {name} hash")
                except OSError as error:
                    validation.errors.append(f"cannot hash live {name}: {error}")
    validation.require(
        isinstance(repositories, dict) and set(repositories) == {"tt_metal", "vllm"},
        "run_contract.repositories must exactly cover tt_metal/vllm",
    )
    if isinstance(repositories, dict):
        for name in ("tt_metal", "vllm"):
            item = repositories.get(name)
            validate_exact_fields(
                item,
                {"path", "head", "dirty", "tracked_status"},
                set(),
                f"run_contract.repositories.{name}",
                validation,
            )
            if isinstance(item, dict):
                validation.require(
                    isinstance(item.get("path"), str) and Path(item.get("path", "x")).is_absolute(),
                    f"{name} repository path must be absolute",
                )
                validation.require(
                    isinstance(item.get("head"), str)
                    and re.fullmatch(r"[0-9a-f]{40}", item.get("head", "")) is not None,
                    f"{name} HEAD must be a full SHA",
                )
                validation.equal(item.get("dirty"), False, f"{name} repository must be tracked-clean")
                validation.equal(item.get("tracked_status"), [], f"{name} repository tracked status")
                if isinstance(item.get("path"), str):
                    try:
                        live_head = subprocess.check_output(
                            ["git", "-C", item["path"], "rev-parse", "HEAD"], text=True
                        ).strip()
                        live_status = subprocess.check_output(
                            ["git", "-C", item["path"], "status", "--porcelain", "--untracked-files=no"], text=True
                        )
                        validation.equal(live_head, item.get("head"), f"live {name} HEAD")
                        validation.equal(bool(live_status), item.get("dirty"), f"live {name} tracked dirty state")
                    except (OSError, subprocess.CalledProcessError) as error:
                        validation.errors.append(f"cannot verify live {name} repository: {error}")
        validation.equal(
            contract.get("tested_code_sha"), repositories.get("tt_metal", {}).get("head"), "tested_code_sha"
        )
        validation.equal(contract.get("vllm_sha"), repositories.get("vllm", {}).get("head"), "vllm_sha")
    validation.equal(inputs, expectations.get("provenance"), "run_contract input provenance")
    base_env = contract.get("base_env")
    allowed_base_env = {"PATH", "HOME", "LD_LIBRARY_PATH", "TT_METAL_HOME", "PYTHONNOUSERSITE"}
    validation.require(
        isinstance(base_env, dict)
        and all(isinstance(key, str) and isinstance(value, str) for key, value in base_env.items())
        and set(base_env) <= allowed_base_env,
        "run_contract.base_env must be a string map from the exact ambient allowlist",
    )
    if isinstance(base_env, dict):
        validation.equal(base_env.get("PYTHONNOUSERSITE"), "1", "run_contract.base_env.PYTHONNOUSERSITE")
    launch_hashes = contract.get("launch_sha256_by_row")
    validation.require(
        isinstance(launch_hashes, dict) and set(launch_hashes) == set(expected_selected),
        "launch hash map must exactly cover selected rows",
    )
    expected_cache_roots = {row["id"]: row["manifest"]["cache_root"] for row in rows if row["id"] in expected_selected}
    validation.equal(contract.get("resolved_cache_roots"), expected_cache_roots, "resolved cache-root map")
    for row_id, cache_root in expected_cache_roots.items():
        try:
            validation.equal(
                str(Path(cache_root).resolve(strict=True)), cache_root, f"{row_id}: live resolved cache root"
            )
            validation.require(
                Path(cache_root).is_dir() and os.access(cache_root, os.W_OK | os.X_OK),
                f"{row_id}: live cache root is not writable/searchable",
            )
        except OSError as error:
            validation.errors.append(f"{row_id}: cannot resolve live cache root: {error}")
    row_sources = contract.get("row_sources")
    if row_sources is not None:
        validation.require(
            isinstance(row_sources, dict) and set(row_sources) == set(expected_selected),
            "run_contract.row_sources must exactly cover selected rows",
        )
    validation.require(
        isinstance(contract.get("host"), str) and bool(contract.get("host")), "run_contract.host is missing"
    )
    validation.require(
        isinstance(contract.get("port"), int) and 0 < contract.get("port", 0) < 65536, "run_contract.port is invalid"
    )
    return contract


def validate_manifest(case_dir: Path, row: dict[str, Any], tier: str, validation: Validation) -> dict[str, Any] | None:
    manifest = load_json(case_dir / "manifest.json", validation)
    if not isinstance(manifest, dict):
        validation.errors.append(f"{row['id']}: manifest.json must contain an object")
        return None
    validate_exact_fields(
        manifest,
        set(row["manifest"]) | {"tier", "row_id", "status", "error_hits"},
        set(),
        f"{row['id']}: manifest.json",
        validation,
    )
    for key, value in row["manifest"].items():
        validation.equal(manifest.get(key), value, f"{row['id']}: manifest.{key}")
    validation.equal(manifest.get("row_id"), row["id"], f"{row['id']}: manifest.row_id")
    validation.equal(manifest.get("tier"), tier, f"{row['id']}: manifest.tier")
    validation.equal(manifest.get("status"), "ok", f"{row['id']}: manifest.status")
    validation.equal(manifest.get("error_hits"), 0, f"{row['id']}: manifest.error_hits")
    return manifest


def validate_launch(
    case_dir: Path,
    expectations: dict[str, Any],
    row: dict[str, Any],
    tier: str,
    contract: dict[str, Any],
    validation: Validation,
) -> None:
    row_id = row["id"]
    launch = load_json(case_dir / "launch.json", validation)
    if not isinstance(launch, dict):
        validation.errors.append(f"{row_id}: launch.json must contain an object")
        return
    required_launch_fields = {"schema_version", "server", "client"}
    if tier == "benchmark":
        required_launch_fields.add("context_clients")
    validate_exact_fields(launch, required_launch_fields, set(), f"{row_id}: launch", validation)
    validation.equal(launch.get("schema_version"), 1, f"{row_id}: launch schema_version")
    try:
        validation.equal(
            sha256_file(case_dir / "launch.json"),
            contract["launch_sha256_by_row"][row_id],
            f"{row_id}: immutable launch hash",
        )
    except (KeyError, OSError) as error:
        validation.errors.append(f"{row_id}: cannot verify launch hash: {error}")
    server = launch.get("server")
    validate_process_spec(server, f"{row_id}: launch.server", validation)
    validation.require(
        bool(row["manifest"].get("cache_root")),
        f"{row_id}: manifest.cache_root is required (per-row TT cache isolation)",
    )
    if isinstance(server, dict):
        validation.equal(server.get("cwd"), expectations["execution"]["vllm_dir"], f"{row_id}: server cwd")
        validation.equal(
            server.get("argv"), expected_server_argv(expectations, row["manifest"], contract), f"{row_id}: server argv"
        )
        expected_env = dict(contract.get("base_env", {}))
        expected_env.update(expected_server_env(expectations, row["manifest"]))
        validation.equal(server.get("env"), expected_env, f"{row_id}: server env")
    client = launch.get("client")
    validation.require(isinstance(client, dict), f"{row_id}: launch.client must be an object")
    if isinstance(client, dict) and tier == "benchmark":
        validate_process_spec(client, f"{row_id}: launch.client", validation)
        provenance_case_dir = case_dir
        row_sources = contract.get("row_sources")
        if isinstance(row_sources, dict) and isinstance(row_sources.get(row_id), str):
            provenance_case_dir = Path(row_sources[row_id]) / row_id
        validation.equal(client.get("cwd"), expectations["execution"]["vllm_dir"], f"{row_id}: client cwd")
        validation.equal(
            client.get("argv"),
            expected_benchmark_argv(expectations, contract, provenance_case_dir),
            f"{row_id}: client argv",
        )
        expected_client_env = dict(contract.get("base_env", {}))
        expected_client_env.update(
            {
                "HF_HOME": expectations["hf_cache"]["hf_home"],
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        validation.equal(client.get("env"), expected_client_env, f"{row_id}: client env")
        context_clients = launch.get("context_clients")
        subcases = row["manifest"]["context_subcases"]
        validation.require(
            isinstance(context_clients, dict) and set(context_clients) == {item["kind"] for item in subcases},
            f"{row_id}: context client set",
        )
        if isinstance(context_clients, dict):
            for item in subcases:
                spec = context_clients.get(item["kind"])
                validate_process_spec(spec, f"{row_id}: context client {item['kind']}", validation)
                if isinstance(spec, dict):
                    validation.equal(
                        spec.get("cwd"), expectations["execution"]["vllm_dir"], f"{row_id}: {item['kind']} cwd"
                    )
                    validation.equal(spec.get("env"), expected_client_env, f"{row_id}: {item['kind']} env")
                    argv = spec.get("argv")
                    validation.require(
                        isinstance(argv, list)
                        and len(argv) == 8
                        and argv[:2] == [expectations["execution"]["python"], "-c"],
                        f"{row_id}: {item['kind']} client argv shape",
                    )
                    if isinstance(argv, list) and len(argv) == 8:
                        validation.equal(
                            argv[2], CONTEXT_CLIENT_PROGRAM, f"{row_id}: {item['kind']} exact context client program"
                        )
                        try:
                            validation.equal(json.loads(argv[3]), item, f"{row_id}: {item['kind']} frozen subcase")
                        except json.JSONDecodeError as error:
                            validation.errors.append(f"{row_id}: {item['kind']} invalid frozen subcase: {error}")
                        validation.equal(
                            argv[4:7],
                            [contract["host"], str(contract["port"]), row["manifest"]["model"]],
                            f"{row_id}: {item['kind']} endpoint/model argv",
                        )
                        context_case_dir = case_dir
                        row_sources = contract.get("row_sources")
                        if isinstance(row_sources, dict) and isinstance(row_sources.get(row_id), str):
                            context_case_dir = Path(row_sources[row_id]) / row_id
                        validation.equal(
                            argv[7],
                            str(context_case_dir / "context_subcases" / item["kind"] / "result.json"),
                            f"{row_id}: {item['kind']} result path",
                        )
    elif isinstance(client, dict):
        validation.equal(
            client,
            {
                "kind": "http",
                "method": "POST",
                "url": f"http://{contract['host']}:{contract['port']}/v1/completions",
                "request_file": "request.json",
                "response_file": "response.json",
            },
            f"{row_id}: HTTP client provenance",
        )

    live = load_json(case_dir / "live_process.json", validation)
    if not isinstance(live, dict) or not isinstance(server, dict):
        validation.errors.append(f"{row_id}: live_process.json must contain an object")
        return
    validate_exact_fields(
        live,
        {"pid", "pgid", "cwd", "executable", "argv", "env"},
        set(),
        f"{row_id}: live_process",
        validation,
    )
    validation.equal(live.get("pid"), int((case_dir / "server.pid").read_text().strip()), f"{row_id}: live pid")
    validation.equal(live.get("pgid"), int((case_dir / "server.pgid").read_text().strip()), f"{row_id}: live pgid")
    validation.equal(live.get("cwd"), server.get("cwd"), f"{row_id}: live cwd")
    validation.equal(live.get("argv"), server.get("argv"), f"{row_id}: live argv")
    validation.equal(live.get("env"), server.get("env"), f"{row_id}: live env")
    validation.require(
        isinstance(live.get("executable"), str) and Path(live["executable"]).is_absolute(),
        f"{row_id}: live executable must be absolute",
    )
    try:
        validation.equal(
            live.get("executable"), str(Path(server["argv"][0]).resolve(strict=True)), f"{row_id}: live executable"
        )
    except (KeyError, IndexError, OSError, TypeError) as error:
        validation.errors.append(f"{row_id}: cannot resolve server executable: {error}")


def validate_process_proof(path: Path, spec: dict[str, Any], validation: Validation, label: str) -> None:
    proof = load_json(path, validation)
    if not isinstance(proof, dict):
        validation.errors.append(f"{label}: process proof must be an object")
        return
    validate_exact_fields(
        proof,
        {"pid", "pgid", "cwd", "executable", "argv", "env"},
        set(),
        f"{label}: process proof",
        validation,
    )
    validation.equal(proof.get("cwd"), spec.get("cwd"), f"{label}: live cwd")
    validation.equal(proof.get("argv"), spec.get("argv"), f"{label}: live argv")
    validation.equal(proof.get("env"), spec.get("env"), f"{label}: live env")
    validation.require(isinstance(proof.get("pid"), int) and proof["pid"] > 0, f"{label}: live pid")
    validation.require(isinstance(proof.get("pgid"), int) and proof["pgid"] > 0, f"{label}: live pgid")
    validation.require(
        isinstance(proof.get("executable"), str) and Path(proof["executable"]).is_absolute(),
        f"{label}: live executable",
    )
    try:
        validation.equal(
            proof.get("executable"), str(Path(spec["argv"][0]).resolve(strict=True)), f"{label}: live executable"
        )
    except (KeyError, IndexError, OSError, TypeError) as error:
        validation.errors.append(f"{label}: cannot resolve executable: {error}")


def validate_context_subcases(
    case_dir: Path, row: dict[str, Any], launch: dict[str, Any], validation: Validation
) -> None:
    row_id = row["id"]
    try:
        server_bytes = (case_dir / "server.log").read_bytes()
    except OSError as error:
        validation.errors.append(f"{row_id}: cannot read server log for context evidence: {error}")
        return
    for subcase in row["manifest"]["context_subcases"]:
        kind = subcase["kind"]
        subdir = case_dir / "context_subcases" / kind
        code = read_exit_code(subdir / "client.exit", validation, f"{row_id}/{kind}")
        if code is not None:
            validation.equal(code, 0, f"{row_id}/{kind}: client.exit")
        spec = launch.get("context_clients", {}).get(kind)
        if isinstance(spec, dict):
            validate_process_proof(subdir / "live_process.json", spec, validation, f"{row_id}/{kind}")
        evidence = load_json(subdir / "evidence.json", validation)
        if not isinstance(evidence, dict):
            validation.errors.append(f"{row_id}/{kind}: evidence must be an object")
            continue
        validate_exact_fields(
            evidence,
            {
                "schema_version",
                "server_log_start",
                "server_log_end",
                "server_segment_sha256",
                "instrumentation_counts",
                "result",
            },
            set(),
            f"{row_id}/{kind}: evidence",
            validation,
        )
        validation.equal(evidence.get("schema_version"), 1, f"{row_id}/{kind}: evidence schema_version")
        start, end = evidence.get("server_log_start"), evidence.get("server_log_end")
        validation.require(
            isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(server_bytes),
            f"{row_id}/{kind}: invalid server-log byte interval",
        )
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= len(server_bytes)):
            continue
        segment = server_bytes[start:end]
        validation.equal(
            evidence.get("server_segment_sha256"),
            hashlib.sha256(segment).hexdigest(),
            f"{row_id}/{kind}: server segment hash",
        )
        actual_counts = {
            "chunk_events": len(re.findall(rb"(?:chunked[^\n]*prefill|prefill[^\n]*chunk)", segment, re.I)),
            "cache_hits": len(re.findall(rb"(?:cache[^\n]*hit|prefix[^\n]*cache[^\n]*hit)", segment, re.I)),
        }
        validate_exact_fields(
            evidence.get("instrumentation_counts"),
            {"chunk_events", "cache_hits"},
            set(),
            f"{row_id}/{kind}: instrumentation_counts",
            validation,
        )
        validation.equal(
            evidence.get("instrumentation_counts"), actual_counts, f"{row_id}/{kind}: instrumentation counts"
        )
        if kind == "chunked_prefill":
            validation.require(
                actual_counts["chunk_events"] >= subcase["expected_min_chunks"],
                f"{row_id}/{kind}: insufficient chunk instrumentation",
            )
        if kind == "cached_prefill":
            validation.require(
                actual_counts["cache_hits"] >= subcase["expected_min_cache_hits"],
                f"{row_id}/{kind}: insufficient cache-hit instrumentation",
            )
        result = evidence.get("result")
        validation.require(
            isinstance(result, dict) and result.get("subcase") == subcase and isinstance(result.get("calls"), list),
            f"{row_id}/{kind}: malformed result",
        )
        if not isinstance(result, dict) or not isinstance(result.get("calls"), list):
            continue
        validate_exact_fields(
            result,
            {"schema_version", "subcase", "calls"},
            set(),
            f"{row_id}/{kind}: result",
            validation,
        )
        validation.equal(result.get("schema_version"), 1, f"{row_id}/{kind}: result schema_version")
        expected_calls = 2 if kind == "cached_prefill" else 1
        validation.equal(len(result["calls"]), expected_calls, f"{row_id}/{kind}: call count")
        for index, call in enumerate(result["calls"]):
            validate_exact_fields(
                call,
                {"request", "response"},
                set(),
                f"{row_id}/{kind}: call {index}",
                validation,
            )
            request = call.get("request") if isinstance(call, dict) else None
            response = call.get("response") if isinstance(call, dict) else None
            validation.require(
                isinstance(request, dict)
                and isinstance(request.get("prompt"), list)
                and len(request["prompt"]) == subcase["input_tokens"],
                f"{row_id}/{kind}: call {index} input-token count",
            )
            if isinstance(request, dict):
                validate_exact_fields(
                    request,
                    {"model", "prompt", "max_tokens", "temperature", "ignore_eos"},
                    set(),
                    f"{row_id}/{kind}: call {index} request",
                    validation,
                )
                validation.equal(request.get("model"), row["manifest"]["model"], f"{row_id}/{kind}: call {index} model")
                expected_prompt = [42] * subcase["input_tokens"]
                if kind == "cached_prefill":
                    common = subcase["common_prefix_tokens"]
                    expected_prompt = [42] * common + [43 + index] * (subcase["input_tokens"] - common)
                validation.equal(
                    request.get("prompt"), expected_prompt, f"{row_id}/{kind}: call {index} exact prompt tokens"
                )
                validation.equal(
                    request.get("max_tokens"), subcase["output_tokens"], f"{row_id}/{kind}: call {index} max_tokens"
                )
                validation.equal(request.get("temperature"), 0, f"{row_id}/{kind}: call {index} temperature")
                validation.equal(request.get("ignore_eos"), True, f"{row_id}/{kind}: call {index} ignore_eos")
            validation.require(isinstance(response, dict), f"{row_id}/{kind}: call {index} response")
            if isinstance(response, dict):
                validation.equal(
                    response.get("usage", {}).get("prompt_tokens"),
                    subcase["input_tokens"],
                    f"{row_id}/{kind}: call {index} server prompt-token count",
                )
                validation.equal(
                    response.get("usage", {}).get("completion_tokens"),
                    subcase["output_tokens"],
                    f"{row_id}/{kind}: call {index} output-token count",
                )
                try:
                    validation.equal(
                        response["choices"][0]["finish_reason"],
                        "length",
                        f"{row_id}/{kind}: call {index} finish_reason",
                    )
                except (KeyError, IndexError, TypeError):
                    validation.errors.append(f"{row_id}/{kind}: call {index} malformed completion")


def validate_lifecycle(case_dir: Path, tier: str, validation: Validation, row_id: str) -> None:
    for filename in ("reset_before.exit", "cleanup.exit", "reset_after.exit", "client.exit"):
        code = read_exit_code(case_dir / filename, validation, row_id)
        if code is not None:
            validation.equal(code, 0, f"{row_id}: {filename}")
    try:
        cleanup_log = (case_dir / "cleanup.log").read_text(errors="replace")
        validation.require("cleanup_status=ok" in cleanup_log, f"{row_id}: cleanup.log lacks cleanup_status=ok")
    except FileNotFoundError:
        validation.errors.append(f"{row_id}: missing file: cleanup.log")
    try:
        validation.require(
            not (case_dir / "process_check_after.log").read_text().strip(),
            f"{row_id}: post-row process check is not empty",
        )
    except FileNotFoundError:
        validation.errors.append(f"{row_id}: missing file: process_check_after.log")


def validate_evidence(
    case_dir: Path, row: dict[str, Any], tier: str, program_log_contract: Any, validation: Validation
) -> None:
    row_id = row["id"]
    evidence = load_json(case_dir / "evidence.json", validation)
    if not isinstance(evidence, dict):
        validation.errors.append(f"{row_id}: evidence.json must contain an object")
        return
    validate_exact_fields(
        evidence,
        {"trace_counts", "program_counts", "trace_region_config_hits", "metrics"},
        set(),
        f"{row_id}: evidence.json",
        validation,
    )
    try:
        server = (case_dir / "server.log").read_text(errors="replace")
    except OSError as error:
        validation.errors.append(f"{row_id}: cannot read server.log for evidence: {error}")
        return
    actual_traces = {name: len(pattern.findall(server)) for name, pattern in TRACE_PATTERNS.items()}
    validation.equal(evidence.get("trace_counts"), actual_traces, f"{row_id}: evidence trace_counts")
    validation.equal(actual_traces, row["expected_traces"], f"{row_id}: expected trace counts")
    programs = evidence.get("program_counts")
    expected_programs = {name: len(pattern.findall(server)) for name, pattern in PROGRAM_PATTERNS.items()}
    validation.equal(programs, expected_programs, f"{row_id}: evidence program_counts")
    log_contract = row.get("expected_program_logs")
    if log_contract is None:
        global_contract = row.get("manifest", {}).get("dp")
        if isinstance(program_log_contract, dict) and isinstance(global_contract, int):
            log_contract = {
                "decode_compiles": program_log_contract.get("decode_compiles_per_lane", 0) * global_contract,
                "sampling_compiles": program_log_contract.get("sampling_compiles_per_lane", 0) * global_contract,
            }
    validation.require(isinstance(log_contract, dict), f"{row_id}: expected program-log contract is absent")
    if isinstance(log_contract, dict):
        validation.equal(expected_programs, log_contract, f"{row_id}: expected program-log counts")
        validation.require(
            all(isinstance(value, int) and value > 0 for value in log_contract.values()),
            f"{row_id}: expected program-log counts must be nonzero integers",
        )
    region_pattern = re.compile(
        rf"(?:['\"])?trace_region_size(?:['\"])?\s*[:=]\s*{row['manifest']['trace_region_size']}\b"
    )
    region_hits = len(region_pattern.findall(server))
    validation.equal(evidence.get("trace_region_config_hits"), region_hits, f"{row_id}: trace-region evidence")
    validation.require(region_hits > 0, f"{row_id}: exact trace_region_size is absent from server.log")
    metrics = evidence.get("metrics")
    validation.require(isinstance(metrics, dict), f"{row_id}: evidence.metrics must be an object")
    expected_metric_fields = BENCHMARK_METRICS if tier == "benchmark" else set()
    validate_exact_fields(metrics, expected_metric_fields, set(), f"{row_id}: evidence.metrics", validation)
    if tier == "benchmark" and isinstance(metrics, dict):
        result = load_json(case_dir / "result.json", validation)
        if isinstance(result, dict):
            for key in (
                "request_throughput",
                "output_throughput",
                "total_token_throughput",
                "median_ttft_ms",
                "p99_ttft_ms",
                "median_tpot_ms",
                "p99_tpot_ms",
            ):
                validation.equal(metrics.get(key), result.get(key), f"{row_id}: evidence.metrics.{key}")
                validation.require(
                    isinstance(metrics.get(key), (int, float)) and math.isfinite(metrics[key]),
                    f"{row_id}: metric {key} is not finite",
                )


def validate_benchmark(
    case_dir: Path,
    expectations: dict[str, Any],
    row: dict[str, Any],
    validation: Validation,
) -> None:
    row_id = row["id"]
    result = load_json(case_dir / "result.json", validation)
    if not isinstance(result, dict):
        validation.errors.append(f"{row_id}: result.json must contain an object")
        return
    common = expectations["common"]
    count = common["num_prompts"]
    validation.equal(result.get("backend"), common["backend"], f"{row_id}: backend")
    validation.equal(result.get("model_id"), expectations["model"], f"{row_id}: model_id")
    validation.equal(result.get("num_prompts"), count, f"{row_id}: num_prompts")
    validation.equal(result.get("max_concurrency"), common["max_concurrency"], f"{row_id}: max_concurrency")
    validation.equal(str(result.get("request_rate")), str(common["request_rate"]), f"{row_id}: request_rate")
    validation.equal(result.get("completed"), count, f"{row_id}: completed")
    validation.equal(result.get("failed"), 0, f"{row_id}: failed")
    validation.equal(
        result.get("total_output_tokens"), count * common["output_tokens"], f"{row_id}: total_output_tokens"
    )
    errors = result.get("errors")
    validation.require(
        isinstance(errors, list) and len(errors) == count and all(value in ("", None) for value in errors),
        f"{row_id}: errors must contain {count} empty entries",
    )
    output_lens = result.get("output_lens")
    validation.require(
        isinstance(output_lens, list)
        and len(output_lens) == count
        and all(value == common["output_tokens"] for value in output_lens),
        f"{row_id}: output_lens must contain {count} copies of {common['output_tokens']}",
    )
    input_lens = result.get("input_lens")
    validation.require(
        isinstance(input_lens, list)
        and len(input_lens) == count
        and all(isinstance(value, int) and 0 < value <= common["input_tokens"] for value in input_lens),
        f"{row_id}: input_lens must contain {count} sensible lengths",
    )
    for key in ("generated_texts", "start_times", "ttfts", "itls"):
        value = result.get(key)
        validation.require(
            isinstance(value, list) and len(value) == count, f"{row_id}: {key} must contain {count} entries"
        )
    if (
        isinstance(result.get("itls"), list)
        and isinstance(output_lens, list)
        and len(result["itls"]) == len(output_lens)
    ):
        # vLLM records one ITL per streamed response chunk, and a chunk may
        # contain multiple output tokens. TPOT is derived independently from
        # request latency, TTFT, and output-token count. Therefore ITL
        # cardinality is bounded by, but need not equal, token cardinality.
        validation.require(
            all(
                isinstance(items, list)
                and isinstance(length, int)
                and not isinstance(length, bool)
                and length >= 0
                and len(items) <= length
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(value)
                    and value >= 0
                    for value in items
                )
                for items, length in zip(result["itls"], output_lens)
            ),
            f"{row_id}: itls must be finite non-negative streamed-event intervals bounded by output_lens",
        )
    thresholds = row["performance_thresholds"]
    for metric, threshold in thresholds.items():
        actual = result.get(metric)
        validation.require(
            isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual),
            f"{row_id}: performance metric {metric} must be finite numeric",
        )
        if not isinstance(actual, (int, float)) or isinstance(actual, bool) or not math.isfinite(actual):
            continue
        limit = threshold["value"]
        if threshold["direction"] == "minimum":
            validation.require(actual >= limit, f"{row_id}: {metric} {actual} is below required minimum {limit}")
        else:
            validation.require(actual <= limit, f"{row_id}: {metric} {actual} exceeds required maximum {limit}")


def validate_quality(
    case_dir: Path, expectations: dict[str, Any], tier: str, validation: Validation, row_id: str
) -> None:
    budget = expectations["smoke"]["tokens"] if tier == "smoke" else expectations["quality"]["quality_tokens"]
    request = load_json(case_dir / "request.json", validation)
    expected_request = {
        "model": expectations["model"],
        "prompt": expectations["execution"]["prompt"],
        "max_tokens": budget,
        "temperature": 0,
        "ignore_eos": True,
    }
    validation.equal(request, expected_request, f"{row_id}: request provenance")
    response = load_json(case_dir / "response.json", validation)
    if not isinstance(response, dict):
        validation.errors.append(f"{row_id}: response.json must contain an object")
        return
    validation.equal(response.get("model"), expectations["model"], f"{row_id}: response model")
    validation.equal(response.get("object"), "text_completion", f"{row_id}: response object")
    try:
        choice = response["choices"][0]
        text = choice["text"]
        finish_reason = choice["finish_reason"]
        completion_tokens = response["usage"]["completion_tokens"]
    except (KeyError, IndexError, TypeError) as error:
        validation.errors.append(f"{row_id}: malformed completion response: {error}")
        return
    validation.require(isinstance(text, str) and bool(text.strip()), f"{row_id}: completion text is empty")
    validation.equal(finish_reason, "length", f"{row_id}: finish_reason")
    validation.equal(completion_tokens, budget, f"{row_id}: completion_tokens")
    if isinstance(text, str):
        lowered = text.lower()
        semantic_groups = expectations.get(tier, {}).get(
            "semantic_term_groups",
            expectations["quality"].get("semantic_term_groups", ()),
        )
        for index, alternatives in enumerate(semantic_groups):
            validation.require(
                any(term.lower() in lowered for term in alternatives),
                f"{row_id}: semantic term group {index} absent ({alternatives!r})",
            )


def validate_pair_reviews(
    root: Path, expectations: dict[str, Any], rows: list[dict[str, Any]], validation: Validation
) -> None:
    if not expectations["quality"].get("require_pair_review", False):
        return
    review = load_json(root / expectations["quality"].get("pair_review_file", "quality_review.json"), validation)
    if not isinstance(review, dict) or not isinstance(review.get("pairs"), dict):
        validation.errors.append("quality pair-review file must contain a pairs object")
        return
    validate_exact_fields(review, {"pairs"}, set(), "quality pair-review", validation)
    pairs: dict[str, dict[str, str]] = {}
    for row in rows:
        manifest = row["manifest"]
        pair = f"{manifest['platform'].lower()}_dp{manifest['dp']}"
        pairs.setdefault(pair, {})[manifest["trace_mode"]] = row["id"]
    expected_pairs = {pair: modes for pair, modes in pairs.items() if set(modes) == {"decode_only", "all"}}
    validation.equal(set(review["pairs"]), set(expected_pairs), "quality pair-review set")
    for pair, modes in expected_pairs.items():
        record = review["pairs"].get(pair)
        validation.require(isinstance(record, dict), f"quality pair {pair}: missing review object")
        if not isinstance(record, dict):
            continue
        validate_exact_fields(
            record,
            {"rows", "response_sha256", "accepted", "reviewer", "note"},
            set(),
            f"quality pair {pair}",
            validation,
        )
        validation.equal(record.get("rows"), modes, f"quality pair {pair}: exact row ids")
        expected_hashes = {mode: sha256_file(root / row_id / "response.json") for mode, row_id in modes.items()}
        validation.equal(record.get("response_sha256"), expected_hashes, f"quality pair {pair}: response hashes")
        validation.require(type(record.get("accepted")) is bool, f"quality pair {pair}: accepted must be boolean")
        validation.equal(record.get("accepted"), True, f"quality pair {pair}: accepted")
        validation.require(
            isinstance(record.get("reviewer"), str) and bool(record["reviewer"].strip()),
            f"quality pair {pair}: reviewer must be a non-empty string",
        )
        validation.require(
            isinstance(record.get("note"), str) and bool(record["note"].strip()),
            f"quality pair {pair}: note must be a non-empty string",
        )
    for pair, modes in pairs.items():
        validation.require(
            set(modes) in ({"decode_only", "all"}, {"decode_only"}),
            f"quality pair {pair}: unsupported trace-mode set {set(modes)!r}",
        )


def validate_root(
    root: Path, expectations_path: Path, expectations: dict[str, Any], tier: str, subset_evidence: bool = False
) -> Validation:
    validation = Validation()
    contract = validate_root_contract(root, expectations_path, expectations, tier, subset_evidence, validation)
    rows_by_id = {row["id"]: row for row in expectations.get("rows", []) if isinstance(row, dict) and "id" in row}
    if subset_evidence and contract is not None and isinstance(contract.get("selected_row_ids"), list):
        expected_ids = contract["selected_row_ids"]
    else:
        expected_ids = (
            [expectations["smoke"]["row_id"]] if tier == "smoke" else expectations.get("canonical_row_ids", [])
        )
    rows = [rows_by_id[row_id] for row_id in expected_ids if row_id in rows_by_id]
    actual_dirs = {path.name for path in root.iterdir() if path.is_dir()} if root.is_dir() else set()
    validation.equal(actual_dirs, set(expected_ids), "artifact row directory set")
    journal_path = root / "attempt_journal.jsonl"
    try:
        entries = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
        journal_fields = {
            "run_started": {"event", "tier", "timestamp_utc"},
            "run_finished": {"event", "tier", "status", "timestamp_utc"},
            "row_started": {"event", "row_id", "tier", "timestamp_utc"},
            "row_finished": {"event", "row_id", "tier", "status", "error_hits", "timestamp_utc"},
        }
        for index, entry in enumerate(entries):
            event = entry.get("event") if isinstance(entry, dict) else None
            validation.require(event in journal_fields, f"attempt journal entry {index} has an unknown event")
            if event in journal_fields:
                validate_exact_fields(entry, journal_fields[event], set(), f"attempt journal entry {index}", validation)
        starts = [entry.get("row_id") for entry in entries if entry.get("event") == "row_started"]
        finishes = [
            entry.get("row_id")
            for entry in entries
            if entry.get("event") == "row_finished" and entry.get("status") == "ok"
        ]
        validation.equal(starts, expected_ids, "attempt journal row_started order")
        validation.equal(finishes, expected_ids, "attempt journal successful row_finished order")
        validation.require(
            all(isinstance(entry.get("timestamp_utc"), str) and entry["timestamp_utc"] for entry in entries),
            "attempt journal timestamps",
        )
    except FileNotFoundError:
        validation.errors.append(f"missing file: {journal_path}")
    except (OSError, json.JSONDecodeError) as error:
        validation.errors.append(f"invalid attempt journal: {error}")
    if tier == "smoke":
        smoke_row = rows_by_id.get(expectations.get("smoke", {}).get("row_id"))
        validation.require(smoke_row is not None, "smoke.row_id is not a canonical row")
        if smoke_row is not None:
            validation.equal(smoke_row["manifest"].get("trace_mode"), "decode_only", "smoke row trace_mode")
    for row in rows:
        row_id = row["id"]
        case_dir = root / row_id
        validate_manifest(case_dir, row, tier, validation)
        if contract is not None:
            validate_launch(case_dir, expectations, row, tier, contract, validation)
        validate_lifecycle(case_dir, tier, validation, row_id)
        context_logs = tuple(sorted((case_dir / "context_subcases").glob("*/client.log")))
        scan_logs((case_dir / "server.log", case_dir / "client.log", *context_logs), validation, row_id)
        validate_evidence(case_dir, row, tier, expectations.get("program_log_contract"), validation)
        if tier == "benchmark":
            launch = load_json(case_dir / "launch.json", validation)
            if isinstance(launch, dict):
                validate_context_subcases(case_dir, row, launch, validation)
                client = launch.get("client")
                if isinstance(client, dict):
                    validate_process_proof(
                        case_dir / "client_live_process.json", client, validation, f"{row_id}/benchmark-client"
                    )
            validate_benchmark(case_dir, expectations, row, validation)
        else:
            validate_quality(case_dir, expectations, tier, validation, row_id)
    try:
        validation.require(
            not (root / "process_check_after.log").read_text().strip(), "final process check is not empty"
        )
    except FileNotFoundError:
        validation.errors.append(f"missing file: {root / 'process_check_after.log'}")
    if tier == "quality" and not subset_evidence:
        validate_pair_reviews(root, expectations, rows, validation)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--expectations", required=True, type=Path)
    parser.add_argument("--tier", choices=("smoke", "benchmark", "quality"))
    parser.add_argument(
        "--subset-evidence", action="store_true", help="validate selected rows but never grant acceptance"
    )
    parser.add_argument(
        "--check-expectations",
        action="store_true",
        help="validate the complete expectations schema without touching hardware",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bootstrap = Validation()
    expectations = load_json(args.expectations, bootstrap)
    if bootstrap.errors or not isinstance(expectations, dict):
        for error in bootstrap.errors or ["expectations must contain an object"]:
            print(f"FAIL: {error}", file=sys.stderr)
        return 1
    schema_validation = validate_expectations_document(expectations, args.expectations)
    if schema_validation.errors:
        for error in schema_validation.errors:
            print(f"FAIL: {error}", file=sys.stderr)
        print(f"FAILED: {len(schema_validation.errors)} expectations violation(s)", file=sys.stderr)
        return 1
    if args.check_expectations:
        if args.artifact_root is not None or args.tier is not None or args.subset_evidence:
            print("FAIL: --check-expectations cannot be combined with artifact/tier/subset arguments", file=sys.stderr)
            return 1
        print(f"PASS: expectations schema ({len(expectations['canonical_row_ids'])} row(s))")
        return 0
    if args.artifact_root is None or args.tier is None:
        print("FAIL: --artifact-root and --tier are required unless --check-expectations is used", file=sys.stderr)
        return 1
    if args.subset_evidence and args.tier == "smoke":
        print("FAIL: smoke is a declared acceptance row, not subset evidence", file=sys.stderr)
        return 1
    try:
        validation = validate_root(args.artifact_root, args.expectations, expectations, args.tier, args.subset_evidence)
    except (AttributeError, KeyError, TypeError, ValueError, OSError) as error:
        print(f"FAIL: malformed expectations or artifact: {error}", file=sys.stderr)
        return 1
    if validation.errors:
        for error in validation.errors:
            print(f"FAIL: {error}", file=sys.stderr)
        print(f"FAILED: {len(validation.errors)} violation(s)", file=sys.stderr)
        return 1
    if args.subset_evidence:
        print("EVIDENCE_OK_NOT_ACCEPTED: selected rows are valid; aggregate the canonical row set", file=sys.stderr)
        return 3
    print(
        f"PASS: {args.tier} acceptance ({len(expectations['canonical_row_ids']) if args.tier != 'smoke' else 1} row(s))"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
