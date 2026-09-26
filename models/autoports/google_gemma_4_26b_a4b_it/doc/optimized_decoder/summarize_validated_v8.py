# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Package completed v8 evidence; exit 0/pass, 1/failed, 2/incomplete. CPU only."""

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

RUNTIME = "5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898"
MODEL_PATH = Path("models/autoports/google_gemma_4_26b_a4b_it")
MODULE = ".".join(MODEL_PATH.parts) + ".tests."
THRESHOLD = 0.995
KINDS = {0: "sliding_attention", 5: "full_attention"}
REUSE_LENGTHS = [31, 32, 33, 1023, 1024, 1025, 2049, 33, 2047]


class MissingEvidence(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    if not path.is_file():
        raise MissingEvidence(str(path))
    return json.loads(path.read_text())


def option(command, name):
    return command[command.index(name) + 1] if name in command else None


def pcc(value):
    return type(value) in (int, float) and math.isfinite(value) and THRESHOLD <= value <= 1.000000000001


def checked_row(row):
    return isinstance(row, dict) and row.get("passed") is True and pcc(row.get("pcc"))


def exact_rows(rows, key, expected):
    require(isinstance(rows, list), f"Missing {key} rows")
    require([row.get(key) for row in rows] == expected, f"Unexpected {key} coverage")
    require(all(checked_row(row) for row in rows), f"Failed or contradictory {key} PCC checks")
    return min(row["pcc"] for row in rows)


def sampled_positions(length):
    return sorted(
        set(
            [
                0,
                min(31, length - 1),
                min(32, length - 1),
                *range(1023, length, 1024),
                *range(max(0, length - 33), length),
            ]
        )
    )


def check_fixture(report, fixture):
    metadata = load(fixture.with_suffix(".json"))
    provenance = report.get("input_fixture")
    if isinstance(provenance, dict):
        actual_path, actual_hash = provenance.get("path"), provenance.get("sha256")
        source = provenance.get("source", {})
    else:
        actual_path, actual_hash = provenance, report.get("input_fixture_sha256")
        source = report.get("input_source", {}).get("source", {})
    require(actual_path and Path(actual_path).resolve() == fixture.resolve(), "Input fixture path differs")
    require(actual_hash == metadata.get("fixture_sha256") and bool(actual_hash), "Input fixture manifest hash differs")
    require(source.get("kind") == "recorded_real_text_hf_layer_inputs", "Input is not recorded real text")
    return actual_hash


def check_v8_policy(report, layer):
    policy = report.get("precision_policy", {})
    qkv = policy.get("prefill_qkv_projection", {})
    require(
        qkv.get("backend") == "minimal_matmul"
        and qkv.get("grid") == [11, 8]
        and qkv.get("k_block") == (8 if layer == 0 else 16),
        "Wrong v8 prefill QKV backend/geometry",
    )
    require(
        qkv.get("fidelity") == ("MathFidelity.HiFi4" if layer == 0 else "MathFidelity.HiFi2")
        and qkv.get("weight_dtype") == "DataType.BFLOAT8_B"
        and qkv.get("output_dtype") == "float32"
        and qkv.get("fp32_dest_acc_en") is True,
        "Wrong v8 prefill QKV precision",
    )
    require(
        qkv.get("m_block") == ("min(4, padded_M_tiles)" if layer == 0 else "min(2, padded_M_tiles)"),
        "Wrong v8 QKV M block",
    )
    require(
        qkv.get("input_memory") == ("unchanged" if layer == 0 else "L1 from input normalization"),
        "Wrong v8 QKV input producer placement",
    )
    for index, program in enumerate(qkv.get("programs", {}).values(), start=1):
        expected_m = min(index, 4 if layer == 0 else 2)
        require(
            f"M_block_size={expected_m},K_block_size={8 if layer == 0 else 16},N_block_size=8,subblock_h=1,subblock_w=4,compute_with_storage_grid_size=11-8"
            in program,
            "Wrong effective minimal program",
        )
    require(len(qkv.get("programs", {})) == 4, "Missing minimal tail configurations")
    output = policy.get("prefill_output_projection", {})
    require(output.get("k_block") == (16 if layer == 0 else 8), "Wrong v8 prefill output K block")
    if layer == 5:
        require(output.get("backend") == "minimal_matmul", "Wrong v8 full prefill output backend")


def check_decoder(report, layer, length, steps, decoder="optimized"):
    require(report.get("decoder") == decoder, "Wrong decoder")
    require(report.get("layer_type") == KINDS[layer] and report.get("real_weights") is True, "Wrong layer/weights")
    require(report.get("length") == length and report.get("prefix_length") == 0, "Wrong prefill length/prefix")
    require(checked_row(report), "Prefill HF failed")
    require(report.get("runtime_prefill_audit") == "clean", "Prefill audit missing/failed")
    require(
        report.get("program_cache_miss_guard") is True and report.get("prefill_cache_entries", 0) > 0,
        "Program-cache guard missing",
    )
    decode = report.get("decode", {})
    require(decode.get("runtime_decode_audit") == "clean", "Decode audit missing/failed")
    require(decode.get("traced") is True and decode.get("repeated_equal") is True, "Trace/determinism missing")
    require(
        decode.get("steps") == steps and decode.get("positions") == [length, length + steps - 1], "Wrong decode range"
    )
    minimum = exact_rows(decode.get("checks"), "position", [length, *range(length, length + steps)])
    require(decode.get("passed") is True and pcc(decode.get("pcc")), "Decode HF failed")
    require(math.isclose(decode.get("min_pcc", -2), minimum, abs_tol=1e-12, rel_tol=0), "Decode minimum differs")
    if decoder == "optimized":
        require(report.get("runtime_sha256") == RUNTIME, "Optimized runtime hash differs")
        check_v8_policy(report, layer)
        require(
            report.get("functional_fallback") == "forbidden" and report.get("contract") == "run_decoder",
            "Default device-only contract missing",
        )
        require("candidate" not in report, "Unexpected candidate override")
    return minimum


def expected_contracts():
    result = []
    for layer in (5, 0):
        for name, contract, length in [
            ("batched", "batched", None),
            ("prefix_continuation", "prefix_continuation", None),
            ("request_reuse", "request_reuse", None),
            ("bf16_cache_prefix", "prefix_continuation", None),
            ("long_262144", "long_context", 262144),
            ("long_262143", "long_context", 262143),
            ("headline", "run_decoder", 4096),
            ("watcher", "run_decoder", 4096),
        ]:
            if layer == 0 and name not in ("headline", "watcher"):
                continue
            result.append((f"validated_v8_{name}_layer{layer}.json", contract, layer, length, name))
    return result


class Evidence:
    def __init__(self, root):
        self.root = root
        self.repo = next(parent for parent in root.parents if (parent / MODEL_PATH).is_dir())
        self.pending, self.errors = [], []
        self.journal = []

    def attempt(self, label, operation):
        try:
            result = operation()
            return {"status": "passed", **result}
        except MissingEvidence as error:
            self.pending.append(f"{label}: {error}")
            return {"status": "incomplete", "missing": str(error)}
        except (ValueError, KeyError, TypeError, OSError, IndexError) as error:
            self.errors.append(f"{label}: {error}")
            return {"status": "failed", "error": str(error)}

    def command(self, filename):
        matches = [
            entry for entry in self.journal if Path(option(entry.get("command", []), "--output") or "").name == filename
        ]
        if not matches:
            raise MissingEvidence(f"Completed journal entry for {filename}")
        require(len(matches) == 1, f"Duplicate entry for {filename}")
        entry = matches[0]
        if "returncode" not in entry:
            raise MissingEvidence(f"Command still running for {filename}")
        require(entry["returncode"] == 0, f"Nonzero process result for {filename}")
        require(entry.get("runtime_sha256") == RUNTIME, "Command runtime differs")
        require(entry.get("driver_sha256") == digest(self.root / "run_validated_v8.py"), "Driver hash differs")
        require(entry.get("output_sha256") == digest(self.root / filename), "Output hash differs from journal")
        return entry

    def driver_provenance(self):
        path = self.root / "validated_v8_environment_snapshot.json"
        snapshot = load(path)
        driver = self.root / "run_validated_v8.py"
        require(
            snapshot.get("runtime_sha256") == RUNTIME and snapshot.get("driver_sha256") == digest(driver),
            "Environment/source snapshot differs",
        )
        environment = snapshot.get("environment", {})
        require(environment and not any(environment.values()), "Validation environment not clean")
        return dict(
            driver=driver.name,
            driver_sha256=digest(driver),
            environment_snapshot=path.name,
            environment_snapshot_sha256=digest(path),
            captured_at_utc=snapshot.get("recorded_at_utc"),
            environment=environment,
        )

    def contract(self, filename, contract, layer, length, name):
        entry = self.command(filename)
        command = entry["command"]
        require(option(command, "-m") == MODULE + "run_optimized_contract", "Wrong contract runner")
        require(
            option(command, "--contract") == contract and option(command, "--layer") == str(layer),
            "Wrong command target",
        )
        fixture = self.root / (
            f"actual_text_long/actual_text_layer{layer}_262144_0.pt"
            if contract == "long_context"
            else f"actual_text_layer{layer}_4096_128.pt"
        )
        require(Path(option(command, "--input-fixture") or "").resolve() == fixture.resolve(), "Wrong command fixture")
        report = load(self.root / filename)
        require(report.get("runtime_sha256") == RUNTIME, "Runtime hash differs")
        require(
            report.get("decoder") == "optimized" and report.get("contract") == contract, "Wrong report decoder/contract"
        )
        require(report.get("functional_fallback") == "forbidden", "Functional fallback guard missing")
        require(report.get("layer_type") == KINDS[layer] and report.get("real_weights") is True, "Wrong layer/weights")
        fixture_hash = check_fixture(report, fixture)
        expected_override = {"kv_cache_dtype": "bfloat16"} if name == "bf16_cache_prefix" else {}
        require(
            json.loads(option(command, "--default-overrides") or "{}") == expected_override,
            "Unexpected command override",
        )
        require(report.get("candidate", {}) == expected_override, "Unexpected report override")
        policy = report.get("precision_policy", {})
        check_v8_policy(report, layer)
        require(policy.get("kv_cache") == ("bfloat16" if expected_override else "bfloat8_b"), "Wrong cache dtype")
        require(
            policy.get("prefill_expert_gate") == ("DataType.BFLOAT8_B" if layer == 0 else "DataType.BFLOAT4_B"),
            "Wrong selected prefill gate precision",
        )
        result = {
            "artifact": filename,
            "artifact_sha256": digest(self.root / filename),
            "contract": contract,
            "layer_type": KINDS[layer],
            "returncode": 0,
            "runtime_sha256": RUNTIME,
            "fixture_sha256": fixture_hash,
        }
        if contract == "batched":
            require(option(command, "--batch") == "32" and "--heterogeneous-positions" in command, "Wrong B32 command")
            require(report.get("batch") == 32 and report.get("heterogeneous_positions") is True, "B32 coverage missing")
            require(report.get("lengths") == list(range(32, 64)), "Wrong B32 lengths")
            result["minimum_prefill_pcc"] = exact_rows(report.get("prefill"), "slot", list(range(32)))
            result["minimum_decode_pcc"] = exact_rows(report.get("decode"), "slot", list(range(32)))
            require([x.get("length") for x in report["prefill"]] == list(range(32, 64)), "Wrong prefill row lengths")
            require([x.get("position") for x in report["decode"]] == list(range(32, 64)), "Wrong decode positions")
            require(
                report.get("traced") is True and report.get("repeated_equal") is True, "B32 trace/determinism missing"
            )
        elif contract == "prefix_continuation":
            require(
                report.get("length") == 65 and report.get("slot") == 1 and checked_row(report),
                "Prefix accuracy/geometry failed",
            )
            rows = report.get("cache_preservation", [])
            require(
                [(x.get("start"), x.get("end")) for x in rows] == [(0, 31), (31, 33), (33, 65)], "Wrong prefix spans"
            )
            require(
                all(x.get("prefix_unchanged") is True and x.get("other_slot_unchanged") is True for x in rows),
                "Cache preservation failed",
            )
        elif contract == "request_reuse":
            rows = report.get("requests", [])
            require([x.get("length") for x in rows] == REUSE_LENGTHS, "Reuse length coverage differs")
            require(
                all(x.get("passed") is True and pcc(x.get("prefill_pcc")) and pcc(x.get("decode_pcc")) for x in rows),
                "Reuse accuracy failed",
            )
            require(report.get("trace_reused") is True, "Trace reuse missing")
            require(
                entry.get("environment_overrides", {}).get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
                and entry.get("environment_overrides", {}).get("TT_METAL_TRACE_ALLOC_TRACEBACKS") == "1",
                "Full allocation tracking flags absent",
            )
            require(
                report.get("program_cache_entries_at_capture") == report.get("program_cache_entries")
                and report.get("program_cache_entries", 0) > 0,
                "Reuse cache grew",
            )
            require(
                report.get("program_cache_misses_while_trace_live") == "forbidden", "Live trace cache guard missing"
            )
            result.update(
                requests=rows,
                full_trace_allocation_tracking=True,
                program_cache_entries=report["program_cache_entries"],
            )
        elif contract == "long_context":
            require(option(command, "--length") == str(length) and report.get("length") == length, "Wrong long context")
            require(
                report.get("cache_positions") == length and report.get("scope") == "subset", "Wrong long cache/scope"
            )
            require(checked_row(report) and report.get("prefill_aggregate_passed") is True, "Aggregate prefill failed")
            require(
                report.get("prefill_sampled_rows_passed") is True,
                "Strict recorded-input sampled-row gate missing/failed",
            )
            samples = sampled_positions(length)
            require(len(samples) == 291 and report.get("compared_query_rows") == samples, "Query sample set differs")
            result["sampled_rows"] = len(samples)
            result["minimum_sampled_row_pcc"] = exact_rows(report.get("sampled_row_diagnostics"), "position", samples)
            exact_rows(report.get("prefill_tail_checks"), "position", [length - 2, length - 1])
            result["minimum_decode_pcc"] = exact_rows(
                report.get("decode"), "position", [length - 1, length - 2, length - 1]
            )
            require(
                report.get("runtime_prefill_audit") == "passed" and report.get("runtime_decode_audit") == "passed",
                "Long runtime audit failed",
            )
            reference = self.root / f"actual_text_long/actual_text_layer{layer}_{length}_reference.pt"
            require(
                Path(option(command, "--reference-file") or "").resolve() == reference.resolve(), "Wrong long oracle"
            )
            meta = load(reference.with_suffix(".json"))
            require(
                meta.get("compared_query_rows") == samples and meta.get("all_kv_positions") == length,
                "Oracle scope differs",
            )
            require(meta.get("input_fixture", {}).get("sha256") == fixture_hash, "Oracle fixture hash differs")
            require(meta.get("reference_sha256") == digest(reference), "Oracle content hash differs")
            result["reference_sha256"] = meta["reference_sha256"]
        else:
            require(
                option(command, "--length") == "4096" and option(command, "--steps") == "128", "Wrong headline command"
            )
            require(
                all(flag in command for flag in ("--real", "--decode", "--verify-program-cache")),
                "Headline flags missing",
            )
            require(entry.get("watcher") is (name == "watcher"), "Watcher journal flag differs")
            result["minimum_decode_pcc"] = check_decoder(report, layer, 4096, 128)
        if contract in ("batched", "prefix_continuation", "request_reuse"):
            require(report.get("runtime_audit") == "clean", "Public runtime audit failed")
        return result

    def watcher(self, layer):
        name = f"validated_v8_watcher_layer{layer}"
        self.contract(name + ".json", "run_decoder", layer, 4096, "watcher")
        self.driver_provenance()
        snapshot = load(self.root / "validated_v8_environment_snapshot.json")
        entry = self.command(name + ".json")
        require(entry.get("environment_overrides", {}).get("TT_METAL_WATCHER") == "10", "Watcher interval not recorded")
        profiler_keys = [
            "TT_METAL_DEVICE_PROFILER",
            "TT_METAL_DEVICE_PROFILER_NOC_EVENTS",
            "TT_METAL_PROFILE_PERF_COUNTERS",
        ]
        require(
            all(
                key in snapshot["environment"] and snapshot["environment"][key] in (None, "", "0")
                for key in profiler_keys
            ),
            "Watcher/profiler separation missing",
        )
        console_path, device_path = self.root / (name + ".log"), self.root / (name + ".device.log")
        if not console_path.is_file() or not device_path.is_file():
            raise MissingEvidence(f"watcher console/device logs for layer {layer}")
        console, device = console_path.read_text(), device_path.read_text()
        require(
            "Watcher server initialized, disabled features: None" in console,
            "Watcher disabled-feature evidence missing",
        )
        require(
            "Watcher checking device" in console and "Watcher thread stopped watching" in console,
            "Watcher lifecycle incomplete",
        )
        require("starting" in device and len(device) > 1024, "Watcher device log empty/incomplete")
        failure = re.compile(
            r"(?<![-\w])(?:ERROR|FATAL|ASSERTION|CORRUPT|CORRUPTION)\b|Traceback \(most recent call last\)|"
            r"invalid (?:NoC|NOC)|(?:stack|circular.buffer).*overflow",
            re.IGNORECASE,
        )
        lines = [line[:500] for text in (console, device) for line in text.splitlines() if failure.search(line)]
        require(not lines, "Watcher failure lines: " + repr(lines[:5]))
        return {
            "layer": layer,
            "runtime_sha256": RUNTIME,
            "watcher_interval": "10",
            "disabled_features": None,
            "result": name + ".json",
            "device_log": device_path.name,
            "console_log": console_path.name,
            "device_log_bytes": device_path.stat().st_size,
            "device_log_sha256": digest(device_path),
            "failure_lines": [],
            "profiler_enabled": False,
            "profiler_evidence": "validated_v8_environment_snapshot.json",
            "profiler_evidence_scope": "live orchestrator inherited environment plus saved driver; watcher logs confirm lifecycle",
            "passed": True,
        }

    def pytest(self):
        entries = [x for x in self.journal if option(x.get("command", []), "-m") == "pytest"]
        if not entries:
            raise MissingEvidence("completed pytest command")
        require(len(entries) == 1 and entries[0].get("returncode") == 0, "Pytest command failed/duplicated")
        command = entries[0]["command"]
        require(str(MODEL_PATH / "tests/test_optimized_decoder.py") in command, "Wrong pytest target")
        log = self.root / "pytest_validated_v8.log"
        if not log.is_file():
            raise MissingEvidence(str(log))
        lines = [line for line in log.read_text().splitlines() if re.search(r"(?:^|=)\s*4 passed(?:,| in)", line)]
        require(
            len(lines) == 1 and not re.search(r"\d+ (failed|error|skipped|deselected)", lines[0]),
            "Expected exactly four pytest passes",
        )
        cases = []
        base = self.root / "pytest_validated_v8_tmp"
        for path in base.glob("*/optimized.json"):
            if path.parent.is_symlink():
                continue
            report = load(path)
            layer = next((layer for layer, kind in KINDS.items() if report.get("layer_type") == kind), None)
            require(layer in KINDS and report.get("length") in (33, 65), "Unexpected pytest case")
            check_decoder(report, layer, report["length"], 8)
            require(
                report.get("input_source", {}).get("source", {}).get("kind") == "recorded_real_text_hf_layer_inputs",
                "Pytest fixture not recorded",
            )
            cases.append({"layer": layer, "length": report["length"], "report": str(path.relative_to(self.root))})
        require(
            sorted((x["layer"], x["length"]) for x in cases) == [(0, 33), (0, 65), (5, 33), (5, 65)],
            "Four raw pytest cases missing",
        )
        return {"artifact": log.name, "passed": 4, "summary_line": lines[0], "cases": cases}

    def stress(self):
        entries = [x for x in self.journal if option(x.get("command", []), "-m") == MODULE + "run_optimized_stress"]
        if not entries:
            raise MissingEvidence("completed stress command")
        require(len(entries) == 1 and entries[0].get("returncode") == 0, "Stress command failed/duplicated")
        directory = self.root / "stress_validated_v8_defaults"
        require(
            Path(option(entries[0]["command"], "--output-dir") or "").resolve() == directory.resolve(),
            "Wrong stress directory",
        )
        summary = load(directory / "summary.json")
        require(
            summary.get("length") == 1025 and summary.get("steps") == 512 and summary.get("threshold") == THRESHOLD,
            "Stress scope differs",
        )
        require(
            summary.get("input_mode") == "recorded" and summary.get("real_weights") is True, "Stress input mode differs"
        )
        require(
            all(
                summary.get(key) is True
                for key in ("sources_unchanged", "input_fixtures_unchanged", "preservation_passed", "exact_hf_passed")
            ),
            "Stress gates failed",
        )
        require(summary.get("errors") == [], "Stress errors present")
        require(
            summary.get("source_sha256", {}).get(str(MODEL_PATH / "tt/optimized_decoder.py")) == RUNTIME,
            "Stress runtime hash differs",
        )
        run_dir = Path(summary["run_directory"])
        require(run_dir.resolve().is_relative_to(directory.resolve()), "Stress run directory outside campaign")
        nested = load(Path(summary["command_journal"]))
        require(
            len(nested) == 4 and all(x.get("exit_code") == 0 for x in nested),
            "Incomplete/failed stress subprocess journal",
        )
        layers = {}
        for layer in KINDS:
            record = summary.get("layers", {}).get(str(layer), {})
            reports = {}
            minima = {}
            for decoder in ("fused", "optimized"):
                path = Path(record.get("reports", {}).get(decoder, ""))
                require(path.resolve().parent == run_dir.resolve(), "Stress report directory differs")
                reports[decoder] = load(path)
                minima[decoder] = check_decoder(reports[decoder], layer, 1025, 512, decoder)
                check_fixture(reports[decoder], self.root / f"actual_text_layer{layer}_1025_512.pt")
                command_rows = [
                    x for x in nested if Path(option(x["command"], "--output") or "").resolve() == path.resolve()
                ]
                require(len(command_rows) == 1, "Stress report not bound to its command")
            require(
                reports["fused"].get("input_source") == reports["optimized"].get("input_source"),
                "Stress input metadata differs",
            )
            comparison_path = Path(record.get("comparison", ""))
            require(comparison_path.resolve().parent == run_dir.resolve(), "Comparison directory differs")
            comparison = load(comparison_path)
            require(
                comparison.get("direct_passed") is True and checked_row(comparison.get("prefill", {})),
                "Direct prefill failed",
            )
            minimum = exact_rows(comparison.get("decode"), "position", list(range(1025, 1537)))
            require(all(row.get("finite") is True for row in comparison["decode"]), "Nonfinite direct decode")
            require(
                math.isclose(comparison.get("minimum_direct_decode_pcc", -2), minimum, abs_tol=1e-12, rel_tol=0),
                "Direct minimum differs",
            )
            for key in (
                "baseline_hf_failed_positions",
                "candidate_hf_failed_positions",
                "shared_hf_failed_positions",
                "new_hf_failed_positions",
                "direct_failed_positions",
            ):
                require(comparison.get(key) == [], f"Stress has {key}")
            for key, decoder in (("baseline", "fused"), ("candidate", "optimized")):
                provenance = comparison.get(key, {})
                report_path = Path(record["reports"][decoder])
                require(
                    Path(provenance.get("report", "")).resolve() == report_path.resolve(), "Comparison report differs"
                )
                require(provenance.get("report_sha256") == digest(report_path), "Comparison report hash differs")
                tensor_path = Path(provenance.get("tensors", ""))
                require(
                    tensor_path.resolve().parent == run_dir.resolve() and tensor_path.is_file(),
                    "Saved stress tensors missing",
                )
                require(provenance.get("tensor_sha256") == digest(tensor_path), "Comparison tensor hash differs")
            layers[str(layer)] = {
                "minimum_hf_decode_pcc": minima,
                "minimum_direct_decode_pcc": minimum,
                "reports": record["reports"],
                "comparison": str(comparison_path),
            }
        return {
            "artifact": "stress_validated_v8_defaults/summary.json",
            "exact_hf_passed": True,
            "preservation_passed": True,
            "steps_per_kind": 512,
            "prefill_tokens": 1025,
            "layers": layers,
        }


def boundary_controls(root):
    journal = load(root / "v8_boundary_commands.json")
    expected = [
        ("tight_cache_v8_layer5_1025.json", 1025, True),
        *((f"prefill_boundary_v8_optimized_layer5_{length}.json", length, False) for length in (65, 1023, 1024, 1025)),
    ]
    if len(journal) < 5:
        raise MissingEvidence(f"Boundary commands {len(journal)}/5")
    require(
        len(journal) == 5 and all(entry.get("returncode") == 0 for entry in journal),
        "Boundary campaign incomplete/failed",
    )
    results = []
    for filename, length, tight in expected:
        matches = [entry for entry in journal if Path(option(entry["command"], "--output")).name == filename]
        require(len(matches) == 1, "Boundary journal scope differs")
        entry = matches[0]
        path = root / filename
        report = load(path)
        require(
            entry.get("runtime_sha256") == RUNTIME
            and entry.get("driver_sha256") == digest(root / "run_v8_boundaries.py"),
            "Boundary source differs",
        )
        require(entry.get("output_sha256") == digest(path), "Boundary report changed")
        minimum = check_decoder(report, 5, length, 1)
        fixture = Path(report["input_fixture"])
        require(
            report["input_fixture_sha256"] == digest(fixture)
            and report["input_source"]["source"]["kind"] == "recorded_real_text_hf_layer_inputs",
            "Boundary actual fixture differs",
        )
        row = dict(
            artifact=filename,
            artifact_sha256=digest(path),
            runtime_sha256=RUNTIME,
            length=length,
            prefill_pcc=report["pcc"],
            minimum_decode_pcc=minimum,
            input_fixture_sha256=report["input_fixture_sha256"],
        )
        if tight:
            require(
                entry["watcher"] is True and report["cache_extent"] == 1152 and report["cache_pages"] == 36,
                "Tight capacity differs",
            )
            checks = report.get("paged_prefill_capacity_checks", [])
            require(
                report.get("paged_prefill_capacity_passed") is True and checks, "Native capacity instrumentation absent"
            )
            require(
                all(
                    item["passed"] is True
                    and item["q_chunk"] == 64
                    and item["k_chunk"] == 128
                    and item["read_end"] == 1152
                    and item["cache_capacity"] == 1152
                    and item["table_pages"] == 36
                    for item in checks
                ),
                "Tight K128 branch differs",
            )
            console = path.with_suffix(".log").read_text()
            device = path.with_suffix(".device.log").read_text()
            require(
                all(
                    text in console
                    for text in [
                        "Watcher server initialized, disabled features: None",
                        "Watcher checking device",
                        "Watcher thread stopped watching",
                    ]
                ),
                "Tight Watcher lifecycle absent",
            )
            require("starting" in device and len(device) > 1024, "Tight Watcher device log missing")
            failure = re.compile(
                r"(?<![-\w])(?:ERROR|FATAL|ASSERTION|CORRUPT|CORRUPTION)\b|Traceback \(most recent call last\)|invalid (?:NoC|NOC)|(?:stack|circular.buffer).*overflow",
                re.I,
            )
            require(
                not any(failure.search(line) for text in (console, device) for line in text.splitlines()),
                "Tight Watcher failure",
            )
            row.update(
                capacity_checks=checks,
                watcher_console_sha256=digest(path.with_suffix(".log")),
                watcher_device_sha256=digest(path.with_suffix(".device.log")),
            )
        else:
            require(
                entry["logical_length"] == length
                and entry["physical_rows"] == ([96] if length == 65 else [1024, 32] if length == 1025 else [1024]),
                "Boundary physical geometry differs",
            )
            times = report.get("warmed_prefill_host_us", [])
            require(
                len(times) == 3
                and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in times),
                "Warmed samples missing",
            )
            row.update(physical_rows=entry["physical_rows"], warmed_prefill_host_us=times)
        results.append(row)
    return dict(
        command_journal="v8_boundary_commands.json",
        command_journal_sha256=digest(root / "v8_boundary_commands.json"),
        command_count=5,
        results=results,
    )


def inherited_scope(root):
    prior = load(root / "validated_v7_validation_summary.json")
    ancestor = "daa82a4a5197a007ccd5d29d912f082e695fd37a596578f25fba96a6694e625b"
    require(
        prior["status"] == "current_targeted_and_inherited_gates_passed" and prior["runtime_sha256"] == ancestor,
        "V7 ancestor acceptance differs",
    )
    delta = load(root / "source_delta_v8.json")
    require(
        delta["current_runtime_sha256"] == RUNTIME and delta["ancestor_runtime_sha256"] == ancestor,
        "Source-delta hashes differ",
    )
    require(digest(root / "runtime_v7_before_placement.py.txt") == ancestor, "Exact ancestor snapshot differs")
    require(
        len(delta["unchanged_methods"]) == 36
        and all(row["remaining_ast_exactly_matches_v7"] for row in delta["exact_delta_reversal_proof"]),
        "Source-delta proof failed",
    )
    require(
        all(
            row["passed"]
            for key in ("auto_resolution_checks", "projection_setup_checks", "normalization_checks")
            for row in delta[key]
        ),
        "Source behavior proof failed",
    )
    require(
        delta["ancestor_manifest_sha256"] == digest(root / "validated_v7_validation_summary.json"),
        "Ancestor manifest hash differs",
    )
    inherited = prior["inherited"]
    retained = inherited["sliding_v5_contracts"]
    require(
        len(retained) == 6
        and all(
            row["status"] == "passed" and digest(root / row["artifact"]) == row["artifact_sha256"] for row in retained
        ),
        "Retained sliding contracts differ",
    )
    reuse = inherited["sliding_v6_allocation_tracked_reuse"]
    require(digest(root / reuse["artifact"]) == reuse["artifact_sha256"], "Retained full-tracker sliding reuse differs")
    return dict(
        source_delta="source_delta_v8.json",
        source_delta_sha256=digest(root / "source_delta_v8.json"),
        ancestor_manifest="validated_v7_validation_summary.json",
        ancestor_manifest_sha256=digest(root / "validated_v7_validation_summary.json"),
        ancestor_snapshot="runtime_v7_before_placement.py.txt",
        ancestor_runtime_sha256=ancestor,
        inherited_source_chain=inherited,
        sliding_v5_contracts=retained,
        sliding_v6_allocation_tracked_reuse=reuse,
        scope="Unchanged sliding auto defaults preserve M4 and no input-L1 branch. Broad sliding contracts retain original v5/v6 hashes through the v7/v8 exact source chain; these six contracts did not rerun on v8. Full prefill placement/M2 is freshly checked.",
    )


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    evidence = Evidence(args.evidence_dir.resolve())
    root = evidence.root
    output = (args.output_dir or root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = evidence.repo / MODEL_PATH / "tt/optimized_decoder.py"
    require(digest(runtime) == RUNTIME, "Current source differs from frozen campaign")
    provenance = evidence.attempt("driver provenance", evidence.driver_provenance)
    inherited = evidence.attempt("inherited sliding scope", lambda: inherited_scope(root))
    journal = root / "validated_v8_contract_commands.json"
    evidence.journal = load(journal) if journal.is_file() else []
    if len(evidence.journal) < 12:
        evidence.pending.append(f"Primary journal commands {len(evidence.journal)}/12")
    if len(evidence.journal) > 12:
        evidence.errors.append("Unexpected primary command count")
    for entry in evidence.journal:
        if "returncode" not in entry:
            evidence.pending.append("Primary command still running: " + str(entry["command"]))
        elif entry["returncode"] != 0:
            evidence.errors.append("Nonzero primary process result: " + str(entry["command"]))
    expected = expected_contracts()
    filenames = {row[0] for row in expected}
    observed = [
        Path(option(entry["command"], "--output")).name
        for entry in evidence.journal
        if option(entry["command"], "--output")
    ]
    if len(observed) != len(set(observed)) or not set(observed).issubset(filenames):
        evidence.errors.append("Unexpected/duplicate public outputs")
    contracts = [evidence.attempt(item[0], lambda item=item: evidence.contract(*item)) for item in expected]
    watchers = [
        evidence.attempt(f"watcher layer{layer}", lambda layer=layer: evidence.watcher(layer)) for layer in (0, 5)
    ]
    pytest = evidence.attempt("pytest", evidence.pytest)
    stress = evidence.attempt("stress", evidence.stress)
    boundaries = evidence.attempt("current boundaries", lambda: boundary_controls(root))
    # The primary driver persists an entry before subprocess completion.
    # In-progress pytest/stress are pending, not accuracy failures.
    for name in ("pytest", "stress"):
        prefix = name + ": "
        matching = [
            entry
            for entry in evidence.journal
            if (
                option(entry["command"], "-m") == "pytest"
                if name == "pytest"
                else option(entry["command"], "-m") == MODULE + "run_optimized_stress"
            )
        ]
        if matching and "returncode" not in matching[0]:
            evidence.errors = [error for error in evidence.errors if not error.startswith(prefix)]
            evidence.pending.append(name + " subprocess still running")
            if name == "pytest":
                pytest = dict(status="incomplete")
            else:
                stress = dict(status="incomplete")
    require(digest(runtime) == RUNTIME, "Runtime changed during CPU validation")
    status = (
        "failed"
        if evidence.errors
        else "incomplete"
        if evidence.pending
        else "current_targeted_and_inherited_gates_passed"
    )
    summary = dict(
        status=status,
        runtime_sha256=RUNTIME,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        generator_sha256=digest(Path(__file__)),
        driver_provenance=provenance,
        validation_basis="12 fresh primary commands plus5 affected-prefill boundary commands; unchanged sliding broad public/lifecycle coverage explicitly inherited through source proof. Not a full18-command v8 rerun.",
        command_journal=journal.name,
        command_journal_sha256=digest(journal) if journal.is_file() else None,
        command_count=len(evidence.journal),
        expected_command_count=12,
        public_contracts=contracts,
        pytest=pytest,
        stress=stress,
        watcher_summary="validated_v8_watcher_summary.json",
        boundaries=boundaries,
        inherited=inherited,
        accuracy_gate="PCC>=.995, including every one of291 sampled long-context rows",
        pending=evidence.pending,
        errors=evidence.errors,
        limits=[
            "Long-context accuracy is sampled291 rows, not all-token/full-model accuracy.",
            "Large standard fixtures bound by run reports/manifests; small boundary fixture bytes rehashed.",
            "Native profiles/timings are separate and are never rebound from ancestor v5/v6/v7.",
        ],
    )
    target = output / "validated_v8_validation_summary.json"
    if target.is_file():
        previous = load(target)
        if {key: value for key, value in previous.items() if key != "generated_at_utc"} == {
            key: value for key, value in summary.items() if key != "generated_at_utc"
        }:
            summary["generated_at_utc"] = previous["generated_at_utc"]
    write(target, summary)
    write(
        output / "validated_v8_watcher_summary.json",
        dict(
            status=(
                "failed"
                if any(row["status"] == "failed" for row in watchers)
                else "incomplete"
                if any(row["status"] == "incomplete" for row in watchers)
                else "passed"
            ),
            runtime_sha256=RUNTIME,
            watchers=watchers,
        ),
    )
    print(
        json.dumps(
            dict(
                status=status,
                command_count=len(evidence.journal),
                pending=len(evidence.pending),
                errors=evidence.errors,
            ),
            indent=2,
        )
    )
    return 1 if evidence.errors else 2 if evidence.pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
