# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed CPU audit of current optimized-decoder device artifacts.

Only a fully successful audit writes hard_gates.json. This checker validates the
fixed baseline comparison, correctness/trace/cache/Watcher battery and actual
profile precision rows. It does not choose the best experimental candidate or
replace the owner's best-correct-candidate ranking and independent stage review.
Only source files are hashed to compare existing provenance; raw artifacts are
neither hashed nor archived. No TTNN import or device access occurs.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[1]
DEFAULT_DOC = ROOT / "doc/optimized_decoder"
KINDS = {0: "linear_attention", 3: "full_attention", 34: "linear_attention"}
OFF = (None, "", "0")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def pcc(value, where):
    require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0.995, f"{where}: PCC < .995")


def read_json(path):
    with path.open() as handle:
        return json.load(handle)


class Audit:
    def __init__(self, doc):
        self.doc = doc.resolve()
        self.sources = {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("tt/optimized_decoder.py", "tt/fused_decoder.py")
        }
        self.checked = []

    def load(self, name, *, optimized=True, tracking=False, clean=False):
        path = self.doc / name
        data = read_json(path)
        provenance = read_json(Path(str(path) + ".provenance.json"))
        expected = "optimized" if optimized else "fused"
        require(provenance["decoder"] == expected, f"{path}: incorrect selected decoder")
        require(provenance["execution"] == "TTNN device", f"{path}: not device evidence")
        require(provenance["regime"] == "serial" and provenance["device"], f"{path}: regime/device missing")
        for source, digest in self.sources.items():
            if source == "tt/optimized_decoder.py" and not optimized:
                continue
            require(provenance["source_sha256"].get(source) == digest, f"{path}: stale {source}")
        if "decoder" in data:
            require(data["decoder"] == ("OptimizedDecoder" if optimized else "FusedDecoder"), f"{path}: wrong class")
        for document in (data, provenance):
            require(document.get("oracle_assisted", False) is False, f"{path}: oracle-assisted output")
            require(document.get("acceptance_eligible", True) is not False, f"{path}: diagnostic-only artifact")
            if "measured_decoder_source_sha256" in document:
                source = f"tt/{expected}_decoder.py"
                require(document["measured_decoder_source_sha256"] == self.sources[source], f"{path}: wrong snapshot")
        env = provenance["env"]
        if tracking:
            require(env.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1", f"{path}: tracking disabled")
            require(env.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") in OFF, f"{path}: program cache excluded")
        if clean:
            for key in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
                require(env.get(key) in OFF, f"{path}: instrumented timing ({key})")
        if optimized:
            command = provenance["command"]
            for index, arg in enumerate(command):
                if arg == "--build-kwargs":
                    require(json.loads(command[index + 1]) == {}, f"{path}: final gate must use production defaults")
                elif arg.startswith("--build-kwargs="):
                    require(json.loads(arg.split("=", 1)[1]) == {}, f"{path}: final gate uses explicit candidate")
        self.checked.append(str(path.relative_to(self.doc)))
        return data, provenance

    def benchmark(self, name, layer, *, optimized=True, headline=True):
        data, provenance = self.load(name, optimized=optimized, clean=True)
        require(data["layer"] == layer and data["kind"] == KINDS[layer], f"{name}: layer/kind")
        require(data["deterministic"] is True and data["runtime_fallback_audit"] == "pass", f"{name}: runtime")
        require(data["latency_tracking_enabled"] is False, f"{name}: instrumented latency")
        for field in ("prefill_pcc", "prefill_contribution_pcc", "decode_eager_pcc", "decode_traced_pcc"):
            pcc(data[field], f"{name}/{field}")
        if "hf_prefill_state_pcc" in data:
            pcc(data["hf_prefill_state_pcc"], f"{name}/state")
        for phase, metric in (("prefill", "prefill_warmed_ms"), ("decode", "decode_traced_warmed_ms")):
            samples = data[phase + "_samples_ms"]
            minimum = (10 if phase == "prefill" else 60) if headline else (3 if phase == "prefill" else 20)
            require(len(samples) >= minimum and all(positive(v) for v in samples), f"{name}: {phase} samples")
            require(positive(data[metric]), f"{name}: {metric}")
            require(math.isclose(data[metric], statistics.median(samples), rel_tol=1e-9), f"{name}: median mismatch")
        return data, provenance

    def stress(self, layer, length):
        name = f"stress_{layer}.json" if length == 128 else f"stress_extended_{layer}.json"
        data, provenance = self.load(name, tracking=True)
        expected = dict(
            layer=layer,
            seed=154 if layer == 34 else 913,
            prompt_length=128,
            decode_length=length,
            continuation_length=33,
        )
        for key, value in expected.items():
            require(data[key] == value, f"{name}: {key}")
        require(data["gate_pass"] is True and data["failed_rows"] == [], f"{name}: failed stress")
        pcc(data["prefill_pcc"], name)
        require(data["captures"] == 1 and data["repeated_bit_equal"] is True, f"{name}: trace/determinism")
        require(data["tracking"] is True and data["program_cache_included"] is True, f"{name}: tracker evidence")
        require(data["cpu_oracle_cache"] is None, f"{name}: fresh HF reference required")
        require(data["cpu_oracle_key"]["parameters"] == expected, f"{name}: oracle parameters")
        require(Path(provenance["command"][0]).name == "stress_optimized_decoder.py", f"{name}: wrong harness")
        repeats = data["repeats"]
        require(type(repeats) is int and repeats >= 3, f"{name}: repeat count")
        rows = data["results"]
        require(len(rows) == repeats * length, f"{name}: missing decode rows")
        require(
            {(r["repeat"], r["position"]) for r in rows}
            == {(r, p) for r in range(repeats) for p in range(128, 128 + length)},
            f"{name}: decode coverage",
        )
        continuation = data["continuation_results"]
        require(
            len(continuation) == 33 and {r["position"] for r in continuation} == set(range(128 + length, 161 + length)),
            f"{name}: continuation coverage",
        )
        if data.get("gate_scope") == "whole-output .995; per-row diagnostic":
            for key in (
                "decode_whole_pcc",
                "continuation_whole_pcc",
                "decode_contribution_whole_pcc",
                "continuation_contribution_whole_pcc",
            ):
                pcc(data[key], f"{name}/{key}")
            expected_diagnostics = [r for r in rows + continuation if min(r["pcc"], r["contribution_pcc"]) < 0.995]
            require(data["row_diagnostics"] == expected_diagnostics, f"{name}: missing per-row diagnostics")
            for row in rows + continuation:
                require(
                    all(math.isfinite(row[k]) and -1 <= row[k] <= 1 for k in ("pcc", "contribution_pcc")),
                    f"{name}: invalid PCC",
                )
        else:
            # Earlier strict per-row passes are stronger evidence; never infer
            # an aggregate pass from a failed/missing aggregate artifact.
            for row in rows + continuation:
                for key in ("pcc", "contribution_pcc"):
                    pcc(row[key], f"{name}/{row.get('position')}/{key}")

    def battery(self, layer):
        name = f"battery_{layer}.json"
        data, _ = self.load(name, tracking=True)
        require(data["layer"] == layer and data["kind"] == KINDS[layer], f"{name}: layer")
        require(data["synthetic"] is False and data["runtime_fallback_audit"] == "pass", f"{name}: real runtime")
        groups = defaultdict(list)
        for row in data["results"]:
            groups[row["mode"]].append(row)
            if "pcc" in row:
                pcc(row["pcc"], f"{name}/{row['mode']}")
        boundaries = {1, 7, 31, 32, 33, 255, 256, 257, 513}
        require({r["length"] for r in groups["prefill"]} == boundaries, f"{name}: prefill boundaries")
        require({r["position"] for r in groups["decode"]} == boundaries, f"{name}: decode boundaries")
        for mode in ("chunk_partition_parity", "chunked_vs_unchunked"):
            require(len(groups[mode]) == 1 and groups[mode][0]["length"] == 256, f"{name}: {mode}")
        require({r["start"] for r in groups["resume"]} == {32, 33, 256, 512, 528}, f"{name}: resume coverage")
        for row in groups["resume"]:
            require(
                len(row["per_row_pcc"]) == row["count"] and len(row["dropped_history_per_row_pcc"]) == row["count"],
                f"{name}: resume count",
            )
            for score in row["per_row_pcc"]:
                pcc(score, f"{name}/resume")
            require(
                all(math.isfinite(v) for v in row["dropped_history_per_row_pcc"])
                and min(row["dropped_history_per_row_pcc"]) < 0.995,
                f"{name}: negative control",
            )
        for mode in ("batch_prefill", "batch_decode"):
            require(
                len(groups[mode]) == 32
                and {r["row"] for r in groups[mode]} == set(range(32))
                and all(r["batch"] == 32 for r in groups[mode]),
                f"{name}: {mode}",
            )
        traced = groups["batch_traced_decode"]
        require(
            len(traced) == 96
            and {(r["repeat"], r["row"]) for r in traced} == {(i, j) for i in range(3) for j in range(32)}
            and all(r["batch"] == 32 for r in traced),
            f"{name}: traced batch coverage",
        )
        deterministic = groups["batch_trace_determinism"]
        require(
            len(deterministic) == 1
            and deterministic[0] == dict(mode="batch_trace_determinism", batch=32, repeats=3, bit_equal=True),
            f"{name}: batch determinism",
        )
        if layer == 3:
            pool = groups["shared_pool_cache"]
            require(
                len(pool) == 64
                and {(r["row"], r["tensor"]) for r in pool} == {(i, k) for i in range(32) for k in ("key", "value")},
                f"{name}: pool coverage",
            )
            guard = groups["unowned_guard_pages"]
            require(len(guard) == 1 and guard[0]["untouched"] is True and guard[0]["pages"], f"{name}: guard pages")

    def trace_and_watcher(self, layer):
        name = f"trace_reuse_{layer}.json"
        data, provenance = self.load(name, tracking=True)
        require(data["layer"] == layer and data["tracking"] is True and data["program_cache_included"] is True, name)
        for key, value in dict(
            positive_control="rejected",
            captures=1,
            request_releases=0,
            teardown_releases=1,
            repeated_request_bit_equal=True,
            changed_input_changed_output=True,
        ).items():
            require(data[key] == value, f"{name}: {key}")
        rows = data["results"]
        expected = [(31, 0), (32, 0), (33, 0), (255, 0), (257, 0), (7, 0), (50, 33), (49, 32), (31, 0)]
        require([(r["length"], r["resume_start"]) for r in rows] == expected, f"{name}: request coverage")
        require(len({r["trace_id"] for r in rows}) == 1 and rows[0]["trace_id"], f"{name}: trace reuse")
        for row in rows:
            pcc(row["pcc"], name)
        require(
            provenance["env"].get("TT_METAL_WATCHER") == "10"
            and provenance["env"].get("TT_METAL_DEVICE_PROFILER") in OFF,
            f"{name}: watcher/profiler separation",
        )
        audit = read_json(self.doc / "watcher_audit.json")
        require(audit["clean"] is True, "Watcher audit failed")
        matching = [r for r in audit["runs"] if r["layer"] == layer]
        require(len(matching) == 1, f"Watcher audit layer {layer} missing")
        entry = matching[0]
        require(
            entry["hardware_errors"] == [] and entry["trace_reuse_pass"] is True and entry["completed_dumps"] > 0,
            f"Watcher layer {layer} failed",
        )
        raw = (self.doc / entry["log"]).read_text()
        console = (self.doc / f"logs/trace_reuse_watcher_{layer}.log").read_text()
        require(len(re.findall(r"Dump #\d+ completed", raw)) == entry["completed_dumps"], "Watcher dump mismatch")
        failure = r"Watcher (?:detected|found corruption|data corruption|unexpected|stopped the device)|tripped assert|corrupted noc sanitization|TT_FATAL"
        require(not re.search(failure, raw + console, re.I), f"Watcher layer {layer} raw error")

    def context(self, layer):
        name = f"context_32768_{layer}.json"
        data, _ = self.load(name, tracking=True)
        require(data["layer"] == layer and data["advertised"] == 262144 and data["tested"] == 32768, name)
        rows = data["results"]
        require(len(rows) == 3, f"{name}: incomplete context run")
        require({r["length"] for r in rows if r["mode"] == "prefill"} == {32767, 32768}, name)
        decode = [r for r in rows if r["mode"] == "traced_decode"]
        require(
            len(decode) == 1
            and decode[0]["position"] == 32767
            and decode[0]["context"] == 32768
            and decode[0]["repeats"] >= 3
            and decode[0]["deterministic"] is True,
            name,
        )
        for row in rows:
            pcc(row["pcc"], name)
            for value in row.get("selected_row_pcc", {}).values():
                pcc(value, f"{name}/selected row")

    def page_checks(self):
        data, _ = self.load("page_extent_regression.json")
        rows = data["results"]
        require(
            len(rows) == 10 and {r["capacity"] for r in rows} == {1, 31, 32, 33, 255, 256, 257, 511, 512, 513},
            "Page extent coverage",
        )
        for row in rows:
            require(
                row["unowned_pages_untouched"] is True and row["future_rows_untouched"] is True,
                "Page extent guard corruption",
            )
            require(
                row["owned_pages"] == (row["capacity"] + 31) // 32
                and row["table_entries"] % 8 == 0
                and row["table_entries"] >= row["owned_pages"],
                "Page extent metadata",
            )
            pcc(row["prefill_pcc"], "page extent prefill")
            pcc(row["decode_pcc"], "page extent decode")
        data, _ = self.load("padded_trace_regression.json", tracking=True)
        require(
            data["captures"] == 1 and data["table_entries"] == 8 and data["logical_capacity"] == 33,
            "Padded trace contract",
        )
        for field in (
            "sentinel_masked_bit_equal",
            "repeated_request_bit_equal",
            "changed_request_changed_output",
            "tracking",
        ):
            require(data[field] is True, f"Padded trace {field}")
        require(len(data["results"]) == 6, "Padded trace request count")
        require(
            [(r["request"], r["sentinel"]) for r in data["results"]]
            == [(0, 0), (0, 64), (1, 0), (1, 64), (0, 0), (0, 64)],
            "Padded trace requests/sentinels",
        )
        for row in data["results"]:
            require(row["guards_untouched"] is True and row["position"] == 32, "Padded trace guards")
            pcc(row["prefill_pcc"], "padded prefill")
            pcc(row["decode_pcc"], "padded decode")

    def pytest(self):
        path = self.doc / "real_weight_pytest.xml"
        xml = ET.parse(path).getroot()
        cases = xml.findall(".//testcase")
        wanted = {
            "test_optimized_target_shapes[linear_early]",
            "test_optimized_target_shapes[full_attention]",
            "test_optimized_target_shapes[linear_late]",
        }
        require(len(cases) == 3 and {case.attrib["name"] for case in cases} == wanted, "Pytest cases missing")
        require(
            all(
                case.attrib["classname"].endswith("test_optimized_decoder")
                and not any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
                for case in cases
            ),
            "Pytest failed/skipped/wrong module",
        )
        for layer in (0, 3, 34):
            paths = list((self.doc / "pytest").rglob(f"optimized_{layer}.json"))
            require(len(paths) == 1, f"Pytest current artifact for {layer} missing/ambiguous")
            data, _ = self.load(str(paths[0].relative_to(self.doc)))
            require(data["layer"] == layer and data["synthetic"] is False, "Pytest must use real weights")
            require(data["runtime_fallback_audit"] == "pass", "Pytest fallback")
            rows = data["results"]
            require(
                {r["length"] for r in rows if r["mode"] == "prefill"} == {1, 7, 31, 32, 33, 255, 256, 257, 513},
                "Pytest boundary coverage",
            )
            for row in rows:
                pcc(row["pcc"], f"pytest layer{layer}")
        self.checked.append(str(path.relative_to(self.doc)))

    def profile(self, layer, policy):
        directory = self.doc / f"tracy/layer_{layer}"
        data, provenance = self.load(f"tracy/layer_{layer}/correctness.json")
        require(
            data["layer"] == layer
            and data["kind"] == KINDS[layer]
            and data["length"] == 256
            and data["context"] == 512,
            "Profile shape",
        )
        require(
            data["profile_decode_replays"] == 16
            and data["deterministic"] is True
            and data["runtime_fallback_audit"] == "pass",
            "Profile correctness/replays",
        )
        require(data["precision_policy"] == policy, "Profile policy differs from final default benchmark")
        for key in ("prefill_pcc", "prefill_contribution_pcc", "decode_eager_pcc", "decode_traced_pcc"):
            pcc(data[key], f"profile layer{layer}/{key}")
        require(
            provenance["env"].get("TT_METAL_WATCHER") in OFF
            and provenance["env"].get("TT_METAL_DEVICE_PROFILER") == "1",
            "Profile instrumentation",
        )
        capture = (directory / "capture.log").read_text()
        require(
            "markers were dropped" not in capture
            and "Profiler DRAM buffers were full" not in capture
            and "TT_FATAL" not in capture,
            "Incomplete profile capture",
        )
        with (directory / "ops.csv").open() as handle:
            raw = list(csv.DictReader(handle))
        phases = {}
        for phase in ("prefill", "decode"):
            start_name = "PERF_" + phase.upper()
            starts = [i for i, r in enumerate(raw) if r["OP CODE"] == start_name]
            ends = [i for i, r in enumerate(raw) if r["OP CODE"] == start_name + "_END"]
            require(len(starts) == len(ends) == 1 and starts[0] < ends[0], f"Profile {phase} signposts")
            rows = [r for r in raw[starts[0] + 1 : ends[0]] if r["DEVICE ID"]]
            require(
                rows and all(r["DEVICE ID"] == "0" and positive(float(r["DEVICE KERNEL DURATION [ns]"])) for r in rows),
                f"Profile {phase} missing durations",
            )
            phases[phase] = rows
        sessions = defaultdict(list)
        for row in phases["decode"]:
            require(row["METAL TRACE ID"] and row["METAL TRACE REPLAY SESSION ID"], "Untraced measured profile")
            sessions[row["METAL TRACE REPLAY SESSION ID"]].append(row)
        require(len(sessions) == 16, "Exactly 16 complete measured trace sessions required")
        signatures = []
        for rows in sessions.values():
            ordered = sorted(rows, key=lambda r: int(r["DEVICE FW START CYCLE"]))
            signatures.append([(r["OP CODE"], r["PROGRAM HASH"]) for r in ordered])
        require(all(s == signatures[0] for s in signatures), "Trace replay operation sequence incomplete")
        report_summary = []
        dtype_names = {"bf16": "BFLOAT16", "bfp8": "BFLOAT8_B", "bfp4": "BFLOAT4_B"}
        for phase, raw_rows in phases.items():
            path = directory / f"{phase}_perf_report.csv"
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            require(rows and len(rows) == len(raw_rows), f"{path}: report/raw row count differs")
            require(
                all(positive(float(r["Device Time [μs]"])) and float(r["Device"]) == 0 for r in rows),
                f"{path}: durations/device",
            )
            require(all(float(r.get("Overlap [μs]") or 0) == 0 for r in rows), f"{path}: overlapped operations")
            require(
                math.isclose(
                    sum(float(r["Device Time [μs]"]) for r in rows),
                    sum(float(r["DEVICE KERNEL DURATION [ns]"]) / 1000 for r in raw_rows),
                    rel_tol=1e-5,
                    abs_tol=0.05,
                ),
                f"{path}: report/raw time mismatch",
            )
            require(
                (directory / f"{phase}_perf_report.txt").stat().st_size > 0 and "Advice" in rows[0],
                f"{path}: missing advice/table",
            )
            expected_roles = {"expert_gu": 0, "expert_down": 0, "shared_gu": 0, "shared_down": 0}
            if layer == 3:
                expected_roles.update(attention_packed=0, attention_out=0)
            for row in rows:
                op = row["OP Code"]
                role = None
                if op.startswith("SparseMatmulDeviceOperation"):
                    role = (
                        "expert_gu"
                        if op.endswith(" x 2048 x 1024")
                        else "expert_down"
                        if op.endswith(" x 512 x 2048")
                        else None
                    )
                elif op.startswith("GroupedMatmulDeviceOperation"):
                    role = "expert_gu" if " x 2048 x 65536" in op else "expert_down" if " x 512 x " in op else None
                elif op.startswith("MatmulDeviceOperation"):
                    role = (
                        "shared_gu"
                        if op.endswith(" x 2048 x 1056")
                        else "shared_down"
                        if op.endswith(" x 512 x 2048")
                        else None
                    )
                if layer == 3 and op.startswith("MatmulDeviceOperation"):
                    if op.endswith(" x 2048 x 9216"):
                        role = "attention_packed"
                    elif op.endswith(" x 4096 x 2048"):
                        role = "attention_out"
                if role is None:
                    continue
                expected_roles[role] += 1
                if role.startswith("attention"):
                    dtype = policy["projection_candidate"]["attention_dtype"]
                    fidelity = policy["projection_candidate"]["attention_fidelity"]
                elif role.startswith("shared"):
                    dtype, fidelity = policy["shared_dtype"], policy["shared_fidelity"]
                elif phase == "prefill":
                    dtype = (
                        policy["grouped_prefill_dtype"]
                        if role == "expert_gu"
                        else policy.get("grouped_prefill_down_dtype", policy["grouped_prefill_dtype"])
                    )
                    fidelity = policy["expert_fidelity"]
                else:
                    dtype = (
                        policy["expert_dtype"]
                        if role == "expert_gu"
                        else policy.get("expert_down_dtype", policy["expert_dtype"])
                    )
                    fidelity = policy["expert_fidelity"]
                require(row["Input 1 Datatype"] == dtype_names[dtype], f"{path}: {role} actual dtype mismatch")
                require(row["Math Fidelity"].startswith(fidelity + " "), f"{path}: {role} actual fidelity mismatch")
            require(all(expected_roles.values()), f"{path}: missing selected precision roles {expected_roles}")
            report_summary.append(
                dict(
                    phase=phase,
                    rows=len(rows),
                    device_us=sum(float(r["Device Time [μs]"]) for r in rows) / (16 if phase == "decode" else 1),
                    measured_precision_roles=expected_roles,
                )
            )
            self.checked.append(str(path.relative_to(self.doc)))
        return dict(
            layer=layer,
            measured_replays=16,
            identical_operation_sequence=True,
            operations_per_replay=len(signatures[0]),
            reports=report_summary,
        )

    def run(self):
        comparisons, profiles = [], []
        for layer in (0, 3):
            final, fp = self.benchmark(f"final_{layer}.json", layer)
            baseline, bp = self.benchmark(f"baseline_{layer}.json", layer, optimized=False)
            require(
                final["length"] == baseline["length"] == 256 and final["context"] == baseline["context"] == 512,
                "Matched headline shape",
            )
            require(
                fp["device"] == bp["device"] and fp["versions"] == bp["versions"], "Baseline/final environment mismatch"
            )
            require(
                final["decode_traced_warmed_ms"] < baseline["decode_traced_warmed_ms"],
                f"Layer {layer} final does not beat fused traced decode",
            )
            comparisons.append(
                dict(
                    layer=layer,
                    baseline_prefill_ms=baseline["prefill_warmed_ms"],
                    final_prefill_ms=final["prefill_warmed_ms"],
                    baseline_decode_ms=baseline["decode_traced_warmed_ms"],
                    final_decode_ms=final["decode_traced_warmed_ms"],
                )
            )
            for length in (127, 128, 129, 257, 4095, 4096, 4097):
                baseline_boundary, _ = self.benchmark(
                    f"baseline_l{layer}_s{length}.json", layer, optimized=False, headline=False
                )
                require(
                    baseline_boundary["length"] == length and baseline_boundary["context"] == 8192,
                    "Matched boundary baseline",
                )
                boundary, _ = self.benchmark(f"final_l{layer}_s{length}.json", layer, headline=False)
                require(boundary["length"] == length and boundary["context"] == 8192, "Boundary benchmark shape")
            self.battery(layer)
            self.trace_and_watcher(layer)
            self.context(layer)
            profiles.append(self.profile(layer, final["precision_policy"]))
        for layer in (0, 3, 34):
            for length in (128, 1024):
                self.stress(layer, length)
        self.page_checks()
        self.pytest()
        natural = []
        for layer in (0, 3, 34):
            data, _ = self.benchmark(f"natural_final_{layer}.json", layer, headline=False)
            require(Path(data["activation_source"]).name == "natural257.pt", "Actual activation source missing")
            natural.append(dict(layer=layer, prefill_pcc=data["prefill_pcc"], decode_pcc=data["decode_traced_pcc"]))
        from models.autoports.qwen_qwen3_6_35b_a3b.tests.summarize_optimized_perf import functional_sequence

        functional = []
        for layer in (0, 3):
            data, _ = self.load(f"functional_{layer}.json", clean=True)
            require(data["gate_pass"] and data["failed_rows"] == [], "FUNCTIONAL correctness failed")
            for key in ("prefill_pcc", "decode_whole_pcc", "continuation_whole_pcc"):
                pcc(data[key], "FUNCTIONAL " + key)
            row = functional_sequence(self.doc, layer)
            require(row["status"] == "pass", "FUNCTIONAL 10% theoretical floor failed")
            functional.append(row)
        # BFP4 is measured only as a candidate; default remains BFP8.
        candidate = read_json(self.doc / "tracy/bfp4_candidate/correctness.json")
        require(candidate["decoder"] == "OptimizedDecoder" and candidate["deterministic"], "BFP4 candidate path")
        pcc(candidate["prefill_pcc"], "BFP4 candidate prefill")
        pcc(candidate["decode_traced_pcc"], "BFP4 candidate decode")
        with (self.doc / "tracy/bfp4_candidate/decode_perf_report.csv").open() as handle:
            sparse = [r for r in csv.DictReader(handle) if r["OP Code"].startswith("SparseMatmulDeviceOperation")]
        require(len(sparse) == 32, "BFP4 candidate missing expert rows")
        require(
            all(r["Input 1 Datatype"] == "BFLOAT4_B" and r["Math Fidelity"].startswith("LoFi ") for r in sparse),
            "BFP4/LoFi did not reach measured experts",
        )
        return dict(
            status="PASS",
            gate_pass=True,
            source_sha256=self.sources,
            checked_artifacts=self.checked,
            before_after=comparisons,
            profiler_completeness=profiles,
            natural_activations=natural,
            functional_128_128=functional,
            bfp4_lofi_candidate_measured=True,
            scope="Current optimized decoder local hard gates; not stage completion",
            remaining_owner_checks=[
                "Rank final defaults against every best correct candidate; fixed baseline comparison alone does not prove this.",
                "Independent stage review, optimization stop-rule/checklist evidence and local commit receipt remain owner responsibilities.",
            ],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    args = parser.parse_args()
    try:
        result = Audit(args.doc).run()
    except (AssertionError, KeyError, ValueError, TypeError, OSError, ET.ParseError) as error:
        print(f"OPTIMIZED_HARD_GATES_FAIL: {error}", file=sys.stderr)
        return 1
    path = args.doc / "hard_gates.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"OPTIMIZED_HARD_GATES_PASS: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
