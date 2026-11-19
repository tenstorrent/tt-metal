# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only stage-5 gates, preserving the strict stage-4 acceptance checks.

Run after final validation and profiler artifacts have been collected. Historical
before controls retain their recorded source identity and may load a saved
single-chip control. Final artifacts must match the current default mesh path.
The timing table compares whole-batch multichip latency, never per-user latency
or the saved single-chip control's timing.
"""

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.summarize_multichip_validation import (
    ENV_KEYS,
    PCC_MIN,
    Gate,
    numeric,
    off,
    option,
    require,
    sequence,
)

DOC = Path(__file__).parents[1] / "doc/optimized_multichip_decoder"
MESH_DEVICE = "Blackhole 1x4 target; optimized baseline 1x1 physical chip 0"
MESH_REGIME = "serial; chips 0,1,2,3"
TIMING_SCOPE = "whole-batch decoder latency; all four chips; serial; milliseconds"
PHASES = (("prefill", "prefill_warmed_ms"), ("decode", "decode_traced_warmed_ms"))


class HistoricalGate(Gate):
    """Use unchanged paired/natural gates with historical provenance rules only."""

    def provenance(self, name, *, tracking=False, clean=False, watcher=False, profiler=False):
        require(clean and not any((tracking, watcher, profiler)), "before controls must be clean timing")
        data = self.read(str(name) + ".provenance.json")
        env = data.get("env")
        require(isinstance(env, dict) and set(ENV_KEYS) <= env.keys(), f"{name}: incomplete environment")
        require(all(off(env[key]) for key in ENV_KEYS[:3]), f"{name}: instrumented before timing")
        require(data.get("regime") == "serial", f"{name}: not serial provenance")
        require(data.get("execution") == "TTNN device", f"{name}: not device execution")
        require(data.get("device") == MESH_DEVICE, f"{name}: wrong target mesh")
        require(data.get("decoder") == "multichip", f"{name}: not multichip provenance")
        identities = data.get("source_sha256")
        require(isinstance(identities, dict), f"{name}: missing source identities")
        for source in ("tt/multichip_decoder.py", "tt/optimized_decoder.py"):
            require(
                isinstance(identities.get(source), str) and re.fullmatch(r"[0-9a-f]{64}", identities[source]),
                f"{name}: missing historical {source} identity",
            )
        command = data.get("command")
        require(isinstance(command, list) and command, f"{name}: missing command")
        require("--baseline-only" not in command and "--profile" not in command, f"{name}: wrong execution mode")
        require(json.loads(option(command, "--mesh-build-kwargs") or "{}") == {}, f"{name}: nondefault before policy")
        require(option(command, "--mesh-projection-geometry") is None, f"{name}: experimental before geometry")
        saved_baseline = option(command, "--baseline-load")
        if saved_baseline:
            self.evidence(saved_baseline)
        return data


class OptimizedGate(Gate):
    def __init__(self, doc):
        super().__init__(doc)
        self.timing_rows = []
        self.before = HistoricalGate(doc)

    def linear_performance_acceptance(self):
        """Owner acceptance is bound to one recorded result, never a quality waiver."""
        record = self.read(str(self.doc.parent / "eval_readiness/limitations.json"))
        require(record.get("model") == "Qwen/Qwen3.6-35B-A3B", "wrong limitation model")
        rows = [r for r in record.get("limitations", []) if r.get("id") == "stage5-linear-decode-performance"]
        require(len(rows) == 1, "missing or ambiguous owner performance decision")
        accepted = rows[0]
        for key, value in dict(
            stage="optimized-multichip-decoder",
            category="performance",
            status="owner-accepted",
            original_model="not_applicable",
            gate="functional_l0",
            layer=0,
            kind="linear_attention",
            batch=1,
            mesh=[1, 4],
            evidence_artifact="restored_functional_l0.json",
            required_floor_fraction=0.1,
            required_max_decode_ms_per_token=0.518283125,
            raw_floor_passed=False,
        ).items():
            require(accepted.get(key) == value, f"owner decision: invalid {key}")
        decision = accepted.get("owner_decision", {})
        require(decision.get("date") == "2026-11-14", "missing owner decision date")
        require(decision.get("further_tuning") == "not_requested", "incorrect tuning decision")
        quote = (
            "Owner decision: accept the linear-attention decode speed (0.7195 ms/token vs the 0.5183 ms floor, "
            "7.2% of roofline) as a known limitation. Record it in limitations with this decision and the evidence, "
            "finish the review and complete stage 5. No further tuning for this floor."
        )
        require(decision.get("text") == quote, "owner decision text missing or changed")
        require(quote in self.evidence("owner_decision.md").read_text(), "missing recorded owner decision")
        for item in sequence(accepted.get("evidence"), "limitation evidence"):
            self.evidence(str(self.doc.parent / item))
        return accepted

    def paired_result_status(self, name, data):
        if name != "restored_functional_l0.json":
            return super().paired_result_status(name, data)
        self.linear_performance_acceptance()
        require(data.get("passed") is False, f"{name}: raw failed result was changed")
        require(
            data.get("multichip", {}).get("functional_sequence", {}).get("passed") is False,
            f"{name}: raw floor failure was changed",
        )
        # paired() still checks every PCC/state/trace/fallback/mesh condition.
        # functional() below validates the complete sequence and exact accepted result.

    def unchanged_kv_policy(self):
        """Stage 5 uses 32K smokes only while the selected KV policy stays BF16."""
        source = self.evidence(str(self.doc.parents[1] / "tt/multichip_decoder.py"))
        tree = ast.parse(source.read_text())
        constructors = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "from_state_dict"
            and any(arg.arg == "kv_dtype" for arg in node.args.kwonlyargs + node.args.args)
        ]
        require(len(constructors) == 1, "ambiguous or missing selected KV constructor")
        args = constructors[0].args
        defaults = dict(zip((arg.arg for arg in args.args[-len(args.defaults) :]), args.defaults))
        defaults.update(zip((arg.arg for arg in args.kwonlyargs), args.kw_defaults))
        require(ast.literal_eval(defaults["kv_dtype"]) == "bf16", "KV default changed: maximum-context gate required")
        for batch in (1, 32):
            name = f"final_l3_b{batch}.json"
            data = self.read(name)
            require(
                data.get("multichip", {}).get("precision_policy", {}).get("kv") == "bf16",
                f"{name}: KV policy changed: maximum-context gate required",
            )
            self.provenance(name, clean=True)
        plan = self.read(str(self.doc.parent / "context_contract.json")).get("multichip_decoder_plan", {})
        require(plan.get("kv_cache_dtype") == "BF16", "context contract no longer uses BF16 KV")
        require(plan.get("kv_head_ownership") == [0, 0, 1, 1], "KV head ownership changed")
        return {
            "kv_dtype": "bf16",
            "max_context_rerun": False,
            "reason": "Selected cache dtype/head layout unchanged; both 32K smoke gates retained",
        }

    def check(self, name, callback):
        if name == "context_fullmax":
            # This is the sole stage-specific replacement of the stage-4 gate
            # set. A changed default fails explicitly instead of silently skipping
            # maximum-context validation or accepting a stale stage-4 max run.
            return super().check("unchanged_kv_policy", self.unchanged_kv_policy)
        return super().check(name, callback)

    def functional(self, layer):
        from models.autoports.qwen_qwen3_6_35b_a3b.tests.multichip_functional import sequence_bound

        name = f"restored_functional_l{layer}.json"
        result, provenance = self.paired(name, layer, 1, length=128, steps=128, clean=True)
        require(result.get("stack_depth") == 1, f"{name}: wrong stack depth")
        require(result["multichip"].get("regime") == MESH_REGIME, f"{name}: not target four-chip mesh")
        require(provenance.get("device") == MESH_DEVICE, f"{name}: wrong mesh provenance")
        require("--functional-sequence" in provenance["command"], f"{name}: FUNCTIONAL not enabled")
        for source in ("tests/run_multichip_decoder.py", "tests/multichip_functional.py"):
            path = self.evidence(str(self.doc.parents[1] / source))
            require(
                provenance["source_sha256"].get(source) == hashlib.sha256(path.read_bytes()).hexdigest(),
                f"{name}: stale FUNCTIONAL runner or byte model",
            )
        row = result["multichip"].get("functional_sequence", {})
        for key, value in dict(
            prompt_length=128,
            decode_length=128,
            batch=1,
            mesh=[1, 4],
            positions=list(range(128, 256)),
            trace_capture_count=1,
            repeated_bit_equal=True,
            all_steps_eager_trace_equal=True,
            input_and_position_refresh_each_step=True,
            state_reset_between_repeats=True,
        ).items():
            require(row.get(key) == value, f"{name}: invalid {key}")
        repeats = row.get("repeats")
        require(
            isinstance(repeats, int) and not isinstance(repeats, bool) and repeats >= 2,
            f"{name}: missing repeated changing-input sequence",
        )
        require(
            repeats == int(option(provenance["command"], "--functional-repeats") or 3),
            f"{name}: repeat provenance mismatch",
        )
        samples = sequence(row.get("samples_ms"), f"{name} functional samples", 128 * repeats)
        for sample in samples:
            require(numeric(sample, "FUNCTIONAL sample") > 0, f"{name}: nonpositive latency")
        mean = statistics.mean(samples)
        require(
            math.isclose(numeric(row.get("decode_ms_per_token_mean"), "FUNCTIONAL mean"), mean, rel_tol=1e-9),
            f"{name}: mean disagrees with samples",
        )
        expected = sequence_bound(layer)
        require(row.get("theoretical") == expected, f"{name}: incorrect per-rank theoretical model")
        fraction = expected["roofline_ms_per_token"] / mean
        require(
            math.isclose(numeric(row.get("roofline_fraction"), "FUNCTIONAL fraction"), fraction, rel_tol=1e-9),
            f"{name}: incorrect theoretical fraction",
        )
        require(row.get("floor_fraction") == 0.1, f"{name}: FUNCTIONAL target changed")
        floor_passed = fraction >= 0.1
        require(row.get("passed") is floor_passed, f"{name}: raw floor status inconsistent")
        acceptance = None
        if layer == 0:
            acceptance = self.linear_performance_acceptance()
            require(not floor_passed, f"{name}: owner exception no longer applies")
            for key, value in (("measured_decode_ms_per_token", mean), ("roofline_fraction", fraction)):
                require(
                    math.isclose(numeric(acceptance.get(key), key), value, rel_tol=1e-12),
                    f"{name}: result differs from owner-accepted measurement",
                )
            require(
                acceptance.get("decoder_sha256") == provenance["source_sha256"]["tt/multichip_decoder.py"],
                f"{name}: decision refers to a different decoder",
            )
            require(
                acceptance.get("native_source_sha256") == provenance.get("native_source_sha256"),
                f"{name}: decision refers to different native sources",
            )
            loaded = {key: value.get("sha256") for key, value in provenance.get("loaded_native_binaries", {}).items()}
            require(loaded and acceptance.get("loaded_native_sha256") == loaded, f"{name}: different native build")
        else:
            require(floor_passed, f"{name}: FUNCTIONAL >=10% floor failed ({fraction:.6%})")
        for source, identity in provenance.get("native_source_sha256", {}).items():
            require(hashlib.sha256(self.evidence(source).read_bytes()).hexdigest() == identity, f"stale {source}")
        require(provenance.get("loaded_native_binaries"), f"{name}: missing native provenance")
        for path, identity in provenance["loaded_native_binaries"].items():
            require(
                hashlib.sha256(self.evidence(path).read_bytes()).hexdigest() == identity.get("sha256"),
                f"{name}: loaded native build differs from current {path}",
            )
        require(
            isinstance(row.get("timing_boundary"), str) and "blocking trace replay only" in row["timing_boundary"],
            f"{name}: missing timing boundary",
        )
        policy = result["multichip"].get("precision_policy", {})
        require(
            policy.get("kv") == "bf16" and policy.get("router") == "FP32",
            f"{name}: theoretical precision contract mismatch",
        )
        for group in ("attention", "shared"):
            require(
                policy.get("projection_precision", {}).get(group, {}).get("dtype", "bfp8") == "bfp8",
                f"{name}: theoretical projection dtype changed",
            )
        return dict(
            roofline_fraction=fraction,
            decode_ms_per_token_mean=mean,
            roofline_ms_per_token=expected["roofline_ms_per_token"],
            mesh=[1, 4],
            timing_boundary=row["timing_boundary"],
            evidence_artifact=name,
            raw_floor_passed=floor_passed,
            disposition="owner-accepted-performance-limitation" if acceptance else "measured-pass",
            limitation_id=acceptance["id"] if acceptance else None,
        )

    def compare_timing(self, layer, batch):
        before_name = f"before_l{layer}_b{batch}.json"
        after_name = f"final_l{layer}_b{batch}.json"
        # The historical exception is restricted to before controls. All final
        # checks (including current source/default identity) use the original Gate.
        try:
            if batch == 32:
                self.before.natural(layer, name=before_name, tracking=False, clean=True)
                self.natural(layer, name=after_name, tracking=False, clean=True)
            before, before_provenance = self.before.paired(before_name, layer, batch, clean=True)
        finally:
            self.sources.update(self.before.sources)
        after, after_provenance = self.paired(after_name, layer, batch, clean=True)
        for field in (
            "layer",
            "batch",
            "length",
            "context",
            "decode_steps",
            "stack_depth",
            "seed",
            "activation_source",
        ):
            require(before.get(field) == after.get(field), f"{after_name}: incomparable before/after {field}")
        for name, result, provenance in (
            (before_name, before, before_provenance),
            (after_name, after, after_provenance),
        ):
            require(provenance.get("device") == MESH_DEVICE, f"{name}: wrong target mesh")
            require(provenance.get("decoder") == "multichip", f"{name}: wrong decoder provenance")
            require(result["multichip"].get("regime") == MESH_REGIME, f"{name}: missing four-chip identity")
            require(result["multichip"].get("changing_inputs") is True, f"{name}: static decode inputs")
            command = provenance["command"]
            require(int(option(command, "--layer")) == layer, f"{name}: command layer mismatch")
            require(int(option(command, "--batch") or 1) == batch, f"{name}: command batch mismatch")
            require(int(option(command, "--length")) == result["length"], f"{name}: command length mismatch")
            output = option(command, "--output")
            require(output and self.evidence(output) == self.evidence(name), f"{name}: output provenance mismatch")
            for mode, field in PHASES:
                samples = sequence(result["multichip"].get(mode + "_samples_ms"), f"{name} {mode} samples")
                for sample in samples:
                    require(numeric(sample, "sample") > 0, f"{name}: nonpositive sample")
                median = numeric(result["multichip"].get(field), f"{name} {mode} median")
                require(
                    median > 0 and math.isclose(median, statistics.median(samples), rel_tol=1e-9),
                    f"{name}: invalid median",
                )
        rows = []
        for mode, field in PHASES:
            old_ms, new_ms = before["multichip"][field], after["multichip"][field]
            rows.append(
                dict(
                    layer=layer,
                    kind=after["multichip"]["kind"],
                    batch=batch,
                    logical_length=after["length"],
                    context=after["context"],
                    phase="warmed_prefill" if mode == "prefill" else "traced_warmed_decode",
                    before_ms=old_ms,
                    after_ms=new_ms,
                    speedup=old_ms / new_ms,
                    latency_reduction_percent=100 * (old_ms - new_ms) / old_ms,
                    duration_scope=TIMING_SCOPE,
                    before_artifact=str(self.evidence(before_name)),
                    after_artifact=str(self.evidence(after_name)),
                    before_decoder_sha256=before_provenance["source_sha256"]["tt/multichip_decoder.py"],
                    after_decoder_sha256=after_provenance["source_sha256"]["tt/multichip_decoder.py"],
                )
            )
        self.timing_rows.extend(rows)
        return {"layer": layer, "batch": batch, "phases": rows}

    def watcher_paths(self):
        """Watcher coverage for nonaligned resume and the one-token padding path."""
        name = "watcher_mixed_non_aligned.json"
        mixed = self.mixed(name, watcher=True)
        data = self.read(name)
        require(data.get("prefix_lengths") == [33, 35], f"{name}: wrong prefix boundaries")
        require(data.get("continuation_lengths") == [32, 34], f"{name}: wrong resumed boundaries")
        require(data.get("decode_steps") == 4, f"{name}: wrong decode coverage")
        provenance = self.provenance(name, tracking=True, watcher=True)
        require(
            provenance.get("device") == "serial 1x1 chip 0 baseline then 1x4 Blackhole mesh",
            f"{name}: wrong mesh provenance",
        )
        records = [(name, "mixed_non_aligned", provenance)]
        for layer in (0, 3):
            name = f"watcher_l{layer}_s1.json"
            data, provenance = self.paired(name, layer, 1, length=1, steps=4, tracking=True, watcher=True)
            require(data["multichip"].get("regime") == MESH_REGIME, f"{name}: wrong mesh regime")
            require(provenance.get("device") == MESH_DEVICE, f"{name}: wrong mesh provenance")
            records.append((name, f"l{layer}_s1", provenance))
        logs = {}
        for name, directory, provenance in records:
            log = f"watcher/{directory}/generated/watcher/watcher.log"
            log_root = provenance["env"].get("TT_METAL_LOGS_PATH")
            require(isinstance(log_root, str) and log_root, f"{name}: missing watcher log root")
            require(
                self.evidence(str(Path(log_root) / "generated/watcher/watcher.log")) == self.evidence(log),
                f"{name}: watcher log provenance mismatch",
            )
            logs[name] = self.watcher_log(log)
        return {"mixed_lengths": mixed["lengths"], "watcher_logs": logs}

    def run(self):
        self.check("watcher_non_aligned_paths", self.watcher_paths)
        for layer in (0, 3):
            self.check(f"functional_l{layer}", lambda layer=layer: self.functional(layer))
        for layer in (0, 3):
            for batch in (1, 32):
                self.check(
                    f"before_after_l{layer}_b{batch}",
                    lambda layer=layer, batch=batch: self.compare_timing(layer, batch),
                )
        # Retain stage-4 correctness, 32K ownership, allocation controls,
        # watcher, natural B32 and raw CSV checks. See check() for the explicit
        # unchanged-KV policy replacing stage-4's maximum-context rerun.
        report, path = super().run()
        csv_path = self.doc / "before_after.csv"
        columns = (
            "layer",
            "kind",
            "batch",
            "logical_length",
            "context",
            "phase",
            "before_ms",
            "after_ms",
            "speedup",
            "latency_reduction_percent",
            "duration_scope",
            "before_artifact",
            "after_artifact",
            "before_decoder_sha256",
            "after_decoder_sha256",
        )
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.timing_rows)
        report.update(
            stage="optimized-multichip-decoder",
            passed=all(row["passed"] for row in self.results),
            gates=self.results,
            evidence_sources=sorted(self.sources),
            before_after_csv=str(csv_path),
            timing_scope=TIMING_SCOPE,
            owner_accepted_limitations=[
                row["details"]["limitation_id"]
                for row in self.results
                if row["passed"]
                and isinstance(row.get("details"), dict)
                and row["details"].get("disposition") == "owner-accepted-performance-limitation"
            ],
            scope="Stage-5 recorded hard gates plus before/after mesh timing; independent review and checkpoint follow",
        )
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DOC)
    args = parser.parse_args()
    report, path = OptimizedGate(args.doc).run()
    failures = [row for row in report["gates"] if not row["passed"]]
    print(
        f"OPTIMIZED_MULTICHIP_GATES {'PASS' if report['passed'] else 'FAIL'}: "
        f"{len(report['gates']) - len(failures)}/{len(report['gates'])}; PCC >= {PCC_MIN}; {path}"
    )
    for row in failures:
        print(f"{row['gate']}: {row['error']}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
