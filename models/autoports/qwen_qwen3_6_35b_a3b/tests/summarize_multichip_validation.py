# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed CPU-only stage gates over fixed multichip acceptance artifacts.

Missing fields, missing artifacts, nonfinite metrics and inconsistent summaries
fail the gate. This checks recorded execution evidence; it neither runs devices
nor substitutes for independent stage review. A failure summary is also emitted.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

DOC = Path(__file__).parents[1] / "doc/multichip_decoder"
REPO = Path(__file__).resolve().parents[4]
PCC_MIN = 0.995
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
ENV_KEYS = (
    "TT_METAL_WATCHER",
    "TT_METAL_DEVICE_PROFILER",
    "TT_METAL_TRACE_ALLOC_TRACKING",
    "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
)
WATCHER_BAD = re.compile(
    r"\bTT_FATAL\b|\bASSERT(?:ION)?(?:\s+FAILED)?\b|corrupt|Watcher detected|"
    r"invalid[^\n]*(?:address|NoC|NOC)|illegal[^\n]*(?:address|access)|"
    r"(?:device.side|watcher)[^\n]*(?:error|failure)|sanitization[^\n]*failed",
    re.IGNORECASE,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value, label, minimum=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label}: missing numeric value")
    require(math.isfinite(value), f"{label}: nonfinite value")
    if minimum is not None:
        require(value >= minimum, f"{label}: {value} < {minimum}")
    return float(value)


def pcc(value, label):
    result = numeric(value, label, PCC_MIN)
    require(result <= 1.000000000001, f"{label}: PCC exceeds 1")
    return result


def sequence(value, label, count=None):
    require(isinstance(value, list) and value, f"{label}: missing nonempty list")
    if count is not None:
        require(len(value) == count, f"{label}: expected {count} entries, got {len(value)}")
    return value


def off(value):
    return value in (None, "", "0", 0, False)


def option(command, name):
    require(isinstance(command, list) and command, "provenance command is missing")
    for index, item in enumerate(command):
        if item == name:
            require(index + 1 < len(command), f"{name}: missing command argument")
            return command[index + 1]
        if isinstance(item, str) and item.startswith(name + "="):
            return item.split("=", 1)[1]
    return None


class Gate:
    def __init__(self, doc):
        self.doc = doc.resolve()
        self.sources = set()
        self.results = []

    def evidence(self, value):
        require(isinstance(value, str) and value, "missing evidence path")
        path = Path(value)
        candidates = [path] if path.is_absolute() else [self.doc / path, REPO / path]
        found = next((p for p in candidates if p.is_file() and p.stat().st_size > 0), None)
        require(found is not None, f"missing/empty evidence: {value}")
        found = found.resolve()
        self.sources.add(str(found))
        return found

    def read(self, name):
        path = self.evidence(str(name))
        data = json.loads(path.read_text())
        require(isinstance(data, dict), f"{path}: expected JSON object")
        return data

    def provenance(self, name, *, tracking=False, clean=False, watcher=False, profiler=False):
        data = self.read(str(name) + ".provenance.json")
        env = data.get("env")
        require(isinstance(env, dict) and set(ENV_KEYS) <= env.keys(), f"{name}: incomplete environment provenance")
        require(data.get("regime") == "serial", f"{name}: not serial provenance")
        require(data.get("execution") == "TTNN device", f"{name}: not device execution")
        require(
            isinstance(data.get("source_sha256"), dict) and data["source_sha256"].get("tt/multichip_decoder.py"),
            f"{name}: no decoder source identity",
        )
        current_source = DOC.parents[1] / "tt/multichip_decoder.py"
        require(
            data["source_sha256"]["tt/multichip_decoder.py"] == hashlib.sha256(current_source.read_bytes()).hexdigest(),
            f"{name}: decoder source identity differs from the final implementation",
        )
        command = data.get("command")
        require(isinstance(command, list) and command, f"{name}: missing command")
        require("--baseline-only" not in command, f"{name}: baseline-only is not mesh evidence")
        require(json.loads(option(command, "--mesh-build-kwargs") or "{}") == {}, f"{name}: nondefault mesh policy")
        require(option(command, "--mesh-projection-geometry") is None, f"{name}: experimental projection geometry")
        if tracking:
            require(env["TT_METAL_TRACE_ALLOC_TRACKING"] == "1", f"{name}: allocation tracking not enabled")
            require(
                env["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] != "1", f"{name}: program-cache allocations excluded"
            )
        if clean:
            require(
                all(off(env[key]) for key in ENV_KEYS[:3]),
                f"{name}: clean timing has watcher/profiler/tracking enabled",
            )
            require(option(command, "--baseline-load") is None, f"{name}: clean timing loaded an unmatched baseline")
            require("--profile" not in command, f"{name}: clean timing used profiling")
        if watcher:
            require(env["TT_METAL_WATCHER"] == "10", f"{name}: watcher interval is not 10")
        if profiler:
            require(env["TT_METAL_DEVICE_PROFILER"] == "1", f"{name}: profiler not enabled")
            require(off(env["TT_METAL_WATCHER"]), f"{name}: watcher mixed with profiling")
        else:
            require(off(env["TT_METAL_DEVICE_PROFILER"]), f"{name}: unexpected profiler")
            require("--profile" not in command, f"{name}: unexpected profile mode")
        return data

    def paired_result_status(self, name, data):
        require(data.get("passed") is True, f"{name}: result not passed")

    def paired(self, name, layer, batch, *, length=None, steps=None, tracking=False, clean=False, watcher=False):
        data = self.read(name)
        self.paired_result_status(name, data)
        require(data.get("layer") == layer and data.get("batch") == batch, f"{name}: layer/batch mismatch")
        if length is not None:
            require(data.get("length") == length, f"{name}: logical length mismatch")
        count = data.get("decode_steps")
        require(isinstance(count, int) and count >= 1, f"{name}: no decode steps")
        if steps is not None:
            require(count == steps, f"{name}: decode count must be {steps}")
        pcc(data.get("prefill_pcc"), name + " prefill")
        for value in sequence(data.get("decode_pcc"), name + " decode", count):
            pcc(value, name + " decode")
        checks = sequence(data.get("state_checks"), name + " states")
        state_names = {"conv", "recurrent"} if layer == 0 else {"k", "v"}
        expected = {
            (depth, row, key)
            for depth in range(data.get("stack_depth", 1))
            for row in range(batch)
            for key in state_names
        }
        observed = {(c.get("stack_index"), c.get("row"), c.get("tensor")) for c in checks}
        require(observed == expected and len(checks) == len(expected), f"{name}: incomplete state coverage")
        for check in checks:
            pcc(check.get("pcc"), name + " state")
        for side, decoder in (("baseline", "OptimizedDecoder"), ("multichip", "MultichipDecoder")):
            row = data.get(side, {})
            require(row.get("decoder") == decoder, f"{name}: incorrect {side} decoder")
            if side == "multichip":
                require(row.get("residual_layout") == "replicated", f"{name}: nonselected residual layout")
                require(row.get("output_replicas_equal") is True, f"{name}: replica equality missing/failed")
                if batch == 32:
                    require(
                        row.get("empty_expert_ranks_zero_after_poison") is True, f"{name}: empty expert ranks untested"
                    )
            require(
                row.get("kind") == ("linear_attention" if layer == 0 else "full_attention"), f"{name}: kind mismatch"
            )
            require(row.get("runtime_fallback_audit") == "pass", f"{name}: {side} fallback audit failed")
            for flag in ("deterministic", "all_step_eager_trace_equal", "live_trace_prefill_replay_equal"):
                require(row.get(flag) is True, f"{name}: {side} {flag} missing/false")
            require(row.get("trace_capture_count") == 1, f"{name}: {side} unexpected recapture")
            numeric(row.get("replay_count"), f"{name} {side} replays", 2 * count)
            require(
                row.get("positions") == list(range(data["length"], data["length"] + count)),
                f"{name}: current positions mismatch",
            )
            if clean:
                for mode, field in (("prefill", "prefill_warmed_ms"), ("decode", "decode_traced_warmed_ms")):
                    samples = sequence(row.get(mode + "_samples_ms"), f"{name} {side} {mode} samples")
                    for sample in samples:
                        require(numeric(sample, "latency") > 0, f"{name}: nonpositive latency")
                    median = numeric(row.get(field), f"{name} {side} median")
                    require(math.isclose(median, statistics.median(samples), rel_tol=1e-9), f"{name}: median mismatch")
        if clean:
            require(data.get("trace_allocation_tracking") is False, f"{name}: clean result claims allocation tracking")
            for mode, field in (("prefill", "prefill_warmed_ms"), ("decode", "decode_traced_warmed_ms")):
                ratio = data["baseline"][field] / data["multichip"][field]
                require(
                    math.isclose(numeric(data.get(mode + "_speedup"), "speedup"), ratio, rel_tol=1e-9),
                    f"{name}: speedup mismatch",
                )
                require(
                    math.isclose(numeric(data.get(mode + "_efficiency"), "efficiency"), ratio / 4, rel_tol=1e-9),
                    f"{name}: efficiency mismatch",
                )
        if tracking:
            require(data.get("trace_allocation_tracking") is True, f"{name}: tracking result missing/false")
        self.evidence(data.get("tensor_artifact"))
        provenance = self.provenance(name, tracking=tracking, clean=clean, watcher=watcher)
        return data, provenance

    def natural(self, layer, *, name=None, tracking=True, clean=False):
        name = name or f"natural_l{layer}_b32.json"
        data, provenance = self.paired(name, layer, 32, tracking=tracking, clean=clean)
        source = data.get("activation_file")
        path = self.evidence(source)
        require(data.get("activation_source") == source, f"{name}: inconsistent activation source")
        declared = option(provenance["command"], "--activation-file")
        require(declared and self.evidence(declared) == path, f"{name}: command lacks matching real activation payload")
        metadata = self.read(str(path.with_suffix(".metadata.json")))
        require(
            metadata.get("checkpoint_revision") == REVISION and metadata.get("batch") == 32,
            f"{name}: activation checkpoint/batch mismatch",
        )
        require(
            metadata.get("generation_performed") is False and metadata.get("padding_or_repetition") is False,
            f"{name}: invalid teacher-forced provenance",
        )
        prompts = sequence(metadata.get("source_prompts"), "source prompts", 32)
        require(len(set(prompts)) == 32, f"{name}: repeated prompts")
        token_ids = sequence(metadata.get("input_ids"), "source token IDs", 32)
        require(len({tuple(row) for row in token_ids}) == 32, f"{name}: repeated token rows")
        require(
            metadata.get("prefill_length") == data["length"]
            and metadata.get("decode_steps", 0) >= data["decode_steps"],
            f"{name}: activation shape mismatch",
        )
        shape = metadata.get("tensor_shapes", {}).get(str(layer), {})
        require(shape.get("x") == [32, data["length"], 2048], f"{name}: missing real batch input geometry")
        require(shape.get("xd") == [metadata["decode_steps"], 32, 1, 2048], f"{name}: missing held-out token geometry")
        return {"activation_metadata": str(path.with_suffix(".metadata.json"))}

    def mixed(self, name="mixed_stack_selected.json", *, watcher=False):
        data = self.read(name)
        require(
            data.get("passed") is True and data.get("layers") == [0, 3], f"{name}: missing passing mixed-layer result"
        )
        require(data.get("trace_allocation_tracking") is True, f"{name}: tracking result missing")
        prefix = sequence(data.get("prefix_lengths"), "prefix lengths")
        continuation = sequence(data.get("continuation_lengths"), "continuation lengths", len(prefix))
        require(any(n % 32 for n in prefix), f"{name}: no nonaligned resume boundary")
        lengths = [a + b for a, b in zip(prefix, continuation)]
        require(len(set(lengths)) > 1, f"{name}: missing heterogeneous positions")
        for field in ("output_comparisons", "state_checks"):
            for row in sequence(data.get(field), f"{name} {field}"):
                pcc(row.get("pcc"), f"{name} {field}")
        expected_comparisons = {
            (phase, mode, step)
            for phase in (0, 1)
            for mode, step in [("prefix", None), ("continuation", None)]
            + [("decode", s) for s in range(data["decode_steps"])]
        }
        actual = {(r.get("phase"), r.get("mode"), r.get("step")) for r in data["output_comparisons"]}
        require(actual == expected_comparisons, f"{name}: missing output phase/step")
        expected_states = {
            (phase, depth, request, key)
            for phase in (0, 1)
            for depth, keys in ((0, ("conv", "recurrent")), (1, ("k", "v")))
            for request in range(len(prefix))
            for key in keys
        }
        actual_states = {
            (row.get("phase"), row.get("stack_index"), row.get("request"), row.get("tensor"))
            for row in data["state_checks"]
        }
        require(
            actual_states == expected_states and len(data["state_checks"]) == len(expected_states),
            f"{name}: incomplete mixed state coverage",
        )
        for side in ("baseline", "multichip"):
            row = data.get(side, {})
            require(row.get("runtime_fallback_audit") == "pass", f"{name}: fallback audit failed")
            require(
                row.get("deterministic") is True and row.get("all_step_eager_trace_equal") is True,
                f"{name}: trace checks failed",
            )
            require(
                row.get("trace_capture_count") == 1 and row.get("page_table_phases") == 2,
                f"{name}: page-table trace reuse missing",
            )
            require(
                row.get("positions") == [[n + step for n in lengths] for step in range(data["decode_steps"])],
                f"{name}: positions mismatch",
            )
            numeric(row.get("trace_replay_count"), "mixed replay count", 4 * data["decode_steps"])
            for phase in sequence(row.get("one_shot_vs_resumed_pcc"), "resumed phases", 2):
                for value in sequence(phase, "resumed rows", len(prefix)):
                    pcc(value, "one-shot versus resumed")
        require(data["multichip"].get("shared_collective_workspace") is True, f"{name}: stack does not share workspace")
        for side in ("baseline", "multichip"):
            row = data[side]
            for flag in (
                "shared_kv_pool",
                "shared_kv_vs_independent_bit_equal",
                "shared_kv_vs_independent_state_bit_equal",
                "shared_kv_disjoint_page_ownership",
                "shared_kv_page_tables_exact",
            ):
                require(row.get(flag) is True, f"{name}: {side} {flag} missing/failed")
            require(row.get("shared_kv_request_count") == len(prefix), f"{name}: {side} shared request count mismatch")
            pool_pages = row.get("shared_kv_pool_pages")
            numeric(pool_pages, f"{name} {side} pool pages", 1)
            for guards in sequence(row.get("shared_kv_guard_pages"), f"{name} {side} guard phases", 2):
                sequence(guards, f"{name} {side} guard pages")
                require(
                    len(set(guards)) == len(guards)
                    and all(isinstance(page, int) and 0 <= page < pool_pages for page in guards),
                    f"{name}: invalid guard pages",
                )
            untouched = sequence(row.get("shared_kv_untouched_checks"), f"{name} {side} untouched KV checks")
            replays = row["trace_replay_count"]
            require(replays % (2 * data["decode_steps"]) == 0, f"{name}: inconsistent replay accounting")
            repeats = replays // (2 * data["decode_steps"])
            expected_untouched = {
                (phase, repeat, key, rank)
                for phase in (0, 1)
                for repeat in range(repeats)
                for key in ("k", "v")
                for rank in range(4 if side == "multichip" else 1)
            }
            actual_untouched = {(r.get("phase"), r.get("repeat"), r.get("tensor"), r.get("rank")) for r in untouched}
            require(
                actual_untouched == expected_untouched and len(untouched) == len(expected_untouched),
                f"{name}: {side} incomplete untouched KV coverage",
            )
            require(
                all(r.get("unwritten_nonzero") == 0 for r in untouched),
                f"{name}: {side} unowned/future cache was written",
            )
        require(data["multichip"].get("output_replicas_bit_equal") is True, f"{name}: residual replicas differ")
        numeric(data["multichip"].get("output_replica_checks"), f"{name} replica checks", 1)
        self.evidence(data.get("tensor_artifact"))
        self.provenance(name, tracking=True, watcher=watcher)
        return {"lengths": lengths}

    def context(self, name, layer, length, context):
        data = self.read(name)
        require(
            data.get("passed") is True and data.get("ownership_passed") is True,
            f"{name}: context/ownership gate failed",
        )
        signature = data.get("signature", {})
        require(
            all(
                signature.get(k) == v
                for k, v in dict(layer=layer, length=length, context=context, decode_steps=2).items()
            ),
            f"{name}: context signature mismatch",
        )
        require(signature.get("checkpoint_revision") == REVISION, f"{name}: checkpoint mismatch")
        pcc(data.get("prefill_pcc"), name + " prefill")
        for value in sequence(data.get("decode_pcc"), name + " decode", 2):
            pcc(value, name + " decode")
        states = data.get("state_pcc", {})
        require(
            set(states) == ({"conv", "recurrent"} if layer == 0 else {"k", "v"}), f"{name}: state comparisons missing"
        )
        for value in states.values():
            pcc(value, name + " state")
        for side in ("baseline", "multichip"):
            row = data.get(side, {})
            require(row.get("runtime_fallback_audit") == "pass", f"{name}: fallback audit failed")
            require(
                row.get("decoder") == ("OptimizedDecoder" if side == "baseline" else "MultichipDecoder"),
                f"{name}: incorrect decoder",
            )
            require(
                row.get("kind") == ("linear_attention" if layer == 0 else "full_attention"), f"{name}: incorrect kind"
            )
            require(row.get("prefill_call_count") == 1, f"{name}: unexpected prefill count")
            require(
                row.get("decode_positions") == [length, length + 1], f"{name}: decode did not reach expected positions"
            )
            require(
                row.get("logical_length") == length and row.get("context") == context, f"{name}: side geometry mismatch"
            )
            require(
                row.get("trace_claim") is False and row.get("timing_claim") is False,
                f"{name}: eager evidence misclassified",
            )
            if layer == 3:
                own = row.get("ownership", {})
                require(
                    own.get("rank_to_global_kv_head") == ([0, 0, 1, 1] if side == "multichip" else [0, 1]),
                    f"{name}: incorrect KV head ownership",
                )
                require(own.get("page_tables_exact") is True, f"{name}: page table check failed")
                require(own.get("written_positions") == length + 2, f"{name}: written position mismatch")
                for key in ("k", "v"):
                    require(own.get(key + "_replicas_equal") is True, f"{name}: local head replicas differ")
                    for rank in (0, 2) if side == "multichip" else (0,):
                        require(
                            own.get(f"{key}_unwritten_nonzero_rank{rank}") == 0, f"{name}: unwritten cache modified"
                        )
            self.evidence(row.get("dram_census_artifact"))
        self.evidence(data.get("baseline_artifact"))
        self.evidence(data.get("multichip_artifact"))
        self.provenance(name)
        return {"decode_context_reached": length + 2}

    def watcher(self, layer, batch):
        self.paired(f"watcher_l{layer}_b{batch}.json", layer, batch, watcher=True)
        return self.watcher_log(f"watcher/l{layer}_b{batch}/generated/watcher/watcher.log")

    def watcher_log(self, name):
        path = self.evidence(name)
        completed = 0
        with path.open(errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                require(
                    not WATCHER_BAD.search(line),
                    f"{path}:{line_number}: watcher failure signature: {line.strip()[:200]}",
                )
                completed += bool(re.search(r"\bDump #\d+ completed at\b", line))
        require(completed > 0, f"{path}: no completed watcher dumps")
        return {"watcher_log": str(path), "completed_dumps": completed}

    def capacity(self):
        data = self.read(str(self.doc.parent / "context_contract.json"))
        require(
            data.get("hf_advertised_context") == data.get("supported_context") == 262144,
            "advertised context was reduced",
        )
        require(data.get("capability_reduction") is None, "context capability reduction present")
        plan = data.get("multichip_decoder_plan", {})
        require(
            plan.get("supported_context") == plan.get("hf_advertised_context") == 262144,
            "multichip context was reduced",
        )
        require(plan.get("capability_reduction") is None and plan.get("mesh") == [1, 4], "multichip contract mismatch")
        return {"supported_context": 262144}

    def profile_seed(self, result, correctness_path, layer):
        """Qualify narrow capture through a fresh complete natural B32 seed run."""
        label = str(correctness_path)
        seed_path = self.evidence(result.get("seed_artifact"))
        acceptance = self.evidence(result.get("seed_acceptance"))
        self.natural(layer, name=str(acceptance), tracking=False, clean=True)
        seed = self.read(str(acceptance))
        require(seed.get("decode_steps") == 8, f"{label}: seed must retain all eight decode controls")
        require(seed.get("stack_depth") == 1, f"{label}: seed stack depth mismatch")
        require(
            all(seed.get(key) == result.get(key) for key in ("layer", "batch", "length", "context")),
            f"{label}: seed/capture geometry mismatch",
        )
        require(
            seed.get("activation_file") == result.get("activation_file")
            and seed.get("activation_source") == result.get("activation_source"),
            f"{label}: seed/capture activation source mismatch",
        )
        require(
            seed.get("multichip", {}).get("precision_policy") == result.get("multichip", {}).get("precision_policy"),
            f"{label}: seed/capture precision policy mismatch",
        )
        seed_provenance = self.provenance(str(acceptance), clean=True)
        require("--save-prefill-states" in seed_provenance["command"], f"{label}: seed lacks prefill-state export")
        exported = self.evidence(seed.get("tensor_artifact"))
        output = option(seed_provenance["command"], "--output")
        require(output and self.evidence(output) == acceptance, f"{label}: seed acceptance output mismatch")
        require(exported == seed_path == acceptance.with_suffix(".pt"), f"{label}: seed export artifact mismatch")
        provenance = self.provenance(label, profiler=True)
        for item, name in ((seed_provenance, str(acceptance)), (provenance, label)):
            for source in ("tt/optimized_decoder.py", "tt/multichip_decoder.py"):
                digest = hashlib.sha256((DOC.parents[1] / source).read_bytes()).hexdigest()
                require(item["source_sha256"].get(source) == digest, f"{name}: current {source} identity mismatch")
        require(off(provenance["env"]["TT_METAL_TRACE_ALLOC_TRACKING"]), f"{label}: tracked profile capture")
        declared = option(provenance["command"], "--seed")
        require(declared and self.evidence(declared) == seed_path, f"{label}: capture seed command mismatch")
        return {"seed_artifact": str(seed_path), "seed_acceptance": str(acceptance)}

    def perf_batch(self, *, prefix, batch):
        # Use the same pure readers as the profiler summarizer, but never call
        # summarize(), which writes files. Reconstruct both phase summaries from
        # raw per-rank CSV rows instead of trusting a prior summary's flags.
        from models.autoports.qwen_qwen3_6_35b_a3b.tests import summarize_multichip_perf as profiler

        root = self.doc / prefix
        data = self.read(str(root / "perf_summary.json"))
        require(data.get("batch") == batch, f"{prefix}: profile batch mismatch")
        require(data.get("devices") == ["0", "1", "2", "3"], f"{prefix}: profile missing target mesh")
        require(data.get("expected_decode_replays") == 16, f"{prefix}: profile lacks 16 replays")
        measurements = sequence(data.get("measurements"), f"{prefix} measurements", 2)
        require({row.get("layer") for row in measurements} == {0, 3}, f"{prefix}: missing layer kind")
        for row in measurements:
            require(row.get("batch") == batch, f"{prefix}: layer batch mismatch")
            correctness_path = self.evidence(str(root / row["correctness"]))
            result = profiler.validate_correctness(correctness_path, row["layer"], batch)
            mesh = result.get("multichip", {})
            seeded = result.get("profile_fixture") == "seeded_b32_v1"
            require(
                row.get("duration_scope") == ("phase-specific; see phases" if seeded else profiler.BATCH_SCOPE),
                f"{prefix}: timing scope mismatch",
            )
            if seeded:
                require(row.get("profile_fixture") == result["profile_fixture"], f"{prefix}: fixture summary mismatch")
                for field in ("seed_artifact", "seed_acceptance"):
                    require(row.get(field) == result[field], f"{prefix}: {field} summary mismatch")
                self.profile_seed(result, correctness_path, row["layer"])
            require(
                row.get("precision_policy") == mesh.get("precision_policy"), f"{prefix}: precision summary mismatch"
            )
            require(
                mesh.get("residual_layout") == "replicated" and mesh.get("output_replicas_equal") is True,
                f"{prefix}: selected residual/replica evidence missing",
            )
            if not seeded:
                require(
                    mesh.get("all_step_eager_trace_equal") is True
                    and mesh.get("live_trace_prefill_replay_equal") is True,
                    f"{prefix}: trace controls missing",
                )
            provenance = self.provenance(str(correctness_path), profiler=True)
            require(off(provenance["env"]["TT_METAL_TRACE_ALLOC_TRACKING"]), f"{prefix}: tracking mixed with profiling")
            if batch == 32 and not seeded:
                require(
                    mesh.get("empty_expert_ranks_zero_after_poison") is True,
                    f"{prefix}: empty-expert poison control missing",
                )
                activation_path = self.evidence(result.get("activation_file"))
                declared = option(provenance["command"], "--activation-file")
                require(
                    declared and self.evidence(declared) == activation_path,
                    f"{prefix}: profile activation command mismatch",
                )
                metadata = self.read(str(activation_path.with_suffix(".metadata.json")))
                require(
                    metadata.get("checkpoint_revision") == REVISION and metadata.get("batch") == 32,
                    f"{prefix}: activation checkpoint/batch mismatch",
                )
                prompts = sequence(metadata.get("source_prompts"), "profile prompts", 32)
                require(
                    len(set(prompts)) == 32 and metadata.get("padding_or_repetition") is False,
                    f"{prefix}: real diverse routing evidence missing",
                )
            ops_path = self.evidence(str(root / row["ops_csv"]))
            raw_rows, fields = profiler.read_csv(ops_path, ("OP CODE", "OP TYPE", "DEVICE ID", profiler.DURATION))
            for mode in ("prefill", "decode"):
                scope = result.get("phase_scope", {}).get(mode, profiler.BATCH_SCOPE)
                expected, groups = profiler.summarize_phase(
                    correctness_path.parent, raw_rows, fields, mode, duration_scope=scope
                )
                phase = row.get("phases", {}).get(mode, {})
                # Includes all four ranks, every replay ID, per-rank op counts,
                # category totals and identical within-rank replay signatures.
                require(phase == expected, f"{prefix}: {mode} summary differs from raw execution rows")
                for field in ("compact_csv", "report_csv", "human_table"):
                    self.evidence(str(root / phase[field]))
                compact = self.evidence(str(root / phase["compact_csv"]))
                with compact.open(newline="") as stream:
                    recorded_groups = list(csv.DictReader(stream))
                expected_groups = [{key: str(value) for key, value in group.items()} for group in groups]
                require(
                    recorded_groups == expected_groups, f"{prefix}: {mode} compact CSV differs from raw execution rows"
                )
        return {"layers": [0, 3], "batch": batch, "devices": 4, "raw_replay_consistency_checked": True}

    def perf(self):
        # Both batch regimes are mandatory; either missing capture set fails.
        return {
            "batch1": self.perf_batch(prefix="", batch=1),
            "batch32": self.perf_batch(prefix="perf_b32", batch=32),
        }

    def trace_positive_control(self):
        name = "trace_alloc_tracking/positive_control.json"
        data = self.read(name)
        provenance = self.provenance(name, tracking=True)
        require(provenance["env"]["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0", "control skips program cache")
        require(provenance["env"]["TT_METAL_WATCHER"] is None, "control watcher must be unset")
        runner = "tests/run_multichip_trace_control.py"
        require(
            provenance["source_sha256"].get(runner)
            == hashlib.sha256((DOC.parents[1] / runner).read_bytes()).hexdigest(),
            "trace control runner identity changed",
        )
        require(data.get("mesh") == [1, 4] and data.get("fabric") == "FABRIC_1D", "wrong control mesh")
        for key in ("passed", "rejected", "unsafe_removed", "clean_replay_passed"):
            require(data.get(key) is True, f"trace control missing {key}")
        require(data.get("exact_output_replicas") == 4, "control did not check all replicas")
        require(data.get("exception_type") == "RuntimeError", "wrong control exception")
        exception = data.get("exception", "")
        require("device buffer(s) still alive before" in exception, "not an allocation rejection")
        require(f"Buffer {data['unsafe_buffer_id']}" in exception, "injected allocation absent from exception")
        require(data["unsafe_buffer_id"] != data["acknowledged_output_buffer_id"], "unsafe allocation acknowledged")
        log = self.evidence("trace_alloc_tracking/positive_control.log").read_text()
        require("EXPECTED_REJECTION " + exception in log, "observed exception missing from log")
        require("TRACE_ALLOCATION_POSITIVE_CONTROL PASS" in log, "control completion missing from log")
        return {"rejected": True, "clean_replay_passed": True, "devices": 4}

    def check(self, name, callback):
        before = set(self.sources)
        try:
            details = callback()
            self.results.append(dict(gate=name, passed=True, details=details if isinstance(details, dict) else None))
        except (
            AssertionError,
            KeyError,
            TypeError,
            ValueError,
            OSError,
            ZeroDivisionError,
            AttributeError,
            IndexError,
            OverflowError,
        ) as error:
            self.results.append(dict(gate=name, passed=False, error=f"{type(error).__name__}: {error}"))
        self.results[-1]["evidence"] = sorted(self.sources - before)

    def run(self):
        for layer in (0, 3):
            self.check(
                f"final_l{layer}_b1", lambda layer=layer: self.paired(f"final_l{layer}_b1.json", layer, 1, clean=True)
            )
            self.check(
                f"final_l{layer}_b32",
                lambda layer=layer: self.natural(layer, name=f"final_l{layer}_b32.json", tracking=False, clean=True),
            )
            self.check(f"natural_l{layer}_b32", lambda layer=layer: self.natural(layer))
            self.check(
                f"stress_l{layer}_s129_d128",
                lambda layer=layer: self.paired(
                    f"stress_l{layer}_s129_d128.json", layer, 1, length=129, steps=128, tracking=True
                ),
            )
            for length in (1, 31, 65, 257):
                self.check(
                    f"boundary_l{layer}_s{length}",
                    lambda layer=layer, length=length: self.paired(
                        f"boundary_l{layer}_s{length}.json", layer, 1, length=length
                    ),
                )
            self.check(
                f"context_l{layer}_32k",
                lambda layer=layer: self.context(f"context_l{layer}_32k.json", layer, 32769, 65536),
            )
            for batch in (1, 32):
                self.check(f"watcher_l{layer}_b{batch}", lambda layer=layer, batch=batch: self.watcher(layer, batch))
        self.check("trace_allocation_positive_control", self.trace_positive_control)
        self.check("mixed_stack_selected", self.mixed)
        self.check("context_fullmax", lambda: self.context("context_fullmax.json", 3, 262142, 262144))
        self.check("context_capability", self.capacity)
        self.check("perf_summary", self.perf)
        report = dict(
            schema_version=1,
            passed=all(row["passed"] for row in self.results),
            pcc_threshold=PCC_MIN,
            gates=self.results,
            evidence_sources=sorted(self.sources),
            scope="Recorded multichip hard gates; stage review and local checkpoint commit occur after these gates",
        )
        self.doc.mkdir(parents=True, exist_ok=True)
        path = self.doc / "gate_summary.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, default=DOC)
    args = parser.parse_args()
    report, path = Gate(args.doc).run()
    failures = [r for r in report["gates"] if not r["passed"]]
    print(
        f"MULTICHIP_GATES {'PASS' if report['passed'] else 'FAIL'}: {len(report['gates']) - len(failures)}/{len(report['gates'])}; {path}"
    )
    for row in failures:
        print(f"{row['gate']}: {row['error']}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
