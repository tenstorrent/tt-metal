# SPDX-License-Identifier: Apache-2.0
"""Verify a canonical same-chunk reference without rewriting a failed lifecycle.

CPU/file inspection only. Requires completed full-model probe evidence; missing,
partial, reduced, changed-source, or mismatching evidence fails closed. This
produces a separate verification record for review, not a replacement lifecycle
pass, numerical chunk-invariance claim, or qualitative/performance approval.
"""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

SIGNATURES = {"decode", "greedy", "sample", "argmax", "prefill_1", "prefill_2", "prefill_4", "prefill_8"}
LABELS = {
    "first_129",
    "first_130",
    "first_131",
    "first_513",
    "first_645",
    "mixed_129",
    "mixed_131",
    "mixed_513",
    "mixed_645",
    "repeat_129",
    "repeat_131",
    "sample_130",
    "return_greedy_130",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def first_difference(left, right):
    return next(
        (i for i, pair in enumerate(zip(left, right)) if pair[0] != pair[1]),
        min(len(left), len(right)) if len(left) != len(right) else None,
    )


def lifecycle_contract(rows, failed):
    require(
        len(rows) == 13 and {r["label"] for r in rows} == LABELS, "Expected exactly the original13 lifecycle requests"
    )
    require(
        failed["status"] == "failed" and failed["requests"] == rows,
        "Original failed lifecycle evidence must be preserved exactly",
    )
    require(failed["completed_requests"] == 13, "Incomplete original lifecycle")
    indexed = {r["label"]: r for r in rows}
    differences, exact = [], []
    for row in rows:
        length = int(row["label"].rsplit("_", 1)[1])
        require(row["prompt_length"] == length == len(row["prompt_token_ids"]), "Invalid prompt length/IDs")
        require(
            row["output_tokens"] == len(row["token_ids"]) == 96, "Every lifecycle row must contain96 exact output IDs"
        )
        require(
            all(type(i) is int and 0 <= i < 248320 for i in row["prompt_token_ids"] + row["token_ids"]),
            "Invalid token ID",
        )
        require(
            row["usage"]["prompt_tokens"] == length and row["usage"]["completion_tokens"] == 96,
            "Usage disagrees with exact token counts",
        )
        require(
            row["start"] < row["end"] and all(math.isfinite(row[k]) for k in ("start", "end")),
            "Invalid lifecycle timing",
        )
        require(
            row["sampling"] == ("sample" if row["label"] == "sample_130" else "greedy"), "Unexpected sampling profile"
        )
        if not row["label"].startswith("first_") and row["sampling"] == "greedy":
            control = indexed[f"first_{length}"]
            require(row["prompt_token_ids"] == control["prompt_token_ids"], "Compared lifecycle prompt IDs differ")
            difference = first_difference(control["token_ids"], row["token_ids"])
            if difference is None:
                exact.append(row["label"])
            else:
                differences.append(
                    dict(
                        label=row["label"],
                        first_difference=difference,
                        control_token=control["token_ids"][difference],
                        candidate_token=row["token_ids"][difference],
                    )
                )
    require(
        len(differences) == 1 and differences[0]["label"] == "mixed_513",
        "A different/additional lifecycle mismatch cannot use this reference",
    )
    require(failed["exact_greedy_differences"] == differences, "Original failure summary disagrees with raw tokens")
    return indexed, differences, exact


def chunks_for_request(events, row):
    selected = [e for e in events if row["start"] <= e["time"] <= row["end"]]
    slots = {
        e["physical_slots"][i]
        for e in selected
        if e["event"] == "prefill_schedule"
        for i, end in enumerate(e["ends"])
        if end == row["prompt_length"]
    }
    require(len(slots) == 1, "Target physical state slot is ambiguous")
    slot = slots.pop()
    chunks = [
        dict(start=start, length=e["chunk_length"], slot=bank)
        for e in selected
        if e["event"] == "prefill_chunk"
        for start, bank, active in zip(e["start_positions"], e["slots"], e["active"])
        if active and bank == slot
    ]
    cursor = 0
    for chunk in chunks:
        require(
            chunk["start"] == cursor and chunk["length"] in (1, 2, 4, 8),
            "Target chunks contain a gap/overlap/invalid size",
        )
        cursor += chunk["length"]
    require(cursor == row["prompt_length"], "Target chunks do not cover the complete prompt")
    return chunks


def trace_contract(events, *, pid=None, workload_bounds=None):
    prepared = [e for e in events if e["event"] == "prepared"]
    require(len(prepared) == 1 and prepared[0]["batch"] == 32, "Expected one B32 preparation")
    baseline = prepared[0]
    captures = [e for e in events if e["event"] == "capture"]
    counts = Counter(e["signature"] for e in captures)
    require(
        set(counts) == SIGNATURES and set(counts.values()) == {1},
        "Each of eight execution signatures must be captured exactly once",
    )
    ids = {e["signature"]: e["trace_id"] for e in captures}
    require(
        ids == baseline["traces"] and all(e["time"] < baseline["time"] for e in captures),
        "Trace IDs/preparation ordering disagree",
    )
    require(baseline["buckets"] == [1, 2, 4, 8], "Unexpected native bucket contract")
    preparations = {e["signature"]: e for e in events if e["event"] == "preparation"}
    require(set(preparations) == SIGNATURES, "Missing warmup execution signatures")
    require(
        all(e["time"] < baseline["time"] for e in preparations.values()),
        "Warmup execution occurred after preparation completed",
    )
    for event in events:
        if pid is not None:
            require(event["pid"] == pid, "Events contain another server process")
        if event["event"] == "replay":
            require(event["trace_id"] == ids[event["signature"]], "Replay refers to an unqualified trace")
        within = workload_bounds is None or workload_bounds[0] <= event["time"] <= workload_bounds[1]
        if within:
            require(
                event["event"]
                not in {
                    "retirement",
                    "retire",
                    "invalidation",
                    "eviction",
                    "evict",
                    "recapture",
                    "release",
                    "shutdown_release",
                },
                "Trace was retired/replaced during workload",
            )
    return dict(
        trace_ids=ids,
        signatures=[
            dict(
                signature=name,
                trace_id=ids[name],
                captures=1,
                replays=sum(e["event"] == "replay" and e["signature"] == name for e in events),
                retirements=0,
            )
            for name in sorted(ids)
        ],
    )


def live_trace_contract(events, runtime, rows):
    trace = trace_contract(
        events, pid=runtime["pid"], workload_bounds=(min(r["start"] for r in rows), max(r["end"] for r in rows))
    )
    require(trace["trace_ids"] == runtime["trace_ids"], "Live runtime trace IDs disagree")
    schedule, cursor, covered = None, None, []
    for event in events:
        if event["event"] == "prefill_schedule":
            require(schedule is None, "Nested/incomplete native prefill schedule")
            schedule, cursor = event, event["starts"].copy()
            require(
                event["suffix_lengths"][: len(cursor)] == [end - start for start, end in zip(cursor, event["ends"])],
                "Schedule suffix lengths disagree",
            )
        elif event["event"] == "prefill_chunk":
            require(schedule is not None, "Native chunk without serving schedule")
            for i in range(len(cursor)):
                if event["active"][i]:
                    require(
                        event["slots"][i] == schedule["physical_slots"][i] and event["start_positions"][i] == cursor[i],
                        "Chunk position/state slot changed within schedule",
                    )
                    cursor[i] += event["chunk_length"]
                    require(cursor[i] <= schedule["ends"][i], "Prefill overruns scheduled suffix")
        elif event["event"] == "prefill_complete":
            require(schedule is not None and cursor == schedule["ends"], "Incomplete native prefill")
            covered.append(schedule)
            schedule, cursor = None, None
    require(schedule is None and covered, "No complete native prefill coverage")
    require(any(any(start > 0 for start in e["starts"]) for e in covered), "No continued prefill")
    require(any(any(n % 32 for n in e["suffix_lengths"]) for e in covered), "No non-aligned prefill")
    decode = [e for e in events if e["event"] == "decode_submit"]
    require(decode and max(sum(e["active"]) for e in decode) >= 3, "Concurrent decode missing")
    require({"greedy", "sample"} <= {e["sampling"] for e in decode}, "Device sampling switch missing")
    require(all(e["sampling"] in ("greedy", "sample") for e in decode), "Unexpected host decode in device lifecycle")
    page_only = [e for e in decode if e["reload_page_table"]]
    require(len(page_only) >= 2 and all(not e["reload_inputs"] for e in page_only), "Page-only updates missing/invalid")
    require(
        any(
            any(e["starts"])
            and any(d["time"] < e["time"] for d in decode)
            and any(d["time"] > e["time"] for d in decode)
            for e in covered
        ),
        "Prefill continuation did not occur between decode submissions",
    )
    memory = [e["memory"]["total_bytes_allocated_per_bank"] for e in events if e["event"] == "prefill_complete"]
    require(memory and len(set(memory)) == 1, "Serving allocations grew across prefill shapes")
    return dict(
        **trace,
        complete_schedules=covered,
        maximum_active_decode=max(sum(e["active"]) for e in decode),
        page_only_updates=len(page_only),
        allocated_bytes_per_bank=memory[0],
    )


def raw_logit_contract(control, candidate, phases, difference):
    import torch

    for raw, phase in zip((control, candidate), phases):
        tensor = raw["logits"]
        require(
            tuple(tensor.shape) == (96, 248320) and tensor.dtype == torch.bfloat16, "Expected96 full BF16 logit vectors"
        )
        require(raw["tokens"] == phase["tokens"] and bool(torch.isfinite(tensor).all()), "Raw logits/tokens invalid")
        selected = tensor[torch.arange(96), torch.tensor(raw["tokens"])]
        require(
            bool(torch.equal(selected, tensor.max(dim=1).values)),
            "Saved canonical greedy token is not a maximum-logit candidate",
        )
    left, right = control["logits"][difference].double(), candidate["logits"][difference].double()
    lc, rc = left - left.mean(), right - right.mean()
    denominator = lc.norm() * rc.norm()
    require(float(denominator) > 0, "Degenerate divergence vectors")
    return dict(
        output_index=difference,
        consumed_position=513 + difference - 1,
        same_consumed_history=control["tokens"][:difference] == candidate["tokens"][:difference],
        pcc=float(torch.dot(lc, rc) / denominator),
        max_abs=float((left - right).abs().max()),
        control_top_margin=float(torch.topk(left, 2).values.diff().abs()[0]),
        candidate_top_margin=float(torch.topk(right, 2).values.diff().abs()[0]),
    )


def completed_full_probe_contract(probe):
    require(
        probe["status"] == "completed"
        and probe["matched_live_token_status"] == "pass"
        and probe["closed_mesh"] is True,
        "Full matched probe is missing/incomplete/failed or did not close mesh",
    )
    require(
        probe["selected_layers"] == "all"
        and probe["args"]["layers"] == "all"
        and probe["args"]["batch"] == 32
        and probe["args"]["steps"] == 95
        and probe["context"] == 262144,
        "Reduced or wrong-shape probe cannot qualify the lifecycle",
    )
    require(
        probe["tracking"] == {"TT_METAL_TRACE_ALLOC_TRACKING": "1", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE": "0"},
        "Trace allocation tracking was not fully enabled",
    )


def evaluate(args):
    import torch

    rows, events = jsonl(args.journal), jsonl(args.events)
    failed, runtime, probe = (json.loads(path.read_text()) for path in (args.failed, args.runtime, args.probe))
    indexed, differences, unchanged = lifecycle_contract(rows, failed)
    require(
        runtime["batch"] == 32
        and runtime["context"] == 262144
        and runtime["public_pool_tokens"] == 263168
        and runtime["scratch_page_base"] == 8224,
        "Unexpected live batch/context/cache pool",
    )
    require(
        [layer["layer"] for layer in runtime["runtime"]["layers"]] == list(range(48)),
        "Live reference is not all48 real layers",
    )
    live_trace = live_trace_contract(events, runtime, rows)
    completed_full_probe_contract(probe)
    require(
        probe["runtime"] == runtime["runtime"] and probe["precision_policy"] == runtime["runtime"]["policy"],
        "Native precision/fidelity/layer exceptions differ",
    )
    geometry = probe["cache_geometry"]
    require(
        geometry
        == dict(
            batch=32,
            context=262144,
            public_pool_tokens=263168,
            public_blocks=8224,
            scratch_page_base=8224,
            private_scratch_tokens=16384,
            cache_shapes=runtime["cache_shapes"],
        ),
        "Probe cache geometry differs",
    )
    require(
        probe["shutdown_counters"]["captures"] == probe["shutdown_counters"]["releases"] == 8,
        "Probe trace release counts disagree",
    )
    fixture = probe["matched_live"]
    for path in (args.journal, args.events, args.runtime):
        stored = {str(Path(k).resolve()): v for k, v in fixture["sources"].items()}
        require(
            stored.get(str(path.resolve())) == sha(path), "Probe used a different/changed live fixture: " + str(path)
        )
    require(fixture["runtime"] == runtime, "Embedded live runtime differs")
    expected_sources = {
        Path(__file__).with_name(name).resolve() for name in ("vllm_chunk_numerics_probe.py", "vllm_lifecycle.py")
    }
    expected_sources |= {
        (Path(__file__).resolve().parents[1] / "tt" / name).resolve()
        for name in (
            "generator_vllm.py",
            "generator.py",
            "model.py",
            "multichip_decoder.py",
            "qsa.py",
            "paged_qsa.py",
            "precision.py",
        )
    }
    source_hashes = {Path(k).resolve(): v for k, v in probe["source_sha256"].items()}
    require(expected_sources <= source_hashes.keys(), "Probe source manifest is incomplete")
    for path, expected_hash in source_hashes.items():
        require(sha(path) == expected_hash, "Probe source changed since collection: " + str(path))
    phases = probe["phases"]
    require(
        [p["name"] for p in phases] == ["control", "forced_tail"],
        "Both and only the required probe phases must complete",
    )
    probe_events_path = args.probe.with_suffix(".events.jsonl")
    probe_events = jsonl(probe_events_path)
    require(len({e["pid"] for e in probe_events}) == 1, "Probe events contain multiple worker processes")
    phase_markers = [(i, e["phase"]) for i, e in enumerate(probe_events) if e["event"] == "chunk_numerics_phase"]
    require([name for _, name in phase_markers] == ["control", "forced_tail"], "Probe event phases missing/reordered")
    probe_trace = trace_contract(
        probe_events,
        workload_bounds=(
            probe_events[phase_markers[0][0]]["time"],
            max(e["time"] for e in probe_events if e["event"] == "decode_submit"),
        ),
    )
    references, raw = [], []
    for index, (phase, label) in enumerate(zip(phases, ("first_513", "mixed_513"))):
        row = indexed[label]
        expected_chunks = chunks_for_request(events, row)
        require(
            phase["tokens"] == row["token_ids"] and len(phase["rows"]) == 96,
            "Canonical output IDs do not exactly match HTTP: " + label,
        )
        require(
            probe["prompts"]["target_513"]
            == row["prompt_token_ids"]
            == fixture["phases"][phase["name"]]["prompt_token_ids"],
            "Canonical prompt IDs differ",
        )
        require(
            phase["observed_native_chunks"] == expected_chunks == fixture["phases"][phase["name"]]["native_chunks"],
            "Stored physical chunk signatures differ",
        )
        require(
            phase["trace_ids_unchanged"] and phase["trace_ids"] == probe_trace["trace_ids"],
            "Probe recaptured/replaced traces",
        )
        segment = probe_events[
            phase_markers[index][0] + 1 : phase_markers[index + 1][0]
            if index + 1 < len(phase_markers)
            else len(probe_events)
        ]
        observed = []
        for event in segment:
            if event["event"] == "prefill_chunk":
                require(
                    sum(event["active"]) == 1 and event["active"][0], "Forced native reference had an active companion"
                )
                observed.append(
                    dict(start=event["start_positions"][0], length=event["chunk_length"], slot=event["slots"][0])
                )
        require(observed == expected_chunks, "Raw probe events disagree with live physical chunks")
        decodes = [e for e in segment if e["event"] == "decode_submit"]
        require(len(decodes) == 95, "Expected95 canonical decode submissions")
        for step, event in enumerate(decodes):
            require(
                event["sampling"] == "greedy" and sum(event["active"]) == 1 and event["active"][0],
                "Probe decode did not use canonical solo device greedy sampling",
            )
            require(event["slots"][0] == expected_chunks[0]["slot"], "Probe target state slot changed")
            require(
                event["reload_inputs"] == (step == 0) and not event["reload_page_table"],
                "Probe lost resident device feedback",
            )
        for output_index, observed_row in enumerate(phase["rows"]):
            require(
                observed_row["output_index"] == output_index
                and observed_row["selected_token"] == phase["tokens"][output_index],
                "Probe per-step token rows disagree",
            )
            if output_index:
                require(observed_row["consumed_position"] == 512 + output_index, "Probe decode positions disagree")
        raw_path = Path(phase["raw_logits"])
        raw.append(torch.load(raw_path, map_location="cpu", weights_only=True))
        references.append(
            dict(
                http_label=label,
                canonical_phase=phase["name"],
                exact_output_tokens=96,
                native_chunks=expected_chunks,
                raw_logits=dict(path=str(raw_path), sha256=sha(raw_path)),
            )
        )
    divergence = first_difference(phases[0]["tokens"], phases[1]["tokens"])
    require(divergence == differences[0]["first_difference"], "Canonical/live divergence index differs")
    logits = raw_logit_contract(*raw, phases, divergence)
    require(logits["same_consumed_history"], "Divergence logits do not share consumed history")
    artifacts = (args.journal, args.events, args.failed, args.runtime, args.probe, probe_events_path)
    return dict(
        status="verified",
        scope="Canonical same-physical-chunk adapter correctness evidence, subject to stage review; original cross-shape invariance test remains failed",
        original_cross_shape_result=dict(status="failed", artifact=str(args.failed), differences=differences),
        unchanged_exact_lifecycle_comparisons=unchanged,
        completed_requests=13,
        canonical_references=references,
        same_history_divergence=logits,
        live_trace_audit=live_trace,
        canonical_trace_audit=probe_trace,
        artifacts={str(path): sha(path) for path in artifacts},
        probe_source_hashes=probe["source_sha256"],
        provenance_limit="Live worker has no recorded loaded-source hash manifest. Probe source integrity is checked against current files; runtime precision/layer metadata, geometry, prompts, chunks and96 output IDs are checked. Live native-argument telemetry is retained in the hashed runtime artifact but is not independently cross-compared to probe call arguments here. This does not retrospectively prove live binary source identity.",
        remaining_gates="Independent stage review, full sampling runner, qualitative judgment, benchmarks and cleanup remain separate. No generated lifecycle pass replaces the original failure.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("journal", "events", "failed", "runtime", "probe", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Use a fresh output; evidence must not be overwritten")
    try:
        result = evaluate(args)
    except Exception as error:
        result = dict(status="failed", failure=repr(error), scope="No canonical acceptance established")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "scope")}))
    if result["status"] != "verified":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
