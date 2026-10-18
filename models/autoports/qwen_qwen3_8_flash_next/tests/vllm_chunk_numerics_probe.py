# SPDX-License-Identifier: Apache-2.0
"""Chunk-boundary differential, not a standalone model accuracy/performance gate.

The explicit host-compatibility prefill result and diagnostic decode-logit reads
never select feedback tokens. Canonical device greedy sampling owns all tokens.
Run only with serialized exclusive TT device ownership.
"""
import os

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
if os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] != "1":
    raise RuntimeError("This correctness probe requires trace allocation tracking")
if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
    raise RuntimeError("Program-cache allocations must remain tracked")

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig
from vllm_tt_plugin.model_input import TTSamplingParams

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = 262144


def prompt(tokenizer, length, topic):
    # Identical rendering/newline padding to vllm_lifecycle.py.
    text = tokenizer.apply_chat_template(
        [dict(role="user", content=topic)], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tokenizer.encode(text)
    end = ids.index(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    ids[end:end] = [198] * (length - len(ids))
    assert len(ids) == length
    return ids


def stats(values):
    values = values.float().flatten()
    top, ids = torch.topk(values, 8)
    return dict(
        top_ids=ids.tolist(),
        top_values=top.tolist(),
        top1_margin=float(top[0] - top[1]),
        finite=bool(torch.isfinite(values).all()),
    )


def compare_logits(reference, candidate):
    left, right = reference.double().flatten(), candidate.double().flatten()
    delta = (left - right).abs()
    lc, rc = left - left.mean(), right - right.mean()
    denom = lc.norm() * rc.norm()
    return dict(
        pcc=float(torch.dot(lc, rc) / denom) if denom else None,
        max_abs=float(delta.max()),
        mean_abs=float(delta.mean()),
        exact=bool(torch.equal(reference, candidate)),
        control=stats(reference),
        candidate=stats(candidate),
    )


def phase_schedule(phase):
    if phase == "forced_tail":
        # Match the mixed target's native boundaries with every other row
        # inactive; external calls carry full prefixes and absolute ends.
        ends = [256, 384, 386, 387, 507, 511, 513]
        starts = [0] + ends[:-1]
        return [dict(starts=[start], ends=[end], slots=[1]) for start, end in zip(starts, ends)]
    first_end = 512 if phase in ("control", "repeat_control") else 256
    schedule = [dict(starts=[0], ends=[first_end], slots=[1]), dict(starts=[first_end], ends=[513], slots=[1])]
    if phase == "mixed":
        schedule[1] = dict(starts=[256, 0], ends=[513, 131], slots=[1, 2])
    return schedule


def live_fixture(args):
    """Reconstruct exact target chunks from completed HTTP rows and adapter events."""
    rows = [json.loads(line) for line in args.live_journal.read_text().splitlines()]
    events = [json.loads(line) for line in args.live_events.read_text().splitlines()]
    runtime = json.loads(args.live_runtime.read_text())
    if args.layers != "all" or args.batch != runtime["batch"] or runtime["context"] != CONTEXT:
        raise ValueError("Matched-live evidence requires all layers and the same batch/context as the running server")
    if runtime["public_pool_tokens"] != CONTEXT + args.batch * 32:
        raise ValueError("Live physical public pool differs from this probe's allocation contract")
    phases = {}
    for phase, label in (("control", args.live_control_label), ("forced_tail", args.live_mixed_label)):
        matches = [row for row in rows if row["label"] == label]
        if len(matches) != 1:
            raise ValueError(f"Expected one completed HTTP row for {label}")
        row = matches[0]
        if row["prompt_length"] != 513 or len(row["token_ids"]) != args.steps + 1 or row["sampling"] != "greedy":
            raise ValueError("Expected the 513-token greedy request with exactly steps+1 saved output token IDs")
        selected = [event for event in events if row["start"] <= event["time"] <= row["end"]]
        slots = {
            event["physical_slots"][i]
            for event in selected
            if event["event"] == "prefill_schedule"
            for i, end in enumerate(event["ends"])
            if end == 513
        }
        if len(slots) != 1:
            raise ValueError(f"Cannot uniquely identify target physical state slot for {label}: {slots}")
        slot = slots.pop()
        chunks = [
            dict(start=start, length=event["chunk_length"], slot=bank)
            for event in selected
            if event["event"] == "prefill_chunk"
            for start, bank, active in zip(event["start_positions"], event["slots"], event["active"])
            if active and bank == slot
        ]
        cursor, schedule = 0, []
        for chunk in chunks:
            if chunk["start"] != cursor or chunk["length"] not in (1, 2, 4, 8):
                raise ValueError(f"Incomplete/noncanonical target chunk sequence for {label}: {chunk}")
            # Merge only consecutive maximum-size8 chunks. Smaller chunks must
            # remain separate calls or the generator could choose a larger bucket.
            if schedule and chunk["length"] == 8 and schedule[-1]["native_length"] == 8:
                schedule[-1]["ends"][0] += 8
            else:
                schedule.append(
                    dict(starts=[cursor], ends=[cursor + chunk["length"]], slots=[slot], native_length=chunk["length"])
                )
            cursor += chunk["length"]
        if cursor != 513:
            raise ValueError(f"Target prefill coverage is {cursor}, expected513")
        phases[phase] = dict(
            http_label=label,
            target_physical_slot=slot,
            native_chunks=chunks,
            schedule=schedule,
            http_tokens=row["token_ids"],
            http_text=row["text"],
            prompt_token_ids=row["prompt_token_ids"],
        )
    if phases["control"]["prompt_token_ids"] != phases["forced_tail"]["prompt_token_ids"]:
        raise ValueError("Live control and mixed request prompts differ")
    return dict(
        phases=phases,
        runtime=runtime,
        sources={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.live_journal, args.live_events, args.live_runtime)
        },
        scope="Independent canonical-generator control and forced-shape solo replay; all companion requests inactive. Exact HTTP token equality is required separately for both phases.",
        page_mapping_limit="Public/private pool geometry matches. Physical page IDs are distinct reversed pages; live events do not expose the scheduler's actual block IDs.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch", type=int, choices=(3, 32), default=3)
    parser.add_argument(
        "--layers", default="0,1,3", help="Comma-separated real layer indices, or all for the complete model"
    )
    parser.add_argument("--steps", type=int, default=64, help="Decode steps after the prefill next-token result")
    parser.add_argument("--trace-region-size", type=int, default=160000000)
    parser.add_argument("--phases", default=None)
    parser.add_argument("--live-journal", type=Path, help="Completed lifecycle JSONL with exact prompts/tokens")
    parser.add_argument("--live-events", type=Path, help="Matching server adapter event journal")
    parser.add_argument("--live-runtime", type=Path, help="Matching server runtime/policy/cache report")
    parser.add_argument("--live-control-label", default="first_513")
    parser.add_argument("--live-mixed-label", default="mixed_513")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture_paths = (args.live_journal, args.live_events, args.live_runtime)
    if any(fixture_paths) and not all(fixture_paths):
        raise ValueError("Matched-live mode requires --live-journal, --live-events and --live-runtime")
    fixture = live_fixture(args) if all(fixture_paths) else None
    phases = (
        args.phases or ("control,forced_tail" if fixture else "control,split_solo,mixed,forced_tail,repeat_control")
    ).split(",")
    if fixture and phases != ["control", "forced_tail"]:
        raise ValueError("Matched-live mode runs control,forced_tail with all other requests inactive")
    if phases[0] != "control" or any(
        p not in ("control", "split_solo", "mixed", "forced_tail", "repeat_control") for p in phases
    ):
        raise ValueError("Start with control, followed by split_solo/mixed/forced_tail/repeat_control")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if os.environ.get("TT_METAL_DEVICE_PROFILER") not in (None, "0"):
        raise RuntimeError("Disable device profiling")
    if args.layers == "all":
        os.environ.pop("QWEN_FLASH_SERVING_LAYERS", None)
        selected_layers = "all"
    else:
        selected_layers = [int(value) for value in args.layers.split(",")]
        if (
            not selected_layers
            or any(value < 0 for value in selected_layers)
            or len(set(selected_layers)) != len(selected_layers)
        ):
            raise ValueError("layers must contain distinct nonnegative layer indices")
        os.environ["QWEN_FLASH_SERVING_LAYERS"] = ",".join(map(str, selected_layers))
    os.environ["QWEN_FLASH_CHECKPOINT"] = args.checkpoint
    os.environ["QWEN_FLASH_ALLOW_HOST_SAMPLING"] = "1"
    os.environ["QWEN_FLASH_TRACE_EVENTS"] = str(args.output.with_suffix(".events.jsonl"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or Path(os.environ["QWEN_FLASH_TRACE_EVENTS"]).exists():
        raise FileExistsError("Use a fresh output to preserve prior evidence")
    report = dict(
        status="running",
        scope=f"Layers {selected_layers} numeric/state differential; no accuracy or speed claim",
        selected_layers=selected_layers,
        args=vars(args),
        context=CONTEXT,
        phases=[],
        matched_live=fixture,
        tracking={
            k: os.environ.get(k, "0")
            for k in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE")
        },
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "tests/vllm_lifecycle.py",
                ROOT / "tt/generator_vllm.py",
                ROOT / "tt/generator.py",
                ROOT / "tt/model.py",
                ROOT / "tt/multichip_decoder.py",
                ROOT / "tt/qsa.py",
                ROOT / "tt/paged_qsa.py",
                ROOT / "tt/kernels/qsa_pooled_paged.cpp",
                ROOT / "tt/precision.py",
            )
        },
    )

    def persist():
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    mesh, adapter = None, None
    persist()
    try:
        torch.set_num_threads(8)
        ttnn.CONFIG.throw_exception_on_fallback = True
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        target = (
            fixture["phases"]["control"]["prompt_token_ids"]
            if fixture
            else prompt(tokenizer, 513, "Explain how bread is baked in a short paragraph.")
        )
        companion = prompt(tokenizer, 131, "Explain what a bicycle does in two sentences.")
        report["prompts"] = {"target_513": target, "companion_131": companion}
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
        cfg = Qwen4ExpConfig.from_pretrained(args.checkpoint, local_files_only=True)
        adapter = QwenFlashNextForCausalLM.initialize_vllm_model(cfg, mesh, args.batch, CONTEXT)
        total_blocks = CONTEXT // 32 + args.batch
        cache = adapter.allocate_kv_cache((total_blocks, 1, 32, 256), torch.float8_e4m3fn, 1)
        adapter.warmup_model_prefill(cache, enable_trace=True, can_sample_on_device=True)
        adapter.warmup_model_decode(cache, enable_trace=True, can_sample_on_device=True)
        g = adapter.generator
        from models.autoports.qwen_qwen3_8_flash_next.tt.precision import runtime_summary

        report["runtime"] = runtime_summary(g.model)
        report["precision_policy"] = g.model.precision_policy
        report["cache_geometry"] = dict(
            batch=args.batch,
            context=CONTEXT,
            public_pool_tokens=g.model.public_pool_tokens,
            public_blocks=total_blocks,
            scratch_page_base=g.model.scratch_page_base,
            private_scratch_tokens=args.batch * 512,
            cache_shapes=[
                {name: list(tensor.shape) for name, tensor in row.items() if name != "ple_cache"}
                for row in cache.layers
            ],
        )
        if fixture:
            if report["runtime"] != fixture["runtime"]["runtime"]:
                raise AssertionError("Loaded precision/runtime contract differs from the live server")
            if report["cache_geometry"]["cache_shapes"] != fixture["runtime"]["cache_shapes"]:
                raise AssertionError("Loaded exported cache geometry differs from the live server")
            if len(g.model.layers) != 48:
                raise AssertionError("Matched-live proof requires all48 real layers")
        trace_ids = dict(g.traces)
        params = TTSamplingParams(
            temperature=[0.0] * args.batch,
            top_k=[1] * args.batch,
            top_p=[1.0] * args.batch,
            seed=[101 + i for i in range(args.batch)],
            enable_log_probs=[False] * args.batch,
        )
        # Distinct reversed physical pages; target consistently lives in bank1.
        page = torch.zeros(args.batch, CONTEXT // 32, dtype=torch.int32)
        page[0, :32] = torch.arange(total_blocks - 1, total_blocks - 33, -1)
        page[1, :32] = torch.arange(total_blocks - 33, total_blocks - 65, -1)
        reference_vectors, reference_tokens = None, None
        pairwise_inputs = {}
        for phase in phases:
            g.reset()
            adapter.slot_map = list(range(args.batch))
            if hasattr(adapter, "sampling"):
                del adapter.sampling
            g.event("chunk_numerics_phase", phase=phase)
            expected = fixture["phases"][phase] if fixture else None
            schedule = expected["schedule"] if expected else phase_schedule(phase)
            target_slot = expected["target_physical_slot"] if expected else 1
            phase_report = dict(name=phase, schedule=schedule, target_physical_slot=target_slot, rows=[], tokens=[])
            report["phases"].append(phase_report)
            persist()
            event_offset = len(Path(os.environ["QWEN_FLASH_TRACE_EVENTS"]).read_text().splitlines())
            for entry in schedule:
                rows = len(entry["ends"])
                tokens = torch.zeros(rows, max(entry["ends"]), dtype=torch.int32)
                tokens[0, : entry["ends"][0]] = torch.tensor(target[: entry["ends"][0]], dtype=torch.int32)
                if rows == 2:
                    tokens[1, :131] = torch.tensor(companion, dtype=torch.int32)
                logits, deltas = adapter.prefill_forward(
                    tokens,
                    page[:rows],
                    cache,
                    prompt_lens=entry["ends"],
                    empty_slots=entry["slots"],
                    start_pos=entry["starts"],
                    sampling_params=None,
                    enable_trace=True,
                )
                assert deltas.tolist() == [0] * rows
            observed = [
                json.loads(line)
                for line in Path(os.environ["QWEN_FLASH_TRACE_EVENTS"]).read_text().splitlines()[event_offset:]
            ]
            phase_report["observed_native_chunks"] = [
                dict(start=start, length=event["chunk_length"], slot=slot)
                for event in observed
                if event["event"] == "prefill_chunk"
                for start, slot, active in zip(event["start_positions"], event["slots"], event["active"])
                if active and slot == target_slot
            ]
            if expected and phase_report["observed_native_chunks"] != expected["native_chunks"]:
                raise AssertionError("Native target chunks do not exactly match the live event sequence")
            initial_logits = logits[0].reshape(-1).clone()
            assert initial_logits.numel() == g.model.cfg.vocab_size
            # Seed resident feedback through the generator's actual sampler.
            mode = adapter._sampling(params)
            g.sampling_mode = "device"
            g._replay(mode)
            first_token = int(g._read_tokens()[0])
            vectors = [initial_logits]
            emitted = [first_token]
            phase_report["rows"].append(
                dict(output_index=0, stage="prefill", selected_token=first_token, logits=stats(initial_logits))
            )
            phase_report["tokens"].append(first_token)
            pending_tokens = torch.zeros(args.batch, 1, dtype=torch.int32)
            pending_tokens[0, 0] = first_token
            positions = [513] + [-1] * (args.batch - 1)
            permutation = list(range(args.batch))
            permutation[0], permutation[target_slot] = target_slot, 0
            for step in range(args.steps):
                output = adapter.decode_forward(
                    tokens=pending_tokens if step == 0 else None,
                    start_pos=positions if step == 0 else None,
                    page_table=page if step == 0 else None,
                    kv_cache=cache,
                    sampling_params=params,
                    enable_trace=True,
                    read_from_device=True,
                    slot_remap=permutation if step == 0 else None,
                    reload_inputs=step == 0,
                    reload_page_table=False,
                    reload_sampling_params=step == 0,
                    reset_sampling_state=step == 0,
                )
                token = int(output.reshape(args.batch)[0])
                # Diagnostic observation only: sampler feedback already resides
                # on device. No argmax or host token write in steady decode.
                vector = g._read_logits(g.tt_logits)[0, 0].clone()
                vectors.append(vector)
                emitted.append(token)
                phase_report["tokens"].append(token)
                row = dict(
                    output_index=step + 1,
                    stage="decode",
                    consumed_position=513 + step,
                    selected_token=token,
                    logits=stats(vector),
                )
                phase_report["rows"].append(row)
                if step % 8 == 0:
                    persist()
            phase_report.update(
                text=tokenizer.decode(emitted),
                counters=dict(g.counters),
                trace_ids=dict(g.traces),
                trace_ids_unchanged=g.traces == trace_ids,
            )
            if g.traces != trace_ids:
                raise AssertionError("Trace IDs changed during diagnostic phases")
            if expected:
                first_http_difference = next(
                    (i for i, (a, b) in enumerate(zip(expected["http_tokens"], emitted)) if a != b), None
                )
                phase_report["http_comparison"] = dict(
                    label=expected["http_label"],
                    exact_tokens=emitted == expected["http_tokens"],
                    compared_outputs=len(emitted),
                    first_divergence=first_http_difference,
                    expected_tokens=expected["http_tokens"],
                    expected_text=expected["http_text"],
                    native_chunks_exact=True,
                    companions_active=False,
                )
                if phase != "control":
                    raw = args.output.with_suffix(f".{phase}_logits.pt")
                    torch.save(dict(logits=torch.stack(vectors), tokens=emitted), raw)
                    phase_report["raw_logits"] = str(raw)
            if reference_vectors is None:
                reference_vectors, reference_tokens = vectors, emitted
                raw = args.output.with_suffix(".control_logits.pt")
                torch.save(dict(logits=torch.stack(vectors), tokens=emitted), raw)
                phase_report["raw_logits"] = str(raw)
            else:
                first_divergence = next((i for i, (a, b) in enumerate(zip(reference_tokens, emitted)) if a != b), None)
                phase_report["first_token_divergence"] = first_divergence
                phase_report["equal_tokens"] = emitted == reference_tokens
                phase_report["prefill_comparison"] = compare_logits(reference_vectors[0], vectors[0])
                for i, (control, candidate) in enumerate(zip(reference_vectors, vectors)):
                    phase_report["rows"][i]["control_comparison"] = compare_logits(control, candidate)
                    phase_report["rows"][i]["same_consumed_history"] = reference_tokens[:i] == emitted[:i]
                if first_divergence is not None:
                    raw = args.output.with_suffix(f".{phase}.first_divergence.pt")
                    indices = sorted({0, max(0, first_divergence - 1), first_divergence})
                    torch.save(
                        dict(
                            indices=indices,
                            control_logits=torch.stack([reference_vectors[i] for i in indices]),
                            candidate_logits=torch.stack([vectors[i] for i in indices]),
                            control_tokens=reference_tokens,
                            candidate_tokens=emitted,
                        ),
                        raw,
                    )
                    phase_report["first_divergence_logits"] = str(raw)
                if phase == "repeat_control" and emitted != reference_tokens:
                    phase_report["state_or_repeat_nondeterminism"] = True
            if phase in ("mixed", "forced_tail"):
                pairwise_inputs[phase] = (vectors, emitted)
                if set(pairwise_inputs) == {"mixed", "forced_tail"}:
                    mixed_vectors, mixed_tokens = pairwise_inputs["mixed"]
                    forced_vectors, forced_tokens = pairwise_inputs["forced_tail"]
                    count = min(33, len(mixed_tokens), len(forced_tokens))
                    comparisons = []
                    for index in range(count):
                        comparisons.append(
                            dict(
                                output_index=index,
                                same_consumed_history=mixed_tokens[:index] == forced_tokens[:index],
                                mixed_token=mixed_tokens[index],
                                forced_tail_token=forced_tokens[index],
                                logits=compare_logits(mixed_vectors[index], forced_vectors[index]),
                            )
                        )
                    raw = args.output.with_suffix(".mixed_vs_forced_tail.pt")
                    torch.save(
                        dict(
                            mixed_logits=torch.stack(mixed_vectors[:count]),
                            forced_tail_logits=torch.stack(forced_vectors[:count]),
                            mixed_tokens=mixed_tokens[:count],
                            forced_tail_tokens=forced_tokens[:count],
                        ),
                        raw,
                    )
                    report["mixed_vs_forced_tail"] = dict(
                        reference="mixed",
                        candidate="forced_tail",
                        compared_outputs=count,
                        includes_33_outputs=count == 33,
                        prefill_comparison=comparisons[0]["logits"],
                        first_33_tokens_equal=mixed_tokens[:count] == forced_tokens[:count],
                        first_33_logits_exact=all(row["logits"]["exact"] for row in comparisons),
                        first_token_divergence=next(
                            (i for i in range(count) if mixed_tokens[i] != forced_tokens[i]), None
                        ),
                        rows=comparisons,
                        raw_logits=str(raw),
                    )
            print(
                json.dumps({k: v for k, v in phase_report.items() if k not in ("rows", "trace_ids", "tokens")}),
                flush=True,
            )
            persist()
        report.update(
            status="completed",
            classification="Inspect same-history logits, margins and repeat_control; divergent reduced text alone is not an accuracy finding",
        )
        if fixture:
            matched = all(phase["http_comparison"]["exact_tokens"] for phase in report["phases"])
            report["matched_live_token_status"] = "pass" if matched else "failed"
            report[
                "classification"
            ] = "Full-model same-physical-chunk differential; inspect exact HTTP matches and same-history logits. This diagnostic does not itself change serving acceptance."
            if not matched:
                raise AssertionError(
                    "Canonical solo outputs failed to match one or both live physical-shape references"
                )
    except BaseException:
        report.update(status="failed", failure=traceback.format_exc())
        raise
    finally:
        persist()
        try:
            if adapter is not None and adapter.generator is not None:
                adapter.release_persistent_capture()
                report["shutdown_counters"] = dict(adapter.generator.counters)
        finally:
            # Even a trace-release failure must not skip the mesh close attempt.
            report["closed_mesh"] = False
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
                    report["closed_mesh"] = True
            finally:
                persist()


if __name__ == "__main__":
    main()
