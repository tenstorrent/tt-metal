"""Reduced real-layer async adapter differential; run only with exclusive devices.

This is a targeted adapter test, not a scheduler/server or model-quality gate.
Tracking is enabled before importing TTNN; this probe is not a performance run.
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
from transformers import AutoConfig
from vllm_tt_plugin.model_input import TTSamplingParams

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.multichip_capacity_probe import view
from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM
from models.autoports.qwen_qwen3_8_flash_next.tt.precision import runtime_summary

ROOT = Path(__file__).resolve().parents[1]
BATCH = 3
CONTEXT = 262144
BLOCK = 32
LENGTHS = [31, 46, 61]
PERMUTATIONS = {12: [2, 0, 1], 27: [1, 2, 0]}


class RequestPages:
    """Deterministic allocator-driven growth at each currently consumed position.

    Deliberately uses the high end of the public pool, with disjoint blocks for
    distinct requests. This is a miniature allocator, not the vLLM scheduler.
    """

    def __init__(self, total_blocks):
        self.total_blocks = total_blocks
        self.allocated = [[] for _ in range(BATCH)]
        self.next_block = total_blocks - 1
        self.events = []

    def ensure(self, request, position, step):
        added = []
        while len(self.allocated[request]) <= position // BLOCK:
            if self.next_block <= 0:
                raise RuntimeError("Probe allocator exhausted the public pool")
            block = self.next_block
            self.next_block -= 1
            self.allocated[request].append(block)
            added.append(block)
        if added:
            self.events.append(dict(step=step, request=request, used_position=position, added_blocks=added))
        return bool(added)

    def table(self, order):
        table = torch.zeros(BATCH, CONTEXT // BLOCK, dtype=torch.int32)
        for row, request in enumerate(order):
            blocks = self.allocated[request]
            table[row, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        return table


def sampling_params(mode, order):
    return TTSamplingParams(
        temperature=[0.0 if mode == "greedy" else 0.8] * BATCH,
        top_k=[1 if mode == "greedy" else 8] * BATCH,
        top_p=[1.0] * BATCH,
        seed=[101 + request * 100 for request in order],
        enable_log_probs=[False] * BATCH,
    )


def read_pending(adapter, pending):
    step, order, host, events = pending
    for event in events:
        ttnn.event_synchronize(event)
    values = adapter.process_decode_output_host(host, is_tokens=True).reshape(BATCH).clone().tolist()
    return step, {request: int(token) for request, token in zip(order, values)}


def check_delta(delta, *, reload_inputs, page_only, table_changed):
    if not reload_inputs:
        for key in ("token_refreshes", "position_refreshes", "slot_refreshes", "activity_refreshes"):
            if delta.get(key, 0) != 0:
                raise AssertionError((key, delta))
    if not reload_inputs and not page_only:
        for key in ("page_table_refreshes", "slot_refreshes"):
            if delta.get(key, 0) != 0:
                raise AssertionError((key, delta))
    if page_only and table_changed and delta.get("page_table_refreshes", 0) != 1:
        raise AssertionError(("Missing changed-page copy", delta))
    if delta.get("decode_replays", 0) != 1:
        raise AssertionError(("Exactly one decode replay required", delta))
    if delta.get("logits_readbacks", 0) != 0:
        raise AssertionError(("Unexpected logits readback", delta))


def run_traffic(adapter, mode, overlap, steps, total_blocks, reference, persist, report):
    generator = adapter.generator
    generator.reset()
    adapter.slot_map = list(range(BATCH))
    order = list(range(BATCH))
    allocator = RequestPages(total_blocks)
    texts = [
        "The botanist studies leaves, sunlight, water, and photosynthesis. ",
        "A sailor navigates the ocean using stars, maps, and a compass. ",
        "A mathematician explains prime numbers, factors, and arithmetic. ",
    ]
    prompt = torch.zeros(BATCH, max(LENGTHS), dtype=torch.int32)
    for request, length in enumerate(LENGTHS):
        text = texts[request % len(texts)]
        if BATCH > len(texts):
            text = f"Request {request}: " + text
        ids = generator.tokenizer.encode(text * 20, add_special_tokens=False)[:length]
        if len(ids) != length:
            raise AssertionError("Prompt construction did not reach its logical length")
        prompt[request, :length] = torch.tensor(ids, dtype=torch.int32)
        allocator.ensure(request, length - 1, -1)
    first, deltas = adapter.prefill_forward(
        prompt,
        allocator.table(order),
        adapter.kv_cache,
        prompt_lens=LENGTHS,
        empty_slots=order,
        start_pos=[0] * BATCH,
        sampling_params=sampling_params(mode, order),
        enable_trace=True,
    )
    if deltas.tolist() != [0] * BATCH:
        raise AssertionError("Text prefill must report zero M-RoPE deltas")
    latest = dict(zip(order, first.reshape(BATCH).tolist()))
    phase = dict(
        mode=mode,
        overlap=overlap,
        prompt_lengths=LENGTHS,
        prompt_tokens=prompt.tolist(),
        prefill_tokens=latest,
        rows=[],
        outputs=[],
        allocations=allocator.events,
        memory_before=view(adapter.mesh),
        submitted_next_before_previous_read=0,
    )
    report["phases"].append(phase)
    if reference is not None and latest != reference["prefill_tokens"]:
        raise AssertionError(("Prefill differs from synchronous control", mode, latest, reference["prefill_tokens"]))
    pending = None

    def retire(item):
        step, by_request = read_pending(adapter, item)
        phase["outputs"].append(dict(step=step, tokens=by_request))
        if reference is not None and by_request != reference["outputs"][step]["tokens"]:
            raise AssertionError(
                ("Async snapshot/control mismatch", mode, step, by_request, reference["outputs"][step]["tokens"])
            )
        return by_request

    for step in range(steps):
        permutation = PERMUTATIONS.get(step)
        if permutation is not None:
            # Match plugin reload authority: drain pending output before a new
            # host token/layout submission. Ordinary steady steps stay overlapped.
            if pending is not None:
                latest = retire(pending)
                pending = None
            order = [order[i] for i in permutation]
        positions = [LENGTHS[request] + step for request in order]
        grew = False
        for request, position in zip(order, positions):
            grew = allocator.ensure(request, position, step) or grew
        reload_inputs = not overlap or step == 0 or permutation is not None
        page_only = grew and not reload_inputs
        before = dict(generator.counters)
        submitted = adapter.decode_forward(
            tokens=torch.tensor([[latest[r]] for r in order], dtype=torch.int32)
            if reload_inputs
            else torch.full((BATCH, 1), 7777, dtype=torch.int32),
            start_pos=positions if reload_inputs else [CONTEXT + 100 + step] * BATCH,
            page_table=allocator.table(order)
            if reload_inputs or page_only
            else torch.full((BATCH, 1), -1, dtype=torch.int32),
            kv_cache=adapter.kv_cache,
            sampling_params=sampling_params(mode, order),
            enable_trace=True,
            read_from_device=False,
            slot_remap=permutation,
            reload_inputs=reload_inputs,
            reload_page_table=page_only,
            reload_sampling_params=step == 0 or permutation is not None,
            reset_sampling_state=step == 0 or permutation is not None,
            rope_deltas_all_users=[0] * BATCH if reload_inputs else None,
        )
        # The deferred CPU snapshot must be queued before another replay writes
        # the same persistent output. Resolve it only after the next submission.
        host, events = adapter.read_decode_output(submitted, async_read=True)
        current = (step, order.copy(), host, events)
        delta = {k: generator.counters[k] - before.get(k, 0) for k in generator.counters}
        check_delta(delta, reload_inputs=reload_inputs, page_only=page_only, table_changed=grew)
        phase["rows"].append(
            dict(
                step=step,
                order=order.copy(),
                consumed_positions=positions,
                reload_inputs=reload_inputs,
                page_only=page_only,
                slot_remap=permutation,
                state_slots=adapter.slot_map.copy(),
                counter_delta=delta,
            )
        )
        if overlap:
            if pending is not None:
                phase["submitted_next_before_previous_read"] += 1
                latest = retire(pending)
            pending = current
        else:
            latest = retire(current)
        persist()
    if pending is not None:
        latest = retire(pending)
    ttnn.synchronize_device(adapter.mesh)
    final_positions = ttnn.to_torch(ttnn.get_device_tensors(generator.tt_pos)[0])[:, 0].tolist()
    expected_positions = [LENGTHS[request] + steps for request in order]
    if final_positions != expected_positions:
        raise AssertionError(("Position advancement", final_positions, expected_positions))
    if overlap and phase["submitted_next_before_previous_read"] < steps - len(PERMUTATIONS) - 1:
        raise AssertionError("Insufficient actual overlapping submissions")
    for request in range(BATCH):
        crossings = [e for e in allocator.events if e["request"] == request and e["step"] >= 0]
        if len(crossings) < 2:
            raise AssertionError(("Multiple used-page boundaries not covered", request))
    changing_steps = sum(
        phase["outputs"][i]["tokens"] != phase["outputs"][i - 1]["tokens"]
        for i in range(1, steps)
        if i not in PERMUTATIONS
    )
    phase.update(
        status="pass",
        final_positions=final_positions,
        final_tokens=latest,
        successive_snapshot_changes=changing_steps,
        memory_after=view(adapter.mesh),
        trace_ids=dict(generator.traces),
        counters=dict(generator.counters),
    )
    persist()
    return phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch", type=int, choices=(3, 32), default=3)
    parser.add_argument("--steps", type=int, default=66)
    parser.add_argument("--sampling", choices=("greedy", "sample", "both"), default="both")
    parser.add_argument("--trace-region-size", type=int, default=160000000)
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/adapter_device_probe.json")
    args = parser.parse_args()
    global BATCH, LENGTHS, PERMUTATIONS
    BATCH = args.batch
    if BATCH == 32:
        LENGTHS = [31 + request % 31 for request in range(BATCH)]
        PERMUTATIONS = {12: list(range(1, BATCH)) + [0], 27: list(range(2, BATCH)) + [0, 1]}
    if args.steps < 66:
        raise ValueError("At least66 steps cover multiple used page boundaries per distinct request")
    if os.environ.get("TT_METAL_DEVICE_PROFILER") not in (None, "0"):
        raise RuntimeError("Disable the device profiler for this serving adapter probe")
    os.environ["QWEN_FLASH_SERVING_LAYERS"] = "0,1,3"
    os.environ["QWEN_FLASH_CHECKPOINT"] = args.checkpoint
    os.environ["QWEN_FLASH_TRACE_EVENTS"] = str(args.output.with_suffix(".events.jsonl"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or Path(os.environ["QWEN_FLASH_TRACE_EVENTS"]).exists():
        raise FileExistsError("Use a fresh --output to preserve prior evidence")
    report = dict(
        status="running",
        scope="Reduced real layers0,1,3 adapter; no server/scheduler or quality claim",
        args=vars(args),
        batch=BATCH,
        context=CONTEXT,
        phases=[],
        tracking={
            k: os.environ.get(k, "0")
            for k in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE")
        },
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / "tt/generator_vllm.py", ROOT / "tt/generator.py")
        },
    )

    def persist():
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")

    mesh = None
    adapter = None
    persist()
    try:
        torch.set_num_threads(8)
        ttnn.CONFIG.throw_exception_on_fallback = True
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
        config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
        adapter = QwenFlashNextForCausalLM.initialize_vllm_model(config, mesh, BATCH, CONTEXT)
        total_blocks = CONTEXT // BLOCK + BATCH
        cache = adapter.allocate_kv_cache((total_blocks, 1, BLOCK, 256), torch.float8_e4m3fn, 1)
        generator = adapter.generator
        if cache is not generator.kv_cache or cache is not generator.model.cache:
            raise AssertionError("Adapter must pass the exact vLLM cache through")
        selected = json.loads((ROOT / "doc/datatype_sweep/selected_precision_config.json").read_text())
        if generator.model.precision_policy != selected:
            raise AssertionError("Serving changed the selected precision policy")
        adapter.warmup_model_prefill(cache, enable_trace=True, can_sample_on_device=True)
        adapter.warmup_model_decode(cache, enable_trace=True, can_sample_on_device=True)
        baseline_ids = dict(generator.traces)
        baseline_counts = dict(generator.counters)
        report.update(
            public_pool_tokens=generator.model.public_pool_tokens,
            public_blocks=total_blocks,
            private_scratch_base=generator.model.scratch_page_base,
            runtime=runtime_summary(generator.model),
            prepared_trace_ids=baseline_ids,
            prepared_counters=baseline_counts,
            prepared_memory=view(mesh),
            prepared_trace_memory=view(mesh, ttnn.BufferType.TRACE),
        )
        modes = ["greedy", "sample"] if args.sampling == "both" else [args.sampling]
        for mode in modes:
            control = run_traffic(adapter, mode, False, args.steps, total_blocks, None, persist, report)
            run_traffic(adapter, mode, True, args.steps, total_blocks, control, persist, report)
            run_traffic(adapter, mode, True, args.steps, total_blocks, control, persist, report)
            if generator.traces != baseline_ids or generator.counters["captures"] != baseline_counts["captures"]:
                raise AssertionError("Trace lifetime violated across request/sampling transitions")
            if generator.counters["releases"] != baseline_counts.get("releases", 0):
                raise AssertionError("A prepared trace was retired during traffic")
        if not any(p["overlap"] and p["successive_snapshot_changes"] > 0 for p in report["phases"]):
            raise AssertionError("Unchanged output tokens make snapshot identity evidence nondiscriminating")
        report.update(
            status="pass",
            final_counters=dict(generator.counters),
            memory_final=view(mesh),
            exact_async_control=True,
            repeated_async_control=True,
            caveat="Automatic vLLM scheduler allocator/page-growth proof remains a separate server gate",
        )
        print("ADAPTER_DEVICE_PROBE_PASS", str(args.output), flush=True)
    except BaseException:
        report.update(status="failed", failure=traceback.format_exc())
        raise
    finally:
        persist()
        if adapter is not None and adapter.generator is not None:
            adapter.release_persistent_capture()
            report["shutdown_counters"] = dict(adapter.generator.counters)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        report["closed_mesh"] = mesh is not None
        persist()


if __name__ == "__main__":
    main()
