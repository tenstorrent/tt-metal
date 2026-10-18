# SPDX-License-Identifier: Apache-2.0
"""Reduced real-layer compact-decode differential; exclusive TP4 only.

Standalone diagnostic readbacks/synchronization are not serving instrumentation.
Exercise bucket boundaries, nonprefix fallback, reactivation, physical slot
permutations, page growth, and actual overlapping token snapshots.
"""

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")

import torch
from transformers import AutoConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import vllm_adapter_device_probe as base
from models.autoports.qwen_qwen3_8_flash_next.tt.generator_vllm import QwenFlashNextForCausalLM

ROOT = Path(__file__).resolve().parents[1]
BATCH = 32
COUNTS = [32, 17, 16, 9, 8, 5, 4, 3, 1, -1, 7, 20, 32]
SEGMENT = 8


def active_mask(segment):
    n = COUNTS[segment]
    return [i in (0, 3, 7, 15) for i in range(BATCH)] if n < 0 else [i < n for i in range(BATCH)]


def bucket(mask):
    n = sum(mask)
    if mask != [i < n for i in range(BATCH)]:
        return 32
    return next((b for b in (4, 8, 16) if n <= b), 32)


def public_state_hashes(model):
    """Hash every public request/page on each ASIC; omit private scratch only.

    Read complete tensors through the existing supported mesh composer (FP8
    cannot use the single-shard torch conversion), then slice on the host.
    These blocking diagnostics occur only after all test traffic is retired.
    """
    result = {}
    for index, layer in zip(model.layer_indices, model.layers):
        attention = layer.attention
        names = (
            ("recurrent_bank", "conv_bank")
            if layer.kind == "linear_attention"
            else ("packed_cache", "raw_key_cache", "pooled_cache")
        )
        for name in names:
            tensor = getattr(attention, name)
            # Concatenate shards along the leading dimension; all cache storage
            # is replicated in geometry but contains ASIC-specific data.
            value = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh, dim=0))
            shards = value.chunk(4, dim=0)
            digests = []
            for shard in shards:
                if name.endswith("bank"):
                    public = shard[: model.max_requests]
                elif name == "packed_cache":
                    public = shard[:, :, : model.public_pool_tokens]
                elif name == "pooled_cache":
                    public = shard[:, :, : model.public_pool_tokens // 4]
                else:
                    # Raw index storage is [physical_pages,1,32,128].
                    public = shard[: model.public_pool_tokens // 32]
                raw = public.contiguous().view(torch.uint8).numpy().tobytes()
                digests.append(hashlib.sha256(raw).hexdigest())
            result[f"{index}/{name}"] = digests
            del value, shards
        if layer.ple is not None:
            for name, tensor in vars(layer.ple_cache).items():
                value = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh, dim=0))
                limit = model.max_requests * (9 if name == "conv_history" else 1)
                result[f"{index}/ple/{name}"] = [
                    hashlib.sha256(shard[:limit].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    for shard in value.chunk(4, dim=0)
                ]
                del value
    return result


def run(adapter, mode, compact, overlap, blocks, reference, report, save):
    g = adapter.generator
    g.compact_decode_enabled = compact
    g.reset()
    adapter.slot_map = list(range(BATCH))
    order = list(range(BATCH))
    lengths = [31 + i % 31 for i in range(BATCH)]
    positions = dict(enumerate(lengths))
    pages = base.RequestPages(blocks)
    prompt = torch.zeros(BATCH, max(lengths), dtype=torch.int32)
    for request, length in enumerate(lengths):
        ids = g.tokenizer.encode(
            (f"Request {request}: A botanist describes sunlight, leaves, water and photosynthesis. " * 20),
            add_special_tokens=False,
        )[:length]
        assert len(ids) == length
        prompt[request, :length] = torch.tensor(ids)
        pages.ensure(request, length - 1, -1)
    first, _ = adapter.prefill_forward(
        prompt,
        pages.table(order),
        adapter.kv_cache,
        prompt_lens=lengths,
        empty_slots=order,
        start_pos=[0] * BATCH,
        sampling_params=base.sampling_params(mode, order),
        enable_trace=True,
    )
    latest = dict(zip(order, first.reshape(BATCH).tolist()))
    result = dict(
        mode=mode,
        compact=compact,
        overlap=overlap,
        outputs=[],
        steps=[],
        prefill_tokens=latest.copy(),
        overlapping_submissions=0,
        allocations=pages.events,
    )
    report["phases"].append(result)
    if reference is not None:
        assert latest == reference["prefill_tokens"]
    pending = None

    def retire(item):
        step, rows, mask, host, events = item
        for event in events:
            ttnn.event_synchronize(event)
        tokens = adapter.process_decode_output_host(host, is_tokens=True).reshape(BATCH).tolist()
        actual = {str(r): int(t) for r, t, enabled in zip(rows, tokens, mask) if enabled}
        if reference is not None:
            assert actual == reference["outputs"][step]["tokens"], (
                mode,
                compact,
                overlap,
                step,
                actual,
                reference["outputs"][step],
            )
        result["outputs"].append(dict(step=step, tokens=actual))
        latest.update({int(r): t for r, t in actual.items()})

    start = time.perf_counter()
    for step in range(len(COUNTS) * SEGMENT):
        segment = step // SEGMENT
        transition = step % SEGMENT == 0
        if transition and pending is not None:
            retire(pending)
            pending = None
        permutation = None
        if transition and segment in (2, 6, 10):
            permutation = list(range(3, BATCH)) + [0, 1, 2]
            order = [order[i] for i in permutation]
        mask = active_mask(segment)
        grew = False
        for request, enabled in zip(order, mask):
            if enabled:
                grew = pages.ensure(request, positions[request], step) or grew
        reload_inputs = not overlap or transition
        page_only = grew and not reload_inputs
        before = dict(g.counters)
        out = adapter.decode_forward(
            tokens=torch.tensor([[latest[r]] for r in order], dtype=torch.int32)
            if reload_inputs
            else torch.full((BATCH, 1), 7777, dtype=torch.int32),
            start_pos=[positions[r] if enabled else -1 for r, enabled in zip(order, mask)]
            if reload_inputs
            else [base.CONTEXT + 99] * BATCH,
            page_table=pages.table(order) if reload_inputs or page_only else torch.full((BATCH, 1), -1),
            kv_cache=adapter.kv_cache,
            sampling_params=base.sampling_params(mode, order),
            enable_trace=True,
            read_from_device=False,
            slot_remap=permutation,
            reload_inputs=reload_inputs,
            reload_page_table=page_only,
            reload_sampling_params=transition,
            reset_sampling_state=transition,
        )
        host, events = adapter.read_decode_output(out, async_read=True)
        current = (step, order.copy(), mask.copy(), host, events)
        delta = {k: g.counters[k] - before.get(k, 0) for k in g.counters}
        base.check_delta(delta, reload_inputs=reload_inputs, page_only=page_only, table_changed=grew)
        expected_bucket = bucket(mask) if compact else 32
        actual_bucket = getattr(g, "last_decode_execution_batch", None)
        assert actual_bucket == expected_bucket, (step, actual_bucket, expected_bucket)
        result["steps"].append(
            dict(
                step=step,
                active=mask,
                order=order.copy(),
                physical_slots=adapter.slot_map.copy(),
                execution_batch=actual_bucket,
                reload_inputs=reload_inputs,
                page_only=page_only,
                counters=delta,
            )
        )
        for r, enabled in zip(order, mask):
            if enabled:
                positions[r] += 1
        if overlap:
            if pending is not None:
                result["overlapping_submissions"] += 1
                retire(pending)
            pending = current
        else:
            retire(current)
        save()
    if pending is not None:
        retire(pending)
    ttnn.synchronize_device(adapter.mesh)
    actual_positions = ttnn.to_torch(ttnn.get_device_tensors(g.tt_pos)[0])[:, 0].tolist()
    assert actual_positions == [positions[r] for r in order]
    if overlap:
        assert result["overlapping_submissions"] >= len(COUNTS) * (SEGMENT - 1)
    result.update(
        status="passed",
        elapsed_seconds=time.perf_counter() - start,
        final_positions=actual_positions,
        final_order=order,
        final_tokens=latest,
    )
    result["public_state_sha256_per_asic"] = public_state_hashes(g.model)
    if reference is not None:
        assert result["public_state_sha256_per_asic"] == reference["public_state_sha256_per_asic"], (
            "Public state differs after bucket transitions/reactivation",
            mode,
            compact,
            overlap,
        )
    save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-region-size", type=int, default=400000000)
    args = parser.parse_args()
    assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
    assert ttnn._ttnn.operations.trace.trace_allocation_tracking_enabled()
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") in ("", "0")
    assert os.environ.get("TT_METAL_WATCHER", "0") in ("", "0")
    args.output.mkdir(parents=True, exist_ok=False)
    base.BATCH = BATCH
    os.environ["QWEN_FLASH_SERVING_LAYERS"] = "0,1,3"
    os.environ["QWEN_FLASH_CHECKPOINT"] = args.checkpoint
    os.environ["QWEN_FLASH_TRACE_EVENTS"] = str(args.output / "events.jsonl")
    report = dict(
        status="started",
        scope="Reduced real layers0/1/3; not a full-model or server measurement",
        phases=[],
        tracking=dict(native=True, python=True, skip_program_cache=False, exemptions=[]),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "tt/generator.py",
                ROOT / "tt/decode_view.py",
                ROOT / "tt/generator_vllm.py",
                ROOT / "tt/multichip_decoder.py",
            ]
        },
    )
    (args.output / "probe_source.py.txt").write_bytes(Path(__file__).read_bytes())

    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2, default=str) + "\n")

    adapter = mesh = None
    save()
    try:
        torch.set_num_threads(8)
        ttnn.CONFIG.throw_exception_on_fallback = True
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
        cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
        adapter = QwenFlashNextForCausalLM.initialize_vllm_model(cfg, mesh, BATCH, base.CONTEXT)
        blocks = base.CONTEXT // 32 + BATCH
        cache = adapter.allocate_kv_cache((blocks, 1, 32, 256), torch.float8_e4m3fn, 1)
        adapter.warmup_model_prefill(cache, enable_trace=True, can_sample_on_device=True)
        g = adapter.generator
        assert hasattr(g, "compact_decode_enabled"), "Probe requires explicit baseline/candidate toggle"
        traces = dict(g.traces)
        captures = g.counters["captures"]
        report.update(
            device_ids=list(mesh.get_device_ids()),
            prepared_traces=traces,
            precision=g.model.precision_policy,
            context=g.model.context,
            memory=base.view(mesh),
            trace_memory=base.view(mesh, ttnn.BufferType.TRACE),
        )
        for mode in ("greedy", "sample"):
            control = run(adapter, mode, False, False, blocks, None, report, save)
            run(adapter, mode, True, False, blocks, control, report, save)
            run(adapter, mode, True, True, blocks, control, report, save)
            run(adapter, mode, True, True, blocks, control, report, save)
            assert traces == g.traces and captures == g.counters["captures"]
        report["status"] = "passed"
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if adapter is not None:
            adapter.release_persistent_capture()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        report["closed_mesh"] = mesh is not None
        save()


if __name__ == "__main__":
    main()
