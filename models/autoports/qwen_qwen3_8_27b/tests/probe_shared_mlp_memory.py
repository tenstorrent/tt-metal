# SPDX-License-Identifier: Apache-2.0
"""Standalone paired-arm MLP memory/correctness fixture; serialized TT lease required.

Run each arm in a fresh process with QWEN38_SHARED_MLP_WEIGHTS=0/1. Native
logical context and page-table shapes are retained; the explicitly bounded
external pool only backs this fixture's short requests. This is not a B32
native-context capacity claim or a qualitative model judgment.
"""
import argparse
import gc
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--full-stack", action="store_true")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lengths", type=int, nargs="+", default=[1, 33, 255, 256, 257, 511, 512, 513])
    p.add_argument("--outputs", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument(
        "--capacity-pool-ids",
        type=int,
        help="Standalone production-shaped checkpoint512 pool; not a live serving profiler",
    )
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--min-free-bytes", type=int, default=512 * 1024**2)
    p.add_argument("--fragment-guard-per-bank", type=int, default=8 * 1024**2)
    a = p.parse_args()
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_8_27b.tests.full_model_evidence import runtime_build
    from models.autoports.qwen_qwen3_8_27b.tests.native_fixture_cleanup import finalize_run
    from models.autoports.qwen_qwen3_8_27b.tests.native_fixture_metadata import capture_linear_calls
    from models.autoports.qwen_qwen3_8_27b.tests.physical_memory_inventory import AllocationLedger, tensor_aliases
    from models.autoports.qwen_qwen3_8_27b.tt.generator import Generator
    from models.autoports.qwen_qwen3_8_27b.tt.model import Model
    from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import PagedState
    from models.autoports.qwen_qwen3_8_27b.tt.serving_state import ServingStatePool

    if (
        os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1"
        or os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0"
    ):
        raise RuntimeError("Correctness fixture requires trace tracking and program-cache tracking")
    capacity_mode = a.capacity_pool_ids is not None
    if a.cycles < 1 or a.cycles > 3:
        raise ValueError("Use one to three fixed stress cycles")
    if capacity_mode and (not a.full_stack or a.batch != 32 or a.outputs != 128):
        raise ValueError("Capacity screen requires all64 layers, B32 and128 fixed outputs")
    if capacity_mode and (a.min_free_bytes < 512 * 1024**2 or a.fragment_guard_per_bank < 8 * 1024**2):
        raise ValueError("Capacity screen cannot weaken the registered512MiB/8MiB-per-bank guards")
    if capacity_mode and (min(a.lengths) < 1 or max(a.lengths) > 16384):
        raise ValueError("Capacity stress lengths must lie within1..16384; native context remains262144")
    if not capacity_mode and (max(a.lengths) + a.outputs > 1024 or min(a.lengths) < 1):
        raise ValueError("Short fixture pool must cover every request")
    # paged_update_cache requires table_width <= total physical cache pages.
    # Preserve native table width; supply the smallest legal aggregate pool.
    pool_context = None if capacity_mode else max(1024, math.ceil(262144 / (a.batch * 256)) * 256)
    trace_region = 128 * 1024**2 if capacity_mode else 0
    if capacity_mode:
        os.environ["QWEN38_MEMORY_ATTESTATION"] = "1"
        # Each row owns whole attention IDs; reserve two IDs/group/row and null0.
        row_blocks = math.ceil((max(a.lengths) + a.outputs) / 512)
        required_ids = a.batch * row_blocks + 2 * 3 * a.batch + 1
        if a.capacity_pool_ids < required_ids:
            raise ValueError(f"Fixed stress needs{required_ids} distinct global IDs, got{a.capacity_pool_ids}")
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    result = dict(
        status="running",
        throw_exception_on_fallback=bool(ttnn.CONFIG.throw_exception_on_fallback),
        shared=os.environ.get("QWEN38_SHARED_MLP_WEIGHTS", "0"),
        shared_mlp_readers=int(os.environ.get("QWEN38_SHARED_MLP_READERS", "1")),
        shared_mlp_cores=int(os.environ.get("QWEN38_SHARED_MLP_CORES", "80")),
        batch=a.batch,
        layers="all64" if a.full_stack else [0, 3],
        native_context=262144,
        physical_pool_tokens_per_row=pool_context,
        requested_outputs=a.outputs,
        capacity_pool_ids=a.capacity_pool_ids,
        capacity_image_digest=os.environ.get("QWEN38_CAPACITY_IMAGE_DIGEST"),
        capacity_study_id=os.environ.get("QWEN38_CAPACITY_STUDY_ID"),
        stress_cycles=a.cycles,
        configured_trace_region_bytes=trace_region,
        capacity_guards=dict(
            min_normal_free_bytes=a.min_free_bytes,
            fragment_guard_bytes_per_bank=a.fragment_guard_per_bank,
            min_trace_free_bytes=8 * 1024**2,
        ),
        scope="native standalone bounded external pool; not serving retention/capacity or quality",
        runtime=runtime_build(),
        records=[],
        environment={
            key: os.environ.get(key)
            for key in (
                "TT_METAL_HOME",
                "TT_METAL_RUNTIME_ROOT",
                "TT_METAL_CACHE",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_WATCHER",
                "QWEN38_PRECISION_CONFIG",
                "QWEN38_GROUPED_RECURRENCE",
                "QWEN38_MASKED_RECURRENCE",
                "QWEN38_FLAT_PREFILL",
                "QWEN38_MEMORY_ATTESTATION",
            )
        },
    )
    root = Path(__file__).resolve().parents[1]
    result["source_sha256"] = {
        str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in [
            Path(__file__),
            root / "tests/physical_memory_inventory.py",
            root / "tests/native_fixture_cleanup.py",
            root / "tests/native_fixture_metadata.py",
            root / "tt/selected_precision_config.json",
            root / "tests/full_model_evidence.py",
            *sorted((root / "tt").glob("*.py")),
        ]
    }
    try:
        result["vllm_distribution_version"] = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        result["vllm_distribution_version"] = None
    result["checkpoint_file_sha256"] = {
        name: hashlib.sha256((Path(a.checkpoint) / name).read_bytes()).hexdigest()
        for name in ("config.json", "tokenizer_config.json")
        if (Path(a.checkpoint) / name).is_file()
    }
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=trace_region)
    gen = None
    state_pool = None
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)

    def archive_graph(graph, label):
        if capacity_mode:
            events = [node for node in graph if node["node_type"] in ("buffer_allocate", "buffer_deallocate")]
            path = target.with_name(target.stem + "." + label + ".allocations.json.gz")
            with gzip.open(path, "wt", compresslevel=1) as stream:
                json.dump(events, stream)
            result.setdefault("allocation_event_artifacts", {})[label] = dict(
                path=str(path), events=len(events), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )

    def memory():
        m = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        trace = ttnn.get_memory_view(mesh, ttnn.BufferType.TRACE)
        return dict(
            banks=m.num_banks,
            allocated=m.total_bytes_allocated_per_bank * m.num_banks,
            free=m.total_bytes_free_per_bank * m.num_banks,
            capacity=m.total_bytes_per_bank * m.num_banks,
            largest_contiguous_free_per_bank=m.largest_contiguous_bytes_free_per_bank,
            trace_allocated=trace.total_bytes_allocated_per_bank * trace.num_banks,
            trace_free=trace.total_bytes_free_per_bank * trace.num_banks,
            trace_capacity=trace.total_bytes_per_bank * trace.num_banks,
        )

    try:
        baseline = memory()
        # Capture construction separately from preparation to bound host graph size.
        l1_banks = ttnn.get_memory_view(mesh, ttnn.BufferType.L1).num_banks
        ledger = AllocationLedger({"DRAM": baseline["banks"], "L1": l1_banks})
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            model = Model(a.checkpoint, mesh, layer_indices=None if a.full_stack else [0, 3])
            result["effective_precision"] = model.precision_config
            result["layer_types"] = {str(layer.layer_idx): layer.layer_type for layer in model.layers}
            if capacity_mode:

                def attention_tensor(layer):
                    return ttnn.zeros(
                        [a.capacity_pool_ids * 16, 1, 32, 256],
                        dtype=getattr(ttnn, layer.policy["kv_dtype"]),
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )

                cache = [
                    layer.allocate_state(batch_size=a.batch)
                    if layer.layer_type == "linear_attention"
                    else PagedState(attention_tensor(layer), attention_tensor(layer))
                    for layer in model.layers
                ]
                pages = row_blocks * 16
            else:
                cache = model.allocate_cache(batch_size=a.batch, context_length=pool_context)
                pages = pool_context // 32
            table = torch.zeros(a.batch, 262144 // 32, dtype=torch.int32)
            table[:, :pages] = (
                torch.arange(a.batch * pages, dtype=torch.int32) + (16 if capacity_mode else 0)
            ).reshape(a.batch, pages)
            gen = Generator(
                model,
                a.checkpoint,
                batch_size=a.batch,
                context_length=262144,
                kv_cache=cache,
                page_table=table,
                prefill_chunk_size=512,
                serving_state=capacity_mode,
            )
            if capacity_mode:
                recurrent = [layer.layer_idx for layer in model.layers if layer.layer_type == "linear_attention"]
                groups = [recurrent[i : i + 16] for i in range(0, 48, 16)]
                state_pool = ServingStatePool(
                    model,
                    cache,
                    num_blocks_by_layer={index: a.capacity_pool_ids for index in recurrent},
                    sharing_groups=groups,
                )
                assert state_pool.unique_pool_count == 16
            ttnn.synchronize_device(mesh)
            if capacity_mode:
                gc.collect()
                ttnn.synchronize_device(mesh)
        finally:
            graph = ttnn.graph.end_graph_capture()
            archive_graph(graph, "construction")
        ledger.update(graph)
        del graph

        def checkpoint(label):
            ttnn.synchronize_device(mesh)
            census = tensor_aliases({"generator": gen, "snapshot_pool": state_pool})
            m = memory()
            snap = ledger.snapshot(
                census["tensors"],
                domain=mesh.id(),
                allocator_bytes=m["allocated"],
                baseline_bytes=baseline["allocated"],
                allocator_bytes_by_domain={"TRACE": m["trace_allocated"]},
                baseline_bytes_by_domain={"TRACE": baseline["trace_allocated"]},
            )
            snap["shared_objects"] = census["shared_objects"]
            snap["allocator"] = m
            result[label] = snap
            if capacity_mode:
                roots = {
                    "attention": [(state.key, state.value) for state in cache if hasattr(state, "key")],
                    "snapshots": state_pool.pools,
                }
                pool_addresses = {
                    row["address"] for row in tensor_aliases(roots)["tensors"] if row["buffer_type"] == "DRAM"
                }
                actual_pool = sum(
                    row["physical_bytes_per_device"]
                    for row in snap["allocations"]
                    if row["type"] == "DRAM" and int(row["address"]) in pool_addresses
                )
                payload = a.capacity_pool_ids * (512 * 8704 + 16 * 801792)
                padding = ((-3 * a.capacity_pool_ids) % 8) * 5120 * 16
                assert actual_pool == payload + padding, (
                    "Physical cache owner reconciliation failed",
                    actual_pool,
                    payload,
                    padding,
                )
                result["physical_cache_pool"] = dict(
                    global_ids=a.capacity_pool_ids,
                    checkpoint_tokens=512,
                    actual_bytes_per_device=actual_pool,
                    payload_bytes_per_device=payload,
                    bank_padding_bytes_per_device=padding,
                    unique_snapshot_pools=state_pool.unique_pool_count,
                    measurement="Actual allocation-event physical owners joined to attention/snapshot buffer roots; active recurrence excluded",
                )
            target.write_text(json.dumps(result, indent=2) + "\n")

        checkpoint("after_construction")
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            if state_pool is not None:
                state_pool.warmup()
            with capture_linear_calls(result, ttnn, model):
                gen.prepare_decode()
            ttnn.synchronize_device(mesh)
            if capacity_mode:
                gc.collect()
                ttnn.synchronize_device(mesh)
        finally:
            graph = ttnn.graph.end_graph_capture()
            archive_graph(graph, "preparation")
        ledger.update(graph)
        del graph
        checkpoint("after_preparation")
        result["preparation"] = gen.last_preparation_memory
        gen.set_sampling_params(top_k=1)
        if capacity_mode:
            # Match serving's new-slot reset: never zero the whole attention pool.
            # Generator.generate still handles request/sampler buffers normally.
            original_reset = gen.reset

            def request_reset():
                bound = gen.kv_cache
                gen.kv_cache = [state for state in bound if not hasattr(state, "key")]
                try:
                    original_reset()
                finally:
                    gen.kv_cache = bound

            gen.reset = request_reset
            result["attention_pool_zeroed_per_request"] = False
            result[
                "capacity_scope"
            ] = "Standalone native production-shaped pools/warmup; exact0.30 serving admission and prefix lifecycle are separate gates"

        def digest(tensor):
            rows = []
            for local in ttnn.get_device_tensors(tensor):
                host = ttnn.to_torch(local).contiguous()
                rows.append(
                    dict(
                        shape=list(host.shape),
                        dtype=str(host.dtype),
                        sha256=hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest(),
                    )
                )
            return rows

        trace_ids = {str(k): str(v) for k, v in gen.decode_traces.items()}
        captures = gen.counters["captures"]
        maximum_request_allocation_per_bank = 0
        capacity_prompts = {}
        capacity_expected = {}
        for request_index, length in enumerate(a.lengths * a.cycles):
            # Deliberately awkward lengths, distinct rows, and a ragged last row.
            # Synthetic token IDs are mechanism coverage, not chat quality evidence.
            prompts = [
                (torch.arange(length - (i % 3 if length > 3 else 0)) + 1000 + i * 17).tolist() for i in range(a.batch)
            ]
            if capacity_mode:
                if length not in capacity_prompts:
                    rendered = []
                    for stream in range(a.batch):
                        header = f"Private research stream {stream} family 0. Continue the technical record.\n"
                        count = max(1, length - 100)
                        for _ in range(10):
                            ids = gen.tokenizer.apply_chat_template(
                                [{"role": "user", "content": header + " observation" * count}],
                                tokenize=True,
                                add_generation_prompt=True,
                                return_dict=False,
                                enable_thinking=True,
                                reasoning_effort="medium",
                                preserve_thinking=True,
                            )
                            if len(ids) == length:
                                break
                            count += length - len(ids)
                        if len(ids) != length:
                            raise ValueError("Cannot render fixed native-chat capacity prompt")
                        rendered.append(ids)
                    capacity_prompts[length] = rendered
                    target.with_suffix(".inputs.json").write_text(json.dumps(capacity_prompts))
                prompts = capacity_prompts[length]
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            try:
                tokens = gen.generate(prompts, a.outputs, stop_on_eos=False, buffered_output=True)
                ttnn.synchronize_device(mesh)
                if capacity_mode:
                    gc.collect()
                    ttnn.synchronize_device(mesh)
            finally:
                graph = ttnn.graph.end_graph_capture()
                archive_graph(graph, f"request{request_index}")
            maximum_request_allocation_per_bank = max(
                maximum_request_allocation_per_bank,
                max(
                    (
                        int(node["params"]["max_size_per_bank"])
                        for node in graph
                        if node["node_type"] == "buffer_allocate" and int(node["params"]["buffer_type"]) == 0
                    ),
                    default=0,
                ),
            )
            ledger.update(graph)
            del graph
            request_memory = memory()
            reconciled = ledger.snapshot(
                [],
                domain=mesh.id(),
                allocator_bytes=request_memory["allocated"],
                baseline_bytes=baseline["allocated"],
                allocator_bytes_by_domain={"TRACE": request_memory["trace_allocated"]},
                baseline_bytes_by_domain={"TRACE": baseline["trace_allocated"]},
            )
            states = (
                {}
                if capacity_mode
                else {
                    f"{layer.layer_idx}.{field}": digest(tensor)
                    for layer, state in zip(model.layers, cache)
                    for field, tensor in vars(state).items()
                }
            )
            record = dict(
                length=length,
                cycle=request_index // len(a.lengths),
                logical_lengths=list(map(len, prompts)),
                input_sha256=[
                    hashlib.sha256(json.dumps(row, separators=(",", ":")).encode()).hexdigest() for row in prompts
                ],
                tokens=tokens,
                logits=digest(gen.logits),
                states=states,
                generation_metrics=dict(gen.last_generation_metrics),
                allocation=dict(
                    allocator=request_memory,
                    cumulative_event_peak_bytes_per_device=reconciled["event_dram_peak_bytes_per_device"],
                    exact_allocator_reconciliation=reconciled["exact_allocator_reconciliation"],
                    scope="Allocation events cover construction, preparation and each completed request through this row; not clean timing",
                ),
            )
            result["records"].append(record)
            if capacity_mode:
                witness = (record["input_sha256"], record["tokens"], record["logits"])
                if length in capacity_expected:
                    assert witness == capacity_expected[length], "Repeated stress request changed outputs/logits"
                capacity_expected[length] = witness
                minimum_free = request_memory["capacity"] - reconciled["event_dram_peak_bytes_per_device"]
                minimum_trace_free = (
                    request_memory["trace_capacity"] - reconciled["event_trace_arena_peak_bytes_per_device"]
                )
                assert minimum_free >= a.min_free_bytes, ("Transient headroom failed", minimum_free, a.min_free_bytes)
                assert minimum_trace_free >= 8 * 1024**2, ("TRACE headroom failed", minimum_trace_free)
                assert (
                    request_memory["largest_contiguous_free_per_bank"]
                    >= maximum_request_allocation_per_bank + a.fragment_guard_per_bank
                ), "Fragmentation guard failed"
                result["capacity_observation"] = dict(
                    minimum_free_from_allocation_events=minimum_free,
                    minimum_trace_free_from_allocation_events=minimum_trace_free,
                    maximum_request_allocation_per_bank=maximum_request_allocation_per_bank,
                    largest_contiguous_free_per_bank=request_memory["largest_contiguous_free_per_bank"],
                )
            assert trace_ids == {str(k): str(v) for k, v in gen.decode_traces.items()}
            assert gen.counters["captures"] == captures
            target.write_text(json.dumps(result, indent=2) + "\n")
        # Same warmed fixed-work native path; diagnostic tracking stays enabled.
        # Clean headline performance must rerun separately with tracking disabled.
        result["diagnostic_timing"] = []
        for _ in range(0 if capacity_mode else a.repeats):
            started = time.perf_counter()
            gen.generate(prompts, a.outputs, stop_on_eos=False, buffered_output=True)
            result["diagnostic_timing"].append(
                dict(wall_s=time.perf_counter() - started, metrics=dict(gen.last_generation_metrics))
            )
        result.update(
            status="pass",
            trace_ids=trace_ids,
            captures=captures,
            exactness=(
                "Capacity screen only: compare tokens/logits with matched arm; full-state/serving lifecycle gates remain separate"
                if capacity_mode
                else "Compare tokens/logits/states against paired arm; pass alone is not equivalence"
            ),
        )
    except BaseException as e:
        result.update(status="fail", error_type=type(e).__name__, error=str(e))
        raise
    finally:

        def release():
            if gen is not None:
                ttnn.synchronize_device(mesh)
                before_release = memory()
                released_ids = []
                for trace in list(gen.decode_traces.values()) + [gen.sample_trace, gen.buffered_sample_trace]:
                    if trace is not None:
                        ttnn.release_trace(mesh, trace)
                        released_ids.append(str(trace))
                ttnn.synchronize_device(mesh)
                after_release = memory()
                result["trace_release"] = dict(
                    trace_ids=released_ids,
                    actual_released_dram_bytes_per_device=before_release["allocated"] - after_release["allocated"],
                    actual_released_trace_arena_bytes_per_device=before_release["trace_allocated"]
                    - after_release["trace_allocated"],
                    configured_trace_region_bytes=trace_region,
                    scope="Standalone trace release: regularDRAM and reservedTRACE allocator deltas recorded separately",
                )

        finalize_run(result, target, release, lambda: ttnn.close_mesh_device(mesh))


if __name__ == "__main__":
    main()
