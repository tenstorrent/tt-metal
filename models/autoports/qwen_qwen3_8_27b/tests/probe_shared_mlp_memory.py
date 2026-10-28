# SPDX-License-Identifier: Apache-2.0
"""Standalone paired-arm MLP memory/correctness fixture; serialized TT lease required.

Run each arm in a fresh process with QWEN38_SHARED_MLP_WEIGHTS=0/1. Native
logical context and page-table shapes are retained; the explicitly bounded
external pool only backs this fixture's short requests. This is not a B32
native-context capacity claim or a qualitative model judgment.
"""
import argparse
import hashlib
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
    a = p.parse_args()
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_8_27b.tests.full_model_evidence import runtime_build
    from models.autoports.qwen_qwen3_8_27b.tests.native_fixture_cleanup import finalize_run
    from models.autoports.qwen_qwen3_8_27b.tests.native_fixture_metadata import capture_linear_calls
    from models.autoports.qwen_qwen3_8_27b.tests.physical_memory_inventory import AllocationLedger, tensor_aliases
    from models.autoports.qwen_qwen3_8_27b.tt.generator import Generator
    from models.autoports.qwen_qwen3_8_27b.tt.model import Model

    if (
        os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1"
        or os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0"
    ):
        raise RuntimeError("Correctness fixture requires trace tracking and program-cache tracking")
    if max(a.lengths) + a.outputs > 1024 or min(a.lengths) < 1:
        raise ValueError("Short fixture pool must cover every request")
    # paged_update_cache requires table_width <= total physical cache pages.
    # Preserve native table width; supply the smallest legal aggregate pool.
    pool_context = max(1024, math.ceil(262144 / (a.batch * 256)) * 256)
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    result = dict(
        status="running",
        throw_exception_on_fallback=bool(ttnn.CONFIG.throw_exception_on_fallback),
        shared=os.environ.get("QWEN38_SHARED_MLP_WEIGHTS", "0"),
        batch=a.batch,
        layers="all64" if a.full_stack else [0, 3],
        native_context=262144,
        physical_pool_tokens_per_row=pool_context,
        requested_outputs=a.outputs,
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
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    gen = None
    target = Path(a.output)
    target.parent.mkdir(parents=True, exist_ok=True)

    def memory():
        m = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        return dict(
            banks=m.num_banks,
            allocated=m.total_bytes_allocated_per_bank * m.num_banks,
            free=m.total_bytes_free_per_bank * m.num_banks,
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
            cache = model.allocate_cache(batch_size=a.batch, context_length=pool_context)
            pages = pool_context // 32
            table = torch.zeros(a.batch, 262144 // 32, dtype=torch.int32)
            table[:, :pages] = torch.arange(a.batch * pages, dtype=torch.int32).reshape(a.batch, pages)
            gen = Generator(
                model,
                a.checkpoint,
                batch_size=a.batch,
                context_length=262144,
                kv_cache=cache,
                page_table=table,
                prefill_chunk_size=512,
            )
            ttnn.synchronize_device(mesh)
        finally:
            graph = ttnn.graph.end_graph_capture()
        ledger.update(graph)
        del graph

        def checkpoint(label):
            ttnn.synchronize_device(mesh)
            census = tensor_aliases({"generator": gen})
            m = memory()
            snap = ledger.snapshot(
                census["tensors"],
                domain=mesh.id(),
                allocator_bytes=m["allocated"],
                baseline_bytes=baseline["allocated"],
            )
            snap["shared_objects"] = census["shared_objects"]
            snap["allocator"] = m
            result[label] = snap
            target.write_text(json.dumps(result, indent=2) + "\n")

        checkpoint("after_construction")
        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        try:
            with capture_linear_calls(result, ttnn, model):
                gen.prepare_decode()
            ttnn.synchronize_device(mesh)
        finally:
            graph = ttnn.graph.end_graph_capture()
        ledger.update(graph)
        del graph
        checkpoint("after_preparation")
        result["preparation"] = gen.last_preparation_memory
        gen.set_sampling_params(top_k=1)

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
        for length in a.lengths:
            # Deliberately awkward lengths, distinct rows, and a ragged last row.
            # Synthetic token IDs are mechanism coverage, not chat quality evidence.
            prompts = [
                (torch.arange(length - (i % 3 if length > 3 else 0)) + 1000 + i * 17).tolist() for i in range(a.batch)
            ]
            ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
            try:
                tokens = gen.generate(prompts, a.outputs, stop_on_eos=False, buffered_output=True)
                ttnn.synchronize_device(mesh)
            finally:
                graph = ttnn.graph.end_graph_capture()
            ledger.update(graph)
            del graph
            request_memory = memory()
            reconciled = ledger.snapshot(
                [], domain=mesh.id(), allocator_bytes=request_memory["allocated"], baseline_bytes=baseline["allocated"]
            )
            states = {
                f"{layer.layer_idx}.{field}": digest(tensor)
                for layer, state in zip(model.layers, cache)
                for field, tensor in vars(state).items()
            }
            record = dict(
                length=length,
                logical_lengths=list(map(len, prompts)),
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
            assert trace_ids == {str(k): str(v) for k, v in gen.decode_traces.items()}
            assert gen.counters["captures"] == captures
            target.write_text(json.dumps(result, indent=2) + "\n")
        # Same warmed fixed-work native path; diagnostic tracking stays enabled.
        # Clean headline performance must rerun separately with tracking disabled.
        result["diagnostic_timing"] = []
        for _ in range(a.repeats):
            started = time.perf_counter()
            gen.generate(prompts, a.outputs, stop_on_eos=False, buffered_output=True)
            result["diagnostic_timing"].append(
                dict(wall_s=time.perf_counter() - started, metrics=dict(gen.last_generation_metrics))
            )
        result.update(
            status="pass",
            trace_ids=trace_ids,
            captures=captures,
            exactness="Compare tokens/logits/states against paired arm; pass alone is not equivalence",
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
                    configured_trace_region_bytes=0,
                    scope="Standalone dynamic traces: allocator delta after releasing only traces at final teardown; not a serving128MiB arena payload measurement",
                )

        finalize_run(result, target, release, lambda: ttnn.close_mesh_device(mesh))


if __name__ == "__main__":
    main()
