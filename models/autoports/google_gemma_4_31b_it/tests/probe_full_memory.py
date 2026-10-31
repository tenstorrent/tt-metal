# SPDX-License-Identifier: Apache-2.0
"""Reduced layers0/5 transient DRAM accounting, without device profiling.

NORMAL graph buffer events already contain physical max_size_per_bank. Detailed
page tracing is optional: --detailed-pages also enumerates every weight/cache
page on allocations and can be very expensive at full context. Neither mode
collects Tracy/device timestamps or claims the all60-layer peak.
"""

import argparse
import gc
import json
import os
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_31b_it.tt.generator import Gemma4Generator


def memory(mesh):
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return {
        name: int(getattr(view, name))
        for name in (
            "num_banks",
            "total_bytes_allocated_per_bank",
            "total_bytes_per_bank",
            "largest_contiguous_bytes_free_per_bank",
        )
    }


def summarize(nodes, before, after):
    """Reconcile event accounting to the same mesh allocator's observed delta."""
    current = peak = 0
    peak_counter = None
    counts = {"buffer_allocate": 0, "buffer_deallocate": 0}
    ids = set()
    for node in nodes:
        kind = node.get("node_type")
        params = node.get("params", {})
        if kind not in counts or params.get("type") != "DRAM":
            continue
        ids.add(str(params["device_id"]))
        # GraphProcessor directly reports Buffer::aligned_size_per_bank, not
        # logical tensor bytes. The lockstep DRAM allocator reserves it per bank.
        amount = int(params["max_size_per_bank"])
        current += amount if kind == "buffer_allocate" else -amount
        counts[kind] += 1
        if current > peak:
            peak, peak_counter = current, node.get("counter")
    baseline = before["total_bytes_allocated_per_bank"]
    observed_delta = after["total_bytes_allocated_per_bank"] - baseline
    reconciled = current == observed_delta and len(ids) == 1 and bool(counts["buffer_allocate"])
    banks = before["num_banks"]
    return dict(
        allocator_device_ids=sorted(ids),
        event_counts=counts,
        event_delta_per_bank=current,
        observed_delta_per_bank=observed_delta,
        reconciliation_bytes_per_bank=current - observed_delta,
        reconciled=reconciled,
        peak_event_counter=peak_counter,
        num_banks=banks,
        peak_dram_bytes_per_device=(baseline + peak) * banks if reconciled else None,
        incremental_peak_dram_bytes_per_device=peak * banks if reconciled else None,
        missing_reasons=(
            []
            if reconciled
            else ["Graph event delta/device identity did not reconcile to mesh allocator; peak withheld."]
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, choices=(1, 8, 16, 32), default=1)
    parser.add_argument("--detailed-pages", action="store_true")
    parser.add_argument("--buffered", action="store_true")
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    for name in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER", "TT_METAL_DPRINT_CORES"):
        if os.environ.get(name, "0") not in ("", "0"):
            raise ValueError(f"This allocator-only probe requires {name} disabled")
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    results = []
    try:
        gen = Gemma4Generator(mesh, layer_indices=(0, 5), batch_size=args.batch, max_seq_len=262144)
        assert gen.model.layer_indices == (0, 5)
        assert gen.model_trace is None and gen.sample_trace is None
        gen.reset()
        gen.prefill.stage(torch.full((1024,), 2, dtype=torch.int32), slot=0, start=0, size=1024, valid=1024)

        def prefill_all_logits():
            hidden = gen.prefill.hidden(size=1024, start=0, kv_cache=gen.kv_cache)
            # Match prepare_decode's largest-tail lifetimes: full head output
            # remains live while selected-row head and B-row assembly execute.
            all_logits = gen.model.terminal(hidden)
            selected = gen.model.terminal(gen.prefill.last_row(hidden))
            rows = selected if gen.batch == 1 else ttnn.concat([selected] * gen.batch, dim=2)
            ttnn.copy(rows, gen.logits)
            del hidden, all_logits, selected, rows

        def decode_sampling_setup():
            gen._decode()
            gen._sample(record=args.buffered)

        for name, operation in (
            ("prefill1024_all_logits", prefill_all_logits),
            ("decode_sampling_setup", decode_sampling_setup),
        ):
            for temperature in ("cold", "warm"):
                gc.collect()
                ttnn.synchronize_device(mesh)
                before = memory(mesh)
                if args.detailed_pages:
                    ttnn.graph.enable_detailed_buffer_tracing()
                ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
                try:
                    operation()
                    gc.collect()
                    ttnn.synchronize_device(mesh)
                    after = memory(mesh)
                finally:
                    graph = ttnn.graph.end_graph_capture()
                    if args.detailed_pages:
                        ttnn.graph.disable_detailed_buffer_tracing()
                # Only compact counters leave this process, never full graph/page dumps.
                row = dict(
                    phase=name, temperature=temperature, before=before, after=after, **summarize(graph, before, after)
                )
                results.append(row)
                print(json.dumps(row), flush=True)
                del graph
        report = dict(
            scope="reduced real-weight layers0/5 only; NOT full-stack memory measurement",
            mesh=[1, 4],
            layer_indices=[0, 5],
            batch=args.batch,
            context=gen.context,
            pool_tokens=gen.pool_tokens,
            detailed_pages=args.detailed_pages,
            cache_shapes=[[list(t.shape) for t in cache] for cache in gen.kv_cache],
            cache_dtypes=[[str(t.dtype) for t in cache] for cache in gen.kv_cache],
            phases=results,
            basis="NORMAL graph allocate/deallocate max_size_per_bank x allocator num_banks; reconciled to get_memory_view",
            notes=[
                "Mesh allocator represents mirrored per-device reservations; no chip-specific occupancy is inferred.",
                "Cold includes new persistent program allocations; warm isolates warmed operation lifetime.",
                "Peak is allocated capacity, not largest contiguous free region or a full-stack fit guarantee.",
                "Trace region is configured but no model/sampling trace is captured; measure full-stack persistent baseline separately.",
            ],
        )
        path = Path(args.result)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        assert all(row["reconciled"] for row in results), "Memory event reconciliation failed; peaks withheld"
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
