# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-layer native hybrid retirement contrast; never a model benchmark.

Run with TT_METAL_TRACE_ALLOC_TRACKING=1 in the normal bringup environment.
All-full *tables* retain every page; the control still has the real sliding/full
attention layers and selected precision. Diagnostic logits/cache reads belong
only to this test, never to the adapter's serving token-feedback path.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
LAYERS = (0, 3)
BOUNDARIES = (2047, 2048, 2078, 2079, 2080)


class AbsoluteTables:
    """Small deterministic allocator reproducing native absolute row semantics.

    Full rows retain their pages. Sliding rows either null retired columns or
    preserve the worker's stale IDs. Freed pages are reused by another request
    before allocating fresh pages. Physical page zero remains reserved.
    """

    def __init__(self, batch, mode):
        self.batch, self.mode = batch, mode
        self.tables = [torch.zeros((batch, 4096), dtype=torch.int32) for _ in range(52)]
        self.owned = [[{} for _ in range(batch)] for _ in LAYERS]
        self.free = [[] for _ in range(batch)]
        self.allocated = 0
        self.retired = 0
        self.events = []

    def advance(self, computed, end):
        retired = max(0, computed - 2047) // 32 if self.mode != "control" else 0
        for row in range(self.batch):
            for logical in range(self.retired, retired):
                physical = self.owned[0][row].pop(logical)
                self.free[row].append(physical)
                if self.mode == "null":
                    self.tables[0][row, logical] = 0
                self.events.append(dict(event="retire", position=computed, row=row, logical=logical, physical=physical))
        blocks = (end + 31) // 32
        for local, layer in enumerate(LAYERS):
            for row in range(self.batch):
                source = (row + 1) % self.batch
                for logical in range(self.allocated, blocks):
                    if local == 0 and self.free[source]:
                        physical = self.free[source].pop(0)
                        self.events.append(
                            dict(
                                event="reuse",
                                position=computed,
                                row=row,
                                from_row=source,
                                logical=logical,
                                physical=physical,
                            )
                        )
                    else:
                        physical = 1 + local * 1024 + row * 256 + logical
                    assert 0 < physical < 4096
                    self.tables[layer][row, logical] = physical
                    self.owned[local][row][logical] = physical
        self.retired, self.allocated = retired, blocks
        # Both layers share the physical backing pair in the adapter. No live
        # page may therefore alias another layer or another active request.
        live = [p for group in self.owned for row in group for p in row.values()]
        assert len(live) == len(set(live)), "Distinct live owners alias a physical page"
        for layer in LAYERS:
            first = retired if layer == 0 else 0
            assert torch.all(self.tables[layer][:, first:blocks] > 0)
            assert torch.all(self.tables[layer][:, blocks:] == 0)


def device_values(ttnn, tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()


def compare(actual, expected, *, atol, rtol):
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), "Nonfinite diagnostic values"
    error = (actual - expected).abs()
    result = dict(max_abs=float(error.max()), mean_abs=float(error.mean()), atol=atol, rtol=rtol)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return result


def cache_snapshot(ttnn, adapter, mapping, end):
    """Read all four KV shards, comparing only mathematically live token rows.

    Host indexing avoids creating new TT allocations after trace capture. The
    pooled tensors are shared, so each physical cache is read only once/shard.
    """
    result = {}
    for kind, cache in zip(("k", "v"), adapter.generator.kv_cache[0]):
        for shard, device_cache in enumerate(ttnn.get_device_tensors(cache)):
            values = ttnn.to_torch(device_cache)
            for layer in LAYERS:
                first = max(0, end - 2048) if layer == 0 else 0
                positions = torch.arange(first, end)
                for row in range(adapter.batch_size):
                    pages = mapping.tables[layer][row, positions // 32].long()
                    result[f"layer{layer}.row{row}.{kind}.shard{shard}"] = values[pages, 0, positions % 32].clone()
    return result


def prompt_tokens(batch, length):
    # Distinct deterministic real vocabulary IDs; no random model weights/KV.
    positions = torch.arange(length)
    tokens = torch.stack([100 + (positions * (row + 3) + 113 * row) % 1900 for row in range(batch)])
    tokens[:, 0] = 200000
    return tokens


def run_trial(ttnn, adapter, cache, mode, scenario, args):
    gen, batch = adapter.generator, adapter.batch_size
    gen.reset(clear_cache=True)
    mapping = AbsoluteTables(batch, mode)
    params = SimpleNamespace(
        temperature=[0.0] * batch, top_k=[1] * batch, top_p=[1.0] * batch, seed=list(range(11, 11 + batch))
    )
    result = dict(
        mode=mode,
        scenario=scenario,
        tokens={},
        logits={},
        cache={},
        page_only=[],
        submissions=[],
        chunks=[],
        table_checks=[],
    )
    first_length = 2047 if scenario == "decode" else 4096
    prompts = prompt_tokens(batch, 4259)
    mapping.advance(0, first_length)
    first, _ = adapter.prefill_forward(
        prompts[:, :first_length],
        mapping.tables[0],
        cache,
        [first_length] * batch,
        page_tables_per_layer=mapping.tables,
        sampling_params=params,
    )
    result["tokens"]["prefill"] = first.tolist()
    result["logits"]["prefill"] = gen._read_logits(gen.logits).reshape(batch, -1).clone()
    before = dict(gen.counters)
    traces = dict(gen.traces)

    def record(name, end, *, with_cache=False):
        result["logits"][name] = gen._read_logits(gen.logits).reshape(batch, -1).clone()
        if with_cache:
            result["cache"][name] = cache_snapshot(ttnn, adapter, mapping, end)

    if scenario == "decode":
        pending = None

        def finish(item):
            position, host, events = item
            for event in events:
                ttnn.event_synchronize(event)
            token, _ = adapter.process_decode_output_host(host, is_tokens=True)
            result["tokens"][str(position)] = token.tolist()

        for position in range(2047, 2082):
            previous = [mapping.tables[layer].clone() for layer in LAYERS]
            mapping.advance(position, position + 1)
            changed = any(not torch.equal(old, mapping.tables[layer]) for old, layer in zip(previous, LAYERS))
            authoritative = position == 2047
            page_only = changed and not authoritative
            counters = dict(gen.counters)
            refresh = gen._refresh_table

            def checked_refresh(*values, **kwargs):
                tensors = (gen.tokens, gen.positions, gen.rope_indices)
                saved = [device_values(ttnn, tensor) for tensor in tensors]
                refresh(*values, **kwargs)
                for tensor, current, value in zip(tensors, (gen.tokens, gen.positions, gen.rope_indices), saved):
                    assert tensor is current, "Page-only refresh replaced persistent input storage"
                    assert torch.equal(value, device_values(ttnn, current)), "Page-only refresh changed resident inputs"
                result["page_only"].append(
                    dict(
                        position=position,
                        tokens=saved[0].flatten().tolist(),
                        positions=saved[1].flatten().tolist(),
                        rope=saved[2].flatten().tolist(),
                        exact_preservation=True,
                    )
                )

            if page_only:
                gen._refresh_table = checked_refresh
            try:
                device = adapter.decode_forward(
                    first if authoritative else torch.zeros_like(first),
                    torch.full((batch,), position if authoritative else 7, dtype=torch.int32),
                    mapping.tables[0],
                    cache,
                    page_tables_per_layer=mapping.tables,
                    sampling_params=params,
                    read_from_device=False,
                    reload_inputs=authoritative,
                    reload_page_table=page_only,
                    reload_sampling_params=authoritative,
                    reset_sampling_state=authoritative,
                )
            finally:
                gen._refresh_table = refresh
            host, events = adapter.read_decode_output(device, async_read=True)
            # The previous read was queued before this decode overwrote the
            # persistent device token buffer; formatting is deliberately delayed.
            if pending is not None:
                finish(pending)
            pending = (position, host, events)
            resident = device_values(ttnn, gen.positions).flatten()[:batch]
            assert torch.equal(resident.int(), torch.full((batch,), position + 1, dtype=torch.int32))
            delta = {key: gen.counters[key] - counters.get(key, 0) for key in gen.counters}
            if not authoritative:
                assert all(
                    delta.get(key, 0) == 0 for key in ("token_refreshes", "position_refreshes", "rope_refreshes")
                ), delta
            if not changed and not authoritative:
                assert delta.get("page_table_refreshes", 0) == 0, delta
            result["submissions"].append(
                dict(position=position, reload_inputs=authoritative, reload_page_table=page_only, counters=delta)
            )
            if position in BOUNDARIES:
                result["table_checks"].append(
                    dict(
                        position=position,
                        sliding_first_visible=max(0, position - 2047),
                        retired_columns=mapping.retired,
                        first_columns=mapping.tables[0][:, :2].tolist(),
                        write_pages={str(layer): mapping.tables[layer][:, position // 32].tolist() for layer in LAYERS},
                    )
                )
                record(str(position), position + 1, with_cache=position == 2080)
        finish(pending)
        result["delayed_async_reads"] = 34
    else:
        # The second call has a genuine continuation start >4096 and a partial
        # final page. It traverses the prepared alignment-32/64/96 and 64 buckets.
        for start, end in ((4096, 4128), (4128, 4259)):
            mapping.advance(start, end)
            tokens, _ = adapter.prefill_forward(
                prompts[:, :end],
                mapping.tables[0],
                cache,
                [end] * batch,
                start_pos=[start] * batch,
                page_tables_per_layer=mapping.tables,
                sampling_params=params,
            )
            name = f"chunk_{start}_{end}"
            result["tokens"][name] = tokens.tolist()
            result["chunks"].append(dict(start=start, length=end - start, end=end))
            record(name, end, with_cache=end == 4259)

    assert traces == gen.traces, "A prepared execution signature was recaptured or retired"
    result["counters"] = {key: gen.counters[key] - before.get(key, 0) for key in gen.counters}
    result["allocator_events"] = mapping.events
    if mode != "control":
        reused = [event for event in mapping.events if event["event"] == "reuse"]
        assert reused, "Test never reused a retired page"
        assert all(event["row"] != event["from_row"] for event in reused) or batch == 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--scenario", choices=("all", "decode", "chunk"), default="all")
    parser.add_argument("--atol", type=float, default=0.03125)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/hybrid_retirement.json")
    args = parser.parse_args()
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        raise RuntimeError("Set TT_METAL_TRACE_ALLOC_TRACKING=1 before importing TTNN")
    # Import only after CLI parsing so --help and AST checks need no devices.
    import ttnn
    from models.autoports.meta_models_muse_glimmer_30b.tt.generator_vllm import MuseGlimmerForConditionalGeneration

    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["MUSE_VLLM_LAYER_INDICES"] = "0,3"
    os.environ["MUSE_VLLM_EVENTS"] = str(args.output.with_suffix(".events.jsonl"))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    adapter = None
    summary = dict(
        status="running",
        layers=list(LAYERS),
        batch_size=args.batch_size,
        boundaries=list(BOUNDARIES),
        trace_alloc_tracking=True,
        diagnostic_only=True,
        cache_shards=[0, 1, 2, 3],
        results=[],
    )
    try:
        adapter = MuseGlimmerForConditionalGeneration.initialize_vllm_model(None, mesh, args.batch_size)
        selected = json.loads((ROOT / "doc/datatype_sweep/selected_precision_config.json").read_text())
        assert adapter.model.precision_config == selected, "Adapter did not load selected precision"
        summary["precision_config"] = selected
        # Native hybrid groups may share a backing tensor_idx. Force that case;
        # physical IDs, not per-layer allocation isolation, enforce ownership.
        specs = [((4096, 1, 32, 128), torch.bfloat16, 0) for _ in range(52)]
        cache = adapter.allocate_kv_cache_per_layer(specs)
        assert all(a is b for a, b in zip(cache[0], cache[1])), "tensor_idx sharing was ignored"
        adapter.warmup_model_prefill(cache, True, True)
        summary["prepared_traces"] = {str(key): int(value) for key, value in adapter.generator.traces.items()}
        scenarios = ("decode", "chunk") if args.scenario == "all" else (args.scenario,)
        for scenario in scenarios:
            baseline = None
            for mode in ("control", "null", "stale"):
                print(f"HYBRID_PROBE scenario={scenario} mode={mode}", flush=True)
                trial = run_trial(ttnn, adapter, cache, mode, scenario, args)
                raw = args.output.with_name(f"{args.output.stem}_{scenario}_{mode}_logits.pt")
                torch.save(dict(tokens=trial["tokens"], logits=trial["logits"]), raw)
                row = {key: value for key, value in trial.items() if key not in ("logits", "cache")}
                row["raw_logits"] = str(raw)
                row["logit_comparisons"], row["cache_comparisons"] = {}, {}
                if baseline is None:
                    baseline = trial
                else:
                    assert trial["tokens"] == baseline["tokens"], f"{scenario}/{mode}: greedy token mismatch"
                    for name, logits in trial["logits"].items():
                        row["logit_comparisons"][name] = compare(
                            logits, baseline["logits"][name], atol=args.atol, rtol=args.rtol
                        )
                    for name, snapshots in trial["cache"].items():
                        row["cache_comparisons"][name] = {
                            key: compare(value, baseline["cache"][name][key], atol=args.atol, rtol=args.rtol)
                            for key, value in snapshots.items()
                        }
                summary["results"].append(row)
                args.output.write_text(json.dumps(summary, indent=2) + "\n")
        summary["status"] = "pass"
    except BaseException as error:
        summary.update(status="fail", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
        try:
            if adapter is not None and adapter.generator is not None:
                adapter.generator.close()
        finally:
            if adapter is not None and adapter.events is not None:
                adapter.events.close()
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
