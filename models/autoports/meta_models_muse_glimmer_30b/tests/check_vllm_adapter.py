# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced real-layer adapter replay, external-cache and stale-input checks."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator_vllm import MuseGlimmerForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grouped", action="store_true", help="Exercise shared tables with real layers 0/1/3")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/autoports/meta_models_muse_glimmer_30b/readiness_vllm/reduced_adapter.json"),
    )
    args = parser.parse_args()
    out = args.output
    indices = [0, 1, 3] if args.grouped else [0, 3]
    os.environ["MUSE_VLLM_LAYER_INDICES"] = ",".join(map(str, indices))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        adapter = MuseGlimmerForConditionalGeneration.initialize_vllm_model(None, mesh, 4)
        if args.grouped:
            adapter.configure_kv_cache_groups([int(i == 3) for i in range(52)])
        specs = [((4096, 1, 32, 128), torch.bfloat16, i) for i in range(52)]
        cache = adapter.allocate_kv_cache_per_layer(specs)
        adapter.warmup_model_prefill(cache, True, True)
        gen = adapter.generator
        if args.grouped:
            assert gen.layer_page_tables[0] is gen.layer_page_tables[1]
            assert gen.layer_page_tables[0] is not gen.layer_page_tables[2]
            assert len(gen._unique_page_table_indices) == 2
        params = SimpleNamespace(temperature=[0.0] * 4, top_k=[1] * 4, top_p=[1.0] * 4, seed=[11, 12, 13, 14])
        tables = []
        for i in range(52):
            table = torch.zeros((4, 4096), dtype=torch.int32)
            for slot in range(4):
                table[slot, :64] = torch.arange(64) + slot * 64 + (512 if i == 3 else 0)
            tables.append(table)
        prompts = torch.tensor([[200000] + [100 + slot] * 130 for slot in range(4)])
        outputs = []
        baseline = None
        for trial in range(2):
            prefill, _ = adapter.prefill_forward(
                prompts, tables[0], cache, [129, 130, 131, 129], page_tables_per_layer=tables, sampling_params=params
            )
            seq = [prefill.tolist()]
            pos = torch.tensor([129, 130, 131, 129], dtype=torch.int32)
            token = prefill
            before = dict(gen.counters)
            for step in range(4):
                authoritative = trial == 0 or step == 0
                device = adapter.decode_forward(
                    token if authoritative else torch.zeros_like(token),
                    pos if authoritative else torch.zeros_like(pos),
                    tables[0],
                    cache,
                    sampling_params=params,
                    page_tables_per_layer=tables,
                    read_from_device=False,
                    reload_inputs=authoritative,
                    reload_sampling_params=step == 0,
                    reset_sampling_state=step == 0,
                )
                host, events = adapter.read_decode_output(device, async_read=True)
                for event in events:
                    ttnn.event_synchronize(event)
                token, _ = adapter.process_decode_output_host(host, is_tokens=True)
                seq.append(token.tolist())
                pos += 1
                actual = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).flatten()[:4]
                assert torch.equal(actual.to(torch.int32), pos), (actual, pos)
            delta = {k: gen.counters[k] - before.get(k, 0) for k in gen.counters}
            if baseline is None:
                baseline = seq
            else:
                assert seq == baseline, "Resident decode differs from authoritative control"
                assert delta["token_refreshes"] == 1 and delta["position_refreshes"] == 1, delta
                assert delta["page_table_refreshes"] == 0, delta
            outputs.append(dict(trial=trial, tokens=seq, counters=delta))
        # A first admission can mix a final row with an intermediate row.
        # Only the final row may consume RNG state; compare the exact known
        # host ledger against the device after every transition.
        gen.reset()
        adapter._sampling["seeds"] = list(range(4))
        adapter._seed_bases = list(range(4))
        seed_checks = []

        def check_seeds(label):
            actual = ttnn.to_torch(ttnn.get_device_tensors(gen.sampler._seeds)[0]).flatten()[:4].tolist()
            assert actual == adapter._sampling["seeds"], (label, actual, adapter._sampling)
            seed_checks.append(dict(label=label, seeds=actual, bases=list(adapter._seed_bases)))

        partial = SimpleNamespace(temperature=[0.8, 0.8], top_k=[8, 8], top_p=[0.9, 0.9], seed=[11, 12])
        adapter.prefill_forward(
            prompts[:2],
            tables[0][:2],
            cache,
            [129, 130],
            page_tables_per_layer=[t[:2] for t in tables],
            sampling_params=partial,
            intermediate_prefill_mask=[False, True],
        )
        check_seeds("mixed_final_intermediate")
        assert adapter._sampling["seeds"] == [140, 1, 2, 3]
        final = SimpleNamespace(temperature=[0.8], top_k=[8], top_p=[0.9], seed=[12])
        adapter.prefill_forward(
            prompts[1:2],
            tables[0][1:2],
            cache,
            [131],
            start_pos=[130],
            empty_slots=[1],
            page_tables_per_layer=[t[1:2] for t in tables],
            sampling_params=final,
        )
        check_seeds("continuation_final")
        assert adapter._sampling["seeds"] == [140, 143, 2, 3]
        # Parameter-only updates must not consult the deliberately stale host
        # positions. Inactive rows are sampled by the canonical fixed graph,
        # and the ledger mirrors that advance exactly.
        all_params = SimpleNamespace(temperature=[0.9] * 4, top_k=[8] * 4, top_p=[0.9] * 4, seed=[11, 12, 2, 3])
        adapter.decode_forward(
            torch.zeros((4, 1), dtype=torch.int64),
            torch.zeros(4, dtype=torch.int32),
            tables[0],
            cache,
            sampling_params=all_params,
            page_tables_per_layer=tables,
            read_from_device=False,
            reload_inputs=False,
            reload_sampling_params=True,
        )
        check_seeds("parameter_only_stale_positions")
        # Dormant sampling state must follow a permutation even in explicit
        # host compatibility mode; no sampled tokens are used as host feedback.
        adapter.allow_host_sampling = True
        adapter.decode_forward(
            torch.full((4, 1), 100),
            torch.tensor([131, 129, -1, -1]),
            tables[0],
            cache,
            sampling_params=None,
            page_tables_per_layer=tables,
            read_from_device=False,
            reload_inputs=True,
            slot_remap=[1, 0, 3, 2],
        )
        check_seeds("dormant_host_remap")
        out.write_text(
            json.dumps(
                dict(
                    status="pass",
                    layers=indices,
                    page_table_groups=adapter.page_table_groups,
                    logical_lengths=[129, 130, 131, 129],
                    traces={str(k): int(v) for k, v in gen.traces.items()},
                    results=outputs,
                    seed_checks=seed_checks,
                    counters=dict(gen.counters),
                    tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
                ),
                indent=2,
            )
            + "\n"
        )
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
