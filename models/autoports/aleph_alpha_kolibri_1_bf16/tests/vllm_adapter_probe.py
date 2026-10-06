# SPDX-License-Identifier: Apache-2.0
"""Small adapter proof with explicit external cache and poisoned stale inputs."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.aleph_alpha_kolibri_1_bf16.tt.generator_vllm import KolibriForCausalLM


def run():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    adapter = None
    try:
        adapter = KolibriForCausalLM.initialize_vllm_model(None, mesh, 4, 4096)
        specs = [((1024, 1, 32, 128), torch.bfloat16, i % 4) for i in range(50)]
        cache = adapter.allocate_kv_cache_per_layer(specs)
        tables = []
        for i in range(50):
            # Layers 0 and 4 share a buffer; keep their pages disjoint.
            base = 0 if i == 0 else 512
            tables.append(torch.arange(base, base + 512, dtype=torch.int32).reshape(4, 128))
        params = SimpleNamespace(temperature=[0.0] * 4, top_k=[1] * 4, top_p=[0.0] * 4, seed=[11] * 4)
        adapter.warmup_model_prefill(cache, True, True)
        assert not adapter.generator.owns_cache
        result = []
        sequences = []
        logits_runs = []
        seed_checks = []
        for poison in (False, True):
            tables[0][0, 5] = 5
            tables[4][0, 5] = 517
            logits_checks = []
            tokens = torch.tensor([[1000 + i % 77 for i in range(129)]], dtype=torch.int64)
            first = adapter.prefill_forward(
                tokens,
                page_table=tables[0],
                page_tables_per_layer=tables,
                kv_cache=cache,
                prompt_lens=torch.tensor([129]),
                empty_slots=[0],
                start_pos=torch.tensor([0]),
                sampling_params=params,
            )
            seq = [int(first[0, 0])]

            def seed_row():
                return int(ttnn.to_torch(ttnn.get_device_tensors(adapter.generator.sampler._seeds)[0]).flatten()[0])

            assert seed_row() == 13, seed_row()
            for step in range(70):
                ids = torch.tensor([seq[-1], 0, 0, 0]).reshape(4, 1)
                pos = torch.tensor([129 + step, -1, -1, -1])
                reload = not poison or step == 0
                if step == 31:
                    tables[0][0, 5] = 300
                    tables[4][0, 5] = 800
                if not reload:
                    ids.fill_(999)
                    pos.fill_(999)
                dev = adapter.decode_forward(
                    ids,
                    pos,
                    tables[0],
                    cache,
                    read_from_device=False,
                    sampling_params=params,
                    reload_inputs=reload,
                    reload_page_table=step == 31,
                    reload_sampling_params=step == 0,
                    reset_sampling_state=step == 0,
                    page_tables_per_layer=tables,
                )
                host, events = adapter.read_decode_output(dev, async_read=True)
                for event in events:
                    ttnn.event_synchronize(event)
                seq.append(int(adapter.process_decode_output_host(host, is_tokens=True)[0, 0]))
                if step == 0:
                    seed_checks.append(seed_row())
                    assert seed_row() == 14, seed_row()
                if step in (0, 31, 69):
                    logits_checks.append(adapter.generator.read_logits()[0, 0, 0].clone())
            sequences.append(seq)
            logits_runs.append(logits_checks)
        assert sequences[0] == sequences[1], sequences
        differences = [float((a - b).abs().max()) for a, b in zip(*logits_runs)]
        assert max(differences) == 0, differences
        adapter.decode_forward(
            torch.tensor([999, 0, 0, 0]).reshape(4, 1),
            torch.tensor([999, -1, -1, -1]),
            tables[0],
            cache,
            sampling_params=params,
            page_tables_per_layer=tables,
        )
        sensitivity = float((adapter.generator.read_logits()[0, 0, 0] - logits_runs[1][-1]).abs().max())
        assert sensitivity > 0.1, sensitivity
        result = dict(
            status="pass",
            logit_max_differences=differences,
            poison_sensitivity=sensitivity,
            sequences=sequences,
            counters=dict(adapter.generator.counters),
            traces=adapter.generator.traces,
            cache_owned_by_generator=adapter.generator.owns_cache,
            first_decode_next_seeds=seed_checks,
            limitation="Direct stale-input proof; allocator-driven growth and overlap remain server gates",
        )
        Path(os.environ["KOLIBRI_ADAPTER_PROBE_RESULT"]).write_text(json.dumps(result, indent=2, default=str) + "\n")
        print("ADAPTER_STALE_INPUT_PASS", flush=True)
    finally:
        if adapter is not None and adapter.generator is not None:
            adapter.generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    run()
