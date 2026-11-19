# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Focused external-cache warmup preservation and split-trace state probe."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator


def digest(cache):
    result = []
    seen = set()
    for states in cache:
        for state in states:
            for name in ("conv", "recurrent", "key", "value"):
                tensor = getattr(state, name)
                if tensor is None or id(tensor) in seen:
                    continue
                seen.add(id(tensor))
                for shard in ttnn.get_device_tensors(tensor):
                    value = ttnn.to_torch(shard).float().contiguous()
                    result.append(hashlib.sha256(value.numpy().tobytes()).hexdigest())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = build_generator(None, mesh, layer_indices=[0, 3], max_seq_len=256)
        external = gen.model.allocate_cache(batch=1, context=256)
        # Prefill before decode preparation: warmup must preserve caller state.
        tokens = torch.arange(100, 133).reshape(1, -1)
        gen.prefill_forward(tokens, page_table=gen.page_table, kv_cache=external, prompt_lens=[33])
        before = digest(external)
        state = gen.prepare_decode(kv_cache=external, page_table=gen.page_table, slots=(0,))
        assert digest(external) == before, "Decode warmup mutated caller-owned prefilled cache"
        first = gen.decode_forward(torch.tensor([[234]]), [33], page_table=gen.page_table, kv_cache=external)

        def read(tensor):
            return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).reshape(-1).tolist()

        records = [{"token": int(first[0]), "position": read(state.positions[0])[0]}]
        assert records[0]["position"] == 34
        counters = dict(gen.counters)
        for expected_position in (35, 36, 37):
            actual_input = read(state.tokens)[0]
            assert actual_input == records[-1]["token"], "Device feedback does not equal sampled token"
            output = gen.replay(state)
            records.append({"input": actual_input, "token": int(output[0]), "position": read(state.positions[0])[0]})
            assert records[-1]["position"] == expected_position
        for name in ("token_refreshes", "position_refreshes", "page_table_refreshes", "captures", "releases"):
            assert counters[name] == gen.counters[name], f"Unexpected steady-state host work: {name}"
        # Change only currently unused pages, preserving the existing causal cache.
        changed = gen.page_table.clone()
        changed[:, -2:] = changed[:, -2:].flip(-1)
        gen.decode_forward(torch.tensor([[records[-1]["token"]]]), [37], page_table=changed, kv_cache=external)
        assert gen.counters["page_table_refreshes"] == counters["page_table_refreshes"] + 1
        assert read(state.tables[0]) == changed.reshape(-1).tolist()
        after_changed = gen.counters["page_table_refreshes"]
        gen.decode_forward(torch.tensor([[345]]), [38], page_table=changed, kv_cache=external)
        assert gen.counters["page_table_refreshes"] == after_changed
        Path(args.output).write_text(
            json.dumps(
                {
                    "status": "pass",
                    "layers": [0, 3],
                    "context": 256,
                    "cache_restored": True,
                    "steps": records,
                    "counters": gen.counters,
                    "limitation": "State-transition probe; reduced context and layers, no quality/performance claim",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        if gen:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
