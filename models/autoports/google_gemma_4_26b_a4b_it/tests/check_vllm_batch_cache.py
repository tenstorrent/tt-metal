# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced multirow cache and decode isolation probe."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import AutoportGemma4ForCausalLM
from models.common.sampling.generator import SamplingParams


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    report = {}
    try:
        gen = Gemma4Generator(mesh, max_seq_len=256, layer_indices=(0, 5), trace_debug=True)
        prompts = [[2] + [100] * 30, [2] + [101] * 62]
        controls = [gen.generate(prompt, 4, stop_on_eos=False) for prompt in prompts]
        report["controls"] = controls
        gen._release_trace()
        adapter = AutoportGemma4ForCausalLM(gen, 32)
        specs = [((32, 2, 32, 256), torch.bfloat16, i) for i in range(30)]
        specs[5] = ((32, 1, 32, 512), torch.bfloat16, 0)
        cache = adapter.allocate_kv_cache_per_layer(specs)
        table = torch.tensor([list(range(8)), list(range(16, 24))], dtype=torch.int32)
        full = table + 8
        tables = [table] * 30
        tables[5] = full
        ids = torch.zeros(2, 63, dtype=torch.long)
        for i, prompt in enumerate(prompts):
            ids[i, : len(prompt)] = torch.tensor(prompt)
        params = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
        output = adapter.prefill_forward(
            ids, table, cache, [31, 63], sampling_params=params, page_tables_per_layer=tables, empty_slots=[5, 7]
        )
        observed = [[int(v)] for v in output.flatten()]
        decode_tables = [torch.nn.functional.pad(t, (0, 0, 0, 30)) for t in tables]
        for step in range(3):
            positions = torch.full((32,), -1, dtype=torch.int32)
            positions[:2] = torch.tensor([31 + step, 63 + step])
            wire_tokens = torch.nn.functional.pad(output, (0, 0, 0, 32 - len(output)))
            output = adapter.decode_forward(
                wire_tokens,
                positions,
                table,
                cache,
                sampling_params=params,
                page_tables_per_layer=decode_tables,
                reset_batch=step == 0,
            )
            for row, value in zip(observed, output.flatten()):
                row.append(int(value))
        report.update(observed=observed, matches=observed == controls, wire_rows=32, logical_decode_rows=gen.batch)
        assert gen.batch == 2
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        assert observed == controls, report
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
