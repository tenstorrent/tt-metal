# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Common sampler traced mode/seed transitions on real TP4 logits."""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tt.generator import build_generator
from models.common.sampling.generator import SamplingParams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen, trace = None, None
    try:
        gen = build_generator(None, mesh, layer_indices=[0, 3])
        sampled = SamplingParams(temperature=0.8, top_k=32, top_p=0.9, seed=42)
        gen.configure_sampling(sampled, reset_seed=True)
        state = gen.prepare_decode(kv_cache=gen.cache, page_table=gen.page_table, slots=(0,))
        gen.reset()
        gen.prefill_forward(
            torch.arange(100, 133).reshape(1, -1), page_table=gen.page_table, kv_cache=gen.cache, prompt_lens=[33]
        )
        gen._copy(torch.full((1, 1, 1, 32), 234, dtype=torch.int32), state.tokens, "token_refreshes")
        gen._copy(torch.tensor([33], dtype=torch.int32), state.positions[0], "position_refreshes", signed=True)
        state.logits = gen._model_step(state)
        reference = int(gen.model.host_logits(state.logits)[0, 0, 0].argmax())
        gen._sample_step(state)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        gen._sample_step(state)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        count = mesh.num_program_cache_entries()
        mesh.set_program_cache_misses_allowed(False)

        def read(tensor):
            return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).reshape(-1).tolist()

        def sequence(params):
            gen.configure_sampling(params, reset_seed=True)
            before = read(gen.sampler.tt_sampling.seeds_tt_tensor)
            tokens = []
            for _ in range(16):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                tokens.append(read(state.tokens)[0])
            after = read(gen.sampler.tt_sampling.seeds_tt_tensor)
            assert after == [x + 16 for x in before]
            return tokens

        first = sequence(sampled)
        greedy = sequence(SamplingParams(temperature=0, top_k=1, top_p=1, seed=42))
        repeated = sequence(sampled)
        different = sequence(SamplingParams(temperature=0.8, top_k=32, top_p=0.9, seed=43))
        assert first == repeated and first != different
        assert len(set(first)) > 1 and greedy == [reference] * 16
        assert mesh.num_program_cache_entries() == count
        Path(args.output).write_text(
            json.dumps(
                {
                    "status": "pass",
                    "sampled": first,
                    "repeat": repeated,
                    "different_seed": different,
                    "greedy": greedy,
                    "greedy_reference": reference,
                    "program_count": count,
                    "scope": "Frozen real reduced-model logits; sample trace advances request seeds on device",
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        mesh.set_program_cache_misses_allowed(True)
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        if gen:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
