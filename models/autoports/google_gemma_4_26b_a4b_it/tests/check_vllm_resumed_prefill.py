# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced device comparison of resumed prefill and a canonical sampling oracle.

The oracle samples the same re-prefilled logits with explicitly restored request
state. Uninterrupted decode is recorded separately because its precision differs.
"""

import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import AutoportGemma4ForCausalLM
from models.common.sampling.generator import SamplingParams, _hash_request_seed_to_device_seed


def read_tokens(value):
    return ttnn.to_torch(ttnn.get_device_tensors(value)[0]).flatten().tolist()


def sampler_state(generator):
    penalties = generator.sampler.tt_penalties
    state = {"seed": [torch.tensor(read_tokens(generator.sampler.tt_sampling.seeds_tt_tensor)[:1])]}
    for name in ("prompt_mask", "output_mask", "output_counts", "output_counts_gathered"):
        state[name] = [
            ttnn.to_torch(shard).reshape(-1, shard.shape[-1])[:1].clone()
            for shard in ttnn.get_device_tensors(getattr(penalties, name))
        ]
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    try:
        gen = Gemma4Generator(mesh, max_seq_len=256, layer_indices=(0, 5))
        params = SamplingParams(
            temperature=0.5,
            top_k=32,
            top_p=1.0,
            seed=6,
            repetition_penalty=1.5,
            presence_penalty=1.0,
            frequency_penalty=1.0,
        )
        prompt = [2] + [100] * 32
        uninterrupted = gen.generate(prompt, 4, sampling_params=params, stop_on_eos=False)
        retained = uninterrupted[:3]
        prefix = prompt + retained
        adapter = AutoportGemma4ForCausalLM(gen, 1)
        specs = [((8, 2, 32, 256), torch.bfloat16, i) for i in range(30)]
        specs[5] = ((8, 1, 32, 512), torch.bfloat16, 0)
        cache = adapter.allocate_kv_cache_per_layer(specs)
        table = torch.arange(8, dtype=torch.int32).reshape(1, -1)
        tables = [table] * 30
        tables[5] = table.flip(1)
        sample_prefill = gen.sample_prefill
        captured = {}

        def observe(logits):
            captured["state"] = sampler_state(gen)
            captured["logits"] = ttnn.clone(logits)
            return sample_prefill(logits)

        gen.sample_prefill = observe
        resumed = (
            adapter.prefill_forward(
                torch.tensor([prefix]),
                table,
                cache,
                [len(prefix)],
                sampling_params=params,
                page_tables_per_layer=tables,
                start_pos=torch.tensor([0]),
                output_token_counts=torch.tensor([len(retained)]),
            )
            .flatten()
            .tolist()
        )
        gen.sample_prefill = sample_prefill
        gen.configure_sampling(params, prompt_tokens=torch.tensor([prompt]), seed_offsets=torch.tensor([len(retained)]))
        gen.sampler.reset_output_state(torch.tensor([retained]))
        reference_state = sampler_state(gen)
        state_matches = {
            name: all(torch.equal(actual, expected) for actual, expected in zip(captured["state"][name], shards))
            for name, shards in reference_state.items()
        }
        reference = read_tokens(gen.sample_prefill(captured["logits"]))[:1]
        actual_seed = int(captured["state"]["seed"][0].item())
        expected_seed = _hash_request_seed_to_device_seed(6, 0) + len(retained)
        report = {
            "scope": "Reduced layers 0 and 5; resumed adapter prefill versus the same device logits and explicit canonical state",
            "prompt_tokens": prompt,
            "retained_output_tokens": retained,
            "logical_prefix_length": len(prefix),
            "sampler_state_matches": state_matches,
            "actual_seed": actual_seed,
            "expected_seed": expected_seed,
            "resumed_next_token": resumed,
            "same_logits_reference_next_token": reference,
            "uninterrupted_next_token_information_only": uninterrupted[3],
            "runtime_precision": gen.model.precision_summary(),
            "counters": gen.counters,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        assert all(state_matches.values()), state_matches
        assert actual_seed == expected_seed, report
        assert resumed == reference, report
    finally:
        if gen is not None:
            gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
