"""Reduced wrapper state/lifetime gate; no full-model quality claim."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn

from ..tt.generator import Gemma4Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    a = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    try:
        gen = Gemma4Generator(mesh, layer_indices=(0, 5), batch_size=a.batch, max_seq_len=4096)
        gen.prepare_decode()
        trace_ids = (gen.model_trace, gen.sample_trace)
        baseline = mesh.num_program_cache_entries()
        prompts = [[2, 125, 89] * ((31 + slot * 17) // 3) + [2] * ((31 + slot * 17) % 3) for slot in range(a.batch)]
        if a.batch > 1:
            prompts[-1] = None
        before = dict(gen.counters)
        feedback = []
        replay = gen._replay

        def checked_replay(*args, **kwargs):
            inputs = [ttnn.to_torch(t).reshape(-1)[: a.batch].tolist() for t in ttnn.get_device_tensors(gen.tokens)]
            positions = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1).tolist()
            assert all(row == inputs[0] for row in inputs)
            if feedback:
                assert inputs[0] == feedback[-1]["sampled"]
            result = replay(*args, **kwargs)
            after = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1).tolist()
            assert after == [p + 1 if p >= 0 else -1 for p in positions]
            feedback.append(
                dict(token_input=inputs[0], sampled=result, positions_before=positions, positions_after=after)
            )
            return result

        with patch.object(gen, "_replay", side_effect=checked_replay):
            first = gen.generate_batch(prompts, 4)
        assert feedback[0]["token_input"] == [
            row[0] if row else feedback[0]["token_input"][i] for i, row in enumerate(first)
        ]
        assert any(a != b for a, b in zip(feedback[0]["token_input"], feedback[1]["token_input"])), feedback
        middle = dict(gen.counters)
        again = gen.generate_batch(prompts, 4)
        assert first == again, (first, again)
        assert gen.counters["token_refreshes"] == before["token_refreshes"]
        assert gen.counters["page_refreshes"] == middle["page_refreshes"]
        assert (gen.model_trace, gen.sample_trace) == trace_ids and gen.counters["captures"] == 2
        assert mesh.num_program_cache_entries() == baseline
        current = ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1).tolist()
        assert current == [len(p) + 3 if p else -1 for p in prompts], current
        if a.batch > 1:
            assert first[-1] == []
        # Change the virtual→physical map, reset, and confirm logical output.
        maps = []
        for layer, table, cache in zip(gen.model.layers, gen.prefill.host_pages, gen.kv_cache):
            table = table.clone()
            # Reverse physical block IDs across the whole cache. Ownership stays
            # disjoint and the cyclic sliding sequence is still a permutation.
            maps.append(cache[0].shape[0] - 1 - table)
        gen.refresh_page_tables(maps, revision="reversed")
        gen.owns_cache = False  # external scheduler now owns the tested mapping
        remapped = gen.generate_batch(prompts, 4)
        assert remapped == first, (remapped, first)
        gen.owns_cache = True  # Longer later requests need fresh adequately sized ownership maps.
        # Alternate parameter values on the same prepared top-k-capable graph.
        params = {"temperature": 0.8, "top_k": 32, "top_p": 0.9, "seed": 101}
        sampled = gen.generate_batch(prompts, 16, sampling_params=params, ignore_eos=True)
        sampled_repeat = gen.generate_batch(prompts, 16, sampling_params=params, ignore_eos=True)
        other_seed = gen.generate_batch(prompts, 16, sampling_params={**params, "seed": 102}, ignore_eos=True)
        assert sampled == sampled_repeat, "Reset must reproduce the seeded sampled sequence"
        active = [slot for slot, prompt in enumerate(prompts) if prompt]
        assert all(sampled[slot] != other_seed[slot] for slot in active), "Seed changes must affect every active slot"
        greedy_long = gen.generate_batch(prompts, 16, ignore_eos=True)
        assert all(sampled[slot] != greedy_long[slot] for slot in active), "Every active slot must sample"

        greedy = gen.generate_batch(prompts, 4)
        assert greedy == first
        assert (gen.model_trace, gen.sample_trace) == trace_ids and gen.counters["captures"] == 2
        assert mesh.num_program_cache_entries() == baseline
        boundary_lengths = [1023, 1024, 1025]
        boundary_outputs = {}
        for length in boundary_lengths:
            boundary_prompts = [[2, 125, 89] * (length // 3) + [2] * (length % 3) for _ in range(a.batch)]
            if a.batch > 1:
                boundary_prompts[-1] = None
            boundary = gen.generate_batch(boundary_prompts, 4, ignore_eos=True)
            assert boundary == gen.generate_batch(boundary_prompts, 4, ignore_eos=True)
            assert (gen.model_trace, gen.sample_trace) == trace_ids and gen.counters["captures"] == 2
            assert mesh.num_program_cache_entries() == baseline
            boundary_outputs[length] = boundary
        report = dict(
            scope="reduced layers0,5",
            batch=a.batch,
            outputs=first,
            feedback=feedback,
            positions=current,
            traces=[str(t) for t in trace_ids],
            program_cache_entries=baseline,
            counters=gen.counters,
            repeat_equal=True,
            page_remap_equal=True,
            greedy_after_sample_equal=True,
            chunk_boundary_repeat_outputs=boundary_outputs,
            seeded_sample_repeat_equal=True,
            changed_seed_changes_output=True,
            sampled_outputs=sampled,
            other_seed_outputs=other_seed,
            zero_steady_token_refreshes=True,
            inactive_rows=a.batch > 1,
        )
        Path(f"models/autoports/google_gemma_4_31b_it/doc/full_model/generator_state_b{a.batch}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(report, flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
