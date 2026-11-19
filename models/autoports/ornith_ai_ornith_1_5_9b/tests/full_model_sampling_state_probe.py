# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Parent-owned bounded device probe for the common sampler's slot-state helper.

No weights or new sampler. Exercises common penalty accumulation with synthetic
token IDs and the existing seed increment, including inactive/offbatch slots.
It is not a full sampling/quality or trace-allocation acceptance gate.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import torch

    import ttnn
    from models.autoports.ornith_ai_ornith_1_5_9b.tt.sampling_state import SamplingSlotState
    from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params

    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
    report = {"synthetic_state_probe": True}
    try:
        sampler = SamplingGenerator(
            args=SimpleNamespace(
                vocab_size=256,
                padded_vocab_size=256,
                max_batch_size=32,
                max_top_k=32,
                cluster_shape=(1, 4),
                sampling_all_gather_axis=1,
            ),
            mesh_device=mesh,
            tt_ccl=None,
        )
        params = SamplingParams(
            temperature=1.0, top_k=32, top_p=0.9, presence_penalty=0.1, frequency_penalty=0.2, repetition_penalty=1.1
        )
        sampler.reset_sampling_params(format_sampling_params(params, 32))
        state = SamplingSlotState(sampler, parameter_owner=SimpleNamespace(sampling_params=params)).prepare()

        def read(tensor, *, sharded=False):
            shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(tensor)]
            return torch.cat(shards, dim=-1) if sharded else shards[0]

        def tables():
            return {
                name: read(getattr(sampler.tt_penalties, name), sharded=name != "output_counts_gathered").clone()
                for name in ("prompt_mask", "output_mask", "output_counts", "output_counts_gathered")
            }

        state.reset_history(slots=[0, 31], prompt_tokens=[[1, 1, 2], [3]], output_tokens=[[7, 7], [8]])
        state.reset_seed_values(slots=list(range(32)), values=list(range(100, 132)))
        original = tables()
        seeds = read(sampler.tt_sampling.seeds_tt_tensor).clone()
        state.snapshot()
        state.reset_history(slots=[0], prompt_tokens=[[4]], output_tokens=[[5]])
        state.reset_seed_values(slots=[0], values=[456])
        changed = tables()
        assert changed["prompt_mask"][0, 4] == 1 and changed["prompt_mask"][0, 1] == 0
        assert all(torch.equal(changed[name][1:], original[name][1:]) for name in changed)
        state.restore()
        assert all(torch.equal(tables()[name], original[name]) for name in original)
        assert torch.equal(read(sampler.tt_sampling.seeds_tt_tensor), seeds)
        report["scoped_reset_and_snapshot"] = "pass"
        permutation = list(reversed(range(32)))
        state.apply_slot_remap(permutation)
        remapped = tables()
        assert all(torch.equal(remapped[name], original[name][permutation]) for name in original)
        assert torch.equal(read(sampler.tt_sampling.seeds_tt_tensor), seeds[permutation])
        report["full32_permutation"] = "pass"
        state.refresh_active_slots([0, 2])
        before = tables()
        seeds_before = read(sampler.tt_sampling.seeds_tt_tensor).clone()
        # Run the same common count-table update used in the sampling trace.
        tokens = ttnn.from_torch(
            torch.arange(32, dtype=torch.int32).reshape(1, 1, 1, 32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        state.before_sample()
        sampler.tt_penalties.update_output_tokens(tokens)
        ttnn.plus_one(sampler.tt_sampling.seeds_tt_tensor, skip_negative_entries=True)
        state.after_sample()
        after = tables()
        seeds_after = read(sampler.tt_sampling.seeds_tt_tensor)
        for slot in range(32):
            if slot in (0, 2):
                assert int(seeds_after[slot]) == int(seeds_before[slot]) + 1
                assert after["output_counts_gathered"][slot, slot] == before["output_counts_gathered"][slot, slot] + 1
            else:
                assert seeds_after[slot] == seeds_before[slot]
                assert all(torch.equal(after[name][slot], before[name][slot]) for name in before)
        report["inactive_seed_and_penalty_state"] = "pass"
        report["program_cache_entries"] = mesh.num_program_cache_entries()
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
