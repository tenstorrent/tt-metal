# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Canonical traced penalty histories, masks, remaps and PC-04 adapter boundary."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator_vllm import QwenVLLM


def first(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    adapter = None
    try:
        adapter = QwenVLLM.initialize_vllm_model(None, mesh, 4, 262144)
        cache = adapter.allocate_kv_cache((8196, 1, 32, 256), torch.bfloat16, 16)
        gen = adapter.generator
        for engine in {gen.prefill_single, gen.prefill_uniform}:
            engine.prepare(_diagnostic_lengths=[1, 2, 3, 32, 33, 65])
        adapter.warmup_model_prefill(kv_cache=cache)
        penalties = gen.sampling.tt_penalties
        baseline_traces = dict(gen.traces)
        results = []
        vocab = gen.model.config.vocab_size
        # Full logical logits are sharded on host upload, outside measured decode.
        mapper = ttnn.ShardTensor2dMesh(mesh, dims=(None, 3), mesh_shape=mesh.shape)

        def upload_logits(values):
            host = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
            ttnn.copy_host_to_device_tensor(host, gen.logits)

        def histories(prompt, output, slots):
            p = torch.full((4, max(1, max(map(len, prompt)))), -1, dtype=torch.int64)
            o = torch.full((4, max(1, max(map(len, output)))), -1, dtype=torch.int64)
            for row in range(4):
                p[row, : len(prompt[row])] = torch.tensor(prompt[row])
                o[row, : len(output[row])] = torch.tensor(output[row])
            gen.penalty_history.reset(slots=slots, prompt_tokens=p, output_tokens=o)
            gen.penalty_history_slots.update(slots)

        def sample(mask):
            gen._write(
                torch.nn.functional.pad(torch.tensor([mask], dtype=torch.bfloat16), (0, 28)), gen.sample_active_rows
            )
            slots = [i for i, active in enumerate(mask) if active]
            gen._execute(gen._sample_trace_key(slots))
            return gen._read_tokens()

        for name, logits_values, prompt, output, presence, frequency, repetition in (
            ("positive_repetition", [4.0, 3.0, 2.0], [0], [], 0.0, 0.0, 2.0),
            ("negative_repetition", [-1.0, -1.5, -3.0], [0], [], 0.0, 0.0, 2.0),
            ("presence_output_only", [4.0, 3.0, 2.0], [1], [0], 2.0, 0.0, 1.0),
            ("frequency_repeats", [4.0, 3.0, 2.0], [1], [0, 0], 0.0, 1.0, 1.0),
            ("prompt_not_output", [4.0, 3.0, 2.0], [0, 0], [], 2.0, 1.0, 1.0),
            ("combined", [5.0, 4.0, 3.0], [0], [1, 1], 1.0, 0.5, 2.0),
        ):
            gen.reset()
            gen.configure_sampling(
                presence_penalty=[presence, 0, 0, 0],
                frequency_penalty=[frequency, 0, 0, 0],
                repetition_penalty=[repetition, 1, 1, 1],
            )
            histories([prompt, [], [], []], [output, [], [], []], [0])
            logits = torch.full((1, 1, 32, vocab), -100.0, dtype=torch.bfloat16)
            logits[0, 0, 0, :3] = torch.tensor(logits_values)
            upload_logits(logits)
            before = first(gen.logits)
            expected = torch.tensor(logits_values)
            for token in set(prompt + output):
                expected[token] = expected[token] / repetition if expected[token] > 0 else expected[token] * repetition
            for token in set(output):
                expected[token] -= presence + frequency * output.count(token)
            expected_token = int(expected.argmax())
            actual = sample([1, 0, 0, 0])[0]
            assert actual == expected_token, (name, actual, expected_token)
            assert torch.equal(first(gen.logits), before), "raw logits modified"
            counts = first(penalties.output_counts_gathered)
            assert int(counts[0, actual]) == output.count(actual) + 1
            assert not counts[1:].any(), "inactive rows were counted"
            results.append(dict(case=name, token=actual, expected=expected_token))

        # Replace, rather than add, committed history after first sampled prefill.
        histories([[0], [], [], []], [[1, 1], [], [], []], [0])
        histories([[0], [], [], []], [[1, 1], [], [], []], [0])
        assert int(first(penalties.output_counts_gathered)[0, 1]) == 2
        # Slot remaps preserve device counts, prompt mask and params, including
        # while host mode is dormant. Return to device uses the same trace.
        gen.configure_sampling(presence_penalty=[1, 0, 0, 0])
        gen.remap_slots([2, 1, 0, 3])
        counts = first(penalties.output_counts_gathered)
        assert int(counts[2, 1]) == 2 and not counts[0].any(), counts[:4, :3].tolist()
        assert int(first(penalties.prompt_mask)[2, 0]) == 1
        assert float(first(penalties.presence_penalties)[2, 0]) == 1
        gen.release_request(2)
        assert not first(penalties.output_counts_gathered)[2].any()
        assert not first(penalties.prompt_mask)[2].any()

        # Actual adapter final prefill distinguishes identical concatenated
        # token history by original prompt boundary; partial chunks may report
        # a boundary beyond the visible tensor width and must not count.
        params = SimpleNamespace(
            temperature=[0.0],
            top_k=[1],
            top_p=[0.0],
            seed=[0],
            presence_penalty=[0.5],
            frequency_penalty=[0.5],
            repetition_penalty=[1.1],
        )
        tokens = torch.tensor([[70000, 70001, 70002]])
        table = gen.page_host[:1]

        def prefill(boundary, final=True, end=3, start=0):
            return adapter.prefill_forward(
                tokens,
                page_table=table,
                kv_cache=cache,
                prompt_lens=[end],
                start_pos=[start],
                empty_slots=[3],
                sampling_params=params,
                sampling_mask=[final],
                original_prompt_lens=[boundary],
            )

        gen.reset()
        prefill(5, False, end=1)
        assert not first(penalties.output_counts_gathered).any()
        prefill(1, True, start=1)
        replay = first(penalties.output_counts_gathered)[3]
        assert replay[70001] >= 1 and replay[70002] >= 1 and replay.sum() == 3
        prefill(3)
        fresh = first(penalties.output_counts_gathered)[3]
        assert fresh.sum() == 1
        assert torch.equal(
            first(penalties.output_counts_gathered)[:3], torch.zeros_like(first(penalties.output_counts_gathered)[:3])
        )
        assert gen.traces == baseline_traces
        assert gen.counters["retirements"] == 0
        result = dict(
            passed=True,
            cases=results,
            raw_logits_preserved=True,
            inactive_count_mask=True,
            replacement_no_double_count=True,
            remap=True,
            release=True,
            pc04_replay_boundary=True,
            intermediate_boundary_exceeds_width=True,
            traces={k: int(v) for k, v in gen.traces.items()},
            counters=dict(gen.counters),
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if adapter is not None and adapter.generator is not None:
            adapter.generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
