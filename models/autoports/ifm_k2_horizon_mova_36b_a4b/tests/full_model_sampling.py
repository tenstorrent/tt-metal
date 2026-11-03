# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sampler state/semantics fixture on the real reduced wrapper and tensor contract.

Synthetic logits isolate sampling semantics; this is not model accuracy evidence.
"""
import argparse
import json
from pathlib import Path

import torch

import ttnn
from models.common.sampling.generator import SamplingParams

from ..tt.generator import DecodeOutput, build_generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        gen = build_generator(None, mesh, override_num_layers=2, max_seq_len=1024, max_batch_size=4)
        gen.capture()
        fixture = torch.full((1, 1, 32, 262144), -80.0, dtype=torch.bfloat16)
        candidates = [101, 10001, 60001, 180000, 250623]
        fixture[0, 0, :, candidates] = torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0], dtype=torch.bfloat16)
        host = ttnn.from_torch(
            fixture, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1)
        )
        ttnn.copy_host_to_device_tensor(host, gen.logits)
        gen.set_active_slots([0, 1, 2, 3])
        history = torch.full((32, 1), -1, dtype=torch.int64)
        result = {"fixture": "five cross-shard logits4,3,2,1,0; others-80; physical32x262144"}

        def reset(params, prompts=history, outputs=history):
            gen.update_sampling_state(
                params,
                start_pos=torch.zeros(4, dtype=torch.int32),
                reload_params=True,
                reset_state=True,
                prompt_tokens=prompts,
                output_tokens=outputs,
            )

        def sample():
            # The production model writes fresh logits before every sample.
            # Penalties intentionally mutate the input, so restore this fixture.
            ttnn.copy_host_to_device_tensor(host, gen.logits)
            tid, output = gen._capture_sampler()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            return gen.process_decode_output_host(gen.read_decode_output(DecodeOutput(gen.tokens, output[1], True)))

        greedy = SamplingParams([0.0] * 4, [1] * 4, [1.0] * 4)
        reset(greedy)
        assert sample().reshape(-1).tolist() == [101] * 4
        sampled = SamplingParams([1.0] * 4, [5] * 4, [1.0] * 4, seed=[42, 42, -7, 0])
        sequences = []
        for _ in range(2):
            reset(sampled)
            sequence = [sample().reshape(-1).tolist() for step in range(12)]
            assert all(row[0] == row[1] for row in sequence), sequence
            sequences.append(sequence)
        assert sequences[0] == sequences[1]
        result["seeded_reproducibility"] = sequences[0]
        reset(SamplingParams([1.0] * 4, [5] * 4, [0.8] * 4))
        unseeded1 = [sample().reshape(-1).tolist() for _ in range(8)]
        reset(SamplingParams([1.0] * 4, [5] * 4, [0.8] * 4))
        unseeded2 = [sample().reshape(-1).tolist() for _ in range(8)]
        assert unseeded1 != unseeded2
        result["unseeded_variety"] = [unseeded1, unseeded2]
        for name in ("presence_penalty", "frequency_penalty", "repetition_penalty"):
            params = SamplingParams([0.0] * 4, [1] * 4, [1.0] * 4, **{name: [2.0] * 4})
            counts = torch.full((32, 1), 101)
            reset(params, prompts=counts, outputs=counts)
            out = sample().reshape(-1).tolist()
            assert out == [10001] * 4, (name, out)
            result[name] = out
        reset(SamplingParams([0.0] * 4, [1] * 4, [1.0] * 4, enable_log_probs=True, num_logprobs=10))
        logprob_result = sample()
        print(
            "LOGPROB_RESULT",
            type(logprob_result),
            gen._sampling_key(),
            gen.sampler.tt_sampling.log_probs_calculator._use_topk_logprobs,
            [type(v) for v in logprob_result],
            flush=True,
        )
        tokens, (lp, indices) = logprob_result
        golden = fixture[0, 0, :4, :250624].float().log_softmax(-1)
        actual = golden.gather(1, indices.long())
        assert torch.allclose(lp.float(), actual, atol=0.15, rtol=0.01), (lp, actual)
        assert torch.equal(indices[:, :5].long(), torch.tensor(candidates).repeat(4, 1))
        result["logprobs_max_abs_error"] = (lp.float() - actual).abs().max().item()
        mixed = SamplingParams(
            temperature=[0.0, 0.7, 1.3, 0.0],
            top_k=[1, 2, 5, 1],
            top_p=[1.0, 0.9, 0.6, 1.0],
            seed=[17, 18, -7, 20],
            presence_penalty=[0.0, 0.0, 0.0, 2.0],
            frequency_penalty=[0.0, 0.2, 0.0, 0.5],
            repetition_penalty=[1.0, 1.0, 1.5, 2.0],
            enable_log_probs=True,
            num_logprobs=10,
        )
        counts = torch.full((32, 1), 101)
        mixed_sequences = []
        for _ in range(2):
            reset(mixed, prompts=counts, outputs=counts)
            first_tokens, (first_lp, first_indices) = sample()
            expected_logits = fixture[0, 0, :4, :250624].float().clone()
            expected_logits[:, 101] = torch.tensor([4.0, 3.8, 4.0 / 1.5, -0.5])
            expected_lp = expected_logits.log_softmax(-1).gather(1, first_indices.long())
            assert torch.allclose(first_lp.float(), expected_lp, atol=0.15, rtol=0.01)
            assert first_tokens[0, 0] == 101 and first_tokens[3, 0] == 10001
            assert first_tokens[1, 0] in (101, 10001)
            sequence = [first_tokens.reshape(-1).tolist()]
            sequence += [sample()[0].reshape(-1).tolist() for _ in range(7)]
            mixed_sequences.append(sequence)
        assert mixed_sequences[0] == mixed_sequences[1]
        result["mixed_parameters_and_penalties_logprobs"] = dict(
            sequences=mixed_sequences,
            first_logprobs_max_abs_error=float((first_lp.float() - expected_lp).abs().max()),
            first_top5_logprobs_max_abs_error=float((first_lp[:, :5].float() - expected_lp[:, :5]).abs().max()),
        )
        reset(greedy)
        assert sample().reshape(-1).tolist() == [101] * 4
        old_seeds = ttnn.to_torch(ttnn.get_device_tensors(gen.sampler.tt_sampling.seeds_tt_tensor)[0]).clone()
        gen.apply_slot_remap([2, 0, 3, 1])
        moved = ttnn.to_torch(ttnn.get_device_tensors(gen.sampler.tt_sampling.seeds_tt_tensor)[0])
        assert torch.equal(moved[:4], old_seeds[[2, 0, 3, 1]])
        result["slot_seed_remap"] = True
        reset(greedy)
        assert sample().reshape(-1).tolist() == [101] * 4
        retained = {k: v[0] for k, v in gen.sampling_traces.items()}
        prefill_ids = {}
        for lengths in ([33, 65, 129, 97], [129, 33, 97, 65], [33, 65, 129, 97]):
            prompt = torch.arange(10, 10 + max(lengths)).repeat(4, 1)
            gen.prefill_forward(
                prompt,
                page_table=gen.page_table_host,
                kv_cache=gen.kv_cache,
                prompt_lens=lengths,
                sampling_mode="device",
                sampling_params=mixed,
            )
            reset(sampled)
            assert sample().reshape(-1).tolist() == sequences[0][0]
            reset(greedy)
            assert sample().reshape(-1).tolist() == [101] * 4
            assert {k: v[0] for k, v in gen.sampling_traces.items()} == retained
            for key, tid in prefill_ids.items():
                assert gen.prefill_traces[key] == tid
            prefill_ids = dict(gen.prefill_traces)
        result["prefill_interleaved_sampling_reuse"] = dict(
            reordered_lengths=True,
            seeded_stream_restored=True,
            prefill_trace_ids={str(k): str(v) for k, v in prefill_ids.items()},
            captures=gen.counters["prefill_captures"],
        )
        assert gen.counters["prefill_captures"] == len(prefill_ids)
        result["sampling_trace_count"] = len(gen.sampling_traces)
        result["trace_ids"] = {str(k): str(v[0]) for k, v in gen.sampling_traces.items()}
        result["counters"] = dict(gen.counters)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
