# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Synchronized B1 full-model TTFT and canonical split-trace token-out benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--layer-indices", type=int, nargs="+", default=None)
    parser.add_argument("--profile-only-decode", action="store_true")
    parser.add_argument("--candidate-gather-greedy", action="store_true")
    parser.add_argument("--force-argmax-greedy", action="store_true")
    parser.add_argument("--disable-power-of-two-sampling-pad", action="store_true")
    parser.add_argument("--feedback-overwrite-probe", action="store_true")
    parser.add_argument("--gather-debug-probe", action="store_true")
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    generator = None
    try:
        generator = build_generator(
            model_dir=Path("models/autoports/qwen_qwen3_6_27b"),
            mesh_device=mesh,
            max_context=512,
            batch=1,
            num_layers=args.num_layers,
            layer_indices=args.layer_indices,
            force_argmax_greedy=args.force_argmax_greedy,
            pad_sampling_logits_to_power_of_2=not args.disable_power_of_two_sampling_pad,
        )
        rendered = generator.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt.read_text().strip()}],
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = generator.tokenizer.encode(rendered, add_special_tokens=False)
        token_ids = token_ids[: args.prompt_tokens]
        tokens = torch.tensor([token_ids], dtype=torch.long)

        generator.reset()
        ttnn.synchronize_device(mesh)
        started = time.perf_counter()
        logits = generator.prefill_forward(
            tokens,
            page_table=generator._page_table,
            kv_cache=generator.kv_cache,
            prompt_lens=[len(token_ids)],
        )
        ttnn.synchronize_device(mesh)
        ttft_seconds = time.perf_counter() - started
        first_token = int(torch.argmax(logits[0, 0]).item())

        capture_started = time.perf_counter()
        generator.sampling.tt_sampling.debug_preserve_force_argmax_gather = args.gather_debug_probe
        generator._capture_token_out_trace(first_token, len(token_ids))
        feedback_probe = None
        if args.feedback_overwrite_probe:
            # Trace capture records the model graph but does not execute it.
            # Populate the persistent captured logits exactly as the real
            # token-out loop does before comparing the sampler with the host
            # semantic-greedy oracle.  Reading the buffer immediately after
            # capture observes allocator/stale contents, not model output.
            ttnn.execute_trace(mesh, generator._decode_trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            gathered_logits = ttnn.to_torch(
                generator._trace_logits,
                mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1),
            )[..., : generator.model.vocab_size]
            expected = int(torch.argmax(gathered_logits.reshape(-1, gathered_logits.shape[-1])[0]).item())
            pre_sample_input_tensors = list(ttnn.get_device_tensors(generator._trace_logits))
            pre_sample_input_shards = [ttnn.to_torch(tensor) for tensor in pre_sample_input_tensors]
            pre_sample_input_addresses = [int(tensor.buffer_address()) for tensor in pre_sample_input_tensors]
            generator._seed_token_out_trace(123, len(token_ids))
            before = generator._read_sampled_token()
            generator.sampling.sample(generator._trace_logits, enable_trace=True, tt_out_tok=generator._trace_token)
            ttnn.synchronize_device(mesh)
            after = generator._read_sampled_token()
            feedback_probe = {
                "seeded": before,
                "expected_global_argmax": expected,
                "sampled": after,
                "overwritten": after != before,
                "semantic_greedy": after == expected,
            }
            if args.gather_debug_probe:
                gathered_tt = generator.sampling.tt_sampling.debug_force_argmax_gather
                post_sample_input_tensors = list(ttnn.get_device_tensors(generator._trace_logits))
                input_shards = [ttnn.to_torch(tensor) for tensor in post_sample_input_tensors]
                gathered_shards = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(gathered_tt)]
                expected_gather = torch.cat(pre_sample_input_shards, dim=-1)

                def summary(tensor):
                    row = tensor.reshape(-1, tensor.shape[-1])[0].float()
                    values, indices = torch.topk(row, k=8)
                    return {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "top_indices": [int(value) for value in indices],
                        "top_values": [float(value) for value in values],
                    }

                def comparison(reference_tensor, actual_tensor):
                    reference = reference_tensor.float().reshape(-1)
                    actual = actual_tensor[..., : reference_tensor.shape[-1]].float().reshape(-1)
                    reference_centered = reference - reference.mean()
                    actual_centered = actual - actual.mean()
                    denominator = torch.linalg.vector_norm(reference_centered) * torch.linalg.vector_norm(
                        actual_centered
                    )
                    pcc = (
                        1.0
                        if denominator == 0 and torch.equal(reference, actual)
                        else (
                            float("nan")
                            if denominator == 0
                            else float(torch.dot(reference_centered, actual_centered) / denominator)
                        )
                    )
                    return {
                        "pcc": pcc,
                        "max_abs": float(torch.max(torch.abs(reference - actual))),
                        "allclose_rtol_1e-2_atol_1e-2": bool(torch.allclose(reference, actual, rtol=1e-2, atol=1e-2)),
                    }

                per_rank = []
                for rank, actual_tensor in enumerate(gathered_shards):
                    actual = actual_tensor[..., : expected_gather.shape[-1]]
                    per_rank.append(
                        {
                            "rank": rank,
                            "vs_host_composed": comparison(expected_gather, actual),
                            "summary": summary(actual),
                        }
                    )
                feedback_probe["gather_debug"] = {
                    "trace_logits_object_id": id(generator._trace_logits),
                    "input_addresses_before": pre_sample_input_addresses,
                    "input_addresses_after": [int(tensor.buffer_address()) for tensor in post_sample_input_tensors],
                    "input_pre_vs_post": [
                        comparison(before_tensor, after_tensor)
                        for before_tensor, after_tensor in zip(pre_sample_input_shards, input_shards)
                    ],
                    "input_shards": [summary(tensor) for tensor in input_shards],
                    "host_composed": summary(expected_gather),
                    "captured_gather_per_rank": per_rank,
                    "gather_addresses": [
                        int(tensor.buffer_address()) for tensor in ttnn.get_device_tensors(gathered_tt)
                    ],
                }
                # Snapshot the captured result above, then release only the
                # sampler trace and execute eagerly on the identical model
                # trace-output tensor/address.  No model replay follows this
                # destructive diagnostic comparison.
                generator.sampling.reset_trace()
                generator.sampling.tt_sampling.debug_force_argmax_gather = None
                generator._seed_token_out_trace(123, len(token_ids))
                generator.sampling.sample(
                    generator._trace_logits, enable_trace=False, tt_out_tok=generator._trace_token
                )
                ttnn.synchronize_device(mesh)
                eager_token = generator._read_sampled_token()
                eager_gather = generator.sampling.tt_sampling.debug_force_argmax_gather
                eager_gather_tensors = list(ttnn.get_device_tensors(eager_gather))
                eager_shards = [ttnn.to_torch(tensor) for tensor in eager_gather_tensors]
                feedback_probe["gather_debug"]["eager_token"] = eager_token
                feedback_probe["gather_debug"]["eager_gather_per_rank"] = [
                    {
                        "rank": rank,
                        "vs_host_composed": comparison(expected_gather, tensor),
                        "vs_captured_gather": comparison(gathered_shards[rank], tensor),
                        "summary": summary(tensor),
                    }
                    for rank, tensor in enumerate(eager_shards)
                ]
                feedback_probe["gather_debug"]["eager_input_addresses"] = [
                    int(tensor.buffer_address()) for tensor in ttnn.get_device_tensors(generator._trace_logits)
                ]
                feedback_probe["gather_debug"]["eager_gather_addresses"] = [
                    int(tensor.buffer_address()) for tensor in eager_gather_tensors
                ]
            if not feedback_probe["semantic_greedy"] or not feedback_probe["overwritten"]:
                if args.gather_debug_probe:
                    print(json.dumps({"feedback_overwrite_probe": feedback_probe}, indent=2), flush=True)
                raise AssertionError(f"sampling trace failed semantic feedback probe: {feedback_probe}")
        generator._seed_token_out_trace(first_token, len(token_ids))
        ttnn.synchronize_device(mesh)
        capture_seconds = time.perf_counter() - capture_started

        output = [first_token]
        if args.profile_only_decode:
            # Construction, prefill, compilation and trace capture contain far
            # more markers than device profiler buffers can retain. Flush them
            # before the requested steady-state terminal/sampler replay.
            ttnn.ReadDeviceProfiler(mesh)
            signpost("FULL_MODEL_DECODE", "one reduced model + canonical sampler replay")
        started = time.perf_counter()
        for _ in range(args.decode_tokens):
            generator.token_out_decode_step(readback=False)
        ttnn.synchronize_device(mesh)
        decode_seconds = time.perf_counter() - started
        # One reporting readback after the measured replay interval. The
        # persistent token remains device-owned for every timed step.
        final_sampled_token = generator._read_sampled_token()
        output.append(final_sampled_token)
        if args.profile_only_decode:
            ttnn.ReadDeviceProfiler(mesh)
            signpost("FULL_MODEL_DECODE_END")

        result = {
            "prompt_tokens": len(token_ids),
            "measured_decode_replays": args.decode_tokens,
            "ttft_seconds": ttft_seconds,
            "ttft_ms": 1000 * ttft_seconds,
            "trace_capture_seconds": capture_seconds,
            "token_out_seconds": decode_seconds,
            "token_out_t_s_u": args.decode_tokens / decode_seconds,
            "trace_counters": dict(generator.trace_counters),
            "canonical_split_sampling": True,
            "model_trace_id": str(generator._decode_trace_id),
            "sampler_trace_live": any(slot["id"] is not None for slot in generator.sampling._trace_states.values()),
            "feedback_overwrite_probe": feedback_probe,
            "measured_per_token_readback": False,
            "final_sampled_token": final_sampled_token,
            "token_ids": output,
            "text": generator.tokenizer.decode(output, skip_special_tokens=False),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key not in ("token_ids", "text")}, indent=2))
    finally:
        if generator is not None:
            generator.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
