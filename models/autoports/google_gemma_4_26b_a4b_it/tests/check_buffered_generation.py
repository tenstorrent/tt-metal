# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare buffered trace output with the token-at-a-time generator contract."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator, build_generator
from models.common.sampling.generator import SamplingParams


def read(tensor):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recorder-only", action="store_true")
    parser.add_argument("--all-layers", action="store_true")
    parser.add_argument("--prompt-len", type=int, default=33)
    parser.add_argument("--gen-len", type=int, default=16)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", help="Additional logical prompt boundary lengths")
    args = parser.parse_args()
    if args.gen_len < 4 or args.prompt_len < 1 or any(length < 1 for length in args.prompt_lengths or []):
        parser.error("The primary generation length must be >= 4 and prompt lengths must be positive")
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
    gen = None
    report = {}
    try:
        if args.recorder_only:
            gen = Gemma4Generator.__new__(Gemma4Generator)
            gen.mesh = mesh
            gen.counters = {}
            gen.output_trace_id = None
            gen.output_buffer = None
            gen.model = SimpleNamespace(
                upload=lambda value, dtype, layout: ttnn.from_torch(
                    value,
                    dtype=dtype,
                    layout=layout,
                    device=mesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
            )
            gen.tokens = gen.model.upload(
                torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
            )
            for steps in (1, 3, 127, 127):
                gen._prepare_output_buffer(steps)
                gen._capture_output_buffer()
                expected = []
                for step in range(steps):
                    values = torch.arange(32, dtype=torch.int32).reshape(1, 1, 1, 32) + step * 1000 + 17
                    gen._copy(values, gen.tokens, "probe_token_writes")
                    expected.append(values)
                    ttnn.execute_trace(mesh, gen.output_trace_id, cq_id=0, blocking=False)
                assert torch.equal(read(gen.output_buffer).long(), torch.cat(expected).long())
                assert read(gen.output_index).item() == steps
            report["recorder_exact_uint32_rows"] = True
        else:
            gen = build_generator(
                None,
                mesh,
                max_seq_len=max(1024, max([args.prompt_len] + (args.prompt_lengths or [])) + args.gen_len + 2),
                layer_indices=None if args.all_layers else (0, 5),
            )
            prompt = [2] + [100] * (args.prompt_len - 1)
            baseline = gen.generate(prompt, args.gen_len, stop_on_eos=False, buffer_tokens=False)
            report["baseline_metrics"] = dict(gen.metrics)
            record_token = gen._record_token
            recorder_calls = []

            def audited_record_token():
                with device_only():
                    record_token()
                recorder_calls.append(True)

            with patch.object(gen, "_record_token", side_effect=audited_record_token):
                buffered = gen.generate(prompt, args.gen_len, stop_on_eos=False)
            assert len(recorder_calls) == 2  # warm and capture
            report["recorder_warm_and_capture_runtime_audit"] = True
            report["buffered_metrics"] = dict(gen.metrics)
            assert baseline == buffered, (baseline, buffered)
            assert read(gen.output_index).item() == args.gen_len - 1
            assert int(read(gen.tokens).flatten()[0]) == buffered[-1]
            counters = gen.metrics["counters"]
            assert counters["token_readbacks"] == 1
            assert counters["model_replays"] == counters["sampling_replays"] == args.gen_len - 1
            assert counters["output_replays"] == args.gen_len - 1
            for name in (
                "token_refreshes",
                "position_refreshes",
                "cache_position_refreshes",
                "page_table_refreshes",
                "synchronizations",
                "full_logits_readbacks",
                "teacher_forcing_token_refreshes",
            ):
                assert counters[name] == 0, (name, counters)
            trace = gen.output_trace_id
            repeated = gen.generate(prompt, args.gen_len, stop_on_eos=False)
            assert repeated == buffered and gen.output_trace_id == trace
            for length in (2, 3, args.gen_len):
                tail = gen.generate(prompt, length, stop_on_eos=False)
                assert tail == baseline[:length]
            report["same_tokens_repeated_and_resized"] = True

            # Generation-length edge cases cannot require a recorder replay.
            previous_trace = gen.trace_id
            previous_positions = read(gen.cache_positions).clone()
            assert gen.generate(prompt, 0, stop_on_eos=False) == []
            assert gen.trace_id == previous_trace and torch.equal(read(gen.cache_positions), previous_positions)
            assert gen.generate(prompt, 1, stop_on_eos=False) == baseline[:1]
            assert gen.generate(prompt, 1, stop_on_eos=False, buffer_tokens=False) == baseline[:1]
            report["zero_and_single_token"] = True

            # Start with a genuinely small allocation, then request a larger
            # one while decode traces are alive. Allocation tracking must pass.
            gen._release_trace()
            assert gen.generate(prompt, 3, stop_on_eos=False) == baseline[:3]
            assert gen.output_buffer.shape[0] == 2
            small_trace = gen.trace_id
            assert gen.generate(prompt, args.gen_len, stop_on_eos=False) == baseline
            assert gen.output_buffer.shape[0] >= args.gen_len - 1
            assert gen.trace_id != small_trace and not gen.metrics["reused_request_trace"]
            report["capacity_growth_recaptures"] = True

            # Audit the actual replay loop and the recorder op construction.
            # Index reset belongs at the request boundary, outside the audit.
            gen._copy(torch.zeros(1, dtype=torch.int32), gen.output_index, "audit_index_reset")
            before_positions = read(gen.cache_positions).clone()
            with device_only():
                for _ in range(2):
                    gen._replay()
                    ttnn.execute_trace(mesh, gen.output_trace_id, cq_id=0, blocking=False)
            assert torch.equal(read(gen.cache_positions), before_positions + 2)
            assert read(gen.output_index).item() == 2
            gen.reset()
            assert gen.generate(prompt, args.gen_len, stop_on_eos=False) == baseline
            report["buffered_replay_runtime_audit"] = True
            report["reset_restores_positions_after_extra_replays"] = True

            changed_prompt = [2] + [101] * (args.prompt_len - 1)
            changed_control = gen.generate(changed_prompt, args.gen_len, stop_on_eos=False, buffer_tokens=False)
            changed_buffered = gen.generate(changed_prompt, args.gen_len, stop_on_eos=False)
            assert changed_control == changed_buffered
            assert gen.generate(prompt, args.gen_len, stop_on_eos=False) == baseline
            report["changed_prompt_contents_no_cross_request_leakage"] = True

            sampled = SamplingParams(temperature=0.8, top_k=16, top_p=0.9, seed=42)
            sampled_control = gen.generate(
                prompt, args.gen_len, sampling_params=sampled, stop_on_eos=False, buffer_tokens=False
            )
            sampled_buffered = gen.generate(prompt, args.gen_len, sampling_params=sampled, stop_on_eos=False)
            assert sampled_control == sampled_buffered, (sampled_control, sampled_buffered)
            assert gen.metrics["counters"]["token_readbacks"] == 1
            report["sampled_counters"] = dict(gen.metrics["counters"])
            assert gen.generate(prompt, args.gen_len, sampling_params=sampled, stop_on_eos=False) == sampled_control
            assert gen.generate(prompt, args.gen_len, stop_on_eos=False) == baseline
            report["seeded_topk_topp_and_greedy_alternation"] = True
            report["sampled_tokens"] = sampled_buffered

            report["prompt_boundaries"] = []
            for length in args.prompt_lengths or []:
                boundary_prompt = [2] + [100] * (length - 1)
                control = gen.generate(boundary_prompt, 4, stop_on_eos=False, buffer_tokens=False)
                candidate = gen.generate(boundary_prompt, 4, stop_on_eos=False)
                assert control == candidate, (length, control, candidate)
                report["prompt_boundaries"].append({"prompt_length": length, "tokens_equal": True})
            report["tokens"] = buffered
            report["all_layers"] = args.all_layers
        report["status"] = "pass"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if gen is not None:
            if args.recorder_only:
                if gen.output_trace_id is not None:
                    ttnn.release_trace(mesh, gen.output_trace_id)
            else:
                gen.teardown()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
