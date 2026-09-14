# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare output-delivery boundaries on the real, warmed TP4 model."""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric
from models.common.readiness_check.schema import load_reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--sampling-strategy", choices=("split", "argmax"), default="split")
    parser.add_argument("--pad-topk", action="store_true")
    parser.add_argument("--fabric-payload-bytes", type=int, default=8192)
    parser.add_argument("--qualitative-story", action="store_true")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--generate", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.generate < 2:
        parser.error("At least two generated tokens are required")
    root = Path("models/autoports/qwen_qwen3_8_27b")
    reference = load_reference(root / "readiness_aime24_chat.refpt").entries[0]
    raw = reference.prompt_tokens[0].tolist()
    # Fixed-length latency fixture only; correctness uses the intact reference elsewhere.
    prompt = (raw * ((args.length + len(raw) - 1) // len(raw)))[: args.length]
    torch.set_num_threads(8)
    configure_fabric(payload_bytes=args.fabric_payload_bytes)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        gen = build_generator(
            root,
            mesh,
            layer_indices=None if args.full else [0, 3],
            sampling_strategy=args.sampling_strategy,
        )
        gen.sampler.pad_to_power_of_2 = args.pad_topk
        report = dict(
            full=args.full,
            layers=gen.model.layer_indices,
            mesh=[1, 4],
            sampling_strategy=args.sampling_strategy,
            pad_topk=args.pad_topk,
            fabric_payload_bytes=args.fabric_payload_bytes,
            workload=dict(prompt_len=args.length, gen_len=args.generate, batch=1),
        )
        expected = gen.generate(prompt, args.generate, defer_token_readback=False, trace_prefill=False)
        warm = gen.generate(prompt, args.generate, defer_token_readback=False)
        assert warm == expected
        repeated = gen.generate(prompt, args.generate, defer_token_readback=False)
        assert repeated == expected
        report["eager_prefill_control_all_tokens_equal"] = True
        report["immediate_delivery"] = gen.last_perf
        # Reuse the same cache allocation and compiled model/sampling traces.
        # Reset and prefill are request-boundary work, outside steady decode timing.
        gen.generate(prompt, 1, defer_token_readback=False)
        ttft = gen.last_perf["ttft_s"]
        before = gen.counters.copy()
        begin = time.perf_counter()
        for _ in range(args.generate - 1):
            result = gen.decode_forward(page_table=gen.page_table, kv_cache=gen.cache, read_from_device=False)
            assert result is gen.tokens
        ttnn.synchronize_device(mesh)
        elapsed = time.perf_counter() - begin
        counters = dict(gen.counters - before)
        for name in (
            "token_refreshes",
            "position_refreshes",
            "rope_refreshes",
            "page_table_refreshes",
            "token_readbacks",
            "full_logits_readbacks",
            "trace_captures",
        ):
            assert not counters.get(name, 0), (name, counters)
        assert counters["model_replays"] == counters["sampling_replays"] == args.generate - 1
        final = int(gen._read_tokens()[0])
        assert final == expected[-1], (final, expected[-1])
        report["queued_token_out"] = dict(
            ttft_s=ttft,
            decode_s=elapsed,
            tokens_per_second=(args.generate - 1) / elapsed,
            steady_state_counters=counters,
            window_end_synchronizations=1,
            replay_blocking=False,
            final_token=final,
            final_token_matches_immediate=True,
            output_boundary="No per-token readback; final token checked after timing",
        )
        for _ in range(2):
            delivered = gen.generate(prompt, args.generate, defer_token_readback=True)
            assert delivered == expected
        perf = gen.last_perf
        steady = perf["steady_state_counters"]
        assert steady == dict(
            model_replays=args.generate - 1, sampling_replays=args.generate - 1, history_appends=args.generate - 1
        ), steady
        assert perf["delivery_counters"] == {"history_readbacks": 1}
        assert perf["counters"]["token_readbacks"] == 1
        report["deferred_delivery"] = dict(perf, all_tokens_match_immediate=True)
        if args.pad_topk:
            gen._release_traces()
            gen.sampler.pad_to_power_of_2 = False
            control = gen.generate(prompt, args.generate, defer_token_readback=False)
            assert control == expected
            report["unpadded_control_all_tokens_equal"] = True
            gen._release_traces()
            gen.sampler.pad_to_power_of_2 = True
        if args.sampling_strategy == "argmax":
            gen._release_traces()
            gen.sampler._allow_force_argmax_sampling = False
            control = gen.generate(prompt, args.generate, defer_token_readback=False)
            assert control == expected
            report["split_greedy_control_all_tokens_equal"] = True
            gen._release_traces()
            gen.sampler._allow_force_argmax_sampling = True
        if args.full:
            forced = reference.generated_tokens[0].tolist()
            for _ in range(2):
                gen.generate(raw, len(forced), next_input=lambda step, predicted: forced[step])
            report["teacher_forcing_with_token_delivery"] = gen.last_perf
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        if args.qualitative_story:
            assert args.full, "Story completion needs the complete layer stack"
            from models.autoports.qwen_qwen3_8_27b.tests.tt_qualitative import run

            report["story_completion_2048"] = str(
                run(
                    gen,
                    root,
                    "hf_qualitative_extended.json",
                    max_new_tokens=2048,
                    output_name="tt_qualitative_story_2048.json",
                    output_dir=args.output.parent,
                    prompt_ids=[2],
                )
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
