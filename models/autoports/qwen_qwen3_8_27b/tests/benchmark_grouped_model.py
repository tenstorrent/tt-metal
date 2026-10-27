# SPDX-License-Identifier: Apache-2.0
"""All64-layer token-out comparison of the opt-in grouped recurrence policy."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="/data/mvasiljev/weights/Qwen3.8-27B")
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--masked-recurrence", action="store_true")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--active-requests", type=int, help="Useful requests; remaining resident rows stay inactive")
    parser.add_argument(
        "--debug-layers", type=int, nargs="+", help="Reduced lifetime-debug fixture; never full-model evidence"
    )
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--outputs", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--qualitative-output-tokens", type=int, default=512)
    parser.add_argument(
        "--skip-qualitative", action="store_true", help="Timing-only run; qualitative acceptance is separate"
    )
    parser.add_argument("--state-checks", action="store_true")
    parser.add_argument("--continuation-prefix-length", type=int, default=4097)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    active_requests = args.active_requests if args.active_requests is not None else args.batch
    if not 1 <= active_requests <= args.batch:
        parser.error("active-requests must be within 1..batch")
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    delta_source = None
    os.environ["QWEN38_GROUPED_RECURRENCE"] = "1" if args.grouped else "0"
    os.environ["QWEN38_MASKED_RECURRENCE"] = "1" if args.masked_recurrence else "0"
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024**2)
    result = dict(
        scope=(
            "reduced lifetime-debug fixture; not full-model performance or SWE accuracy"
            if args.debug_layers
            else "all64-layer traced autoregressive token-out; fixed-work fixture, not SWE accuracy"
        ),
        layer_indices=args.debug_layers or list(range(64)),
        grouped=args.grouped,
        masked_recurrence=args.masked_recurrence,
        batch=args.batch,
        active_requests=active_requests,
        tested_cache_context=args.context,
        supported_model_context=262144,
        length=args.length,
        outputs=args.outputs,
        physical_prefill_chunk_size=args.prefill_chunk_size,
        continuation_prefix_length=args.continuation_prefix_length,
        allocation_tracking=os.getenv("TT_METAL_TRACE_ALLOC_TRACKING"),
        program_cache_skip=os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"),
        generated_delta_source=delta_source,
        records=[],
        status="running",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    def state_receipt():
        # Host inspection stays outside every measured loop. Preserve FP32 bits.
        return {
            f"{index}/{field}/{device}": hashlib.sha256(
                ttnn.to_torch(part).contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for index, state in enumerate(gen.kv_cache)
            for field in ("conv", "recurrent")
            if hasattr(state, field)
            for device, part in enumerate(ttnn.get_device_tensors(getattr(state, field)))
        }

    try:
        started = time.perf_counter()
        gen = build_generator(
            Path(__file__).resolve().parents[1],
            mesh,
            checkpoint=args.checkpoint,
            batch_size=args.batch,
            context_length=args.context,
            prepare=False,
            prefill_chunk_size=args.prefill_chunk_size,
            layer_indices=args.debug_layers,
        )
        result["load_seconds"] = time.perf_counter() - started
        result["precision"] = gen.model.precision_config
        print("FULL_MODEL_LOADED", result["load_seconds"], flush=True)
        save()
        started = time.perf_counter()
        gen.prepare_decode()
        result["prepare_seconds"] = time.perf_counter() - started
        print("FULL_MODEL_PREPARED", result["prepare_seconds"], flush=True)
        save()
        suite_path = Path(__file__).resolve().parents[1] / "doc/qualitative_prompts.json"
        suite = json.loads(suite_path.read_text())["prompts"]
        result["qualitative"] = dict(
            source=str(suite_path),
            source_sha256=hashlib.sha256(suite_path.read_bytes()).hexdigest(),
            prompt_mode="checkpoint native chat template; default thinking and reasoning effort",
            max_output_tokens=args.qualitative_output_tokens,
            skipped=args.skip_qualitative,
            outputs=[],
        )
        for item in [] if args.skip_qualitative else suite:
            messages = item["messages"]
            prompt = gen.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=False
            )
            qualitative = gen.generate([prompt[:] for _ in range(active_requests)], args.qualitative_output_tokens)
            result["qualitative"]["outputs"].append(
                dict(
                    id=item["id"],
                    messages=messages,
                    prompt_tokens=prompt,
                    tokens=qualitative,
                    completions=[gen.tokenizer.decode(row, skip_special_tokens=False) for row in qualitative],
                )
            )
            save()
            print("FULL_MODEL_QUALITATIVE", item["id"], flush=True)
        words = gen.tokenizer.encode(
            "The research team records each observation carefully and checks the evidence before drawing conclusions. ",
            add_special_tokens=False,
        )
        body_length = max(1, args.length - 48)
        for _ in range(8):
            body = gen.tokenizer.decode((words * ((body_length + len(words) - 1) // len(words)))[:body_length])
            prompt = gen.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "Continue with a detailed, several-paragraph account of this research process. Context: "
                        + body,
                    }
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            if len(prompt) == args.length:
                break
            body_length += args.length - len(prompt)
        if len(prompt) != args.length:
            raise ValueError("Could not construct the exact-length native chat fixture")
        prompts = [prompt[:] for _ in range(active_requests)]
        result["prompt_sha256"] = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        gen.generate(prompts, args.outputs, stop_on_eos=False)
        trace_ids = [str(t) for t in gen.decode_traces.values()] + [
            str(gen.sample_trace),
            str(gen.buffered_sample_trace),
        ]
        captures = gen.counters["captures"]
        for repeat in range(args.repeats):
            tokens = gen.generate(prompts, args.outputs, stop_on_eos=False)
            metrics = dict(gen.last_generation_metrics)
            assert all(len(row) == args.outputs for row in tokens)
            assert not metrics["host_sampling"]
            assert gen.counters["captures"] == captures
            result["records"].append(
                dict(
                    repeat=repeat,
                    metrics=metrics,
                    tokens_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    tokens=tokens,
                )
            )
            if args.state_checks:
                result["records"][-1]["state_sha256"] = state_receipt()
            save()
            print("FULL_MODEL_REPEAT", repeat, json.dumps(metrics), flush=True)
        mixed = [[100] * 33, [101] * 65, [102]]
        first = gen.generate(mixed, 4, stop_on_eos=False)
        mixed_state = state_receipt() if args.state_checks else None
        second = gen.generate(mixed, 4, stop_on_eos=False)
        assert first == second
        if args.state_checks:
            assert mixed_state == state_receipt(), "Reset/repeated ragged generation state changed"
            result["mixed_state_sha256"] = mixed_state
            gen.reset()
            gen.prefill_forward(
                torch.full((args.batch, args.continuation_prefix_length), 100, dtype=torch.int64),
                page_table=gen.page_table,
                kv_cache=gen.kv_cache,
                prompt_lens=[args.continuation_prefix_length] * args.batch,
                return_device=True,
            )
            result["continuation"] = []
            # Both groups independently advance7 tokens from a nonzero prefix
            # while the other group's bound slots are inactive.
            for parity in (0, 1):
                lengths = [7 if row % 2 == parity else 0 for row in range(args.batch)]
                logits = gen.prefill_continuation(
                    torch.full((args.batch, 7), 101 + parity, dtype=torch.int64),
                    page_table=gen.page_table,
                    kv_cache=gen.kv_cache,
                    prompt_lens=lengths,
                    start_position=args.continuation_prefix_length,
                )
                result["continuation"].append(
                    dict(
                        start_position=args.continuation_prefix_length,
                        prompt_lengths=lengths,
                        state_sha256=state_receipt(),
                        logits_sha256=[
                            hashlib.sha256(
                                ttnn.to_torch(part).contiguous().view(torch.uint8).numpy().tobytes()
                            ).hexdigest()
                            for part in ttnn.get_device_tensors(logits)
                        ],
                    )
                )
                del logits  # A materialized bridge return must not survive another replay.
            # Both groups are now aligned at prefix+7. Cross the public decode
            # bridge into an eager padded prefill tail without resetting state.
            start_position = args.continuation_prefix_length + 7
            logits = gen.prefill_continuation(
                torch.full((args.batch, 129), 103, dtype=torch.int64),
                page_table=gen.page_table,
                kv_cache=gen.kv_cache,
                prompt_lens=[129] * args.batch,
                start_position=start_position,
            )
            result["continuation"].append(
                dict(
                    start_position=start_position,
                    prompt_lengths=[129] * args.batch,
                    state_sha256=state_receipt(),
                    logits_sha256=[
                        hashlib.sha256(ttnn.to_torch(part).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                        for part in ttnn.get_device_tensors(logits)
                    ],
                )
            )
            del logits
        assert trace_ids == [str(t) for t in gen.decode_traces.values()] + [
            str(gen.sample_trace),
            str(gen.buffered_sample_trace),
        ]
        assert gen.counters["captures"] == captures
        result.update(
            status="pass",
            mixed_outputs=first,
            trace_ids=trace_ids,
            captures=captures,
            mixed_repeat_exact=True,
            counters=gen.counters,
        )
    except BaseException as exc:
        result.update(status="fail", error=repr(exc))
        raise
    finally:
        root = Path(__file__).resolve().parents[1]
        sources = [
            Path(__file__),
            root / "tt/grouped_recurrence.py",
            root / "tt/precision.py",
            root / "tt/generator.py",
            root / "tt/multichip_decoder.py",
            *root.glob("tt/kernels/*grouped_recurrence.cpp"),
        ]
        result["source_sha256"] = {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        }
        save()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
