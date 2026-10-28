# SPDX-License-Identifier: Apache-2.0
"""Clean paired native B32 MLP timing, separate from allocation diagnostics.

Native262144 tables, legal aggregate cache pool,512 physical prefill chunks.
The standard4096-in/128-out row is throughput mechanism evidence, not scored
quality or32 simultaneous native-length histories. Requires serialized TT lease.
"""
import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reduced", action="store_true")
    parser.add_argument("--qualitative", action="store_true", help="Run shared native-chat suite after timed repeats")
    parser.add_argument("--hf-qualitative-controls", type=Path, help="Existing prompt-correct HF512-token controls")
    args = parser.parse_args()
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_8_27b.tests.full_model_evidence import runtime_build
    from models.autoports.qwen_qwen3_8_27b.tests.native_fixture_cleanup import finalize_run
    from models.autoports.qwen_qwen3_8_27b.tests.native_fixture_metadata import capture_linear_calls
    from models.autoports.qwen_qwen3_8_27b.tt.generator import Generator
    from models.autoports.qwen_qwen3_8_27b.tt.model import Model

    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "0":
        raise RuntimeError("Clean timing requires trace tracking explicitly disabled")
    if any(
        os.environ.get(key) not in (None, "0")
        for key in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER", "TT_METAL_PROFILER_TRACE_TRACKING")
    ):
        raise RuntimeError("Profiler/watcher must be disabled for clean timing")
    if args.qualitative and args.hf_qualitative_controls is None:
        raise ValueError("Qualitative run requires retained HF controls")
    if args.reduced and args.qualitative:
        raise ValueError("Qualitative checks require all64layers")
    if not 1 <= args.batch <= 32 or args.repeats < 1:
        raise ValueError("Invalid native batch/repeat count")
    pool_context = max(1024, math.ceil(262144 / (args.batch * 256)) * 256)
    if args.input_tokens + args.output_tokens > pool_context:
        raise ValueError("Fixed request exceeds bounded physical per-row cache")
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    result = dict(
        status="running",
        throw_exception_on_fallback=bool(ttnn.CONFIG.throw_exception_on_fallback),
        shared=os.environ.get("QWEN38_SHARED_MLP_WEIGHTS", "0"),
        shared_mlp_readers=int(os.environ.get("QWEN38_SHARED_MLP_READERS", "1")),
        shared_mlp_cores=int(os.environ.get("QWEN38_SHARED_MLP_CORES", "80")),
        batch=args.batch,
        layers=[0, 3] if args.reduced else "all64",
        native_context=262144,
        physical_pool_tokens_per_row=pool_context,
        physical_prefill_chunk=512,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
        prompt_mode="HF checkpoint native chat template; fixed-length repeated-text throughput fixture",
        generation="Greedy on-device, fixed output length, EOS disabled for timing; not task quality",
        runtime=runtime_build(),
        environment={
            key: os.environ.get(key)
            for key in (
                "TT_METAL_HOME",
                "TT_METAL_RUNTIME_ROOT",
                "TT_METAL_CACHE",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_DEVICE_PROFILER",
                "TT_METAL_WATCHER",
                "QWEN38_PRECISION_CONFIG",
                "TT_METAL_PROFILER_TRACE_TRACKING",
                "QWEN38_GROUPED_RECURRENCE",
                "QWEN38_MASKED_RECURRENCE",
                "QWEN38_FLAT_PREFILL",
            )
        },
        records=[],
    )
    root = Path(__file__).resolve().parents[1]
    result["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            root / "tests/native_fixture_cleanup.py",
            root / "tests/native_fixture_metadata.py",
            root / "tt/selected_precision_config.json",
            root / "tests/full_model_evidence.py",
            *sorted((root / "tt").glob("*.py")),
        ]
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    gen = None
    try:
        model = Model(args.checkpoint, mesh, layer_indices=[0, 3] if args.reduced else None)
        result["effective_precision"] = model.precision_config
        result["layer_types"] = {str(layer.layer_idx): layer.layer_type for layer in model.layers}
        cache = model.allocate_cache(batch_size=args.batch, context_length=pool_context)
        pages = pool_context // 32
        table = torch.zeros(args.batch, 262144 // 32, dtype=torch.int32)
        table[:, :pages] = torch.arange(args.batch * pages, dtype=torch.int32).reshape(args.batch, pages)
        gen = Generator(
            model,
            args.checkpoint,
            batch_size=args.batch,
            context_length=262144,
            kv_cache=cache,
            page_table=table,
            prefill_chunk_size=512,
        )
        with capture_linear_calls(result, ttnn, model):
            gen.prepare_decode()
        gen.set_sampling_params(top_k=1)
        words = gen.tokenizer.encode(
            "The research team records each observation carefully and checks the evidence before drawing conclusions. ",
            add_special_tokens=False,
        )
        body_length = max(1, args.input_tokens - 48)
        for _ in range(10):
            body = gen.tokenizer.decode((words * ((body_length + len(words) - 1) // len(words)))[:body_length])
            prompt = gen.tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "Continue with a detailed account of this research process. Context: " + body,
                    }
                ],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
            )
            if len(prompt) == args.input_tokens:
                break
            body_length += args.input_tokens - len(prompt)
        else:
            raise RuntimeError("Cannot render exact native-chat input length")
        result["prompt_sha256"] = hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest()
        prompts = [list(prompt) for _ in range(args.batch)]
        warm_tokens = gen.generate(prompts, args.output_tokens, stop_on_eos=False, buffered_output=True)
        trace_ids = {str(k): str(v) for k, v in gen.decode_traces.items()}
        captures = gen.counters["captures"]
        print("CLEAN_TIMING_BEGIN", flush=True)
        for repeat in range(args.repeats):
            tokens = gen.generate(prompts, args.output_tokens, stop_on_eos=False, buffered_output=True)
            metrics = dict(gen.last_generation_metrics)
            assert tokens == warm_tokens, "Warmed repetition changed greedy tokens"
            assert trace_ids == {str(k): str(v) for k, v in gen.decode_traces.items()}
            assert captures == gen.counters["captures"]
            steady = metrics["steady_state_counters"]
            assert all(
                steady[key] == 0
                for key in ("token_refreshes", "position_refreshes", "page_table_refreshes", "captures")
            )
            assert not metrics["host_sampling"] and not metrics["blocking_replay"]
            result["records"].append(dict(repeat=repeat, metrics=metrics))
        print("CLEAN_TIMING_END", flush=True)
        if args.qualitative:
            suite_path = root / "doc/qualitative_prompts.json"
            suite = json.loads(suite_path.read_text())
            result["qualitative_prompt_format"] = dict(
                hf_model="Qwen/Qwen3.8-27B",
                revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
                tokenizer=type(gen.tokenizer).__name__,
                chat_template=bool(gen.tokenizer.chat_template),
                prompt_mode="chat",
                endpoint="native Generator with tokenizer.apply_chat_template",
                do_sample=False,
                max_new_tokens=512,
                stop_on_eos=True,
                suite_sha256=hashlib.sha256(suite_path.read_bytes()).hexdigest(),
                scope="Post-timing shared suite; output quality and HF/control comparison reviewed separately",
            )
            result["qualitative"] = []
            for item in suite["prompts"]:
                ids = gen.tokenizer.apply_chat_template(
                    item["messages"], tokenize=True, add_generation_prompt=True, return_dict=False
                )
                hf_path = args.hf_qualitative_controls / (item["id"] + "_hf.json")
                hf = json.loads(hf_path.read_text())
                if (
                    hf["prompt_tokens"] != ids
                    or hf["messages"] != item["messages"]
                    or hf["revision"] != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
                    or hf["max_new_tokens"] != 512
                    or hf["do_sample"] is not False
                ):
                    raise ValueError("HF qualitative control prompt/revision/policy mismatch")
                generated = gen.generate(ids, 512, stop_on_eos=True, buffered_output=True)
                result["qualitative"].append(
                    dict(
                        id=item["id"],
                        messages=item["messages"],
                        prompt_token_ids=ids,
                        generated_token_ids=generated,
                        stopped_on_eos=bool(generated and generated[-1] in gen.eos_token_ids),
                        completion=gen.tokenizer.decode(generated, skip_special_tokens=False),
                        hf_control=dict(
                            path=str(hf_path),
                            sha256=hashlib.sha256(hf_path.read_bytes()).hexdigest(),
                            generated_token_ids=hf["generated_tokens"],
                            completion=hf["completion"],
                            stopped_on_eos=hf["stopped_on_eos"],
                        ),
                    )
                )
                assert captures == gen.counters["captures"], "Qualitative request recaptured traces"
        result.update(
            status="pass",
            tokens=warm_tokens,
            trace_ids=trace_ids,
            captures=captures,
            median_ttft_ms=statistics.median(r["metrics"]["ttft_ms"] for r in result["records"]),
            median_decode_tps_per_user=statistics.median(
                r["metrics"]["decode_tps_per_user"] for r in result["records"]
            ),
        )
    except BaseException as error:
        result.update(status="fail", error_type=type(error).__name__, error=str(error))
        raise
    finally:

        def release():
            if gen is not None:
                for trace in list(gen.decode_traces.values()) + [gen.sample_trace, gen.buffered_sample_trace]:
                    if trace is not None:
                        ttnn.release_trace(mesh, trace)

        finalize_run(result, target, release, lambda: ttnn.close_mesh_device(mesh))


if __name__ == "__main__":
    main()
