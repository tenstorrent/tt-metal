# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Matching HF sampled or same-code TT controls for serving's six prompts."""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.6-35B-A3B"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hf", "tt"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thinking", choices=("default", "on", "off"), default="default")
    parser.add_argument("--hf-modes", choices=("sampled", "all"), default="sampled")
    parser.add_argument(
        "--recall-fixtures", type=Path, help="Append native chat recall controls from preserved raw fixtures"
    )
    parser.add_argument("--recall-max-tokens", type=int, default=1024)
    parser.add_argument("--recall-thinking", choices=("default", "off"), default="default")
    parser.add_argument(
        "--served-geometry",
        type=Path,
        help="TT control using current adapter construction and cache specs from this real server log",
    )
    parser.add_argument(
        "--diagnostic-bucket",
        type=int,
        choices=(2, 32),
        help="TT-only forced decode bucket; defaults to the original q01 greedy256 discriminator",
    )
    parser.add_argument(
        "--diagnostic-prefix",
        action="store_true",
        help="With --diagnostic-bucket, run q01 greedy256, q01 sampled256, then q02 greedy256",
    )
    parser.add_argument(
        "--diagnostic-capture-buckets",
        choices=("2,32", "2"),
        default="2,32",
        help="Diagnostic-only retained model graphs; capacity and sampler remain 32 (default: 2,32)",
    )
    args = parser.parse_args(argv)
    if args.served_geometry and (args.backend != "tt" or args.diagnostic_bucket is not None):
        parser.error("Served geometry requires the complete TT suite, without forced diagnostic buckets")
    if args.recall_fixtures and not args.served_geometry:
        parser.error("Recall controls require the current served-geometry TT control")
    if args.diagnostic_bucket is not None and args.backend != "tt":
        parser.error("Bucket discrimination requires the TT backend")
    if (args.diagnostic_prefix or args.diagnostic_capture_buckets != "2,32") and args.diagnostic_bucket is None:
        parser.error("Diagnostic prefix/capture selection requires --diagnostic-bucket")
    if args.diagnostic_bucket is not None and args.diagnostic_bucket not in tuple(
        map(int, args.diagnostic_capture_buckets.split(","))
    ):
        parser.error("The forced bucket must be included in --diagnostic-capture-buckets")
    return args


def request_plan(prompts, modes, *, diagnostic_bucket=None, diagnostic_prefix=False):
    if diagnostic_prefix:
        return [(prompts[0], "greedy"), (prompts[0], "sampled"), (prompts[1], "greedy")]
    if diagnostic_bucket is not None:
        return [(prompts[0], "greedy")]
    return [(prompt, mode) for prompt in prompts for mode in modes]


def main():
    args = parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh control artifact")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    suite = Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "runtime/readiness_check/vllm_prompts.txt"
    prompts = []
    template_kwargs = {} if args.thinking == "default" else {"enable_thinking": args.thinking == "on"}
    for index, text in enumerate(filter(None, (x.strip() for x in suite.read_text().split("\n\n"))), 1):
        messages = [{"role": "user", "content": text}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        prompts.append(
            dict(
                prompt_id=f"q{index:02}",
                messages=messages,
                rendered_prompt=rendered,
                prompt_tokens=tokenizer.encode(rendered, add_special_tokens=False),
            )
        )
    result = dict(
        status="RUNNING",
        backend=args.backend,
        model=MODEL,
        revision=REVISION,
        tokenizer_class=type(tokenizer).__name__,
        chat_template_present=bool(tokenizer.chat_template),
        prompt_mode="chat",
        thinking=args.thinking,
        chat_template_kwargs=template_kwargs,
        max_new_tokens=256,
        seed=0,
        generation=dict(
            greedy=dict(temperature=0.0, top_k=20, top_p=0.95), sampled=dict(temperature=0.7, top_k=20, top_p=0.9)
        ),
        prompts=prompts,
        outputs=[],
        quality_verdict="pending reading",
        diagnostic_bucket=args.diagnostic_bucket,
        diagnostic_prefix=args.diagnostic_prefix,
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    mesh = generator = None
    current_request = None

    def snapshot_generator():
        if generator is not None:
            result["trace_ids"] = {key: str(value) for key, value in generator.traces.items()}
            result["trace_replays"] = dict(generator.trace_replays)
            result["generator_counters"] = dict(generator.counters)

    try:
        if args.backend == "hf":
            torch.set_num_interop_threads(1)
            model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True).eval()
            modes = ("greedy", "sampled") if args.hf_modes == "all" else ("sampled",)
            result["resolved_model_dtype"] = str(model.dtype)
        else:
            import ttnn
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
            from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel

            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=805306368)
            if args.served_geometry:
                from models.autoports.qwen_qwen3_6_35b_a3b.doc.vllm_integration.sampling_debug.run_matched_b2_overhead import (
                    serving_geometry,
                )
                from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator_vllm import QwenForCausalLM

                initial, specs = serving_geometry(args.served_geometry)
                adapter = QwenForCausalLM.initialize_vllm_model(
                    SimpleNamespace(model_type="qwen3_5_moe", _name_or_path=None), mesh, 32, 262144
                )
                cache = adapter.allocate_kv_cache_per_layer(
                    [(tuple(shape), torch.bfloat16, index) for shape, _, index in specs]
                )
                model, generator = adapter.model, adapter.generator
                assert model.precision_config == initial["precision"]
                assert generator.cache is cache and not generator.host_sampling
                assert [tuple(group[0]["k"].shape) for group in cache if "k" in group[0]] == [
                    tuple(shape) for shape, _, _ in specs
                ]
                adapter.warmup_model_prefill(kv_cache=cache, can_sample_on_device=True, enable_trace=False)
                adapter.warmup_model_decode(
                    kv_cache=cache,
                    max_batch_size=32,
                    num_blocks=specs[0][0][0],
                    can_sample_on_device=True,
                    enable_trace=True,
                )
                capture_buckets = generator.decode_buckets
                root = Path(__file__).resolve().parents[1]
                sources = [*sorted((root / "tt").glob("*.py")), Path(__file__)]
                native = root.parents[2] / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill"
                for family in ("moe_fused_swiglu", "post_combine_reduce"):
                    sources.extend(path for path in (native / family).rglob("*") if path.suffix in (".cpp", ".hpp"))
                result.update(
                    served_geometry=str(args.served_geometry),
                    served_cache_specs=specs,
                    source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                    control_scope="Current adapter-constructed full model and served cache; direct generator.generate",
                )
            else:
                model = QwenModel(mesh, max_seq_len=1024)
                generator = QwenGenerator(model, max_batch_size=32, decode_buckets=(2, 32))
                capture_buckets = tuple(map(int, args.diagnostic_capture_buckets.split(",")))
            if capture_buckets != (2, 32):
                # Diagnostic only: retain the B2 graph at unchanged capacity32.
                # Production construction still requires the capacity bucket;
                # this harness deliberately isolates its retention before capture.
                generator.decode_buckets = capture_buckets
            result.update(
                layers=list(model.layer_indices),
                precision=model.precision_config,
                capacity=32,
                context=model.max_seq_len,
                decode_buckets=list(capture_buckets),
                host_sampling=False,
            )
            generator.prepare()
            modes = ("greedy", "sampled")
            if args.diagnostic_bucket is not None:
                # Selection changes only after the requested graphs are captured;
                # the canonical sampling/token-feedback path remains unchanged.
                generator.decode_buckets = (args.diagnostic_bucket,)
                result["scope"] = (
                    "q01 greedy256 -> q01 sampled256 -> q02 greedy256 bucket discriminator"
                    if args.diagnostic_prefix
                    else "q01 greedy256 bucket discriminator"
                ) + ", not complete qualitative acceptance"
                snapshot_generator()
                result["captured_trace_ids"] = dict(result["trace_ids"])
                result["partial_tokens"] = []
                result["request_progress"] = []
                original_read = generator._read_tokens

                def observed_read(slots):
                    values = original_read(slots)
                    result["partial_tokens"].extend(values)
                    current_request["partial_tokens"].extend(values)
                    if len(current_request["partial_tokens"]) % 32 == 0:
                        snapshot_generator()
                        current_request["trace_replays"] = dict(generator.trace_replays)
                        current_request["generator_counters"] = dict(generator.counters)
                        save()
                        print(
                            "CONTROL_PROGRESS",
                            current_request["prompt_id"],
                            current_request["mode"],
                            len(current_request["partial_tokens"]),
                            flush=True,
                        )
                    return values

                generator._read_tokens = observed_read
        plan = request_plan(
            prompts, modes, diagnostic_bucket=args.diagnostic_bucket, diagnostic_prefix=args.diagnostic_prefix
        )
        result["request_plan"] = [dict(prompt_id=prompt["prompt_id"], mode=mode) for prompt, mode in plan]
        save()
        for prompt, mode in plan:
            if args.diagnostic_bucket is not None:
                snapshot_generator()
                current_request = dict(
                    prompt_id=prompt["prompt_id"],
                    mode=mode,
                    status="RUNNING",
                    partial_tokens=[],
                    before_trace_replays=dict(generator.trace_replays),
                    before_generator_counters=dict(generator.counters),
                )
                result["request_progress"].append(current_request)
                save()
            torch.manual_seed(0)
            before = time.monotonic()
            params = result["generation"][mode]
            if args.backend == "hf":
                ids = torch.tensor([prompt["prompt_tokens"]], dtype=torch.long)
                with torch.no_grad():
                    sequence = model.generate(
                        input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        max_new_tokens=256,
                        do_sample=mode == "sampled",
                        use_cache=True,
                        **(params if mode == "sampled" else {}),
                    )
                tokens = sequence[0, ids.shape[1] :].tolist()
            else:
                tokens = generator.generate(prompt["prompt_tokens"], 256, seed=0, stop_on_eos=True, **params)
            result["outputs"].append(
                dict(
                    prompt_id=prompt["prompt_id"],
                    mode=mode,
                    generated_tokens=tokens,
                    completion=tokenizer.decode(tokens, skip_special_tokens=False),
                    elapsed_s=time.monotonic() - before,
                )
            )
            if current_request is not None:
                snapshot_generator()
                current_request.update(
                    status="PASS",
                    trace_replays=dict(generator.trace_replays),
                    generator_counters=dict(generator.counters),
                )
            save()
            print(prompt["prompt_id"], mode, len(tokens), flush=True)
        if args.recall_fixtures:
            fixture_record = json.loads(args.recall_fixtures.read_text())
            assert fixture_record["model"] == MODEL and fixture_record["revision"] == REVISION
            fixture_ids = [case["id"] for case in fixture_record["fixtures"]]
            assert len(fixture_ids) == len(set(fixture_ids)) == 14, "Require all fourteen unique canonical fixtures"
            eos = json.loads((model.checkpoint.path / "generation_config.json").read_text())["eos_token_id"]
            eos = eos if isinstance(eos, list) else [eos]
            result["recall_control"] = dict(
                fixtures=str(args.recall_fixtures),
                fixture_sha256=hashlib.sha256(args.recall_fixtures.read_bytes()).hexdigest(),
                max_tokens=args.recall_max_tokens,
                prompt_mode="chat; reasoning_effort low as in supported plugin recall mode",
                thinking=args.recall_thinking,
                generation=dict(temperature=0.0, top_p=1.0, top_k=20),
                eos_token_ids=eos,
                outputs=[],
                scope="Native recall and answer-length control, separate from the unchanged shared256 suite",
            )
            for case in fixture_record["fixtures"]:
                messages = [dict(role="user", content=case["prompt"])]
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    reasoning_effort="low",
                    **({} if args.recall_thinking == "default" else dict(enable_thinking=False)),
                )
                prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
                before = time.monotonic()
                tokens = generator.generate(
                    prompt_ids,
                    args.recall_max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=20,
                    seed=0,
                    stop_on_eos=True,
                    stop_token_ids=eos,
                )
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                final = (
                    text
                    if args.recall_thinking == "off"
                    else text.split("</think>", 1)[-1]
                    if "</think>" in text
                    else ""
                )
                normalized = final.translate(str.maketrans({"\N{HYPHEN}": "-", "\N{NON-BREAKING HYPHEN}": "-"}))
                recalled = re.search(r"(?<![\w-])" + re.escape(case["passphrase"]) + r"(?![\w-])", normalized.lower())
                result["recall_control"]["outputs"].append(
                    dict(
                        id=case["id"],
                        messages=messages,
                        prompt_tokens=prompt_ids,
                        generated_tokens=tokens,
                        completion=text,
                        completed=bool(tokens) and tokens[-1] in eos,
                        recalled=bool(recalled),
                        elapsed_s=time.monotonic() - before,
                    )
                )
                save()
                print("recall", case["id"], len(tokens), bool(recalled), flush=True)
            result["recall_control"]["all_recalled_and_completed"] = all(
                row["recalled"] and row["completed"] for row in result["recall_control"]["outputs"]
            )
            assert len(result["recall_control"]["outputs"]) == 14
            assert result["recall_control"]["all_recalled_and_completed"], "Native recall control failed"
        result["status"] = "PASS"
        if generator is not None:
            snapshot_generator()
    except BaseException as error:
        result.update(status="FAIL", error=repr(error))
        if current_request is not None:
            current_request.update(status="FAIL", error=repr(error))
        snapshot_generator()
        save()
        raise
    finally:
        try:
            if generator is not None:
                generator.close()
                result["shutdown_generator_counters"] = dict(generator.counters)
        except BaseException as error:
            result.update(status="FAIL", generator_cleanup_error=repr(error))
            raise
        finally:
            try:
                if mesh is not None:
                    ttnn.close_mesh_device(mesh)
            except BaseException as error:
                result.update(status="FAIL", mesh_cleanup_error=repr(error))
                raise
            finally:
                save()


if __name__ == "__main__":
    main()
