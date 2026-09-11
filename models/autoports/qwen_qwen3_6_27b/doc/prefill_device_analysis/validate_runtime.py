# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""One weight load: full-stack prefill timing, HF checks and traced generation."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator
from models.common.readiness_check.run_prefill_check import _run_one_entry_prefill
from models.common.readiness_check.run_teacher_forcing import _run_one_entry
from models.common.readiness_check.schema import load_reference
from models.common.readiness_check.teacher_forcing import TokenAccuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--silu-ab", action="store_true")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--program-block-limit", type=int)
    parser.add_argument(
        "--pad-prefill", action="store_true", help="Pad physical input to tiles; preserve logical lengths"
    )
    parser.add_argument("--sequences", type=int, nargs="+", default=[128, 4096])
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.program_block_limit:
        from models.autoports.qwen_qwen3_6_27b.tt import multichip_decoder as md

        original_program = md._prefill_program

        def block_program(**kwargs):
            kwargs["in0_block_w_limit"] = args.program_block_limit
            return original_program(**kwargs)

        md._prefill_program = block_program
    if args.silu_ab:
        # Restore the historical defect explicitly so this A/B remains meaningful
        # after the repair becomes the default. The second variant removes it.
        import inspect
        import textwrap

        from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder

        original = MultichipDecoder._tp_linear
        source = textwrap.dedent(inspect.getsource(original))
        source = source.replace(
            'if not decode and "program_config" not in kwargs and fused_activation == ttnn.UnaryOpType.SILU:',
            "if not decode and fused_activation == ttnn.UnaryOpType.SILU:",
        )
        # A real source file lets the second variant's inspect-based patch read it.
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as saved:
            saved.write(source)
        namespace = dict(original.__globals__)
        exec(compile(source, saved.name, "exec"), namespace)
        MultichipDecoder._tp_linear = namespace["_tp_linear"]
    reference = load_reference(args.reference)
    if "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0" not in reference.hf_model_id:
        raise ValueError("Validation requires the pinned Qwen3.8 reference")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=300_000_000)
    generator = None
    result = {
        "reference": str(args.reference),
        "hf_model_id": reference.hf_model_id,
        "invocation": sys.argv,
        "program_block_limit_override": args.program_block_limit,
        "pad_prefill": args.pad_prefill,
        "variants": {},
    }
    try:
        generator = build_generator(Path(__file__).parents[2], mesh, batch=1, max_context=4608)
        result["runtime"] = generator.model.precision_summary()
        if args.pad_prefill:
            unpadded_forward = generator.prefill_forward
            result["physical_prefill_shapes"] = []

            def padded_forward(tokens, **kwargs):
                logical_width = tokens.shape[1]
                physical_width = (logical_width + 31) // 32 * 32
                tokens = torch.nn.functional.pad(tokens, (0, physical_width - logical_width))
                result["physical_prefill_shapes"].append(
                    {
                        "supplied_width": logical_width,
                        "physical_width": physical_width,
                        "prompt_lens": list(kwargs["prompt_lens"]),
                    }
                )
                output = unpadded_forward(tokens, **kwargs)
                if kwargs.get("return_all_logits", False):
                    output = output[:, :logical_width]
                return output

            generator.prefill_forward = padded_forward
        result["native_branch_calls"] = {"flat": 0, "rank4": 0}
        from models.autoports.qwen_qwen3_6_27b.tt.prefill_recurrence import NativeGatedDeltaRule

        for method, label in (("flat_forward", "flat"), ("__call__", "rank4")):
            original_method = getattr(NativeGatedDeltaRule, method)

            def counted(self, *values, _method=original_method, _label=label, **kwargs):
                result["native_branch_calls"][_label] += 1
                return _method(self, *values, **kwargs)

            setattr(NativeGatedDeltaRule, method, counted)
        prompt = "Explain a stable merge sort and its time and space complexity. "
        rendered = generator.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt * 500}], tokenize=False, add_generation_prompt=True
        )
        ids = generator.tokenizer.encode(rendered, add_special_tokens=False)
        for variant in ["current_silu", "single_silu"] if args.silu_ab else ["current_silu"]:
            if variant == "single_silu":
                from probe import configure

                configure("single_silu")
            metrics = {}
            result["variants"][variant] = metrics
            for sequence in args.sequences:
                tokens = torch.tensor([ids[:sequence]])
                samples = []
                for iteration in range(4):
                    generator.reset()
                    ttnn.synchronize_device(mesh)
                    started = time.perf_counter()
                    generator.prefill_forward(
                        tokens, page_table=generator._page_table, kv_cache=generator.kv_cache, prompt_lens=[sequence]
                    )
                    ttnn.synchronize_device(mesh)
                    elapsed = (time.perf_counter() - started) * 1000
                    print(f"FULL_PREFILL {variant} S{sequence} iteration={iteration} ms={elapsed:.3f}", flush=True)
                    samples.append(elapsed)
                metrics[f"prefill_s{sequence}"] = {
                    "cold_ms": samples[0],
                    "warmed_ms": samples[1:],
                    "median_ms": statistics.median(samples[1:]),
                }
            if args.bench_only:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                continue
            generator.reset()
            metrics["prefill_accuracy"] = _run_one_entry_prefill(
                generator=generator, entry=reference.entries[0], reference=reference
            )
            print("PREFILL_ACCURACY", json.dumps(metrics["prefill_accuracy"]), flush=True)
            generator.reset()
            metrics["teacher_forcing"] = _run_one_entry(
                generator=generator, acc=TokenAccuracy(args.reference), entry_idx=0
            )
            print("TEACHER_ACCURACY", json.dumps(metrics["teacher_forcing"]), flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result["validation_status"] = "accuracy_passed_generation_pending"
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            generator.reset()
            prompt_ids = reference.entries[0].prompt_tokens[0].tolist()
            output = generator.generate(prompt_ids, 100)
            result["validation_status"] = "generation_completed"
            metrics["generation"] = {
                "prompt_ids": prompt_ids,
                "tt_token_ids": output,
                "tt_text": generator.tokenizer.decode(output, skip_special_tokens=False),
                "hf_text": generator.tokenizer.decode(
                    reference.entries[0].generated_tokens[0], skip_special_tokens=False
                ),
                "trace_counters": dict(generator.trace_counters),
            }
            metrics["accuracy_pass"] = all(
                metrics[name]["top5"] >= 0.98 and metrics[name]["top100"] == 1.0
                for name in ("prefill_accuracy", "teacher_forcing")
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("FULL_VALIDATION", json.dumps(metrics), flush=True)
    finally:
        if generator is not None:
            generator.teardown()
        ttnn.close_mesh_device(mesh)
    if not args.bench_only and not all(item["accuracy_pass"] for item in result["variants"].values()):
        raise AssertionError("Full-model accuracy gate failed; see saved result")


if __name__ == "__main__":
    main()
