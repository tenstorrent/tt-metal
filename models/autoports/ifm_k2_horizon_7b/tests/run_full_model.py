"""All-layer readiness gates, sharing one model load across official runners."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_prefill_check, run_teacher_forcing

import ttnn

from ..tt.generator import K2Generator

DOC = Path("models/autoports/ifm_k2_horizon_7b/doc/full_model")
MODEL_DIR = DOC.parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gate", choices=["prefill", "teacher", "both"], default="both")
    p.add_argument("--output", default=str(DOC / "accuracy.json"))
    p.add_argument("--head-dtype")
    p.add_argument("--head-fidelity")
    p.add_argument("--extended-quality", action="store_true")
    p.add_argument("--quality-output-name", default="qualitative_extended.json")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--benchmark-output", default=str(DOC / "performance_final.json"))
    p.add_argument("--batch-output")
    p.add_argument("--continuation-output")
    p.add_argument("--generator-controls-output")
    p.add_argument("--transition-output")
    p.add_argument("--autoregressive", action="store_true")
    p.add_argument("--autoregressive-output-dir", default="autoregressive")
    args = p.parse_args()
    torch.set_num_threads(16)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    result = {
        "head_dtype": args.head_dtype,
        "head_fidelity": args.head_fidelity,
        "head_split_size": 8192,
        "head_workers": 1,
        "head_k": 4,
        "layers": 36,
    }
    try:
        gen = K2Generator(mesh, head_dtype=args.head_dtype, head_fidelity=args.head_fidelity)
        result.update(
            head_dtype=gen.model.head_dtype,
            head_fidelity=gen.model.head_fidelity,
            precision_config=gen.model.precision_config,
        )
        result["runtime_policies"] = [asdict(layer.policy) for layer in gen.model.layers]
        result["weight_allocations"] = [layer.weight_allocations for layer in gen.model.layers]
        result["source_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((MODEL_DIR / "tt").glob("*.py"))
        }
        native_paths = {
            Path(line.split()[-1])
            for line in Path("/proc/self/maps").read_text().splitlines()
            if line.split()[-1].endswith(("/_ttnncpp.so", "/libtt_metal.so"))
        }
        result["runtime_native_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(native_paths)
        }
        if args.benchmark:
            # Measure the standalone256-token cache regime before validation
            # grows cache/history allocations, matching the pre-edit baseline.
            from .benchmark_full_model import run

            run(gen, args.benchmark_output)
        for name, runner, method in [
            ("prefill", run_prefill_check, "run_prefill_check"),
            ("teacher", run_teacher_forcing, "run_teacher_forcing"),
        ]:
            if args.gate not in ("both", name):
                continue
            # Only reuse the factory result; actual packaged runner logic and
            # real generator paths remain unchanged.
            with patch.object(runner, "_import_build_generator", return_value=lambda **kw: gen):
                result[name] = getattr(runner, method)(
                    model_dir=MODEL_DIR, reference_path=MODEL_DIR / "readiness_aime24_chat.refpt", mesh_device=mesh
                )
            result[name + "_generator_perf"] = gen.last_perf
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
            print("GATE", name, json.dumps(result[name]), flush=True)
            for score in result[name]:
                assert score["total"] == 100 and score["top5"] >= 0.98 and score["top100"] == 1.0
        if args.extended_quality:
            from .run_qualitative_extended import run

            run(gen, output_name=args.quality_output_name)
        if args.batch_output:
            from .probe_full_batch import run

            run(gen, args.batch_output)
        if args.continuation_output:
            from .probe_full_continuation import compare

            gen._ensure_owned_cache(1, 4356)
            prompt = (
                gen.tokenizer.encode("A careful scientist checks the evidence before drawing a conclusion. ") * 30
            )[:257]
            tokens = torch.tensor([prompt])
            gen.reset()
            whole = gen.prefill_forward(
                tokens, page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[257], sampling_mode="host"
            )
            checks = []
            reference = json.loads((DOC / "stress_hf_topk.json").read_text())["records"][0]
            assert reference["prompt_token_ids"] == prompt
            hf_ranks = {row["id"]: row["rank"] for row in reference["hf_top100"]}
            for cut in [31, 32, 65]:
                gen.reset()
                gen.prefill_forward(
                    tokens[:, :cut], page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[cut]
                )
                split = gen.prefill_forward(
                    tokens[:, cut:],
                    page_table=gen.page_table,
                    kv_cache=gen.kv_cache,
                    prompt_lens=[257 - cut],
                    start_pos=[cut],
                    sampling_mode="host",
                )
                metrics = compare(whole, split)
                checks.append(
                    {
                        "cut": cut,
                        "metrics": metrics,
                        "whole_token": int(whole.flatten().argmax()),
                        "split_token": int(split.flatten().argmax()),
                        "passes_pcc_0_995": metrics["pcc"] >= 0.995,
                        "hf_top1": reference["hf_top1"]["id"],
                        "whole_hf_rank": hf_ranks.get(int(whole.flatten().argmax())),
                        "split_hf_rank": hf_ranks.get(int(split.flatten().argmax())),
                    }
                )
            Path(args.continuation_output).write_text(
                json.dumps({"prompt_token_ids": prompt, "checks": checks}, indent=2) + "\n"
            )
            assert all(row["passes_pcc_0_995"] for row in checks)
            assert all(row["whole_hf_rank"] == row["split_hf_rank"] == 1 for row in checks)
        if args.autoregressive:
            from .run_full_quality import run

            run(gen, shared_suite=False, output_dir=args.autoregressive_output_dir)
        if args.generator_controls_output:
            from .probe_optimized_generator import run

            # That probe explicitly verifies history growth128→256. Prior
            # quality/benchmark requests may already own a larger history.
            gen._release_traces()
            gen.token_history = None
            run(gen, args.generator_controls_output)
        if args.transition_output:
            from .probe_batch_transition import all_logits_retirement_regression, prepared_prefill_regression

            prompt = (
                gen.tokenizer.encode("A careful scientist checks the evidence before drawing a conclusion. ") * 30
            )[:257]
            gen.generate(prompt, 4)
            transitions = {"layers": gen.model.num_layers, "all_logits": all_logits_retirement_regression(gen, prompt)}

            def save():
                Path(args.transition_output).write_text(json.dumps(transitions, indent=2) + "\n")

            save()
            for batch in [2, 32]:
                record = transitions.setdefault(f"prepared_batch{batch}", {})
                prepared_prefill_regression(gen, prompt, record, save, batch=batch)
            transitions["pass"] = True
            save()
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
