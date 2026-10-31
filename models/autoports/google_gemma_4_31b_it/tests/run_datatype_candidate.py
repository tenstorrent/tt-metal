"""Full-model precision candidate measurement against the pinned readiness reference."""

import argparse
import copy
import hashlib
import json
import statistics
import subprocess
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_prefill_check, run_teacher_forcing

import ttnn

from ..tt.generator import build_generator
from ..tt.precision import POLICIES, load_precision
from ..tt.weights import HF_REVISION
from .optimized_generation_evidence import performance
from .run_full_batch import inventory, memory


def kernel_summary(kernel):
    return {
        key: str(getattr(kernel, key))
        for key in ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--state", action="store_true")
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--acceptance", action="store_true")
    parser.add_argument("--id", default="baseline")
    args = parser.parse_args()
    policy = load_precision(args.config)
    torch.set_num_threads(8)
    model_dir = Path("models/autoports/google_gemma_4_31b_it")
    source = model_dir / "doc/optimized_full_model"
    root = model_dir / "doc/datatype_sweep" / args.id
    root.mkdir(parents=True, exist_ok=True)
    reference = json.loads((source / "reference_metadata.json").read_text())
    assert reference["revision"] == HF_REVISION and reference["actual_generated_tokens"] == 100
    (root / "aime24_prompt.json").write_text((source / "aime24_prompt.json").read_text())
    manifest = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((model_dir / "tt").glob("*.py"))
        },
    }
    (root / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    try:
        gen = build_generator(
            model_dir,
            mesh,
            layer_indices=[0, 5] if args.smoke else None,
            batch_size=args.batch,
            **({"precision_config": policy} if args.config else {}),
        )
        summary = {
            "policy": gen.model.precision_config,
            "builder_module": build_generator.__module__,
            "generator_class": type(gen).__module__,
            "model_class": type(gen.model).__module__,
            "layers": [
                {
                    "index": i,
                    "weights": {n: str(w.dtype) for n, w in layer.weights.items()},
                    "kernels": {n: kernel_summary(k) for n, k in layer.linear_kernels.items()},
                    "cache_dtype": str(layer.cache_dtype),
                    "residual_dtype": str(layer.residual_dtype),
                    "decode_ccl_dtype": str(layer.decode_ccl_dtype),
                    "activation_dtype": str(layer.prefill_activation_dtype),
                    "prefill_ccl_bfp8": layer.prefill_ccl_bfp8,
                }
                for i, layer in zip(gen.model.layer_indices, gen.model.layers)
            ],
            "head_dtype": str(gen.model.lm_head[0].dtype),
            "embedding_dtype": str(gen.model.embedding.dtype),
            "norm_dtype": str(gen.model.norm_weight.dtype),
            "sampling_logits_dtype": str(gen.logits.dtype),
            "max_seq_len": gen.context,
            "head_kernel": kernel_summary(gen.model.kernel),
        }
        (root / "runtime_policy.json").write_text(json.dumps(summary, indent=2) + "\n")
        for index, layer in zip(gen.model.layer_indices, gen.model.layers):
            groups = dict(policy["weight_groups"])
            groups.update(policy["layer_exceptions"].get(str(index), {}))
            for group in ("attention", "mlp", "down"):
                assert layer.linear_kernels[group].math_fidelity == getattr(
                    ttnn.MathFidelity, POLICIES[groups[group]][1]
                )
            for name, weight in layer.weights.items():
                if "proj" in name or "prefill_qkv" in name:
                    group = "attention" if name.startswith("self_attn.") else "down" if "down_proj" in name else "mlp"
                    assert weight.dtype == getattr(ttnn, POLICIES[groups[group]][0]), name
            assert layer.cache_dtype == getattr(ttnn, policy["kv_cache_dtype"])
            assert layer.prefill_activation_dtype == getattr(ttnn, policy["activation_dtype"])
        assert gen.model.kernel.math_fidelity == getattr(ttnn.MathFidelity, policy["compute_fidelities"]["lm_head"])
        constructed = memory(mesh)
        owned = inventory(gen)
        gen.prepare_decode()
        (root / "memory.json").write_text(
            json.dumps({"constructed": constructed, "prepared": memory(mesh), "inventory": owned}, indent=2) + "\n"
        )
        if args.smoke:
            prompt = json.loads((root / "aime24_prompt.json").read_text())["tokens"]
            records = []
            for length in (129, 1023, 1025):
                ids = (prompt * ((length + len(prompt) - 1) // len(prompt)))[:length]
                tokens = gen.generate(ids, 4)
                records.append({"length": length, "tokens": tokens})
            (root / "smoke.json").write_text(json.dumps(records, indent=2) + "\n")
            return
        if args.state:
            from .datatype_state import run as state

            state(gen, root)
            return
        if args.long_only:
            from .datatype_long_prefix import run as long_prefix

            long_prefix(gen, root)
            return
        if args.batch != 1:
            performance(gen, root)
            if args.acceptance:
                from .datatype_long_prefix import run as long_prefix

                long_prefix(gen, root)
            return
        result = dict(scope="full60layers", reference=reference["reference"], revision=HF_REVISION)
        for name, module, fn in [
            ("prefill", run_prefill_check, run_prefill_check.run_prefill_check),
            ("decode", run_teacher_forcing, run_teacher_forcing.run_teacher_forcing),
        ]:
            before = dict(gen.counters)
            with patch.object(module, "_import_build_generator", return_value=lambda **kwargs: gen):
                measurements = [
                    fn(model_dir=model_dir, reference_path=Path(reference["reference"]), mesh_device=mesh)
                    for _ in range(3 if name == "decode" else 1)
                ]
                result[name] = copy.deepcopy(measurements[0])
                if name == "decode":
                    result["teacher_forcing_repeats"] = measurements
                    for key in ("decode_t/s/u", "ttft_ms"):
                        result[name][0][key] = statistics.median(r[0][key] for r in measurements)
            result[name + "_counters"] = {k: gen.counters[k] - v for k, v in before.items()}
            result["traces"] = {"model": str(gen.model_trace), "sampling": str(gen.sample_trace)}
            (root / "readiness_accuracy.json").write_text(json.dumps(result, indent=2) + "\n")
        performance(gen, root)
        from .datatype_long_prefix import run as long_prefix

        long_prefix(gen, root)
        if args.acceptance:
            from .full_generation_evidence import qualitative
            from .run_optimized_full_readiness import context_boundaries

            for filename in ("qualitative_hf.json", "qualitative_prompt_format.json"):
                (root / filename).write_text((source / filename).read_text())
            qualitative(gen, root)
            context_boundaries(gen, root)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
