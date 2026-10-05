"""Full-stack precision evaluation; only traced teacher forcing ranks policies."""

import argparse
import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_prefill_check, run_teacher_forcing
from readiness_check.schema import load_reference

import ttnn

from ..tt.generator import build_generator
from ..tt.precision_config import layer_policy, load_precision_config

MODEL = Path("models/demos/k2_horizon_7b_qb2")
DOC = MODEL / "doc/datatype_sweep"


def runtime_summary(gen):
    config = gen.model.precision_config
    rows = []
    for index, layer in enumerate(gen.model.layers):
        assert layer.policy == layer_policy(config, index)
        row = {"layer": index, "policy": asdict(layer.policy), "matmuls": {}}
        for role, weight in [("qkv", layer.wqkv), ("o", layer.wo), ("mlp", layer.wgate), ("down", layer.wdown)]:
            group = "attention" if role in {"qkv", "o"} else role
            dtype = getattr(layer.policy, role + "_dtype", None) or getattr(layer.policy, group)
            fidelity = getattr(layer.policy, role + "_fidelity", None) or getattr(layer.policy, group + "_fidelity")
            compute = layer.decode_computes[id(weight)]
            assert weight.dtype == layer.decode_weights[id(weight)].dtype == getattr(ttnn, dtype)
            assert compute.math_fidelity == getattr(ttnn.MathFidelity, fidelity)
            assert compute.fp32_dest_acc_en == config["accumulation"]["decode_fp32"]
            assert compute.math_approx_mode == config["accumulation"]["math_approx_mode"]
            assert compute.packer_l1_acc == config["accumulation"]["packer_l1_acc"]
            prefill = layer._weight_compute(weight)
            assert prefill.math_fidelity == getattr(ttnn.MathFidelity, fidelity)
            assert prefill.fp32_dest_acc_en == config["accumulation"]["prefill_fp32"]
            assert prefill.math_approx_mode == config["accumulation"]["math_approx_mode"]
            assert prefill.packer_l1_acc == config["accumulation"]["packer_l1_acc"]
            row["matmuls"][role] = {
                "prefill_weight_dtype": str(weight.dtype),
                "decode_weight_dtype": str(layer.decode_weights[id(weight)].dtype),
                "decode_fidelity": str(compute.math_fidelity),
                "prefill_fidelity": str(prefill.math_fidelity),
                "decode_fp32": compute.fp32_dest_acc_en,
                "prefill_fp32": prefill.fp32_dest_acc_en,
                "math_approx_mode": compute.math_approx_mode,
                "packer_l1_acc": compute.packer_l1_acc,
                "decode_input_dtype": (
                    layer.policy.attention_activation if group == "attention" else layer.policy.mlp_activation
                ),
            }
        assert layer.ccl_dtype == getattr(ttnn, config["dtypes"]["ccl"])
        assert layer.wup.dtype == layer.wgate.dtype
        assert (
            layer.decode_computes[id(layer.wup)].math_fidelity == layer.decode_computes[id(layer.wgate)].math_fidelity
        )
        assert layer.compute.math_fidelity == getattr(ttnn.MathFidelity, config["accumulation"]["norm_fidelity"])
        assert layer.residual_dtype == getattr(ttnn, config["dtypes"]["residual"])
        assert layer.matmul_output_dtype == getattr(ttnn, config["dtypes"]["matmul_output"])
        assert layer.attention_compute.math_fidelity == getattr(ttnn.MathFidelity, layer.policy.sdpa_fidelity)
        assert layer.prefill_qkv_dtype == getattr(ttnn, layer.policy.prefill_qkv_activation)
        assert layer.prefill_mlp_dtype == getattr(ttnn, layer.policy.prefill_mlp_activation)
        row.update(
            ccl_dtype=str(layer.ccl_dtype),
            residual_dtype=str(layer.residual_dtype),
            kv_dtype=str(layer.kv_dtype),
            norm_fidelity=str(layer.compute.math_fidelity),
            matmul_output_dtype=str(layer.matmul_output_dtype),
            gather_output_dtype=str(layer.gather_output_dtype),
            sdpa_fidelity=str(layer.attention_compute.math_fidelity),
            prefill_qkv_activation_dtype=str(layer.prefill_qkv_dtype),
            prefill_mlp_activation_dtype=str(layer.prefill_mlp_dtype),
        )
        rows.append(row)
    assert gen.model.head.dtype == getattr(ttnn, config["head"]["weight_dtype"])
    assert gen.model.head_compute.math_fidelity == getattr(ttnn.MathFidelity, config["head"]["compute_fidelity"])
    assert all(weight.dtype == gen.model.head.dtype for weight in gen.model.head_decode.output_weights)
    assert gen.model.head_decode.config.compute_kernel_config.math_fidelity == gen.model.head_compute.math_fidelity
    return {
        "construction": "tt.generator.build_generator -> K2Generator -> K2Model -> required precision artifact",
        "config": config,
        "layers": rows,
        "embedding_dtype": str(gen.model.embedding.dtype),
        "norm_weight_dtype": str(gen.model.norm_weight.dtype),
        "head_dtype": str(gen.model.head.dtype),
        "head_compute_fidelity": str(gen.model.head_compute.math_fidelity),
        "logits_dtype": str(gen.model.logits_dtype),
        "head_decode_weight_dtypes": [str(weight.dtype) for weight in gen.model.head_decode.output_weights],
        "head_decode_compute_fidelity": str(gen.model.head_decode.config.compute_kernel_config.math_fidelity),
        "sampling_values_dtype": str(gen.p.dtype),
        "sampling_token_dtype": str(gen.k.dtype),
        "max_context": gen.model.context,
    }


def continuation_check(gen):
    """Preserve the stage6/7 repeated-BOS whole/split numerical contract."""
    from .probe_full_continuation import compare

    fixture = json.loads((MODEL / "doc/full_model/stress_hf_topk.json").read_text())["records"][0]
    prompt = torch.tensor([fixture["prompt_token_ids"]])
    assert prompt.shape == (1, 257)
    gen._ensure_owned_cache(1, 512)
    gen.reset()
    whole = gen.prefill_forward(
        prompt, page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[257], sampling_mode="host"
    )
    rows = []
    for cut in [31, 32]:
        gen.reset()
        gen.prefill_forward(prompt[:, :cut], page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[cut])
        split = gen.prefill_forward(
            prompt[:, cut:],
            page_table=gen.page_table,
            kv_cache=gen.kv_cache,
            prompt_lens=[257 - cut],
            start_pos=[cut],
            sampling_mode="host",
        )
        metrics = compare(whole, split)
        tokens = [int(value.flatten().argmax()) for value in (whole, split)]
        rows.append(
            dict(
                cut=cut,
                metrics=metrics,
                whole_token=tokens[0],
                split_token=tokens[1],
                hf_token=fixture["hf_top1"]["id"],
                passed=metrics["pcc"] >= 0.995 and tokens == [fixture["hf_top1"]["id"]] * 2,
            )
        )
    return {
        "prompt_len": 257,
        "fixture": str(MODEL / "doc/full_model/stress_hf_topk.json"),
        "inherited_minimum_pcc": 0.995,
        "checks": rows,
        "pass": all(row["passed"] for row in rows),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--quality", action="store_true")
    p.add_argument("--continuation", action="store_true")
    args = p.parse_args()
    config = load_precision_config(args.config)
    reference = MODEL / "readiness_aime24_chat.refpt"
    ref = load_reference(reference)
    assert len(ref.entries) == 1 and ref.entries[0].generated_tokens.shape[-1] == 100
    result = {
        "config_id": config["config_id"],
        "precision_config": str(args.config) if args.config else "default selected artifact",
        "dtype_policy": config,
        "compute_fidelity_policy": {
            "defaults": {k: v for k, v in config["layer_defaults"].items() if "fidelity" in k},
            "layer_exceptions": config["layer_exceptions"],
            "head": config["head"],
            "accumulation": config["accumulation"],
        },
        "command": shlex.join([sys.executable, "-m", __spec__.name, *sys.argv[1:]]),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "hardware": "4x Blackhole P300c, 11x10 worker cores, 8 DRAM banks/rank",
        "mesh": [1, 4],
        "reference": str(reference),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "prompt_len": ref.entries[0].prompt_tokens.shape[-1],
        "generation_len": 100,
        "batch_size": 1,
        "measurement_regime": "warmed traced teacher forcing with prediction readback and forced-token refresh; median repeats",
        "layers": 1 if args.smoke else 36,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((MODEL / "tt").glob("*.py"))
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {
            key: os.environ.get(key)
            for key in [
                "HF_HUB_OFFLINE",
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
            ]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    torch.set_num_threads(16)
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200_000_000)
    gen = None
    try:
        kwargs = {"precision_config": args.config} if args.config else {}
        if args.smoke:
            kwargs["override_num_layers"] = 1
        gen = build_generator(MODEL, mesh, **kwargs)
        result["runtime_summary"] = runtime_summary(gen)
        result["runtime_policies"] = [asdict(layer.policy) for layer in gen.model.layers]
        result["weight_allocations"] = [layer.weight_allocations for layer in gen.model.layers]
        save()
        if args.smoke:
            prompt = ref.entries[0].prompt_tokens[0].tolist()
            checks = []
            for length in [31, 33, 257]:
                ids = (prompt * 3)[:length]
                a = gen.generate(ids, 4)
                b = gen.generate(ids, 4)
                assert a == b
                checks.append({"length": length, "tokens": a, "repeat_exact": True, "perf": gen.last_perf.copy()})
            result.update(smoke=checks, status="smoke_pass")
        else:
            if args.benchmark:
                from .benchmark_full_model import run

                result["token_out_artifact"] = str(args.output.with_name(args.output.stem + "_token_out.json"))
                run(gen, result["token_out_artifact"])
            with patch.object(run_prefill_check, "_import_build_generator", return_value=lambda **kw: gen):
                result["prefill"] = run_prefill_check.run_prefill_check(
                    model_dir=MODEL, reference_path=reference, mesh_device=mesh
                )
            result["teacher_runs"] = []
            result["teacher_perf"] = []
            for iteration in range(args.repeats + 1):
                with patch.object(run_teacher_forcing, "_import_build_generator", return_value=lambda **kw: gen):
                    scores = run_teacher_forcing.run_teacher_forcing(
                        model_dir=MODEL, reference_path=reference, mesh_device=mesh
                    )
                perf = gen.last_perf.copy()
                counters = perf["steady_state_counters"]
                assert perf["teacher_forcing"] and counters["model_replays"] == counters["sampling_replays"] == 99
                assert counters["token_refreshes"] == counters["token_readbacks"] == 99
                if iteration:
                    result["teacher_runs"].append(scores[0])
                    result["teacher_perf"].append(perf)
                save()
            for key in ["top1", "top5", "top100"]:
                assert len({row[key] for row in result["teacher_runs"]}) == 1
                result[key] = result["teacher_runs"][0][key]
            result["token_count"] = 100
            result["cache_capacity_tokens"] = gen.capacity
            result["ttft_ms"] = statistics.median(row["ttft_seconds"] * 1000 for row in result["teacher_perf"])
            result["decode_tokens_per_second_per_user"] = statistics.median(
                row["decode_tokens_per_second_per_user"] for row in result["teacher_perf"]
            )
            result["trace_verified"] = True
            result["accuracy_pass"] = all(
                row["top1"] >= 0.90 and row["top5"] >= 0.98 and row["top100"] == 1
                for row in [*result["prefill"], *result["teacher_runs"]]
            )
            result["status"] = "pass" if result["accuracy_pass"] else "accuracy_fail"
            if args.continuation:
                result["continuation"] = continuation_check(gen)
                result["capability_pass"] = result["continuation"]["pass"]
                if result["accuracy_pass"] and not result["capability_pass"]:
                    result["status"] = "continuation_fail"
            if args.quality and result.get("capability_pass", True):
                from .run_qualitative_extended import run

                path = args.output.with_name(args.output.stem + "_quality.json")
                run(gen, output_name=str(path.resolve()))
                result["quality_artifact"] = str(path)
            elif args.quality:
                result["quality_skipped"] = "Candidate already rejected by the inherited continuation gate"
        result["runtime_cache"] = [
            {"dtypes": [str(t.dtype) for t in pair], "shapes": [list(t.shape) for t in pair]} for pair in gen.kv_cache
        ]
        for layer, pair in zip(gen.model.layers, gen.kv_cache):
            assert all(t.dtype == layer.kv_dtype for t in pair)
        if gen.state is not None and "logits" in gen.state:
            result["runtime_logits_dtype"] = str(gen.state["logits"].dtype)
            assert gen.state["logits"].dtype == gen.model.logits_dtype
            result["runtime_token_dtype"] = str(gen.state["tokens"].dtype)
            assert gen.state["tokens"].dtype == gen.token_dtype
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        save()
        print(
            "CANDIDATE_RESULT",
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k
                    in {"config_id", "status", "top1", "top5", "top100", "ttft_ms", "decode_tokens_per_second_per_user"}
                }
            ),
            flush=True,
        )
    except Exception as exc:
        result.update(status="runtime_error", error=str(exc)[:4000])
        save()
        raise
    finally:
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
