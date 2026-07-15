# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Evaluate several precision policies without reloading unchanged full-model weights."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import replace
from pathlib import Path

import ttnn
from models.autoports.google_gemma_4_31b.tests.run_datatype_sweep_candidate import _git_revision, _teacher_forcing
from models.autoports.google_gemma_4_31b.tt.generator import build_generator
from models.autoports.google_gemma_4_31b.tt.precision import load_precision_config
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device


def _compute(mesh, fidelity):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=fidelity,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def _weight_signature(resolved) -> tuple:
    policy = resolved.default_decoder_policy
    return (
        policy.attention_weight_dtype,
        policy.resolved_attention_qkv_weight_dtype,
        policy.resolved_attention_o_weight_dtype,
        policy.mlp_gate_up_weight_dtype,
        policy.mlp_down_weight_dtype,
        resolved.lm_head_weight_dtype,
        policy.kv_cache_dtype,
        resolved.logits_dtype,
        resolved.sampling_dtype,
        tuple(
            (
                layer,
                override.attention_weight_dtype,
                override.resolved_attention_qkv_weight_dtype,
                override.resolved_attention_o_weight_dtype,
                override.mlp_gate_up_weight_dtype,
                override.mlp_down_weight_dtype,
                override.kv_cache_dtype,
            )
            for layer, override in resolved.layer_decoder_policies
        ),
    )


def _apply_runtime_policy(generator, resolved) -> None:
    model = generator.model
    layer_overrides = dict(resolved.layer_decoder_policies)
    for layer in model.layers:
        policy = layer_overrides.get(layer.layer_idx, resolved.default_decoder_policy)
        layer.policy = policy
        layer.communication_dtype = resolved.decode_ccl_dtype
        layer.prefill_communication_dtype = resolved.prefill_ccl_dtype
        layer.residual_dtype = resolved.residual_dtype
        layer.attention_qkv_compute = _compute(model.mesh_device, policy.resolved_attention_qkv_math_fidelity)
        layer.attention_o_compute = _compute(model.mesh_device, policy.resolved_attention_o_math_fidelity)
        layer.attention_compute = layer.attention_qkv_compute
        mlp = layer.layer.shared_mlp
        mlp.policy = replace(
            mlp.policy,
            name=f"{policy.name}_tp4_square_mlp_14c",
            mlp_gate_up_weight_dtype=policy.mlp_gate_up_weight_dtype,
            mlp_down_weight_dtype=policy.mlp_down_weight_dtype,
            mlp_gate_up_math_fidelity=policy.mlp_gate_up_math_fidelity,
            mlp_down_math_fidelity=policy.mlp_down_math_fidelity,
            kv_cache_dtype=policy.kv_cache_dtype,
        )
        mlp.gate_up_compute = _compute(model.mesh_device, policy.mlp_gate_up_math_fidelity)
        mlp.down_compute = _compute(model.mesh_device, policy.mlp_down_math_fidelity)

    model.config = replace(
        model.config,
        precision_config_id=resolved.config_id,
        precision_config_path=str(resolved.source_path),
        decoder_optimization_policy=resolved.default_decoder_policy,
        layer_decoder_policies=resolved.layer_decoder_policies,
        activation_dtype=resolved.activation_dtype,
        residual_dtype=resolved.residual_dtype,
        prefill_ccl_dtype=resolved.prefill_ccl_dtype,
        decode_ccl_dtype=resolved.decode_ccl_dtype,
        lm_head_math_fidelity=resolved.lm_head_math_fidelity,
    )
    generator.model_config = model.config
    model.lm_head_compute = _compute(model.mesh_device, resolved.lm_head_math_fidelity)


def _result(generator, resolved, stats: dict, args: argparse.Namespace) -> dict:
    counters = dict(generator.model.trace_state.counters)
    trace_verified = counters["model_trace_replays"] == stats["decode_tokens"]
    if not trace_verified:
        raise RuntimeError("same-weight group candidate did not execute one model trace replay per decode token")
    passed = stats["top1"] >= args.min_top1 and stats["top5"] >= args.min_top5 and stats["top100"] >= args.min_top100
    layer_count = len(generator.model.layers)
    return {
        "config_id": resolved.config_id,
        "precision_config": str(resolved.source_path),
        "dtype_policy": resolved.raw,
        "runtime_policy_summary": generator.model.precision_runtime_summary(),
        "accuracy": {
            "teacher_forcing": stats,
            "prefill": None,
            "thresholds": {"top1": args.min_top1, "top5": args.min_top5, "top100": args.min_top100},
        },
        "performance": {
            "ttft_ms": stats["ttft_ms"],
            "trace_verified_teacher_forcing_decode_t/s/u": stats["decode_t/s/u"],
            "teacher_forcing_e2e_t/s/u": stats["e2e_t/s/u"],
            "measurement_regime": (
                f"{'full' if layer_count == 60 else 'reduced'}-{layer_count}-layer batch-1 traced teacher forcing; "
                "AIME24 149-token prompt; 100 reference tokens"
            ),
        },
        "trace_verified": trace_verified,
        "trace_counters": counters,
        "pass": passed,
        "status": "pass" if passed else "fail_accuracy",
        "command": shlex.join(sys.argv) + f" # candidate={resolved.config_id}",
        "git_commit": _git_revision(),
        "hardware": "4x Blackhole P150b",
        "mesh": "MeshShape(1,4) TP4 FABRIC_1D",
        "reference": str(args.reference.resolve()),
    }


def run(args: argparse.Namespace) -> None:
    configs = [load_precision_config(path) for path in args.precision_config]
    base_signature = _weight_signature(configs[0])
    incompatible = [config.config_id for config in configs[1:] if _weight_signature(config) != base_signature]
    if incompatible:
        raise ValueError(f"same-weight group contains incompatible physical dtype/cache policies: {incompatible}")
    mesh = open_readiness_mesh_device("P150_X4", "FABRIC_1D")
    generator = None
    try:
        generator = build_generator(
            model_dir=args.model_dir.resolve(),
            mesh_device=mesh,
            precision_config_path=configs[0].source_path,
            tensor_cache_path=args.tensor_cache,
            layer_indices=tuple(args.layer_indices) if args.layer_indices else None,
        )
        for index, resolved in enumerate(configs):
            if index:
                generator.reset()
            _apply_runtime_policy(generator, resolved)
            stats = _teacher_forcing(generator, args.reference.resolve())
            result = _result(generator, resolved, stats, args)
            output = args.output_dir / f"{resolved.config_id}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                prior = json.loads(output.read_text(encoding="utf-8"))
                if (
                    prior.get("config_id") == resolved.config_id
                    and prior.get("accuracy", {}).get("prefill") is not None
                ):
                    result["accuracy"]["prefill"] = prior["accuracy"]["prefill"]
            output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(
                json.dumps(
                    {
                        "config_id": resolved.config_id,
                        "top1": stats["top1"],
                        "top5": stats["top5"],
                        "top100": stats["top100"],
                        "decode_t/s/u": stats["decode_t/s/u"],
                        "status": result["status"],
                    }
                )
            )
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh, "FABRIC_1D")


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--precision-config", type=Path, action="append", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-cache", type=Path, default=Path("/tmp/gemma4_31b_full_model_tensor_cache"))
    parser.add_argument("--layer-indices", type=int, nargs="*")
    parser.add_argument("--min-top1", type=float, default=0.90)
    parser.add_argument("--min-top5", type=float, default=0.98)
    parser.add_argument("--min-top100", type=float, default=1.0)
    run(parser.parse_args())


if __name__ == "__main__":
    _main()
