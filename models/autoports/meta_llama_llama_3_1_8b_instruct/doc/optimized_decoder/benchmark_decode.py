# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Evidence runner for the Llama 3.1 8B optimized decoder stage.

This is intentionally kept beside the optimized-decoder report artifacts.  It
uses the same test harness inputs as the stage tests, compares functional and
optimized decode latency in the same process, and records lower-precision
candidate outcomes used by the work log.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tests.test_optimized_decoder import (
    _build_decode_case,
    _pcc,
    _synthetic_state_dict,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    EMITTED_BATCH_SIZE,
    EMITTED_CACHE_LEN,
    MODEL_ID,
    FunctionalDecoder,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import OptimizedDecoderPolicy

DOC_DIR = Path(__file__).resolve().parent


def main() -> None:
    hf_config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    hf_config._attn_implementation = "eager"
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=8 << 20, physical_device_ids=[0])
    try:
        latency = _latency_comparison(mesh_device, hf_config)
        candidates = _candidate_trials(mesh_device, hf_config)
    finally:
        ttnn.close_mesh_device(mesh_device)

    _write_json("benchmark_decode_latency.json", latency)
    _write_json("candidate_precision_trials.json", candidates)


def _latency_comparison(mesh_device, hf_config) -> dict:
    state_dict = _synthetic_state_dict(hf_config, layer_idx=0)
    optimized = _build_decode_case(mesh_device, hf_config, state_dict, 0, "benchmark", scale=0.02)
    functional = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH_SIZE,
        cache_len=EMITTED_CACHE_LEN,
    )
    functional_inputs = dict(optimized["tt_inputs"])

    functional_ms = _time_decode(functional, functional_inputs)
    optimized_ms = _time_decode(optimized["decoder"], optimized["tt_inputs"])

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = optimized["decoder"].decode_forward(**optimized["tt_inputs"])
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    start = time.perf_counter()
    for _ in range(5):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)
    optimized_traced_ms = (time.perf_counter() - start) * 1000.0 / 5.0
    ttnn.release_trace(mesh_device, trace_id)

    actual = ttnn.to_torch(trace_output).reshape(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size)
    pcc = _pcc(optimized["hf_out"].float(), actual.float())
    tt_key_cache = ttnn.to_torch(optimized["tt_inputs"]["key_cache"])
    tt_value_cache = ttnn.to_torch(optimized["tt_inputs"]["value_cache"])
    cache_position = EMITTED_CACHE_LEN - 1
    key_cache_pcc = _pcc(
        optimized["hf_key_cache"][:, :, cache_position : cache_position + 1, :].float(),
        tt_key_cache[:, :, cache_position : cache_position + 1, :].float(),
    )
    value_cache_pcc = _pcc(
        optimized["hf_value_cache"][:, :, cache_position : cache_position + 1, :].float(),
        tt_value_cache[:, :, cache_position : cache_position + 1, :].float(),
    )
    return {
        "weights": "synthetic",
        "batch": EMITTED_BATCH_SIZE,
        "cache_len": EMITTED_CACHE_LEN,
        "functional_eager_ms": functional_ms,
        "optimized_eager_ms": optimized_ms,
        "optimized_traced_ms_per_replay": optimized_traced_ms,
        "optimized_vs_functional_eager_speedup": functional_ms / optimized_ms,
        "optimized_traced_vs_functional_eager_speedup": functional_ms / optimized_traced_ms,
        "optimized_pcc": pcc,
        "optimized_key_cache_pcc": key_cache_pcc,
        "optimized_value_cache_pcc": value_cache_pcc,
    }


def _candidate_trials(mesh_device, hf_config) -> dict:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True)
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    model.eval()
    state_dict = model.state_dict()
    policies = {
        "default_bfp8_hifi2_dram_sharded": OptimizedDecoderPolicy(),
        "split_gate_up_bfp8_hifi2": OptimizedDecoderPolicy(use_packed_gate_up_projection=False),
        "mlp_bfp4_attention_bfp8": OptimizedDecoderPolicy(
            attention_weight_dtype=ttnn.bfloat8_b,
            mlp_weight_dtype=ttnn.bfloat4_b,
            output_weight_dtype=ttnn.bfloat8_b,
        ),
        "mlp_bfp4_lofi_attention_bfp8": OptimizedDecoderPolicy(
            attention_weight_dtype=ttnn.bfloat8_b,
            mlp_weight_dtype=ttnn.bfloat4_b,
            output_weight_dtype=ttnn.bfloat8_b,
            mlp_math_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "mlp_bfp4_hifi4_attention_bfp8": OptimizedDecoderPolicy(
            attention_weight_dtype=ttnn.bfloat8_b,
            mlp_weight_dtype=ttnn.bfloat4_b,
            output_weight_dtype=ttnn.bfloat8_b,
            mlp_math_fidelity=ttnn.MathFidelity.HiFi4,
        ),
        "all_linear_bfp4": OptimizedDecoderPolicy(
            attention_weight_dtype=ttnn.bfloat4_b,
            mlp_weight_dtype=ttnn.bfloat4_b,
            output_weight_dtype=ttnn.bfloat4_b,
        ),
        "all_linear_bfp4_lofi": OptimizedDecoderPolicy(
            attention_weight_dtype=ttnn.bfloat4_b,
            mlp_weight_dtype=ttnn.bfloat4_b,
            output_weight_dtype=ttnn.bfloat4_b,
            attention_math_fidelity=ttnn.MathFidelity.LoFi,
            mlp_math_fidelity=ttnn.MathFidelity.LoFi,
        ),
        "interleaved_bfp8_no_dram_sharded_program": OptimizedDecoderPolicy(
            attention_weight_dtype=ttnn.bfloat8_b,
            mlp_weight_dtype=ttnn.bfloat8_b,
            output_weight_dtype=ttnn.bfloat8_b,
            use_dram_sharded_weights=False,
        ),
        "decode_create_heads_candidate": OptimizedDecoderPolicy(use_decode_create_heads=True),
        "packed_gate_up_bfp8_hifi2": OptimizedDecoderPolicy(use_packed_gate_up_projection=True),
    }
    results = {}
    for name, policy in policies.items():
        try:
            layer_results = {}
            for layer_idx in (0, 31):
                case = _build_decode_case(
                    mesh_device,
                    hf_config,
                    state_dict,
                    layer_idx,
                    "real",
                    scale=0.01,
                    policy=policy,
                )
                ms = _time_decode(case["decoder"], case["tt_inputs"])
                out = case["decoder"].decode_forward(**case["tt_inputs"])
                actual = ttnn.to_torch(out).reshape(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size)
                pcc = _pcc(case["hf_out"].float(), actual.float())
                layer_results[f"layer{layer_idx}"] = {"eager_ms": ms, "pcc": pcc}
            results[name] = {
                "status": "pass" if min(v["pcc"] for v in layer_results.values()) >= 0.99 else "fail",
                "layers": layer_results,
                "policy": _policy_name(policy),
            }
        except Exception as exc:  # noqa: BLE001 - evidence runner records candidate failures.
            results[name] = {"status": "fail", "error": f"{type(exc).__name__}: {exc}", "policy": _policy_name(policy)}
    synthetic_layer_kind_checks = {}
    reduced_precision_policies = {
        "mlp_bfp4_attention_bfp8": OptimizedDecoderPolicy(
            attention_weight_dtype=ttnn.bfloat8_b,
            mlp_weight_dtype=ttnn.bfloat4_b,
            output_weight_dtype=ttnn.bfloat8_b,
        ),
        "all_linear_bfp4": OptimizedDecoderPolicy(
            attention_weight_dtype=ttnn.bfloat4_b,
            mlp_weight_dtype=ttnn.bfloat4_b,
            output_weight_dtype=ttnn.bfloat4_b,
        ),
    }
    for policy_name, reduced_policy in reduced_precision_policies.items():
        for layer_idx in (0, 31):
            try:
                case = _build_decode_case(
                    mesh_device,
                    hf_config,
                    _synthetic_state_dict(hf_config, layer_idx=layer_idx),
                    layer_idx,
                    "synthetic",
                    scale=0.02,
                    policy=reduced_policy,
                )
                out = case["decoder"].decode_forward(**case["tt_inputs"])
                actual = ttnn.to_torch(out).reshape(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size)
                pcc = _pcc(case["hf_out"].float(), actual.float())
                synthetic_layer_kind_checks[f"{policy_name}_layer{layer_idx}"] = {
                    "status": "pass" if pcc >= 0.99 else "fail",
                    "pcc": pcc,
                }
            except Exception as exc:  # noqa: BLE001 - evidence runner records candidate failures.
                synthetic_layer_kind_checks[f"{policy_name}_layer{layer_idx}"] = {
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                }
    try:
        case = _build_decode_case(
            mesh_device,
            hf_config,
            _synthetic_state_dict(hf_config, layer_idx=31),
            31,
            "synthetic",
            scale=0.02,
            policy=OptimizedDecoderPolicy(use_decode_create_heads=True),
        )
        out = case["decoder"].decode_forward(**case["tt_inputs"])
        actual = ttnn.to_torch(out).reshape(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size)
        pcc = _pcc(case["hf_out"].float(), actual.float())
        synthetic_layer_kind_checks["decode_create_heads_layer31"] = {
            "status": "pass" if pcc >= 0.99 else "fail",
            "pcc": pcc,
        }
    except Exception as exc:  # noqa: BLE001 - evidence runner records candidate failures.
        synthetic_layer_kind_checks["decode_create_heads_layer31"] = {
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "weights": "real",
        "batch": EMITTED_BATCH_SIZE,
        "cache_len": EMITTED_CACHE_LEN,
        "pcc_threshold": 0.99,
        "candidates": results,
        "geometry_sweep": _geometry_sweep(mesh_device, hf_config, state_dict),
        "synthetic_layer_kind_checks": synthetic_layer_kind_checks,
    }


def _geometry_sweep(mesh_device, hf_config, state_dict) -> dict:
    roles = {
        "qvk": ("qvk_in0_block_w", (1, 2, 4, 8, 16)),
        "o": ("o_in0_block_w", (1, 2, 4, 8, 16)),
        "packed_gate_up": ("gate_in0_block_w", (1, 2, 4, 8, 16)),
        "down": ("down_in0_block_w", (1, 2, 4, 7, 8, 16)),
    }
    results = {}
    for role, (field_name, block_values) in roles.items():
        role_results = {}
        for in0_block_w in block_values:
            kwargs = {field_name: in0_block_w}
            try:
                policy = OptimizedDecoderPolicy(**kwargs)
                case = _build_decode_case(
                    mesh_device,
                    hf_config,
                    state_dict,
                    0,
                    "real",
                    scale=0.01,
                    policy=policy,
                )
                ms = _time_decode(case["decoder"], case["tt_inputs"], repeats=3)
                out = case["decoder"].decode_forward(**case["tt_inputs"])
                actual = ttnn.to_torch(out).reshape(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size)
                pcc = _pcc(case["hf_out"].float(), actual.float())
                role_results[str(in0_block_w)] = {
                    "status": "pass" if pcc >= 0.99 else "fail",
                    "eager_ms": ms,
                    "pcc": pcc,
                }
            except Exception as exc:  # noqa: BLE001 - evidence runner records candidate failures.
                role_results[str(in0_block_w)] = {
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        results[role] = role_results

    adapted_roles = {
        "qvk": ("qvk_l1_core_count", "qvk_in0_block_w", ((32, None), (32, 4), (16, None), (16, 8))),
        "o": ("o_l1_core_count", "o_in0_block_w", ((32, None), (32, 4), (16, None), (16, 8))),
        "packed_gate_up": ("gate_l1_core_count", "gate_in0_block_w", ((32, None), (32, 4), (16, None), (16, 8))),
        "down": ("down_l1_core_count", "down_in0_block_w", ((32, None), (32, 7), (32, 14), (16, None), (16, 14))),
    }
    adapted_results = {}
    for role, (core_field, block_field, candidates) in adapted_roles.items():
        role_results = {}
        for core_count, in0_block_w in candidates:
            kwargs = {core_field: core_count}
            if in0_block_w is not None:
                kwargs[block_field] = in0_block_w
            name = f"cores{core_count}_in0{in0_block_w or 'auto'}"
            try:
                policy = OptimizedDecoderPolicy(**kwargs)
                case = _build_decode_case(
                    mesh_device,
                    hf_config,
                    state_dict,
                    0,
                    "real",
                    scale=0.01,
                    policy=policy,
                )
                ms = _time_decode(case["decoder"], case["tt_inputs"], repeats=3)
                out = case["decoder"].decode_forward(**case["tt_inputs"])
                actual = ttnn.to_torch(out).reshape(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size)
                pcc = _pcc(case["hf_out"].float(), actual.float())
                role_results[name] = {
                    "status": "pass" if pcc >= 0.99 else "fail",
                    "eager_ms": ms,
                    "pcc": pcc,
                    "policy": _policy_name(policy),
                }
            except Exception as exc:  # noqa: BLE001 - evidence runner records candidate failures.
                role_results[name] = {
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}",
                    "policy": _policy_name(OptimizedDecoderPolicy(**kwargs)),
                }
        adapted_results[role] = role_results

    return {"in0_block_w_same_shard_shape": results, "adapted_l1_core_count": adapted_results}


def _time_decode(decoder, inputs: dict, *, warmups: int = 2, repeats: int = 5) -> float:
    for _ in range(warmups):
        decoder.decode_forward(**inputs)
    ttnn.synchronize_device(decoder.mesh_device)
    start = time.perf_counter()
    for _ in range(repeats):
        decoder.decode_forward(**inputs)
    ttnn.synchronize_device(decoder.mesh_device)
    return (time.perf_counter() - start) * 1000.0 / repeats


def _policy_name(policy: OptimizedDecoderPolicy) -> dict[str, str | bool]:
    return {
        "attention_weight_dtype": str(policy.attention_weight_dtype),
        "mlp_weight_dtype": str(policy.mlp_weight_dtype),
        "output_weight_dtype": str(policy.output_weight_dtype),
        "activation_dtype": str(policy.activation_dtype),
        "attention_math_fidelity": str(policy.attention_math_fidelity),
        "mlp_math_fidelity": str(policy.mlp_math_fidelity),
        "qvk_in0_block_w": policy.qvk_in0_block_w,
        "o_in0_block_w": policy.o_in0_block_w,
        "gate_in0_block_w": policy.gate_in0_block_w,
        "up_in0_block_w": policy.up_in0_block_w,
        "down_in0_block_w": policy.down_in0_block_w,
        "qvk_l1_core_count": policy.qvk_l1_core_count,
        "o_l1_core_count": policy.o_l1_core_count,
        "gate_l1_core_count": policy.gate_l1_core_count,
        "up_l1_core_count": policy.up_l1_core_count,
        "down_l1_core_count": policy.down_l1_core_count,
        "use_dram_sharded_weights": policy.use_dram_sharded_weights,
        "use_decode_create_heads": policy.use_decode_create_heads,
        "use_explicit_sdpa_program_config": policy.use_explicit_sdpa_program_config,
        "use_packed_gate_up_projection": policy.use_packed_gate_up_projection,
    }


def _write_json(name: str, data: dict) -> None:
    (DOC_DIR / name).write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
