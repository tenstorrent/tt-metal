# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Instrumented real layer-0 upstream precision diagnostic, outside measured passes.

All HF injections and host readbacks are diagnostic/setup operations. This probe
does not validate runtime fallback, timing, or traced execution and changes no
runtime implementation. Each path starts with fresh recurrent/conv state.
"""

import argparse
import hashlib
import json
from pathlib import Path
from unittest import mock

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp import modeling_qwen4_exp as hf_source

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights


def metrics(expected, actual):
    actual = actual.reshape(expected.shape)
    left = expected.float().reshape(-1, expected.shape[-1])
    right = actual.float().reshape_as(left)
    a, b = left - left.mean(-1, keepdim=True), right - right.mean(-1, keepdim=True)
    pcc = (a * b).sum(-1) / (a.square().sum(-1) * b.square().sum(-1)).sqrt()
    flat_a, flat_b = left.flatten(), right.flatten()
    flat_a, flat_b = flat_a - flat_a.mean(), flat_b - flat_b.mean()
    global_pcc = (flat_a * flat_b).sum() / (flat_a.square().sum() * flat_b.square().sum()).sqrt()
    error = (left - right).abs()
    equal = (expected.bfloat16() == actual.bfloat16()).reshape_as(left).float()
    finite = pcc[torch.isfinite(pcc)]
    return {
        "shape": list(expected.shape),
        "global_pcc": global_pcc.item() if torch.isfinite(global_pcc) else None,
        "minimum_row_pcc": finite.min().item() if finite.numel() else None,
        "per_row_pcc": [v.item() if torch.isfinite(v) else None for v in pcc],
        "nonfinite_pcc_rows": torch.where(~torch.isfinite(pcc))[0].tolist(),
        "maximum_absolute_error": error.max().item(),
        "per_row_maximum_absolute_error": error.max(-1).values.tolist(),
        "bf16_equal_fraction": equal.mean().item(),
        "per_row_bf16_equal_fraction": equal.mean(-1).tolist(),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fp32-residual-write",
        action="store_true",
        help="Diagnostic-only FP32 multiply/add with separate BF16 rounding at each HF boundary",
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(932)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    weights = load_weights(args.snapshot, 0, "")
    with torch.device("meta"):
        reference = hf_source.Qwen4ExpTextDecoderLayer(config, 0)
    reference.load_state_dict(weights, assign=True)
    reference.eval()
    hidden = (torch.randn(1, 65, 10240) * 0.02).bfloat16()
    other = (torch.randn(1, 7, 10240) * 0.02).bfloat16()
    requests = {"primary_first32": hidden[:, :32].contiguous(), "independent7": other}
    captured = {}

    def attn_read(_module, _inputs, output):
        captured["attn_mixed"] = output[0].clone()
        captured["attn_injection"] = output[2].clone()

    def attention(_module, _inputs, output):
        captured["gdn_output"] = output.clone()

    def norm(_module, inputs, output):
        captured["gdn_norm_input"] = inputs[0].clone().reshape(1, -1, 6144)
        captured["gdn_norm_gate"] = inputs[1].clone().reshape(1, -1, 6144)
        captured["gdn_norm_output"] = output.clone().reshape(1, -1, 6144)

    def mlp_read(_module, inputs, output):
        captured["post_attention"] = inputs[0].clone()
        captured["mlp_mixed"] = output[0].clone()
        captured["mlp_injection"] = output[2].clone()

    def gate(_module, inputs, output):
        captured["router_input"] = inputs[0].clone()
        for name, value in zip(("router_logits", "router_scores", "router_indices"), output):
            captured[name] = value.clone()

    hooks = [
        reference.attn_hyper_connection.register_forward_hook(attn_read),
        reference.linear_attn.register_forward_hook(attention),
        reference.linear_attn.norm.register_forward_hook(norm),
        reference.mlp_hyper_connection.register_forward_hook(mlp_read),
        reference.mlp.gate.register_forward_hook(gate),
    ]
    references = {}
    for name, value in requests.items():
        captured = {}
        reference(value, position_embeddings=(None, None), past_key_values=DynamicCache(config=config))
        references[name] = captured
    for hook in hooks:
        hook.remove()

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt import decoder_components as components
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import (
        GatedDeltaNet,
        GatedResidual,
        WeightLoader,
        linear,
    )

    report = {
        "scope": "Instrumented upstream component and exact-HF injection diagnostics; no measured runtime claim",
        "seed": 932,
        "layer": 0,
        "cpu_threads": 8,
        "weights": "real",
        "residual_write_policy": (
            "test_local_fp32_multiply_to_bf16_then_fp32_add_to_bf16"
            if args.fp32_residual_write
            else "unmodified_runtime"
        ),
        "snapshot": str(args.snapshot),
        "hf_source_sha256": hashlib.sha256(Path(hf_source.__file__).read_bytes()).hexdigest(),
        "requests": {},
    }
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        loader = WeightLoader(weights, mesh)
        attn_mix = GatedResidual(loader, "attn_hyper_connection.", config)
        gdn = GatedDeltaNet(loader, config)
        mlp_mix = GatedResidual(loader, "mlp_hyper_connection.", config)
        router = loader.matrix("mlp.gate.weight", ttnn.float32)

        def residual_write(component, residual, block, injection):
            if not args.fp32_residual_write:
                return component.write(residual, block, injection)
            tokens = residual.shape[-2]
            update = ttnn.typecast(
                ttnn.multiply(ttnn.typecast(block, ttnn.float32), ttnn.typecast(injection, ttnn.float32)),
                ttnn.bfloat16,
            )
            update = ttnn.reshape(
                ttnn.permute(update, (0, 2, 1, 3)), (1, 1, tokens, component.streams * component.width)
            )
            return ttnn.typecast(
                ttnn.add(ttnn.typecast(residual, ttnn.float32), ttnn.typecast(update, ttnn.float32)),
                ttnn.bfloat16,
            )

        def run_attention(value):
            conv = ttnn.zeros((1, 1, 3, 10240), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            state = ttnn.zeros((1, 48, 128, 128), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
            internals = {}
            original_linear = components.linear
            original_recurrence = components.fp32_recurrent_gated_delta_rule

            def capture_linear(x, weight, compute):
                output = original_linear(x, weight, compute)
                if weight is gdn.z:
                    internals["gdn_norm_gate"] = ttnn.to_torch(output).reshape(1, -1, 6144)
                elif weight is gdn.out:
                    internals["gdn_norm_output"] = ttnn.to_torch(x).reshape(1, -1, 6144)
                return output

            def capture_recurrence(*values, **kwargs):
                output, final_state = original_recurrence(*values, **kwargs)
                internals["gdn_norm_input"] = ttnn.to_torch(output).bfloat16().reshape(1, -1, 6144)
                return output, final_state

            # Instrumented host readbacks deliberately run outside any measured
            # or traced pass; wrapped functions retain their original results.
            with (
                mock.patch.object(components, "linear", capture_linear),
                mock.patch.object(components, "fp32_recurrent_gated_delta_rule", capture_recurrence),
            ):
                result = gdn(value, conv_state=conv, recurrent_state=state)
            ttnn.deallocate(conv)
            ttnn.deallocate(state)
            return result, internals

        for request, hidden_states in requests.items():
            expected = references[request]
            tokens = hidden_states.shape[1]
            x = loader.tensor(hidden_states.unsqueeze(0))
            exact = {}
            for name in ("attn_mixed", "attn_injection", "gdn_output", "post_attention", "mlp_mixed", "mlp_injection"):
                value = expected[name]
                if name.endswith("injection"):
                    value = value.unsqueeze(-1).permute(0, 2, 1, 3)
                else:
                    value = value.unsqueeze(0)
                exact[name] = loader.tensor(value)

            def host(name, value):
                result = ttnn.to_torch(value)
                if name.endswith("injection"):
                    result = result.permute(0, 2, 1, 3)
                return result.reshape(expected[name].shape)

            request_report = {"tokens": tokens, "variants": {}}
            report["requests"][request] = request_report
            variants = (
                "baseline",
                "exact_attn_mixed",
                "exact_attn_injection",
                "exact_attn_read_both",
                "exact_gdn_output",
                "exact_gdn_and_attn_injection",
                "exact_post_attention",
                "exact_mlp_mixed",
            )
            for variant in variants:
                mixed, injection = attn_mix.read(x)
                if variant in ("exact_attn_mixed", "exact_attn_read_both"):
                    mixed = exact["attn_mixed"]
                if variant in ("exact_attn_injection", "exact_attn_read_both", "exact_gdn_and_attn_injection"):
                    injection = exact["attn_injection"]
                block, gdn_internals = run_attention(mixed)
                if variant in ("exact_gdn_output", "exact_gdn_and_attn_injection"):
                    block = exact["gdn_output"]
                updated = residual_write(attn_mix, x, block, injection)
                if variant == "exact_post_attention":
                    updated = exact["post_attention"]
                router_input, mlp_injection = mlp_mix.read(updated)
                if variant == "exact_mlp_mixed":
                    router_input = exact["mlp_mixed"]
                counterparts = {
                    "attn_mixed": mixed,
                    "attn_injection": injection,
                    "gdn_output": block,
                    "post_attention": updated,
                    "mlp_mixed": router_input,
                    "mlp_injection": mlp_injection,
                }
                result = {
                    "components": {
                        name: metrics(expected[name], host(name, value)) for name, value in counterparts.items()
                    }
                }
                result["gdn_internal_before_output_override"] = {
                    name: metrics(expected[name], actual) for name, actual in gdn_internals.items()
                }
                logits = linear(ttnn.typecast(router_input, ttnn.float32), router, loader.compute)
                result["router"] = {}
                expected_sets = expected["router_indices"].sort(-1).values
                for rounding in (False, True):
                    routed = ttnn.typecast(ttnn.typecast(logits, ttnn.bfloat16), ttnn.float32) if rounding else logits
                    _, indices = ttnn.topk(routed, k=10, dim=-1, sorted=True)
                    actual_logits = ttnn.to_torch(routed).reshape(tokens, 512)
                    actual_indices = ttnn.to_torch(indices).reshape(tokens, 10).long()
                    changed = torch.where((actual_indices.sort(-1).values != expected_sets).any(-1))[0].tolist()
                    details = []
                    for row in sorted(set(changed + ([1] if request == "primary_first32" else [0, 2]))):
                        ids = torch.unique(torch.cat((expected["router_indices"][row], actual_indices[row]))).long()
                        details.append(
                            {
                                "row": row,
                                "hf_indices": expected["router_indices"][row].tolist(),
                                "tt_indices": actual_indices[row].tolist(),
                                "expert_ids": ids.tolist(),
                                "hf_logits_at_experts": expected["router_logits"][row, ids].float().tolist(),
                                "tt_logits_at_experts": actual_logits[row, ids].float().tolist(),
                            }
                        )
                    result["router"]["bf16_rounded" if rounding else "runtime_fp32"] = {
                        "logits": metrics(expected["router_logits"], actual_logits),
                        "expert_set_changed_rows": changed,
                        "details": details,
                    }
                # Setup-only host counterfactual separates upstream input error
                # from device router matmul/topk; it is never a runtime fallback.
                hf_on_tt = reference.mlp.gate(host("mlp_mixed", router_input).reshape(-1, 2560))
                result["hf_router_on_tt_input_changed_rows"] = torch.where(
                    (hf_on_tt[2].sort(-1).values != expected_sets).any(-1)
                )[0].tolist()
                request_report["variants"][variant] = result
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "request": request,
                            "variant": variant,
                            "router_changed_rows": result["router"]["runtime_fp32"]["expert_set_changed_rows"],
                            "hf_router_on_tt_input_changed_rows": result["hf_router_on_tt_input_changed_rows"],
                        }
                    ),
                    flush=True,
                )
    finally:
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
