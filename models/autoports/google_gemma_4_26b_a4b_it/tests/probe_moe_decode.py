# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Focused same-input routing and sparse-expert fidelity controls."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import MODEL, REVISION, load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tt.decode_attention import DecodeAttention
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.common.utility_functions import comp_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attention-fidelity", choices=["default", "hifi4", "hifi4_fp32", "hifi2_fp32"], default="default"
    )
    parser.add_argument("--projection", choices=["both", "qkv", "o"], default="both")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION).text_config
    config._attn_implementation = "eager"
    hf = load_layer(config, 5, True)
    x = torch.randn(1, 33, config.hidden_size).bfloat16().float()
    cos, sin = Gemma4TextRotaryEmbedding(config)(x, torch.arange(192)[None], layer_type="full_attention")
    hf_cache = DynamicCache()
    mask = torch.zeros(33, 33).masked_fill(torch.ones(33, 33, dtype=torch.bool).triu(1), float("-inf"))[None, None]
    with torch.no_grad():
        hf(x, position_embeddings=(cos[:, :33], sin[:, :33]), attention_mask=mask, past_key_values=hf_cache)
    dx = torch.randn(1, 1, config.hidden_size).bfloat16().float()
    hf_router = {}

    def router_hook(module, inputs, output):
        hf_router["input"] = inputs[0].detach().clone()
        hf_router["values"] = output[1].detach().clone()
        hf_router["indices"] = output[2].detach().clone()

    hf.router.register_forward_hook(router_hook)
    with torch.no_grad():
        reference = hf(
            dx,
            position_embeddings=(cos[:, 33:34], sin[:, 33:34]),
            attention_mask=torch.zeros(1, 1, 1, 34),
            past_key_values=hf_cache,
        )
    original_router = dict(hf_router)
    results = {}
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:

        def device(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value, device=mesh, dtype=dtype, layout=layout)

        def host(value):
            return ttnn.to_torch(value).float()

        def compare(name, expected, actual):
            passed, pcc = comp_pcc(expected.reshape(-1), actual.reshape(-1), 0.995)
            results[name] = dict(pcc=float(pcc), passed=bool(passed))
            print(name, results[name], flush=True)

        decoder = FunctionalDecoder.from_state_dict(hf.state_dict(), hf_config=config, layer_idx=5, mesh_device=mesh)
        layer = decoder.layer
        # Preserve the original baseline so every control remains reproducible
        # after the passing precision policy is installed by the decoder.
        if isinstance(layer.self_attn, DecodeAttention):
            layer.self_attn = layer.self_attn.source
        layer.self_attn.weights = replace(layer.self_attn.weights, wqkv=layer.self_attn.weights.wqkv.weight)
        layer.moe.router = layer.moe.router.source
        if args.attention_fidelity != "default":

            class AttentionPrecision:
                def __init__(self, wrapped):
                    self.wrapped = wrapped

                def __getattr__(self, name):
                    return getattr(self.wrapped, name)

                def __call__(self, *a, **kw):
                    original = ttnn.linear
                    compute = ttnn.init_device_compute_kernel_config(
                        mesh.arch(),
                        math_fidelity=(
                            ttnn.MathFidelity.HiFi2
                            if args.attention_fidelity == "hifi2_fp32"
                            else ttnn.MathFidelity.HiFi4
                        ),
                        math_approx_mode=False,
                        fp32_dest_acc_en="fp32" in args.attention_fidelity,
                        packer_l1_acc=False,
                    )

                    def linear(*a, **kw):
                        if args.projection == "both" or a[1] is getattr(
                            self.wrapped.weights, "wqkv" if args.projection == "qkv" else "o_proj"
                        ):
                            kw["compute_kernel_config"] = compute
                        return original(*a, **kw)

                    ttnn.linear = linear
                    try:
                        return self.wrapped(*a, **kw)
                    finally:
                        ttnn.linear = original

            layer.self_attn = AttentionPrecision(layer.self_attn)
        table = device(torch.arange(5, -1, -1, dtype=torch.int32)[None], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        cache = [
            device(torch.zeros(6, layer.self_attn.config.num_key_value_heads, 32, layer.self_attn.config.head_dim))
            for _ in range(2)
        ]
        decoder.prefill_forward(
            device(x[None]), rope_mats=tuple(device(v[None]) for v in (cos, sin)), page_table=table, kv_cache=cache
        )
        positions = torch.zeros(1, 32, dtype=torch.int32)
        positions[0, 0] = 33
        dt = device(dx[None])
        attention = layer.self_attn(
            layer.input_layernorm.forward(dt),
            rope_mats=tuple(device(v.squeeze(0)) for v in (cos, sin)),
            position_idx=device(positions, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            position_idx_cache=device(torch.tensor([33], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            page_table=table,
            kv_cache=cache,
            is_decode=True,
        )
        residual = ttnn.add(dt, layer.post_attention_layernorm.forward(attention))
        expert_input = layer.pre_feedforward_layernorm_2.forward(residual)
        shared = layer.post_feedforward_layernorm_1.forward(
            layer.shared_mlp(layer.pre_feedforward_layernorm.forward(residual))
        )
        routing = layer.moe.router(residual)
        cpu_routing = host(routing).reshape(1, -1)
        cpu_input = host(expert_input).reshape(1, -1)
        with torch.no_grad():
            _, oracle_values, oracle_indices = hf.router(host(residual).reshape(1, -1))
            oracle_dense = torch.zeros_like(cpu_routing).scatter(-1, oracle_indices, oracle_values)
            oracle_experts = hf.experts(cpu_input, oracle_indices, oracle_values)
            tt_indices = cpu_routing.nonzero()[:, 1][None]
            tt_values = cpu_routing.gather(-1, tt_indices)
            same_routes_reference = hf.experts(cpu_input, tt_indices, tt_values)
        results["routes"] = dict(
            tt_ids=tt_indices.tolist(),
            oracle_ids=oracle_indices.tolist(),
            intersection=len(set(tt_indices[0].tolist()) & set(oracle_indices[0].tolist())),
        )
        print("routes", results["routes"], flush=True)
        compare("routing_weights", oracle_dense, cpu_routing)
        compare("residual", original_router["input"], host(residual))
        results["hf_original_ids"] = original_router["indices"].tolist()
        with torch.no_grad():
            for label, value in [
                ("original", original_router["input"]),
                ("bf16_residual", original_router["input"].bfloat16().float()),
                ("tt_residual", host(residual).reshape(1, -1)),
            ]:
                probs, vals, ids = hf.router(value)
                scores = hf.router.proj(hf.router.norm(value) * hf.router.scale * hf.router.scalar_root_size)
                top = scores.topk(10)
                results[label + "_router"] = dict(
                    ids=ids.tolist(), top10_indices=top.indices.tolist(), top10_scores=top.values.tolist()
                )
                print(label + "_router", results[label + "_router"], flush=True)

        def finish(raw):
            routed = layer.post_feedforward_layernorm_2.forward(raw)
            combined = layer.post_feedforward_layernorm.forward(ttnn.add(shared, routed))
            return ttnn.mul(ttnn.add(residual, combined), layer.layer_scalar)

        baseline = layer.moe.experts(expert_input, routing)
        compare("baseline_experts_same_routes", same_routes_reference, host(baseline))
        compare("baseline_final", reference, host(finish(baseline)))
        oracle_device = device(oracle_dense.reshape(1, 1, 1, -1))
        oracle_raw = layer.moe.experts(expert_input, oracle_device)
        compare("oracle_routing_experts", oracle_experts, host(oracle_raw))
        compare("oracle_routing_final", reference, host(finish(oracle_raw)))
        compare("exact_expert_final", reference, host(finish(device(oracle_experts.reshape(1, 1, 1, -1)))))
        original_dense = torch.zeros_like(cpu_routing).scatter(
            -1, original_router["indices"], original_router["values"]
        )
        original_raw = layer.moe.experts(expert_input, device(original_dense.reshape(1, 1, 1, -1)))
        compare("original_hf_routing_final", reference, host(finish(original_raw)))

        router = layer.moe.router
        scaled = ttnn.mul(ttnn.mul(router.norm.forward(residual), router.scale), router.scalar_root_size)
        fp32_compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        float_residual = ttnn.typecast(residual, ttnn.float32)
        float_norm = ttnn.rms_norm(float_residual, epsilon=config.rms_norm_eps, compute_kernel_config=fp32_compute)
        float_scaled = ttnn.mul(ttnn.mul(float_norm, router.scale), router.scalar_root_size)
        for label, projection_input, kwargs in [
            ("logits_topk", scaled, {}),
            ("fp32_logits_topk", scaled, dict(dtype=ttnn.float32, compute_kernel_config=fp32_compute)),
            ("fp32_router", float_scaled, dict(dtype=ttnn.float32, compute_kernel_config=fp32_compute)),
            ("fp32_router_softmax", float_scaled, dict(dtype=ttnn.float32, compute_kernel_config=fp32_compute)),
        ]:
            scores = ttnn.linear(projection_input, router.proj_weight, **kwargs)
            if label == "fp32_router_softmax":
                values, indices = ttnn.topk(ttnn.softmax(scores, dim=-1), k=router.top_k, dim=-1)
            else:
                values, indices = ttnn.topk(scores, k=router.top_k, dim=-1)
                values = ttnn.exp(ttnn.subtract(values, ttnn.max(values, dim=-1, keepdim=True)))
            values = ttnn.div(values, ttnn.sum(values, dim=-1, keepdim=True))
            candidate = ttnn.scatter(
                ttnn.zeros_like(routing), dim=-1, index=indices, src=ttnn.typecast(values, ttnn.bfloat16)
            )
            candidate = ttnn.mul(candidate, router.per_expert_scale)
            candidate = ttnn.typecast(candidate, ttnn.bfloat16)
            results[label + "_ids"] = host(indices).long().tolist()
            top = host(scores).flatten().topk(10)
            results[label + "_scores"] = dict(indices=top.indices.tolist(), scores=top.values.tolist())
            print(label + "_scores", results[label + "_scores"], flush=True)
            compare(label + "_routing", oracle_dense, host(candidate))
            compare(label + "_final", reference, host(finish(layer.moe.experts(expert_input, candidate))))

        original_sparse_matmul = ttnn.sparse_matmul
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        def hifi4_matmul(*args, **kwargs):
            kwargs["compute_kernel_config"] = compute
            return original_sparse_matmul(*args, **kwargs)

        try:
            ttnn.sparse_matmul = hifi4_matmul
            high_fidelity = layer.moe.experts(expert_input, routing)
        finally:
            ttnn.sparse_matmul = original_sparse_matmul
        compare("hifi4_experts_same_routes", same_routes_reference, host(high_fidelity))
        compare("hifi4_final", reference, host(finish(high_fidelity)))
        print("MEMORY_APIS", [name for name in dir(mesh) if "memory" in name or "device" in name], flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    Path(
        f"models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/decode_moe_probe_{args.attention_fidelity}_{args.projection}.json"
    ).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
