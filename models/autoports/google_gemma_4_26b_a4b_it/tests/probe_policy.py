# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run an existing parity harness with one explicit diagnostic precision policy."""

import argparse
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import MethodType

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_fp32_attention import PrecisePagedAttention
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.demos.gemma4.tt.dram_sharded import DramShardedLinear


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--qkv-fidelity", choices=["hifi2", "hifi4"], required=True)
    parser.add_argument("--sdpa-compute", choices=["default", "exact", "fp32"], required=True)
    parser.add_argument("--runner", choices=["run_decoder", "request_reuse", "batched"], default="run_decoder")
    parser.add_argument("--input-norm-fp32", action="store_true")
    parser.add_argument("--norm-all-phases", action="store_true")
    parser.add_argument("--head-fp32", action="store_true")
    parser.add_argument("--precise-attention", action="store_true")
    parser.add_argument("--residual-fp32", action="store_true")
    parser.add_argument("--oracle-prefix", action="store_true")
    parser.add_argument("--rope-fp32", action="store_true")
    parser.add_argument("--attention-output-fp32", action="store_true")
    parser.add_argument("--router-report", type=Path)
    parser.add_argument("--head-norm-exact", action="store_true")
    parser.add_argument("--rope-exact", action="store_true")
    parser.add_argument("--rope-elementwise", action="store_true")
    parser.add_argument("--router-logit-topk", action="store_true")
    parser.add_argument("--composite-norm", action="store_true")
    parser.add_argument("--split-linear", choices=["qkv", "o", "router", "qkv_router", "all"])
    parser.add_argument("--split-attention", action="store_true")
    parser.add_argument("--oracle-sdpa", action="store_true")
    parser.add_argument("--oracle-qkv", action="store_true")
    parser.add_argument("--cast-cache-update", action="store_true")
    parser.add_argument("--decode-only-front", action="store_true")
    parser.add_argument("--router-sfpu", action="store_true")
    parser.add_argument("--router-scale-fp32", action="store_true")
    args, remaining = parser.parse_known_args()
    original = FunctionalDecoder.from_state_dict.__func__
    precise = None
    oracle_host = None
    oracle_device = None
    hf_router = None
    hf_residual = None
    router_rows = []
    hf_stages = {}
    tt_stages = {}
    hf_layer = None
    hf_rope = None
    full_cache = None
    attention_stages = {}
    oracle_sdpa = []
    attention_calls = 0

    def capture_tt(name, value):
        if args.router_report and value.shape[-2] == 1 and len(router_rows) < hf_residual.shape[0]:
            tt_stages[name] = ttnn.to_torch(value).float().reshape(-1)

    def correlation(actual, expected):
        return float(torch.corrcoef(torch.stack((actual.flatten().double(), expected.flatten().double())))[0, 1])

    def norm_op(value, *, weight=None, epsilon, compute_kernel_config=None, memory_config=None):
        if not args.composite_norm:
            return ttnn.rms_norm(
                value,
                weight=weight,
                epsilon=epsilon,
                compute_kernel_config=compute_kernel_config,
                memory_config=memory_config,
            )
        value = ttnn.typecast(value, ttnn.float32)
        squared = ttnn.mul(value, value)
        mean = ttnn.mean(squared, dim=-1, keepdim=True)
        scale = ttnn.rsqrt(ttnn.add(mean, epsilon), fast_and_approximate_mode=False)
        output = ttnn.mul(value, scale)
        if weight is not None:
            weight = ttnn.to_layout(
                ttnn.typecast(ttnn.reshape(weight, (1, 1, 1, value.shape[-1])), ttnn.float32), ttnn.TILE_LAYOUT
            )
            output = ttnn.mul(output, weight)
        return output

    def linear(value, weight, *, group, **kw):
        if args.router_sfpu and group == "router" and value.shape[-2] == 1:
            matrix = ttnn.transpose(ttnn.typecast(weight, ttnn.float32), -2, -1)
            repeated = ttnn.repeat(value, (1, 1, matrix.shape[-2], 1))
            products = ttnn.mul(repeated, matrix)
            return ttnn.transpose(ttnn.sum(products, dim=-1, keepdim=True), -2, -1)
        if group not in (args.split_linear or "").split("_") and args.split_linear != "all":
            return ttnn.linear(value, weight, **kw)
        high = ttnn.typecast(value, ttnn.bfloat16)
        low = ttnn.typecast(
            ttnn.subtract(ttnn.typecast(value, ttnn.float32), ttnn.typecast(high, ttnn.float32)), ttnn.bfloat16
        )
        return ttnn.add(ttnn.linear(high, weight, **kw), ttnn.linear(low, weight, **kw))

    def factory(cls, *a, **kw):
        nonlocal precise, oracle_device
        result = original(cls, *a, **kw)
        attention = result.layer.self_attn
        if args.oracle_prefix:
            assert oracle_host is not None
            oracle_device = []
            for tensor in oracle_host:
                padded = torch.nn.functional.pad(tensor, (0, 0, 0, (-tensor.shape[-2]) % 32))
                oracle_device.append(
                    [
                        ttnn.from_torch(
                            slot[None], device=attention.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                        )
                        for slot in padded
                    ]
                )
        if args.precise_attention:
            precise = PrecisePagedAttention(
                attention.mesh_device,
                attention.config,
                result.config.max_position_embeddings,
                output_dtype=ttnn.float32 if args.attention_output_fp32 else ttnn.bfloat16,
                split_inputs=args.split_attention,
            )
            if args.router_report:

                def observer(q, k, v, out):
                    nonlocal attention_calls
                    index = attention_calls % hf_residual.shape[0]
                    attention_calls += 1
                    if len(router_rows) >= hf_residual.shape[0]:
                        return oracle_sdpa[index] if args.oracle_sdpa else None
                    slot = len(router_rows)
                    q, k, v, out = [ttnn.to_torch(t).float() for t in (q, k, v, out)]
                    length = full_cache[0].shape[-2]
                    from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

                    reference_q = apply_rotary_pos_emb(
                        hf_stages["self_attn.q_norm"][slot : slot + 1], *hf_rope, unsqueeze_dim=2
                    )
                    attention_stages["q_after_rope"] = correlation(q, reference_q)
                    rounded_rope = tuple(t.bfloat16().float() for t in hf_rope)
                    same_table_q = apply_rotary_pos_emb(
                        hf_stages["self_attn.q_norm"][slot : slot + 1], *rounded_rope, unsqueeze_dim=2
                    )
                    same_table_k = (
                        apply_rotary_pos_emb(
                            hf_stages["self_attn.k_norm"][slot : slot + 1], *rounded_rope, unsqueeze_dim=2
                        )
                        .transpose(1, 2)
                        .bfloat16()
                        .float()
                    )
                    expected_v = hf_stages["self_attn.v_norm"][slot : slot + 1].transpose(1, 2).bfloat16().float()
                    attention_stages["q_same_bf16_table"] = correlation(q, same_table_q)
                    attention_stages["current_k_same_bf16_table"] = correlation(
                        k[:, :, length - 1 : length], same_table_k
                    )
                    attention_stages["current_v_quantized_hf"] = correlation(v[:, :, length - 1 : length], expected_v)
                    attention_stages["current_v_max_error"] = float(
                        (v[:, :, length - 1 : length] - expected_v).abs().max()
                    )
                    attention_stages["cached_k"] = correlation(k[:, :, :length], full_cache[0][slot : slot + 1])
                    attention_stages["cached_v"] = correlation(v[:, :, :length], full_cache[1][slot : slot + 1])
                    cpu_attention = torch.nn.functional.scaled_dot_product_attention(
                        q.transpose(1, 2), k[:, :, :length], v[:, :, :length], enable_gqa=True, scale=1.0
                    ).transpose(1, 2)
                    attention_stages["sdpa_same_qkv"] = correlation(out, cpu_attention)
                    tt_stages["attention_input"] = out.flatten()
                    if args.oracle_sdpa:
                        oracle_sdpa.append(
                            ttnn.from_torch(
                                cpu_attention, device=attention.mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
                            )
                        )
                        tt_stages["attention_input"] = cpu_attention.flatten()
                        return oracle_sdpa[-1]

                precise.observer = observer
        attention.source.weights.wqkv.compute = ttnn.init_device_compute_kernel_config(
            attention.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4 if args.qkv_fidelity == "hifi4" else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        attention.compute = ttnn.init_device_compute_kernel_config(
            attention.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=args.sdpa_compute != "exact",
            fp32_dest_acc_en=args.sdpa_compute == "fp32",
            packer_l1_acc=False,
        )
        if args.input_norm_fp32:
            input_norm = result.layer.input_layernorm
            original_forward = input_norm.forward
            exact = ttnn.init_device_compute_kernel_config(
                attention.mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )

            def norm(value):
                if value.shape[-2] != 1 and not args.norm_all_phases:
                    return original_forward(value)
                return norm_op(
                    ttnn.typecast(value, ttnn.float32),
                    weight=input_norm.tt_weight,
                    epsilon=input_norm.eps,
                    compute_kernel_config=exact,
                )

            input_norm.forward = norm
        if args.input_norm_fp32 or args.head_fp32:
            if args.oracle_qkv:
                assert args.router_report
                names = ["q", "k", "k"] if attention.config.use_kv_tying else ["q", "k", "v"]
                values = torch.cat([hf_stages["self_attn." + name + "_proj"] for name in names], dim=-1)
                oracle_qkv = [
                    ttnn.from_torch(
                        row[None, None], device=attention.mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
                    )
                    for row in values
                ]

            class Projection(DramShardedLinear):
                def __init__(self, source):
                    self.source = source
                    self.calls = 0

                def __call__(self, value, out_memory_config=None):
                    if args.decode_only_front and value.shape[-2] != 1:
                        out = self.source(value, out_memory_config=out_memory_config)
                    elif args.oracle_qkv and value.shape[-2] == 1:
                        out = ttnn.clone(oracle_qkv[self.calls % len(oracle_qkv)])
                        self.calls += 1
                    else:
                        out = linear(
                            value,
                            self.source.weight,
                            group="qkv",
                            dtype=ttnn.float32 if args.head_fp32 else ttnn.bfloat16,
                            compute_kernel_config=self.source.compute,
                            memory_config=out_memory_config,
                        )
                    capture_tt("qkv", out)
                    return out

            attention.source.weights = replace(attention.source.weights, wqkv=Projection(attention.source.weights.wqkv))
        if args.residual_fp32:
            compute = ttnn.init_device_compute_kernel_config(
                attention.mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )

            class OutputProjection(DramShardedLinear):
                def __init__(self, weight):
                    self.weight = weight

                def __call__(self, value):
                    if args.decode_only_front and value.shape[-2] != 1:
                        return ttnn.linear(value, self.weight)
                    return linear(value, self.weight, group="o", dtype=ttnn.float32, compute_kernel_config=compute)

            attention.source.weights = replace(
                attention.source.weights, o_proj=OutputProjection(attention.source.weights.o_proj)
            )
            post_attention_norm = result.layer.post_attention_layernorm
            original_post_norm = post_attention_norm.forward

            def post_norm(value):
                if args.decode_only_front and value.shape[-2] != 1:
                    return original_post_norm(value)
                return norm_op(
                    ttnn.typecast(value, ttnn.float32),
                    weight=post_attention_norm.tt_weight,
                    epsilon=post_attention_norm.eps,
                    compute_kernel_config=compute,
                )

            result.layer.post_attention_layernorm.forward = post_norm
            original_decoder_forward = result._forward

            def forward(self, x, **attention_kwargs):
                if args.decode_only_front and not attention_kwargs.get("is_decode", True):
                    return original_decoder_forward(x, **attention_kwargs)
                layer = self.layer
                attn = layer.self_attn(layer.input_layernorm.forward(x), **attention_kwargs)
                residual = ttnn.add(ttnn.typecast(x, ttnn.float32), layer.post_attention_layernorm.forward(attn))
                shared = layer.shared_mlp(
                    ttnn.typecast(layer.pre_feedforward_layernorm.forward(residual), ttnn.bfloat16)
                )
                shared = layer.post_feedforward_layernorm_1.forward(shared)
                expert_input = ttnn.typecast(layer.pre_feedforward_layernorm_2.forward(residual), ttnn.bfloat16)
                routed = layer.post_feedforward_layernorm_2.forward(layer.moe(residual, expert_input))
                combined = layer.post_feedforward_layernorm.forward(ttnn.add(shared, routed))
                return ttnn.typecast(ttnn.mul(ttnn.add(residual, combined), layer.layer_scalar), ttnn.bfloat16)

            result._forward = MethodType(forward, result)
        if args.router_logit_topk:
            router_source = result.layer.moe.router
            router_scale = (
                ttnn.typecast(router_source.source.scale, ttnn.float32)
                if args.router_scale_fp32
                else router_source.source.scale
            )

            class LogitRouter:
                def __getattr__(self, name):
                    return getattr(router_source, name)

                def __call__(self, value):
                    source = router_source.source
                    normed = norm_op(
                        ttnn.typecast(value, ttnn.float32),
                        epsilon=router_source.epsilon,
                        compute_kernel_config=router_source.compute,
                    )
                    scaled = ttnn.mul(ttnn.mul(normed, router_scale), source.scalar_root_size)
                    scores = linear(
                        scaled,
                        source.proj_weight,
                        group="router",
                        dtype=ttnn.float32,
                        compute_kernel_config=router_source.compute,
                    )
                    selected, indices = ttnn.topk(scores, k=source.top_k, dim=-1)
                    probabilities = ttnn.softmax(selected, dim=-1)
                    routed = ttnn.scatter(
                        ttnn.zeros_like(ttnn.typecast(scores, ttnn.bfloat16)),
                        dim=-1,
                        index=indices,
                        src=ttnn.typecast(probabilities, ttnn.bfloat16),
                    )
                    return ttnn.mul(routed, source.per_expert_scale)

            result.layer.moe.router = LogitRouter()
        if args.router_report:
            for name in ["input_layernorm", "post_attention_layernorm"]:
                module = getattr(result.layer, name)
                module_forward = module.forward

                def record(value, name=name, module_forward=module_forward):
                    out = module_forward(value)
                    capture_tt(name, out)
                    return out

                module.forward = record
            base_attention = result.layer.self_attn

            class AttentionProbe:
                def __getattr__(self, name):
                    return getattr(base_attention, name)

                def __call__(self, value, **kw):
                    out = base_attention(value, **kw)
                    capture_tt("self_attn", out)
                    return out

            result.layer.self_attn = AttentionProbe()
            base_router = result.layer.moe.router

            class RouterProbe:
                def __call__(self, value):
                    routed = base_router(value)
                    if value.shape[-2] == 1 and len(router_rows) < hf_residual.shape[0]:
                        slot = len(router_rows)
                        residual = ttnn.to_torch(value).float().reshape(1, -1)
                        actual = ttnn.to_torch(routed).float().reshape(-1)
                        with torch.no_grad():
                            oracle = hf_router(hf_residual[slot : slot + 1])
                            same = hf_router(residual)
                        stages = {
                            name: correlation(value, hf_stages[name][slot])
                            for name, value in tt_stages.items()
                            if name not in ["qkv", "attention_input"]
                        }
                        stages.update(attention_stages)
                        if "attention_input" in tt_stages:
                            with torch.no_grad():
                                cpu_o = hf_layer.self_attn.o_proj(tt_stages["attention_input"])
                            stages["o_same_input"] = correlation(tt_stages["self_attn"], cpu_o)
                        router_detail = {}
                        if set(torch.nonzero(actual).flatten().tolist()) != set(same[2].flatten().tolist()):
                            source = base_router.source
                            normal_tt = norm_op(
                                ttnn.typecast(value, ttnn.float32),
                                epsilon=base_router.epsilon,
                                compute_kernel_config=base_router.compute,
                            )
                            scaled_tt = ttnn.mul(
                                ttnn.mul(normal_tt, router_scale if args.router_logit_topk else source.scale),
                                source.scalar_root_size,
                            )
                            scores_tt = linear(
                                scaled_tt,
                                source.proj_weight,
                                group="router",
                                dtype=ttnn.float32,
                                compute_kernel_config=base_router.compute,
                            )
                            normal, scaled, scores = [
                                ttnn.to_torch(t).float().reshape(1, -1) for t in (normal_tt, scaled_tt, scores_tt)
                            ]
                            with torch.no_grad():
                                norm_cpu = hf_router.norm(residual)
                                scores_cpu = hf_router.proj(scaled)
                            router_detail = dict(
                                norm_pcc=correlation(normal, norm_cpu),
                                linear_same_input_pcc=correlation(scores, scores_cpu),
                                linear_max_error=float((scores - scores_cpu).abs().max()),
                                tt_score_ranks=torch.topk(scores, 12).indices.flatten().tolist(),
                                cpu_same_scaled_ranks=torch.topk(scores_cpu, 12).indices.flatten().tolist(),
                            )
                        qkv = torch.cat(
                            [
                                hf_stages["self_attn." + name + "_proj"][slot].flatten()
                                for name in (["q", "k", "k"] if attention.config.use_kv_tying else ["q", "k", "v"])
                            ]
                        )
                        stages["qkv"] = correlation(tt_stages["qkv"], qkv)
                        router_rows.append(
                            dict(
                                slot=slot,
                                residual_pcc=correlation(residual, hf_residual[slot]),
                                stages=stages,
                                router_detail=router_detail,
                                hf_routes=oracle[2].flatten().tolist(),
                                cpu_same_residual_routes=same[2].flatten().tolist(),
                                tt_routes=torch.nonzero(actual).flatten().tolist(),
                            )
                        )
                    return routed

            result.layer.moe.router = RouterProbe()
        return result

    FunctionalDecoder.from_state_dict = classmethod(factory)
    sys.argv = [sys.argv[0], *remaining]
    originals = []
    if args.cast_cache_update:
        update_cache = ttnn.experimental.paged_update_cache

        def update(cache, value, *a, **kw):
            return update_cache(cache, ttnn.typecast(value, cache.dtype), *a, **kw)

        originals.append((ttnn.experimental, "paged_update_cache", update_cache))
        ttnn.experimental.paged_update_cache = update
    if args.rope_elementwise:
        source_rope = importlib.import_module("models.demos.gemma4.tt.attention.operations").apply_rope

        def elementwise_rope(value, cos, sin, token_index=None, memory_config=None):
            if args.decode_only_front and token_index is None:
                return source_rope(value, cos, sin, token_index, memory_config)
            if token_index is not None:
                cos = ttnn.repeat(cos[:, :, token_index : token_index + 1, :], (1, 1, value.shape[-2], 1))
                sin = ttnn.repeat(sin[:, :, token_index : token_index + 1, :], (1, 1, value.shape[-2], 1))
            else:
                cos, sin = cos[:, :, : value.shape[-2]], sin[:, :, : value.shape[-2]]
            half = value.shape[-1] // 2
            rotated = ttnn.concat((ttnn.neg(value[..., half:]), value[..., :half]), dim=-1)
            return ttnn.add(
                ttnn.mul(value, ttnn.typecast(cos, ttnn.float32)), ttnn.mul(rotated, ttnn.typecast(sin, ttnn.float32))
            )

        for name in [
            "models.demos.gemma4.tt.attention.operations",
            "models.demos.gemma4.tt.attention.prefill",
            "models.autoports.google_gemma_4_26b_a4b_it.tt.decode_attention",
        ]:
            module = importlib.import_module(name)
            originals.append((module, "apply_rope", module.apply_rope))
            module.apply_rope = elementwise_rope
    if args.rope_exact:
        original_rope = ttnn.experimental.rotary_embedding

        def exact_rope(value, *a, **kw):
            kw["compute_kernel_config"] = ttnn.init_device_compute_kernel_config(
                value.device().arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            return original_rope(value, *a, **kw)

        originals.append((ttnn.experimental, "rotary_embedding", original_rope))
        ttnn.experimental.rotary_embedding = exact_rope
    if args.head_norm_exact:
        source_head_norm = importlib.import_module("models.demos.gemma4.tt.attention.operations").apply_per_head_norm

        def head_norm(value, weight, eps, with_scale=True, memory_config=None):
            if args.decode_only_front and value.shape[1] != 1:
                return source_head_norm(value, weight, eps, with_scale, memory_config)
            shape = value.shape
            flat = ttnn.reshape(value, (1, 1, shape[0] * shape[1] * shape[2], shape[-1]))
            compute = ttnn.init_device_compute_kernel_config(
                value.device().arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            normed = norm_op(
                flat,
                weight=weight if with_scale else None,
                epsilon=eps,
                memory_config=memory_config,
                compute_kernel_config=compute,
            )
            return ttnn.reshape(normed, shape)

        for name in [
            "models.demos.gemma4.tt.attention.operations",
            "models.demos.gemma4.tt.attention.prefill",
            "models.autoports.google_gemma_4_26b_a4b_it.tt.decode_attention",
        ]:
            module = importlib.import_module(name)
            originals.append((module, "apply_per_head_norm", module.apply_per_head_norm))
            module.apply_per_head_norm = head_norm
    if args.head_fp32 or args.oracle_prefix:
        original_fill = ttnn.experimental.paged_fill_cache
        fill_count = 0

        def fill(cache, value, *a, **kw):
            nonlocal fill_count
            if args.oracle_prefix:
                value = oracle_device[fill_count % 2][kw["batch_idx"]]
                fill_count += 1
            return original_fill(cache, ttnn.typecast(value, cache.dtype), *a, **kw)

        originals.append((ttnn.experimental, "paged_fill_cache", original_fill))
        ttnn.experimental.paged_fill_cache = fill
    if args.head_fp32:
        for name in [
            "scaled_dot_product_attention",
            "chunked_scaled_dot_product_attention",
            "paged_scaled_dot_product_attention_decode",
        ]:
            operation = getattr(ttnn.transformer, name)

            def sdpa(q, k, v, *a, operation=operation, **kw):
                return operation(
                    ttnn.typecast(q, ttnn.bfloat16),
                    ttnn.typecast(k, ttnn.bfloat16),
                    ttnn.typecast(v, ttnn.bfloat16),
                    *a,
                    **kw,
                )

            originals.append((ttnn.transformer, name, operation))
            setattr(ttnn.transformer, name, sdpa)
    if args.precise_attention:
        operation = ttnn.transformer.paged_scaled_dot_product_attention_decode
        originals.append((ttnn.transformer, "paged_scaled_dot_product_attention_decode", operation))

        def precise_sdpa(*a, **kw):
            return precise(*a, **kw)

        ttnn.transformer.paged_scaled_dot_product_attention_decode = precise_sdpa
    try:
        runner = importlib.import_module(f"models.autoports.google_gemma_4_26b_a4b_it.tests.{args.runner}")
        if args.rope_fp32:
            rotary = runner.Gemma4TextRotaryEmbedding
            rotary_forward = rotary.forward
            rope_pointers = set()

            def capture_rope(*a, **kw):
                result = rotary_forward(*a, **kw)
                rope_pointers.update(t.data_ptr() for t in result)
                return result

            originals.append((rotary, "forward", rotary_forward))
            rotary.forward = capture_rope
            from_torch = ttnn.from_torch

            def upload(value, *a, **kw):
                if value.data_ptr() in rope_pointers:
                    kw["dtype"] = ttnn.float32
                return from_torch(value, *a, **kw)

            originals.append((ttnn, "from_torch", from_torch))
            ttnn.from_torch = upload
            embedding = ttnn.embedding

            def select_rope(indices, table, *a, **kw):
                if table.dtype != ttnn.float32:
                    return embedding(indices, table, *a, **kw)
                row = ttnn.repeat(indices[:, :1], (1, table.shape[-1]))
                gathered = ttnn.gather(ttnn.to_layout(table, ttnn.ROW_MAJOR_LAYOUT), dim=0, index=row)
                return ttnn.to_layout(gathered, ttnn.TILE_LAYOUT)

            originals.append((ttnn, "embedding", embedding))
            ttnn.embedding = select_rope
        if args.oracle_prefix or args.router_report:
            assert args.runner == "batched", "Oracle prefix control supports the fixed batched probe only"
            load_layer = runner.load_layer

            def load(*a, **kw):
                nonlocal hf_router, hf_layer
                hf = load_layer(*a, **kw)
                hf_layer = hf
                hf_router = hf.router
                for name, module in hf.named_modules():
                    if name in [
                        "input_layernorm",
                        "self_attn.q_proj",
                        "self_attn.k_proj",
                        "self_attn.v_proj",
                        "self_attn.q_norm",
                        "self_attn.k_norm",
                        "self_attn.v_norm",
                        "self_attn",
                        "post_attention_layernorm",
                    ]:

                        def stage_hook(module, inputs, output, name=name):
                            hf_stages[name] = (output[0] if isinstance(output, tuple) else output).detach().clone()

                        module.register_forward_hook(stage_hook)
                latest_residual = None

                def router_hook(module, inputs):
                    nonlocal latest_residual
                    latest_residual = inputs[0].detach().clone()

                hf.router.register_forward_pre_hook(router_hook)
                forward = hf.forward

                def capture(value, *a, **kw):
                    nonlocal oracle_host, hf_residual, full_cache, hf_rope
                    out = forward(value, *a, **kw)
                    if value.shape[-2] > 1:
                        cache = kw["past_key_values"].layers[hf.self_attn.layer_idx]
                        oracle_host = (cache.keys.clone(), cache.values.clone())
                    else:
                        hf_residual = latest_residual.reshape(value.shape[0], -1)
                        cache = kw["past_key_values"].layers[hf.self_attn.layer_idx]
                        full_cache = (cache.keys.clone(), cache.values.clone())
                        hf_rope = kw["position_embeddings"]
                    return out

                hf.forward = capture
                return hf

            originals.append((runner, "load_layer", load_layer))
            runner.load_layer = load
        runner.main()
    finally:
        if args.router_report:
            args.router_report.write_text(json.dumps(router_rows, indent=2) + "\n")
        FunctionalDecoder.from_state_dict = classmethod(original)
        for namespace, name, operation in reversed(originals):
            setattr(namespace, name, operation)


if __name__ == "__main__":
    main()
