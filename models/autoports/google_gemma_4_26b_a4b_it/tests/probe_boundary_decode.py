# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact request-reuse inputs with cache, attention, and routing controls."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding, apply_rotary_pos_emb

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_fp32_attention import PrecisePagedAttention
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import MODEL, REVISION, load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tt.decode_attention import DecodeAttention
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder
from models.common.utility_functions import comp_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-index", type=int, default=0)
    parser.add_argument("--qkv-fidelity", choices=["hifi2", "hifi4"], default="hifi2")
    parser.add_argument(
        "--sdpa-compute", choices=["default", "hifi2_exact", "hifi2_fp32", "hifi4_fp32", "precise"], default="default"
    )
    args = parser.parse_args()
    torch.manual_seed(19)
    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(MODEL, revision=REVISION).text_config
    cfg._attn_implementation = "sdpa"
    hf = load_layer(cfg, 0, True)
    extent = 4096
    cos, sin = Gemma4TextRotaryEmbedding(cfg)(
        torch.zeros(1, 1, cfg.hidden_size), torch.arange(extent)[None], layer_type="sliding_attention"
    )
    for length in [31, 32, 33, 1023, 1024, 1025, 2049, 33][: args.request_index + 1]:
        x = torch.randn(1, length, cfg.hidden_size).bfloat16().float()
        dx = torch.randn(1, 1, cfg.hidden_size).bfloat16().float()
        table = torch.randperm(extent // 32).int()[None]
    idx = torch.arange(length)
    allowed = (idx[:, None] >= idx[None, :]) & (idx[:, None] - idx[None, :] < cfg.sliding_window)
    mask = torch.zeros(length, length).masked_fill(~allowed, float("-inf"))[None, None]
    dm = torch.zeros(1, 1, 1, length + 1)
    dm[..., : max(0, length + 1 - cfg.sliding_window)] = float("-inf")
    hc = DynamicCache()
    with torch.no_grad():
        hf(x, position_embeddings=(cos[:, :length], sin[:, :length]), attention_mask=mask, past_key_values=hc)
    hf_stages = {}

    def hook(name):
        def record(module, inputs, output):
            hf_stages[name] = (
                output.detach().clone()
                if isinstance(output, torch.Tensor)
                else tuple(v.detach().clone() if isinstance(v, torch.Tensor) else v for v in output)
            )
            if name == "router":
                hf_stages["residual"] = inputs[0].detach().clone()

        return record

    for name, module in hf.named_modules():
        if name in [
            "input_layernorm",
            "self_attn",
            "self_attn.q_norm",
            "post_attention_layernorm",
            "router",
            "experts",
            "mlp",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ]:
            module.register_forward_hook(hook(name))
    with torch.no_grad():
        reference = hf(
            dx,
            position_embeddings=(cos[:, length : length + 1], sin[:, length : length + 1]),
            attention_mask=dm,
            past_key_values=hc,
        )
    original_hf = dict(hf_stages)
    results = dict(length=length, request_index=args.request_index)

    def compare(name, expected, actual):
        passing, pcc = comp_pcc(expected.reshape(-1).float(), actual.reshape(-1).float(), 0.995)
        results[name] = dict(pcc=float(pcc), passed=bool(passing))
        print(name, results[name], flush=True)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:

        def device(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value, dtype=dtype, layout=layout, device=mesh)

        def host(value):
            return ttnn.to_torch(value).float()

        layer = FunctionalDecoder.from_state_dict(hf.state_dict(), hf_config=cfg, layer_idx=0, mesh_device=mesh)
        ll = layer.layer
        # Make baseline controls reproducible after the production policy changes.
        if isinstance(ll.self_attn, DecodeAttention):
            ll.self_attn = ll.self_attn.source
        ll.self_attn.weights.wqkv.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4 if args.qkv_fidelity == "hifi4" else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        cache = [device(torch.zeros(extent // 32, cfg.num_key_value_heads, 32, cfg.head_dim)) for _ in range(2)]
        pt = device(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        layer.prefill_forward(
            device(x[None]), rope_mats=tuple(device(v[None]) for v in (cos, sin)), page_table=pt, kv_cache=cache
        )
        dt = device(dx[None])
        rope = tuple(device(v.squeeze(0)) for v in (cos, sin))
        p = torch.zeros(1, 32, dtype=torch.int32)
        p[0, 0] = length
        rp = device(p, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        cp = device(torch.tensor([length], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        captured = {}

        class MoEProbe:
            def __init__(self, base):
                self.base = base
                self.oracle = None

            def __call__(self, residual, expert_input):
                captured["residual"] = host(residual).reshape(1, -1)
                captured["expert_input"] = host(expert_input).reshape(1, -1)
                routing = self.base.router(residual)
                captured["routing"] = host(routing).reshape(1, -1)
                raw = self.base.experts(expert_input, routing if self.oracle is None else self.oracle)
                captured["experts"] = host(raw)
                return raw

        moe = MoEProbe(ll.moe)
        ll.moe = moe
        oracle_o = False
        for name in [
            "input_layernorm",
            "post_attention_layernorm",
            "post_feedforward_layernorm_1",
            "post_feedforward_layernorm_2",
        ]:
            module = getattr(ll, name)
            original = module.forward

            def norm(value, name=name, original=original):
                if name == "post_attention_layernorm":
                    captured["attention"] = host(value)
                    if oracle_o:
                        cpu_input = captured["used_sdpa"].transpose(1, 2).reshape(1, 1, -1)
                        cpu_output = torch.nn.functional.linear(cpu_input, hf.self_attn.o_proj.weight.detach())
                        value = device(cpu_output[None])
                result = original(value)
                captured[name] = host(result)
                return result

            module.forward = norm
        original_sdpa = ttnn.transformer.paged_scaled_dot_product_attention_decode
        precise_sdpa = (
            PrecisePagedAttention(mesh, ll.self_attn.config, extent) if args.sdpa_compute == "precise" else None
        )
        oracle_sdpa = False
        oracle_source = "tt"
        hf_q = apply_rotary_pos_emb(
            original_hf["self_attn.q_norm"],
            cos[:, length : length + 1],
            sin[:, length : length + 1],
            unsqueeze_dim=2,
        ).transpose(1, 2)

        def logical_cache(value):
            return (
                value[table[0].long()]
                .permute(1, 0, 2, 3)
                .reshape(1, cfg.num_key_value_heads, extent, cfg.head_dim)[:, :, : length + 1]
            )

        def sdpa(q, k, v, *a, **kw):
            if args.sdpa_compute not in ["default", "precise"]:
                kw["compute_kernel_config"] = ttnn.init_device_compute_kernel_config(
                    mesh.arch(),
                    math_fidelity=(
                        ttnn.MathFidelity.HiFi4 if args.sdpa_compute == "hifi4_fp32" else ttnn.MathFidelity.HiFi2
                    ),
                    math_approx_mode=args.sdpa_compute == "hifi2_fp32",
                    fp32_dest_acc_en=args.sdpa_compute != "hifi2_exact",
                    packer_l1_acc=False,
                )
            out = (precise_sdpa or original_sdpa)(q, k, v, *a, **kw)
            captured["q"] = host(q).transpose(1, 2)
            captured["k"] = logical_cache(host(k))
            captured["v"] = logical_cache(host(v))
            captured["sdpa"] = host(out).transpose(1, 2)
            groups = cfg.num_attention_heads // cfg.num_key_value_heads
            captured["cpu_sdpa"] = torch.nn.functional.scaled_dot_product_attention(
                hf_q if "q" in oracle_source else captured["q"],
                (hc.layers[0].keys if "k" in oracle_source else captured["k"]).repeat_interleave(groups, dim=1),
                (hc.layers[0].values if "v" in oracle_source else captured["v"]).repeat_interleave(groups, dim=1),
                attn_mask=dm,
                scale=1.0,
            )
            captured["used_sdpa"] = captured["cpu_sdpa"] if oracle_sdpa else captured["sdpa"]
            if oracle_sdpa:
                return device(captured["cpu_sdpa"].transpose(1, 2))
            return out

        ttnn.transformer.paged_scaled_dot_product_attention_decode = sdpa

        def forward():
            return layer.decode_forward(dt, rope_mats=rope, current_pos=rp, cache_pos=cp, page_table=pt, kv_cache=cache)

        try:
            actual = host(forward())
            compare("eager_final", reference, actual)
            compare("attention", original_hf["self_attn"][0], captured["attention"])
            for name in [
                "input_layernorm",
                "post_attention_layernorm",
                "post_feedforward_layernorm_1",
                "post_feedforward_layernorm_2",
                "residual",
                "experts",
            ]:
                compare(name, original_hf[name], captured[name])
            compare("cache_k", hc.layers[0].keys, captured["k"])
            compare("cache_v", hc.layers[0].values, captured["v"])
            compare("cache_current_k", hc.layers[0].keys[:, :, -1:], captured["k"][:, :, -1:])
            compare("cache_current_v", hc.layers[0].values[:, :, -1:], captured["v"][:, :, -1:])
            compare("query", hf_q, captured["q"])
            compare("sdpa_same_qkv", captured["cpu_sdpa"], captured["sdpa"])
            _, original_values, original_ids = original_hf["router"]
            with torch.no_grad():
                _, values, ids = hf.router(captured["residual"])
            results["routes"] = dict(
                hf_original=original_ids.tolist(),
                hf_same_residual=ids.tolist(),
                tt=captured["routing"].nonzero()[:, 1].tolist(),
            )
            print("routes", results["routes"], flush=True)
            original_dense = torch.zeros_like(captured["routing"]).scatter(-1, original_ids, original_values)
            moe.oracle = device(original_dense.reshape(1, 1, 1, -1))
            compare("original_routes_final", reference, host(forward()))
            moe.oracle = None
            oracle_sdpa = True
            compare("oracle_sdpa_final", reference, host(forward()))
            results["oracle_sdpa_routes"] = captured["routing"].nonzero()[:, 1].tolist()
            print("oracle_sdpa_routes", results["oracle_sdpa_routes"], flush=True)
            oracle_sdpa = False
            oracle_o = True
            compare("oracle_output_projection_final", reference, host(forward()))
            oracle_sdpa = True
            compare("oracle_sdpa_and_output_final", reference, host(forward()))
            oracle_o = False
            for oracle_source in ["q", "k", "v", "qkv"]:
                compare("oracle_sdpa_" + oracle_source + "_final", reference, host(forward()))
        finally:
            ttnn.transformer.paged_scaled_dot_product_attention_decode = original_sdpa
    finally:
        ttnn.close_mesh_device(mesh)
    with torch.no_grad():
        hf.bfloat16()
        bf_cache = DynamicCache()
        hf(
            x.bfloat16(),
            position_embeddings=(cos[:, :length].bfloat16(), sin[:, :length].bfloat16()),
            attention_mask=mask.bfloat16(),
            past_key_values=bf_cache,
        )
        bf_reference = hf(
            dx.bfloat16(),
            position_embeddings=(cos[:, length : length + 1].bfloat16(), sin[:, length : length + 1].bfloat16()),
            attention_mask=dm.bfloat16(),
            past_key_values=bf_cache,
        )
    compare("hf_bf16_vs_fp32", reference, bf_reference.float())
    results["hf_bf16_routes"] = hf_stages["router"][2].tolist()
    print("hf_bf16_routes", results["hf_bf16_routes"], flush=True)
    Path(
        f"models/autoports/google_gemma_4_26b_a4b_it/doc/functional_decoder/boundary_probe_{args.request_index}_{args.sdpa_compute}_{args.qkv_fidelity}.json"
    ).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
