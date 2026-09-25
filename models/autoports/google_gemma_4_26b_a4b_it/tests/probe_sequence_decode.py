# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Localize failing steps of the normal trace harness without changing compute."""

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests import run_decoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.functional_decoder import FunctionalDecoder


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage-report", type=Path, required=True)
    parser.add_argument("--rope-fp32", action="store_true")
    parser.add_argument("--save-inputs", type=Path)
    parser.add_argument("--sfpu-qkv-decode", action="store_true")
    parser.add_argument("--inspect-position", type=int)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    references, captured, rows = {}, {}, []
    inputs = []
    hf_layer = None
    decode_position = 0
    original_load = run_decoder.load_layer
    original_factory = FunctionalDecoder.from_state_dict.__func__
    original_pcc = run_decoder.comp_pcc
    restore = []
    if args.rope_fp32:
        rotary_forward = run_decoder.Gemma4TextRotaryEmbedding.forward
        rope_pointers = set()

        def rotary_forward_probe(*a, **kw):
            result = rotary_forward(*a, **kw)
            rope_pointers.update(t.data_ptr() for t in result)
            return result

        restore.append((run_decoder.Gemma4TextRotaryEmbedding, "forward", rotary_forward))
        run_decoder.Gemma4TextRotaryEmbedding.forward = rotary_forward_probe
        from_torch = ttnn.from_torch

        def upload(value, *a, **kw):
            if value.data_ptr() in rope_pointers:
                kw["dtype"] = ttnn.float32
            return from_torch(value, *a, **kw)

        restore.append((ttnn, "from_torch", from_torch))
        ttnn.from_torch = upload
        embedding = ttnn.embedding

        def select_rope(indices, table, *a, **kw):
            if table.dtype != ttnn.float32:
                return embedding(indices, table, *a, **kw)
            row = ttnn.repeat(indices[:, :1], (1, table.shape[-1]))
            selected = ttnn.gather(ttnn.to_layout(table, ttnn.ROW_MAJOR_LAYOUT), dim=0, index=row)
            return ttnn.to_layout(selected, ttnn.TILE_LAYOUT)

        restore.append((ttnn, "embedding", embedding))
        ttnn.embedding = select_rope

    def load(*a, **kw):
        nonlocal hf_layer
        hf = original_load(*a, **kw)
        hf_layer = hf
        stages = {}
        for name in ("self_attn", "router"):
            module = getattr(hf, name)

            def hook(module, inputs, output, name=name):
                stages[name] = inputs[0].detach().clone() if name == "router" else output[0].detach().clone()

            module.register_forward_hook(hook)
        original_forward = hf.forward

        def forward(value, *a, **kw):
            nonlocal decode_position
            out = original_forward(value, *a, **kw)
            inputs.append(value.detach().clone())
            if value.shape[-2] > 1:
                decode_position = value.shape[-2]
            else:
                references[out.data_ptr()] = dict(
                    position=decode_position, input=value.detach().clone(), **{k: v.clone() for k, v in stages.items()}
                )
                decode_position += 1
            return out

        hf.forward = forward
        return hf

    def factory(cls, *a, **kw):
        result = original_factory(cls, *a, **kw)
        source_router = result.layer.moe.router
        source_attention = result.layer.self_attn
        source_sdpa = source_attention.decode_sdpa
        projection = source_attention.source.weights.wqkv
        captured["projection"] = projection
        if args.sfpu_qkv_decode:
            projection = source_attention.source.weights.wqkv
            matrix = ttnn.transpose(ttnn.typecast(projection.weight, ttnn.float32), -2, -1)

            class ExactProjection:
                def __call__(self, value):
                    if value.shape[-2] != 1:
                        return projection(value)
                    parts = []
                    for start in range(0, matrix.shape[-2], 256):
                        weight = matrix[:, :, start : start + 256, :]
                        repeated = ttnn.repeat(value, (1, 1, weight.shape[-2], 1))
                        products = ttnn.mul(repeated, weight)
                        parts.append(ttnn.transpose(ttnn.sum(products, dim=-1, keepdim=True), -2, -1))
                    return ttnn.concat(parts, dim=-1)

            source_attention.source.weights = replace(source_attention.source.weights, wqkv=ExactProjection())
        current_projection = source_attention.source.weights.wqkv

        class ProjectionProbe:
            def __call__(self, value):
                out = current_projection(value)
                if value.shape[-2] == 1:
                    captured.update(qkv=out, input_norm=value)
                return out

        source_attention.source.weights = replace(source_attention.source.weights, wqkv=ProjectionProbe())

        class SDPAProbe:
            def __call__(self, q, k, v, **kwargs):
                output = source_sdpa(q, k, v, **kwargs)
                captured.update(q=q, k=k, v=v, sdpa=output, **kwargs)
                return output

        class AttentionProbe:
            def __getattr__(self, name):
                return getattr(source_attention, name)

            def __call__(self, value, **kwargs):
                output = source_attention(value, **kwargs)
                if kwargs.get("is_decode", True):
                    captured["self_attn"] = output
                return output

        class RouterProbe:
            def __call__(self, value):
                output = source_router(value)
                if value.shape[-2] == 1:
                    captured.update(residual=value, routing=output)
                return output

        source_attention.decode_sdpa = SDPAProbe()
        result.layer.self_attn = AttentionProbe()
        result.layer.moe.router = RouterProbe()
        return result

    def host(value):
        return ttnn.to_torch(value).float()

    def corr(expected, actual):
        return float(original_pcc(expected.reshape(-1).float(), actual.reshape(-1).float(), 0.995)[1])

    def compare(reference, actual, *a, **kw):
        passed, pcc = original_pcc(reference, actual, *a, **kw)
        state = references.get(reference.data_ptr())
        if state is not None:
            row = dict(position=state["position"], pcc=float(pcc), passed=bool(passed))
            if not passed or state["position"] == args.inspect_position:
                residual = host(captured["residual"]).reshape(1, -1)
                actual_routes = torch.nonzero(host(captured["routing"]).reshape(-1)).flatten().tolist()
                with torch.no_grad():
                    original_routes = hf_layer.router(state["router"].reshape(1, -1))[2].flatten().tolist()
                    same_routes = hf_layer.router(residual)[2].flatten().tolist()
                row.update(
                    residual_pcc=corr(state["router"], residual),
                    attention_pcc=corr(state["self_attn"], host(captured["self_attn"])),
                    hf_routes=original_routes,
                    cpu_same_residual_routes=same_routes,
                    tt_routes=actual_routes,
                )
                position = state["position"]
                table = ttnn.to_torch(captured["page_table_tensor"])[0].long()
                q = host(captured["q"]).transpose(1, 2)
                selected = []
                for name in ("k", "v"):
                    cache = host(captured[name])
                    logical = cache[table].permute(1, 0, 2, 3).reshape(1, cache.shape[1], -1, cache.shape[-1])
                    start = (
                        max(0, position + 1 - hf_layer.config.sliding_window) if hf_layer.self_attn.is_sliding else 0
                    )
                    selected.append(logical[:, :, start : position + 1])
                with torch.no_grad():
                    sdpa = torch.nn.functional.scaled_dot_product_attention(q, *selected, scale=1.0, enable_gqa=True)
                    sdpa = sdpa.transpose(1, 2)
                    projected = hf_layer.self_attn.o_proj(sdpa.flatten(-2))
                    rebuilt_residual = state["input"] + hf_layer.post_attention_layernorm(projected)
                    rebuilt_routes = hf_layer.router(rebuilt_residual.reshape(1, -1))[2].flatten().tolist()
                row.update(sdpa_same_qkv_pcc=corr(sdpa, host(captured["sdpa"])), cpu_sdpa_o_routes=rebuilt_routes)
                projection = captured["projection"]
                baseline_qkv = ttnn.linear(
                    captured["input_norm"],
                    projection.weight,
                    dtype=ttnn.float32,
                    compute_kernel_config=projection.compute,
                )
                normalized = host(captured["input_norm"]).reshape(1, 1, -1)
                with torch.no_grad():
                    cpu_parts = [hf_layer.self_attn.q_proj(normalized), hf_layer.self_attn.k_proj(normalized)]
                    cpu_parts.append(
                        hf_layer.self_attn.v_proj(normalized) if hf_layer.self_attn.v_proj is not None else cpu_parts[1]
                    )
                    expected_qkv = torch.cat(cpu_parts, dim=-1)
                actual_qkv, baseline = host(captured["qkv"]).reshape_as(expected_qkv), host(baseline_qkv).reshape_as(
                    expected_qkv
                )
                row.update(
                    qkv_same_input_pcc=corr(expected_qkv, actual_qkv),
                    qkv_same_input_max_error=float((actual_qkv - expected_qkv).abs().max()),
                    fpu_qkv_same_input_pcc=corr(expected_qkv, baseline),
                    fpu_qkv_same_input_max_error=float((baseline - expected_qkv).abs().max()),
                )
                print("SEQUENCE_FAILURE", row, flush=True)
            rows.append(row)
        return passed, pcc

    run_decoder.load_layer = load
    FunctionalDecoder.from_state_dict = classmethod(factory)
    run_decoder.comp_pcc = compare
    try:
        run_decoder.main()
    finally:
        run_decoder.load_layer = original_load
        FunctionalDecoder.from_state_dict = classmethod(original_factory)
        run_decoder.comp_pcc = original_pcc
        for namespace, name, value in reversed(restore):
            setattr(namespace, name, value)
        args.stage_report.write_text(json.dumps(rows, indent=2) + "\n")
        if args.save_inputs:
            torch.save(dict(x=inputs[0], decode_inputs=torch.cat(inputs[1:], dim=1)), args.save_inputs)
            hashes = [hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in inputs]
            args.save_inputs.with_suffix(".sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
