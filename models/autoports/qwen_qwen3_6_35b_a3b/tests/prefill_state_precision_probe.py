# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unmeasured real-prefix frontend/state capture from the actual fused prefill."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import l2norm

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import REVISION, config, hf_forward, layer, weights

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder"


def compare(expected, actual):
    actual = actual.reshape(expected.shape)
    a, b = expected.double().flatten(), actual.double().flatten()
    ac, bc = a - a.mean(), b - b.mean()
    pcc = 1.0 if torch.equal(a, b) else float(ac @ bc / (ac.norm() * bc.norm()))
    return dict(
        pcc=pcc,
        relative_l2=float((a - b).norm() / a.norm()),
        max_abs=float((a - b).abs().max()),
        bf16_different_codes=int((expected.bfloat16() != actual.bfloat16()).sum()),
        elements=a.numel(),
        expected_dtype=str(expected.dtype),
        actual_dtype=str(actual.dtype),
    )


def by_head(expected, actual):
    expected, actual = expected.reshape(32, 128, 128), actual.reshape(32, 128, 128)
    return [dict(head=i, **compare(expected[i], actual[i])) for i in range(32)]


def main(args):
    torch.set_num_threads(8)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    record(args.output)
    oracle_path = Path(args.oracle_cache)
    oracle = torch.load(oracle_path, map_location="cpu", weights_only=False)
    assert oracle["key"]["revision"] == REVISION
    assert oracle["key"]["parameters"]["layer"] == 0
    assert oracle["key"]["parameters"]["prompt_length"] == 128
    for path, expected in oracle["key"]["source_hashes"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, f"Oracle source changed: {path}"
    prefix = oracle["prefix"]
    assert tuple(prefix.shape) == (1, 128, 2048)
    c, sd = config(), weights(0)
    hf = layer(sd, 0)
    hv = {}
    handles = []
    for name, module in (
        ("input_norm", hf.input_layernorm),
        ("attention", hf.linear_attn),
        ("projection_qkv", hf.linear_attn.in_proj_qkv),
        ("projection_z", hf.linear_attn.in_proj_z),
        ("projection_a", hf.linear_attn.in_proj_a),
        ("projection_b", hf.linear_attn.in_proj_b),
    ):
        handles.append(module.register_forward_hook(lambda m, i, o, key=name: hv.__setitem__(key, o.detach().clone())))
    original_hf_chunk = hf.linear_attn.chunk_gated_delta_rule
    hf_calls = 0

    def hf_chunk(q, k, v, *positional, **kwargs):
        nonlocal hf_calls
        hf_calls += 1
        normalize = kwargs.get("use_qk_l2norm_in_kernel", False)
        hv.update(
            q=(l2norm(q) if normalize else q).clone(),
            k=(l2norm(k) if normalize else k).clone(),
            v=v.clone(),
            g=kwargs["g"].clone(),
            beta=kwargs["beta"].clone(),
        )
        output, final_state = original_hf_chunk(q, k, v, *positional, **kwargs)
        hv.update(core=output.detach().clone(), raw_final_state=final_state.detach().clone())
        return output, final_state

    hf.linear_attn.chunk_gated_delta_rule = hf_chunk
    cache = DynamicCache(config=c)
    try:
        ref = hf_forward(hf, prefix, cache, 0)
    finally:
        hf.linear_attn.chunk_gated_delta_rule = original_hf_chunk
        for handle in handles:
            handle.remove()
    assert hf_calls == 1
    assert torch.equal(ref, oracle["ref_p"]), "Recomputed prefix differs from immutable stress oracle"
    hv["public_recurrent"] = cache.layers[0].recurrent_states.clone()
    hv["conv_cache_full"] = cache.layers[0].conv_states.clone()
    hv["conv_latest_three"] = hv["conv_cache_full"][:, :, -3:].transpose(1, 2).contiguous()
    hv["packed_projection"] = torch.cat([hv["projection_" + key] for key in ("qkv", "z", "a", "b")], dim=-1)
    result = dict(
        oracle_cache=str(oracle_path.resolve()),
        oracle_sha256=hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
        prefix_source="immutable stress oracle prefix; recomputed HF output bit-identical",
        hf_chunk_calls=hf_calls,
        hf_cache_dtype=str(hv["public_recurrent"].dtype),
        execution=(
            "CPU oracle only" if args.oracle_only else "serial physical chip 0; actual FusedDecoder.prefill_forward"
        ),
        note="Diagnostic host reads capture outputs only; no CPU tensor substitutes for device forward results",
    )
    if args.oracle_only:
        torch.save(dict(prefix=prefix, hf=hv, ref=ref), args.output + ".tensors.pt")
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return
    del hf, cache

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import FusedDecoder

    assert FusedDecoder.__module__.endswith(".fused_decoder")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    original_tt_chunk = ttnn.transformer.chunk_gated_delta_rule
    tv, tt_calls = {}, []
    try:
        dec = FusedDecoder.from_state_dict(sd, hf_config=c, layer_idx=0, mesh_device=mesh)
        state = dec.new_recurrent_state()
        original_norm, original_linear, original_attention = dec.norm, dec.linear, dec._linear_attention

        def norm(value, key):
            output = original_norm(value, key)
            if key == "input_layernorm.weight":
                tv["input_norm"] = ttnn.to_torch(output)
            return output

        def linear(value, key, *positional, **kwargs):
            output = original_linear(value, key, *positional, **kwargs)
            if key == "attention_packed":
                tv["packed_projection"] = ttnn.to_torch(output)
            return output

        def attention(*positional, **kwargs):
            output = original_attention(*positional, **kwargs)
            tv["attention"] = ttnn.to_torch(output)
            return output

        def tt_chunk(q, k, v, g, beta, *positional, **kwargs):
            tt_calls.append(dict(chunk_size=kwargs.get("chunk_size"), num_key_heads=kwargs.get("num_key_heads")))
            for key, value in dict(q=q, k=k, v=v, g=g, beta=beta, initial_state=kwargs["initial_state"]).items():
                tv[key] = ttnn.to_torch(value)
            output, final_state = original_tt_chunk(q, k, v, g, beta, *positional, **kwargs)
            tv.update(core=ttnn.to_torch(output), raw_final_state=ttnn.to_torch(final_state))
            return output, final_state

        dec.norm, dec.linear, dec._linear_attention = norm, linear, attention
        ttnn.transformer.chunk_gated_delta_rule = tt_chunk
        tensor = ttnn.from_torch(prefix, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        got = ttnn.to_torch(dec.prefill_forward(tensor, state=state, plan=dec.prepare_prefill(seq_len=128)))
        assert len(tt_calls) == 1, f"Expected actual native fused prefill chunk path, got {tt_calls}"
        tv["public_recurrent_raw_storage"] = ttnn.to_torch(state["recurrent"])
        tv["public_recurrent"] = tv["public_recurrent_raw_storage"].bfloat16()
        tv["conv_latest_three"] = ttnn.to_torch(state["conv"])
        result["tt_chunk_calls"] = tt_calls
        result["output"] = compare(ref, got)
        result["contribution"] = compare(ref - prefix, got.reshape(prefix.shape) - prefix)
        result["intermediates"] = {}
        for key in (
            "input_norm",
            "packed_projection",
            "attention",
            "core",
            "g",
            "beta",
            "conv_latest_three",
            "raw_final_state",
            "public_recurrent",
        ):
            result["intermediates"][key] = compare(hv[key], tv[key])
        for key in ("q", "k", "v"):
            value = tv[key].reshape(1, 128, -1, 128)
            if value.shape[2] == 16:
                value = value.repeat_interleave(2, dim=2)
            result["intermediates"][key] = compare(hv[key], value)
        result["public_state_per_head"] = by_head(hv["public_recurrent"], tv["public_recurrent"])
        result["raw_state_per_head"] = by_head(hv["raw_final_state"], tv["raw_final_state"])
        for label, begin, end in (("qkv", 0, 8192), ("z", 8192, 12288), ("a", 12288, 12320), ("b", 12320, 12352)):
            result["intermediates"]["projection_" + label] = compare(
                hv["projection_" + label], tv["packed_projection"][..., begin:end]
            )
        torch.save(dict(prefix=prefix, hf=hv, tt=tv, ref=ref, got=got), args.output + ".tensors.pt")
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        ttnn.transformer.chunk_gated_delta_rule = original_tt_chunk
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-cache", default=str(DOC / "oracle_cache/stress_8192_0.pt"))
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
