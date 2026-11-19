# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Selected-step GDN attribution; diagnostic host reads are NEVER perf evidence.

Preserves run_decode_sequence's RNG/trace/reference sequence. Selected steps are
replayed eagerly from saved state with temporary hooks, then traced state is
restored. No model implementation or acceptance threshold is changed.
"""
import argparse
from contextlib import ExitStack, contextmanager
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding, l2norm,
)
import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.evidence import decoder_class, decoder_kwargs, provenance
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import MODEL, REVISION, load_layer, pcc


def metrics(a, b):
    a, b = a.float(), b.float()
    assert a.shape == b.shape, (a.shape, b.shape)
    error = a - b
    correlation = pcc(a, b)
    return {
        "pcc": correlation if torch.isfinite(torch.tensor(correlation)) else None,
        "max_abs": error.abs().max().item(),
        "rmse": error.square().mean().sqrt().item(),
        "relative_rms": (error.square().mean() / a.square().mean().clamp_min(1e-30)).sqrt().item(),
        "equal": torch.equal(a, b),
    }


def selected(t, users):
    return t.detach().cpu()[users].clone()


@contextmanager
def hf_capture(ref, batch, users, data):
    attn = ref.linear_attn
    original = attn.recurrent_gated_delta_rule

    def recurrence(*args, **kwargs):
        bound = inspect.signature(original).bind(*args, **kwargs)
        bound.apply_defaults()
        for key in ("query", "key", "value", "g", "beta", "initial_state"):
            data[key] = selected(bound.arguments[key], users)
        result = original(*args, **kwargs)
        data["core"] = selected(result[0], users).reshape(len(users), 48, 128)
        data["updated_state"] = selected(result[1], users)
        return result

    def norm_pre(module, args):
        x, z = args
        data["norm_input"] = selected(x.reshape(batch, 48, 128), users)
        data["z"] = selected(z.reshape(batch, 48, 128), users)
        # HF's actual ungated intermediate, reproduced with its exact dtype boundaries.
        xf = data["norm_input"].float()
        denom = (xf.square().mean(-1, keepdim=True) + module.variance_epsilon).sqrt()
        data["norm_denominator"] = denom
        data["norm_ungated"] = module.weight.detach().cpu() * (xf * (xf.square().mean(-1, keepdim=True) + module.variance_epsilon).rsqrt()).to(x.dtype)

    def norm_post(module, args, result):
        data["gated"] = selected(result.reshape(batch, 48, 128), users)

    def attention_post(module, args, result):
        data["attention"] = selected(result, users).reshape(len(users), 5120)

    def record_projection(name):
        def hook(module, args, result):
            data[name] = selected(result.reshape(batch, -1), users)
        return hook

    handles = [attn.norm.register_forward_pre_hook(norm_pre), attn.norm.register_forward_hook(norm_post),
               attn.register_forward_hook(attention_post),
               ref.input_layernorm.register_forward_hook(record_projection("input_norm")),
               ref.post_attention_layernorm.register_forward_hook(record_projection("post_attention_norm"))]
    for name in ("qkv", "a", "b"):
        handles.append(getattr(attn, "in_proj_" + name).register_forward_hook(record_projection("projection_" + name)))
    try:
        with patch.object(attn, "recurrent_gated_delta_rule", recurrence):
            yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def tt_capture(layer, batch, users, read, data):
    """Read only in an eager replay; never put host conversions in trace capture."""
    original_delta, original_linear, original_norm = layer._delta_chunks, layer._linear, ttnn.rms_norm
    original_multiply = ttnn.multiply
    compact_module = inspect.getmodule(layer._gdn_compact) if hasattr(layer, "_gdn_compact") else None
    original_compact = getattr(compact_module, "_recurrent_sfpu_compact", None)
    original_conv = getattr(layer, "_batched_conv", None)
    original_layer_norm = getattr(layer, "_norm", None)

    def record_vectors(q, k, v, packed=False):
        if packed:
            packed_value = read(q).reshape(batch, 1, 10240)
            tensors = (packed_value[..., :2048], packed_value[..., 2048:4096], packed_value[..., 4096:])
        else:
            tensors = tuple(read(t)[:, :1] for t in (q, k, v))
        for name, value, heads in zip(("query", "key", "value"), tensors, (16, 16, 48)):
            value = selected(value, users).reshape(len(users), 1, heads, 128)
            data[name] = value.repeat_interleave(3, dim=2) if heads == 16 else value

    def conv(*args, **kwargs):
        result = original_conv(*args, **kwargs)
        # Save raw vectors before optional external q/k normalization. Packed RM
        # returns the same [B,1,10240] buffer three times; split its logical channels.
        record_vectors(*result, packed=result[0].shape[-1] == 10240)
        return result

    def compact(q, k, v, g, beta, state, *args, **kwargs):
        if not kwargs.get("normalized", False):
            record_vectors(q, k, v, packed=kwargs.get("packed_qkv", False))
        else:
            assert "query" in data, "Normalized compact inputs require the raw convolution hook"
            data["kernel_normalized_scaled_q"] = selected(read(q).reshape(batch, 16, 128), users)
            data["kernel_normalized_k"] = selected(read(k).reshape(batch, 16, 128), users)
        data["beta"] = selected(read(beta).reshape(batch, 1, 48), users)
        data["g"] = selected(read(g).reshape(batch, 1, 48), users)
        data["initial_state"] = selected(read(state).reshape(batch, 48, 128, 128), users)
        result = original_compact(q, k, v, g, beta, state, *args, **kwargs)
        data["core"] = selected(read(result).reshape(batch, 48, 128), users)
        # This is the FP32 kernel result before any public cache boundary cast.
        data["updated_state"] = selected(read(state).reshape(batch, 48, 128, 128), users)
        return result

    def recurrence(q, k, v, beta, g, state):
        for name, tensor, heads in (("query", q, 16), ("key", k, 16), ("value", v, 48)):
            value = selected(read(tensor)[:, :1], users).reshape(len(users), 1, heads, 128)
            data[name] = value.repeat_interleave(3, dim=2) if heads == 16 else value
        data["beta"] = selected(read(beta)[:, :1], users)
        data["g"] = selected(read(g)[:, :1], users)
        data["initial_state"] = selected(read(state), users)
        result = original_delta(q, k, v, beta, g, state)
        data["core"] = selected(read(result), users).reshape(len(users), 48, 128)
        data["updated_state"] = selected(read(state), users)
        return result

    def norm(x, *args, **kwargs):
        affine = kwargs.get("weight") is layer.weights["linear_attn.norm.weight"]
        unweighted_gdn = (kwargs.get("weight") is None and x.shape[-1] == 128
                          and tuple(x.shape) in ((1, batch * 48, 128), (batch, 48, 128)))
        match = affine or unweighted_gdn
        other_norm = next((name for key, name in (("input_layernorm.weight", "input_norm"),
                          ("post_attention_layernorm.weight", "post_attention_norm"))
                          if kwargs.get("weight") is layer.weights[key]), None)
        if match:
            data["norm_input"] = selected(read(x).reshape(batch, 48, 128), users)
        result = original_norm(x, *args, **kwargs)
        if other_norm:
            # Some policies cast this FP32 return once more to BF16; keep both.
            raw = selected(read(result).reshape(batch, 5120), users)
            data[other_norm + "_native_output"] = raw
            data[other_norm] = raw.to(torch.bfloat16)
        if match:
            key = "norm_unweighted" if unweighted_gdn else "norm_ungated"
            data[key] = selected(read(result).reshape(batch, 48, 128), users)
            xf = data["norm_input"].float()
            data["norm_denominator"] = (xf.square().mean(-1, keepdim=True) + layer.config.rms_norm_eps).sqrt()
        return result

    def multiply(a, b, *args, **kwargs):
        result = original_multiply(a, b, *args, **kwargs)
        if a is layer.weights["linear_attn.norm.weight"] or b is layer.weights["linear_attn.norm.weight"]:
            data["norm_ungated"] = selected(read(result).reshape(batch, 48, 128), users)
        return result

    def layer_norm(x, key, *args, **kwargs):
        result = original_layer_norm(x, key, *args, **kwargs)
        name = {"input_layernorm.weight": "input_norm", "post_attention_layernorm.weight": "post_attention_norm"}.get(key)
        if name:
            data[name] = selected(read(result).reshape(batch, 5120), users)
        return result

    def linear(x, key, *args, **kwargs):
        if key == "linear_attn.out_proj.weight":
            data["gated"] = selected(read(x).reshape(batch, 48, 128), users)
        result = original_linear(x, key, *args, **kwargs)
        if key == "linear_attn.out_proj.weight":
            data["attention"] = selected(read(result).reshape(batch, 5120), users)
        elif key == "packed_gdn":
            raw = selected(read(result).reshape(batch, -1), users)
            data["z"] = raw[:, 10240:16384].reshape(len(users), 48, 128)
            for name, start, end in (("qkv", 0, 10240), ("b", 16384, 16432), ("a", 16448, 16496)):
                data["projection_" + name] = raw[:, start:end].clone()
        elif key == "linear_attn.in_proj_z.weight":
            data["z"] = selected(read(result).reshape(batch, 48, 128), users)
        elif key in ("linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"):
            name = key.split("in_proj_")[1].split(".")[0]
            data["projection_" + name] = selected(read(result).reshape(batch, -1), users)
        return result

    with ExitStack() as hooks:
        for obj, name, replacement in ((layer, "_delta_chunks", recurrence), (layer, "_linear", linear),
                                        (ttnn, "rms_norm", norm), (ttnn, "multiply", multiply)):
            hooks.enter_context(patch.object(obj, name, replacement))
        if original_compact is not None:
            hooks.enter_context(patch.object(compact_module, "_recurrent_sfpu_compact", compact))
        if original_conv is not None:
            hooks.enter_context(patch.object(layer, "_batched_conv", conv))
        if original_layer_norm is not None:
            hooks.enter_context(patch.object(layer, "_norm", layer_norm))
        yield
    required = {"query", "key", "value", "g", "beta", "initial_state", "updated_state", "core", "norm_input", "norm_ungated", "gated", "attention", "z"}
    assert required <= data.keys(), ("Unsupported decoder hook topology", required - data.keys())


def cpu_recurrence(inputs, state, hf_norm):
    """IEEE FP32 attribution oracle, not a claim of native arithmetic equivalence."""
    q, k = inputs["query"], inputs["key"]
    if hf_norm:
        q, k = l2norm(q, dim=-1, eps=1e-6), l2norm(k, dim=-1, eps=1e-6)
    else:
        q, k = [x.float() * (x.float().square().sum(-1, keepdim=True) + 1e-6).rsqrt() for x in (q, k)]
    q, k = q[:, 0].float() * 128**-0.5, k[:, 0].float()
    value = inputs["value"][:, 0].float()
    beta = inputs["beta"][:, 0].float().unsqueeze(-1)
    decay = inputs["g"][:, 0].float().exp().unsqueeze(-1).unsqueeze(-1)
    decayed = state.float() * decay
    prediction = (decayed * k.unsqueeze(-1)).sum(-2)
    delta = (value - prediction) * beta
    updated = decayed + k.unsqueeze(-1) * delta.unsqueeze(-2)
    core = (updated * q.unsqueeze(-1)).sum(-2)
    return {"normalized_scaled_q": q, "normalized_k": k, "exp_g": decay, "prediction": prediction,
            "delta": delta, "updated_state": updated, "core": core}


def summarize(hf, tt):
    shared = sorted(set(hf) & set(tt))
    comparisons = {key: metrics(hf[key], tt[key]) for key in shared}
    # Per-head statistics expose a small core readout followed by norm amplification.
    heads = []
    for user in range(hf["core"].shape[0]):
        heads.append([{
            "head": h,
            "hf_core_rms": hf["core"][user, h].float().square().mean().sqrt().item(),
            "tt_core_rms": tt["core"][user, h].float().square().mean().sqrt().item(),
            "core": metrics(hf["core"][user, h], tt["core"][user, h]),
            "gated": metrics(hf["gated"][user, h], tt["gated"][user, h]),
        } for h in range(48)])
    return {"comparisons": comparisons, "per_user_heads": heads}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--target", action="append", help="Zero-based STEP:USER, repeatable; defaults 19:20,58:30,99:27")
    ap.add_argument("--dump-intermediates", type=Path, help="Optional directory for selected-user .pt tensors")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    targets = {}
    for item in args.target or ["19:20", "58:30", "99:27"]:
        step, user = map(int, item.split(":"))
        assert 0 <= step < args.steps and 0 <= user < args.batch, item
        targets.setdefault(step, set()).add(user)
    targets = {step: sorted(users) for step, users in targets.items()}
    prov = provenance()
    torch.set_num_threads(8)
    torch.manual_seed(1729)
    c = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True).text_config
    c._attn_implementation = "sdpa"
    weights = load_layer(args.layer)
    ref = Qwen3_5DecoderLayer(c, args.layer).eval().to(torch.bfloat16)
    ref.load_state_dict(weights)
    rot = Qwen3_5TextRotaryEmbedding(c)
    cache = DynamicCache(config=c)
    assert hasattr(ref, "linear_attn"), "This diagnostic is intentionally GDN-only"
    # Do not move fixture sampling before HF construction: that changes the failed inputs.
    prefix = torch.randn(args.batch, 128, 5120, dtype=torch.bfloat16) * 0.1
    tokens = torch.randn(args.steps, args.batch, 1, 5120, dtype=torch.bfloat16) * 0.1
    with torch.no_grad():
        expected_pre = ref(prefix, position_embeddings=rot(prefix, torch.arange(128).expand(args.batch, -1)), past_key_values=cache)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    tid = None
    try:
        layer = decoder_class().from_state_dict(weights, hf_config=c, layer_idx=args.layer, mesh_device=mesh, **decoder_kwargs())
        assert hasattr(layer, "_delta_chunks"), "Use optimized/fused decoder; functional hook differs"
        read = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        state = layer.create_recurrent_state(args.batch)
        buffers = [state.conv, state.recurrent]
        def snapshot():
            return [read(t).clone() for t in buffers]
        def restore(values):
            for value, destination in zip(values, buffers):
                host = ttnn.from_torch(value, dtype=destination.dtype, layout=destination.layout)
                ttnn.copy_host_to_device_tensor(host, destination)
            ttnn.synchronize_device(mesh)
        pre = read(layer.prefill_forward(layer._upload(prefix), state=state))
        saved = snapshot()
        hf_prefill_state = cache.layers[args.layer].recurrent_states.detach().clone()
        prefill_state_metrics = metrics(hf_prefill_state, saved[1])
        td = layer._upload(tokens[0])
        hosts = [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for t in tokens]
        layer.decode_forward(td, state=state)
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = layer.decode_forward(td, state=state)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        restore(saved)
        del saved, hf_prefill_state
        rows, diagnostics = [], []
        for step in range(args.steps):
            users = targets.get(step)
            ttnn.copy_host_to_device_tensor(hosts[step], td)
            ttnn.synchronize_device(mesh)
            before = snapshot() if users else None
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            actual = read(out).clone()
            hf, tt = {}, {}
            if users:
                hf["conv_before"] = selected(cache.layers[args.layer].conv_states.transpose(1, 2)[:, -3:], users)
                tt["conv_before"] = selected(before[0], users)
                after = snapshot()
                restore(before)
                try:
                    with tt_capture(layer, args.batch, users, read, tt):
                        eager = read(layer.decode_forward(td, state=state)).clone()
                    eager_state = snapshot()
                    replay = {"output": metrics(actual, eager), "conv_state": metrics(after[0], eager_state[0]),
                              "recurrent_state": metrics(after[1], eager_state[1])}
                    tt["output"] = selected(eager, users).reshape(len(users), 5120)
                finally:
                    # The diagnostic rerun must not advance the traced sequence a second time.
                    restore(after)
                assert state.conv is buffers[0] and state.recurrent is buffers[1], "State tensor identity changed"
            with torch.no_grad():
                if users:
                    with hf_capture(ref, args.batch, users, hf):
                        expected = ref(tokens[step], position_embeddings=rot(tokens[step], torch.full((args.batch, 1), 128 + step, dtype=torch.long)), past_key_values=cache)
                else:
                    expected = ref(tokens[step], position_embeddings=rot(tokens[step], torch.full((args.batch, 1), 128 + step, dtype=torch.long)), past_key_values=cache)
            scores = [pcc(expected[b], actual[b]) for b in range(args.batch)]
            rows.append({"step": step, "per_user_pcc": scores})
            if users:
                hf["output"] = selected(expected, users).reshape(len(users), 5120)
                hf["cached_state"] = selected(cache.layers[args.layer].recurrent_states, users)
                hf["conv_after"] = selected(cache.layers[args.layer].conv_states.transpose(1, 2)[:, -3:], users)
                tt["conv_after"] = selected(after[0], users)
                tt["cached_state"] = selected(after[1], users)
                oracles = {
                    "hf_inputs_hf_state_hf_norm": cpu_recurrence(hf, hf["initial_state"], True),
                    "tt_inputs_tt_state_hf_norm": cpu_recurrence(tt, tt["initial_state"], True),
                    "tt_inputs_tt_state_fp32_norm": cpu_recurrence(tt, tt["initial_state"], False),
                    "hf_inputs_tt_state_hf_norm": cpu_recurrence(hf, tt["initial_state"], True),
                    "tt_inputs_hf_state_hf_norm": cpu_recurrence(tt, hf["initial_state"], True),
                }
                diagnostic = {"step": step, "users": users, "eager_vs_trace": replay, **summarize(hf, tt),
                              "cpu_oracles": {name: {"vs_hf_core": metrics(hf["core"], value["core"]),
                                                       "vs_tt_core": metrics(tt["core"], value["core"]),
                                                       "vs_tt_state": metrics(tt["updated_state"], value["updated_state"])}
                                              for name, value in oracles.items()}}
                if args.dump_intermediates:
                    args.dump_intermediates.mkdir(parents=True, exist_ok=True)
                    path = args.dump_intermediates / f"layer{args.layer}_step{step}.pt"
                    torch.save({"provenance": prov, "step": step, "users": users, "hf": hf, "tt": tt,
                                "oracles": oracles, "input": selected(tokens[step], users),
                                "traced_output": selected(actual, users),
                                "hf_norm_weight": ref.linear_attn.norm.weight.detach().cpu().clone()}, path)
                    diagnostic["tensor_artifact"] = str(path)
                diagnostics.append(diagnostic)
                print("DIAGNOSTIC", step, users, "traced PCC", [scores[u] for u in users], "eager/trace", replay["output"], flush=True)
                del before, after, eager_state, hf, tt, oracles
        result = {"provenance": prov, "diagnostic_only_no_perf_claim": True, "layer": args.layer,
                  "batch": args.batch, "steps": args.steps, "prefill_pcc": pcc(expected_pre, pre),
                  "prefill_state": prefill_state_metrics,
                  "minimum_per_user_pcc": min(s for row in rows for s in row["per_user_pcc"]),
                  "rows": rows, "diagnostics": diagnostics}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        assert result["prefill_pcc"] >= .995 and result["minimum_per_user_pcc"] >= .995, result["minimum_per_user_pcc"]
    finally:
        if tid is not None:
            ttnn.release_trace(mesh, tid)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
