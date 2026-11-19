# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unmeasured diagnostic: localize a sequential decoder routing discontinuity."""
import argparse
import json
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, hf_forward, layer, pcc, weights
from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import FusedDecoder


def run(args):
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    record(args.output)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    c = config()
    sd = weights(0)
    hf = layer(sd, 0)
    prefix = torch.randn(1, 128, 2048, dtype=torch.bfloat16) * 0.05
    tokens = torch.randn(1, args.steps, 2048, dtype=torch.bfloat16) * 0.05
    hc = DynamicCache(config=c)
    hf_values = {}
    capture_intermediates = False
    for key, module in [
        ("input_norm", hf.input_layernorm),
        ("attention", hf.linear_attn),
        ("post_norm", hf.post_attention_layernorm),
        ("router", hf.mlp.gate),
        ("projection_a", hf.linear_attn.in_proj_a),
        ("projection_b", hf.linear_attn.in_proj_b),
        ("projection_qkv", hf.linear_attn.in_proj_qkv),
    ]:
        module.register_forward_hook(
            lambda m, i, o, k=key: hf_values.__setitem__(k, o) if capture_intermediates else None
        )
    hf.linear_attn.norm.register_forward_pre_hook(
        lambda m, i: hf_values.__setitem__("core", i[0].clone()) if capture_intermediates else None
    )
    old_hf_delta = hf.linear_attn.recurrent_gated_delta_rule

    def hf_delta(q, k, v, **kwargs):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import l2norm

        if capture_intermediates:
            hf_values.update(
                delta_q=l2norm(q),
                delta_k=l2norm(k),
                delta_v=v,
                delta_g=kwargs["g"],
                delta_decay=kwargs["g"].float().exp(),
                delta_beta=kwargs["beta"],
                delta_state_input=kwargs["initial_state"].clone(),
            )
        return old_hf_delta(q, k, v, **kwargs)

    hf.linear_attn.recurrent_gated_delta_rule = hf_delta
    hf_forward(hf, prefix, hc, 0)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        dec = FusedDecoder.from_state_dict(sd, hf_config=c, layer_idx=0, mesh_device=mesh)
        state = dec.new_recurrent_state()

        def upload(x):
            return ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        dec.prefill_forward(upload(prefix), state=state, plan=dec.prepare_prefill(seq_len=128))
        tt_values = {}
        old_tt_delta = dec._recurrent_step

        def tt_delta(q, k, v, g, beta, state, **kwargs):
            if not capture_intermediates:
                return old_tt_delta(q, k, v, g, beta, state, **kwargs)
            tt_values.update(delta_q=q, delta_k=k, delta_v=v, delta_beta=beta, delta_state_input=state)
            if g is not None:
                tt_values["delta_g"] = g
            if kwargs.get("decay_value") is not None:
                tt_values["delta_decay"] = kwargs["decay_value"]
            saved_ops = {name: getattr(ttnn, name) for name in ("multiply", "sum", "subtract", "addcmul")}
            old_recurrent_sum = getattr(dec, "_recurrent_sum", None)
            counts = {"sum": 0, "recurrent_sum": 0}

            def recurrent_sum(products):
                # Observe the model's complete contraction, including its current
                # fast_reduce/cumsum/permute implementation, without replacing it.
                out = old_recurrent_sum(products)
                counts["recurrent_sum"] += 1
                name = "memory" if counts["recurrent_sum"] == 1 else "reduced_output"
                tt_values[name] = out
                tt_values[name + "_products"] = products
                return out

            def observe(name):
                def invoke(*a, **kw):
                    out = saved_ops[name](*a, **kw)
                    if name == "sum" and old_recurrent_sum is None:
                        counts["sum"] += 1
                        tt_values["memory" if counts["sum"] == 1 else "reduced_output"] = out
                    elif name == "subtract":
                        tt_values["value_minus_memory"] = out
                    elif name == "addcmul":
                        tt_values["new_state"] = out
                    elif name == "multiply" and (a[0] is state or kw.get("input_tensor_b_activations")):
                        tt_values["decayed_state"] = out
                    elif tuple(out.shape) == (1, 32, 1, 128):
                        tt_values["delta_update"] = out
                    return out

                return invoke

            try:
                for name in saved_ops:
                    setattr(ttnn, name, observe(name))
                if old_recurrent_sum is not None:
                    dec._recurrent_sum = recurrent_sum
                result = old_tt_delta(q, k, v, g, beta, state, **kwargs)
            finally:
                for name, op in saved_ops.items():
                    setattr(ttnn, name, op)
                if old_recurrent_sum is not None:
                    dec._recurrent_sum = old_recurrent_sum
            tt_values["core"] = result[0]
            tt_values["new_state"] = result[1]
            return result

        dec._recurrent_step = tt_delta
        old_norm, old_attention, old_linear = dec.norm, dec._linear_attention, dec.linear

        def norm(x, key):
            out = old_norm(x, key)
            if capture_intermediates and key in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                tt_values["input_norm" if key == "input_layernorm.weight" else "post_norm"] = out
            return out

        def attention(*a, **k):
            if capture_intermediates:
                tt_values["conv_input_state"] = ttnn.clone(a[1]["conv"])
            out = old_attention(*a, **k)
            if capture_intermediates:
                tt_values["attention"] = out
            return out

        def linear(x, key, *a, **k):
            out = old_linear(x, key, *a, **k)
            if not capture_intermediates:
                return out
            if key == "mlp.gate.weight":
                tt_values["router"] = out
            elif key == "attention_packed":
                tt_values["projection_packed"] = out
            elif key == "output_only_packed":
                tt_values["projection_output_only"] = out
            return out

        dec.norm, dec._linear_attention, dec.linear = norm, attention, linear
        rows = []
        for i in range(args.steps):
            capture_intermediates = 128 + i in args.positions
            tt_values.clear()
            hf_values.clear()
            x = tokens[:, i : i + 1]
            ref = hf_forward(hf, x, hc, 128 + i)
            device_out = dec.decode_forward(upload(x), state=state)
            if 128 + i not in args.positions:
                del device_out
                continue
            got = ttnn.to_torch(device_out).reshape(1, 1, 2048)
            del device_out
            comparisons = {}
            tensors = {k: ttnn.to_torch(v) for k, v in tt_values.items()}
            for k in ("input_norm", "attention", "post_norm"):
                h = hf_values[k]
                t = tensors[k].reshape(h.shape)
                comparisons[k] = dict(pcc=pcc(h, t), max_abs=float((h - t).abs().max()))
            for k in (
                "core",
                "delta_q",
                "delta_k",
                "delta_v",
                "delta_g",
                "delta_decay",
                "delta_beta",
                "delta_state_input",
            ):
                if k not in tensors:
                    continue  # Decode intentionally supplies decay without g.
                h = hf_values[k]
                t = tensors[k]
                if k in ("delta_q", "delta_k"):
                    t = t.repeat_interleave(2, dim=2)
                t = t.reshape(h.shape)
                comparisons[k] = dict(
                    pcc=pcc(h, t),
                    max_abs=float((h - t).abs().max()),
                    relative_l2=float((h.float() - t.float()).norm() / h.float().norm()),
                )
            hs = hc.layers[0].recurrent_states
            ts = ttnn.to_torch(state["recurrent"]).reshape(hs.shape)
            comparisons["state"] = dict(
                pcc=pcc(hs, ts), relative_l2=float((hs - ts).norm() / hs.norm()), max_abs=float((hs - ts).abs().max())
            )
            hf_values["state"] = hs.clone()
            tensors["state"] = ts
            hf_logits, hf_scores, hf_ids = hf_values["router"]
            tt_logits = tensors["router"].reshape(1, 256)
            cpu_tt_logits = torch.nn.functional.linear(tensors["post_norm"].reshape(1, 2048), sd["mlp.gate.weight"])

            def top(logits):
                v, idx = torch.topk(logits.float(), 16, dim=-1)
                return dict(ids=idx.flatten().tolist(), logits=v.flatten().tolist())

            native_ids = ttnn.to_torch(dec.router_out_ids).reshape(32, 32)[0, :8].tolist()
            row = dict(
                position=128 + i,
                seed=args.seed,
                pcc=pcc(ref, got),
                contribution_pcc=pcc(ref - x, got - x),
                intermediates=comparisons,
                hf=top(hf_logits),
                tt_projected=top(tt_logits),
                tt_bf16=top(tt_logits.bfloat16()),
                cpu_on_tt_input=top(cpu_tt_logits),
                hf_ids=hf_ids.flatten().tolist(),
                native_ids=native_ids,
            )
            rows.append(row)
            torch.save(
                dict(
                    hf_values=hf_values, tt_values=tensors, ref=ref, got=got, token=x, seed=args.seed, position=128 + i
                ),
                args.output + f".pos{128+i}.pt",
            )
            Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
            print("ROUTING_CAPTURE", 128 + i, "pcc", row["pcc"], "contribution", row["contribution_pcc"], flush=True)
        Path(args.output).write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(rows))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=110)
    p.add_argument("--seed", type=int, default=913)
    p.add_argument("--positions", type=int, nargs="+", default=[128, 236, 237])
    p.add_argument("--output", required=True)
    run(p.parse_args())
