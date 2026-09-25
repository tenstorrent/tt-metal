# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""torch.fx view of the reference blocks at concrete (per-device chunk) shapes.

Traces one sliding and one full ``DecoderLayer`` with a tracer that keeps the
bring-up blocks (norms, linears, rope, sdpa, mlp, router, experts) as leaves,
propagates real shapes, and tags each node with the dashboard block id it
belongs to. Output is plain JSON (nodes + edges) consumed by the dashboard.

    python -m models.demos.gemma4_26b_d_p.reference.fx_graph --chunk 4096 --out graph.json
"""

from __future__ import annotations

import argparse
import json

import torch
from torch import fx
from torch.fx.passes.shape_prop import ShapeProp

from .blocks import SDPA, DecoderLayer, DenseMLP, Experts, Rope, RMSNorm, Router, attention_mask, rope_cos_sin
from .config import FULL, SLIDING, Gemma4TextConfig

LEAF_TYPES = (RMSNorm, torch.nn.Linear, Rope, SDPA, DenseMLP, Router, Experts)

# fx module path (within DecoderLayer) -> dashboard block id. Order matters: first prefix match wins.
BLOCK_OF_PATH = [
    ("input_layernorm", "input_norm"),
    ("self_attn.q_proj", "qkv_proj"),
    ("self_attn.k_proj", "qkv_proj"),
    ("self_attn.v_proj", "qkv_proj"),
    ("self_attn.q_norm", "qk_norm"),
    ("self_attn.k_norm", "qk_norm"),
    ("self_attn.v_norm", "v_norm"),
    ("self_attn.rope", "rope"),
    ("self_attn.sdpa", "sdpa"),
    ("self_attn.o_proj", "o_proj"),
    ("post_attention_layernorm", "post_attn_norm"),
    ("pre_feedforward_layernorm_2", "pre_ff_norm_2"),
    ("pre_feedforward_layernorm", "pre_ff_norm"),
    ("mlp", "dense_mlp"),
    ("post_feedforward_layernorm_1", "post_ff_norm_1"),
    ("post_feedforward_layernorm_2", "post_ff_norm_2"),
    ("post_feedforward_layernorm", "ffn_out_norm"),
    ("router", "router"),
    ("experts", "moe_experts"),
    ("layer_scalar", "layer_scalar"),
]

# Which sub-layer each block lives in (dashboard "layer" grouping inside a decoder layer).
SUBLAYER = {
    "input_norm": "attention",
    "qkv_proj": "attention",
    "qk_norm": "attention",
    "v_norm": "attention",
    "rope": "attention",
    "kv_cache_write": "attention",
    "sdpa": "attention",
    "o_proj": "attention",
    "post_attn_norm": "attention",
    "attn_residual": "attention",
    "pre_ff_norm": "ffn",
    "dense_mlp": "ffn",
    "post_ff_norm_1": "ffn",
    "router": "ffn",
    "pre_ff_norm_2": "ffn",
    "moe_experts": "ffn",
    "post_ff_norm_2": "ffn",
    "ffn_out_norm": "ffn",
    "ffn_residual": "ffn",
    "layer_scalar": "ffn",
}


class BlockTracer(fx.Tracer):
    def is_leaf_module(self, m, qualname):
        return isinstance(m, LEAF_TYPES) or super().is_leaf_module(m, qualname)


def _block_for(node: fx.Node, gm: fx.GraphModule) -> str | None:
    if node.op in ("call_module", "get_attr"):
        for prefix, bid in BLOCK_OF_PATH:
            if node.target == prefix or node.target.startswith(prefix + "."):
                return bid
    return None


def _shape(meta):
    tm = meta.get("tensor_meta")
    if tm is None:
        return None
    if hasattr(tm, "shape"):
        return list(tm.shape)
    if isinstance(tm, (tuple, list)):
        return [list(t.shape) for t in tm if hasattr(t, "shape")]
    return None


def trace_layer(cfg: Gemma4TextConfig, layer_idx: int, chunk: int, prefix: int, seed: int = 0):
    """fx-trace one decoder layer for a chunk of ``chunk`` tokens after ``prefix`` cached tokens."""
    torch.manual_seed(seed)
    layer = DecoderLayer(cfg, layer_idx).to(torch.bfloat16)
    for p in layer.parameters():
        torch.nn.init.normal_(p, std=0.02)
    lt = cfg.layer_types[layer_idx]
    spec = cfg.rope_spec(lt)
    q_pos = torch.arange(prefix, prefix + chunk)
    k_pos = torch.arange(0, prefix + chunk)
    cos, sin = rope_cos_sin(q_pos, spec.theta, spec.head_dim, spec.rotated_pairs)
    mask = attention_mask(q_pos, k_pos, cfg.sliding_window if lt == SLIDING else None)
    n_kv, hd = cfg.layer_kv_heads(layer_idx), cfg.layer_head_dim(layer_idx)
    kp = torch.randn(1, n_kv, prefix, hd, dtype=torch.bfloat16)
    x = torch.randn(1, chunk, cfg.hidden_size, dtype=torch.bfloat16)

    graph = BlockTracer().trace(layer)
    gm = fx.GraphModule(layer, graph)
    ShapeProp(gm).propagate(x, cos, sin, kp, kp.clone(), mask)
    return gm


def graph_to_json(gm: fx.GraphModule, name: str) -> dict:
    nodes, edges = [], []
    last_block = {}
    for n in gm.graph.nodes:
        bid = _block_for(n, gm)
        if n.op == "placeholder":
            kind = "input"
        elif n.op == "output":
            kind = "output"
        elif n.op == "call_module":
            kind = type(gm.get_submodule(n.target)).__name__
        else:
            kind = n.target if isinstance(n.target, str) else getattr(n.target, "__name__", str(n.target))
        nodes.append(
            {
                "id": n.name,
                "op": n.op,
                "kind": kind,
                "target": str(n.target),
                "block": bid,
                "shape": _shape(n.meta),
            }
        )
        for a in n.all_input_nodes:
            edges.append([a.name, n.name])
    # Drop non-tensor bookkeeping (x.shape unpacking etc.), rewiring nothing: those nodes only feed views.
    keep = {n["id"] for n in nodes if n["op"] in ("placeholder", "output") or n["shape"] is not None}
    keep |= {n["id"] for n in nodes if n["op"] == "get_attr"}
    nodes = [n for n in nodes if n["id"] in keep]
    edges = [e for e in edges if e[0] in keep and e[1] in keep]
    by_id = {n["id"]: n for n in nodes}
    order = {n["id"]: i for i, n in enumerate(nodes)}
    preds = {}
    for s_, d in edges:
        preds.setdefault(d, []).append(s_)
    glue = lambda n: n["op"] in ("call_function", "call_method") and n["block"] is None
    two_in_adds = [n for n in nodes if glue(n) and n["kind"] == "add" and len(preds.get(n["id"], [])) == 2]
    if two_in_adds:
        two_in_adds[0]["block"] = "attn_residual"
        two_in_adds[-1]["block"] = "ffn_residual"
        if len(two_in_adds) >= 3:
            two_in_adds[-2]["block"] = "ffn_out_norm"  # m1 + m2 feeds the shared post-ffn norm
    for n in nodes:
        if glue(n) and n["kind"] == "cat" and any(by_id[p]["op"] == "placeholder" for p in preds.get(n["id"], [])):
            n["block"] = "kv_cache_write"
    # Remaining glue inherits the block of its latest tagged producer.
    for n in nodes:
        if glue(n):
            tagged = [p for p in preds.get(n["id"], []) if by_id[p]["block"]]
            if tagged:
                n["block"] = by_id[max(tagged, key=order.get)]["block"]
    for n in nodes:
        n["sublayer"] = SUBLAYER.get(n["block"])
    return {"name": name, "nodes": nodes, "edges": edges}


def build(cfg: Gemma4TextConfig, chunk: int, prefix: int) -> dict:
    s_idx = cfg.layer_types.index(SLIDING)
    f_idx = cfg.layer_types.index(FULL)
    out = {"chunk": chunk, "prefix": prefix, "graphs": {}}
    for name, idx in (("sliding_layer", s_idx), ("full_layer", f_idx)):
        gm = trace_layer(cfg, idx, chunk, prefix)
        out["graphs"][name] = graph_to_json(gm, name)
        out["graphs"][name]["code"] = gm.code
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=256, help="tokens per device chunk")
    ap.add_argument("--prefix", type=int, default=256, help="cached tokens before this chunk")
    ap.add_argument("--out", default="gemma4_26b_d_p_fx_graph.json")
    a = ap.parse_args()
    cfg = Gemma4TextConfig.from_json()
    res = build(cfg, a.chunk, a.prefix)
    json.dump(res, open(a.out, "w"), indent=1)
    for g in res["graphs"].values():
        print(g["name"], len(g["nodes"]), "nodes")


if __name__ == "__main__":
    main()
