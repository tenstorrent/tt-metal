# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device-layout weight dicts built from vendored-reference modules (a test helper, not an oracle).

``device_weights`` turns a reference ``Block`` (e.g. from ``oracle.build_reference``) into the ``TtV41Block``
``weights`` dict; ``dequant`` gives a reference ``Linear``'s weight as bf16 ``[out, in]``.
"""

from __future__ import annotations

import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import model as v41
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.kernel_cpu import unpack_fp4


def dequant(linear) -> torch.Tensor:
    """A reference Linear's weight as bf16 [out, in] (FP8 32x32 / FP4 per-32 dequantization is exact)."""
    w = linear.weight
    if w.dtype == torch.float8_e4m3fn:
        s = linear.scale.float()
        full = s.repeat_interleave(32, 0)[: w.shape[0]].repeat_interleave(32, 1)[:, : w.shape[1]]
        return (w.float() * full).to(torch.bfloat16)
    if w.dtype == torch.float4_e2m1fn_x2:
        return (unpack_fp4(w) * linear.scale.float().repeat_interleave(32, 1)).to(torch.bfloat16)
    return w.detach().to(torch.bfloat16)


MOE_KEYS = ("gate_weights", "routed_expert_weights", "shared_expert_weights")


@torch.no_grad()
def device_weights(model: v41.Transformer, layer: int, include_moe: bool = True) -> dict:
    """The TtV41Block ``weights`` dict of the reference backbone layer at position ``layer``. With
    ``include_moe=False`` the MoE entries (``MOE_KEYS``) are omitted and not computed (the routed experts' FP4
    dequantization dominates the cost), for callers whose device MoE tensors are cached."""
    blk = model.layers[layer]
    attn, ffn = blk.attn, blk.ffn
    extra = {}
    if attn.compressor is not None:
        c = attn.compressor
        extra["compressor"] = {"wkv": c.wkv.weight.detach(), "norm": c.norm.weight.detach()} | (
            {"wgate": c.wgate.weight.detach()} if c.compress_ratio > 1 else {}
        )
    if attn.indexer is not None:
        ind = attn.indexer
        extra["indexer"] = {"wq_b": dequant(ind.wq_b), "weights_proj": ind.weights_proj.weight.detach()} | (
            {"wk": ind.wk.weight.detach(), "k_norm": ind.k_norm.weight.detach()} if ind.owns_k else {}
        )
    weights = extra | {
        "attn": {
            "wq_a": dequant(attn.wq_a),
            "q_norm": attn.q_norm.weight.detach(),
            "wq_b": dequant(attn.wq_b),
            "wkv": dequant(attn.wkv),
            "kv_norm": attn.kv_norm.weight.detach(),
            "wo_a": dequant(attn.wo_a),
            "wo_b": dequant(attn.wo_b),
            "attn_sink": attn.attn_sink.detach(),
        },
        "attn_norm": blk.attn_norm.weight.detach(),
        "ffn_norm": blk.ffn_norm.weight.detach(),
        "hc_attn": (blk.hc_attn_fn.detach(), blk.hc_attn_base.detach(), blk.hc_attn_scale.detach()),
        "hc_ffn": (blk.hc_ffn_fn.detach(), blk.hc_ffn_base.detach(), blk.hc_ffn_scale.detach()),
    }
    if not include_moe:
        return weights
    se = ffn.shared_experts
    return weights | {
        "gate_weights": {
            "weight": ffn.gate.weight.detach().to(torch.bfloat16),
            "e_score_correction_bias": ffn.gate.bias.detach().float(),
        },
        "routed_expert_weights": [
            {"gate_proj": dequant(e.w1), "up_proj": dequant(e.w3), "down_proj": dequant(e.w2)} for e in ffn.experts
        ],
        "shared_expert_weights": {
            "gate_proj": dequant(se.w1),
            "up_proj": dequant(se.w3),
            "down_proj": dequant(se.w2),
        },
    }
