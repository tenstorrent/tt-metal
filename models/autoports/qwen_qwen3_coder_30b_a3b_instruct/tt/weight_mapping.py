# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace -> TTNN weight mapping for Qwen3-Coder-30B-A3B-Instruct.

RoPE convention (the decision this file encodes)
------------------------------------------------
TTNN offers two rotary embedding ops, and each demands a different weight
layout. They are interchangeable only as a *pair*; crossing them runs fine and
silently produces garbage.

  Meta-style  ``rotary_embedding_llama``     head channels interleaved
              -> q/k rows must be reordered (``reverse_permute``), and because
                 that reorders channels *within* a head, Qwen3's per-head
                 QK-norm weights must be reordered to match as well.

  HF-style    ``ttnn.experimental.rotary_embedding``   HF's native layout
              -> no weight transformation at all.

This port uses **HF-style**, matching models/demos/gemma4 (the Blackhole
exemplar this attention block is written against, and the only other supported
model with Qwen3-shaped per-head QK-norm). Keeping the checkpoint layout
untouched removes both permutation steps, so the QK-norm weights are copied
verbatim. If the RoPE op in functional_decoder.py is ever swapped for the llama
variant, both permutations have to come back together.

Expert fusion (MoE)
-------------------
The checkpoint stores 3 tensors per expert. TTNN wants them batched, with gate
and up fused as ``[gate ; up]`` along the output dim, matching
``Qwen3MoeExperts.forward``'s ``chunk(2, dim=-1)``.

The fused QKV layout follows models/tt_transformers/tt/attention.py: transpose
each projection to ``[in, out]``, then concatenate ``[q, k, v]`` along the
output dim.
"""

from __future__ import annotations

import torch


def convert_attention_weights(
    sd: dict[str, torch.Tensor],
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    """Return the fused ``wqkv`` plus ``wo`` and the QK-norm weights.

    Channel order is left exactly as HuggingFace stores it -- see the RoPE
    convention note in the module docstring. The only structural change is
    fusing Q, K and V into one matmul.

    ``sd`` uses layer-relative keys (``self_attn.q_proj.weight``, ...).
    Single-device layout; tensor-parallel chunking is a later stage.
    """
    q = sd["self_attn.q_proj.weight"].float()
    k = sd["self_attn.k_proj.weight"].float()
    v = sd["self_attn.v_proj.weight"].float()
    o = sd["self_attn.o_proj.weight"].float()

    assert q.shape[0] == n_heads * head_dim, f"q_proj {tuple(q.shape)} != {n_heads}x{head_dim}"
    assert k.shape[0] == n_kv_heads * head_dim, f"k_proj {tuple(k.shape)} != {n_kv_heads}x{head_dim}"

    # torch stores nn.Linear as [out, in]; TTNN matmuls right-multiply, so
    # transpose to [in, out] before concatenating along the output dim.
    wqkv = torch.cat([q.T, k.T, v.T], dim=-1).unsqueeze(0).unsqueeze(0)

    out = {
        "wqkv": wqkv,
        "wo": o.T.contiguous(),
    }

    for name, key in (("q_norm", "self_attn.q_norm.weight"), ("k_norm", "self_attn.k_norm.weight")):
        if key in sd:
            out[name] = sd[key].float()

    return out


def convert_moe_weights(sd: dict[str, torch.Tensor], *, n_experts: int) -> dict[str, torch.Tensor]:
    """Batch the per-expert checkpoint tensors and fuse gate/up.

    gate first, then up -- ``Qwen3MoeExperts.forward`` chunks the matmul output
    in half and treats the FIRST half as gate.
    """
    gate_up = torch.stack(
        [
            torch.cat(
                [
                    sd[f"mlp.experts.{e}.gate_proj.weight"].float(),
                    sd[f"mlp.experts.{e}.up_proj.weight"].float(),
                ],
                dim=0,
            )
            for e in range(n_experts)
        ]
    )
    down = torch.stack([sd[f"mlp.experts.{e}.down_proj.weight"].float() for e in range(n_experts)])
    return {
        "router": sd["mlp.gate.weight"].float(),
        "experts_gate_up": gate_up,
        "experts_down": down,
    }


def convert_norm_weights(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Decoder-layer RMSNorm weights. Plain RMSNorm -- no zero-centering."""
    return {
        "input_layernorm": sd["input_layernorm.weight"].float(),
        "post_attention_layernorm": sd["post_attention_layernorm.weight"].float(),
    }


def convert_layer_weights(sd: dict[str, torch.Tensor], config) -> dict[str, torch.Tensor]:
    """Full layer conversion: attention + MoE + norms."""
    weights = convert_attention_weights(
        sd,
        n_heads=config.num_attention_heads,
        n_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
    )
    weights.update(convert_moe_weights(sd, n_experts=config.num_experts))
    weights.update(convert_norm_weights(sd))
    return weights
