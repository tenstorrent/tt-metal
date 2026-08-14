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


def hf_to_meta_channels(head_dim: int) -> torch.Tensor:
    """Index vector mapping an HF-ordered head to Meta (llama) channel order.

    HF's ``rotate_half`` pairs channel ``i`` with ``i + head_dim/2``; the Meta
    convention that ``rotary_embedding_llama``'s transformation matrix encodes
    pairs ``2i`` with ``2i + 1``. So ``meta[2i] = hf[i]`` and
    ``meta[2i+1] = hf[i + head_dim/2]``, i.e. an interleave of the two halves.

    The *same* vector converts a cos/sin row, because HF stores
    ``[c0 .. c_{d/2-1}, c0 .. c_{d/2-1}]`` and Meta stores ``[c0, c0, c1, c1, ...]``.

    Applied per head, so it commutes with any reordering of whole heads --
    which is what lets ``multichip_decoder`` apply it before
    ``head_interleaved_wqkv`` splits the heads across dies.
    """
    assert head_dim % 2 == 0, head_dim
    half = head_dim // 2
    return torch.stack([torch.arange(half), torch.arange(half) + half], dim=1).reshape(-1)


def permute_wqkv_to_meta(wqkv: torch.Tensor, *, n_heads: int, n_kv_heads: int, head_dim: int) -> torch.Tensor:
    """Reorder the Q and K channels of a fused ``[..., in, out]`` wqkv to Meta order.

    **V is deliberately untouched** -- RoPE is applied to Q and K only, so V's
    channels keep their HF meaning, and so does ``wo``, which consumes the
    attention output in V's space.

    This is the ``reverse_permute`` step the module docstring says comes back
    the moment the llama rotary op is used. It is a *checkpoint layout* change
    and is applied once at upload, never per token.
    """
    out = wqkv.clone()
    perm = hf_to_meta_channels(head_dim)
    q_end = n_heads * head_dim
    k_end = q_end + n_kv_heads * head_dim
    for start, count in ((0, n_heads), (q_end, n_kv_heads)):
        for h in range(count):
            lo = start + h * head_dim
            out[..., lo : lo + head_dim] = wqkv[..., lo : lo + head_dim][..., perm]
    assert torch.equal(out[..., k_end:], wqkv[..., k_end:]), "V must not be permuted"
    return out


def permute_head_vector_to_meta(vec: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    """Reorder a per-head vector (Qwen3's ``q_norm`` / ``k_norm``) to Meta order.

    Qwen3 applies these **between the head split and RoPE**, so they index the
    same channels the rotary op does. Permuting Q/K without permuting these
    scales the wrong channel and is silent -- it does not change any shape and
    it does not raise. ``test_meta_rope_weights_match_hf`` is the assertion that
    catches it.
    """
    flat = vec.reshape(-1)
    assert flat.numel() == head_dim, (flat.shape, head_dim)
    return flat[hf_to_meta_channels(head_dim)].reshape(vec.shape)


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
