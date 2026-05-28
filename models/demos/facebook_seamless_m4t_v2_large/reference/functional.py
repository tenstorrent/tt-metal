# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Standalone PyTorch reference implementations for SeamlessM4T-v2-Large blocks.

These functions are pure-PyTorch (no TTNN imports) and are used to generate
golden outputs for TTNN block verification. Each function should be
bit-equivalent to the corresponding HuggingFace `transformers` module forward
pass when run in float32.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def layernorm_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Standard `torch.nn.LayerNorm` forward pass over the last dim.

    SeamlessM4T-v2 uses this everywhere with `layer_norm_eps=1e-5`:
        - Conformer feature_projection LayerNorm(160)
        - Conformer encoder layer macaron / attn / final LayerNorm(1024)
        - Conformer adapter LayerNorm(1024)
        - Conformer conv module LayerNorm(1024) / depthwise_layer_norm
        - NLLB text encoder/decoder layer norms and final LayerNorm
        - T2U encoder/decoder layer norms and final LayerNorm
        - VariancePredictor LayerNorm

    Args:
        x: Input tensor of shape (..., dim).
        weight: LayerNorm gamma of shape (dim,).
        bias: LayerNorm beta of shape (dim,).
        eps: Numerical stability epsilon (default 1e-5; matches
            `SeamlessM4Tv2Config.layer_norm_eps`).

    Returns:
        Tensor of the same shape as `x`, normalized over the last dim.
    """
    normalized_shape = (x.shape[-1],)
    return F.layer_norm(x, normalized_shape, weight=weight, bias=bias, eps=eps)


def scaled_word_embedding_forward(
    input_ids: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    padding_idx: Optional[int] = None,
) -> torch.Tensor:
    """`SeamlessM4Tv2ScaledWordEmbedding` forward pass.

    Implements the NLLB / M2M100 style scaled embedding used by the text
    encoder, text decoder, and T2U decoder:

        out = F.embedding(input_ids, weight, padding_idx=padding_idx) * scale

    where ``scale`` is ``sqrt(hidden_size)`` when ``config.scale_embedding``
    is True (the default for SeamlessM4T-v2). For the large model
    ``hidden_size = 1024`` so ``scale = sqrt(1024) = 32.0``.

    The HF class is
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2ScaledWordEmbedding``
    (copied from ``M2M100ScaledWordEmbedding``). It subclasses ``nn.Embedding``
    and overrides ``forward`` to multiply the gathered embeddings by
    ``self.embed_scale``.

    Args:
        input_ids: Integer token-id tensor of shape ``(batch, seq_len)`` or any
            shape — output adds a trailing ``embedding_dim`` axis.
        weight: Embedding table of shape ``(vocab_size, embedding_dim)``.
        scale: Scalar multiplier applied to gathered embeddings
            (``math.sqrt(hidden_size)`` for SeamlessM4T-v2).
        padding_idx: Optional padding index, matching the underlying
            ``nn.Embedding(padding_idx=...)`` semantics. The forward output
            is unaffected for non-pad ids; included for parity with HF.

    Returns:
        Tensor of shape ``(*input_ids.shape, embedding_dim)``.
    """
    return F.embedding(input_ids, weight, padding_idx=padding_idx) * scale


def build_sinusoidal_positional_embedding_weights(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: Optional[int] = None,
) -> torch.Tensor:
    """Build the NLLB-style sinusoidal positional embedding lookup table.

    This is a verbatim port of
    ``SeamlessM4Tv2SinusoidalPositionalEmbedding.get_embedding`` from the
    HuggingFace ``transformers`` library
    (``transformers/models/seamless_m4t_v2/modeling_seamless_m4t_v2.py``).
    It matches the tensor2tensor convention (concat ``sin || cos`` along the
    last dim) and zeros the row at ``padding_idx`` when provided.

    Args:
        num_embeddings: Number of positions to materialize (including the
            offset of 2 reserved for ``padding_idx`` and the "no-position"
            slot, plus any caller-side margin).
        embedding_dim: Hidden size for the embedding (must match the model's
            ``hidden_size``; e.g. 1024 for SeamlessM4T-v2 text encoder /
            decoder / T2U encoder / T2U decoder).
        padding_idx: Optional padding index (default 1 for SeamlessM4T-v2);
            the corresponding row is zeroed.

    Returns:
        Tensor of shape ``(num_embeddings, embedding_dim)`` in the default
        dtype (typically ``torch.float32``).
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
    emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
    if embedding_dim % 2 == 1:
        # zero pad to match the requested embedding_dim
        emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
    if padding_idx is not None:
        emb[padding_idx, :] = 0
    return emb.to(torch.get_default_dtype())


def _create_position_ids_from_input_ids(
    input_ids: torch.Tensor,
    padding_idx: int,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    """Replace non-padding tokens with their (offset) position indices.

    Mirrors ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2
    .create_position_ids_from_input_ids``. Padding positions keep
    ``padding_idx``; non-padding tokens at column ``c`` (after skipping pads
    via a per-row cumulative sum) map to
    ``padding_idx + 1 + c + past_key_values_length``.
    """
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx


def sinusoidal_positional_embedding_forward(
    weights: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    padding_idx: int = 1,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    """Standalone forward pass for ``SeamlessM4Tv2SinusoidalPositionalEmbedding``.

    Given a precomputed sinusoidal weight table (typically produced by
    :func:`build_sinusoidal_positional_embedding_weights`), reproduces the
    NLLB-style padding-aware lookup used by the SeamlessM4T-v2 text
    encoder / decoder and T2U encoder / decoder.

    Exactly one of ``input_ids`` or ``inputs_embeds`` must be provided.

    Args:
        weights: Sinusoidal weight table of shape
            ``(num_embeddings, embedding_dim)``. The caller is responsible
            for sizing this so ``num_embeddings >= padding_idx + 1 + seq_len
            + past_key_values_length``.
        input_ids: Integer token ids of shape ``(batch, seq_len)``. When
            provided, positions are computed from the non-padding mask
            (padding tokens keep ``padding_idx``).
        inputs_embeds: Float tensor of shape ``(batch, seq_len, hidden)``.
            When provided, positions are sequential
            (``padding_idx + 1 .. padding_idx + seq_len``) without padding
            awareness.
        padding_idx: Padding token id (default 1, matching
            SeamlessM4T-v2).
        past_key_values_length: Number of cached tokens already consumed
            (used during decoder incremental decoding to offset positions).

    Returns:
        Tensor of shape ``(batch, seq_len, embedding_dim)``.
    """
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Exactly one of input_ids or inputs_embeds must be provided.")

    if input_ids is not None:
        bsz, seq_len = input_ids.size()
        position_ids = _create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length).to(
            input_ids.device
        )
    else:
        bsz, seq_len = inputs_embeds.size()[:-1]
        position_ids = torch.arange(
            padding_idx + 1,
            seq_len + padding_idx + 1,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        position_ids = position_ids.unsqueeze(0).expand((bsz, seq_len)).contiguous() + past_key_values_length

    max_pos = padding_idx + 1 + seq_len + past_key_values_length
    if max_pos > weights.size(0):
        raise ValueError(
            f"Sinusoidal weight table too small: have {weights.size(0)} rows, "
            f"need at least {max_pos} for seq_len={seq_len}, "
            f"padding_idx={padding_idx}, past_key_values_length={past_key_values_length}."
        )

    return weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, weights.shape[-1])


def seamless_mha_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, Dict[str, torch.Tensor]],
    num_heads: int,
    head_dim: int,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standalone ``SeamlessM4Tv2Attention`` forward pass (BART-style MHA w/ bias).

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2Attention``
    for the un-cached (no KV-cache) path. Supports BOTH self-attention
    (``encoder_hidden_states is None``) and cross-attention
    (``encoder_hidden_states`` provided -> keys/values are projected from it).

    The op sequence intentionally matches HF v2 exactly:

        Q = q_proj(hidden_states)
        K = k_proj(current_states); V = v_proj(current_states)
        Q, K, V reshaped to ``[batch, num_heads, seq, head_dim]``
        Q = Q * (head_dim ** -0.5)
        scores = Q @ K^T  (shape ``[batch, num_heads, tgt_len, src_len]``)
        scores += attention_mask  (if provided; additive log-mask)
        attn_weights = softmax(scores.float(), dim=-1).type_as(scores)
        out = attn_weights @ V  -> permute -> reshape -> out_proj

    ``attention_mask`` is the standard additive log-mask broadcast-compatible
    with ``[batch, 1, tgt_len, src_len]`` (e.g. ``0.0`` for valid positions
    and a large negative number / ``-inf`` for masked positions).

    Note: the softmax is computed in float32 then cast back to the activation
    dtype (matches HF v2). For a fp32 reference this is a no-op.

    Args:
        hidden_states: Query input, shape ``[batch, tgt_len, embed_dim]``.
        state_dict: Mapping from ``q_proj`` / ``k_proj`` / ``v_proj`` /
            ``out_proj`` to sub-dicts containing ``weight`` (shape
            ``[embed_dim, embed_dim]``) and optional ``bias`` (shape
            ``[embed_dim]``). SeamlessM4Tv2 uses ``bias=True`` everywhere.
        num_heads: Number of attention heads.
        head_dim: Per-head dimension. Must satisfy
            ``num_heads * head_dim == embed_dim``.
        encoder_hidden_states: Optional cross-attention K/V source, shape
            ``[batch, src_len, embed_dim]``. If ``None``, runs self-attention.
        attention_mask: Optional additive log-mask broadcast-compatible with
            ``[batch, 1, tgt_len, src_len]``.

    Returns:
        Output tensor of shape ``[batch, tgt_len, embed_dim]``.
    """
    batch_size, tgt_len, embed_dim = hidden_states.shape
    if num_heads * head_dim != embed_dim:
        raise ValueError(f"num_heads({num_heads}) * head_dim({head_dim}) != embed_dim({embed_dim})")
    scaling = head_dim**-0.5

    current_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

    # Project K, V from current_states (encoder for cross-attn, self for self-attn)
    key_states = F.linear(
        current_states,
        state_dict["k_proj"]["weight"],
        state_dict["k_proj"].get("bias"),
    )
    value_states = F.linear(
        current_states,
        state_dict["v_proj"]["weight"],
        state_dict["v_proj"].get("bias"),
    )
    # Reshape K, V to [batch, num_heads, src_len, head_dim]
    key_states = key_states.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

    # Project Q from hidden_states; reshape to [batch, num_heads, tgt_len, head_dim]
    query_states = F.linear(
        hidden_states,
        state_dict["q_proj"]["weight"],
        state_dict["q_proj"].get("bias"),
    )
    query_states = query_states.reshape(batch_size, tgt_len, num_heads, head_dim).transpose(1, 2)

    # Scale Q (HF v2 scales after projection/reshape, matches BART behavior)
    query_states = query_states * scaling

    # Attention scores: [batch, num_heads, tgt_len, src_len]
    attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2))
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    # Softmax in float32 then cast back to scores dtype (HF v2 idiom)
    attn_weights = F.softmax(attention_scores.float(), dim=-1).type_as(attention_scores)

    # Context: [batch, num_heads, tgt_len, head_dim] -> [batch, tgt_len, embed_dim]
    context_states = torch.matmul(attn_weights, value_states)
    context_states = context_states.permute(0, 2, 1, 3).contiguous().view(batch_size, tgt_len, -1)

    # Output projection
    attn_output = F.linear(
        context_states,
        state_dict["out_proj"]["weight"],
        state_dict["out_proj"].get("bias"),
    )
    return attn_output
