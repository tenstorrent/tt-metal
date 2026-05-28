# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Standalone PyTorch reference implementations for SeamlessM4T-v2-Large blocks.

These functions are pure-PyTorch (no TTNN imports) and are used to generate
golden outputs for TTNN block verification. Each function should be
bit-equivalent to the corresponding HuggingFace `transformers` module forward
pass when run in float32.
"""

import math
from typing import Dict, Optional, Sequence

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


def seamless_ffn_forward(
    x: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Standalone ``SeamlessM4Tv2FeedForwardNetwork`` forward pass.

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2FeedForwardNetwork``:

        h = fc1(x)            # Linear(hidden_size, ffn_dim) + bias
        h = act(h)             # ReLU (config.activation_function = "relu")
        h = dropout(h)         # config.activation_dropout (default 0.0)
        h = fc2(h)             # Linear(ffn_dim, hidden_size) + bias

    This block is reused throughout SeamlessM4T-v2:
        - NLLB text encoder layer FFN
        - NLLB text decoder layer FFN
        - T2U encoder / decoder layer FFN

    For the ``-large`` model: ``hidden_size = 1024``, ``ffn_dim = 8192``,
    bias is enabled on both projections, activation is ReLU, and
    ``activation_dropout = 0.0``.

    Note: the HF forward additionally casts ``hidden_states`` to
    ``self.fc2.weight.dtype`` before the final projection (handles mixed
    int8/uint8 weight types). For an fp32 reference with fp32 weights this is
    a no-op, so we omit it here.

    Args:
        x: Input tensor of shape ``[..., hidden_size]``.
        fc1_weight: ``[ffn_dim, hidden_size]`` weight of the first linear.
        fc1_bias: ``[ffn_dim]`` bias of the first linear.
        fc2_weight: ``[hidden_size, ffn_dim]`` weight of the second linear.
        fc2_bias: ``[hidden_size]`` bias of the second linear.
        dropout_p: Dropout probability applied to the post-activation
            hidden states (default 0.0, matching
            ``SeamlessM4Tv2Config.activation_dropout``). Dropout is only
            applied in training mode; this function uses ``training=False``
            so it is effectively a no-op for verification.

    Returns:
        Tensor of the same shape as ``x``.
    """
    h = F.linear(x, fc1_weight, fc1_bias)
    h = F.relu(h)
    if dropout_p > 0.0:
        # training=False -> identity, included for parity with HF signature.
        h = F.dropout(h, p=dropout_p, training=False)
    return F.linear(h, fc2_weight, fc2_bias)


def conformer_convolution_module_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, Dict[str, torch.Tensor]],
    kernel_size: int = 31,
    eps: float = 1e-5,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standalone ``SeamlessM4Tv2ConformerConvolutionModule`` forward pass.

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2ConformerConvolutionModule``
    bit-for-bit (no dropout, default ``activation = swish = SiLU``):

        x = LayerNorm(x)                              # (B, T, C)
        # optional mask zeroing
        x = x.transpose(1, 2)                         # (B, C, T)
        x = pointwise_conv1(x)                        # (B, 2C, T), no bias, k=1
        x = GLU(dim=1)(x)                             # (B, C, T)
        x = F.pad(x, (kernel_size - 1, 0))            # left-only causal pad
        x = depthwise_conv(x)                         # (B, C, T), groups=C, k=31, no bias
        x = LayerNorm(x.transpose(1, 2)).transpose(1, 2)  # (B, C, T)
        x = swish(x)
        x = pointwise_conv2(x)                        # (B, C, T), no bias, k=1
        return x.transpose(1, 2)                      # (B, T, C)

    All three Conv1d layers have ``bias=False``. The depthwise Conv1d has
    ``groups = hidden_size`` and is CAUSAL (left padding ``kernel_size - 1``,
    right padding 0). ``layer_norm`` and ``depthwise_layer_norm`` are
    ``nn.LayerNorm`` over the channel dim with eps 1e-5.

    For SeamlessM4T-v2-Large: ``hidden_size = 1024`` and
    ``conv_depthwise_kernel_size = 31``.

    Args:
        hidden_states: Input tensor of shape ``(B, T, hidden_size)``.
        state_dict: Mapping with the following entries:
            - ``"layer_norm"``: dict with ``"weight"``, ``"bias"`` (shape
              ``(hidden_size,)``).
            - ``"pointwise_conv1"``: dict with ``"weight"`` of shape
              ``(2 * hidden_size, hidden_size, 1)``.
            - ``"depthwise_conv"``: dict with ``"weight"`` of shape
              ``(hidden_size, 1, kernel_size)`` (groups=hidden_size).
            - ``"depthwise_layer_norm"``: dict with ``"weight"``, ``"bias"``
              (shape ``(hidden_size,)``).
            - ``"pointwise_conv2"``: dict with ``"weight"`` of shape
              ``(hidden_size, hidden_size, 1)``.
        kernel_size: Depthwise conv kernel size (default 31, matches
            ``SeamlessM4Tv2Config.conv_depthwise_kernel_size``).
        eps: LayerNorm epsilon (default 1e-5, matches
            ``SeamlessM4Tv2Config.layer_norm_eps``).
        attention_mask: Optional bool/int mask of shape ``(B, T)`` where
            ``False/0`` indicates a padded position to zero out before the
            depthwise convolution (mirrors HF ``masked_fill`` step).

    Returns:
        Tensor of shape ``(B, T, hidden_size)``.
    """
    # 1. LayerNorm over channels.
    ln_weight = state_dict["layer_norm"]["weight"]
    ln_bias = state_dict["layer_norm"]["bias"]
    x = F.layer_norm(hidden_states, (hidden_states.shape[-1],), weight=ln_weight, bias=ln_bias, eps=eps)

    # 2. Optional mask: zero padded positions BEFORE depthwise conv.
    if attention_mask is not None:
        x = x.masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)

    # 3. Transpose to channel-first for Conv1d: (B, T, C) -> (B, C, T).
    x = x.transpose(1, 2)

    # 4. Pointwise conv1 (1x1, bias=False) -> (B, 2C, T).
    x = F.conv1d(x, state_dict["pointwise_conv1"]["weight"], bias=None)

    # 5. GLU along channel dim: (B, 2C, T) -> (B, C, T).
    x = F.glu(x, dim=1)

    # 6. Causal left-padding by (kernel_size - 1, 0).
    x = F.pad(x, (kernel_size - 1, 0))

    # 7. Depthwise conv (groups=channels, bias=False).
    channels = x.shape[1]
    x = F.conv1d(x, state_dict["depthwise_conv"]["weight"], bias=None, groups=channels)

    # 8. LayerNorm over channels (transpose to last dim, normalize, transpose back).
    dwln_weight = state_dict["depthwise_layer_norm"]["weight"]
    dwln_bias = state_dict["depthwise_layer_norm"]["bias"]
    x = x.transpose(1, 2)
    x = F.layer_norm(x, (x.shape[-1],), weight=dwln_weight, bias=dwln_bias, eps=eps)
    x = x.transpose(1, 2)

    # 9. Swish (= SiLU) activation.
    x = F.silu(x)

    # 10. Pointwise conv2 (1x1, bias=False).
    x = F.conv1d(x, state_dict["pointwise_conv2"]["weight"], bias=None)

    # 11. Back to (B, T, C).
    return x.transpose(1, 2)


def conformer_ffn_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, Dict[str, torch.Tensor]],
    act_fn: str = "swish",
) -> torch.Tensor:
    """Standalone ``SeamlessM4Tv2ConformerFeedForward`` forward pass.

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2ConformerFeedForward``
    used inside the W2v-BERT-2.0-style Conformer speech encoder. Each Conformer
    encoder layer contains two of these (``ffn1`` / ``ffn2``), wrapped at the
    parent layer level in the half-step residual pattern
    ``residual + 0.5 * FFN(LayerNorm(residual))``. This function implements
    ONLY the inner FFN (the surrounding LayerNorm + 0.5 scaling + residual
    live in :class:`SeamlessM4Tv2ConformerEncoderLayer`).

    Op sequence (matches HF v2 exactly; dropouts are no-ops at eval):

        h = intermediate_dense(hidden_states)   # Linear(d, 4d), bias=True
        h = swish(h)                            # i.e. SiLU
        h = output_dense(h)                     # Linear(4d, d), bias=True

    For SeamlessM4T-v2-Large the dims are ``d = config.hidden_size = 1024``
    and ``4d = config.speech_encoder_intermediate_size = 4096``. The activation
    is selected by ``config.speech_encoder_hidden_act`` which defaults to
    ``"swish"`` (== ``"silu"``).

    NOTE: this is the Conformer (speech encoder) FFN — it uses SiLU/swish,
    NOT ReLU. The NLLB text encoder/decoder and T2U FFNs use
    :func:`seamless_ffn_forward` with ReLU instead.

    Args:
        hidden_states: Input tensor of shape ``[batch, seq_len, hidden_size]``.
        state_dict: Mapping with keys ``intermediate_dense`` and
            ``output_dense``, each containing ``weight`` (shape
            ``[out_features, in_features]``) and ``bias`` (shape
            ``[out_features]``). HF uses ``bias=True`` on both Linear layers.
        act_fn: Activation function name. ``"swish"`` and ``"silu"`` both map
            to ``F.silu`` (SeamlessM4T-v2 default). ``"gelu"`` and ``"relu"``
            are also accepted for completeness with other ACT2FN names.

    Returns:
        Tensor of shape ``[batch, seq_len, hidden_size]``.
    """
    h = F.linear(
        hidden_states,
        state_dict["intermediate_dense"]["weight"],
        state_dict["intermediate_dense"].get("bias"),
    )

    if act_fn in ("swish", "silu"):
        h = F.silu(h)
    elif act_fn == "gelu":
        h = F.gelu(h)
    elif act_fn == "relu":
        h = F.relu(h)
    else:
        raise ValueError(f"Unsupported activation: {act_fn!r}")

    h = F.linear(
        h,
        state_dict["output_dense"]["weight"],
        state_dict["output_dense"].get("bias"),
    )
    return h


def conformer_self_attention_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, Dict[str, torch.Tensor]],
    num_heads: int,
    head_dim: int,
    distance_embedding_weight: Optional[torch.Tensor] = None,
    left_max_position_embeddings: int = 64,
    right_max_position_embeddings: int = 8,
    position_embeddings_type: Optional[str] = "relative_key",
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standalone ``SeamlessM4Tv2ConformerSelfAttention`` forward pass.

    Reproduces the forward pass of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2
    .SeamlessM4Tv2ConformerSelfAttention`` for the un-cached path. This is
    the Conformer speech-encoder MHA used in the speech encoder layers of
    SeamlessM4T-v2. It is identical to a standard 4-projection MHA except
    for an optional ``relative_key`` positional bias term added to the
    attention logits.

    The op sequence intentionally matches HF exactly:

        Q = linear_q(x).view(B, T, H, D).transpose(1, 2)
        K = linear_k(x).view(B, T, H, D).transpose(1, 2)
        V = linear_v(x).view(B, T, H, D).transpose(1, 2)

        scores = (Q @ K^T) / sqrt(D)

        if position_embeddings_type == "relative_key":
            distance = clamp(
                arange(K)[None] - arange(Q)[:, None],
                -left_max_position_embeddings,
                +right_max_position_embeddings,
            )                                                       # [T, T]
            P = distance_embedding(distance + left_max_position_embeddings)  # [T, T, D]
            rel = einsum("bhld,lrd->bhlr", Q, P)
            scores = scores + rel / sqrt(D)

        scores += attention_mask  (if provided; additive log-mask)
        attn   = softmax(scores, dim=-1)
        out    = (attn @ V).transpose(1, 2).reshape(B, T, H*D)
        out    = linear_out(out)

    For SeamlessM4T-v2-Large the relevant config values are:
        - hidden_size = 1024
        - speech_encoder_attention_heads = 16  -> head_dim = 64
        - position_embeddings_type = "relative_key"
        - left_max_position_embeddings = 64
        - right_max_position_embeddings = 8
        - speech_encoder_dropout = 0.0
    so the dropout on attention weights is a no-op and is omitted here.

    Args:
        hidden_states: Self-attention input, shape ``[batch, seq_len, hidden]``.
        state_dict: Mapping from ``linear_q`` / ``linear_k`` / ``linear_v`` /
            ``linear_out`` to sub-dicts with ``weight`` (shape
            ``[hidden, hidden]``) and optional ``bias`` (shape
            ``[hidden]``). HF uses ``bias=True`` everywhere.
        num_heads: Number of attention heads (16 for v2-Large).
        head_dim: Per-head dimension (64 for v2-Large). Must satisfy
            ``num_heads * head_dim == hidden_size``.
        distance_embedding_weight: ``[L + R + 1, head_dim]`` learned table
            for the ``relative_key`` positional bias. Required when
            ``position_embeddings_type == "relative_key"``.
        left_max_position_embeddings: ``L`` clamp range on the left side
            (matches ``config.left_max_position_embeddings``; 64 for v2-Large).
        right_max_position_embeddings: ``R`` clamp range on the right side
            (matches ``config.right_max_position_embeddings``; 8 for v2-Large).
        position_embeddings_type: Either ``"relative_key"`` (add relative
            position bias) or ``None`` (standard MHA, no positional term).
        attention_mask: Optional additive log-mask broadcast-compatible
            with ``[batch, 1, seq_len, seq_len]``.

    Returns:
        Output tensor of shape ``[batch, seq_len, hidden_size]``.
    """
    batch_size, seq_len, hidden = hidden_states.shape
    if num_heads * head_dim != hidden:
        raise ValueError(f"num_heads({num_heads}) * head_dim({head_dim}) != hidden({hidden})")

    # Project Q/K/V and reshape to [batch, num_heads, seq, head_dim].
    query = (
        F.linear(
            hidden_states,
            state_dict["linear_q"]["weight"],
            state_dict["linear_q"].get("bias"),
        )
        .view(batch_size, -1, num_heads, head_dim)
        .transpose(1, 2)
    )
    key = (
        F.linear(
            hidden_states,
            state_dict["linear_k"]["weight"],
            state_dict["linear_k"].get("bias"),
        )
        .view(batch_size, -1, num_heads, head_dim)
        .transpose(1, 2)
    )
    value = (
        F.linear(
            hidden_states,
            state_dict["linear_v"]["weight"],
            state_dict["linear_v"].get("bias"),
        )
        .view(batch_size, -1, num_heads, head_dim)
        .transpose(1, 2)
    )

    # Content-based scaled dot-product scores: [batch, num_heads, seq, seq].
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)

    if position_embeddings_type == "relative_key":
        if distance_embedding_weight is None:
            raise ValueError("distance_embedding_weight is required when position_embeddings_type='relative_key'")
        query_length, key_length = query.shape[2], key.shape[2]
        position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        distance = position_ids_r - position_ids_l
        distance = torch.clamp(distance, -left_max_position_embeddings, right_max_position_embeddings)

        # Gather from learned table [L+R+1, head_dim] -> [query_len, key_len, head_dim].
        positional_embedding = F.embedding(distance + left_max_position_embeddings, distance_embedding_weight)
        positional_embedding = positional_embedding.to(dtype=query.dtype)

        relative_position_attn_weights = torch.einsum("bhld,lrd->bhlr", query, positional_embedding)
        attn_weights = attn_weights + (relative_position_attn_weights / math.sqrt(head_dim))

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.softmax(attn_weights, dim=-1)
    # Dropout omitted: speech_encoder_dropout=0.0 in v2-Large config.

    # Context: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden].
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, num_heads * head_dim)

    return F.linear(
        attn_output,
        state_dict["linear_out"]["weight"],
        state_dict["linear_out"].get("bias"),
    )


def hifigan_residual_block_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict,
    kernel_size: int = 3,
    dilation=(1, 3, 5),
    leaky_relu_slope: float = 0.1,
) -> torch.Tensor:
    """`HifiGanResidualBlock` forward pass (HiFi-GAN MRF residual block).

    Bit-equivalent (in fp32) to
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.HifiGanResidualBlock``
    (itself copied from ``speecht5``). This is one of the multi-receptive-field
    (MRF) residual blocks inside the HiFi-GAN vocoder used by SeamlessM4T-v2.

    Structure (``N = len(dilation)`` pairs):
        for i in range(N):
            residual = hidden_states
            hidden_states = leaky_relu(hidden_states, slope)
            hidden_states = conv1[i](hidden_states)      # Conv1d k=kernel_size, dilation=dilation[i]
            hidden_states = leaky_relu(hidden_states, slope)
            hidden_states = conv2[i](hidden_states)      # Conv1d k=kernel_size, dilation=1
            hidden_states = hidden_states + residual
        return hidden_states

    All Conv1d layers use ``stride=1`` and "same" padding computed as
    ``padding = (kernel_size * dilation - dilation) // 2`` so the time dimension
    is preserved.

    Args:
        hidden_states: Input tensor of shape ``[batch, channels, time]``.
        state_dict: Dict with two parallel lists of conv weights:
            - ``state_dict["convs1"][i] = {"weight": [C, C, k], "bias": [C]}``
              with dilation ``dilation[i]``.
            - ``state_dict["convs2"][i] = {"weight": [C, C, k], "bias": [C]}``
              with dilation 1.
            ``len(convs1) == len(convs2) == len(dilation)``.
        kernel_size: Conv kernel size (default 3, matches HF default).
        dilation: Tuple of dilation factors for ``convs1`` (default (1, 3, 5)).
        leaky_relu_slope: Negative slope for LeakyReLU (default 0.1, matches
            ``SeamlessM4Tv2Config.leaky_relu_slope``).

    Returns:
        Tensor of the same shape as ``hidden_states``.
    """
    convs1 = state_dict["convs1"]
    convs2 = state_dict["convs2"]
    assert len(convs1) == len(convs2) == len(dilation), (
        f"convs1/convs2/dilation length mismatch: " f"{len(convs1)} vs {len(convs2)} vs {len(dilation)}"
    )

    def _same_padding(k: int, d: int) -> int:
        return (k * d - d) // 2

    for i, d in enumerate(dilation):
        residual = hidden_states
        hidden_states = F.leaky_relu(hidden_states, negative_slope=leaky_relu_slope)
        hidden_states = F.conv1d(
            hidden_states,
            convs1[i]["weight"],
            bias=convs1[i].get("bias"),
            stride=1,
            padding=_same_padding(kernel_size, d),
            dilation=d,
        )
        hidden_states = F.leaky_relu(hidden_states, negative_slope=leaky_relu_slope)
        hidden_states = F.conv1d(
            hidden_states,
            convs2[i]["weight"],
            bias=convs2[i].get("bias"),
            stride=1,
            padding=_same_padding(kernel_size, 1),
            dilation=1,
        )
        hidden_states = hidden_states + residual
    return hidden_states


def variance_predictor_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, Dict[str, torch.Tensor]],
    kernel_size: int = 3,
    eps: float = 1e-5,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standalone ``SeamlessM4Tv2VariancePredictor`` forward pass.

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2
    .SeamlessM4Tv2VariancePredictor`` bit-for-bit (dropout is a no-op at eval
    mode, so it is omitted here).

    The HF module instantiates two ``nn.Conv1d(..., padding="same")`` layers
    with odd kernels (default ``kernel_size = 3``), a ``ReLU`` activation
    (hardcoded — there is NO config knob switching to SiLU), two
    ``nn.LayerNorm(hidden_dim)`` layers (eps 1e-5), and a final
    ``nn.Linear(hidden_dim, 1)``. The output is squeezed to drop the
    singleton channel dim (``B x T x 1`` -> ``B x T``).

    Op sequence (matches HF exactly):

        if padding_mask is not None:
            x = x.masked_fill(~padding_mask.unsqueeze(-1), 0)        # (B, T, C)
        x = conv1(x.transpose(1, 2))                                 # (B, H, T)
        x = relu(x).transpose(1, 2)                                  # (B, T, H)
        x = ln1(x)                                                   # (B, T, H)
        if padding_mask is not None:
            x = x.masked_fill(~padding_mask.unsqueeze(-1), 0)        # (B, T, H)
        x = conv2(x.transpose(1, 2))                                 # (B, H, T)
        x = relu(x).transpose(1, 2)                                  # (B, T, H)
        x = ln2(x)                                                   # (B, T, H)
        return proj(x).squeeze(-1)                                   # (B, T)

    Used in two places inside SeamlessM4T-v2:
        - T2U char-duration variance predictor
          (``embed_dim=1024``, ``hidden_dim=256``, ``kernel_size=3``)
        - Vocoder duration predictor
          (``embed_dim=1280``, ``hidden_dim=1280``, ``kernel_size=3``)

    Both Conv1d layers use ``padding="same"`` with odd kernels, which is
    equivalent to symmetric padding of ``(kernel_size - 1) // 2`` on each
    side — reproduced here via ``F.conv1d(..., padding=kernel_size // 2)``.
    Both Conv1d layers have ``bias=True``.

    Args:
        hidden_states: Input tensor of shape ``(B, T, embed_dim)``.
        state_dict: Mapping with the following entries:
            - ``"conv1"``: dict with ``"weight"`` of shape
              ``(hidden_dim, embed_dim, kernel_size)`` and ``"bias"`` of shape
              ``(hidden_dim,)``.
            - ``"ln1"``: dict with ``"weight"``, ``"bias"`` (shape
              ``(hidden_dim,)``).
            - ``"conv2"``: dict with ``"weight"`` of shape
              ``(hidden_dim, hidden_dim, kernel_size)`` and ``"bias"`` of
              shape ``(hidden_dim,)``.
            - ``"ln2"``: dict with ``"weight"``, ``"bias"`` (shape
              ``(hidden_dim,)``).
            - ``"proj"``: dict with ``"weight"`` of shape
              ``(1, hidden_dim)`` and ``"bias"`` of shape ``(1,)``.
        kernel_size: Conv1d kernel size (must be odd; default 3, matches
            ``SeamlessM4Tv2Config.variance_predictor_kernel_size`` and
            ``t2u_variance_predictor_kernel_size``).
        eps: LayerNorm epsilon (default 1e-5, matches
            ``SeamlessM4Tv2Config.layer_norm_eps``).
        padding_mask: Optional bool/int mask of shape ``(B, T)`` where
            ``False/0`` indicates a padded position to zero out before each
            Conv1d (mirrors the HF ``masked_fill`` step).

    Returns:
        Tensor of shape ``(B, T)``.
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd for padding='same', got {kernel_size}")
    pad = kernel_size // 2

    # 1. Optional mask: zero padded positions BEFORE conv1.
    x = hidden_states
    if padding_mask is not None:
        x = x.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)

    # 2. Conv1 on (B, C, T); ReLU; back to (B, T, H).
    x = F.conv1d(
        x.transpose(1, 2),
        state_dict["conv1"]["weight"],
        state_dict["conv1"].get("bias"),
        padding=pad,
    )
    x = F.relu(x).transpose(1, 2)

    # 3. LayerNorm 1 over the hidden_dim axis (dropout is a no-op at eval).
    x = F.layer_norm(
        x,
        (x.shape[-1],),
        weight=state_dict["ln1"]["weight"],
        bias=state_dict["ln1"]["bias"],
        eps=eps,
    )

    # 4. Optional mask: zero padded positions BEFORE conv2.
    if padding_mask is not None:
        x = x.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)

    # 5. Conv2 on (B, H, T); ReLU; back to (B, T, H).
    x = F.conv1d(
        x.transpose(1, 2),
        state_dict["conv2"]["weight"],
        state_dict["conv2"].get("bias"),
        padding=pad,
    )
    x = F.relu(x).transpose(1, 2)

    # 6. LayerNorm 2 over the hidden_dim axis.
    x = F.layer_norm(
        x,
        (x.shape[-1],),
        weight=state_dict["ln2"]["weight"],
        bias=state_dict["ln2"]["bias"],
        eps=eps,
    )

    # 7. Final projection to scalar per time step; squeeze trailing dim.
    x = F.linear(
        x,
        state_dict["proj"]["weight"],
        state_dict["proj"].get("bias"),
    )
    return x.squeeze(dim=2)


def conformer_encoder_layer_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, Dict[str, torch.Tensor]],
    num_heads: int,
    head_dim: int,
    distance_embedding_weight: Optional[torch.Tensor] = None,
    left_max_position_embeddings: int = 64,
    right_max_position_embeddings: int = 8,
    position_embeddings_type: Optional[str] = "relative_key",
    conv_kernel_size: int = 31,
    act_fn: str = "swish",
    eps: float = 1e-5,
    attention_mask: Optional[torch.Tensor] = None,
    conv_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standalone ``SeamlessM4Tv2ConformerEncoderLayer`` forward pass.

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2
    .SeamlessM4Tv2ConformerEncoderLayer`` bit-for-bit (no dropout — both
    ``speech_encoder_dropout`` and ``activation_dropout`` are 0.0 for v2-Large).

    This is one full Conformer block of the W2v-BERT-2.0 speech encoder. It
    uses the macaron (half-step FFN sandwich) pattern:

        residual = x
        x = ffn1_layer_norm(x)
        x = ffn1(x)
        x = x * 0.5 + residual                       # half-step FFN 1
        residual = x

        x = self_attn_layer_norm(x)
        x, _ = self_attn(x, attention_mask=attention_mask)
        x = x + residual                              # self-attention

        residual = x
        x = conv_module(x, attention_mask=conv_attention_mask)
        x = residual + x                              # convolution module

        residual = x
        x = ffn2_layer_norm(x)
        x = ffn2(x)
        x = x * 0.5 + residual                       # half-step FFN 2
        x = final_layer_norm(x)                       # final post-norm

        return x

    The inner sub-blocks are implemented by the existing reference functions:
        - LayerNorms via inline ``F.layer_norm`` (cheap, no extra dispatch).
        - ``ffn1`` / ``ffn2`` via :func:`conformer_ffn_forward`
          (SwiGLU-shaped path with SiLU/swish activation).
        - ``self_attn`` via :func:`conformer_self_attention_forward`
          (4-proj MHA with optional relative-key positional bias).
        - ``conv_module`` via :func:`conformer_convolution_module_forward`
          (causal depthwise + GLU + dual pointwise convs).

    For SeamlessM4T-v2-Large the relevant config values are:
        - hidden_size = 1024
        - speech_encoder_attention_heads = 16  -> head_dim = 64
        - position_embeddings_type = "relative_key"
        - left_max_position_embeddings = 64
        - right_max_position_embeddings = 8
        - conv_depthwise_kernel_size = 31
        - speech_encoder_hidden_act = "swish" (== SiLU)
        - layer_norm_eps = 1e-5
        - speech_encoder_dropout = 0.0
        - activation_dropout = 0.0

    Args:
        hidden_states: Input tensor of shape ``[batch, seq_len, hidden]``.
        state_dict: Nested mapping with the following top-level keys (each a
            sub-dict pulled from the matching HF submodule):
                - ``"ffn1_layer_norm"``: ``{"weight", "bias"}`` shape
                  ``(hidden,)``.
                - ``"ffn1"``: ``{"intermediate_dense", "output_dense"}`` each
                  ``{"weight", "bias"}`` — see
                  :func:`conformer_ffn_forward`.
                - ``"self_attn_layer_norm"``: ``{"weight", "bias"}`` shape
                  ``(hidden,)``.
                - ``"self_attn"``:
                  ``{"linear_q", "linear_k", "linear_v", "linear_out"}`` each
                  ``{"weight", "bias"}`` — see
                  :func:`conformer_self_attention_forward`. The
                  ``distance_embedding`` table is passed separately via the
                  ``distance_embedding_weight`` arg.
                - ``"conv_module"``:
                  ``{"layer_norm", "pointwise_conv1", "depthwise_conv",
                  "depthwise_layer_norm", "pointwise_conv2"}`` — see
                  :func:`conformer_convolution_module_forward`.
                - ``"ffn2_layer_norm"``: ``{"weight", "bias"}`` shape
                  ``(hidden,)``.
                - ``"ffn2"``: same shape as ``"ffn1"``.
                - ``"final_layer_norm"``: ``{"weight", "bias"}`` shape
                  ``(hidden,)``.
        num_heads: Number of self-attention heads (16 for v2-Large).
        head_dim: Per-head dimension (64 for v2-Large). Must satisfy
            ``num_heads * head_dim == hidden``.
        distance_embedding_weight: ``[L + R + 1, head_dim]`` learned relative
            position table; required when
            ``position_embeddings_type == "relative_key"``.
        left_max_position_embeddings: ``L`` left clamp (64 for v2-Large).
        right_max_position_embeddings: ``R`` right clamp (8 for v2-Large).
        position_embeddings_type: ``"relative_key"`` or ``None``.
        conv_kernel_size: Depthwise kernel size for the conv module
            (31 for v2-Large).
        act_fn: FFN activation name (default ``"swish"`` == SiLU).
        eps: LayerNorm epsilon (default 1e-5 for v2-Large).
        attention_mask: Optional additive log-mask broadcast-compatible with
            ``[batch, 1, seq_len, seq_len]`` for the self-attention.
        conv_attention_mask: Optional bool/int mask of shape ``(batch, seq_len)``
            passed to the conv module to zero padded positions before the
            depthwise conv.

    Returns:
        Tensor of shape ``[batch, seq_len, hidden]``.
    """
    # 1. Half-step FFN 1.
    residual = hidden_states
    x = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=state_dict["ffn1_layer_norm"]["weight"],
        bias=state_dict["ffn1_layer_norm"]["bias"],
        eps=eps,
    )
    x = conformer_ffn_forward(x, state_dict["ffn1"], act_fn=act_fn)
    x = x * 0.5 + residual
    residual = x

    # 2. Self-attention.
    x = F.layer_norm(
        x,
        (x.shape[-1],),
        weight=state_dict["self_attn_layer_norm"]["weight"],
        bias=state_dict["self_attn_layer_norm"]["bias"],
        eps=eps,
    )
    x = conformer_self_attention_forward(
        x,
        state_dict["self_attn"],
        num_heads=num_heads,
        head_dim=head_dim,
        distance_embedding_weight=distance_embedding_weight,
        left_max_position_embeddings=left_max_position_embeddings,
        right_max_position_embeddings=right_max_position_embeddings,
        position_embeddings_type=position_embeddings_type,
        attention_mask=attention_mask,
    )
    x = x + residual

    # 3. Convolution module (causal depthwise + GLU).
    residual = x
    x = conformer_convolution_module_forward(
        x,
        state_dict["conv_module"],
        kernel_size=conv_kernel_size,
        eps=eps,
        attention_mask=conv_attention_mask,
    )
    x = residual + x

    # 4. Half-step FFN 2 + final post-LayerNorm.
    residual = x
    x = F.layer_norm(
        x,
        (x.shape[-1],),
        weight=state_dict["ffn2_layer_norm"]["weight"],
        bias=state_dict["ffn2_layer_norm"]["bias"],
        eps=eps,
    )
    x = conformer_ffn_forward(x, state_dict["ffn2"], act_fn=act_fn)
    x = x * 0.5 + residual

    # 5. Final LayerNorm (post-norm).
    x = F.layer_norm(
        x,
        (x.shape[-1],),
        weight=state_dict["final_layer_norm"]["weight"],
        bias=state_dict["final_layer_norm"]["bias"],
        eps=eps,
    )
    return x


def conformer_feature_projection_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, Dict[str, torch.Tensor]],
    eps: float = 1e-5,
) -> torch.Tensor:
    """Standalone ``SeamlessM4Tv2ConformerFeatureProjection`` forward pass.

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2
    .SeamlessM4Tv2ConformerFeatureProjection`` bit-for-bit at eval time
    (dropout is a no-op). This block is the very front of the W2v-BERT-2.0
    Conformer speech encoder: it takes the raw feature-extractor output
    (``feature_projection_input_dim = 160`` for v2-Large, sourced from the
    SeamlessM4T-v2 feature extractor's 80-bin log-mel with stacked frames)
    and projects it up to ``hidden_size = 1024`` after a per-feature
    LayerNorm.

    Op sequence (matches HF exactly):

        x = LayerNorm(x)            # nn.LayerNorm(feature_projection_input_dim, eps=1e-5)
        x = Linear(x)               # nn.Linear(feature_projection_input_dim, hidden_size), bias=True
        x = Dropout(x)              # config.speech_encoder_dropout (no-op at eval)
        return x

    The HF forward additionally casts the input to ``self.layer_norm.weight.dtype``
    before the LayerNorm; for an fp32 reference with fp32 weights this is a
    no-op, so it is omitted here.

    For SeamlessM4T-v2-Large the relevant config values are:
        - feature_projection_input_dim = 160
        - hidden_size = 1024
        - layer_norm_eps = 1e-5
        - speech_encoder_dropout = 0.0

    The LayerNorm step is delegated to :func:`layernorm_forward` for parity
    with the other reference blocks (matches the same op as
    ``F.layer_norm`` over the last dim).

    Args:
        hidden_states: Input tensor of shape
            ``(batch, seq_len, feature_projection_input_dim)``.
        state_dict: Mapping with the following entries:
            - ``"layer_norm"``: dict with ``"weight"`` and ``"bias"`` of shape
              ``(feature_projection_input_dim,)``.
            - ``"projection"``: dict with ``"weight"`` of shape
              ``(hidden_size, feature_projection_input_dim)`` and ``"bias"``
              of shape ``(hidden_size,)`` (HF uses ``bias=True``).
        eps: LayerNorm epsilon (default 1e-5, matches
            ``SeamlessM4Tv2Config.layer_norm_eps``).

    Returns:
        Tensor of shape ``(batch, seq_len, hidden_size)``.
    """
    # 1. LayerNorm over the last (feature) dim, reusing the shared helper.
    norm_hidden_states = layernorm_forward(
        hidden_states,
        weight=state_dict["layer_norm"]["weight"],
        bias=state_dict["layer_norm"]["bias"],
        eps=eps,
    )

    # 2. Up-projection to hidden_size. Dropout is a no-op at eval, so omitted.
    return F.linear(
        norm_hidden_states,
        state_dict["projection"]["weight"],
        state_dict["projection"].get("bias"),
    )


def text_encoder_layer_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, Dict[str, torch.Tensor]],
    num_heads: int,
    head_dim: int,
    attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    activation_function: str = "relu",
) -> torch.Tensor:
    """Standalone ``SeamlessM4Tv2EncoderLayer`` forward pass.

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2
    .SeamlessM4Tv2EncoderLayer`` bit-for-bit at eval time (dropout is a
    no-op). This is one full layer of the NLLB-style text encoder (and is
    structurally also used by the T2U encoder layer). It follows the
    standard pre-norm Transformer block pattern:

        residual = x
        x = self_attn_layer_norm(x)
        x, _ = self_attn(x, attention_mask=attention_mask)
        x = residual + x

        residual = x
        x = ffn_layer_norm(x)
        x = ffn(x)                # Linear -> ReLU -> Linear (no GLU)
        x = residual + x

        return x

    The inner sub-blocks are implemented by the existing reference functions:
        - ``self_attn`` via :func:`seamless_mha_forward` (BART-style 4-proj
          MHA with bias and softmax-in-fp32 cast).
        - ``ffn`` via :func:`seamless_ffn_forward` (Linear -> ReLU -> Linear,
          activation defaults to ReLU per HF ``activation_function="relu"``).
        - LayerNorms via inline ``F.layer_norm`` (cheap, no extra dispatch).

    For SeamlessM4T-v2-Large the relevant config values are:
        - hidden_size = 1024
        - encoder_attention_heads = 16  -> head_dim = 64
        - encoder_ffn_dim = 8192
        - activation_function = "relu"
        - layer_norm_eps = 1e-5
        - dropout = 0.0 (effective at eval)
        - attention_dropout = 0.0
        - activation_dropout = 0.0

    Args:
        hidden_states: Input tensor of shape ``[batch, seq_len, hidden]``.
        state_dict: Nested mapping with the following top-level keys (each a
            sub-dict pulled from the matching HF submodule):
                - ``"self_attn_layer_norm"``: ``{"weight", "bias"}`` shape
                  ``(hidden,)``.
                - ``"self_attn"``:
                  ``{"q_proj", "k_proj", "v_proj", "out_proj"}`` each
                  ``{"weight", "bias"}`` — see :func:`seamless_mha_forward`.
                - ``"ffn_layer_norm"``: ``{"weight", "bias"}`` shape
                  ``(hidden,)``.
                - ``"ffn"``: ``{"fc1", "fc2"}`` each ``{"weight", "bias"}``
                  — see :func:`seamless_ffn_forward`.
        num_heads: Number of self-attention heads (16 for v2-Large).
        head_dim: Per-head dimension (64 for v2-Large). Must satisfy
            ``num_heads * head_dim == hidden``.
        attention_mask: Optional additive log-mask broadcast-compatible with
            ``[batch, 1, seq_len, seq_len]`` for the self-attention.
        eps: LayerNorm epsilon (default 1e-5 for v2-Large).
        activation_function: FFN activation name (default ``"relu"`` per
            ``SeamlessM4Tv2Config.activation_function``).

    Returns:
        Tensor of shape ``[batch, seq_len, hidden]``.
    """
    # 1. Self-attention block (pre-norm).
    residual = hidden_states
    x = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=state_dict["self_attn_layer_norm"]["weight"],
        bias=state_dict["self_attn_layer_norm"]["bias"],
        eps=eps,
    )
    x = seamless_mha_forward(
        x,
        state_dict["self_attn"],
        num_heads=num_heads,
        head_dim=head_dim,
        encoder_hidden_states=None,
        attention_mask=attention_mask,
    )
    x = residual + x

    # 2. Feed-forward block (pre-norm).
    residual = x
    x = F.layer_norm(
        x,
        (x.shape[-1],),
        weight=state_dict["ffn_layer_norm"]["weight"],
        bias=state_dict["ffn_layer_norm"]["bias"],
        eps=eps,
    )
    ffn_sd = state_dict["ffn"]
    if activation_function != "relu":
        raise ValueError(
            f"text_encoder_layer_forward only supports activation_function='relu' "
            f"(SeamlessM4T-v2 default), got {activation_function!r}"
        )
    x = seamless_ffn_forward(
        x,
        fc1_weight=ffn_sd["fc1"]["weight"],
        fc1_bias=ffn_sd["fc1"].get("bias"),
        fc2_weight=ffn_sd["fc2"]["weight"],
        fc2_bias=ffn_sd["fc2"].get("bias"),
    )
    x = residual + x

    return x


def text_decoder_layer_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, Dict[str, torch.Tensor]],
    num_heads: int,
    head_dim: int,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    self_attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    activation_function: str = "relu",
) -> torch.Tensor:
    """Standalone ``SeamlessM4Tv2DecoderLayer`` forward pass (NLLB text decoder layer).

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2DecoderLayer``
    bit-for-bit at eval time (all three dropouts -- ``attn_dropout``,
    ``ffn_dropout``, and the inner attention ``dropout`` -- are no-ops when
    ``training=False``, so we omit them here).

    Op sequence (matches HF exactly):

        # 1. Pre-norm self-attention (causal at full-model level via attention_mask).
        residual = x
        x = self_attn_layer_norm(x)
        x = self_attn(x, attention_mask=self_attention_mask)
        x = residual + x

        # 2. Pre-norm cross-attention (only when encoder_hidden_states is provided).
        if encoder_hidden_states is not None:
            residual = x
            x = cross_attention_layer_norm(x)
            x = cross_attention(x, encoder_hidden_states,
                                attention_mask=encoder_attention_mask)
            x = residual + x

        # 3. Pre-norm FFN (NLLB-style: Linear -> ReLU -> Linear, both with bias).
        residual = x
        x = ffn_layer_norm(x)
        x = ffn(x)
        x = residual + x

        return x

    Note the HF parameter naming for the LayerNorms / submodules:
        - ``self_attn_layer_norm`` / ``self_attn``
        - ``cross_attention_layer_norm`` / ``cross_attention``
          (NOT ``encoder_attn_layer_norm`` / ``encoder_attn`` as in some BART
          forks)
        - ``ffn_layer_norm`` / ``ffn``
    We follow HF naming in the ``state_dict`` keys to make weight-loading
    one-to-one.

    Self-attention causality is supplied through the additive
    ``self_attention_mask`` (the SeamlessM4T-v2 text decoder builds a triangular
    log-mask externally and passes it in -- there is no ``is_causal`` knob on
    the inner attention forward, matching the BART/M2M100 lineage these classes
    are copied from). The HF attention forward does NOT internally apply a
    causal mask, so to get causal behaviour the caller must pass a triangular
    ``self_attention_mask``.

    For SeamlessM4T-v2-Large the relevant config values are:
        - hidden_size = 1024
        - decoder_attention_heads = 16   -> head_dim = 64
        - decoder_ffn_dim = 8192
        - activation_function = "relu"
        - layer_norm_eps = 1e-5
        - dropout = 0.1, attention_dropout = 0.1, activation_dropout = 0.0
          (all dropouts are no-ops at eval; we don't apply them).

    Args:
        hidden_states: Decoder input tensor of shape
            ``[batch, tgt_len, hidden_size]``.
        state_dict: Mapping with the following top-level keys (each a sub-dict
            pulled from the matching HF submodule):
                - ``"self_attn_layer_norm"``: ``{"weight", "bias"}`` shape
                  ``(hidden,)``.
                - ``"self_attn"``: ``{"q_proj", "k_proj", "v_proj", "out_proj"}``
                  each ``{"weight", "bias"}`` -- see
                  :func:`seamless_mha_forward`.
                - ``"cross_attention_layer_norm"``: ``{"weight", "bias"}`` shape
                  ``(hidden,)``. Only required when ``encoder_hidden_states``
                  is provided.
                - ``"cross_attention"``: ``{"q_proj", "k_proj", "v_proj",
                  "out_proj"}`` each ``{"weight", "bias"}`` -- see
                  :func:`seamless_mha_forward`. Only required when
                  ``encoder_hidden_states`` is provided.
                - ``"ffn_layer_norm"``: ``{"weight", "bias"}`` shape
                  ``(hidden,)``.
                - ``"ffn"``: ``{"fc1", "fc2"}`` each ``{"weight", "bias"}``
                  (``fc1`` shape ``[ffn_dim, hidden]``; ``fc2`` shape
                  ``[hidden, ffn_dim]``) -- see :func:`seamless_ffn_forward`.
        num_heads: Number of attention heads (16 for v2-Large decoder).
        head_dim: Per-head dimension (64 for v2-Large decoder). Must satisfy
            ``num_heads * head_dim == hidden_size``.
        encoder_hidden_states: Optional encoder K/V source for the
            cross-attention, shape ``[batch, src_len, hidden_size]``. If
            ``None``, the cross-attention block is skipped (matches HF
            behaviour: ``cross_attn_weights`` is ``None`` and the residual
            stream is untouched between self-attn and FFN).
        self_attention_mask: Optional additive log-mask broadcast-compatible
            with ``[batch, 1, tgt_len, tgt_len]``. For causal self-attention
            the caller supplies a triangular ``-inf`` mask here.
        encoder_attention_mask: Optional additive log-mask broadcast-compatible
            with ``[batch, 1, tgt_len, src_len]`` for the cross-attention.
        eps: LayerNorm epsilon (default 1e-5, matches
            ``SeamlessM4Tv2Config.layer_norm_eps``).
        activation_function: FFN activation name (default ``"relu"`` per
            ``SeamlessM4Tv2Config.activation_function``). Kept as an arg only
            to fail loudly on accidental misconfiguration.

    Returns:
        Tensor of shape ``[batch, tgt_len, hidden_size]``.
    """
    # 1. Self-attention block (pre-norm, residual).
    residual = hidden_states
    x = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=state_dict["self_attn_layer_norm"]["weight"],
        bias=state_dict["self_attn_layer_norm"]["bias"],
        eps=eps,
    )
    x = seamless_mha_forward(
        x,
        state_dict["self_attn"],
        num_heads=num_heads,
        head_dim=head_dim,
        encoder_hidden_states=None,
        attention_mask=self_attention_mask,
    )
    x = residual + x

    # 2. Cross-attention block (pre-norm, residual). Skipped if no encoder states.
    if encoder_hidden_states is not None:
        residual = x
        x = F.layer_norm(
            x,
            (x.shape[-1],),
            weight=state_dict["cross_attention_layer_norm"]["weight"],
            bias=state_dict["cross_attention_layer_norm"]["bias"],
            eps=eps,
        )
        x = seamless_mha_forward(
            x,
            state_dict["cross_attention"],
            num_heads=num_heads,
            head_dim=head_dim,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        x = residual + x

    # 3. Feed-forward block (pre-norm, residual). NLLB-style: Linear -> ReLU -> Linear.
    if activation_function != "relu":
        raise ValueError(
            f"text_decoder_layer_forward only supports activation_function='relu' "
            f"(SeamlessM4T-v2 default), got {activation_function!r}"
        )
    residual = x
    x = F.layer_norm(
        x,
        (x.shape[-1],),
        weight=state_dict["ffn_layer_norm"]["weight"],
        bias=state_dict["ffn_layer_norm"]["bias"],
        eps=eps,
    )
    ffn_sd = state_dict["ffn"]
    x = seamless_ffn_forward(
        x,
        fc1_weight=ffn_sd["fc1"]["weight"],
        fc1_bias=ffn_sd["fc1"].get("bias"),
        fc2_weight=ffn_sd["fc2"]["weight"],
        fc2_bias=ffn_sd["fc2"].get("bias"),
    )
    x = residual + x

    return x


def conformer_adapter_layer_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, Dict[str, torch.Tensor]],
    num_heads: int,
    head_dim: int,
    kernel_size: int = 8,
    stride: int = 8,
    eps: float = 1e-5,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standalone ``SeamlessM4Tv2ConformerAdapterLayer`` forward pass.

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2
    .SeamlessM4Tv2ConformerAdapterLayer`` bit-for-bit at eval time
    (``adaptor_dropout`` is a no-op when ``training=False``).

    This is the adapter layer that sits between the W2v-BERT-2.0 Conformer
    speech-encoder stack and the rest of the SeamlessM4T-v2 pipeline. Its
    purpose is to *down-sample* the temporal axis by ``stride`` (=8) using a
    pair of strided Conv1d + GLU stacks (one for the residual path and one
    for the self-attention path), and then apply a standard MHA + FFN block
    on top of the down-sampled hidden states.

    Op sequence (matches HF exactly):

        # 1. Residual path: LN -> transpose -> Conv1d(stride=s) -> GLU(dim=1) -> transpose
        residual = residual_layer_norm(hidden_states)
        residual = residual.transpose(1, 2)
        residual = residual_conv(residual)
        residual = GLU(dim=1)(residual)
        residual = residual.transpose(1, 2)

        # 2. Main path: LN -> transpose -> Conv1d(stride=s) -> GLU(dim=1) -> transpose
        h = self_attn_layer_norm(hidden_states)
        h = h.transpose(1, 2)
        h = self_attn_conv(h)
        h = GLU(dim=1)(h)
        h = h.transpose(1, 2)

        # 3. Self-attention (no positional embeddings -- use_position_embeddings=False).
        h, _ = self_attn(h, attention_mask=attention_mask)
        h = h + residual

        # 4. Pre-norm FFN (ReLU activation -- adapter uses act_fn="relu", not swish).
        residual = h
        h = ffn_layer_norm(h)
        h = ffn(h) + residual

        return h

    The two Conv1d layers (``residual_conv`` and ``self_attn_conv``) are
    identical in shape: both take ``hidden`` -> ``2 * hidden`` with kernel
    ``kernel_size`` (=8) and stride ``stride`` (=8) and padding
    ``stride // 2`` (=4). Bias is ``True`` (HF default for ``nn.Conv1d``).
    The GLU halves the channel dim so the final hidden dim is unchanged.

    For SeamlessM4T-v2-Large the relevant config values are:
        - hidden_size = 1024
        - speech_encoder_attention_heads = 16  -> head_dim = 64
        - adaptor_kernel_size = 8
        - adaptor_stride = 8
        - adaptor_dropout = 0.1 (no-op at eval)
        - layer_norm_eps = 1e-5
    The inner ``self_attn`` uses ``use_position_embeddings=False``, so NO
    relative-key positional bias is applied (unlike the encoder-layer
    ``self_attn``). The FFN uses ``act_fn="relu"`` (NOT ``swish``).

    Args:
        hidden_states: Input tensor of shape ``[batch, seq_len, hidden]``.
        state_dict: Nested mapping with the following top-level keys (each a
            sub-dict pulled from the matching HF submodule):
                - ``"residual_layer_norm"``: ``{"weight", "bias"}`` shape
                  ``(hidden,)``.
                - ``"residual_conv"``: ``{"weight", "bias"}`` with
                  ``weight`` of shape ``(2 * hidden, hidden, kernel_size)``
                  and ``bias`` of shape ``(2 * hidden,)``.
                - ``"self_attn_layer_norm"``: ``{"weight", "bias"}`` shape
                  ``(hidden,)``.
                - ``"self_attn_conv"``: ``{"weight", "bias"}`` with
                  ``weight`` of shape ``(2 * hidden, hidden, kernel_size)``
                  and ``bias`` of shape ``(2 * hidden,)``.
                - ``"self_attn"``: ``{"linear_q", "linear_k", "linear_v",
                  "linear_out"}`` each ``{"weight", "bias"}`` -- see
                  :func:`conformer_self_attention_forward`. NOTE: no
                  ``distance_embedding`` because the adapter's inner attention
                  has ``use_position_embeddings=False``.
                - ``"ffn_layer_norm"``: ``{"weight", "bias"}`` shape
                  ``(hidden,)``.
                - ``"ffn"``: ``{"intermediate_dense", "output_dense"}`` each
                  ``{"weight", "bias"}`` -- see
                  :func:`conformer_ffn_forward`.
        num_heads: Number of attention heads (16 for v2-Large).
        head_dim: Per-head dimension (64 for v2-Large). Must satisfy
            ``num_heads * head_dim == hidden``.
        kernel_size: Conv1d kernel size (default 8, matches
            ``SeamlessM4Tv2Config.adaptor_kernel_size``).
        stride: Conv1d stride (default 8, matches
            ``SeamlessM4Tv2Config.adaptor_stride``). Output time dim follows
            ``floor((seq_len + 2 * (stride // 2) - kernel_size) / stride) + 1``.
        eps: LayerNorm epsilon (default 1e-5).
        attention_mask: Optional additive log-mask broadcast-compatible with
            ``[batch, 1, sub_seq_len, sub_seq_len]`` (post-downsampling). The
            HF adapter computes a new attention mask from the pre-downsampling
            mask via ``_compute_sub_sample_lengths_from_attention_mask`` and
            ``_compute_new_attention_mask`` -> ``_prepare_4d_attention_mask``;
            here we leave that computation to the caller and just accept the
            already-prepared 4D additive log-mask.

    Returns:
        Tensor of shape ``[batch, sub_seq_len, hidden]`` where
        ``sub_seq_len = floor((seq_len + 2 * (stride // 2) - kernel_size) / stride) + 1``.
        For ``seq_len=128, kernel_size=stride=8, padding=stride//2=4`` the
        output time dim is ``floor((128 + 8 - 8) / 8) + 1 = 17``.
    """
    if num_heads * head_dim != hidden_states.shape[-1]:
        raise ValueError(f"num_heads({num_heads}) * head_dim({head_dim}) != hidden({hidden_states.shape[-1]})")

    conv_padding = stride // 2

    # 1. Residual convolution branch.
    residual = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=state_dict["residual_layer_norm"]["weight"],
        bias=state_dict["residual_layer_norm"]["bias"],
        eps=eps,
    )
    # (B, T, C) -> (B, C, T) for Conv1d.
    residual = residual.transpose(1, 2)
    residual = F.conv1d(
        residual,
        state_dict["residual_conv"]["weight"],
        bias=state_dict["residual_conv"].get("bias"),
        stride=stride,
        padding=conv_padding,
    )
    # GLU along channel dim: (B, 2C, T_sub) -> (B, C, T_sub).
    residual = F.glu(residual, dim=1)
    # (B, C, T_sub) -> (B, T_sub, C).
    residual = residual.transpose(1, 2)

    # 2. Main self-attention convolution branch.
    x = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=state_dict["self_attn_layer_norm"]["weight"],
        bias=state_dict["self_attn_layer_norm"]["bias"],
        eps=eps,
    )
    x = x.transpose(1, 2)
    x = F.conv1d(
        x,
        state_dict["self_attn_conv"]["weight"],
        bias=state_dict["self_attn_conv"].get("bias"),
        stride=stride,
        padding=conv_padding,
    )
    x = F.glu(x, dim=1)
    x = x.transpose(1, 2)

    # 3. Self-attention (no positional embeddings -- use_position_embeddings=False).
    x = conformer_self_attention_forward(
        x,
        state_dict["self_attn"],
        num_heads=num_heads,
        head_dim=head_dim,
        distance_embedding_weight=None,
        position_embeddings_type=None,
        attention_mask=attention_mask,
    )
    x = x + residual

    # 4. Pre-norm FFN (ReLU, matches HF act_fn="relu").
    residual = x
    x = F.layer_norm(
        x,
        (x.shape[-1],),
        weight=state_dict["ffn_layer_norm"]["weight"],
        bias=state_dict["ffn_layer_norm"]["bias"],
        eps=eps,
    )
    x = conformer_ffn_forward(x, state_dict["ffn"], act_fn="relu")
    x = x + residual

    return x


def t2u_decoder_layer_forward(
    hidden_states: torch.Tensor,
    state_dict: Dict[str, Dict[str, torch.Tensor]],
    num_heads: int,
    head_dim: int,
    attention_mask: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    conv_kernel_size: int = 7,
    activation_function: str = "relu",
) -> torch.Tensor:
    """Standalone ``SeamlessM4Tv2TextToUnitDecoderLayer`` forward pass.

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2
    .SeamlessM4Tv2TextToUnitDecoderLayer`` bit-for-bit at eval time (the
    ``conv_dropout`` is a no-op when ``training=False``, so we omit it).

    This is one layer of the **non-autoregressive** Text-to-Unit decoder used
    by SeamlessM4T-v2's speech generation path. Unlike a standard Transformer
    decoder layer, this block:

      1. Uses **post-norm** residual placement (LayerNorm AFTER the
         ``residual + sub_layer`` add), not pre-norm.
      2. Replaces the FFN with a **two-Conv1d + activation** "conv" branch
         (both Conv1d layers have ``kernel_size = 7`` and ``padding = "same"``,
         which is equivalent to symmetric padding of 3 since 7 is odd).
      3. Skips cross-attention entirely (T2U decoder is NAR and conditions on
         encoder hidden states upstream via character expansion, not via a
         per-layer cross-attention).

    Op sequence (matches HF exactly):

        residual = x
        x = self_attn(x, attention_mask=attention_mask)   # BART-style MHA (no causal mask inside)
        x = residual + x
        x = self_attn_layer_norm(x)                       # POST-NORM

        residual = x
        if padding_mask is not None:
            x = x.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        x = conv1(x.transpose(1, 2)).transpose(1, 2)      # Conv1d k=7, padding='same'
        if padding_mask is not None:
            x = x.masked_fill(~padding_mask.unsqueeze(-1), 0.0)
        x = activation_fn(x)                              # ReLU per config
        x = conv2(x.transpose(1, 2)).transpose(1, 2)      # Conv1d k=7, padding='same'
        x = residual + x
        x = conv_layer_norm(x)                            # POST-NORM

        return x

    The self-attention is delegated to :func:`seamless_mha_forward`. Although
    HF instantiates the inner ``SeamlessM4Tv2Attention`` with ``is_decoder=True``,
    NAR decoding means the caller controls causality entirely via the
    ``attention_mask`` argument (the T2U decoder is bidirectional by default
    -- ``SeamlessM4Tv2TextToUnitDecoder.forward`` builds a NON-causal padding
    mask only). Pass a triangular log-mask via ``attention_mask`` if you want
    causal behaviour.

    HF v2 instantiates this layer at the T2U decoder with
    ``decoder_attention_heads = config.decoder_attention_heads`` (== 16 for
    v2-Large). ``decoder_ffn_dim`` is accepted by ``__init__`` for signature
    parity but is NOT used inside this layer (no FFN; the conv branch replaces
    it). For SeamlessM4T-v2-Large the relevant config values are:
        - hidden_size = 1024
        - decoder_attention_heads = 16   -> head_dim = 64
        - activation_function = "relu"
        - layer_norm_eps = 1e-5
        - Conv1d kernel_size = 7 (hardcoded in the HF class)
        - dropout = 0.1 (no-op at eval; ``conv_dropout`` is the only one
          inside the layer that would fire and is a no-op when
          ``training=False``).

    Args:
        hidden_states: Input tensor of shape ``[batch, seq_len, hidden_size]``.
        state_dict: Mapping with the following top-level keys:
                - ``"self_attn"``: ``{"q_proj", "k_proj", "v_proj", "out_proj"}``
                  each ``{"weight", "bias"}`` -- see :func:`seamless_mha_forward`.
                - ``"self_attn_layer_norm"``: ``{"weight", "bias"}`` shape
                  ``(hidden,)``.
                - ``"conv1"``: ``{"weight", "bias"}`` with ``"weight"`` of
                  shape ``(hidden, hidden, conv_kernel_size)`` and ``"bias"``
                  of shape ``(hidden,)`` (HF uses ``bias=True``).
                - ``"conv2"``: ``{"weight", "bias"}`` (same shapes as
                  ``conv1``).
                - ``"conv_layer_norm"``: ``{"weight", "bias"}`` shape
                  ``(hidden,)``.
        num_heads: Number of self-attention heads (16 for v2-Large).
        head_dim: Per-head dimension (64 for v2-Large). Must satisfy
            ``num_heads * head_dim == hidden_size``.
        attention_mask: Optional additive log-mask broadcast-compatible with
            ``[batch, 1, seq_len, seq_len]`` for the self-attention. Pass a
            triangular ``-inf`` mask here for causal behaviour.
        padding_mask: Optional bool/int mask of shape ``(batch, seq_len)``
            where ``False/0`` indicates a padded position to zero out before
            each Conv1d (mirrors the HF ``masked_fill`` step).
        eps: LayerNorm epsilon (default 1e-5, matches
            ``SeamlessM4Tv2Config.layer_norm_eps``).
        conv_kernel_size: Conv1d kernel size (default 7, hardcoded in the HF
            class). Must be odd for the symmetric ``padding='same'`` mapping
            used here to be correct.
        activation_function: Activation between conv1 and conv2 (default
            ``"relu"`` per ``SeamlessM4Tv2Config.activation_function``). Kept
            as an arg only to fail loudly on accidental misconfiguration.

    Returns:
        Tensor of shape ``[batch, seq_len, hidden_size]``.
    """
    if conv_kernel_size % 2 == 0:
        raise ValueError(
            f"conv_kernel_size must be odd for symmetric padding='same' mapping, " f"got {conv_kernel_size}"
        )
    if activation_function != "relu":
        raise ValueError(
            f"t2u_decoder_layer_forward only supports activation_function='relu' "
            f"(SeamlessM4T-v2 default), got {activation_function!r}"
        )
    pad = conv_kernel_size // 2

    # 1. Self-attention block (POST-NORM, residual).
    residual = hidden_states
    x = seamless_mha_forward(
        hidden_states,
        state_dict["self_attn"],
        num_heads=num_heads,
        head_dim=head_dim,
        encoder_hidden_states=None,
        attention_mask=attention_mask,
    )
    x = residual + x
    x = F.layer_norm(
        x,
        (x.shape[-1],),
        weight=state_dict["self_attn_layer_norm"]["weight"],
        bias=state_dict["self_attn_layer_norm"]["bias"],
        eps=eps,
    )

    # 2. Conv branch (POST-NORM, residual). Replaces the standard FFN.
    residual = x

    # Pre-conv1 padding-mask zeroing.
    if padding_mask is not None:
        x = x.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)

    # Conv1: (B, T, C) -> (B, C, T) -> Conv1d k=7 padding='same' -> (B, C, T) -> (B, T, C).
    x = F.conv1d(
        x.transpose(1, 2),
        state_dict["conv1"]["weight"],
        state_dict["conv1"].get("bias"),
        padding=pad,
    ).transpose(1, 2)

    # Pre-activation padding-mask zeroing (matches HF: applied AFTER conv1, BEFORE activation/conv2).
    if padding_mask is not None:
        x = x.masked_fill(~padding_mask.bool().unsqueeze(-1), 0.0)

    # Activation: ReLU (per SeamlessM4Tv2Config.activation_function).
    x = F.relu(x)

    # Conv2: same shape sequence as Conv1.
    x = F.conv1d(
        x.transpose(1, 2),
        state_dict["conv2"]["weight"],
        state_dict["conv2"].get("bias"),
        padding=pad,
    ).transpose(1, 2)

    # conv_dropout is a no-op at eval (training=False) -> skipped.

    x = residual + x
    x = F.layer_norm(
        x,
        (x.shape[-1],),
        weight=state_dict["conv_layer_norm"]["weight"],
        bias=state_dict["conv_layer_norm"]["bias"],
        eps=eps,
    )

    return x


def text_encoder_forward(
    input_ids: torch.Tensor,
    state_dict: Dict[str, object],
    num_heads: int,
    head_dim: int,
    attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    activation_function: str = "relu",
    padding_idx: int = 1,
    embed_scale: Optional[float] = None,
) -> torch.Tensor:
    """Standalone forward pass for ``SeamlessM4Tv2Encoder`` (NLLB text encoder).

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2Encoder``
    bit-for-bit at eval time when ``is_t2u_encoder=False`` (the
    standard text-encoder branch of the SeamlessM4T-v2 model). All
    dropouts (``self.dropout`` and the inner layer's ``attn_dropout`` /
    ``ffn_dropout`` / attention ``dropout``) are no-ops at eval, so we
    omit them here.

    Op sequence (matches HF exactly):

        inputs_embeds = scaled_word_embedding(input_ids)             # *embed_scale
        embed_pos     = sinusoidal_positional_embedding(input_ids)   # padding-aware
        hidden        = inputs_embeds + embed_pos
        for layer in self.layers:                                    # N x encoder_layer
            hidden = text_encoder_layer(hidden, attention_mask)
        hidden = final_layer_norm(hidden)

    Note on the attention mask:
        HF expands a ``[batch, seq_len]`` boolean / 0-1 mask into a 4-D
        additive log-mask via ``_prepare_4d_attention_mask`` before
        passing it into each encoder layer. To keep this reference
        decoupled from HF utilities, the caller supplies the already
        expanded additive 4-D mask directly (broadcast-compatible with
        ``[batch, 1, src_len, src_len]``). For "no mask" the caller
        passes ``None``.

    Reuses the existing, separately-verified reference functions:
        - :func:`scaled_word_embedding_forward`
        - :func:`build_sinusoidal_positional_embedding_weights` +
          :func:`sinusoidal_positional_embedding_forward`
        - :func:`text_encoder_layer_forward` (N times)
        - inline :func:`F.layer_norm` for the final LayerNorm.

    For SeamlessM4T-v2-Large the relevant config values are:
        - hidden_size = 1024
        - encoder_layers = 24
        - encoder_attention_heads = 16  -> head_dim = 64
        - encoder_ffn_dim = 8192
        - activation_function = "relu"
        - layer_norm_eps = 1e-5
        - scale_embedding = True  -> embed_scale = sqrt(1024) = 32.0
        - pad_token_id = 1
        - max_position_embeddings = 4096

    Args:
        input_ids: Integer token ids of shape ``[batch, seq_len]``.
        state_dict: Nested mapping holding all parameters of one
            ``SeamlessM4Tv2Encoder`` (with ``is_t2u_encoder=False``):

            - ``"embed_tokens"``: ``{"weight"}`` shape
              ``(vocab_size, hidden_size)``. The token embedding table
              shared with the rest of the model.
            - ``"embed_positions_weights"``: precomputed sinusoidal table
              of shape ``(num_embeddings, hidden_size)`` with
              ``num_embeddings >= padding_idx + 1 + seq_len``. Typically
              produced by
              :func:`build_sinusoidal_positional_embedding_weights`.
              Allowing the caller to pass it in avoids rebuilding the
              table per call and keeps this function dependency-free.
            - ``"layers"``: ``list`` of ``num_layers`` per-layer
              state_dicts, each in the format consumed by
              :func:`text_encoder_layer_forward` (keys
              ``"self_attn_layer_norm"``, ``"self_attn"``,
              ``"ffn_layer_norm"``, ``"ffn"``).
            - ``"final_layer_norm"``: ``{"weight", "bias"}`` shape
              ``(hidden_size,)`` for the encoder's terminal LayerNorm.

        num_heads: Number of self-attention heads (16 for v2-Large).
        head_dim: Per-head dimension (64 for v2-Large). Must satisfy
            ``num_heads * head_dim == hidden_size``.
        attention_mask: Optional precomputed additive 4-D log-mask
            broadcast-compatible with ``[batch, 1, src_len, src_len]``.
            Use ``None`` for fully-attending sequences (no padding).
        eps: LayerNorm epsilon (default 1e-5, matches
            ``SeamlessM4Tv2Config.layer_norm_eps``; also the default of
            ``nn.LayerNorm`` used for the encoder's final LayerNorm).
        activation_function: FFN activation name (default ``"relu"`` per
            ``SeamlessM4Tv2Config.activation_function``).
        padding_idx: Padding token id (default 1, matches
            SeamlessM4T-v2).
        embed_scale: Scalar multiplier applied to gathered token
            embeddings. If ``None``, defaults to
            ``math.sqrt(hidden_size)`` (matches ``scale_embedding=True``
            for SeamlessM4T-v2).

    Returns:
        Tensor of shape ``[batch, seq_len, hidden_size]`` — the
        encoder's final hidden states (``last_hidden_state``).
    """
    embed_weight = state_dict["embed_tokens"]["weight"]
    hidden_size = embed_weight.shape[-1]
    if embed_scale is None:
        embed_scale = math.sqrt(hidden_size)

    # 1. Token embeddings (scaled).
    inputs_embeds = scaled_word_embedding_forward(
        input_ids,
        embed_weight,
        scale=embed_scale,
        padding_idx=padding_idx,
    )

    # 2. Sinusoidal positional embeddings (padding-aware).
    embed_pos = sinusoidal_positional_embedding_forward(
        state_dict["embed_positions_weights"],
        input_ids=input_ids,
        padding_idx=padding_idx,
        past_key_values_length=0,
    )

    hidden_states = inputs_embeds + embed_pos.to(inputs_embeds.dtype)

    # 3. Stack of encoder layers.
    for layer_sd in state_dict["layers"]:
        hidden_states = text_encoder_layer_forward(
            hidden_states,
            layer_sd,
            num_heads=num_heads,
            head_dim=head_dim,
            attention_mask=attention_mask,
            eps=eps,
            activation_function=activation_function,
        )

    # 4. Final LayerNorm.
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=state_dict["final_layer_norm"]["weight"],
        bias=state_dict["final_layer_norm"]["bias"],
        eps=eps,
    )

    return hidden_states


def text_decoder_forward(
    input_ids: torch.Tensor,
    state_dict: Dict[str, object],
    num_heads: int,
    head_dim: int,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    activation_function: str = "relu",
    padding_idx: int = 0,
    embed_scale: Optional[float] = None,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    """Standalone forward pass for ``SeamlessM4Tv2Decoder`` (NLLB text decoder).

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2Decoder``
    bit-for-bit at eval time (no KV cache path, ``output_attentions=False``,
    ``output_hidden_states=False``, ``use_cache=False``). All dropouts
    (``self.dropout`` and the inner layer dropouts ``attn_dropout`` /
    ``ffn_dropout`` / attention ``dropout``) are no-ops at eval, so we
    omit them here.

    Op sequence (matches HF exactly):

        inputs_embeds = scaled_word_embedding(input_ids)            # *embed_scale
        positions     = sinusoidal_positional_embedding(input_ids,
                                                        past_key_values_length)
        hidden        = inputs_embeds + positions
        # dropout(p=config.dropout, training=False) is a no-op -> skip

        # Build a 4D causal additive log-mask for self-attention from the
        # caller's 2D ``[batch, tgt_len]`` padding mask (or None) using the
        # same HF helper used internally by ``SeamlessM4Tv2Decoder``. This
        # adds the triangular causal part on top of any padding mask.
        self_attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # Expand the encoder 2D padding mask -> 4D additive log-mask, again
        # via the same helper. If the caller already passed a 4D mask we
        # leave it alone (HF does it on the model edge, the layer is dumb).
        if encoder_hidden_states is not None and encoder_attention_mask is not None
            and encoder_attention_mask.dim() == 2:
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

        for layer in self.layers:                          # config.decoder_layers
            hidden = text_decoder_layer(hidden, encoder_hidden_states,
                                        self_attention_mask=self_attention_mask,
                                        encoder_attention_mask=encoder_attention_mask)

        hidden = layer_norm(hidden)                        # final nn.LayerNorm
        return hidden

    The NLLB-style ``SeamlessM4Tv2ScaledWordEmbedding`` multiplies the
    looked-up embeddings by ``sqrt(hidden_size)`` when
    ``config.scale_embedding`` is True (the default for v2-Large, so
    ``embed_scale=32.0``). The sinusoidal positional embedding follows the
    BART/M2M100 convention used by SeamlessM4T-v2: positions are computed
    from the *non-padding* mask of ``input_ids`` and offset by
    ``past_key_values_length`` for incremental decoding.

    Reuses the existing, separately-verified reference functions:
        - :func:`scaled_word_embedding_forward`
        - :func:`sinusoidal_positional_embedding_forward`
        - :func:`text_decoder_layer_forward` (N times)
        - inline :func:`F.layer_norm` for the final LayerNorm.

    For SeamlessM4T-v2-Large the relevant config values are:
        - hidden_size = 1024
        - decoder_layers = 24
        - decoder_attention_heads = 16  -> head_dim = 64
        - decoder_ffn_dim = 8192
        - activation_function = "relu"
        - layer_norm_eps = 1e-5
        - scale_embedding = True  -> embed_scale = sqrt(1024) = 32.0
        - pad_token_id = 0  -> decoder ``padding_idx = 0``
        - max_position_embeddings = 4096

    Note: the final LayerNorm in HF is ``nn.LayerNorm(config.hidden_size)``
    constructed with its *default* ``eps=1e-5``. This matches the
    ``layer_norm_eps`` used everywhere else in the decoder layers, so we
    reuse the same ``eps`` argument for both.

    Args:
        input_ids: Integer decoder token-id tensor of shape
            ``[batch, tgt_len]``.
        state_dict: Nested mapping holding all parameters of one
            ``SeamlessM4Tv2Decoder``:

            - ``"embed_tokens"``: ``{"weight"}`` shape
              ``(vocab_size, hidden_size)``.
            - ``"embed_positions_weights"``: precomputed sinusoidal table
              of shape ``(num_embeddings, hidden_size)`` with
              ``num_embeddings >= padding_idx + 1 + tgt_len +
              past_key_values_length``. Typically produced by
              :func:`build_sinusoidal_positional_embedding_weights`, or
              simply the HF buffer ``decoder.embed_positions.weights``
              passed straight through.
            - ``"layers"``: ``list`` of ``config.decoder_layers`` per-layer
              state_dicts, each in the format consumed by
              :func:`text_decoder_layer_forward` (keys
              ``"self_attn_layer_norm"``, ``"self_attn"``,
              ``"cross_attention_layer_norm"``, ``"cross_attention"``,
              ``"ffn_layer_norm"``, ``"ffn"``).
            - ``"layer_norm"``: ``{"weight", "bias"}`` shape
              ``(hidden_size,)`` for the decoder's terminal LayerNorm
              (HF attribute name ``self.layer_norm``).

        num_heads: Number of self/cross-attention heads (16 for v2-Large).
        head_dim: Per-head dimension (64 for v2-Large). Must satisfy
            ``num_heads * head_dim == hidden_size``.
        encoder_hidden_states: Optional encoder K/V source for the
            cross-attention, shape ``[batch, src_len, hidden_size]``. When
            ``None`` every layer's cross-attention block is skipped.
        attention_mask: Optional 2D ``[batch, tgt_len]`` padding mask
            (1 = keep, 0 = pad), 4D additive ``[batch, 1, tgt_len, tgt_len]``
            mask, or ``None``. Combined with a triangular causal mask via
            ``_prepare_4d_causal_attention_mask`` (matches HF).
        encoder_attention_mask: Optional 2D ``[batch, src_len]`` padding
            mask, 4D additive ``[batch, 1, tgt_len, src_len]`` mask, or
            ``None``. 2D masks are expanded to 4D via
            ``_prepare_4d_attention_mask`` (matches HF). Ignored when
            ``encoder_hidden_states is None``.
        eps: LayerNorm epsilon (default 1e-5; matches both
            ``SeamlessM4Tv2Config.layer_norm_eps`` and the
            ``nn.LayerNorm`` default used for ``self.layer_norm``).
        activation_function: FFN activation name (default ``"relu"`` per
            ``SeamlessM4Tv2Config.activation_function``).
        padding_idx: Padding token id (default 0, matches
            ``SeamlessM4Tv2Config.pad_token_id``).
        embed_scale: Scalar multiplier applied to gathered token
            embeddings. If ``None`` (default) we compute
            ``math.sqrt(hidden_size)`` from the embedding weight shape
            (matches ``scale_embedding=True`` for v2-Large).
        past_key_values_length: Number of cached tokens already consumed
            (default 0). Used to offset sinusoidal positions for
            incremental decoding.

    Returns:
        Tensor of shape ``[batch, tgt_len, hidden_size]`` — the decoder's
        final hidden states (``last_hidden_state``), *before* the LM head
        (which lives outside this block in HF).
    """
    # Lazy-import HF mask helpers so this module can still be imported
    # without forcing the transformers attention-utils path on consumers
    # that only need the non-decoder reference functions.
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask

    embed_weight = state_dict["embed_tokens"]["weight"]
    hidden_size = embed_weight.shape[-1]
    if embed_scale is None:
        embed_scale = math.sqrt(hidden_size)

    # 1. Token embeddings (scaled).
    inputs_embeds = scaled_word_embedding_forward(
        input_ids,
        embed_weight,
        scale=embed_scale,
        padding_idx=padding_idx,
    )

    # 2. Sinusoidal positional embeddings (padding-aware, with KV-cache offset).
    positions = sinusoidal_positional_embedding_forward(
        state_dict["embed_positions_weights"],
        input_ids=input_ids,
        padding_idx=padding_idx,
        past_key_values_length=past_key_values_length,
    )

    hidden_states = inputs_embeds + positions.to(inputs_embeds.dtype)
    # dropout(p=config.dropout, training=False) is a no-op at eval -> skip.

    # 3. Build the 4D causal additive log-mask for self-attention. HF's helper
    # handles every flavor (None, 2D padding, already-4D) and always overlays
    # the triangular causal part -- we delegate to it for parity.
    input_shape = input_ids.shape  # (batch, tgt_len)
    self_attention_mask = _prepare_4d_causal_attention_mask(
        attention_mask, input_shape, inputs_embeds, past_key_values_length
    )

    # 4. Expand the encoder 2D padding mask to a 4D additive log-mask if needed.
    # If the caller already supplied a 4D mask, leave it alone (HF does the
    # 2D->4D expansion only).
    if encoder_hidden_states is not None and encoder_attention_mask is not None:
        if encoder_attention_mask.dim() == 2:
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            )

    # 5. Stack of decoder layers.
    for layer_sd in state_dict["layers"]:
        hidden_states = text_decoder_layer_forward(
            hidden_states,
            layer_sd,
            num_heads=num_heads,
            head_dim=head_dim,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=self_attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            eps=eps,
            activation_function=activation_function,
        )

    # 6. Final LayerNorm (HF attribute name ``self.layer_norm``).
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=state_dict["layer_norm"]["weight"],
        bias=state_dict["layer_norm"]["bias"],
        eps=eps,
    )

    return hidden_states


def t2u_encoder_forward(
    inputs_embeds: torch.Tensor,
    state_dict: Dict[str, object],
    num_heads: int,
    head_dim: int,
    attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    activation_function: str = "relu",
) -> torch.Tensor:
    """Standalone forward pass for the SeamlessM4T-v2 Text-to-Unit encoder.

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2Encoder``
    bit-for-bit at eval time when ``is_t2u_encoder=True`` (the T2U branch of
    the model). When the encoder is constructed with ``is_t2u_encoder=True``
    HF skips token embeddings and sinusoidal positional embeddings entirely:
    the caller supplies ``inputs_embeds`` directly and the encoder just runs
    a stack of pre-norm Transformer layers + a final LayerNorm.

    Op sequence (matches HF exactly):

        hidden = inputs_embeds                              # no embed lookup
                                                            # no positional add
        for layer in self.layers:                           # N x t2u encoder layer
            hidden = encoder_layer(hidden, attention_mask)
        hidden = final_layer_norm(hidden)

    The T2U encoder layer is structurally identical to the text encoder
    layer (same pre-norm self-attn + FFN block, same ReLU activation,
    same hidden_size). The only differences vs the text encoder are:
        - There are 6 layers (vs 24 in the text encoder).
        - No token embedding / no scaled embedding.
        - No sinusoidal positional embedding.
        - Input is ``inputs_embeds`` directly (consumed from upstream).

    For SeamlessM4T-v2-Large the relevant config values are:
        - hidden_size = 1024
        - t2u_encoder_layers = 6
        - t2u_encoder_attention_heads = 16  -> head_dim = 64
        - t2u_encoder_ffn_dim = 8192
        - activation_function = "relu"
        - layer_norm_eps = 1e-5

    Note on the attention mask:
        Mirrors :func:`text_encoder_forward`. HF expands a
        ``[batch, seq_len]`` boolean / 0-1 mask into a 4-D additive log-mask
        via ``_prepare_4d_attention_mask`` before passing it into each
        encoder layer. To keep this reference decoupled from HF utilities,
        the caller supplies the already-expanded additive 4-D mask
        directly (broadcast-compatible with ``[batch, 1, src_len, src_len]``).
        For "no mask" the caller passes ``None``.

    Reuses the existing, separately-verified reference functions:
        - :func:`text_encoder_layer_forward` (N times) — structurally
          identical to the T2U encoder layer.
        - inline :func:`F.layer_norm` for the final LayerNorm.

    Args:
        inputs_embeds: Input embeddings of shape ``[batch, seq_len, hidden]``.
            For SeamlessM4T-v2 these come from upstream of the T2U encoder
            (e.g., from the text decoder hidden states).
        state_dict: Nested mapping holding all parameters of one
            ``SeamlessM4Tv2Encoder`` (with ``is_t2u_encoder=True``):

            - ``"layers"``: ``list`` of ``num_layers`` per-layer
              state_dicts, each in the format consumed by
              :func:`text_encoder_layer_forward` (keys
              ``"self_attn_layer_norm"``, ``"self_attn"``,
              ``"ffn_layer_norm"``, ``"ffn"``).
            - ``"final_layer_norm"``: ``{"weight", "bias"}`` shape
              ``(hidden_size,)`` for the encoder's terminal LayerNorm
              (HF attribute name ``self.layer_norm``).

        num_heads: Number of self-attention heads (16 for v2-Large).
        head_dim: Per-head dimension (64 for v2-Large). Must satisfy
            ``num_heads * head_dim == hidden_size``.
        attention_mask: Optional precomputed additive 4-D log-mask
            broadcast-compatible with ``[batch, 1, src_len, src_len]``.
            Use ``None`` for fully-attending sequences (no padding).
        eps: LayerNorm epsilon (default 1e-5, matches
            ``SeamlessM4Tv2Config.layer_norm_eps``; also the default of
            ``nn.LayerNorm`` used for the encoder's final LayerNorm).
        activation_function: FFN activation name (default ``"relu"`` per
            ``SeamlessM4Tv2Config.activation_function``).

    Returns:
        Tensor of shape ``[batch, seq_len, hidden_size]`` — the T2U
        encoder's final hidden states (``last_hidden_state``).
    """
    hidden_states = inputs_embeds

    # 1. Stack of T2U encoder layers (structurally identical to text encoder layer).
    for layer_sd in state_dict["layers"]:
        hidden_states = text_encoder_layer_forward(
            hidden_states,
            layer_sd,
            num_heads=num_heads,
            head_dim=head_dim,
            attention_mask=attention_mask,
            eps=eps,
            activation_function=activation_function,
        )

    # 2. Final LayerNorm (HF attribute name ``self.layer_norm``).
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=state_dict["final_layer_norm"]["weight"],
        bias=state_dict["final_layer_norm"]["bias"],
        eps=eps,
    )

    return hidden_states


def _build_conformer_chunk_attention_mask(
    seq_len: int,
    chunk_size: int,
    left_chunk_num: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the W2v-BERT-2.0 chunked attention mask used by the Conformer encoder.

    Bit-for-bit port of
    ``SeamlessM4Tv2ConformerEncoder._apply_chunk_attention`` when the input
    ``attention_mask`` is ``None``. Returns a *boolean* mask of shape
    ``[1, 1, seq_len, seq_len]`` whose ``True`` entries indicate positions
    that must be *masked out* (i.e. fall outside the allowed chunk window).

    The HF op then multiplies this mask by ``finfo(dtype).min`` to produce an
    additive log-mask, which the caller can do after we cast to ``dtype``.
    Here we return the raw bool mask plus the additive-log version cast to
    ``dtype`` so the caller can choose what to do.

    Args:
        seq_len: Number of time steps (post feature-projection).
        chunk_size: ``config.speech_encoder_chunk_size`` (default 20000).
        left_chunk_num: ``config.speech_encoder_left_chunk_num`` (default 128).
            Negative values disable the left-chunk overlap (every chunk only
            attends within itself).
        dtype: dtype for the returned additive log-mask.
        device: device for the constructed tensors.

    Returns:
        Additive log-mask of shape ``[1, 1, seq_len, seq_len]`` in ``dtype``,
        ready to be added to attention scores (entries that should be masked
        are ``finfo(dtype).min``; allowed entries are ``0``).
    """
    chunk_indices = torch.arange(seq_len, device=device)
    chunk_indices = torch.div(chunk_indices, chunk_size, rounding_mode="trunc").long()

    if left_chunk_num >= 0:
        start_indices = (chunk_indices - left_chunk_num).clamp_(min=0)
        start_indices = start_indices * chunk_size
    else:
        start_indices = torch.zeros_like(chunk_indices)
    start_indices = start_indices.unsqueeze(1).expand(-1, seq_len)

    end_indices = ((chunk_indices + 1) * chunk_size).clamp_(max=seq_len)
    end_indices = end_indices.unsqueeze(1).expand(-1, seq_len)

    indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(seq_len, -1)

    chunk_mask = (indices < start_indices) | (indices >= end_indices)
    chunk_mask = chunk_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

    # Convert to additive log-mask in ``dtype``.
    additive = chunk_mask.to(dtype=dtype) * torch.finfo(dtype).min
    return additive


def speech_encoder_forward(
    input_features: torch.Tensor,
    state_dict: Dict[str, object],
    num_heads: int,
    head_dim: int,
    attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    speech_encoder_hidden_act: str = "swish",
    left_max_position_embeddings: int = 64,
    right_max_position_embeddings: int = 8,
    position_embeddings_type: Optional[str] = "relative_key",
    conv_depthwise_kernel_size: int = 31,
    adaptor_kernel_size: int = 8,
    adaptor_stride: int = 8,
    speech_encoder_chunk_size: Optional[int] = 20000,
    speech_encoder_left_chunk_num: int = 128,
    add_adapter: bool = True,
) -> torch.Tensor:
    """Standalone forward pass for ``SeamlessM4Tv2SpeechEncoder``.

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2SpeechEncoder``
    bit-for-bit at eval time (no dropouts; ``output_attentions=False``,
    ``output_hidden_states=False``).

    This is the full W2v-BERT-2.0 Conformer speech encoder of SeamlessM4T-v2.
    Op sequence (matches HF exactly):

        hidden = feature_projection(input_features)            # LN(160) + Linear(160->1024)
        hidden = conformer_encoder(hidden, attention_mask)     # N x Conformer + final LN
        expanded = intermediate_ffn(hidden)                    # ReLU FFN (Conformer-style)
        hidden = hidden + 0.5 * expanded
        if add_adapter:
            hidden = conformer_adapter(hidden, attention_mask) # num_adapter_layers x layer
        hidden = inner_layer_norm(hidden)

    The Conformer encoder (``conformer_encoder``) itself handles:
      - masked_fill of padded positions to 0 (when 2-D ``attention_mask`` given),
      - expansion of the 2-D mask -> additive 4-D log-mask,
      - chunked attention masking (``speech_encoder_chunk_size`` /
        ``speech_encoder_left_chunk_num``) — a 4-D bool mask that blocks
        attention outside the current (possibly left-extended) chunk,
      - per-layer Conformer forward (FFN-macaron + MHA + ConvModule + FFN-macaron
        + final LN),
      - a final LayerNorm over the stack output.

    Reuses the existing, separately-verified reference functions:
      - :func:`conformer_feature_projection_forward`
      - :func:`conformer_encoder_layer_forward` (N times)
      - :func:`conformer_ffn_forward` (for ``intermediate_ffn``)
      - :func:`conformer_adapter_layer_forward` (for the adapter layers)
      - inline :func:`F.layer_norm` for the encoder final and ``inner_layer_norm``.

    For SeamlessM4T-v2-Large the relevant config values are:
        - hidden_size = 1024
        - speech_encoder_attention_heads = 16  -> head_dim = 64
        - speech_encoder_layers = 24
        - speech_encoder_intermediate_size = 4096
        - speech_encoder_hidden_act = "swish"
        - left_max_position_embeddings = 64
        - right_max_position_embeddings = 8
        - position_embeddings_type = "relative_key"
        - conv_depthwise_kernel_size = 31
        - feature_projection_input_dim = 160
        - adaptor_kernel_size = 8
        - adaptor_stride = 8
        - num_adapter_layers = 1
        - add_adapter = True
        - speech_encoder_chunk_size = 20000
        - speech_encoder_left_chunk_num = 128
        - layer_norm_eps = 1e-5
        - speech_encoder_dropout / activation_dropout / adaptor_dropout = 0.0 (eval)

    Args:
        input_features: Feature-extractor output of shape
            ``[batch, seq_len, feature_projection_input_dim]`` (160 for v2-Large).
        state_dict: Nested mapping holding all parameters of one
            ``SeamlessM4Tv2SpeechEncoder``:

            - ``"feature_projection"``: ``{"layer_norm", "projection"}``
              matching :func:`conformer_feature_projection_forward`.
            - ``"encoder"``: ``{"layers", "final_layer_norm"}`` where
              ``"layers"`` is a list of per-layer state_dicts in the format
              consumed by :func:`conformer_encoder_layer_forward` (keys
              ``"ffn1_layer_norm"``, ``"ffn1"``, ``"self_attn_layer_norm"``,
              ``"self_attn"``, ``"conv_module"``, ``"ffn2_layer_norm"``,
              ``"ffn2"``, ``"final_layer_norm"``). Each layer's state_dict
              MUST also include a ``"distance_embedding_weight"`` tensor of
              shape ``[L + R + 1, head_dim]`` for the relative-key bias.
            - ``"intermediate_ffn"``: ``{"intermediate_dense", "output_dense"}``
              matching :func:`conformer_ffn_forward` (activation is ``"relu"``
              — HF builds it with ``act_fn="relu"``).
            - ``"adapter"`` (optional, present only when ``add_adapter=True``):
              ``{"layers": [...]}`` where each layer's state_dict matches
              :func:`conformer_adapter_layer_forward`.
            - ``"inner_layer_norm"``: ``{"weight", "bias"}`` shape
              ``(hidden,)`` — the encoder's terminal LayerNorm applied
              AFTER the adapter (or the intermediate-FFN residual when
              ``add_adapter=False``).

        num_heads: Number of self-attention heads (16 for v2-Large).
        head_dim: Per-head dimension (64 for v2-Large). Must satisfy
            ``num_heads * head_dim == hidden_size``.
        attention_mask: Optional 2-D ``[batch, seq_len]`` bool/long mask
            (1=keep, 0=pad) over the post-feature-projection time axis,
            following the HF convention. ``None`` means "no padding".
        eps: LayerNorm epsilon (default 1e-5, matches
            ``SeamlessM4Tv2Config.layer_norm_eps``).
        speech_encoder_hidden_act: Conformer FFN activation (default
            ``"swish"`` per v2-Large).
        left_max_position_embeddings, right_max_position_embeddings,
        position_embeddings_type, conv_depthwise_kernel_size: Forwarded to
            :func:`conformer_encoder_layer_forward`.
        adaptor_kernel_size, adaptor_stride: Forwarded to
            :func:`conformer_adapter_layer_forward`.
        speech_encoder_chunk_size: ``config.speech_encoder_chunk_size``
            (default 20000). Set ``None`` to disable chunk-attention masking.
        speech_encoder_left_chunk_num: ``config.speech_encoder_left_chunk_num``
            (default 128). Number of *previous* chunks each position is
            allowed to attend to; negative disables.
        add_adapter: Whether to apply the Conformer adapter
            (default ``True`` for v2-Large). When ``False``, the
            ``state_dict["adapter"]`` entry is ignored.

    Returns:
        Tensor of shape ``[batch, sub_seq_len, hidden_size]`` (the post-adapter
        last hidden states; ``sub_seq_len == seq_len`` when ``add_adapter=False``).
    """
    # --- 1. Mask preprocessing (mirrors SeamlessM4Tv2ConformerEncoder.forward) ---
    # The Conformer encoder mutates the input + builds its own 4-D mask from
    # the 2-D HF-style ``attention_mask``. We do the same here for bit-equiv.
    hidden_states = conformer_feature_projection_forward(
        input_features,
        state_dict["feature_projection"],
        eps=eps,
    )
    batch_size, seq_len, _ = hidden_states.shape

    conv_attention_mask = attention_mask  # 2-D bool/long mask, used by ConvModule.
    encoder_attention_mask: Optional[torch.Tensor] = None

    if attention_mask is not None:
        # HF: zero padded positions on the residual stream.
        hidden_states = hidden_states.masked_fill(
            ~attention_mask.bool().unsqueeze(-1),
            0.0,
        )
        # HF: expand to additive 4-D log-mask broadcast as [B, 1, T, T].
        expanded = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
        expanded = expanded.expand(batch_size, 1, seq_len, seq_len)
        encoder_attention_mask = expanded

    # Combine with chunked-attention mask (HF: chunk_mask = bool OR existing mask).
    if speech_encoder_chunk_size is not None:
        chunk_additive = _build_conformer_chunk_attention_mask(
            seq_len=seq_len,
            chunk_size=speech_encoder_chunk_size,
            left_chunk_num=speech_encoder_left_chunk_num,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        # HF combines via boolean OR on the un-scaled masks, then multiplies
        # by finfo.min once. To stay bit-equiv we replicate that exactly: any
        # entry that is "masked" in either branch must end up as finfo.min.
        if encoder_attention_mask is None:
            # No 2-D padding mask -> the chunk mask alone is the full mask
            # (already in additive-log form).
            encoder_attention_mask = chunk_additive
        else:
            # Convert ``expanded`` (currently 1.0 where pad, 0.0 where keep)
            # to a bool, OR with chunk_mask bool, then multiply by finfo.min.
            pad_bool = expanded.bool()
            chunk_bool = (chunk_additive != 0).expand(batch_size, 1, seq_len, seq_len)
            combined_bool = pad_bool | chunk_bool
            encoder_attention_mask = combined_bool.to(hidden_states.dtype) * torch.finfo(hidden_states.dtype).min
    elif encoder_attention_mask is not None:
        # No chunk mask -> just apply finfo.min scale (HF does the same on the
        # ``expanded`` tensor when no chunk attention is in play).
        encoder_attention_mask = encoder_attention_mask * torch.finfo(hidden_states.dtype).min

    # --- 2. Conformer encoder stack ---
    for layer_sd in state_dict["encoder"]["layers"]:
        hidden_states = conformer_encoder_layer_forward(
            hidden_states,
            layer_sd,
            num_heads=num_heads,
            head_dim=head_dim,
            distance_embedding_weight=layer_sd.get("distance_embedding_weight"),
            left_max_position_embeddings=left_max_position_embeddings,
            right_max_position_embeddings=right_max_position_embeddings,
            position_embeddings_type=position_embeddings_type,
            conv_kernel_size=conv_depthwise_kernel_size,
            act_fn=speech_encoder_hidden_act,
            eps=eps,
            attention_mask=encoder_attention_mask,
            conv_attention_mask=conv_attention_mask,
        )

    # --- 3. Conformer encoder final LayerNorm ---
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=state_dict["encoder"]["final_layer_norm"]["weight"],
        bias=state_dict["encoder"]["final_layer_norm"]["bias"],
        eps=eps,
    )

    # --- 4. Intermediate FFN with half-step residual ---
    # HF: hidden_states = hidden_states + 0.5 * self.intermediate_ffn(hidden_states)
    expanded = conformer_ffn_forward(
        hidden_states,
        state_dict["intermediate_ffn"],
        act_fn="relu",  # HF: SeamlessM4Tv2ConformerFeedForward(config, act_fn="relu")
    )
    hidden_states = hidden_states + 0.5 * expanded

    # --- 5. Adapter (optional) ---
    if add_adapter:
        # HF: each adapter layer recomputes its own 4-D mask from the
        # input 2-D ``attention_mask`` when it is not None. To match HF we
        # forward the 2-D padding mask through and rely on the adapter
        # layer's own sub-sampling computation. For ``attention_mask=None``
        # the adapter layers skip the mask entirely.
        for layer_sd in state_dict["adapter"]["layers"]:
            if attention_mask is not None:
                # Per-adapter 4-D mask is rebuilt from the (now-current)
                # pre-downsample 2-D mask. The adapter's
                # ``_compute_sub_sample_lengths_from_attention_mask`` operates
                # on the 2-D padding mask, then
                # ``_compute_new_attention_mask`` +
                # ``_prepare_4d_attention_mask`` produce a 4-D additive
                # log-mask of shape [B, 1, T_sub, T_sub]. We import and
                # reuse the HF helpers to stay bit-equiv (rather than
                # re-deriving the formulas here).
                from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
                from transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2 import _compute_new_attention_mask

                # Reproduce HF's per-adapter `_compute_sub_sample_lengths_from_attention_mask`:
                # seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)
                # seq_lens = ((seq_lens + 2 * pad - kernel_size) / stride) + 1
                pad = adaptor_kernel_size // 2
                seq_lens = attention_mask.size(1) - (1 - attention_mask.int()).sum(1)
                sub_sampled_lengths = (((seq_lens + 2 * pad - adaptor_kernel_size) / adaptor_stride) + 1).floor()

                # We need a dummy hidden tensor of the post-adapter time
                # length for _compute_new_attention_mask to produce the
                # right shape; use the actual post-conv hidden_states
                # length computed from the formula above.
                # Note: HF passes ``hidden_states`` AFTER the strided conv,
                # which has the same length as
                # ``floor((T + 2*pad - kernel)/stride) + 1``.
                t_sub = int(((seq_len + 2 * pad - adaptor_kernel_size) // adaptor_stride) + 1)
                dummy_sub_hidden = torch.zeros(
                    batch_size,
                    t_sub,
                    hidden_states.shape[-1],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                sub_2d = _compute_new_attention_mask(hidden_states=dummy_sub_hidden, seq_lens=sub_sampled_lengths)
                sub_4d = _prepare_4d_attention_mask(sub_2d, dummy_sub_hidden.dtype)
                adapter_attention_mask = sub_4d
            else:
                adapter_attention_mask = None

            hidden_states = conformer_adapter_layer_forward(
                hidden_states,
                layer_sd,
                num_heads=num_heads,
                head_dim=head_dim,
                kernel_size=adaptor_kernel_size,
                stride=adaptor_stride,
                eps=eps,
                attention_mask=adapter_attention_mask,
            )
            # After the strided adapter conv, ``seq_len`` shrinks; the next
            # iteration (if any) would need a recomputed 2-D mask. v2-Large
            # uses ``num_adapter_layers=1`` so we don't need to handle that
            # cascade in practice; raise if the caller asked for a stack of
            # >1 adapter layers AND a non-None mask.
            if attention_mask is not None and len(state_dict["adapter"]["layers"]) > 1:
                raise NotImplementedError(
                    "Stacked Conformer adapter (>1 layer) with a non-None "
                    "attention_mask requires recomputing the 2-D mask after "
                    "each strided downsample. v2-Large uses num_adapter_layers=1 "
                    "so this path is unused; add the cascade if you need it."
                )

    # --- 6. Inner LayerNorm (HF attribute name ``self.inner_layer_norm``) ---
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=state_dict["inner_layer_norm"]["weight"],
        bias=state_dict["inner_layer_norm"]["bias"],
        eps=eps,
    )

    return hidden_states


def hifigan_vocoder_forward(
    input_embeds: torch.Tensor,
    state_dict: Dict,
    upsample_rates: Sequence[int] = (5, 4, 4, 2, 2),
    upsample_kernel_sizes: Sequence[int] = (11, 8, 8, 4, 4),
    resblock_kernel_sizes: Sequence[int] = (3, 7, 11),
    resblock_dilation_sizes: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    leaky_relu_slope: float = 0.1,
) -> torch.Tensor:
    """`SeamlessM4Tv2HifiGan` forward pass (HiFi-GAN vocoder, no embeddings).

    Bit-equivalent (in fp32) to
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2HifiGan``
    forward. This is the upsampler + MRF residual stack + conv_post + tanh that
    maps a (batch, model_in_dim, time) spectrogram-like tensor to a 1-D waveform
    (batch, time_out). The unit/speaker/language embeddings sit OUTSIDE this
    block (in ``SeamlessM4Tv2CodeHifiGan``) and are not part of this function.

    Architecture:

        x = conv_pre(input_embeds)                                       # (B, C0, T)
        for i in range(num_upsamples):                                   # 5 stages
            x = leaky_relu(x, slope)
            x = upsampler[i](x)                                          # ConvTranspose1d, stride=upsample_rates[i]
            # MRF (multi-receptive-field): sum of K residual blocks then average
            r = resblocks[i*K + 0](x)
            for j in 1..K-1:
                r += resblocks[i*K + j](x)
            x = r / K
        x = leaky_relu(x)                                                # NOTE: HF uses default slope=0.01 here
        x = conv_post(x)                                                 # (B, 1, T_out)
        x = tanh(x)
        return x.squeeze(1)                                              # (B, T_out)

    NOTE on the final ``leaky_relu``: HF calls ``nn.functional.leaky_relu(x)``
    without passing ``self.leaky_relu_slope``. Per PyTorch's default this uses
    ``negative_slope=0.01`` (NOT the configured ``leaky_relu_slope=0.1``). This
    is a (probably accidental) quirk of the HF implementation that we
    reproduce faithfully here to stay bit-equivalent.

    ConvTranspose1d padding follows HF: ``(kernel_size - upsample_rate) // 2``.

    Args:
        input_embeds: Input tensor of shape ``[batch, model_in_dim, time]``.
            ``model_in_dim = unit_embed_dim + lang_embed_dim + spkr_embed_dim``
            (default ``1280 + 256 + 256 = 1792`` for v2-Large).
        state_dict: Nested dict of weights:
            - ``state_dict["conv_pre"] = {"weight": [C0, in_dim, 7], "bias": [C0]}``
            - ``state_dict["upsampler"] = [ {"weight": [Cin, Cout, k_up], "bias": [Cout]}, ... ]``
              one entry per upsample stage; ConvTranspose1d weight shape is
              ``[in_channels, out_channels, kernel_size]``.
            - ``state_dict["resblocks"] = [ <residual_block_state_dict>, ... ]``
              flat list of length ``num_upsamples * len(resblock_kernel_sizes)``;
              each entry is the state dict expected by
              ``hifigan_residual_block_forward`` (``{"convs1": [...], "convs2": [...]}``).
              Indexed as ``resblocks[i*num_kernels + j]``.
            - ``state_dict["conv_post"] = {"weight": [1, Clast, 7], "bias": [1]}``
        upsample_rates: Stride per upsample stage (default (5,4,4,2,2)).
        upsample_kernel_sizes: Kernel size per upsample stage (default (11,8,8,4,4)).
        resblock_kernel_sizes: Kernel sizes for the K MRF residual blocks at
            each stage (default (3, 7, 11), so K=3 blocks per stage).
        resblock_dilation_sizes: Per-block dilation tuples, length must equal
            ``len(resblock_kernel_sizes)``; each inner tuple is passed to the
            corresponding ``hifigan_residual_block_forward`` call.
        leaky_relu_slope: Negative slope for the per-stage leaky_relu inside
            the upsampler loop (default 0.1). Note: NOT used for the final
            leaky_relu before ``conv_post`` (see NOTE above).

    Returns:
        Waveform tensor of shape ``[batch, time_out]``, in [-1, 1] (after tanh).
    """
    assert len(upsample_rates) == len(upsample_kernel_sizes), (
        f"upsample_rates/upsample_kernel_sizes length mismatch: "
        f"{len(upsample_rates)} vs {len(upsample_kernel_sizes)}"
    )
    assert len(resblock_kernel_sizes) == len(resblock_dilation_sizes), (
        f"resblock_kernel_sizes/resblock_dilation_sizes length mismatch: "
        f"{len(resblock_kernel_sizes)} vs {len(resblock_dilation_sizes)}"
    )

    num_upsamples = len(upsample_rates)
    num_kernels = len(resblock_kernel_sizes)

    upsampler_sd = state_dict["upsampler"]
    resblocks_sd = state_dict["resblocks"]
    assert (
        len(upsampler_sd) == num_upsamples
    ), f"upsampler state_dict length {len(upsampler_sd)} != num_upsamples {num_upsamples}"
    assert len(resblocks_sd) == num_upsamples * num_kernels, (
        f"resblocks state_dict length {len(resblocks_sd)} != "
        f"num_upsamples*num_kernels = {num_upsamples * num_kernels}"
    )

    # --- 1. conv_pre: Conv1d(in_dim, C0, k=7, stride=1, padding=3) ---
    hidden_states = F.conv1d(
        input_embeds,
        state_dict["conv_pre"]["weight"],
        bias=state_dict["conv_pre"].get("bias"),
        stride=1,
        padding=3,
    )

    # --- 2. Upsample stages, each with MRF (multi-receptive-field) ---
    for i in range(num_upsamples):
        hidden_states = F.leaky_relu(hidden_states, negative_slope=leaky_relu_slope)

        # ConvTranspose1d with stride=upsample_rates[i], padding=(k-stride)//2
        up_kernel = upsample_kernel_sizes[i]
        up_stride = upsample_rates[i]
        up_padding = (up_kernel - up_stride) // 2
        hidden_states = F.conv_transpose1d(
            hidden_states,
            upsampler_sd[i]["weight"],
            bias=upsampler_sd[i].get("bias"),
            stride=up_stride,
            padding=up_padding,
        )

        # MRF: sum K residual block outputs then divide by K
        res_state = hifigan_residual_block_forward(
            hidden_states,
            resblocks_sd[i * num_kernels + 0],
            kernel_size=resblock_kernel_sizes[0],
            dilation=tuple(resblock_dilation_sizes[0]),
            leaky_relu_slope=leaky_relu_slope,
        )
        for j in range(1, num_kernels):
            res_state = res_state + hifigan_residual_block_forward(
                hidden_states,
                resblocks_sd[i * num_kernels + j],
                kernel_size=resblock_kernel_sizes[j],
                dilation=tuple(resblock_dilation_sizes[j]),
                leaky_relu_slope=leaky_relu_slope,
            )
        hidden_states = res_state / num_kernels

    # --- 3. Final leaky_relu (HF uses default slope=0.01, NOT leaky_relu_slope) ---
    hidden_states = F.leaky_relu(hidden_states)

    # --- 4. conv_post: Conv1d(Clast, 1, k=7, stride=1, padding=3) ---
    hidden_states = F.conv1d(
        hidden_states,
        state_dict["conv_post"]["weight"],
        bias=state_dict["conv_post"].get("bias"),
        stride=1,
        padding=3,
    )

    # --- 5. tanh + squeeze the singleton channel dim ---
    hidden_states = torch.tanh(hidden_states)
    waveform = hidden_states.squeeze(1)
    return waveform


def _t2u_compute_new_attention_mask(
    hidden_states: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """Helper port of HF ``_compute_new_attention_mask``.

    Produces a ``(batch, seq_len)`` float mask of 1.0 / 0.0, where 0.0 marks
    positions at index >= seq_lens[batch_idx]. Identical to HF's helper used
    by both the speech encoder pre-adapter and the T2U decoder.

    Args:
        hidden_states: Tensor whose first two dims are ``(batch, seq_len, ...)``.
            Only its shape is consulted.
        seq_lens: Long/Int tensor of shape ``(batch,)`` with per-sample valid
            lengths in ``[0, seq_len]``.

    Returns:
        Float mask of shape ``(batch, seq_len)``; valid positions = 1.0,
        padded positions = 0.0. Dtype/device match ``hidden_states``.
    """
    batch_size, mask_seq_len = hidden_states.shape[:2]
    indices = torch.arange(mask_seq_len, device=seq_lens.device).expand(batch_size, -1)
    bool_mask = indices >= seq_lens.unsqueeze(1).expand(-1, mask_seq_len)
    mask = hidden_states.new_ones((batch_size, mask_seq_len))
    mask = mask.masked_fill(bool_mask, 0)
    return mask


def _t2u_hard_upsample(
    hidden_states: torch.Tensor,
    durations: torch.Tensor,
) -> torch.Tensor:
    """Helper port of HF ``SeamlessM4Tv2TextToUnitDecoder._hard_upsample``.

    Repeats each time-step of every sample in the batch according to the
    integer durations. For ``batch == 1`` HF uses a single
    ``torch.repeat_interleave`` (no padding needed). For ``batch > 1`` HF
    interleaves per-sample then zero-right-pads with
    ``nn.utils.rnn.pad_sequence(..., batch_first=True)`` to the longest
    expanded sequence in the batch.

    Args:
        hidden_states: Tensor of shape ``(batch, seq_len, *)`` (we only use
            ``(batch, seq_len, hidden)`` here).
        durations: Long tensor of shape ``(batch, seq_len)`` with integer
            repeat counts (>= 0).

    Returns:
        Upsampled tensor of shape ``(batch, sum_t(durations[b, t])_max, *)``
        (zero-padded along time for ragged batches).
    """
    if hidden_states.size(0) == 1:
        return torch.repeat_interleave(hidden_states, durations.view(-1), dim=1)
    pieces = [
        torch.repeat_interleave(hidden_state, duration, dim=0)
        for (hidden_state, duration) in zip(hidden_states, durations)
    ]
    return torch.nn.utils.rnn.pad_sequence(pieces, batch_first=True)


def t2u_decoder_forward(
    char_input_ids: torch.Tensor,
    char_count_per_id: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    state_dict: Dict[str, object],
    num_heads: int,
    head_dim: int,
    embed_scale: float,
    char_positional_weights: torch.Tensor,
    positional_weights: torch.Tensor,
    padding_idx: int = 1,
    eps: float = 1e-5,
    activation_function: str = "relu",
    variance_predictor_kernel_size: int = 3,
    conv_kernel_size: int = 7,
) -> Dict[str, torch.Tensor]:
    """Standalone forward pass for ``SeamlessM4Tv2TextToUnitDecoder`` (NAR T2U decoder).

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2TextToUnitDecoder``
    bit-for-bit at eval time (all dropouts are no-ops with ``training=False``;
    LayerDrop is skipped because ``self.training=False`` makes the per-layer
    coin-flip never fire).

    This is the *non-autoregressive* Text-to-Unit decoder: it consumes
    character ids + per-text-token character counts + encoder hidden states,
    upsamples encoder hidden states to character resolution, predicts a
    per-character duration via a variance predictor, hard-upsamples by the
    predicted durations to unit resolution, and runs a stack of decoder
    layers (BART-style self-attn + Conv1d conv branch, post-norm).

    Op sequence (matches HF exactly):

        char_padding_mask = _compute_new_attention_mask(char_input_ids, char_count_per_id.sum(1))

        # upsample encoder hidden to char resolution
        char_hidden_states = _hard_upsample(encoder_hidden_states, char_count_per_id)
        char_positions = pos_emb_alpha_char * embed_char_positions(inputs_embeds=char_hidden_states)
        char_hidden_states = embed_char(char_input_ids) * embed_scale + char_positions + char_hidden_states

        # predict per-character duration
        log_dur_pred = duration_predictor(char_hidden_states, padding_mask=char_padding_mask)
        dur_out = clamp(round(expm1(log_dur_pred)).long(), min=1)
        dur_out = dur_out.masked_fill(~char_padding_mask.bool(), 0)

        # upsample to unit resolution
        char_hidden_states = _hard_upsample(char_hidden_states, dur_out)
        positions = pos_emb_alpha * embed_positions(inputs_embeds=char_hidden_states)
        hidden_states = char_hidden_states + positions

        # build per-layer attention mask
        padding_mask = _compute_new_attention_mask(hidden_states, dur_out.sum(1))
        attention_mask = _prepare_4d_attention_mask(padding_mask, hidden_states.dtype)

        for layer in self.layers:
            hidden_states = decoder_layer(hidden_states, attention_mask=attention_mask, padding_mask=padding_mask)
        hidden_states = layer_norm(hidden_states)

    For SeamlessM4T-v2-Large the relevant config values are:
        - hidden_size = 1024
        - t2u_decoder_layers = 6 (after HF's t2u_ -> "" prefix strip)
        - t2u_decoder_attention_heads = 16  -> head_dim = 64
        - t2u_decoder_ffn_dim = 8192  (unused; conv branch replaces FFN)
        - activation_function = "relu"
        - layer_norm_eps = 1e-5
        - char_vocab_size = 10943
        - t2u_variance_predictor_embed_dim = 1024
        - t2u_variance_predictor_hidden_dim = 256
        - variance_predictor_kernel_size = 3
        - scale_embedding = True -> embed_scale = sqrt(1024) = 32.0
        - pad_token_id = 1 (after t2u_ prefix strip; t2u_pad_token_id = 1)
        - dropout = 0.1 (no-op at eval)
        - decoder_layerdrop = 0.05 (no-op at eval)

    Reuses the existing separately-verified reference functions:
        - :func:`scaled_word_embedding_forward` for the character embedding lookup.
        - :func:`sinusoidal_positional_embedding_forward` for both
          ``embed_char_positions`` and ``embed_positions``.
        - :func:`variance_predictor_forward` for the duration predictor.
        - :func:`t2u_decoder_layer_forward` for each of the N stacked layers.
        - inline :func:`F.layer_norm` for the final LayerNorm.

    Args:
        char_input_ids: ``(batch, char_seq_len)`` long tensor of character ids.
        char_count_per_id: ``(batch, encoder_seq_len)`` long tensor with the
            number of characters per text-token, used to upsample
            ``encoder_hidden_states`` to character resolution.
        encoder_hidden_states: ``(batch, encoder_seq_len, hidden)`` float
            tensor (output of the T2U encoder).
        state_dict: Nested mapping holding all parameters of one
            ``SeamlessM4Tv2TextToUnitDecoder``:

            - ``"embed_char"``: ``{"weight"}`` shape ``(char_vocab_size, hidden)``
              (HF uses plain ``nn.Embedding`` — no padding_idx applied at the
              char level; the scale is applied via ``embed_scale`` here).
            - ``"pos_emb_alpha_char"``: scalar tensor (HF ``nn.Parameter``,
              shape ``(1,)``).
            - ``"pos_emb_alpha"``: scalar tensor (HF ``nn.Parameter``, shape
              ``(1,)``).
            - ``"duration_predictor"``: ``state_dict`` for
              :func:`variance_predictor_forward` (keys ``conv1``, ``ln1``,
              ``conv2``, ``ln2``, ``proj``).
            - ``"layers"``: ``list`` of ``num_layers`` per-layer
              :func:`t2u_decoder_layer_forward` state_dicts (keys
              ``self_attn``, ``self_attn_layer_norm``, ``conv1``, ``conv2``,
              ``conv_layer_norm``).
            - ``"layer_norm"``: ``{"weight", "bias"}`` shape ``(hidden,)``
              for the decoder's terminal LayerNorm.

        num_heads: Number of decoder self-attention heads (16 for v2-Large).
        head_dim: Per-head dim (64 for v2-Large). Must satisfy
            ``num_heads * head_dim == hidden_size``.
        embed_scale: Multiplier applied to gathered character embeddings
            (``sqrt(hidden_size)`` for SeamlessM4T-v2 when
            ``scale_embedding=True``).
        char_positional_weights: Sinusoidal weight table for
            ``embed_char_positions`` (precomputed via
            :func:`build_sinusoidal_positional_embedding_weights` with
            ``embedding_dim=hidden_size``,
            ``num_embeddings >= padding_idx + 1 + max_char_seq_len``).
        positional_weights: Sinusoidal weight table for ``embed_positions``
            (same shape as ``char_positional_weights`` but sized for the
            UPSAMPLED unit-length sequence:
            ``num_embeddings >= padding_idx + 1 + max_unit_seq_len``).
        padding_idx: Padding token id used by both positional embeddings
            (default 1 — for the T2U sub-model this is ``t2u_pad_token_id``
            after HF's prefix strip).
        eps: LayerNorm epsilon (default 1e-5, matches
            ``SeamlessM4Tv2Config.layer_norm_eps``).
        activation_function: Decoder-layer conv-branch activation (default
            ``"relu"``).
        variance_predictor_kernel_size: Kernel size for both Conv1d layers
            inside the duration predictor (default 3, matches
            ``SeamlessM4Tv2Config.variance_predictor_kernel_size``).
        conv_kernel_size: Decoder-layer Conv1d kernel size (default 7,
            hardcoded in HF ``SeamlessM4Tv2TextToUnitDecoderLayer``).

    Returns:
        Dict with three tensors (mirrors HF's ``ModelOutput`` minus the
        hidden_states / attentions stacks, which we never request):

        - ``"last_hidden_state"``: ``(batch, unit_seq_len, hidden)`` float
          tensor — the decoder's final hidden states.
        - ``"padding_mask"``: ``(batch, unit_seq_len)`` float mask in
          ``{0.0, 1.0}`` (HF ``padding_mask`` output).
        - ``"dur_out"``: ``(batch, char_seq_len)`` long tensor — the
          per-character integer durations actually used to upsample. Useful
          for downstream tests that want to know the upsampled length.
    """
    # ---------------------------------------------------------------
    # 1. Build the character-level padding mask.
    # ---------------------------------------------------------------
    char_padding_mask = _t2u_compute_new_attention_mask(
        char_input_ids,
        char_count_per_id.sum(1),
    )

    # ---------------------------------------------------------------
    # 2. Upsample encoder hidden states to character resolution.
    # ---------------------------------------------------------------
    char_hidden_states = _t2u_hard_upsample(encoder_hidden_states, char_count_per_id)

    # ---------------------------------------------------------------
    # 3. Add character positional embeddings and the (scaled) character
    #    embedding lookup. Note HF passes inputs_embeds (not input_ids)
    #    into the sinusoidal positional embedding, so positions are simply
    #    padding_idx + 1 .. padding_idx + seq_len.
    # ---------------------------------------------------------------
    char_positions = sinusoidal_positional_embedding_forward(
        char_positional_weights,
        input_ids=None,
        inputs_embeds=char_hidden_states,
        padding_idx=padding_idx,
        past_key_values_length=0,
    )
    char_positions = state_dict["pos_emb_alpha_char"] * char_positions

    # HF: embed_char is a plain nn.Embedding (no padding_idx); embed_scale
    # is applied via scaled_word_embedding semantics.
    char_embeds = scaled_word_embedding_forward(
        char_input_ids,
        state_dict["embed_char"]["weight"],
        scale=embed_scale,
        padding_idx=None,
    )
    char_hidden_states = char_embeds + char_positions + char_hidden_states

    # ---------------------------------------------------------------
    # 4. Predict per-character duration (log-domain), then convert to
    #    integer repeat counts: clamp(round(expm1(log_dur)).long(), min=1)
    #    and zero out padded character positions.
    # ---------------------------------------------------------------
    log_dur_pred = variance_predictor_forward(
        char_hidden_states,
        state_dict["duration_predictor"],
        kernel_size=variance_predictor_kernel_size,
        eps=eps,
        padding_mask=char_padding_mask,
    )
    dur_out = torch.clamp(torch.round(torch.expm1(log_dur_pred)).long(), min=1)
    dur_out = dur_out.masked_fill(~char_padding_mask.bool(), 0)

    # ---------------------------------------------------------------
    # 5. Upsample char hidden states to unit resolution by the predicted
    #    durations, add unit positional embeddings.
    # ---------------------------------------------------------------
    char_hidden_states = _t2u_hard_upsample(char_hidden_states, dur_out)
    positions = sinusoidal_positional_embedding_forward(
        positional_weights,
        input_ids=None,
        inputs_embeds=char_hidden_states,
        padding_idx=padding_idx,
        past_key_values_length=0,
    )
    positions = state_dict["pos_emb_alpha"] * positions
    hidden_states = char_hidden_states + positions

    # ---------------------------------------------------------------
    # 6. Build the per-layer attention mask. HF does:
    #        padding_mask = _compute_new_attention_mask(hidden_states, dur_out.sum(1))
    #        attention_mask = _prepare_4d_attention_mask(padding_mask, hidden_states.dtype)
    #    `_prepare_4d_attention_mask` inverts the boolean mask (1.0 -> 0.0,
    #    0.0 -> ~ very large negative) and expands to [B, 1, 1, T] which
    #    broadcasts against [B, num_heads, T_q, T_k].
    # ---------------------------------------------------------------
    padding_mask = _t2u_compute_new_attention_mask(hidden_states, dur_out.sum(1))

    # Reproduce _prepare_4d_attention_mask (inverted, additive, dtype-aware).
    inverted_mask = 1.0 - padding_mask.to(hidden_states.dtype)
    additive_mask = inverted_mask.masked_fill(
        inverted_mask.bool(),
        torch.finfo(hidden_states.dtype).min,
    )
    # Shape: (B, 1, 1, T_k)
    attention_mask = additive_mask[:, None, None, :]

    # ---------------------------------------------------------------
    # 7. Stack of decoder layers (dropout / layerdrop are no-ops in eval).
    # ---------------------------------------------------------------
    for layer_sd in state_dict["layers"]:
        hidden_states = t2u_decoder_layer_forward(
            hidden_states,
            layer_sd,
            num_heads=num_heads,
            head_dim=head_dim,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            eps=eps,
            conv_kernel_size=conv_kernel_size,
            activation_function=activation_function,
        )

    # ---------------------------------------------------------------
    # 8. Final LayerNorm.
    # ---------------------------------------------------------------
    hidden_states = F.layer_norm(
        hidden_states,
        (hidden_states.shape[-1],),
        weight=state_dict["layer_norm"]["weight"],
        bias=state_dict["layer_norm"]["bias"],
        eps=eps,
    )

    return {
        "last_hidden_state": hidden_states,
        "padding_mask": padding_mask,
        "dur_out": dur_out,
    }


def seamless_m4t_v2_forward(
    input_ids: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    state_dict: Dict[str, object],
    num_heads: int,
    head_dim: int,
    attention_mask: Optional[torch.Tensor] = None,
    decoder_attention_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    activation_function: str = "relu",
    encoder_padding_idx: int = 1,
    decoder_padding_idx: int = 0,
    embed_scale: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Standalone forward pass for the SeamlessM4T-v2 Text-to-Text (T2TT) pipeline.

    Top-level composition reference: verifies that the already-verified
    sub-blocks (``text_encoder_forward`` -> ``text_decoder_forward`` -> ``lm_head``)
    plug together bit-for-bit with HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2.SeamlessM4Tv2ForTextToText``
    at eval time (all dropouts no-op, ``use_cache=False``,
    ``output_attentions=False``, ``output_hidden_states=False``).

    NOTE on scope: The full SeamlessM4T-v2 model has *five* sub-models
    (speech_encoder, text_encoder, text_decoder, t2u_encoder, t2u_decoder,
    code_hifigan_vocoder). The full Speech-to-Speech (S2ST) pipeline involves
    generation (auto-regressive sampling), NAR character upsampling, and
    speaker / language embedding lookups in the code_hifigan_vocoder -- all of
    which sit outside the deterministic forward-pass composition this
    reference is intended to verify. We deliberately scope this top-level
    reference to the *deterministic, single-step* T2TT forward
    (``text_encoder -> text_decoder -> lm_head``) because it (a) is the only
    end-to-end path that does not require a generation loop, (b) exercises
    the embedding + position + mask plumbing of both encoder + decoder, and
    (c) shares its lm_head weights with the encoder/decoder token-embedding
    table (HF ``self.shared``). The remaining sub-blocks are independently
    verified against HF.

    Op sequence (matches HF ``SeamlessM4Tv2ForTextToText.forward`` exactly):

        encoder_hidden = text_encoder(input_ids, attention_mask)
        decoder_hidden = text_decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=attention_mask,   # encoder padding mask
        )
        lm_logits = decoder_hidden @ lm_head_weight.T  # bias=False in HF

    Reuses the existing, separately-verified reference functions:
        - :func:`text_encoder_forward`
        - :func:`text_decoder_forward`
        - inline :func:`F.linear` for the LM head (HF: ``bias=False``).

    For SeamlessM4T-v2-Large the relevant config values are:
        - hidden_size = 1024
        - vocab_size = 256102
        - encoder_layers = decoder_layers = 24
        - encoder_attention_heads = decoder_attention_heads = 16 -> head_dim = 64
        - encoder_ffn_dim = decoder_ffn_dim = 8192
        - activation_function = "relu"
        - layer_norm_eps = 1e-5
        - scale_embedding = True -> embed_scale = sqrt(1024) = 32.0
        - encoder pad_token_id = 1, decoder pad_token_id = 0

    Note on the embedding sharing (``self.shared``):
        HF ``SeamlessM4Tv2ForTextToText`` constructs a single
        ``nn.Embedding(vocab_size, hidden_size, pad_token_id)`` and passes it
        as the shared token-embedding table for both ``text_encoder`` and
        ``text_decoder``. The ``lm_head`` is a SEPARATE ``nn.Linear`` (not
        tied), so callers should pass the lm_head weight explicitly in
        ``state_dict["lm_head"]["weight"]``.

    Args:
        input_ids: Encoder input token ids of shape ``[batch, src_len]``.
        decoder_input_ids: Decoder input token ids of shape ``[batch, tgt_len]``.
        state_dict: Nested mapping with:

            - ``"text_encoder"``: state_dict consumed by
              :func:`text_encoder_forward`.
            - ``"text_decoder"``: state_dict consumed by
              :func:`text_decoder_forward`.
            - ``"lm_head"``: ``{"weight": (vocab_size, hidden_size)}`` -- HF
              uses ``bias=False`` so no bias key is required.

        num_heads: Number of self/cross-attention heads (16 for v2-Large).
            Shared between encoder and decoder (same in HF config).
        head_dim: Per-head dim (64 for v2-Large). Must satisfy
            ``num_heads * head_dim == hidden_size``.
        attention_mask: Optional ``[batch, src_len]`` 2D padding mask (1=keep,
            0=pad) or 4D additive log-mask, or ``None``. Used both as the
            encoder self-attention padding mask AND as the decoder
            cross-attention encoder-padding mask (HF does exactly this).
        decoder_attention_mask: Optional ``[batch, tgt_len]`` 2D padding mask
            or 4D additive log-mask, or ``None``. The decoder always adds the
            triangular causal mask internally on top.
        eps: LayerNorm epsilon (default 1e-5).
        activation_function: FFN activation name (default ``"relu"``).
        encoder_padding_idx: Encoder padding token id (default 1).
        decoder_padding_idx: Decoder padding token id (default 0).
        embed_scale: Optional explicit embedding scale; if ``None``, defaults
            to ``sqrt(hidden_size)`` (matches ``scale_embedding=True``).

    Returns:
        Dict with two tensors:

        - ``"logits"``: ``[batch, tgt_len, vocab_size]`` float tensor -- the
          unnormalized token logits.
        - ``"encoder_last_hidden_state"``: ``[batch, src_len, hidden]`` -- the
          encoder output (useful for downstream verification).
    """
    # --- 1. Encoder ---
    # HF expands the 2D mask -> 4D internally. text_encoder_forward expects
    # an already-4D additive mask, so we expand here if needed (matches HF
    # ``_prepare_4d_attention_mask``).
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

    if attention_mask is not None and attention_mask.dim() == 2:
        # We need the encoder embed dtype to build the additive mask. Cheapest
        # way is to peek at the embed_tokens weight dtype.
        enc_embed_dtype = state_dict["text_encoder"]["embed_tokens"]["weight"].dtype
        encoder_self_mask = _prepare_4d_attention_mask(attention_mask, enc_embed_dtype, tgt_len=input_ids.shape[-1])
    else:
        encoder_self_mask = attention_mask  # None or already 4D

    encoder_hidden_states = text_encoder_forward(
        input_ids,
        state_dict["text_encoder"],
        num_heads=num_heads,
        head_dim=head_dim,
        attention_mask=encoder_self_mask,
        eps=eps,
        activation_function=activation_function,
        padding_idx=encoder_padding_idx,
        embed_scale=embed_scale,
    )

    # --- 2. Decoder ---
    # HF uses the *raw* ``attention_mask`` (the encoder padding mask) as the
    # encoder_attention_mask for cross-attention. Pass it through verbatim;
    # text_decoder_forward handles the 2D->4D expansion internally for both
    # the self-attn causal mask and the cross-attn encoder mask.
    decoder_hidden_states = text_decoder_forward(
        decoder_input_ids,
        state_dict["text_decoder"],
        num_heads=num_heads,
        head_dim=head_dim,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=decoder_attention_mask,
        encoder_attention_mask=attention_mask,
        eps=eps,
        activation_function=activation_function,
        padding_idx=decoder_padding_idx,
        embed_scale=embed_scale,
    )

    # --- 3. LM head (HF: nn.Linear(hidden, vocab, bias=False)) ---
    lm_logits = F.linear(
        decoder_hidden_states,
        state_dict["lm_head"]["weight"],
        bias=None,
    )

    return {
        "logits": lm_logits,
        "encoder_last_hidden_state": encoder_hidden_states,
    }


def code_hifigan_vocoder_forward(
    input_ids: torch.Tensor,
    speaker_id: torch.Tensor,
    lang_id: torch.Tensor,
    state_dict: Dict,
    pad_token_id: int = 1,
    variance_predictor_kernel_size: int = 3,
    variance_predictor_eps: float = 1e-5,
    upsample_rates: Sequence[int] = (5, 4, 4, 2, 2),
    upsample_kernel_sizes: Sequence[int] = (11, 8, 8, 4, 4),
    resblock_kernel_sizes: Sequence[int] = (3, 7, 11),
    resblock_dilation_sizes: Sequence[Sequence[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
    leaky_relu_slope: float = 0.1,
):
    """Standalone ``SeamlessM4Tv2CodeHifiGan`` forward pass.

    Reproduces the forward of HuggingFace
    ``transformers.models.seamless_m4t_v2.modeling_seamless_m4t_v2
    .SeamlessM4Tv2CodeHifiGan`` bit-for-bit (in fp32). This is the vocoder
    wrapper that adds unit/speaker/language embeddings + duration-based
    hard-upsampling around the raw ``hifigan_vocoder`` block.

    Op sequence (matches HF):

        # 1. Embed RVQ unit ids (channels-first).
        hidden = unit_embedding(input_ids).transpose(1, 2)        # (B, C_u, T)
        spkr   = speaker_embedding(speaker_id).transpose(1, 2)    # (B, C_s, 1)
        lang   = language_embedding(lang_id).transpose(1, 2)      # (B, C_l, 1)

        # 2. Duration predictor (variance_predictor with embed=hidden=unit_embed_dim,
        # kernel_size=variance_predictor_kernel_size). Input is (B, T, C_u).
        log_dur = dur_predictor(hidden.transpose(1, 2))           # (B, T)
        dur_out = clamp(round(expm1(log_dur)).long(), min=1)      # (B, T)

        # 3. Hard-upsample by predicted durations (batch_size=1 fast path).
        hidden = repeat_interleave(hidden, dur_out.view(-1), dim=2)  # (B, C_u, T_up)

        # 4. Tile speaker/language to T_up and concat on channel dim.
        spkr = spkr.repeat(1, 1, T_up)
        lang = lang.repeat(1, 1, T_up)
        x = cat([lang, hidden, spkr], dim=1)                       # (B, C_l+C_u+C_s, T_up)

        # 5. HiFi-GAN vocoder -> waveform (B, T_out).
        waveform = hifigan_vocoder(x)

        # 6. Compute output lengths (per-batch valid sample count after dur + conv).
        unit_lengths = _get_dur_output_lengths(input_ids, dur_out)
        lengths      = _get_output_hifigan_lengths(unit_lengths)

    Embedding shapes (SeamlessM4T-v2-Large defaults):
        - ``unit_embedding``     : (10000, 1280)   [unit_hifi_gan_vocab_size, unit_embed_dim]
        - ``speaker_embedding``  : (200,    256)   [vocoder_num_spkrs, spkr_embed_dim]
        - ``language_embedding`` : (36,     256)   [vocoder_num_langs, lang_embed_dim]
        - vocoder ``model_in_dim`` = unit_embed_dim + lang_embed_dim + spkr_embed_dim
          = 1280 + 256 + 256 = 1792

    Duration predictor sizes:
        - ``embed_dim = hidden_dim = unit_embed_dim`` (1280)
        - ``kernel_size = variance_predictor_kernel_size`` (3)

    Batching: HF supports both ``B == 1`` and ``B > 1``. For ``B > 1`` HF builds
    a per-sample interleaved list and uses ``nn.utils.rnn.pad_sequence`` to
    re-batch (loses parallelism). This reference reproduces that path too.

    Args:
        input_ids: ``(B, T)`` long tensor of RVQ unit ids in
            ``[0, unit_hifi_gan_vocab_size)``.
        speaker_id: ``(B,)`` long tensor of speaker indices in
            ``[0, vocoder_num_spkrs)``. (Also accepts shape ``(B, 1)`` per HF.)
        lang_id: ``(B,)`` long tensor of language indices in
            ``[0, vocoder_num_langs)``. (Also accepts shape ``(B, 1)`` per HF.)
        state_dict: Nested dict of weights:
            - ``"unit_embedding"``: ``{"weight": (vocab, unit_embed_dim)}``
            - ``"speaker_embedding"``: ``{"weight": (num_spkrs, spkr_embed_dim)}``
            - ``"language_embedding"``: ``{"weight": (num_langs, lang_embed_dim)}``
            - ``"dur_predictor"``: state dict expected by
              ``variance_predictor_forward``.
            - ``"hifi_gan"``: state dict expected by ``hifigan_vocoder_forward``.
        pad_token_id: ``config.t2u_pad_token_id`` (default 1). Used by
            ``_get_dur_output_lengths`` to compute per-sample valid unit length.
        variance_predictor_kernel_size: Kernel size of the duration predictor
            (default 3, matches ``config.variance_predictor_kernel_size``).
        variance_predictor_eps: LayerNorm eps inside the duration predictor.
        upsample_rates, upsample_kernel_sizes, resblock_kernel_sizes,
        resblock_dilation_sizes, leaky_relu_slope: forwarded to
            ``hifigan_vocoder_forward``. Defaults match SeamlessM4T-v2-Large.

    Returns:
        dict with:
            - ``"waveform"``: ``(B, T_out)`` fp32 in ``[-1, 1]``.
            - ``"lengths"``: ``(B,)`` (or ``()`` if HF squeezed it) long tensor
              of valid output sample counts per batch element.
            - ``"dur_out"``: ``(B, T)`` long tensor of predicted durations
              (clamped >= 1), useful for debugging / sample-rate tests.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2-D (B, T), got shape {tuple(input_ids.shape)}")
    batch = input_ids.shape[0]

    # 1. Embeddings (channels-first like HF).
    unit_emb_w = state_dict["unit_embedding"]["weight"]
    spkr_emb_w = state_dict["speaker_embedding"]["weight"]
    lang_emb_w = state_dict["language_embedding"]["weight"]

    hidden_states = F.embedding(input_ids, unit_emb_w).transpose(1, 2)  # (B, C_u, T)
    spkr = F.embedding(speaker_id, spkr_emb_w).transpose(1, 2)  # (B, C_s, 1)
    lang = F.embedding(lang_id, lang_emb_w).transpose(1, 2)  # (B, C_l, 1)

    # 2. Duration predictor expects (B, T, C). Take the same (B, C_u, T) tensor
    #    and transpose back (matches HF: dur_predictor(hidden_states.transpose(1,2))).
    log_dur_pred = variance_predictor_forward(
        hidden_states.transpose(1, 2),
        state_dict["dur_predictor"],
        kernel_size=variance_predictor_kernel_size,
        eps=variance_predictor_eps,
    )
    dur_out = torch.clamp(torch.round(torch.expm1(log_dur_pred)).long(), min=1)  # (B, T)

    # 3. Hard-upsample via repeat_interleave along time.
    if batch == 1:
        hidden_states = torch.repeat_interleave(hidden_states, dur_out.view(-1), dim=2)
    else:
        # Batched path: per-sample interleave + pad_sequence (matches HF).
        per_sample = [torch.repeat_interleave(h, d, dim=-1).transpose(0, 1) for h, d in zip(hidden_states, dur_out)]
        # pad_sequence: (B, T_up_max, C) -> transpose to (B, C, T_up_max).
        hidden_states = torch.nn.utils.rnn.pad_sequence(per_sample, batch_first=True).transpose(1, 2)

    t_up = hidden_states.shape[-1]

    # 4. Tile speaker/lang to T_up and concat: [lang, hidden, spkr] on channel dim.
    spkr = spkr.repeat(1, 1, t_up)
    lang = lang.repeat(1, 1, t_up)
    cat_features = torch.cat([lang, hidden_states, spkr], dim=1)

    # 5. Vocoder -> waveform.
    waveform = hifigan_vocoder_forward(
        cat_features,
        state_dict["hifi_gan"],
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        leaky_relu_slope=leaky_relu_slope,
    )

    # 6. Per-batch lengths (matches HF _get_dur_output_lengths +
    #    _get_output_hifigan_lengths).
    unit_lengths = _code_hifigan_get_dur_output_lengths(input_ids, dur_out, pad_token_id)
    lengths = _code_hifigan_get_output_hifigan_lengths(
        unit_lengths,
        upsample_rates=upsample_rates,
        upsample_kernel_sizes=upsample_kernel_sizes,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
    )

    return {
        "waveform": waveform,
        "lengths": lengths,
        "dur_out": dur_out,
    }


def _code_hifigan_get_dur_output_lengths(
    input_ids: torch.Tensor,
    dur_out: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Port of HF ``SeamlessM4Tv2CodeHifiGan._get_dur_output_lengths``.

    Computes the per-batch length of the upsampled unit sequence (i.e.
    ``cumsum(dur_out)`` indexed at the last non-pad unit position).
    """
    unit_lengths = (input_ids != pad_token_id).sum(1)
    unit_lengths = torch.clamp(unit_lengths, 0, dur_out.shape[1] - 1)
    cumulative_dur_out = torch.cumsum(dur_out, dim=1)
    unit_lengths = cumulative_dur_out.gather(dim=1, index=unit_lengths.unsqueeze(1)).squeeze()
    return unit_lengths


def _code_hifigan_get_output_hifigan_lengths(
    input_lengths,
    upsample_rates: Sequence[int],
    upsample_kernel_sizes: Sequence[int],
    resblock_kernel_sizes: Sequence[int],
    resblock_dilation_sizes: Sequence[Sequence[int]],
):
    """Port of HF ``SeamlessM4Tv2CodeHifiGan._get_output_hifigan_lengths``.

    Walks the vocoder's conv stack (conv_pre, upsampler ConvTranspose1d's,
    resblock convs1 + convs2 for every MRF block, conv_post) and updates
    the integer length using the standard PyTorch formulas. Accepts both
    Python ``int`` and ``torch.LongTensor`` inputs.
    """

    def _conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
        # 1D convolutional layer output length (matches HF + PyTorch).
        if isinstance(input_length, torch.Tensor):
            return (
                torch.div(
                    input_length + 2 * pad - dilation * (kernel_size - 1) - 1,
                    stride,
                    rounding_mode="floor",
                )
                + 1
            )
        return (input_length + 2 * pad - dilation * (kernel_size - 1) - 1) // stride + 1

    def _transpose_conv_out_length(input_length, kernel_size, stride, pad, dilation=1):
        return (input_length - 1) * stride - 2 * pad + dilation * (kernel_size - 1) + 1

    # conv_pre
    input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

    # upsampler ConvTranspose1d's
    for upsample_rate, kernel_size in zip(upsample_rates, upsample_kernel_sizes):
        input_lengths = _transpose_conv_out_length(
            input_lengths, kernel_size, upsample_rate, (kernel_size - upsample_rate) // 2
        )

    # resblock convs1 (dilated) + convs2 (dilation=1) per upsample stage.
    for _ in range(len(upsample_rates)):
        for kernel_size, dilation in zip(resblock_kernel_sizes, resblock_dilation_sizes):
            for dil in dilation:
                input_lengths = _conv_out_length(
                    input_lengths, kernel_size, 1, (kernel_size - 1) * dil // 2, dilation=dil
                )
            for _dil in dilation:
                input_lengths = _conv_out_length(input_lengths, kernel_size, 1, (kernel_size - 1) // 2, dilation=1)

    # conv_post
    input_lengths = _conv_out_length(input_lengths, 7, 1, 3)

    return input_lengths
