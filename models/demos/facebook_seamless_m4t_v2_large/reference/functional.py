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
