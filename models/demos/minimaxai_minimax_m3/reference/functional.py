# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Pure-PyTorch reference (functional) implementations for the MiniMax-M3 model.

These mirror the HuggingFace `transformers` (>=5.12.0) MiniMaxM3VL modeling
submodules and are used to generate golden tensors for the TTNN bring-up.
No ttnn imports here.
"""

import torch


def rms_norm_forward(hidden: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Gemma-style RMSNorm matching MiniMaxM3VLRMSNorm.

    Normalizes in fp32 and scales by ``(1.0 + weight)`` (the Gemma ``+1`` on the
    gamma), with the multiply by weight performed before casting back to the
    input dtype, i.e. ``(x_norm * (1 + w)).to(input_dtype)``.

    Args:
        hidden: Input tensor of shape [..., hidden_size].
        weight: Gamma weight of shape [hidden_size].
        eps: Variance epsilon.

    Returns:
        Tensor with the same dtype/shape as ``hidden``.
    """
    x = hidden.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x = x * (1.0 + weight.float())
    return x.type_as(hidden)


def embedding_forward(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Text token embedding matching MiniMaxM3VLTextModel.embed_tokens.

    A plain ``nn.Embedding`` row lookup. MiniMax-M3 applies NO scaling
    (no sqrt(hidden_size) multiplier): the HF forward sets
    ``inputs_embeds = self.embed_tokens(input_ids)`` and feeds it directly
    into the decoder stack. ``tie_word_embeddings`` is False.

    The embedding ``padding_idx`` only affects gradient/initialization, not
    the forward output values, so it is irrelevant for inference golden gen.

    Args:
        input_ids: Long tensor of token ids, shape [..., seq_len].
        weight: Embedding matrix of shape [vocab_size, hidden_size].

    Returns:
        Tensor of shape [..., seq_len, hidden_size] with the weight dtype.
    """
    return torch.nn.functional.embedding(input_ids.long(), weight)


def vision_layernorm_forward(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """CLIP-style vision LayerNorm matching the vision tower's ``nn.LayerNorm``.

    True LayerNorm over the last dim with mean subtraction AND a bias term
    (unlike the text RMSNorm): ``(x - mean) / sqrt(var + eps) * weight + bias``.
    Computed in fp32 for numerical stability, then cast back to the input dtype.

    Args:
        x: Input tensor of shape [..., hidden_size].
        weight: Gamma scale of shape [hidden_size].
        bias: Beta shift of shape [hidden_size].
        eps: Variance epsilon (vision_config.layer_norm_eps, default 1e-5).

    Returns:
        Tensor with the same dtype/shape as ``x``.
    """
    orig_dtype = x.dtype
    xf = x.float()
    mean = xf.mean(dim=-1, keepdim=True)
    var = xf.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (xf - mean) / torch.sqrt(var + eps)
    out = x_norm * weight.float() + bias.float()
    return out.type_as(x).to(orig_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Half-split rotation matching HF MiniMaxM3VL ``rotate_half``.

    This is the GPT-NeoX / Llama style ``rotate_half`` (NOT the interleaved
    GPT-J style): the last dim is split into two contiguous halves and rotated
    as ``cat(-x2, x1)``::

        x1 = x[..., : d/2]
        x2 = x[..., d/2 :]
        return cat(-x2, x1)
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def build_rope_cos_sin(
    seq_len: int,
    rotary_dim: int = 64,
    theta: float = 5e6,
    position_ids: torch.Tensor = None,
    dtype: torch.dtype = torch.float32,
    device=None,
):
    """Build the partial-RoPE cos/sin tables matching MiniMaxM3VLRotaryEmbedding.

    Mirrors the HF ``compute_default_rope_parameters`` + ``forward`` for
    ``rope_type == "default"`` (``attention_scaling == 1.0``):

      inv_freq = 1 / theta^(arange(0, rotary_dim, 2) / rotary_dim)   # len rotary_dim/2
      freqs    = position_ids[:, :, None] * inv_freq[None, None, :]  # [B, S, rotary_dim/2]
      emb      = cat(freqs, freqs, dim=-1)                           # [B, S, rotary_dim]
      cos, sin = emb.cos(), emb.sin()

    Note ``rotary_dim`` is the PARTIAL rotary dim (head_dim * partial_rotary_factor
    = 128 * 0.5 = 64), so the produced cos/sin have last dim == 64, and only the
    first 64 dims of each head get rotated downstream (see ``rope_forward``).

    Args:
        seq_len: Sequence length (used to build default position_ids 0..seq_len-1).
        rotary_dim: The partial rotary dimension (64 for MiniMax-M3 text).
        theta: RoPE base / theta (5e6 for MiniMax-M3 text).
        position_ids: Optional explicit positions, shape [B, S]; defaults to
            a single batch arange(seq_len).
        dtype: Output dtype for cos/sin.
        device: Output device.

    Returns:
        ``(cos, sin)`` each of shape [B, seq_len, rotary_dim].
    """
    if position_ids is None:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_ids = position_ids.to(device=device)

    inv_freq = 1.0 / (
        theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / rotary_dim)
    )
    # Force fp32 accumulation as HF does.
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [B, S, rotary_dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [B, S, rotary_dim]
    cos = emb.cos()
    sin = emb.sin()
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rope_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
):
    """Apply PARTIAL rotary position embedding matching HF ``apply_rotary_pos_emb``.

    PARTIAL rope: ``rotary_dim = cos.shape[-1]`` (== 64 for MiniMax-M3 text).
    The first ``rotary_dim`` channels of each head are rotated with the
    half-split ``rotate_half`` convention; the remaining ``head_dim - rotary_dim``
    (== 64) channels PASS THROUGH unchanged, then both parts are concatenated
    back to full ``head_dim``::

        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        q_out   = cat(q_embed, q_pass)

    ``cos``/``sin`` are unsqueezed at ``unsqueeze_dim`` (default 1) so a
    [B, S, rotary_dim] table broadcasts against [B, H, S, head_dim] q/k.

    Args:
        q: Query tensor [B, num_heads, S, head_dim].
        k: Key tensor   [B, num_kv_heads, S, head_dim].
        cos: Cosine table [B, S, rotary_dim].
        sin: Sine table   [B, S, rotary_dim].
        unsqueeze_dim: Dim to unsqueeze cos/sin on (1 for [B,H,S,D] layout).

    Returns:
        ``(q_embed, k_embed)`` rotated, same shapes/dtypes as inputs.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


def build_vision_rope_3d(grid_thw, head_dim: int = 80, theta: float = 1e4, spatial_merge_size: int = 2):
    """Build the CLIP vision 3D-RoPE cos/sin tables matching MiniMaxM3VL3DRotaryEmbedding.

    3D RoPE: every visual token has a ``(t, h, w)`` grid position and the per-head
    rotary dims are split *evenly across the three axes*. With ``rope_dims =
    2 * (head_dim // 2)`` total rotary dims, each axis gets::

        axis_dim = 2 * ((rope_dims // 3) // 2)   # rounded down to a multiple of 2

    so each axis owns ``axis_dim`` channels (``axis_dim // 2`` frequencies). For
    ``head_dim = 80``: ``rope_dims = 80``, ``axis_dim = 26`` (13 freqs/axis),
    ``rot_dim = 3 * axis_dim = 78`` rotated channels, and the final
    ``head_dim - 3 * axis_dim = 2`` channels are NEVER rotated (pass through).

    The single ``inv_freq`` table (length ``axis_dim // 2``) is SHARED across axes;
    each axis multiplies it by *its own* coordinate. The three resulting frequency
    bands are concatenated in ``T | H | W`` order, then duplicated as
    ``cat([freqs, freqs])`` (GPT-NeoX half-rotation pairing). So the cos/sin layout is::

        [ T(13) | H(13) | W(13) | T(13) | H(13) | W(13) ]   -> rot_dim = 78

    Coordinate generation handles ``spatial_merge_size`` (m): within each frame the
    H/W indices are reordered into ``m x m`` spatial-merge blocks via
    ``reshape(h//m, m, w//m, m).permute(0,2,1,3).flatten()`` so that the 2x2 merged
    patches are contiguous; the temporal index ``t`` is ``repeat_interleave(h*w)``.

    Args:
        grid_thw: Long tensor of shape [num_images, 3] with rows ``(t, h, w)``.
        head_dim: Per-head dimension (80 for the M3 vision tower).
        theta: RoPE base (10000.0).
        spatial_merge_size: Spatial merge factor m (2 for M3).

    Returns:
        ``(cos, sin)`` each of shape [total_tokens, rot_dim] where
        ``rot_dim = 3 * axis_dim``, dtype float32.
    """
    import torch as _torch

    if not isinstance(grid_thw, _torch.Tensor):
        grid_thw = _torch.tensor(grid_thw, dtype=_torch.long)

    rope_dims = 2 * (head_dim // 2)
    axis_dim = 2 * ((rope_dims // 3) // 2)
    m = spatial_merge_size

    coords = []
    for t, h, w in grid_thw.tolist():
        hi = _torch.arange(h).unsqueeze(1).expand(-1, w)
        hi = hi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        wi = _torch.arange(w).unsqueeze(0).expand(h, -1)
        wi = wi.reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        ti = _torch.arange(t).repeat_interleave(h * w)
        coords.append(_torch.stack([ti, hi.repeat(t), wi.repeat(t)], dim=-1))
    coords = _torch.cat(coords).to(dtype=_torch.float32)

    inv_freq = 1.0 / (theta ** (_torch.arange(0, axis_dim, 2, dtype=_torch.float32) / axis_dim))
    freqs = _torch.cat([coords[:, i : i + 1] * inv_freq for i in range(3)], dim=-1)
    emb = _torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def vision_rope_3d_forward(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply CLIP vision 3D RoPE matching HF ``apply_rotary_pos_emb_vision``.

    Only the first ``rot_dim = cos.shape[-1]`` head channels carry the 3D rotation
    (using the half-split ``rotate_half`` convention); the tail channels pass
    through untouched, then both parts are concatenated back to full ``head_dim``::

        q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
        q_rot = q_rot * cos + rotate_half(q_rot) * sin
        q_out = cat(q_rot, q_pass)

    cos/sin are [total_tokens, rot_dim]; they are unsqueezed to broadcast against q/k.
    This implementation accepts q/k in ``[B, num_heads, S, head_dim]`` layout (the
    bring-up golden layout) and broadcasts cos/sin as ``[1, 1, S, rot_dim]``. The HF
    reference uses ``[B, S, num_heads, head_dim]`` with cos/sin ``[1, S, 1, rot_dim]``;
    the per-token math is identical.

    Args:
        q: Query tensor [B, num_heads, S, head_dim].
        k: Key tensor   [B, num_heads, S, head_dim].
        cos: Cosine table [S, rot_dim].
        sin: Sine table   [S, rot_dim].

    Returns:
        ``(q_embed, k_embed)`` rotated, same shapes/dtypes as inputs.
    """
    rot_dim = cos.shape[-1]
    # [S, rot_dim] -> [1, 1, S, rot_dim] to broadcast against [B, H, S, D]
    cos_b = cos[None, None, :, :].to(q.dtype)
    sin_b = sin[None, None, :, :].to(q.dtype)

    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]

    q_rot = q_rot * cos_b + _rotate_half(q_rot) * sin_b
    k_rot = k_rot * cos_b + _rotate_half(k_rot) * sin_b

    q_embed = torch.cat([q_rot, q_pass], dim=-1)
    k_embed = torch.cat([k_rot, k_pass], dim=-1)
    return q_embed, k_embed


def qk_norm_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float = 1e-6,
):
    """Per-head QK-norm matching MiniMaxM3VLAttention's q_norm/k_norm.

    M3 attention uses ``use_qk_norm=True`` with ``qk_norm_type="per_head"``: it
    applies a Gemma-style RMSNorm (``MiniMaxM3VLRMSNorm``) over the per-head
    ``head_dim`` (128) of the query and key states. In HF the norm is applied to
    the ``[B, S, num_heads, head_dim]`` view BEFORE ``transpose(1, 2)`` and
    BEFORE RoPE:

        query_states = self.q_norm(self.q_proj(h).view(*shape, -1, head_dim)).transpose(1, 2)
        key_states   = self.k_norm(self.k_proj(h).view(*shape, -1, head_dim)).transpose(1, 2)
        ...
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    Because RMSNorm only reduces over the last dim (``head_dim``), it is
    equivalent whether q/k are laid out as ``[B, S, H, D]`` or
    ``[B, H, S, D]``. This implementation accepts the post-transpose layout
    used elsewhere in the bring-up:

        q: [..., num_attention_heads (64), S, head_dim (128)]
        k: [..., num_key_value_heads (4), S, head_dim (128)]

    The weight is shaped ``[head_dim]`` (==128) and is broadcast identically
    across every head. The Gemma ``+1`` on gamma applies (delegated to
    :func:`rms_norm_forward`). This is APPLIED PRE-ROPE.

    Args:
        q: Query tensor, head_dim as last axis.
        k: Key tensor, head_dim as last axis.
        q_weight: q_norm gamma, shape [head_dim].
        k_weight: k_norm gamma, shape [head_dim].
        eps: RMSNorm variance epsilon (config.rms_norm_eps, default 1e-6).

    Returns:
        ``(q_normed, k_normed)`` with the same shapes/dtypes as the inputs.
    """
    q_normed = rms_norm_forward(q, q_weight, eps=eps)
    k_normed = rms_norm_forward(k, k_weight, eps=eps)
    return q_normed, k_normed


def patch_embedding_forward(
    pixel_values: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    in_channels: int = 3,
    temporal_patch_size: int = 2,
    patch_size: int = 14,
) -> torch.Tensor:
    """CLIP/Qwen2.5-VL-style patch embedding matching MiniMaxM3VLVisionEmbeddings.

    The vision patchify is a **Conv3d** (NOT a Conv2d), identical to
    ``Qwen2_5_VisionPatchEmbed``. The checkpoint stores the weight under
    ``vision_tower.vision_model.embeddings.patch_embedding.weight`` (renamed to
    the inherited ``proj`` at load time), with shape
    ``[embed_dim, in_channels, temporal_patch_size, patch_size, patch_size]``
    (e.g. ``[1280, 3, 2, 14, 14]``). There is **NO bias** (Conv3d ``bias=False``).

    HF forward::

        hidden_states = pixel_values.view(
            -1, in_channels, temporal_patch_size, patch_size, patch_size
        )
        hidden_states = self.proj(hidden_states).view(-1, embed_dim)

    Because the Conv3d kernel size equals its stride and exactly covers each
    flattened patch, this is mathematically a linear projection of each
    ``[C * tps * p * p]``-flattened patch onto ``embed_dim``. Position
    embeddings / 3D-RoPE are applied SEPARATELY downstream and are intentionally
    kept OUT of this block.

    Args:
        pixel_values: Pre-patchified pixel tensor. Either already shaped
            ``[num_patches, C * tps * p * p]`` (Qwen-VL processor output) or any
            tensor whose total numel is a multiple of ``C*tps*p*p``; it is
            ``view``-reshaped to ``[-1, C, tps, p, p]``.
        weight: Conv3d weight, shape
            ``[embed_dim, in_channels, temporal_patch_size, patch_size, patch_size]``.
        bias: Optional bias of shape ``[embed_dim]`` (None for MiniMax-M3).
        in_channels: Number of input channels (3).
        temporal_patch_size: Temporal patch size (2).
        patch_size: Spatial patch size (14).

    Returns:
        Tensor of shape ``[num_patches, embed_dim]`` with the weight dtype.
    """
    target_dtype = weight.dtype
    embed_dim = weight.shape[0]
    hidden_states = pixel_values.view(-1, in_channels, temporal_patch_size, patch_size, patch_size)
    hidden_states = torch.nn.functional.conv3d(
        hidden_states.to(dtype=target_dtype),
        weight,
        bias=bias,
        stride=(temporal_patch_size, patch_size, patch_size),
    )
    return hidden_states.view(-1, embed_dim)


def _repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA key/value head expansion matching HF ``repeat_kv``.

    Expands [B, num_kv_heads, S, head_dim] to [B, num_kv_heads*n_rep, S, head_dim]
    via interleaved expand+reshape (NOT a tile), so kv head ``i`` is repeated into
    output heads ``i*n_rep : (i+1)*n_rep``.
    """
    b, n_kv, s, d = hidden.shape
    if n_rep == 1:
        return hidden
    hidden = hidden[:, :, None, :, :].expand(b, n_kv, n_rep, s, d)
    return hidden.reshape(b, n_kv * n_rep, s, d)


def gqa_attention_forward(
    x: torch.Tensor,
    weights_dict: dict,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_attention_heads: int = 64,
    num_key_value_heads: int = 4,
    head_dim: int = 128,
    eps: float = 1e-6,
    attention_mask: torch.Tensor = None,
):
    """Dense-layer full GQA attention matching MiniMaxM3VLAttention.forward (layers 0-2).

    Exact HF pipeline order (full_attention layers, indexer is None):

      1. q = q_proj(x).view(B, S, num_heads, head_dim)     # [hidden -> 64*128=8192]
         k = k_proj(x).view(B, S, num_kv_heads, head_dim)  # [hidden -> 4*128=512]
         v = v_proj(x).view(B, S, num_kv_heads, head_dim)  # [hidden -> 4*128=512]
      2. PER-HEAD Gemma QK-norm on the LAST dim (head_dim) BEFORE transpose:
            q = q_norm(q); k = k_norm(k)   # rms_norm_forward over head_dim, (1+w) scale
         (v is NOT normed)
      3. transpose(1,2) -> [B, H, S, head_dim]
      4. PARTIAL rope (rotary_dim 64) on q,k via rope_forward (qk_norm BEFORE rope)
      5. repeat_kv(k/v, n_rep=16) : 4 kv heads -> 64
      6. scaled_dot_product, scale = head_dim**-0.5 = 1/sqrt(128), causal
      7. attn.transpose(1,2).reshape(B, S, 8192) -> o_proj -> [hidden]

    Args:
        x: hidden states [B, S, hidden_size].
        weights_dict: dict with keys q_proj,k_proj,v_proj,o_proj,q_norm,k_norm
            (each the ``.weight`` tensor; proj are nn.Linear weights [out,in]).
        cos, sin: partial-rope tables [B, S, rotary_dim] (rotary_dim 64), from
            build_rope_cos_sin(theta=5e6, rotary_dim=64).
        num_attention_heads, num_key_value_heads, head_dim, eps: config.
        attention_mask: optional additive mask [B,1,S,S]; if None a causal mask
            is built internally (matching is_causal=True / sdpa causal).

    Returns:
        Attention block output [B, S, hidden_size] (post o_proj), x's dtype.
    """
    B, S, _ = x.shape
    n_rep = num_attention_heads // num_key_value_heads

    q = torch.nn.functional.linear(x, weights_dict["q_proj"])
    k = torch.nn.functional.linear(x, weights_dict["k_proj"])
    v = torch.nn.functional.linear(x, weights_dict["v_proj"])

    q = q.view(B, S, num_attention_heads, head_dim)
    k = k.view(B, S, num_key_value_heads, head_dim)
    v = v.view(B, S, num_key_value_heads, head_dim)

    # Per-head Gemma RMSNorm over head_dim, applied BEFORE transpose & rope.
    q = rms_norm_forward(q, weights_dict["q_norm"], eps=eps)
    k = rms_norm_forward(k, weights_dict["k_norm"], eps=eps)

    q = q.transpose(1, 2)  # [B, H, S, head_dim]
    k = k.transpose(1, 2)  # [B, n_kv, S, head_dim]
    v = v.transpose(1, 2)  # [B, n_kv, S, head_dim]

    # Partial rope (rotary_dim = cos.shape[-1] = 64), qk_norm done first above.
    q, k = rope_forward(q, k, cos, sin, unsqueeze_dim=1)

    # GQA: expand kv heads 4 -> 64.
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)

    scaling = head_dim**-0.5
    attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling

    if attention_mask is None:
        causal = torch.full((S, S), float("-inf"), dtype=attn_weights.dtype, device=x.device)
        causal = torch.triu(causal, diagonal=1)
        attn_weights = attn_weights + causal[None, None, :, :]
    else:
        attn_weights = attn_weights + attention_mask

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)  # [B, H, S, head_dim]

    attn_output = attn_output.transpose(1, 2).reshape(B, S, -1).contiguous()  # [B, S, 8192]
    out = torch.nn.functional.linear(attn_output, weights_dict["o_proj"])  # [B, S, hidden]
    return out


def vision_attention_forward(
    x: torch.Tensor,
    weights_dict: dict,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_attention_heads: int = 16,
    head_dim: int = 80,
):
    """CLIP vision self-attention matching MiniMaxM3VLVisionAttention.forward.

    BIDIRECTIONAL (non-causal) multi-head self-attention over visual tokens, with
    a 3D RoPE applied to q/k. This is plain MHA -- 16 query heads == 16 key/value
    heads, NO grouped-query attention (``num_key_value_groups = 1``) -- and there
    is NO qk-norm in the vision tower (unlike the language GQA blocks). All four
    projections (q/k/v/out) are ``nn.Linear`` WITH bias.

    Exact HF pipeline order:

      1. q = q_proj(x).view(B, S, num_heads, head_dim)   # 1280 -> 16*80
         k = k_proj(x).view(B, S, num_heads, head_dim)
         v = v_proj(x).view(B, S, num_heads, head_dim).transpose(1, 2)
      2. 3D RoPE on q,k via apply_rotary_pos_emb_vision (cos/sin [S, rot_dim]),
         applied in [B, S, H, D] layout, then transpose q,k -> [B, H, S, D].
         (Here we reuse vision_rope_3d_forward, which expects [B, H, S, D]; the
         per-token math is identical, so we transpose BEFORE the rope call.)
      3. scaled_dot_product_attention, NON-causal, scale = head_dim**-0.5 = 1/sqrt(80)
      4. attn.transpose(1,2).reshape(B, S, hidden) -> out_proj.

    Args:
        x: hidden states [B, S, hidden_size (1280)].
        weights_dict: dict with keys q_proj,k_proj,v_proj,out_proj (the .weight
            tensors, [out,in]) and q_proj_bias,k_proj_bias,v_proj_bias,out_proj_bias
            (the .bias tensors, [out]).
        cos, sin: 3D-rope tables [S, rot_dim] from build_vision_rope_3d(grid_thw,
            head_dim=80, theta=10000.0, spatial_merge_size=2).
        num_attention_heads: 16. head_dim: 80.

    Returns:
        Attention block output [B, S, hidden_size] (post out_proj), x's dtype.
    """
    B, S, _ = x.shape
    scale = head_dim**-0.5

    q = torch.nn.functional.linear(x, weights_dict["q_proj"], weights_dict.get("q_proj_bias"))
    k = torch.nn.functional.linear(x, weights_dict["k_proj"], weights_dict.get("k_proj_bias"))
    v = torch.nn.functional.linear(x, weights_dict["v_proj"], weights_dict.get("v_proj_bias"))

    q = q.view(B, S, num_attention_heads, head_dim).transpose(1, 2)  # [B, H, S, D]
    k = k.view(B, S, num_attention_heads, head_dim).transpose(1, 2)  # [B, H, S, D]
    v = v.view(B, S, num_attention_heads, head_dim).transpose(1, 2)  # [B, H, S, D]

    # 3D RoPE on q,k only (reuse the shared vision rope; tail dims pass through).
    q, k = vision_rope_3d_forward(q, k, cos, sin)

    # Non-causal full self-attention.
    attn = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=scale
    )

    attn = attn.transpose(1, 2).reshape(B, S, -1).contiguous()
    out = torch.nn.functional.linear(attn, weights_dict["out_proj"], weights_dict.get("out_proj_bias"))
    return out


def swigluoai_mlp_forward(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """SwiGLU-OAI gated MLP matching MiniMaxM3VLDenseMLP (GPT-OSS lineage).

    Exact HF forward (transformers 5.12.0, ``MiniMaxM3VLDenseMLP.forward``)::

        gate_up = gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=limit)              # upper clamp ONLY on gate
        up   = up.clamp(min=-limit, max=limit)    # symmetric clamp on up
        glu  = gate * sigmoid(gate * alpha)       # alpha multiplies gate inside sigmoid
        out  = down_proj((up + 1.0) * glu)        # NOTE the (up + 1.0) bias term

    Weight-storage note: the HF *module* uses a single fused
    ``gate_up_proj`` of shape ``[2*inter, hidden]`` and splits it with
    ``chunk(2, dim=-1)`` so the FIRST half is ``gate`` and the second is
    ``up``. The MiniMax-M3 *checkpoint*, however, stores SEPARATE
    ``layers.N.mlp.gate_proj.weight`` and ``...up_proj.weight`` tensors (each
    ``[inter, hidden]``); concatenating them ``cat([gate_w, up_w], dim=0)``
    reproduces the fused weight.

    This function accepts either layout:
      * Separate: pass ``gate_w`` and ``up_w`` (each ``[inter, hidden]``).
      * Fused: pass the fused weight as ``gate_w`` (``[2*inter, hidden]``)
        and ``up_w=None``.

    Args:
        x: Input activations, shape ``[..., hidden_size]``.
        gate_w: Gate projection weight ``[inter, hidden]`` (or fused
            ``[2*inter, hidden]`` if ``up_w is None``).
        up_w: Up projection weight ``[inter, hidden]``, or ``None`` for fused.
        down_w: Down projection weight ``[hidden, inter]``.
        alpha: ``swiglu_alpha`` (sigmoid temperature), default 1.702.
        limit: ``swiglu_limit`` clamp bound, default 7.0.

    Returns:
        Tensor of shape ``[..., hidden_size]``.
    """
    if up_w is None:
        gate_up = torch.nn.functional.linear(x, gate_w)
        gate, up = gate_up.chunk(2, dim=-1)
    else:
        gate = torch.nn.functional.linear(x, gate_w)
        up = torch.nn.functional.linear(x, up_w)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return torch.nn.functional.linear((up + 1.0) * glu, down_w)


def moe_gate_forward(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int = 4,
    routed_scaling_factor: float = 2.0,
):
    """MiniMax-M3 MoE router (DeepSeek-V3 lineage, sigmoid scoring).

    Mirrors HF ``MiniMaxM3VLTopKRouter.forward`` (transformers 5.12.0). Exact
    routing for ``text_config``: num_local_experts=128, num_experts_per_tok=4,
    scoring_func="sigmoid", use_routing_bias=True, routed_scaling_factor=2.0.
    There is NO grouped / node-limited routing (no n_group / topk_group in this
    config) -- plain top-4 of 128.

    HF order (router):
      1. router_logits = x @ gate_weight.T                          (fp32)
      2. routing_weights = sigmoid(router_logits)                   (fp32 scores)
      3. scores_for_choice = routing_weights + e_score_correction_bias
         -> top-k *selection* uses the BIASED scores
      4. top_k_index = topk(scores_for_choice, k)                   (selection only)
      5. top_k_weights = routing_weights.gather(top_k_index)        (ORIGINAL sigmoid
         scores, i.e. the bias is NOT included in the gating weight)
      6. top_k_weights /= top_k_weights.sum(-1, keepdim=True)       (normalize to 1)

    The ``routed_scaling_factor`` (2.0) is applied in HF on the *expert output*
    in ``MiniMaxM3VLSparseMoeBlock`` (``hidden = experts(...) * routed_scaling_factor``),
    NOT inside the router. Here we fold that same factor into the returned gating
    weights so the full effective per-expert weight is captured in one place; the
    downstream MoE op must therefore NOT scale again. Set routed_scaling_factor=1.0
    to obtain the bare HF router weights.

    Args:
        x: hidden states, [..., hidden_size]; flattened to [tokens, hidden].
        gate_weight: router weight [num_experts, hidden_size] (HF Linear layout).
        e_score_correction_bias: [num_experts]; added to scores for SELECTION ONLY.
        top_k: experts per token (4).
        routed_scaling_factor: factor folded into the returned weights (2.0).

    Returns:
        (topk_indices, topk_weights):
          topk_indices: [tokens, top_k] long, selected expert ids (sorted=False).
          topk_weights: [tokens, top_k] fp32, normalized sigmoid scores * factor.
    """
    hidden_dim = gate_weight.shape[-1]
    hidden_states = x.reshape(-1, hidden_dim)
    router_logits = torch.nn.functional.linear(hidden_states.to(gate_weight.dtype), gate_weight)
    routing_weights = torch.sigmoid(router_logits.float())
    scores_for_choice = routing_weights + e_score_correction_bias.float()
    _, topk_indices = torch.topk(scores_for_choice, top_k, dim=-1, sorted=False)
    topk_weights = routing_weights.gather(1, topk_indices)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_indices, topk_weights


def _lightning_indexer_block_indices(
    x: torch.Tensor,
    weights_dict: dict,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    index_head_dim: int = 128,
    index_n_heads: int = 4,
    block_size: int = 128,
    topk_blocks: int = 16,
    local_blocks: int = 1,
    eps: float = 1e-6,
):
    """Lightning indexer selection branch matching MiniMaxM3VLIndexer.forward.

    Pure-PyTorch port of the M3 "lightning indexer" that produces the per-query
    selected key-block indices. This is a SELECTION-ONLY branch: it has no value
    projection and contributes no residual output (the upstream checkpoint
    disables the index-value path on every sparse layer via
    ``sparse_disable_index_value``).

    EXACT indexer math (confirmed against HF ``MiniMaxM3VLIndexer.forward``):

      idx_q = q_norm(index_q_proj(x).view(B, S, H_idx, D_idx)).transpose(1,2)   # [B, H_idx, S, D_idx]
      idx_k = k_norm(index_k_proj(x).view(B, S, 1,    D_idx)).transpose(1,2)    # [B, 1,     S, D_idx]
      # partial RoPE on the index q/k using cos[..., :D_idx], sin[..., :D_idx].
      # Note: cos/sin last dim == rotary_dim == 64 < D_idx(128), so the slice is a
      # no-op clamp and only the first 64 channels are rotated (same partial rope
      # as the main path); channels [64:128] pass through.
      idx_q, idx_k = apply_rotary_pos_emb(idx_q, idx_k, cos[..., :D_idx], sin[..., :D_idx])

    Block-score reduction (score_type == "max"):

      scores = (idx_q.float() @ idx_k.float().transpose(-1,-2))               # [B, H_idx, S_q, S_k]
      token_future = key_pos[None,None,None,:] > position_ids[:,None,:,None]  # strict causal per content pos
      scores = scores.masked_fill(token_future, -inf)
      scores = pad(scores, (0, pad), value=-inf)                             # pad S_k up to a block multiple
      scores = scores.view(B, H_idx, S_q, num_key_blocks, block_size)
      block_scores = scores.amax(dim=-1).amax(dim=1)                          # max over keys-in-block, then over index heads
                                                                              # -> [B, S_q, num_key_blocks]

    Selection rule (init + local + topk, dedup):

      q_block = position_ids // block_size                                    # [B, S_q]
      # local/sliding boost: force the local_blocks blocks ENDING at the query's
      # own block (q_block, q_block-1, ...) to +inf so they always win a slot.
      local_idx = (q_block[...,None] - arange(local_blocks)).clamp(min=0)
      block_scores.scatter_(-1, local_idx, +inf)
      # NOTE: with this config sparse_init_block == 0, so NO init-block boost is
      # applied (block 0 is only kept if it wins on score or is a local block).
      # If sparse_init_block were > 0 the first init blocks would likewise be
      # forced to +inf the same way (HF has no separate init scatter here because
      # config.index has init folded out; we honor init_block via init_blocks arg).
      topk = min(topk_blocks, num_key_blocks)
      topk_scores, topk_indices = block_scores.topk(topk, dim=-1)             # [B, S_q, topk]
      # future/empty blocks keep -inf and sort to the end; tag them -1 (left-packed).
      return topk_indices.masked_fill(topk_scores == -inf, -1)

    The +inf local boost guarantees the local blocks appear exactly once (topk
    over distinct block columns), so the returned indices are deduplicated.

    Args:
        x: hidden states [B, S, hidden_size].
        weights_dict: needs keys ``index_q_proj``, ``index_k_proj``,
            ``index_q_norm``, ``index_k_norm`` (each a ``.weight`` tensor).
        cos, sin: partial-rope tables [B, S, rotary_dim] (rotary_dim 64).
        position_ids: [B, S] content positions (causal anchor).
        index_head_dim, index_n_heads, block_size, topk_blocks, local_blocks: config.
        eps: RMSNorm eps.

    Returns:
        block_indices [B, S_q, topk] long tensor; valid entries left-packed,
        ``-1`` right-pads unused/future/empty slots.
    """
    B, S, _ = x.shape
    D = index_head_dim
    H = index_n_heads

    idx_q = torch.nn.functional.linear(x, weights_dict["index_q_proj"]).view(B, S, H, D)
    idx_q = rms_norm_forward(idx_q, weights_dict["index_q_norm"], eps=eps).transpose(1, 2)  # [B,H,S,D]
    idx_k = torch.nn.functional.linear(x, weights_dict["index_k_proj"]).view(B, S, 1, D)
    idx_k = rms_norm_forward(idx_k, weights_dict["index_k_norm"], eps=eps).transpose(1, 2)  # [B,1,S,D]

    # Partial rope on index q/k. cos/sin last dim (64) < D (128); slicing to D is a
    # clamp to the available rotary dim, so first 64 channels rotate, [64:] pass.
    rot = min(cos.shape[-1], D)
    idx_q, idx_k = rope_forward(idx_q, idx_k, cos[..., :rot], sin[..., :rot], unsqueeze_dim=1)

    k_len = idx_k.shape[2]
    num_key_blocks = -(-k_len // block_size)  # ceil-div
    pad = num_key_blocks * block_size - k_len

    scores = torch.matmul(idx_q.float(), idx_k.float().transpose(-1, -2))  # [B,H,S_q,S_k]
    k_positions = torch.arange(k_len, device=x.device)
    token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]
    scores = scores.masked_fill(token_future, float("-inf"))
    if pad:
        scores = torch.nn.functional.pad(scores, (0, pad), value=float("-inf"))
    scores = scores.view(B, H, S, num_key_blocks, block_size)
    block_scores = scores.amax(dim=-1).amax(dim=1)  # [B, S_q, num_key_blocks]

    q_block = position_ids // block_size  # [B, S_q]

    if local_blocks > 0:
        local = torch.arange(local_blocks, device=x.device)
        local_idx = (q_block[..., None] - local.view(1, 1, -1)).clamp(min=0)  # [B, S_q, local]
        block_scores.scatter_(-1, local_idx, float("inf"))

    topk = min(topk_blocks, num_key_blocks)
    topk_scores, topk_indices = block_scores.topk(topk, dim=-1)
    return topk_indices.masked_fill(topk_scores == float("-inf"), -1)


def _build_block_mask(
    block_indices: torch.Tensor,
    key_length: int,
    position_ids: torch.Tensor,
    block_size: int,
    dtype: torch.dtype,
    device,
):
    """Expand selected block indices into a dense additive attention mask.

    Pure-PyTorch port of ``MiniMaxM3VLIndexer.build_block_mask`` (eager/SDPA path).
    Builds the full 4D additive mask ``[B, 1, S_q, S_k]`` where allowed
    (query, key) pairs are ``0`` and disallowed pairs are ``finfo(dtype).min``.

    Combine rule (block-selection AND causality):

      safe = block_indices.masked_fill(block_indices < 0, num_key_blocks)  # park -1 in throwaway col
      bias = full([B, S_q, num_key_blocks+1], -inf); bias.scatter_(-1, safe, 0.0)
      bias = bias[..., :num_key_blocks]                                    # drop throwaway col
      block_keep = (bias == 0).repeat_interleave(block_size, -1)[..., :key_length].unsqueeze(1)
      # combine with strict causal (token_future) since attention_mask is None here:
      token_future = key_pos[None,None,None,:] > position_ids[:,None,:,None]
      keep = block_keep & ~token_future
      mask = zeros(keep.shape).masked_fill(~keep, finfo(dtype).min)

    So a key is attended iff its BLOCK was selected by the indexer AND it is not
    in the future of the query (block granularity selection, token granularity
    causality). The init/local boosting in the indexer guarantees the local
    blocks (and, when configured, the init blocks) are always present.

    Args:
        block_indices: [B, S_q, topk] from the indexer (-1 padded).
        key_length: S_k.
        position_ids: [B, S_q] content positions.
        block_size: keys per block.
        dtype, device: output mask dtype/device.

    Returns:
        Additive mask [B, 1, S_q, S_k].
    """
    B, S_q, _ = block_indices.shape
    num_key_blocks = -(-key_length // block_size)

    safe = block_indices.masked_fill(block_indices < 0, num_key_blocks)
    bias = block_indices.new_full((B, S_q, num_key_blocks + 1), float("-inf"), dtype=dtype)
    bias.scatter_(-1, safe, 0.0)
    bias = bias[..., :num_key_blocks]

    block_keep = (bias == 0.0).repeat_interleave(block_size, dim=-1)[..., :key_length].unsqueeze(1)

    k_positions = torch.arange(key_length, device=device)
    token_future = k_positions[None, None, None, :] > position_ids[:, None, :, None]
    keep = block_keep & ~token_future

    min_dtype = torch.finfo(dtype).min
    return torch.zeros(keep.shape, dtype=dtype, device=device).masked_fill(~keep, min_dtype)


def sparse_lightning_attention_forward(
    x: torch.Tensor,
    weights_dict: dict,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None,
    num_attention_heads: int = 64,
    num_key_value_heads: int = 4,
    head_dim: int = 128,
    index_head_dim: int = 128,
    index_n_heads: int = 4,
    block_size: int = 128,
    topk_blocks: int = 16,
    local_blocks: int = 1,
    init_blocks: int = 0,
    eps: float = 1e-6,
):
    """Sparse lightning attention matching MiniMaxM3VLAttention.forward (sparse layers 3-59).

    Composes the lightning indexer selection branch + a block-sparse dense GQA
    attention. EXACT HF pipeline (``layer_types[layer_idx] == "minimax_m3_sparse"``):

      1. Main q/k/v projections + PER-HEAD Gemma QK-norm (q_norm/k_norm over
         head_dim, BEFORE transpose; v not normed), transpose to [B,H,S,D].
      2. Partial rope (rotary_dim 64) on main q,k (qk_norm BEFORE rope).
      3. INDEXER (run on raw ``hidden_states`` x, NOT on the main q/k):
           block_indices = lightning_indexer(x, ...)            # [B, S_q, topk], -1 padded
           attention_mask = build_block_mask(block_indices, ...) # [B,1,S_q,S_k] additive
         The indexer projects x to a low-dim index space (index_head_dim 128,
         index_n_heads 4), applies its own qk_norm + partial rope, scores
         q.k per index head, max-pools per key-block (amax over block_size then
         over index heads), boosts local blocks to +inf, and keeps top-``topk``
         blocks per query (left-packed, -1 right-pad). The mask keeps a key iff
         its block was selected AND key is causal w.r.t. the query.
      4. repeat_kv (4 -> 64), SDPA with scale head_dim**-0.5 and the block-sparse
         additive mask (which already encodes causality), softmax fp32.
      5. attn.transpose + reshape -> o_proj.

    The ONLY difference from ``gqa_attention_forward`` is that the additive mask
    is the block-sparse-AND-causal mask from the indexer instead of a plain
    causal mask. When every block is selected (e.g. num_key_blocks <= topk) the
    mask reduces EXACTLY to the dense causal mask and this equals dense GQA.

    Args:
        x: hidden states [B, S, hidden_size].
        weights_dict: q_proj/k_proj/v_proj/o_proj/q_norm/k_norm (main) and
            index_q_proj/index_k_proj/index_q_norm/index_k_norm (indexer), each a
            ``.weight`` tensor.
        cos, sin: partial-rope tables [B, S, rotary_dim] (rotary_dim 64).
        position_ids: [B, S] content positions; defaults to arange(S).
        config dims as named; ``init_blocks`` (sparse_init_block, 0 here) reserved
            for parity -- with 0 it is a no-op (HF folds it out of this config).

    Returns:
        Attention block output [B, S, hidden_size] (post o_proj).
    """
    B, S, _ = x.shape
    n_rep = num_attention_heads // num_key_value_heads
    if position_ids is None:
        position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

    # --- main projections + qk_norm + rope ---
    q = torch.nn.functional.linear(x, weights_dict["q_proj"]).view(B, S, num_attention_heads, head_dim)
    k = torch.nn.functional.linear(x, weights_dict["k_proj"]).view(B, S, num_key_value_heads, head_dim)
    v = torch.nn.functional.linear(x, weights_dict["v_proj"]).view(B, S, num_key_value_heads, head_dim)

    q = rms_norm_forward(q, weights_dict["q_norm"], eps=eps).transpose(1, 2)
    k = rms_norm_forward(k, weights_dict["k_norm"], eps=eps).transpose(1, 2)
    v = v.transpose(1, 2)

    q, k = rope_forward(q, k, cos, sin, unsqueeze_dim=1)

    # --- indexer selection branch (on raw hidden states x) ---
    block_indices = _lightning_indexer_block_indices(
        x,
        weights_dict,
        cos,
        sin,
        position_ids,
        index_head_dim=index_head_dim,
        index_n_heads=index_n_heads,
        block_size=block_size,
        topk_blocks=topk_blocks,
        local_blocks=local_blocks,
        eps=eps,
    )
    attention_mask = _build_block_mask(block_indices, k.shape[2], position_ids, block_size, q.dtype, x.device)

    # --- block-sparse GQA SDPA ---
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)

    scaling = head_dim**-0.5
    attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling
    attn_weights = attn_weights + attention_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_output = torch.matmul(attn_weights, v)

    attn_output = attn_output.transpose(1, 2).reshape(B, S, -1).contiguous()
    out = torch.nn.functional.linear(attn_output, weights_dict["o_proj"])
    return out, block_indices


def vision_mlp_forward(
    x: torch.Tensor,
    fc1_w: torch.Tensor,
    fc1_b: torch.Tensor,
    fc2_w: torch.Tensor,
    fc2_b: torch.Tensor,
) -> torch.Tensor:
    """CLIP vision FFN matching MiniMaxM3VLVisionMLP.forward.

    The standard CLIP vision-tower MLP: a 1280 -> 5120 up-projection (fc1),
    a GELU non-linearity, then a 5120 -> 1280 down-projection (fc2). BOTH
    ``fc1`` and ``fc2`` are ``nn.Linear`` WITH bias (the HF module uses the
    default ``bias=True``).

    Activation: ``hidden_act = "gelu"`` in vision_config maps via
    ``ACT2FN["gelu"]`` to ``transformers.activations.GELUActivation``, which is
    the EXACT / erf-based GELU (``nn.functional.gelu(..., approximate="none")``),
    i.e. ``0.5 * x * (1 + erf(x / sqrt(2)))`` -- NOT the tanh approximation.

    Exact HF pipeline order:
      1. hidden = fc1(x)              # Linear 1280 -> 5120, +bias
      2. hidden = gelu(hidden)        # erf GELU
      3. hidden = fc2(hidden)         # Linear 5120 -> 1280, +bias

    Args:
        x: hidden states [..., hidden_size (1280)].
        fc1_w: fc1 weight [intermediate_size (5120), hidden_size (1280)].
        fc1_b: fc1 bias [intermediate_size (5120)].
        fc2_w: fc2 weight [hidden_size (1280), intermediate_size (5120)].
        fc2_b: fc2 bias [hidden_size (1280)].

    Returns:
        MLP output [..., hidden_size (1280)], x's dtype.
    """
    h = torch.nn.functional.linear(x, fc1_w, fc1_b)
    h = torch.nn.functional.gelu(h, approximate="none")
    h = torch.nn.functional.linear(h, fc2_w, fc2_b)
    return h


def shared_expert_forward(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """MiniMax-M3 always-on SHARED expert (n_shared_experts=1, SwiGLU-OAI).

    Mirrors HF (transformers 5.12.0) ``MiniMaxM3VLSparseMoeBlock``: the shared
    expert is a ``MiniMaxM3VLDenseMLP(intermediate_size=shared_intermediate_size)``
    (text_config: hidden 6144, shared_intermediate_size 3072, swiglu_alpha 1.702,
    swiglu_limit 7.0). It runs on ALL tokens (no routing).

    HF block forward::

        shared_output = self.shared_experts(hidden_states)      # this function
        hidden_states = self.experts(...) * routed_scaling_factor
        hidden_states = hidden_states + shared_output           # plain add

    IMPORTANT -- no extra gating/scaling on the shared output:
      * The shared expert output is added DIRECTLY to the routed output.
      * ``routed_scaling_factor`` (2.0) is applied ONLY to the routed-expert
        output, NOT to the shared output.
      * Unlike DeepSeek/Qwen "shared-expert-gate" variants, there is NO sigmoid
        gate on the shared output here -- it is a bare DenseMLP.

    Weight-storage note: the HF *module* fuses gate+up into a single
    ``gate_up_proj`` ``[2*inter, hidden]`` and chunks it ``gate, up``. The
    MiniMax-M3 *checkpoint* stores SEPARATE
    ``...block_sparse_moe.shared_experts.{gate,up,down}_proj.weight`` (gate/up
    each ``[3072, 6144]``, down ``[6144, 3072]``). Passing the separate gate_w
    and up_w to ``swigluoai_mlp_forward`` reproduces the fused-chunk behaviour
    exactly (chunk's first half == gate_proj, second half == up_proj).

    Args:
        x: Input activations ``[..., hidden_size]`` (hidden 6144).
        gate_w: ``shared_experts.gate_proj.weight`` ``[3072, 6144]``.
        up_w: ``shared_experts.up_proj.weight`` ``[3072, 6144]``.
        down_w: ``shared_experts.down_proj.weight`` ``[6144, 3072]``.
        alpha: ``swiglu_alpha`` (sigmoid temperature), default 1.702.
        limit: ``swiglu_limit`` clamp bound, default 7.0.

    Returns:
        Tensor ``[..., hidden_size]`` -- the shared-expert contribution to add
        to the routed-expert output.
    """
    return swigluoai_mlp_forward(x, gate_w, up_w, down_w, alpha=alpha, limit=limit)


def dense_decoder_layer_forward(
    x: torch.Tensor,
    weights_dict: dict,
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_attention_heads: int = 64,
    num_key_value_heads: int = 4,
    head_dim: int = 128,
    eps: float = 1e-6,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Full DENSE decoder layer matching MiniMaxM3VLDecoderLayer (layers 0-2).

    Composes the existing reference helpers in the EXACT HF order for a dense
    layer (``mlp_layer_types[layer_idx] != "sparse"`` and ``layer_types[layer_idx]
    != "minimax_m3_sparse"`` -> full GQA attention + dense SwiGLU-OAI MLP). The
    HF ``MiniMaxM3VLDecoderLayer.forward`` is plain PRE-NORM with NO residual
    scaling::

        residual = x
        h = input_layernorm(x)                # Gemma RMSNorm, (1+w) scale, eps 1e-6
        h, _ = self_attn(h)                   # full GQA (q/k-norm + partial rope)
        x = residual + h
        residual = x
        h = post_attention_layernorm(x)       # Gemma RMSNorm
        h = mlp(h)                            # dense SwiGLU-OAI, dense_intermediate 12288
        x = residual + h

    Both norms are the same Gemma-style :func:`rms_norm_forward` (``(1+w)`` scale,
    fp32 reduce). The attention path is :func:`gqa_attention_forward` and the MLP
    path is :func:`swigluoai_mlp_forward` (separate gate/up weights, the
    ``(up + 1.0) * glu`` GPT-OSS form, clamp limit 7.0, sigmoid temp 1.702).
    There is NO residual scaling anywhere in this layer. The dense intermediate
    size is 12288 (``dense_intermediate_size``), vs the MoE expert intermediate
    of 3072.

    Args:
        x: hidden states [B, S, hidden_size (6144)].
        weights_dict: keys ``input_layernorm`` and ``post_attention_layernorm``
            (norm gammas [hidden]); attention ``q_proj/k_proj/v_proj/o_proj/
            q_norm/k_norm``; MLP ``gate_proj/up_proj/down_proj`` (each a
            ``.weight`` tensor).
        cos, sin: partial-rope tables [B, S, rotary_dim (64)] from
            build_rope_cos_sin(theta=5e6, rotary_dim=64).
        num_attention_heads, num_key_value_heads, head_dim, eps: config.
        swiglu_alpha, swiglu_limit: dense MLP SwiGLU-OAI params.
        attention_mask: optional additive mask [B,1,S,S]; None -> internal causal.

    Returns:
        Decoder layer output [B, S, hidden_size], x's dtype.
    """
    residual = x
    h = rms_norm_forward(x, weights_dict["input_layernorm"], eps=eps)
    h = gqa_attention_forward(
        h,
        weights_dict,
        cos,
        sin,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        eps=eps,
        attention_mask=attention_mask,
    )
    x = residual + h

    residual = x
    h = rms_norm_forward(x, weights_dict["post_attention_layernorm"], eps=eps)
    h = swigluoai_mlp_forward(
        h,
        weights_dict["gate_proj"],
        weights_dict["up_proj"],
        weights_dict["down_proj"],
        alpha=swiglu_alpha,
        limit=swiglu_limit,
    )
    x = residual + h
    return x


def moe_experts_forward(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_weights: dict,
    moe_intermediate: int = 3072,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """MiniMax-M3 routed sparse-MoE experts (128 experts, top-4, SwiGLU-OAI).

    This is the ROUTED-experts contribution ONLY of
    ``MiniMaxM3VLSparseMoeBlock`` (the always-on ``shared_experts`` DenseMLP is a
    SEPARATE block and is NOT included here).

    Mirrors HF ``MiniMaxM3VLExperts.forward`` (transformers 5.12.0): for each
    token, run its ``top_k`` selected experts (each a SwiGLU-OAI MLP with
    ``moe_intermediate_size`` 3072), weight each expert output by its routing
    weight, and sum. Each expert is exactly ``swigluoai_mlp_forward`` with the
    fused ``gate_up_proj`` split into separate gate/up:

    Weight-layout note (checkpoint vs HF module):
      * The MiniMax-M3 *checkpoint* stores SEPARATE per-expert tensors
        ``...experts.{i}.{w1,w2,w3}.weight`` (Mixtral/DeepSeek convention):
          - ``w1`` -> gate  ``[moe_inter, hidden]``  (3072, 6144)
          - ``w3`` -> up     ``[moe_inter, hidden]``  (3072, 6144)
          - ``w2`` -> down  ``[hidden, moe_inter]``  (6144, 3072)
      * The HF *module* (``MiniMaxM3VLExperts``) stores STACKED 3D parameters
        ``gate_up_proj`` ``[num_experts, 2*moe_inter, hidden]`` and ``down_proj``
        ``[num_experts, hidden, moe_inter]``, where ``gate_up_proj[i]`` ==
        ``cat([w1_i, w3_i], dim=0)`` (gate first, up second; NOT interleaved,
        consumed via ``chunk(2, dim=-1)`` in ``_apply_gate``).
    This decomposition matches the gpt_oss MoE experts lineage
    (``models/demos/gpt_oss/tt/experts``, ``mlp.py``, ``topk.py``): a top-k
    router selects experts, per-expert SwiGLU-OAI MLPs are run on the gathered
    tokens, scaled by the routing weight, and scatter-added back.

    Scaling: ``topk_weights`` already include the ``routed_scaling_factor`` (2.0)
    folded in by ``moe_gate_forward`` (which HF applies as
    ``hidden = experts(...) * routed_scaling_factor`` in the SparseMoeBlock).
    Therefore this function does NOT scale again.

    Args:
        x: hidden states ``[..., hidden_size]``; flattened to ``[tokens, hidden]``.
        topk_indices: ``[tokens, top_k]`` long, selected expert ids (from gate).
        topk_weights: ``[tokens, top_k]`` fp32, routing weights (scaling folded in).
        expert_weights: dict ``{expert_id: {"gate_w": w1 [inter,hidden],
            "up_w": w3 [inter,hidden], "down_w": w2 [hidden,inter]}}``. Only the
            experts actually selected by ``topk_indices`` need be present.
        moe_intermediate: SwiGLU-OAI intermediate size (3072).
        alpha, limit: SwiGLU-OAI ``swiglu_alpha`` / ``swiglu_limit``.

    Returns:
        Routed-expert output, same shape as ``x`` (``[..., hidden_size]``).
    """
    orig_shape = x.shape
    hidden_dim = orig_shape[-1]
    hidden_states = x.reshape(-1, hidden_dim)
    tokens, top_k = topk_indices.shape

    final = torch.zeros_like(hidden_states)
    for t in range(tokens):
        for j in range(top_k):
            eid = int(topk_indices[t, j].item())
            w = expert_weights[eid]
            expert_out = swigluoai_mlp_forward(
                hidden_states[t : t + 1],
                gate_w=w["gate_w"],
                up_w=w["up_w"],
                down_w=w["down_w"],
                alpha=alpha,
                limit=limit,
            )
            final[t] += (expert_out * topk_weights[t, j]).squeeze(0).to(final.dtype)
    return final.reshape(orig_shape)
