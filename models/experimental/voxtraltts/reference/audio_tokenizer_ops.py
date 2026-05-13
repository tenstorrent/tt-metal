# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU reference ops for Voxtral audio tokenizer — goldens for TT PCC tests, no vllm_omni dependency."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def causal_conv1d_pad_amounts(seq_len: int, kernel_size: int, stride: int, dilation: int = 1) -> tuple[int, int]:
    """Left and right pad widths applied before ``conv1d`` in ``CausalConv1d.forward``."""
    effective_kernel_size = (kernel_size - 1) * dilation + 1
    padding_total = effective_kernel_size - stride
    n_frames = (seq_len - effective_kernel_size + padding_total) / stride + 1
    target_length = (math.ceil(n_frames) - 1) * stride + (effective_kernel_size - padding_total)
    extra_padding = int(target_length - seq_len)
    return int(padding_total), extra_padding


def pad1d_non_reflect(x: torch.Tensor, paddings: tuple[int, int], *, mode: str, value: float = 0.0) -> torch.Tensor:
    """``F.pad`` on last dim; handles short-sequence reflect via temporary extend."""
    if mode == "reflect":
        length = x.shape[-1]
        padding_left, padding_right = paddings
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    return F.pad(x, paddings, mode, value)


def causal_conv1d_reference(
    x_ncl: torch.Tensor,
    weight_oik: torch.Tensor,
    *,
    kernel_size: int,
    stride: int,
    dilation: int = 1,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """``CausalConv1d.forward`` in float32: NCL → NCL."""
    x = x_ncl.to(torch.float32)
    w = weight_oik.to(torch.float32)
    assert w.dim() == 3 and w.shape[2] == kernel_size
    t = x.shape[2]
    pl, pr = causal_conv1d_pad_amounts(t, kernel_size, stride, dilation)
    x = pad1d_non_reflect(x, (pl, pr), mode=pad_mode)
    return F.conv1d(x, w, bias=None, stride=stride, padding=0, dilation=dilation, groups=1)


def causal_conv_transpose1d_reference(
    x_ncl: torch.Tensor,
    weight_iok: torch.Tensor,
    *,
    kernel_size: int,
    stride: int,
    trim_ratio: float = 1.0,
) -> torch.Tensor:
    """``CausalConvTranspose1d.forward`` in float32 (``bias=None``, ``groups=1``)."""
    x = x_ncl.to(torch.float32)
    w = weight_iok.to(torch.float32)
    assert w.dim() == 3 and w.shape[2] == kernel_size
    out = F.conv_transpose1d(x, w, bias=None, stride=stride, groups=1)
    total_padding = kernel_size - stride
    right_padding = math.ceil(total_padding * trim_ratio)
    left_padding = total_padding - right_padding
    if right_padding == 0:
        return out[..., left_padding:]
    return out[..., left_padding : out.shape[-1] - right_padding]


def fuse_weight_norm_conv1d(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Fuse ``weight_norm`` storage ``(g, v)`` → normalized conv weight in bf16."""
    g = g.to(torch.float32)
    v = v.to(torch.float32)
    if g.shape[-1] == 1 and g.ndim == 3:
        norm = v.reshape(v.shape[0], -1).norm(dim=1).reshape(-1, 1, 1)
    else:
        norm = v.reshape(v.shape[0], -1).norm(dim=1)
        while norm.ndim < v.ndim:
            norm = norm.unsqueeze(-1)
    return (g * (v / norm)).to(dtype=torch.bfloat16)


def causal_conv1d_left_pad_reference(
    x_ncl: torch.Tensor,
    weight_oik: torch.Tensor,
    *,
    left_pad: int,
    stride: int = 1,
) -> torch.Tensor:
    """Left causal pad + ``conv1d`` in bf16 (``input_proj`` / encoder strided blocks)."""
    x = x_ncl.to(torch.bfloat16)
    if left_pad:
        x = F.pad(x, (left_pad, 0))
    return F.conv1d(x, weight_oik.to(torch.bfloat16), bias=None, stride=stride, padding=0, dilation=1, groups=1)


def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm over last dim."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def audio_tokenizer_alibi_slopes(n_heads: int) -> torch.Tensor:
    """ALiBi slopes for ``n_heads`` (Mistral / vLLM audio tokenizer pattern)."""

    def slopes_power_of_2(n: int) -> torch.Tensor:
        r = 2.0 ** (-8.0 / n)
        return torch.tensor([r**i for i in range(n)], dtype=torch.float32)

    if math.log2(n_heads).is_integer():
        return slopes_power_of_2(n_heads)
    m = 2 ** math.floor(math.log2(n_heads))
    return torch.cat([slopes_power_of_2(m), slopes_power_of_2(2 * m)[::2][: n_heads - m]])


def audio_tokenizer_sliding_window_attention_bias(n_heads: int, seq_len: int, sliding_window: int) -> torch.Tensor:
    """Additive attention bias ``[1, n_heads, seq_len, seq_len]`` (ALiBi + causal + sliding window)."""
    pos = torch.arange(seq_len)
    rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
    slopes = audio_tokenizer_alibi_slopes(n_heads).view(n_heads, 1, 1)
    mask = slopes * rel_pos.unsqueeze(0).to(torch.float32)
    mask = mask.masked_fill(rel_pos.unsqueeze(0) > 0, -1e4)
    outside_window = rel_pos < -sliding_window
    mask = mask.masked_fill(outside_window.unsqueeze(0), -1e4)
    return mask.unsqueeze(0)


def decoder_transformer_block_reference(
    x_btd: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    tokenizer_cfg: Any,
    *,
    block_index: int,
    layer_index: int,
) -> torch.Tensor:
    """One ``decoder_blocks.{block_index}.layers.{layer_index}`` TransformerBlock in float32."""
    prefix = f"decoder_blocks.{block_index}.layers.{layer_index}"
    x = x_btd.float()

    r = rms_norm_reference(x, state_dict[f"{prefix}.attention_norm.weight"].float(), float(tokenizer_cfg.norm_eps))
    wq = state_dict[f"{prefix}.attention.wq.weight"].float()
    wk = state_dict[f"{prefix}.attention.wk.weight"].float()
    wv = state_dict[f"{prefix}.attention.wv.weight"].float()
    wo = state_dict[f"{prefix}.attention.wo.weight"].float()
    q = F.linear(r, wq)
    k = F.linear(r, wk)
    v = F.linear(r, wv)
    if tokenizer_cfg.qk_norm:
        q = rms_norm_reference(
            q, state_dict[f"{prefix}.attention.q_norm.weight"].float(), float(tokenizer_cfg.qk_norm_eps)
        )
        k = rms_norm_reference(
            k, state_dict[f"{prefix}.attention.k_norm.weight"].float(), float(tokenizer_cfg.qk_norm_eps)
        )

    b, t, _ = q.shape
    q = q.view(b, t, tokenizer_cfg.n_heads, tokenizer_cfg.head_dim).transpose(1, 2)
    k = k.view(b, t, tokenizer_cfg.n_kv_heads, tokenizer_cfg.head_dim).transpose(1, 2)
    v = v.view(b, t, tokenizer_cfg.n_kv_heads, tokenizer_cfg.head_dim).transpose(1, 2)
    mask = audio_tokenizer_sliding_window_attention_bias(
        tokenizer_cfg.n_heads, t, tokenizer_cfg.attn_sliding_window_size
    )
    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    attn = attn.transpose(1, 2).reshape(b, t, tokenizer_cfg.n_heads * tokenizer_cfg.head_dim)
    attn = F.linear(attn, wo)
    if tokenizer_cfg.layer_scale:
        attn = attn * state_dict[f"{prefix}.attention_scale"].float()
    h = x + attn

    r = rms_norm_reference(h, state_dict[f"{prefix}.ffn_norm.weight"].float(), float(tokenizer_cfg.norm_eps))
    w1 = state_dict[f"{prefix}.feed_forward.w1.weight"].float()
    w2 = state_dict[f"{prefix}.feed_forward.w2.weight"].float()
    w3 = state_dict[f"{prefix}.feed_forward.w3.weight"].float()
    ff = F.linear(F.silu(F.linear(r, w1)) * F.linear(r, w3), w2)
    if tokenizer_cfg.layer_scale:
        ff = ff * state_dict[f"{prefix}.ffn_scale"].float()
    return (h + ff).to(torch.bfloat16)


def causal_conv1d_reference_bf16(
    x_ncl: torch.Tensor,
    weight_oik: torch.Tensor,
    *,
    kernel_size: int,
    stride: int,
    dilation: int = 1,
    pad_mode: str = "replicate",
) -> torch.Tensor:
    """Same as :func:`causal_conv1d_reference` but in bf16 (tracks TTNN decoder convs)."""
    x = x_ncl.to(torch.bfloat16)
    w = weight_oik.to(torch.bfloat16)
    assert w.dim() == 3 and w.shape[2] == kernel_size
    t = x.shape[2]
    pl, pr = causal_conv1d_pad_amounts(t, kernel_size, stride, dilation)
    x = pad1d_non_reflect(x, (pl, pr), mode=pad_mode)
    return F.conv1d(x, w, bias=None, stride=stride, padding=0, dilation=dilation, groups=1)


def causal_conv_transpose1d_reference_bf16(
    x_ncl: torch.Tensor,
    weight_iok: torch.Tensor,
    *,
    kernel_size: int,
    stride: int,
    trim_ratio: float = 1.0,
) -> torch.Tensor:
    """Same as :func:`causal_conv_transpose1d_reference` but in bf16."""
    x = x_ncl.to(torch.bfloat16)
    w = weight_iok.to(torch.bfloat16)
    assert w.dim() == 3 and w.shape[2] == kernel_size
    out = F.conv_transpose1d(x, w, bias=None, stride=stride, groups=1)
    total_padding = kernel_size - stride
    right_padding = math.ceil(total_padding * trim_ratio)
    left_padding = total_padding - right_padding
    if right_padding == 0:
        return out[..., left_padding:]
    return out[..., left_padding : out.shape[-1] - right_padding]


def decoder_transformer_block_reference_bf16(
    x_btd: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    tokenizer_cfg: Any,
    *,
    block_index: int,
    layer_index: int,
) -> torch.Tensor:
    """Same as :func:`decoder_transformer_block_reference` but in bf16 (RMS reductions stay float32)."""
    prefix = f"decoder_blocks.{block_index}.layers.{layer_index}"
    x = x_btd.to(torch.bfloat16)
    eps = float(tokenizer_cfg.norm_eps)
    qke = float(tokenizer_cfg.qk_norm_eps)

    def _rms_bf16(t_bf: torch.Tensor, w_key: str) -> torch.Tensor:
        w = state_dict[f"{prefix}.{w_key}"].to(torch.bfloat16)
        return rms_norm_reference(t_bf.float(), w.float(), eps).to(torch.bfloat16)

    r = _rms_bf16(x, "attention_norm.weight")
    wq = state_dict[f"{prefix}.attention.wq.weight"].to(torch.bfloat16)
    wk = state_dict[f"{prefix}.attention.wk.weight"].to(torch.bfloat16)
    wv = state_dict[f"{prefix}.attention.wv.weight"].to(torch.bfloat16)
    wo = state_dict[f"{prefix}.attention.wo.weight"].to(torch.bfloat16)
    q = F.linear(r, wq)
    k = F.linear(r, wk)
    v = F.linear(r, wv)
    if tokenizer_cfg.qk_norm:
        q = rms_norm_reference(q.float(), state_dict[f"{prefix}.attention.q_norm.weight"].float(), qke).to(
            torch.bfloat16
        )
        k = rms_norm_reference(k.float(), state_dict[f"{prefix}.attention.k_norm.weight"].float(), qke).to(
            torch.bfloat16
        )

    b, t, _ = q.shape
    q = q.view(b, t, tokenizer_cfg.n_heads, tokenizer_cfg.head_dim).transpose(1, 2)
    k = k.view(b, t, tokenizer_cfg.n_kv_heads, tokenizer_cfg.head_dim).transpose(1, 2)
    v = v.view(b, t, tokenizer_cfg.n_kv_heads, tokenizer_cfg.head_dim).transpose(1, 2)
    mask = audio_tokenizer_sliding_window_attention_bias(
        tokenizer_cfg.n_heads, t, tokenizer_cfg.attn_sliding_window_size
    ).to(torch.bfloat16)
    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    attn = attn.transpose(1, 2).reshape(b, t, tokenizer_cfg.n_heads * tokenizer_cfg.head_dim)
    attn = F.linear(attn, wo)
    if tokenizer_cfg.layer_scale:
        attn = attn * state_dict[f"{prefix}.attention_scale"].to(torch.bfloat16)
    h = x + attn

    r = _rms_bf16(h, "ffn_norm.weight")
    w1 = state_dict[f"{prefix}.feed_forward.w1.weight"].to(torch.bfloat16)
    w2 = state_dict[f"{prefix}.feed_forward.w2.weight"].to(torch.bfloat16)
    w3 = state_dict[f"{prefix}.feed_forward.w3.weight"].to(torch.bfloat16)
    ff = F.linear(F.silu(F.linear(r, w1)) * F.linear(r, w3), w2)
    if tokenizer_cfg.layer_scale:
        ff = ff * state_dict[f"{prefix}.ffn_scale"].to(torch.bfloat16)
    return h + ff


def decoder_blocks_stack_reference(
    latent_ncl_bf16: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    tokenizer_cfg: Any,
) -> torch.Tensor:
    """Full ``decoder_blocks`` 0→7 in bf16 (no ``output_proj``). Returns ``[B, T_out, dim]`` bf16."""
    from models.experimental.voxtraltts.reference.voxtral_config import parse_csv_ints
    from models.experimental.voxtraltts.tt.audio_tokenizer.conv import (
        resolve_decoder_block_causal_conv_fused_weight,
        resolve_decoder_block_conv_transpose_fused_weight,
    )

    kerns = parse_csv_ints(tokenizer_cfg.decoder_convs_kernels_str)
    strides = parse_csv_ints(tokenizer_cfg.decoder_convs_strides_str)
    w0 = resolve_decoder_block_causal_conv_fused_weight(state_dict, 0)
    x_ncl = causal_conv1d_reference_bf16(
        latent_ncl_bf16, w0.to(torch.bfloat16), kernel_size=kerns[0], stride=strides[0], pad_mode="replicate"
    )
    x_btd = x_ncl.permute(0, 2, 1).contiguous()

    for tb_idx, tr_idx in ((1, 2), (3, 4), (5, 6)):
        for li in (0, 1):
            x_btd = decoder_transformer_block_reference_bf16(
                x_btd, state_dict, tokenizer_cfg, block_index=tb_idx, layer_index=li
            )
        kern_i = tr_idx // 2
        w_t = resolve_decoder_block_conv_transpose_fused_weight(state_dict, tr_idx)
        x_ncl = x_btd.permute(0, 2, 1).contiguous()
        x_ncl = causal_conv_transpose1d_reference_bf16(
            x_ncl, w_t.to(torch.bfloat16), kernel_size=kerns[kern_i], stride=strides[kern_i]
        )
        x_btd = x_ncl.permute(0, 2, 1).contiguous()

    for li in (0, 1):
        x_btd = decoder_transformer_block_reference_bf16(
            x_btd, state_dict, tokenizer_cfg, block_index=7, layer_index=li
        )
    return x_btd


def output_proj_mel_ncl_reference_bf16(
    hidden_btd_bf16: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    """``output_proj`` causal conv: ``[B, T, dim]`` → ``[B, patch_size, T_mel]`` NCL bf16."""
    from models.experimental.voxtraltts.tt.audio_tokenizer.conv import resolve_output_proj_causal_conv_fused_weight

    w = resolve_output_proj_causal_conv_fused_weight(state_dict)
    k = int(w.shape[2])
    x_ncl = hidden_btd_bf16.permute(0, 2, 1).contiguous()
    return causal_conv1d_reference_bf16(x_ncl, w.to(torch.bfloat16), kernel_size=k, stride=1, pad_mode="replicate")


def semantic_codebook_centroids_bf16(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    """EMA centroid table ``[n_semantic, semantic_dim]`` bf16 from ``quantizer.semantic_codebook.*``."""
    es = state_dict["quantizer.semantic_codebook.embedding_sum"]
    u = state_dict["quantizer.semantic_codebook.cluster_usage"]
    return (es.to(torch.float32) / u.to(torch.float32).clamp(min=1.0).unsqueeze(-1)).to(torch.bfloat16)


def semantic_codebook_quantize_indices_reference(
    x_bts: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Argmin L2 to EMA centroids. ``x_bts``: ``[B, T, semantic_dim]`` → ``[B, T]`` int64 indices."""
    c = semantic_codebook_centroids_bf16(state_dict).to(torch.float32)
    x = x_bts.to(torch.float32)
    b, t, s = x.shape
    if s != c.shape[1]:
        raise ValueError(f"semantic dim mismatch: x has {s}, centroids have {c.shape[1]}")
    x2 = x.reshape(b * t, s)
    xn = x2.pow(2).sum(-1, keepdim=True)
    cn = (c.pow(2).sum(-1)).unsqueeze(0)
    dots = x2 @ c.transpose(0, 1)
    return (xn + cn - 2.0 * dots).argmin(dim=-1).to(torch.int64).view(b, t)


def audio_codebook_embedding_reference(indices_bt: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``F.embedding`` for MM codebook table ``weight`` ``[num, dim]``; ``indices_bt`` int ``[B, T]``."""
    return torch.nn.functional.embedding(indices_bt.long(), weight.to(torch.bfloat16))


def acoustic_latent_from_codes(acoustic_codes: torch.Tensor, n_levels: int = 21) -> torch.Tensor:
    """FSQ decode: ``[B, 36, T]`` integer codes → float ``[-1, 1]`` via ``codes * 2/(n_levels-1) - 1``."""
    return acoustic_codes.float() * (2.0 / (n_levels - 1)) - 1.0


def audio_tokenizer_latent_from_codes(
    codes_b37t: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    *,
    n_acoustic_levels: int = 21,
) -> torch.Tensor:
    """``[B, 37, T]`` codes → ``[B, semantic_dim+acoustic_dim, T]`` float32 latent.

    Codebook 0: semantic EMA centroid lookup. Codebooks 1-36: acoustic FSQ rescale.
    """
    b, n_cb, t = codes_b37t.shape
    semantic_codes = codes_b37t[:, 0, :].long()
    centroids = semantic_codebook_centroids_bf16(state_dict).float()
    semantic_emb = F.embedding(semantic_codes, centroids).permute(0, 2, 1)  # [B, 256, T]
    acoustic_emb = acoustic_latent_from_codes(codes_b37t[:, 1:, :].float(), n_levels=n_acoustic_levels)
    return torch.cat([semantic_emb, acoustic_emb], dim=1)


def pretransform_decode(mel_b_patches_t: torch.Tensor, *, channels: int = 1) -> torch.Tensor:
    """Inverse pretransform: ``[B, channels*patch_size, T]`` → ``[B, channels, T*patch_size]`` waveform."""
    b, c_patch, t = mel_b_patches_t.shape
    patch_size = c_patch // channels
    return mel_b_patches_t.reshape(b, channels, patch_size, t).permute(0, 1, 3, 2).reshape(b, channels, t * patch_size)


def audio_tokenizer_decode_reference(
    codes_b37t: torch.Tensor,
    state_dict: dict[str, torch.Tensor],
    tokenizer_cfg: Any,
) -> torch.Tensor:
    """Full decode: ``[B, 37, T]`` codes → ``[B, 1, T_audio]`` float32 waveform (CPU, bf16 path)."""
    latent = audio_tokenizer_latent_from_codes(
        codes_b37t.cpu(), state_dict, n_acoustic_levels=tokenizer_cfg.acoustic_codebook_size
    ).to(torch.bfloat16)
    hidden = decoder_blocks_stack_reference(latent, state_dict, tokenizer_cfg)
    mel = output_proj_mel_ncl_reference_bf16(hidden, state_dict)
    return pretransform_decode(mel, channels=tokenizer_cfg.channels).float()


def audio_tokenizer_mm_embedding_offsets(audio_model_cfg: Any) -> torch.Tensor:
    """Per-codebook index offsets for the shared MM embedding table (matches ``MultiVocabEmbeddings.offsets``)."""
    n_special = 2  # EMPTY_AUDIO + END_AUDIO
    sizes = [audio_model_cfg.semantic_codebook_size + n_special] + [
        audio_model_cfg.acoustic_codebook_size + n_special
    ] * audio_model_cfg.n_acoustic_codebook
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    return torch.tensor(offsets, dtype=torch.long)


def audio_tokenizer_encode_tokens_reference(
    codes_b_ncb_t: torch.Tensor,
    mm_embedding_weight: torch.Tensor,
    audio_model_cfg: Any,
) -> torch.Tensor:
    """``[B, n_codebooks, T]`` codes → ``[T, mm_dim]`` multimodal embeddings (offset + lookup + sum over codebooks)."""
    offsets = audio_tokenizer_mm_embedding_offsets(audio_model_cfg)
    indices = codes_b_ncb_t.long() + offsets.to(codes_b_ncb_t.device).view(1, -1, 1)
    emb = F.embedding(indices, mm_embedding_weight.to(torch.bfloat16))  # [B, n_cb, T, mm_dim]
    return emb.sum(dim=1).squeeze(0)
