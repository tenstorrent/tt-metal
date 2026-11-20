# SPDX-FileCopyrightText: © 2025 Meituan
# SPDX-License-Identifier: MIT
#
# Adapted from https://github.com/meituan-longcat/LongCat-Video (longcat_video/modules/, commit 58eb336),
# licensed under MIT. Modified by Tenstorrent as the PyTorch reference for the tt-metal port: single-speaker
# paths only (no multitalk masks, block-sparse attention, context parallelism or KV cache), and torch SDPA in
# place of flash-attn so it runs on host CPU.

"""LongCat-Video-Avatar DiT (the 1.5 / 1.6 architecture).

48 single-stream blocks, each: per-frame adaLN-modulated self-attention with 3D RoPE, text cross-attention,
per-frame audio cross-attention gated by a second adaLN, and a modulated SwiGLU feed-forward.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from safetensors.torch import load_file


def _attn(q, k, v):
    """q, k, v: [B, H, S, D]."""
    return F.scaled_dot_product_attention(q, k, v)


# ---------------------------------------------------------------------------------------------------------------------
# Shared blocks (longcat_video/modules/blocks.py)
# ---------------------------------------------------------------------------------------------------------------------


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm_FP32(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        out = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)
        return out * self.weight


class LayerNorm_FP32(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        ).to(inputs.dtype)


def modulate_fp32(norm_func, x, shift, scale):
    dtype = x.dtype
    x = norm_func(x.to(torch.float32))
    x = x * (scale + 1) + shift
    return x.to(dtype)


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size, in_chans, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class FinalLayer_FP32(nn.Module):
    def __init__(self, hidden_size, num_patch, out_channels, adaln_tembed_dim):
        super().__init__()
        self.norm_final = LayerNorm_FP32(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 2 * hidden_size, bias=True))

    def forward(self, x, t, latent_shape):
        B, N, C = x.shape
        T = latent_shape[0]
        shift, scale = self.adaLN_modulation(t.float()).unsqueeze(2).chunk(2, dim=-1)
        x = modulate_fp32(self.norm_final, x.view(B, T, -1, C), shift, scale).view(B, N, C)
        return self.linear(x.to(self.linear.weight.dtype))


class TimestepEmbedder(nn.Module):
    def __init__(self, t_embed_dim, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, t_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(dtype)
        return self.mlp(t_freq)


class CaptionEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.y_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, caption):
        return self.y_proj(caption)


# ---------------------------------------------------------------------------------------------------------------------
# RoPE (longcat_video/modules/avatar/rope_3d.py)
# ---------------------------------------------------------------------------------------------------------------------


def broadcat(tensors, dim=-1):
    shape_len = len(tensors[0].shape)
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * len(tensors)), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def rope_3d_freqs(head_dim, grid_size, frame_index=None, num_ref_latents=None, base=10000):
    """[T*H*W, head_dim] rotation angles, interleaved pairs (``(n r)`` with r=2)."""
    num_frames, height, width = grid_size
    dim_t = head_dim - 4 * (head_dim // 6)
    dim_h = 2 * (head_dim // 6)
    dim_w = 2 * (head_dim // 6)
    freqs_t = 1.0 / (base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
    freqs_h = 1.0 / (base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
    freqs_w = 1.0 / (base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
    if frame_index is not None and num_ref_latents is not None:
        grid_t = torch.concat(
            [
                torch.tensor([frame_index], dtype=torch.float32),
                torch.arange(0, num_frames - num_ref_latents, dtype=torch.float32),
            ],
            dim=0,
        )
    else:
        grid_t = torch.from_numpy(np.linspace(0, num_frames, num_frames, endpoint=False, dtype=np.float32)).float()
    grid_h = torch.from_numpy(np.linspace(0, height, height, endpoint=False, dtype=np.float32)).float()
    grid_w = torch.from_numpy(np.linspace(0, width, width, endpoint=False, dtype=np.float32)).float()
    freqs_t = repeat(torch.einsum("..., f -> ... f", grid_t, freqs_t), "... n -> ... (n r)", r=2)
    freqs_h = repeat(torch.einsum("..., f -> ... f", grid_h, freqs_h), "... n -> ... (n r)", r=2)
    freqs_w = repeat(torch.einsum("..., f -> ... f", grid_w, freqs_w), "... n -> ... (n r)", r=2)
    freqs = broadcat((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
    return rearrange(freqs, "T H W D -> (T H W) D")


def apply_rope(q, k, freqs):
    cos = freqs.cos()[None, None].float()
    sin = freqs.sin()[None, None].float()
    q_, k_ = q.float(), k.float()
    q_ = (q_ * cos) + (rotate_half(q_) * sin)
    k_ = (k_ * cos) + (rotate_half(k_) * sin)
    return q_.type_as(q), k_.type_as(k)


# ---------------------------------------------------------------------------------------------------------------------
# Attention (longcat_video/modules/avatar/attention.py, longcat_video/modules/attention.py)
# ---------------------------------------------------------------------------------------------------------------------


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, shape, num_cond_latents=None, num_ref_latents=None, ref_img_index=None, mask_frame_range=None):
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, self.num_heads, self.head_dim).permute((2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, k, rope_3d_freqs(self.head_dim, shape, ref_img_index, num_ref_latents).to(q.device))

        N_t = shape[0]
        if num_cond_latents is not None and num_cond_latents == 1:
            n_cond = num_cond_latents * (N // N_t)
            x_cond = _attn(q[:, :, :n_cond], k[:, :, :n_cond], v[:, :, :n_cond])
            x_noise = _attn(q[:, :, n_cond:], k, v)
            x = torch.cat([x_cond, x_noise], dim=2)
        elif num_cond_latents is not None and num_cond_latents > 1:
            assert num_ref_latents is not None and ref_img_index is not None
            n_ref = N // N_t
            n_cond = num_cond_latents * (N // N_t)
            x_ref = _attn(q[:, :, :n_ref], k[:, :, :n_ref], v[:, :, :n_ref])
            x_cond = _attn(q[:, :, n_ref:n_cond], k[:, :, n_ref:n_cond], v[:, :, n_ref:n_cond])
            if num_cond_latents == N_t:
                x = torch.cat([x_ref, x_cond], dim=2)
            else:
                q_noise = q[:, :, n_cond:]
                start_noise, end_noise, num_noisy_frames = 0, 0, N_t - num_cond_latents
                if mask_frame_range is not None and mask_frame_range > 0:
                    start_noise = ref_img_index - mask_frame_range - num_cond_latents + num_ref_latents
                    end_noise = ref_img_index + mask_frame_range - num_cond_latents + num_ref_latents + 1
                if start_noise >= 0 and end_noise > start_noise and end_noise <= num_noisy_frames:
                    s, e = start_noise * (N // N_t), end_noise * (N // N_t)
                    x_noise = torch.cat(
                        [
                            _attn(q_noise[:, :, :s], k, v),
                            _attn(q_noise[:, :, s:e], k[:, :, n_ref:], v[:, :, n_ref:]),
                            _attn(q_noise[:, :, e:], k, v),
                        ],
                        dim=2,
                    )
                else:
                    x_noise = _attn(q_noise, k, v)
                x = torch.cat([x_ref, x_cond, x_noise], dim=2)
        else:
            x = _attn(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)

    def _process_cross_attn(self, x, cond, kv_seqlen):
        B, N, C = x.shape
        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        k, v = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim).unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        outs, k_start = [], 0
        for b in range(B):
            qb = q[:, b * N : (b + 1) * N].transpose(1, 2)
            kb = k[:, k_start : k_start + kv_seqlen[b]].transpose(1, 2)
            vb = v[:, k_start : k_start + kv_seqlen[b]].transpose(1, 2)
            k_start += kv_seqlen[b]
            outs.append(_attn(qb, kb, vb).transpose(1, 2))
        x = torch.cat(outs, dim=1).reshape(B, -1, C)
        return self.proj(x)

    def forward(self, x, cond, kv_seqlen, num_cond_latents=None, shape=None):
        if num_cond_latents is None or num_cond_latents == 0:
            return self._process_cross_attn(x, cond, kv_seqlen)
        B, N, C = x.shape
        n_cond = num_cond_latents * (N // shape[0])
        out_noise = self._process_cross_attn(x[:, n_cond:], cond, kv_seqlen)
        return torch.cat([torch.zeros((B, n_cond, C), dtype=out_noise.dtype, device=out_noise.device), out_noise], 1)


class SingleStreamAttention(nn.Module):
    """Per-frame audio cross-attention: the H*W tokens of each latent frame attend to that frame's audio tokens."""

    def __init__(self, dim: int, encoder_hidden_states_dim: int, num_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_linear = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=eps)
        self.proj = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=True)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=eps)

    def _process_cross_attn(self, x, cond, frames_num):
        out_dtype = x.dtype
        x = rearrange(x, "B (N_t S) C -> (B N_t) S C", N_t=frames_num)
        B, N, C = x.shape
        q = self.q_norm(self.q_linear(x).view(B, N, self.num_heads, self.head_dim).permute((0, 2, 1, 3)))
        N_a = cond.shape[1]
        encoder_k, encoder_v = (
            self.kv_linear(cond).view(B, N_a, 2, self.num_heads, self.head_dim).permute((2, 0, 3, 1, 4)).unbind(0)
        )
        encoder_k = self.k_norm(encoder_k)
        x = _attn(q, encoder_k, encoder_v)
        x = self.proj(x.transpose(1, 2).reshape(B, N, C))
        return rearrange(x, "(B N_t) S C -> B (N_t S) C", N_t=frames_num).type(out_dtype)

    def forward(self, x, cond, shape, num_cond_latents=None):
        B, N, C = x.shape
        if num_cond_latents is None or num_cond_latents == 0:
            return None, self._process_cross_attn(x, cond, shape[0])
        n_cond = num_cond_latents * (N // shape[0])
        cond = rearrange(cond, "(B N_t) M C -> B N_t M C", B=B)[:, num_cond_latents:]
        cond = rearrange(cond, "B N_t M C -> (B N_t) M C")
        out_noise = self._process_cross_attn(x[:, n_cond:], cond, shape[0] - num_cond_latents)
        out_cond = torch.zeros((B, n_cond, C), dtype=out_noise.dtype, device=out_noise.device)
        return out_cond, out_noise


class AudioProjModel(nn.Module):
    def __init__(
        self, seq_len=5, seq_len_vf=12, blocks=12, channels=768, intermediate_dim=512, output_dim=768, context_tokens=32
    ):
        super().__init__()
        self.context_tokens = context_tokens
        self.output_dim = output_dim
        self.proj1 = nn.Linear(seq_len * blocks * channels, intermediate_dim)
        self.proj1_vf = nn.Linear(seq_len_vf * blocks * channels, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_embeds, audio_embeds_vf):
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B = audio_embeds.shape[0]
        a = rearrange(audio_embeds, "bz f w b c -> (bz f) (w b c)")
        a_vf = rearrange(audio_embeds_vf, "bz f w b c -> (bz f) (w b c)")
        a = rearrange(torch.relu(self.proj1(a)), "(bz f) c -> bz f c", bz=B)
        a_vf = rearrange(torch.relu(self.proj1_vf(a_vf)), "(bz f) c -> bz f c", bz=B)
        c = torch.concat([a, a_vf], dim=1)
        c = c.view(-1, c.shape[-1])
        c = torch.relu(self.proj2(c))
        tokens = self.proj3(c).reshape(-1, self.context_tokens, self.output_dim)
        tokens = self.norm(tokens)
        return rearrange(tokens, "(bz f) m c -> bz f m c", f=video_length)


def audio_tokens(audio_proj, audio_cond, N_t, vae_scale, audio_window, num_ref_latents=None):
    """Per-latent-frame audio context tokens: the first video frame keeps its own window, each later latent frame
    pools the windows of its ``vae_scale`` video frames. Returns [B*T, context_tokens, output_dim]."""
    first = audio_cond[:, :1]
    latter = rearrange(audio_cond[:, 1:], "b (n_t n) w s c -> b n_t n w s c", n=vae_scale)
    mid = audio_window // 2
    latter_first = rearrange(latter[:, :, :1, : mid + 1], "b n_t n w s c -> b n_t (n w) s c")
    latter_last = rearrange(latter[:, :, -1:, mid:], "b n_t n w s c -> b n_t (n w) s c")
    latter_middle = rearrange(latter[:, :, 1:-1, mid : mid + 1], "b n_t n w s c -> b n_t (n w) s c")
    latter = torch.concat([latter_first, latter_middle, latter_last], dim=2)
    audio_hidden_states = audio_proj(first, latter)
    if num_ref_latents is not None and num_ref_latents > 0:
        audio_hidden_states = torch.cat([audio_hidden_states[:, [0]], audio_hidden_states], dim=1)
    audio_hidden_states = audio_hidden_states[:, -N_t:]
    return rearrange(audio_hidden_states, "b t n c -> (b t) n c")


# ---------------------------------------------------------------------------------------------------------------------
# Block and model (longcat_video/modules/avatar/longcat_video_dit_avatar.py)
# ---------------------------------------------------------------------------------------------------------------------


class LongCatAvatarSingleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, adaln_tembed_dim, output_dim=768):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True))
        self.audio_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 3 * hidden_size, bias=True))
        self.mod_norm_attn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)
        self.pre_video_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)
        self.attn = Attention(dim=hidden_size, num_heads=num_heads)
        self.cross_attn = MultiHeadCrossAttention(dim=hidden_size, num_heads=num_heads)
        self.audio_cross_attn = SingleStreamAttention(
            dim=hidden_size, encoder_hidden_states_dim=output_dim, num_heads=num_heads
        )
        self.ffn = FeedForwardSwiGLU(dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio))

    def forward(
        self,
        x,
        y,
        t,
        y_seqlen,
        latent_shape,
        num_cond_latents=None,
        num_ref_latents=None,
        audio_hidden_states=None,
        ref_img_index=None,
        mask_frame_range=None,
    ):
        """x: [B, N, C]; y: [1, N_valid_tokens, C]; t: [B, T, C_t] (fp32); audio_hidden_states: [B*T, 32, 768]."""
        x_dtype = x.dtype
        B, N, C = x.shape
        T = latent_shape[0]
        n_cond = num_cond_latents or 0

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(t.float()).unsqueeze(2).chunk(6, dim=-1)
        )

        x_m = modulate_fp32(self.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa).view(B, N, C)
        x_s = self.attn(x_m, latent_shape, num_cond_latents, num_ref_latents, ref_img_index, mask_frame_range)
        x = (x + (gate_msa * x_s.view(B, -1, N // T, C)).view(B, -1, C)).to(x_dtype)

        x = x + self.cross_attn(self.pre_crs_attn_norm(x), y, y_seqlen, num_cond_latents=n_cond, shape=latent_shape)

        audio_shift, audio_scale, audio_gate = (
            self.audio_adaLN_modulation(t[:, n_cond:].float()).unsqueeze(2).chunk(3, -1)
        )
        out_cond, out_noise = self.audio_cross_attn(
            self.pre_video_crs_attn_norm(x), audio_hidden_states, shape=latent_shape, num_cond_latents=n_cond
        )
        out_noise = modulate_fp32(self.mod_norm_attn, out_noise.view(B, T - n_cond, -1, C), audio_shift, audio_scale)
        audio_add_x = (audio_gate * out_noise.view(B, T - n_cond, -1, C)).view(B, -1, C)
        if out_cond is not None:
            audio_add_x = torch.cat([out_cond, audio_add_x], dim=1)
        x = (x + audio_add_x).to(x_dtype)

        x_m = modulate_fp32(self.mod_norm_ffn, x.view(B, -1, N // T, C), shift_mlp, scale_mlp).view(B, -1, C)
        x_s = self.ffn(x_m)
        x = (x + (gate_mlp * x_s.view(B, -1, N // T, C)).view(B, -1, C)).to(x_dtype)
        return x


class LongCatVideoAvatarTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 4096,
        depth: int = 48,
        num_heads: int = 32,
        caption_channels: int = 4096,
        mlp_ratio: int = 4,
        adaln_tembed_dim: int = 512,
        frequency_embedding_size: int = 256,
        patch_size=(1, 2, 2),
        text_tokens_zero_pad: bool = True,
        audio_window: int = 5,
        audio_block: int = 5,
        audio_channel: int = 1280,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 32,
        vae_scale: int = 4,
        **_unused,
    ) -> None:
        super().__init__()
        self.patch_size = tuple(patch_size)
        self.out_channels = out_channels
        self.vae_scale = vae_scale
        self.audio_window = audio_window
        self.text_tokens_zero_pad = text_tokens_zero_pad
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed3D(self.patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(adaln_tembed_dim, frequency_embedding_size)
        self.y_embedder = CaptionEmbedder(caption_channels, hidden_size)
        self.blocks = nn.ModuleList(
            [
                LongCatAvatarSingleStreamBlock(hidden_size, num_heads, mlp_ratio, adaln_tembed_dim, output_dim)
                for _ in range(depth)
            ]
        )
        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=audio_window + vae_scale - 1,
            blocks=audio_block,
            channels=audio_channel,
            intermediate_dim=intermediate_dim,
            output_dim=output_dim,
            context_tokens=context_tokens,
        )
        self.final_layer = FinalLayer_FP32(hidden_size, int(np.prod(self.patch_size)), out_channels, adaln_tembed_dim)

    def fp32_modules(self) -> list[nn.Module]:
        """Modules upstream runs under fp32 autocast; kept in fp32 here since CPU has no CUDA autocast."""
        mods = [self.t_embedder, self.final_layer]
        for block in self.blocks:
            mods += [block.adaLN_modulation, block.audio_adaLN_modulation]
        return mods

    def embed_timestep(self, timestep, N_t):
        """timestep: [B] or [B, T] -> [B, T, C_t] fp32."""
        B = timestep.shape[0]
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t)
        dtype = self.x_embedder.proj.weight.dtype
        timestep = timestep.to(dtype)
        return self.t_embedder(timestep.float().flatten(), dtype=torch.float32).reshape(B, N_t, -1)

    def embed_audio(self, audio_embs, N_t, num_ref_latents=None):
        """audio_embs: [B, num_frames, window, blocks, channels] -> [B*T, context_tokens, output_dim]."""
        audio_cond = audio_embs.to(self.x_embedder.proj.weight.dtype)
        return audio_tokens(self.audio_proj, audio_cond, N_t, self.vae_scale, self.audio_window, num_ref_latents)

    def embed_text(self, encoder_hidden_states, encoder_attention_mask=None):
        """[B, 1, L, C_text] -> ([1, sum(valid), C], y_seqlens)."""
        y = self.y_embedder(encoder_hidden_states.to(self.x_embedder.proj.weight.dtype))
        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            y = y * encoder_attention_mask[:, None, :, None]
            encoder_attention_mask = (encoder_attention_mask * 0 + 1).to(encoder_attention_mask.dtype)
        if encoder_attention_mask is not None:
            mask = encoder_attention_mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            return y, mask.sum(dim=1).tolist()
        seqlens = [y.shape[2]] * y.shape[0]
        return y.squeeze(1).view(1, -1, self.hidden_size), seqlens

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        num_cond_latents=0,
        audio_embs=None,
        num_ref_latents=None,
        ref_img_index=None,
        mask_frame_range=None,
    ):
        B, _, T, H, W = hidden_states.shape
        N_t, N_h, N_w = T // self.patch_size[0], H // self.patch_size[1], W // self.patch_size[2]

        dtype = self.x_embedder.proj.weight.dtype
        x = self.x_embedder(hidden_states.to(dtype))
        t = self.embed_timestep(timestep, N_t)
        y, y_seqlens = self.embed_text(encoder_hidden_states, encoder_attention_mask)
        audio_hidden_states = self.embed_audio(audio_embs, N_t, num_ref_latents)

        for block in self.blocks:
            x = block(
                x,
                y,
                t,
                y_seqlens,
                (N_t, N_h, N_w),
                num_cond_latents,
                num_ref_latents,
                audio_hidden_states,
                ref_img_index,
                mask_frame_range,
            )

        x = self.final_layer(x, t, (N_t, N_h, N_w))
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=self.patch_size[0],
            H_p=self.patch_size[1],
            W_p=self.patch_size[2],
            C_out=self.out_channels,
        )
        return x.to(torch.float32)


# ---------------------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------------------


def load_config(checkpoint_dir: str, subfolder: str = "base_model") -> dict:
    with open(os.path.join(checkpoint_dir, subfolder, "config.json")) as f:
        return json.load(f)


def load_state_dict(
    checkpoint_dir: str,
    subfolder: str = "base_model",
    num_blocks: Optional[int] = None,
    lora_path: Optional[str] = None,
) -> dict[str, torch.Tensor]:
    """Loads the DiT shards, keeping only the first ``num_blocks`` blocks when given, with the LoRA at
    ``lora_path`` merged in."""
    folder = os.path.join(checkpoint_dir, subfolder)
    with open(os.path.join(folder, "diffusion_pytorch_model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]

    def keep(key):
        if num_blocks is None or not key.startswith("blocks."):
            return True
        return int(key.split(".")[1]) < num_blocks

    state = {}
    for shard in sorted({s for k, s in weight_map.items() if keep(k)}):
        for k, v in load_file(os.path.join(folder, shard)).items():
            if keep(k):
                state[k] = v
    if lora_path is not None:
        merge_lora_into_state_dict(state, lora_deltas(lora_path))
    return state


def lora_deltas(lora_path: str, multiplier: float = 1.0, network_alpha: float = 64.0) -> dict[str, torch.Tensor]:
    """Dense fp32 weight deltas ``{module.weight: multiplier * alpha/rank * up @ down}`` of a LongCat LoRA file.

    Block-split up projections (qkv: 3, kv: 2) concatenate their per-block products on the output dim,
    matching ``LongCatVideoAvatarTransformer3DModel.merge_lora`` upstream.
    """
    sd = load_file(lora_path)
    names = sorted({k[: -len(".lora_down.weight")] for k in sd if k.endswith(".lora_down.weight")})
    deltas = {}
    for name in names:
        module = name.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
        down = sd[f"{name}.lora_down.weight"].float()
        rank = down.shape[0]
        ups = sorted((k for k in sd if k.startswith(f"{name}.lora_up.blocks.")), key=lambda k: int(k.split(".")[-2]))
        if ups:
            down_chunks = torch.chunk(down, len(ups), dim=0)
            delta = torch.cat([sd[k].float() @ down_chunks[i] for i, k in enumerate(ups)], dim=0)
            rank = down_chunks[0].shape[0]
        else:
            delta = sd[f"{name}.lora_up.weight"].float() @ down
        deltas[f"{module}.weight"] = delta * (multiplier * network_alpha / rank)
    return deltas


def merge_lora_into_state_dict(state: dict[str, torch.Tensor], deltas: dict[str, torch.Tensor]) -> None:
    for key, delta in deltas.items():
        if key in state:
            w = state[key]
            state[key] = (w.float() + delta).to(w.dtype)


def build_model(
    checkpoint_dir: str,
    num_blocks: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
    lora_path: Optional[str] = None,
) -> LongCatVideoAvatarTransformer3DModel:
    config = load_config(checkpoint_dir)
    if num_blocks is not None:
        config["depth"] = num_blocks
    model = LongCatVideoAvatarTransformer3DModel(**config)
    state = load_state_dict(checkpoint_dir, num_blocks=num_blocks, lora_path=lora_path)
    model.load_state_dict(state, strict=True)
    model = model.to(dtype).eval()
    for module in model.fp32_modules():
        module.float()
    return model
