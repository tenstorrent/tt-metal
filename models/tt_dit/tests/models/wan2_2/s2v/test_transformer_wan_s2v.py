# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parity tests for ``WanS2VTransformer3DModel`` (block / model / inner_step)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from loguru import logger

import ttnn

from .....models.transformers.wan2_2.s2v.motioner import rope_params
from .....models.transformers.wan2_2.s2v.rope_s2v import rope_precompute
from .....models.transformers.wan2_2.s2v.transformer_wan_s2v import WanS2VTransformer3DModel
from .....parallel.config import DiTParallelConfig, ParallelFactor
from .....parallel.manager import CCLManager
from .....utils.check import assert_quality
from .....utils.mochi import get_rot_transformation_mat
from .....utils.padding import get_padded_vision_seq_len, pad_vision_seq_parallel
from .....utils.tensor import bf16_tensor, bf16_tensor_2dshard, float32_tensor, from_torch, local_device_to_torch
from .....utils.test import line_params, ring_params
from .....utils.wan_s2v_checkpoint import (
    find_s2v_snapshot,
    load_s2v_config,
    load_s2v_state_dict,
    translate_s2v_state_dict,
)

# Production model config (Wan-AI/Wan2.2-S2V-14B / config.json).
PATCH_SIZE = (1, 2, 2)
DIM = 5120
NUM_HEADS = 40
HEAD_DIM = DIM // NUM_HEADS  # 128
IN_CHANNELS = 16
OUT_CHANNELS = 16
TEXT_DIM = 4096
FREQ_DIM = 256
FFN_DIM = 13824
AUDIO_DIM = 1024
NUM_AUDIO_TOKEN = 4
NUM_AUDIO_LAYERS = 25
AUDIO_INJECT_LAYERS = (0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39)
ROPE_MAX_SEQ_LEN = 1024
EPS = 1e-6


# Torch reference shared by the block + model tests. Inlined ports of upstream
# wan/modules/model.py + wan/modules/s2v/model_s2v.py with CUDA / distributed /
# flash_attention deps replaced by plain torch ops. Class names + state_dict
# keys match upstream so ``translate_s2v_state_dict`` round-trips cleanly.


class TorchWanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class TorchWanLayerNorm(nn.LayerNorm):
    """Always computes in fp32 and casts back."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False) -> None:
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


def torch_rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to ``x`` for the noisy [F, H, W] grid; tokens past F*H*W pass through."""
    n, c = x.size(2), x.size(3) // 2
    freqs_split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs_split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs_split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs_split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


def torch_rope_apply_per_token(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply precomputed per-token rope freqs (complex). Matches upstream's
    ``wan/modules/s2v/model_s2v.py:rope_apply`` — freqs is the complex per-token
    table from ``rope_precompute``, shape ``[B, N, num_heads_or_1, head_dim/2]``.
    """
    output = []
    for i in range(x.size(0)):
        x_i = torch.view_as_complex(x[i].to(torch.float64).reshape(*x[i].shape[:-1], -1, 2))
        f_i = freqs[i]
        x_i = torch.view_as_real(x_i * f_i).flatten(2)
        output.append(x_i)
    return torch.stack(output).float()


def torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q/k/v shape ``[B, L, H, D]``; returns ``[B, L, H, D]``."""
    qh = q.transpose(1, 2)
    kh = k.transpose(1, 2)
    vh = v.transpose(1, 2)
    out = F.scaled_dot_product_attention(qh, kh, vh)
    return out.transpose(1, 2)


class TorchWanSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = TorchWanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = TorchWanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        # If freqs is precomputed per-token (complex 4D), apply directly; else fall
        # back to grid-derived 2D-table rope (block test path).
        if freqs.is_complex() and freqs.ndim >= 3:
            q = torch_rope_apply_per_token(q, freqs)
            k = torch_rope_apply_per_token(k, freqs)
        else:
            q = torch_rope_apply(q, grid_sizes, freqs)
            k = torch_rope_apply(k, grid_sizes, freqs)
        out = torch_attention(q, k, v)
        return self.o(out.flatten(2))


class TorchWanCrossAttention(TorchWanSelfAttention):
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        b = x.size(0)
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        out = torch_attention(q, k, v)
        return self.o(out.flatten(2))


class TorchWanS2VAttentionBlock(nn.Module):
    """Port of upstream ``WanS2VAttentionBlock`` with segmented timestep modulation."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.norm1 = TorchWanLayerNorm(dim, eps)
        self.self_attn = TorchWanSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = TorchWanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = TorchWanCrossAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = TorchWanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: tuple[torch.Tensor, int],
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        e_full, seg_idx_raw = e
        seg_idx = min(max(0, int(seg_idx_raw)), x.size(1))
        seg_idx = [0, seg_idx, x.size(1)]
        e_mod = (self.modulation.unsqueeze(2) + e_full.float()).chunk(6, dim=1)
        e_mod = [chunk.squeeze(1) for chunk in e_mod]

        norm_x = self.norm1(x).float()
        parts = [
            norm_x[:, seg_idx[i] : seg_idx[i + 1]] * (1 + e_mod[1][:, i : i + 1]) + e_mod[0][:, i : i + 1]
            for i in range(2)
        ]
        norm_x = torch.cat(parts, dim=1)
        y = self.self_attn(norm_x, grid_sizes, freqs)
        z = [y[:, seg_idx[i] : seg_idx[i + 1]] * e_mod[2][:, i : i + 1] for i in range(2)]
        x = x + torch.cat(z, dim=1)

        x = x + self.cross_attn(self.norm3(x), context)

        norm2_x = self.norm2(x).float()
        parts = [
            norm2_x[:, seg_idx[i] : seg_idx[i + 1]] * (1 + e_mod[4][:, i : i + 1]) + e_mod[3][:, i : i + 1]
            for i in range(2)
        ]
        norm2_x = torch.cat(parts, dim=1)
        y = self.ffn(norm2_x)
        z = [y[:, seg_idx[i] : seg_idx[i + 1]] * e_mod[5][:, i : i + 1] for i in range(2)]
        x = x + torch.cat(z, dim=1)
        return x


def torch_sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Vendored from wan/modules/motioner.py."""
    half = dim // 2
    position = position.type(torch.float32)
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half, dtype=torch.float32) / half))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)


class TorchCausalConv1d(nn.Module):
    """nn.Conv1d with causal (left-replicate) padding. Matches wan/modules/s2v/auxi_blocks.py:CausalConv1d."""

    def __init__(self, chan_in: int, chan_out: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]; causal-pad on T axis.
        pad = self.kernel_size - 1
        if pad > 0:
            x = F.pad(x, (pad, 0), mode="replicate")
        return self.conv(x)


class TorchMotionEncoder_tc(nn.Module):
    """Match upstream wan/modules/s2v/auxi_blocks.py:MotionEncoder_tc."""

    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int = 4, need_global: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.need_global = need_global
        H4 = hidden_dim // 4
        self.conv1_local = TorchCausalConv1d(in_dim, H4 * num_heads, kernel_size=3, stride=1)
        if need_global:
            self.conv1_global = TorchCausalConv1d(in_dim, H4, kernel_size=3, stride=1)
            self.final_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(H4, elementwise_affine=False, eps=1e-6)
        self.act = nn.SiLU()
        self.conv2 = TorchCausalConv1d(H4, hidden_dim // 2, kernel_size=3, stride=2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6)
        self.conv3 = TorchCausalConv1d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def _stage(self, x: torch.Tensor, conv: TorchCausalConv1d, norm: nn.LayerNorm) -> torch.Tensor:
        # x: [B*, T, C] → conv on [B*, C, T] → norm + silu
        x = conv(x.transpose(1, 2)).transpose(1, 2)
        return self.act(norm(x))

    def forward(self, x_BTC: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = x_BTC.shape
        H = self.hidden_dim
        H4 = H // 4
        x_in = x_BTC

        # Local branch.
        x = self.conv1_local(x_in.transpose(1, 2)).transpose(1, 2)  # [B, T, H4*num_heads]
        x = x.view(B, T, self.num_heads, H4).permute(0, 2, 1, 3).reshape(B * self.num_heads, T, H4)
        x = self.act(self.norm1(x))
        x = self._stage(x, self.conv2, self.norm2)
        x = self._stage(x, self.conv3, self.norm3)
        T4 = x.shape[1]
        local = x.view(B, self.num_heads, T4, H).permute(0, 2, 1, 3)  # [B, T4, num_heads, H]
        pad = self.padding_tokens.expand(B, T4, 1, H)
        local = torch.cat([local, pad], dim=2)  # [B, T4, num_heads+1, H]

        if not self.need_global:
            return local

        # Global branch.
        x = self.conv1_global(x_in.transpose(1, 2)).transpose(1, 2)  # [B, T, H4]
        x = self.act(self.norm1(x))
        x = self._stage(x, self.conv2, self.norm2)
        x = self._stage(x, self.conv3, self.norm3)
        x = self.final_linear(x)  # [B, T4, H]
        glob = x.unsqueeze(2)  # [B, T4, 1, H]
        return glob, local


class TorchCausalAudioEncoder(nn.Module):
    """Weighted-sum of wav2vec2 hidden states + TorchMotionEncoder_tc.
    Matches wan/modules/s2v/audio_utils.py:CausalAudioEncoder.
    """

    def __init__(
        self, dim: int, num_layers: int = 25, out_dim: int = 5120, num_token: int = 4, need_global: bool = False
    ) -> None:
        super().__init__()
        self.encoder = TorchMotionEncoder_tc(
            in_dim=dim, hidden_dim=out_dim, num_heads=num_token, need_global=need_global
        )
        self.weights = nn.Parameter(torch.ones(1, num_layers, 1, 1) * 0.01)
        self.act = nn.SiLU()

    def forward(self, features_BLDT: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # features: [B, num_layers, dim, T]
        w = self.act(self.weights)
        w_sum = w.sum(dim=1, keepdim=True)
        weighted = ((features_BLDT.float() * w) / w_sum).sum(dim=1)  # [B, dim, T]
        weighted = weighted.permute(0, 2, 1)  # [B, T, dim]
        return self.encoder(weighted)


class TorchAdaLayerNorm(nn.Module):
    """Diffusers ``AdaLayerNorm(output_dim=2*dim, embedding_dim, chunk_dim=1)``.

    silu + linear + chunk(2, dim=1) → (shift, scale), then
    ``norm(x) * (1 + scale) + shift`` broadcasting along the sequence dim.
    """

    def __init__(self, output_dim: int, embedding_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        assert output_dim % 2 == 0
        self.norm = nn.LayerNorm(output_dim // 2, elementwise_affine=False, eps=eps)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim, bias=True)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        proj = self.linear(self.silu(temb))
        shift, scale = proj.chunk(2, dim=1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TorchAudioInjector_WAN(nn.Module):
    """Holds the cross-attn slots + optional AdaIN projections. Matches
    wan/modules/s2v/audio_utils.py:AudioInjector_WAN (subset; no need_adain_ont)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        n_inject: int,
        enable_adain: bool,
        adain_dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.injector = nn.ModuleList(
            [TorchWanCrossAttention(dim, num_heads, qk_norm=True, eps=eps) for _ in range(n_inject)]
        )
        # Reference uses no-affine LayerNorm; included for state_dict-key parity.
        self.injector_pre_norm_feat = nn.ModuleList(
            [nn.LayerNorm(dim, elementwise_affine=False, eps=eps) for _ in range(n_inject)]
        )
        self.injector_pre_norm_vec = nn.ModuleList(
            [nn.LayerNorm(dim, elementwise_affine=False, eps=eps) for _ in range(n_inject)]
        )
        if enable_adain:
            self.injector_adain_layers = nn.ModuleList(
                [TorchAdaLayerNorm(output_dim=dim * 2, embedding_dim=adain_dim) for _ in range(n_inject)]
            )
        else:
            self.injector_adain_layers = nn.ModuleList()


class TorchHead_S2V(nn.Module):
    """Port of upstream Head_S2V (Head + chunk-2 modulation)."""

    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        full_out = math.prod(patch_size) * out_dim
        self.norm = TorchWanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, full_out)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        chunks = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + chunks[1]) + chunks[0])


class TorchWanS2VModel(nn.Module):
    """Inline port of wan/modules/s2v/model_s2v.py:WanModel_S2V.

    Audio path supported (CausalAudioEncoder + AudioInjector). Motioner /
    FramePacker NOT included — tests should pass ``drop_first_motion=True``
    or provide an empty motion_latents path.
    """

    def __init__(
        self,
        *,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_dim: int = 16,
        out_dim: int = 16,
        dim: int = 5120,
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        text_dim: int = 4096,
        text_len: int = 512,
        num_heads: int = 40,
        num_layers: int = 40,
        audio_dim: int = 1024,
        num_audio_token: int = 4,
        audio_inject_layers: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39),
        enable_adain: bool = True,
        cond_dim: int = 16,
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0 and (dim // num_heads) % 2 == 0
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.num_heads = num_heads
        self.text_len = text_len
        self.freq_dim = freq_dim
        self.enable_adain = enable_adain
        self.audio_inject_layers = tuple(audio_inject_layers)
        self.injected_block_id = {layer: i for i, layer in enumerate(self.audio_inject_layers)}

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                TorchWanS2VAttentionBlock(dim, ffn_dim, num_heads, qk_norm=True, cross_attn_norm=True, eps=eps)
                for _ in range(num_layers)
            ]
        )
        self.head = TorchHead_S2V(dim, out_dim, patch_size, eps)
        self.cond_encoder = nn.Conv3d(cond_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.trainable_cond_mask = nn.Embedding(3, dim)
        self.casual_audio_encoder = TorchCausalAudioEncoder(
            dim=audio_dim, num_layers=25, out_dim=dim, num_token=num_audio_token, need_global=enable_adain
        )
        self.audio_injector = TorchAudioInjector_WAN(
            dim=dim,
            num_heads=num_heads,
            n_inject=len(self.audio_inject_layers),
            enable_adain=enable_adain,
            adain_dim=dim,
            eps=eps,
        )
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(rope_max_seq_len, d - 4 * (d // 6)),
                rope_params(rope_max_seq_len, 2 * (d // 6)),
                rope_params(rope_max_seq_len, 2 * (d // 6)),
            ],
            dim=1,
        )

    def after_transformer_block(
        self,
        block_idx: int,
        hidden_states: torch.Tensor,
        merged_audio_emb: torch.Tensor,
        audio_emb_global: torch.Tensor | None,
        original_seq_len: int,
    ) -> torch.Tensor:
        """Faithful port of upstream WanModel_S2V.after_transformer_block.

        Per-frame rearrange + diffusers AdaLayerNorm + cross-attention into
        each spatial frame's own audio tokens. No global mask — the per-frame
        batching enforces the block-diagonal structure naturally.
        """
        if block_idx not in self.injected_block_id:
            return hidden_states
        aid = self.injected_block_id[block_idx]
        num_frames = merged_audio_emb.shape[1]  # T_video
        # input_hidden_states: noisy tokens reshaped to per-frame batches.
        input_hidden_states = hidden_states[:, :original_seq_len].clone()
        # "b (t n) c -> (b t) n c"
        B, _, C = input_hidden_states.shape
        n_per_frame = original_seq_len // num_frames
        input_hidden_states = input_hidden_states.view(B, num_frames, n_per_frame, C).reshape(
            B * num_frames, n_per_frame, C
        )

        if self.enable_adain and audio_emb_global is not None:
            # audio_emb_global: [B, T, 1, C] → "(b t) n c" → take token 0: [(B*T), C].
            T_g, K_g = audio_emb_global.shape[1], audio_emb_global.shape[2]
            audio_emb_global_btnc = audio_emb_global.reshape(B * T_g, K_g, C)
            attn_hidden_states = self.audio_injector.injector_adain_layers[aid](
                input_hidden_states, temb=audio_emb_global_btnc[:, 0]
            )
        else:
            attn_hidden_states = self.audio_injector.injector_pre_norm_feat[aid](input_hidden_states)

        # audio_emb: [B, T, n_tok, C] → "(b t) n c" → [(B*T), n_tok, C].
        n_tok = merged_audio_emb.shape[2]
        attn_audio_emb = merged_audio_emb.reshape(B * num_frames, n_tok, C)
        residual_out = self.audio_injector.injector[aid](attn_hidden_states, attn_audio_emb)
        # "(b t) n c -> b (t n) c"
        residual_out = residual_out.reshape(B, num_frames * n_per_frame, C)
        hidden_states = hidden_states.clone()
        hidden_states[:, :original_seq_len] = hidden_states[:, :original_seq_len] + residual_out
        return hidden_states

    def forward(
        self,
        noisy_BCFHW: torch.Tensor,
        timestep_B: torch.Tensor,
        prompt_BLP: torch.Tensor,
        ref_BCFHW: torch.Tensor,
        audio_input_BLCT: torch.Tensor,
        cond_BCFHW: torch.Tensor | None = None,
        motion_frames: tuple[int, int] = (17, 5),
    ) -> torch.Tensor:
        """Single-batch forward. Returns ``[B, out_dim, F, H, W]``.

        Matches the no-motion / drop_motion_frames=True clip-0 path.
        """
        B = noisy_BCFHW.shape[0]
        # Audio encoder. Pre-pad audio with motion_frames[0] repetitions of first frame.
        first = audio_input_BLCT[..., 0:1].repeat(1, 1, 1, motion_frames[0])
        audio_padded = torch.cat([first, audio_input_BLCT], dim=-1)
        audio_emb_res = self.casual_audio_encoder(audio_padded)
        if self.enable_adain:
            audio_emb_global, audio_emb = audio_emb_res
            audio_emb_global = audio_emb_global[:, motion_frames[1] :]
        else:
            audio_emb = audio_emb_res
            audio_emb_global = None
        merged_audio_emb = audio_emb[:, motion_frames[1] :, :]  # [B, T_video, num_token+1, dim]
        T_video = merged_audio_emb.shape[1]

        # Patch noisy + (optional pose) + ref.
        x_p = self.patch_embedding(noisy_BCFHW)
        if cond_BCFHW is not None:
            x_p = x_p + self.cond_encoder(cond_BCFHW)
        F_p, H_p, W_p = x_p.shape[2:]
        original_grid = torch.tensor([[F_p, H_p, W_p]], dtype=torch.long)
        ref_p = self.patch_embedding(ref_BCFHW)
        ref_F, ref_H, ref_W = ref_p.shape[2:]

        x_noisy = x_p.flatten(2).transpose(1, 2)  # [B, N_noisy, dim]
        x_ref = ref_p.flatten(2).transpose(1, 2)  # [B, N_ref, dim]
        n_noisy = x_noisy.shape[1]

        # Build segmented grid_sizes for rope_precompute.
        noisy_grid = [
            torch.zeros(B, 3, dtype=torch.long),
            torch.tensor([[F_p, H_p, W_p]] * B, dtype=torch.long),
            torch.tensor([[F_p, H_p, W_p]] * B, dtype=torch.long),
        ]
        ref_grid = [
            torch.tensor([[30, 0, 0]] * B, dtype=torch.long),
            torch.tensor([[31, ref_H, ref_W]] * B, dtype=torch.long),
            torch.tensor([[1, ref_H, ref_W]] * B, dtype=torch.long),
        ]
        x = torch.cat([x_noisy, x_ref], dim=1)
        N_total = x.shape[1]
        placeholder = torch.zeros(B, N_total, self.num_heads, self.dim // self.num_heads, dtype=torch.float32)
        freqs_per_token = rope_precompute(placeholder, [noisy_grid, ref_grid], self.freqs, start=None)

        # Mask: 0=noisy, 1=ref. (drop_first_motion=True so no motion=2 tokens here.)
        mask = torch.zeros(B, N_total, dtype=torch.long)
        mask[:, n_noisy:] = 1
        x = x + self.trainable_cond_mask(mask).to(x.dtype)

        # Time embedding + 6-way projection.
        e = self.time_embedding(torch_sinusoidal_embedding_1d(self.freq_dim, timestep_B.float()))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        # Replicate across the two segments (noisy / const).
        e0 = e0.unsqueeze(2).repeat(1, 1, 2, 1)
        e_packed = [e0, n_noisy]

        # Text embedding (pad to text_len).
        pad_len = self.text_len - prompt_BLP.shape[1]
        if pad_len > 0:
            prompt_BLP = torch.cat([prompt_BLP, prompt_BLP.new_zeros(B, pad_len, prompt_BLP.shape[2])], dim=1)
        context = self.text_embedding(prompt_BLP)

        # Use precomputed per-token freqs (matches upstream — applies rope to
        # noisy AND ref tokens with their respective grid positions).
        grid_sizes_block = torch.tensor([[F_p, H_p, W_p]] * B, dtype=torch.long)
        for block_idx, block in enumerate(self.blocks):
            x = block(x, e_packed, grid_sizes_block, freqs_per_token, context)
            x = self.after_transformer_block(block_idx, x, merged_audio_emb, audio_emb_global, n_noisy)

        # Keep noisy tokens, head, unpatchify.
        x = x[:, :n_noisy]
        x = self.head(x, e)
        p_t, p_h, p_w = self.patch_size
        x = x.view(B, F_p, H_p, W_p, p_t, p_h, p_w, self.out_dim)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return x.reshape(B, self.out_dim, F_p * p_t, H_p * p_h, W_p * p_w)


BLOCK_PROMPT_SEQ_LEN = 32
BLOCK_PCC = 0.99


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((2, 4), (2, 4), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="bh_2x4sp1tp0"),
        pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("BLOCK_T", "BLOCK_H", "BLOCK_W"),
    [
        pytest.param(20, 30, 52, id="480p"),  # patched (T, H/2, W/2) at 480p latent (60, 104)
        pytest.param(20, 45, 80, id="720p"),  # patched at 720p latent (90, 160)
    ],
)
def test_wan_s2v_transformer_block(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    BLOCK_T: int,
    BLOCK_H: int,
    BLOCK_W: int,
    reset_seeds,
) -> None:
    """Single-block parity through ``_s2v_segmented_block_forward`` (two-segment modulation)."""
    torch.manual_seed(0)
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    sp_factor = parallel_config.sequence_parallel.factor

    # ---- Build torch reference + matching TT model (num_layers=1) ----
    ref_block = (
        TorchWanS2VAttentionBlock(
            dim=DIM,
            ffn_dim=FFN_DIM,
            num_heads=NUM_HEADS,
            qk_norm=True,
            cross_attn_norm=True,
            eps=EPS,
        )
        .eval()
        .to(torch.float32)
    )

    # Construct full WanS2VTransformer3DModel with num_layers=1 (one block) plus
    # the S2V-specific bits (frame_packer, audio_injector, etc.). Audio injection
    # is disabled by passing audio_inject_layers=() so after_transformer_block is
    # a no-op.
    tt_model = WanS2VTransformer3DModel(
        patch_size=PATCH_SIZE,
        num_heads=NUM_HEADS,
        dim=DIM,
        in_channels=16,
        out_channels=16,
        text_dim=512,
        freq_dim=64,
        ffn_dim=FFN_DIM,
        num_layers=1,
        cross_attn_norm=True,
        eps=EPS,
        rope_max_seq_len=1024,
        audio_dim=128,
        num_audio_layers=3,
        num_audio_token=4,
        audio_inject_layers=(),
        enable_adain=False,
        enable_motioner=False,
        enable_framepack=True,
        cond_dim=16,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        model_type="s2v",
    )

    # Copy reference block weights into the TT block. The base block uses
    # Diffusers-style keys (attn1/attn2 instead of self_attn/cross_attn,
    # to_q/to_k/to_v/to_out instead of q/k/v/o, etc.). norm1/norm3 are no-affine
    # and contribute no state-dict entries — only norm2 (cross-attn pre-norm)
    # has affine params, and reference's norm3 → tt norm2.
    ref_sd = ref_block.state_dict()
    block_sd: dict[str, torch.Tensor] = {}
    block_sd["norm2.weight"] = ref_sd["norm3.weight"]
    block_sd["norm2.bias"] = ref_sd["norm3.bias"]
    for src_prefix, dst_prefix in [("self_attn", "attn1"), ("cross_attn", "attn2")]:
        block_sd[f"{dst_prefix}.to_q.weight"] = ref_sd[f"{src_prefix}.q.weight"]
        block_sd[f"{dst_prefix}.to_q.bias"] = ref_sd[f"{src_prefix}.q.bias"]
        block_sd[f"{dst_prefix}.to_k.weight"] = ref_sd[f"{src_prefix}.k.weight"]
        block_sd[f"{dst_prefix}.to_k.bias"] = ref_sd[f"{src_prefix}.k.bias"]
        block_sd[f"{dst_prefix}.to_v.weight"] = ref_sd[f"{src_prefix}.v.weight"]
        block_sd[f"{dst_prefix}.to_v.bias"] = ref_sd[f"{src_prefix}.v.bias"]
        block_sd[f"{dst_prefix}.to_out.0.weight"] = ref_sd[f"{src_prefix}.o.weight"]
        block_sd[f"{dst_prefix}.to_out.0.bias"] = ref_sd[f"{src_prefix}.o.bias"]
        block_sd[f"{dst_prefix}.norm_q.weight"] = ref_sd[f"{src_prefix}.norm_q.weight"]
        block_sd[f"{dst_prefix}.norm_k.weight"] = ref_sd[f"{src_prefix}.norm_k.weight"]
    block_sd["ffn.ff1.weight"] = ref_sd["ffn.0.weight"]
    block_sd["ffn.ff1.bias"] = ref_sd["ffn.0.bias"]
    block_sd["ffn.ff2.weight"] = ref_sd["ffn.2.weight"]
    block_sd["ffn.ff2.bias"] = ref_sd["ffn.2.bias"]
    # ref.modulation is [1, 6, dim]; block._prepare_torch_state unsqueezes to
    # [1, 1, 6, dim] for the Parameter (which expects rank-4). Pass through.
    block_sd["scale_shift_table"] = ref_sd["modulation"]

    tt_block = tt_model.blocks[0]
    incompat = tt_block.load_torch_state_dict(block_sd, strict=False)
    logger.info(f"S2V block load: missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}")

    # ---- Build inputs ----
    B = 1
    n_noisy = BLOCK_T * BLOCK_H * BLOCK_W  # 256
    n_const = 64
    n_total = n_noisy + n_const

    spatial = torch.randn(B, n_total, DIM, dtype=torch.float32)
    prompt = torch.randn(B, BLOCK_PROMPT_SEQ_LEN, DIM, dtype=torch.float32)
    # e_full: [B, 6, 2, dim] — segment 0 = noisy, segment 1 = const.
    e_full = torch.randn(B, 6, 2, DIM, dtype=torch.float32) * 0.5

    # ---- Build rope features compatible with both paths ----
    # Single (F, H, W) grid for the noisy portion only. Tokens past n_noisy
    # pass through without rope (matches torch_rope_apply behavior).
    grid_sizes = torch.tensor([[BLOCK_T, BLOCK_H, BLOCK_W]], dtype=torch.long)
    d = HEAD_DIM
    freqs = torch.cat(
        [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
        dim=1,
    )

    # ---- Run torch reference ----
    with torch.no_grad():
        ref_out = ref_block(spatial.clone(), (e_full, n_noisy), grid_sizes, freqs, prompt)
    logger.info(f"torch reference output: {tuple(ref_out.shape)}")

    # ---- Build TT side state ----
    # Pad noisy + const segments separately (matches production layout).
    padded_n_noisy = get_padded_vision_seq_len(n_noisy, sp_factor)
    padded_n_const = get_padded_vision_seq_len(n_const, sp_factor)
    padded_n_total = padded_n_noisy + padded_n_const

    noisy_seg = spatial[:, :n_noisy].unsqueeze(0)
    const_seg = spatial[:, n_noisy:].unsqueeze(0)
    noisy_padded = pad_vision_seq_parallel(noisy_seg, num_devices=sp_factor)
    const_padded = pad_vision_seq_parallel(const_seg, num_devices=sp_factor)
    # Per-device interleaved layout [noisy_0 | const_0 | noisy_1 | const_1 | ...]
    sp_chunks = []
    pn_per_dev = padded_n_noisy // sp_factor
    pc_per_dev = padded_n_const // sp_factor
    for d_idx in range(sp_factor):
        sp_chunks.append(noisy_padded[:, :, d_idx * pn_per_dev : (d_idx + 1) * pn_per_dev, :])
        sp_chunks.append(const_padded[:, :, d_idx * pc_per_dev : (d_idx + 1) * pc_per_dev, :])
    spatial_interleaved = torch.cat(sp_chunks, dim=2)

    tt_spatial = bf16_tensor_2dshard(
        spatial_interleaved.contiguous(), device=mesh_device, shard_mapping={sp_axis: 2, tp_axis: 3}
    )
    tt_prompt = bf16_tensor(prompt.unsqueeze(0), device=mesh_device)

    # Rope cos/sin from same freqs — apply to noisy positions only (const get
    # identity rope = no-op, matching torch_rope_apply's pass-through path).
    c = HEAD_DIM // 2
    fs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    rope_complex_noisy = torch.zeros(B, n_noisy, 1, c, dtype=torch.complex128)
    for f in range(BLOCK_T):
        for h in range(BLOCK_H):
            for w in range(BLOCK_W):
                idx = f * BLOCK_H * BLOCK_W + h * BLOCK_W + w
                rope_complex_noisy[0, idx, 0, :] = torch.cat([fs[0][f], fs[1][h], fs[2][w]])
    rope_complex_full = torch.zeros(B, n_total, 1, c, dtype=torch.complex128)
    rope_complex_full[:, :n_noisy] = rope_complex_noisy
    rope_complex_full[:, n_noisy:] = 1.0 + 0.0j  # identity rope on const tokens

    cos_global = rope_complex_full.real.float().repeat_interleave(2, dim=-1).permute(0, 2, 1, 3)
    sin_global = rope_complex_full.imag.float().repeat_interleave(2, dim=-1).permute(0, 2, 1, 3)

    cos_noisy = pad_vision_seq_parallel(cos_global[:, :, :n_noisy, :], num_devices=sp_factor)
    cos_const = pad_vision_seq_parallel(cos_global[:, :, n_noisy:, :], num_devices=sp_factor)
    sin_noisy = pad_vision_seq_parallel(sin_global[:, :, :n_noisy, :], num_devices=sp_factor)
    sin_const = pad_vision_seq_parallel(sin_global[:, :, n_noisy:, :], num_devices=sp_factor)
    rope_upload_kwargs = dict(device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., sp_axis, None])
    rope_cos_tt = ttnn.concat(
        [
            from_torch(cos_noisy.contiguous(), **rope_upload_kwargs),
            from_torch(cos_const.contiguous(), **rope_upload_kwargs),
        ],
        dim=-2,
    )
    rope_sin_tt = ttnn.concat(
        [
            from_torch(sin_noisy.contiguous(), **rope_upload_kwargs),
            from_torch(sin_const.contiguous(), **rope_upload_kwargs),
        ],
        dim=-2,
    )
    trans_mat = bf16_tensor(get_rot_transformation_mat(), device=mesh_device)

    # ---- Build segmented modulation tensors (real-t for noisy, zero-t for const) ----
    timestep_proj_real_torch = e_full[:, :, 0].unsqueeze(0)  # [1, B, 6, dim]
    timestep_proj_zero_torch = e_full[:, :, 1].unsqueeze(0)
    tt_temb_real = from_torch(
        timestep_proj_real_torch.contiguous(), device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., tp_axis]
    )
    tt_temb_zero = from_torch(
        timestep_proj_zero_torch.contiguous(), device=mesh_device, dtype=ttnn.float32, mesh_axes=[..., tp_axis]
    )

    mask_n_noisy = torch.ones(1, 1, padded_n_noisy, 1, dtype=torch.bfloat16)
    mask_n_const = torch.zeros(1, 1, padded_n_const, 1, dtype=torch.bfloat16)
    mask_c_noisy = torch.zeros(1, 1, padded_n_noisy, 1, dtype=torch.bfloat16)
    mask_c_const = torch.ones(1, 1, padded_n_const, 1, dtype=torch.bfloat16)
    seg_upload = dict(device=mesh_device, mesh_axis=sp_axis, shard_dim=2, layout=ttnn.TILE_LAYOUT)
    tt_mask_noisy = ttnn.concat(
        [bf16_tensor(mask_n_noisy.contiguous(), **seg_upload), bf16_tensor(mask_n_const.contiguous(), **seg_upload)],
        dim=-2,
    )
    tt_mask_constant = ttnn.concat(
        [bf16_tensor(mask_c_noisy.contiguous(), **seg_upload), bf16_tensor(mask_c_const.contiguous(), **seg_upload)],
        dim=-2,
    )

    # ---- Run TT segmented block ----
    tt_out = tt_model._s2v_segmented_block_forward(
        tt_block,
        spatial_1BND=tt_spatial,
        prompt_1BLP=tt_prompt,
        N=padded_n_total,
        rope_cos=rope_cos_tt,
        rope_sin=rope_sin_tt,
        trans_mat=trans_mat,
        timestep_proj_real=tt_temb_real,
        timestep_proj_zero=tt_temb_zero,
        mask_noisy=tt_mask_noisy,
        mask_constant=tt_mask_constant,
    )

    # ---- Gather and de-interleave the TT output ----
    sp_gathered = ccl_manager.all_gather_persistent_buffer(tt_out, dim=2, mesh_axis=sp_axis)
    tp_gathered = ccl_manager.all_gather_persistent_buffer(sp_gathered, dim=3, mesh_axis=tp_axis)
    tt_full = local_device_to_torch(tp_gathered).squeeze(0).float()

    # After SP-gather the layout is [noisy_0 | const_0 | noisy_1 | const_1 | ...].
    per_dev_len = pn_per_dev + pc_per_dev
    noisy_chunks = []
    const_chunks = []
    for d_idx in range(sp_factor):
        chunk = tt_full[:, d_idx * per_dev_len : (d_idx + 1) * per_dev_len, :]
        noisy_chunks.append(chunk[:, :pn_per_dev, :])
        const_chunks.append(chunk[:, pn_per_dev:, :])
    tt_noisy_full = torch.cat(noisy_chunks, dim=1)[:, :n_noisy, :]
    tt_const_full = torch.cat(const_chunks, dim=1)[:, :n_const, :]
    tt_assembled = torch.cat([tt_noisy_full, tt_const_full], dim=1)
    logger.info(f"TT assembled output: {tuple(tt_assembled.shape)}, ref: {tuple(ref_out.shape)}")

    assert tt_assembled.shape == ref_out.shape, f"shape: tt={tuple(tt_assembled.shape)} ref={tuple(ref_out.shape)}"
    assert_quality(tt_assembled, ref_out.float(), pcc=BLOCK_PCC)


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("F_lat", "H_lat", "W_lat"),
    [
        pytest.param(20, 60, 104, id="480p"),
        pytest.param(20, 90, 160, id="720p"),
    ],
)
def test_wan_s2v_transformer_model(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    F_lat: int,
    H_lat: int,
    W_lat: int,
) -> None:
    """PCC parity between TorchWanS2VModel and tt_dit at 1 layer with audio
    injection + AdaIN enabled at block 0. Exercises block stacking +
    embeddings + head + audio cross-attn + AdaIN modulation + state-dict
    translation end-to-end at production latent dims (480p / 720p).
    """
    torch.manual_seed(0)
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    B = 1

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    # Torch reference at full production dims, 1 layer, audio injection off.
    torch_ref = TorchWanS2VModel(
        patch_size=PATCH_SIZE,
        in_dim=16,
        out_dim=16,
        dim=DIM,
        ffn_dim=FFN_DIM,
        freq_dim=FREQ_DIM,
        text_dim=TEXT_DIM,
        text_len=512,
        num_heads=NUM_HEADS,
        num_layers=1,
        audio_dim=AUDIO_DIM,
        num_audio_token=NUM_AUDIO_TOKEN,
        audio_inject_layers=(0,),
        enable_adain=True,
        cond_dim=16,
        eps=EPS,
        rope_max_seq_len=ROPE_MAX_SEQ_LEN,
    ).eval()

    # Build TT model with matching config (audio_inject at block 0, AdaIN on).
    tt_model = WanS2VTransformer3DModel(
        patch_size=PATCH_SIZE,
        num_heads=NUM_HEADS,
        dim=DIM,
        in_channels=16,
        out_channels=16,
        text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM,
        ffn_dim=FFN_DIM,
        num_layers=1,
        cross_attn_norm=True,
        eps=EPS,
        rope_max_seq_len=ROPE_MAX_SEQ_LEN,
        audio_dim=AUDIO_DIM,
        num_audio_layers=NUM_AUDIO_LAYERS,
        num_audio_token=NUM_AUDIO_TOKEN,
        audio_inject_layers=(0,),
        enable_adain=True,
        enable_motioner=False,
        enable_framepack=True,
        cond_dim=16,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        model_type="s2v",
    )

    # Translate torch ref state dict → TT naming → load.
    ref_sd = torch_ref.state_dict()
    tt_sd = translate_s2v_state_dict(ref_sd)
    tt_model.load_torch_state_dict(tt_sd, strict=False)

    # Synthetic inputs.
    noisy = torch.randn(B, 16, F_lat, H_lat, W_lat, dtype=torch.float32)
    ref_latent = torch.randn(B, 16, 1, H_lat, W_lat, dtype=torch.float32)
    motion_latents = torch.zeros(B, 16, 19, H_lat, W_lat, dtype=torch.float32)
    audio_input = torch.randn(B, NUM_AUDIO_LAYERS, AUDIO_DIM, 80, dtype=torch.float32)
    prompt = torch.randn(B, 32, TEXT_DIM, dtype=torch.float32)
    timestep = torch.tensor([500.0], dtype=torch.float32)

    # Torch reference forward.
    with torch.no_grad():
        torch_out = torch_ref(
            noisy_BCFHW=noisy,
            timestep_B=timestep,
            prompt_BLP=prompt,
            ref_BCFHW=ref_latent,
            audio_input_BLCT=audio_input,
            cond_BCFHW=None,
            motion_frames=(17, 5),
        )
    logger.info(
        f"Torch ref output: shape={tuple(torch_out.shape)} range=[{torch_out.min():.4f}, {torch_out.max():.4f}]"
    )

    # TT forward: prepare_audio_emb + prepare_cond_emb + rope + prompt + inner_step + postprocess.
    tt_model.prepare_audio_emb(audio_input, motion_frames=(17, 5), target_num_frames=F_lat)
    tt_model.prepare_cond_emb(
        noisy_latents_torch=noisy,
        ref_latent_torch=ref_latent,
        motion_latents_torch=motion_latents,
        cond_states_torch=None,
        drop_first_motion=True,
    )
    rope_cos, rope_sin, trans_mat = tt_model.get_rope_features(noisy)
    prompt_dev = bf16_tensor(prompt.unsqueeze(0), device=mesh_device)
    tt_prompt = tt_model.prepare_text_conditioning(prompt_dev)
    spatial_dev, N = tt_model.preprocess_spatial_input(noisy)
    tt_timestep = float32_tensor(timestep.reshape(B, 1, 1, 1), device=mesh_device)

    tt_out_dev = tt_model.inner_step(
        spatial_1BNI=spatial_dev,
        prompt_1BLP=tt_prompt,
        rope_cos_1HND=rope_cos,
        rope_sin_1HND=rope_sin,
        trans_mat=trans_mat,
        N=N,
        timestep=tt_timestep,
        gather_output=True,
    )
    tt_out_host = local_device_to_torch(tt_out_dev).float()
    tt_final = tt_model.postprocess_spatial_output_host(tt_out_host, F_lat, H_lat, W_lat, N)
    logger.info(f"TT output: shape={tuple(tt_final.shape)} range=[{tt_final.min():.4f}, {tt_final.max():.4f}]")

    assert (
        tt_final.shape == torch_out.shape
    ), f"shape mismatch tt={tuple(tt_final.shape)} torch={tuple(torch_out.shape)}"
    assert_quality(tt_final, torch_out.float(), pcc=0.99)


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("F_lat", "H_lat", "W_lat"),
    [
        pytest.param(20, 60, 104, id="480p"),
        pytest.param(20, 90, 160, id="720p"),
    ],
)
def test_wan_s2v_transformer_inner_step(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    F_lat: int,
    H_lat: int,
    W_lat: int,
) -> None:
    """``inner_step`` smoke with production weights + synthetic inputs."""
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    snapshot = find_s2v_snapshot()

    cfg = load_s2v_config(snapshot)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    tt_model = WanS2VTransformer3DModel(
        patch_size=PATCH_SIZE,
        num_heads=NUM_HEADS,
        dim=DIM,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        text_dim=TEXT_DIM,
        freq_dim=FREQ_DIM,
        ffn_dim=FFN_DIM,
        num_layers=cfg["num_layers"],
        cross_attn_norm=True,
        eps=EPS,
        rope_max_seq_len=ROPE_MAX_SEQ_LEN,
        audio_dim=AUDIO_DIM,
        num_audio_layers=NUM_AUDIO_LAYERS,
        num_audio_token=NUM_AUDIO_TOKEN,
        audio_inject_layers=AUDIO_INJECT_LAYERS,
        enable_adain=cfg.get("enable_adain", True),
        enable_motioner=False,
        enable_framepack=True,
        cond_dim=16,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        is_fsdp=is_fsdp,
        model_type="s2v",
    )
    ref_sd = load_s2v_state_dict(snapshot)
    tt_sd = translate_s2v_state_dict(ref_sd)
    tt_model.load_torch_state_dict(tt_sd, strict=True)
    logger.info(f"Loaded production weights ({cfg['num_layers']} layers)")

    # Synthesize inputs at the parametrized resolution.
    B = 1
    T_audio = 80  # production per-clip audio length (fixed)

    torch.manual_seed(0)
    noisy_latents = torch.randn(B, 16, F_lat, H_lat, W_lat, dtype=torch.float32)
    ref_latent = torch.randn(B, 16, 1, H_lat, W_lat, dtype=torch.float32)
    motion_latents = torch.zeros(B, 16, 19, H_lat, W_lat, dtype=torch.float32)  # zero motion (clip 0)
    wav2vec2_layers = torch.randn(B, NUM_AUDIO_LAYERS, AUDIO_DIM, T_audio, dtype=torch.float32)
    prompt = torch.randn(B, 32, TEXT_DIM, dtype=torch.float32)
    timestep = torch.tensor([500.0], dtype=torch.float32)

    # Populate per-clip caches (audio embeddings, cond embeddings).
    tt_model.prepare_audio_emb(wav2vec2_layers, motion_frames=(17, 5), target_num_frames=F_lat)
    tt_model.prepare_cond_emb(
        noisy_latents_torch=noisy_latents,
        ref_latent_torch=ref_latent,
        motion_latents_torch=motion_latents,
        cond_states_torch=None,
        drop_first_motion=True,
    )

    # Cache rope features + prompt embedding (outer denoise loop does this once).
    rope_cos, rope_sin, trans_mat = tt_model.get_rope_features(noisy_latents)
    # prepare_text_conditioning expects a ttnn.Tensor (production gets it from UMT5).
    prompt_dev = bf16_tensor(prompt.unsqueeze(0), device=mesh_device)
    tt_prompt = tt_model.prepare_text_conditioning(prompt_dev)
    spatial_dev, N = tt_model.preprocess_spatial_input(noisy_latents)
    tt_timestep = float32_tensor(timestep.reshape(B, 1, 1, 1), device=mesh_device)

    # gather_output=True does the SP-gather on-device so local_device_to_torch
    # returns the full noisy-token tensor instead of a per-device slice.
    logger.info(f"Running inner_step on {cfg['num_layers']}-layer model")
    tt_out = tt_model.inner_step(
        spatial_1BNI=spatial_dev,
        prompt_1BLP=tt_prompt,
        rope_cos_1HND=rope_cos,
        rope_sin_1HND=rope_sin,
        trans_mat=trans_mat,
        N=N,
        timestep=tt_timestep,
        gather_output=True,
    )
    out_torch = local_device_to_torch(tt_out).float()
    logger.info(f"inner_step output: shape={tuple(out_torch.shape)}, dtype={out_torch.dtype}")
    logger.info(f"output range: [{out_torch.min():.4f}, {out_torch.max():.4f}]")

    assert torch.isfinite(out_torch).all(), "inner_step output contains NaN/Inf"
    final = tt_model.postprocess_spatial_output_host(out_torch, F_lat, H_lat, W_lat, N)
    logger.info(f"postprocessed output: {tuple(final.shape)}")
    assert final.shape == (
        B,
        16,
        F_lat,
        H_lat,
        W_lat,
    ), f"postprocessed shape mismatch: got {tuple(final.shape)} expected {(B, 16, F_lat, H_lat, W_lat)}"
    del tt_model
