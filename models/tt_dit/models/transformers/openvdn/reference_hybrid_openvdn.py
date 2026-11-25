# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Torch (CPU) reference for the OpenVDN hybrid attention, copied from OpenVDN/vdn-minimax-h3 @e262cb57 (Apache-2.0):
`src/models/hybrid_attention.py`, `attention_gates.py`, `linear_attention/{branch,features,layers}.py` and
`ops/rms_norm.py`. Training path (eager) only, in the released checkpoints' configuration: `vdn_solve` delta rule,
`bridge="alpha"`, short conv on K and V, text state on, `anchor_frames="both"`, reference window softmax. The
array math lives in `reference_openvdn.py`.

Parameter names match the checkpoint: `orig.*` (the H3 attention), `softmax_gate.*`, `linear_attention.*` and
`to_out_linear.*`. Used by the module unit tests; the model's own repo stays the source of truth.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn

from . import reference_openvdn as ref


def apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """diffusers `transformer_minimax_h3._apply_rotary_emb`. hidden_states `[B, N, H, d]`, cos/sin `[N, rotary_dim]`."""
    rotary_dim = cos.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    cos = cos.to(hidden_states.dtype)[None, :, None, :]
    sin = sin.to(hidden_states.dtype)[None, :, None, :]
    x1, x2 = rotary.chunk(2, dim=-1)
    rotary = rotary * cos + torch.cat((-x2, x1), dim=-1) * sin
    return torch.cat((rotary, passthrough), dim=-1)


@dataclasses.dataclass(frozen=True)
class SequenceLayout:
    """`src/models/sequence_layout.py`: video rows contiguous and frame-major, the prompt rows at `text_start`."""

    seq_len: int
    video_start: int
    num_frames: int
    tokens_per_frame: int
    frame_height: int
    frame_width: int
    text_start: int = 0
    text_len: int = 0

    @property
    def video_end(self) -> int:
        return self.video_start + self.num_frames * self.tokens_per_frame

    @property
    def frame_size(self) -> tuple[int, int]:
        assert self.frame_height * self.frame_width == self.tokens_per_frame
        return self.frame_height, self.frame_width


class OutputGate(nn.Module):
    """`attention_gates.OutputGate`: per head (softmax) or per channel, low rank (linear)."""

    def __init__(self, hidden_size, num_heads, head_dim=None, bottleneck=None, init_value=0.9, init="constant"):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.down = None if bottleneck is None else nn.Linear(hidden_size, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck or hidden_size, num_heads * (head_dim or 1), bias=True)
        if init == "constant":
            nn.init.zeros_(self.up.weight)
            nn.init.constant_(self.up.bias, math.log(init_value / (1.0 - init_value)))

    def forward(self, x):
        gate = torch.sigmoid(self.up(x if self.down is None else self.down(x)))
        return gate.view(-1, self.num_heads, self.head_dim or 1)


class FrameKDAAlpha(nn.Module):
    """`linear_attention.layers.FrameKDAAlpha`: `alpha = exp(-exp(A_log) * softplus(up(down(mean)) + dt_bias))`."""

    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.down = nn.Linear(hidden_size, head_dim, bias=False)
        self.up = nn.Linear(head_dim, num_heads * head_dim, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(num_heads).uniform_(1, 16)))
        dt = torch.exp(
            torch.rand(num_heads * head_dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, frame_mean_x):  # [F, hidden] -> [F, H, d_k] fp32
        delta = F.linear(frame_mean_x.float(), self.down.weight.float())
        delta = F.linear(delta, self.up.weight.float()) + self.dt_bias.float()
        scale = torch.exp(self.A_log.float())[:, None]
        delta = delta.view(-1, self.num_heads, self.head_dim)
        return torch.exp(-scale * F.softplus(delta))


class RMSNorm(nn.Module):
    """`ops.rms_norm.RMSNorm`, fp32 second moment."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        ms = torch.linalg.vector_norm(x, dim=-1, keepdim=True, dtype=torch.float32).pow(2) / x.shape[-1]
        return x * torch.rsqrt(ms + self.eps).to(x.dtype) * self.weight.to(x.dtype)


class LinearAttentionSepConv(nn.Module):
    """`linear_attention.features.LinearAttentionSepConv`: depthwise 5x5 spatial conv per frame, then a depthwise
    5-tap temporal conv across frames (zero padded, both bias-free)."""

    KERNEL = 5

    def __init__(self, channels, projs=("k", "v")):
        super().__init__()
        self.projs = tuple(projs)
        k = self.KERNEL
        for name in self.projs:
            sp = nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels, bias=False)
            nn.init.normal_(sp.weight, std=(k * k) ** -0.5)
            tm = nn.Conv1d(channels, channels, k, padding=k // 2, groups=channels, bias=False)
            nn.init.normal_(tm.weight, std=k**-0.5)
            setattr(self, f"{name}_sp", sp)
            setattr(self, f"{name}_tm", tm)

    def apply_conv(self, proj, tokens, num_frames, frame_size):
        """[F*S, H, d] -> [F*S, H, d]."""
        heads, head_dim = tokens.shape[-2], tokens.shape[-1]
        grid_h, grid_w = frame_size
        channels = heads * head_dim
        volume = tokens.reshape(num_frames, grid_h, grid_w, channels).permute(0, 3, 1, 2)
        volume = F.conv2d(volume, getattr(self, f"{proj}_sp").weight, padding=self.KERNEL // 2, groups=channels)
        x = volume.permute(0, 2, 3, 1).reshape(num_frames, grid_h * grid_w, channels)
        w = getattr(self, f"{proj}_tm").weight.squeeze(1).to(x.dtype)  # [C, K]
        pad = self.KERNEL // 2
        xp = F.pad(x, (0, 0, 0, 0, pad, pad))
        out = None
        for dt in range(self.KERNEL):  # `_temporal_shift`
            part = xp[dt : dt + num_frames] * w[:, dt].view(1, 1, -1)
            out = part if out is None else out + part
        return out.reshape(-1, heads, head_dim)


def prepare_linear_features(tokens, l2norm, conv=None, proj=None, num_frames=None, frame_size=None):
    """`linear_attention.features.prepare_linear_features`: [conv ->] SiLU [-> L2Norm]."""
    if conv is not None and proj in conv.projs:
        tokens = conv.apply_conv(proj, tokens, num_frames, frame_size)
    x = F.silu(tokens)
    return F.normalize(x, dim=-1, eps=1e-6).to(x.dtype) if l2norm else x


class BidirectionalLinearBranch(nn.Module):
    """`linear_attention.branch.BidirectionalLinearBranch`, training body (`_readout`), vdn_solve, bridge alpha."""

    TEXT_STATE_SCALE = 0.5

    def __init__(self, hidden_size, num_heads, head_dim, short_conv=("k", "v")):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.short_conv = LinearAttentionSepConv(num_heads * head_dim, short_conv)
        self.alpha = FrameKDAAlpha(hidden_size, num_heads, head_dim)
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=False)
        self.output_gate = OutputGate(hidden_size, num_heads, head_dim, bottleneck=head_dim, init="random")
        self.norm = RMSNorm(head_dim)

    def feature(self, tokens, proj, num_frames, frame_size, use_conv=True):
        conv = self.short_conv if use_conv else None
        return prepare_linear_features(
            tokens, l2norm=(proj != "v"), conv=conv, proj=proj, num_frames=num_frames, frame_size=frame_size
        )

    def text_state(self, text_x, text_qkv_raw):
        """`_text_state`: half the prompt written into a zero state as one delta-rule chunk. [H, d_v, d_k]."""
        if text_x is None:
            return None
        heads, head_dim = self.num_heads, self.head_dim
        length = text_qkv_raw[1].shape[0]
        key = self.feature(text_qkv_raw[1], "k", None, None, use_conv=False)
        value = self.feature(text_qkv_raw[2], "v", None, None, use_conv=False)
        key = key.view(1, length, heads, head_dim).permute(0, 2, 1, 3)
        value = value.view(1, length, heads, head_dim).permute(0, 2, 1, 3)
        beta = torch.sigmoid(self.beta_proj(text_x)).view(1, length, heads).permute(0, 2, 1)
        a, b = ref.frame_statistics(key, value, beta)
        _, injection = ref.vdn_factor_apply(torch.ones(1, heads, head_dim), a, b)
        return self.TEXT_STATE_SCALE * injection[0]

    def frame_alpha(self, xv, num_frames, tokens_per_frame):
        return self.alpha(xv.view(num_frames, tokens_per_frame, -1).mean(dim=1, dtype=torch.float32))

    def forward(
        self, xv, num_frames, tokens_per_frame, bounds, qkv_raw, frame_size, skip_ends=True, text_x=None, text_qkv_raw=None
    ):
        """xv [F*S, hidden], qkv_raw raw q/k/v [F*S, H, d] -> readout [F*S, H*d] (gated, normed; zero on the anchor
        frames when skip_ends)."""
        if not skip_ends:
            return self._readout(xv, num_frames, tokens_per_frame, bounds, qkv_raw, frame_size, text_x, text_qkv_raw)
        if num_frames <= 2:
            return xv.new_zeros(num_frames * tokens_per_frame, self.num_heads * self.head_dim)
        inner = slice(tokens_per_frame, (num_frames - 1) * tokens_per_frame)
        readout = self._readout(
            xv[inner],
            num_frames - 2,
            tokens_per_frame,
            [(lo - 1, hi - 1) for lo, hi in bounds[1 : num_frames - 1]],
            tuple(t[inner] for t in qkv_raw),
            frame_size,
            text_x,
            text_qkv_raw,
        )
        out = readout.new_zeros(num_frames * tokens_per_frame, readout.shape[-1])
        out[inner] = readout
        return out

    def _readout(self, xv, num_frames, tokens_per_frame, bounds, qkv_raw, frame_size, text_x, text_qkv_raw):
        num_tokens = num_frames * tokens_per_frame
        heads, head_dim = self.num_heads, self.head_dim
        shape_per_frame = (num_frames, tokens_per_frame, heads, head_dim)

        query, key, value = (
            self.feature(t, p, num_frames, frame_size) for t, p in zip(qkv_raw, ("q", "k", "v"))
        )
        query_by_frame = query.view(shape_per_frame)
        key_by_frame = key.view(shape_per_frame).permute(0, 2, 1, 3)
        value_by_frame = value.view(shape_per_frame).permute(0, 2, 1, 3)
        beta = torch.sigmoid(self.beta_proj(xv)).view(num_frames, tokens_per_frame, heads).permute(0, 2, 1)

        a, b = ref.frame_statistics(key_by_frame, value_by_frame, beta)
        alpha = self.frame_alpha(xv, num_frames, tokens_per_frame)
        text_state = self.text_state(text_x, text_qkv_raw)
        transitions, injections = ref.vdn_factor_apply(alpha, a, b)
        prefix, suffix = ref.run_scans(transitions, injections, text_state)
        linear_state = ref.gather_linear_state(prefix, suffix, alpha, bounds, text_state).to(xv.dtype)

        readout = torch.einsum("fhvk,fshk->fshv", linear_state, query_by_frame)
        readout = self.norm(readout.reshape(num_tokens, heads, head_dim))
        return (readout * self.output_gate(xv)).reshape(num_tokens, heads * head_dim)


class HybridAttention(nn.Module):
    """`hybrid_attention.HybridAttention` around a diffusers `MiniMaxH3Attention` (`orig`), softmax_impl "ref"."""

    def __init__(self, orig_attn, hidden_size, radius=1, chunk=5, anchor_frames="both", enable_text_state=True):
        super().__init__()
        self.orig = orig_attn
        self.num_heads, self.head_dim = orig_attn.heads, orig_attn.head_dim
        self.radius, self.chunk, self.anchor_frames = radius, chunk, anchor_frames
        self.enable_text_state = enable_text_state
        self.layout: SequenceLayout | None = None
        self.linear_attention = BidirectionalLinearBranch(hidden_size, self.num_heads, self.head_dim)
        self.to_out_linear = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        self.softmax_gate = OutputGate(hidden_size, self.num_heads, init_value=0.99)

    def forward(self, hidden_states, rotary_emb=None, attention_mask=None):
        assert attention_mask is None and hidden_states.shape[0] == 1
        return self._hybrid_forward(hidden_states[0], rotary_emb).unsqueeze(0)

    def _qkv(self, x, rotary_emb):
        orig = self.orig
        query_raw, key_raw, value = (p(x).unflatten(-1, (orig.heads, -1)) for p in (orig.to_q, orig.to_k, orig.to_v))
        query, key = orig.norm_q(query_raw), orig.norm_k(key_raw)
        if rotary_emb is not None:
            query = apply_rotary_emb(query.unsqueeze(0), *rotary_emb).squeeze(0)
            key = apply_rotary_emb(key.unsqueeze(0), *rotary_emb).squeeze(0)
        return query, key, value, (query_raw, key_raw, value)

    def _hybrid_forward(self, x, rotary_emb):
        layout = self.layout
        bounds = ref.window_bounds(layout.num_frames, self.radius, self.chunk)
        full_cover = all(lo <= 0 and hi >= layout.num_frames - 1 for lo, hi in bounds)
        scale = self.head_dim**-0.5

        query, key, value, qkv_raw = self._qkv(x, rotary_emb)
        if full_cover:
            softmax_out = F.scaled_dot_product_attention(
                query.permute(1, 0, 2), key.permute(1, 0, 2), value.permute(1, 0, 2), scale=scale
            ).permute(1, 0, 2)
        else:
            softmax_out = ref.window_softmax(
                query,
                key,
                value,
                video_start=layout.video_start,
                num_frames=layout.num_frames,
                tokens_per_frame=layout.tokens_per_frame,
                bounds=bounds,
                scale=scale,
                anchor_frames=self.anchor_frames,
            )
        flat = (softmax_out * self.softmax_gate(x).to(softmax_out.dtype)).reshape(x.shape[0], -1)
        out = self.orig.to_out[1](self.orig.to_out[0](flat.type_as(x)))
        if full_cover:
            return out

        vs, ve = layout.video_start, layout.video_end
        text_x = text_qkv_raw = None
        if self.enable_text_state:
            ts, te = layout.text_start, layout.text_start + layout.text_len
            text_x, text_qkv_raw = x[ts:te], tuple(t[ts:te] for t in qkv_raw)
        readout = self.linear_attention(
            x[vs:ve],
            layout.num_frames,
            layout.tokens_per_frame,
            bounds,
            tuple(t[vs:ve] for t in qkv_raw),
            layout.frame_size,
            skip_ends=self.anchor_frames == "both",
            text_x=text_x,
            text_qkv_raw=text_qkv_raw,
        )
        out = out.clone()
        out[vs:ve] += self.to_out_linear(readout.type_as(x))
        return out
