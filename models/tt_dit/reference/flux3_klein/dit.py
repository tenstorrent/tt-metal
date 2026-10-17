# SPDX-FileCopyrightText: © 2026 Black Forest Labs, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from https://github.com/black-forest-labs/flux-3-image-klein (src/flux3/dit.py) at e53f4ea,
# licensed under Apache-2.0. Modified by Tenstorrent as the PyTorch reference for the tt-metal port.

""" Image DiT: per-mode blocks, then shared single-stream blocks."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

MODES = ("image", "image_cond", "text")  # sorted; the order tokens take inside a sample
IN_CHANNELS = {"image": 128, "image_cond": 128, "text": 20480}
HIDDEN = 3072
HEADS = 24
DEPTH = 5
DEPTH_SINGLE = 28
AXES_DIM = (32, 32, 32, 32)
MLP_RATIO = 3.0
THETA = 10000


@dataclass
class Packed:
    """Mode-major token pack. mode_indices[m][s] = (start, end) of sample s in data[m]."""

    data: dict[str, Tensor]
    ids: dict[str, Tensor]
    timesteps: dict[str, Tensor]
    mode_indices: dict[str, list[tuple[int, int]]]

    @property
    def num_samples(self) -> int:
        return len(self.mode_indices["image"])


def timestep_embedding(t: Tensor, dim: int = 256, max_period: int = 10000) -> Tensor:
    t = 1000.0 * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    out = torch.einsum("...n,d->...nd", pos, 1.0 / (theta**scale))
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def rope_embed(ids: Tensor) -> Tensor:
    """ids (B, L, 4) -> rotation matrices (B, 1, L, head_dim // 2, 2, 2)."""
    with torch.autocast(device_type=ids.device.type, enabled=False):
        emb = torch.cat([rope(ids[..., i], AXES_DIM[i], THETA) for i in range(len(AXES_DIM))], dim=-3)
    return emb.unsqueeze(1)


def rmsnorm_rope(x: Tensor, scale: Tensor, freqs: Tensor, eps: float = 1e-6) -> Tensor:
    with torch.autocast(device_type=x.device.type, enabled=False):
        b, h, l, d = x.shape
        x_f = x.float()
        x_norm = x_f * x_f.pow(2).mean(-1, keepdim=True).add(eps).rsqrt() * scale.float()
        x_r, freqs = x_norm.reshape(b, h, l, d // 2, 1, 2), freqs.float()
        rotated = freqs[..., 0] * x_r[..., 0] + freqs[..., 1] * x_r[..., 1]
        return rotated.reshape(b, h, l, d).to(x.dtype)


def layernorm_modulation(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return (1.0 + scale) * F.layer_norm(x, (x.shape[-1],), eps=1e-6) + shift


def packed_attention(q: Tensor, k: Tensor, v: Tensor, bounds: list[tuple[int, int]]) -> Tensor:
    """Per-sample attention over (1, H, L, D); samples never attend to each other."""
    _, heads, length, dim = q.shape
    out = torch.zeros(length, heads * dim, device=q.device, dtype=q.dtype)
    backend = SDPBackend.CUDNN_ATTENTION if q.is_cuda else SDPBackend.MATH
    for start, end in bounds:
        if start == end:
            continue
        with sdpa_kernel(backend):
            chunk = F.scaled_dot_product_attention(q[:, :, start:end], k[:, :, start:end], v[:, :, start:end])
        out[start:end] = chunk[0].transpose(0, 1).reshape(end - start, heads * dim)
    return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
        return rmsnorm_rope(q, self.query_norm.scale, freqs), rmsnorm_rope(k, self.key_norm.scale, freqs)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(F.silu(self.in_layer(x)))


class ModeModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, vec: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return tuple(self.lin(F.silu(vec)).chunk(3, dim=-1))  # shift, scale, gate


class LastLayer(nn.Module):
    def __init__(self, hidden: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(hidden, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)
        return self.linear(layernorm_modulation(x, scale, shift))


class Block(nn.Module):
    def __init__(self, hidden: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.heads = heads
        mlp_hidden = int(hidden * mlp_ratio)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, mlp_hidden * 2, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden, bias=False)
        self.norm = QKNorm(hidden // heads)

    def forward(
        self, x: Tensor, freqs: Tensor, bounds: list[tuple[int, int]], mod: tuple[Tensor, Tensor, Tensor]
    ) -> Tensor:
        shift, scale, gate = mod
        x_mod = layernorm_modulation(x, scale, shift)
        split = lambda t: t.reshape(1, t.shape[0], self.heads, -1).transpose(1, 2)  # L (H D) -> 1 H L D
        q, k = self.norm(split(self.q_proj(x_mod)), split(self.k_proj(x_mod)), freqs)
        v = split(self.v_proj(x_mod))
        attn = packed_attention(q, k, v.to(q.dtype), bounds)
        gate_in, up = self.mlp_in(x_mod).chunk(2, dim=-1)
        return x + gate * (self.attn_out(attn) + self.mlp_out(F.silu(gate_in) * up))


class DiT(nn.Module):
    def __init__(self, flow_map: bool = False) -> None:
        super().__init__()
        self.flow_map = flow_map
        self.emb_in = nn.ModuleDict({m: nn.Linear(IN_CHANNELS[m], HIDDEN, bias=False) for m in ("image", "image_cond")})
        self.txt_in = nn.Linear(IN_CHANNELS["text"], HIDDEN, bias=False)
        self.time_in = MLPEmbedder(256, HIDDEN)
        if flow_map:
            self.ref_time_in = MLPEmbedder(256, HIDDEN)
        self.early_stream_modulations = nn.ModuleDict(
            {k: ModeModulation(HIDDEN) for k in ("image", "image_cond", "txt")}
        )
        self.single_stream_modulations = nn.ModuleDict(
            {k: ModeModulation(HIDDEN) for k in ("image", "image_cond", "txt")}
        )
        self.content_mode_blocks = nn.ModuleDict(
            {m: nn.ModuleList([Block(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH)]) for m in ("image", "image_cond")}
        )
        self.txt_mode_blocks = nn.ModuleList([Block(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH)])
        self.single_blocks = nn.ModuleList([Block(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH_SINGLE)])
        self.final_layer = nn.ModuleDict({"image": LastLayer(HIDDEN, IN_CHANNELS["image"])})

    def forward(self, packed: Packed, timesteps_ref: Tensor | None = None) -> Tensor:
        """Velocity prediction for the image tokens, in packed.data["image"] order.
        timesteps_ref: flow map only, the time each sample jumps to."""
        if (timesteps_ref is not None) != self.flow_map:
            raise ValueError("timesteps_ref is required by, and only by, a flow-map DiT")
        data, ids, timesteps = dict(packed.data), dict(packed.ids), dict(packed.timesteps)
        for m in MODES:  # an empty mode gets one phantom token so the projections run; it is never attended
            if data[m].shape[0] == 0:
                data[m], ids[m], timesteps[m] = (
                    data[m].new_zeros(1, data[m].shape[1]),
                    ids[m].new_zeros(1, 4),
                    timesteps[m].new_zeros(1),
                )
        mod_key = lambda m: "txt" if m == "text" else m
        vec = {m: self.time_in(timestep_embedding(timesteps[m])) for m in MODES}
        if timesteps_ref is not None:
            ref = self.ref_time_in(timestep_embedding(timesteps_ref.to(timesteps["image"].dtype)))
            for m in MODES:
                parts = [ref[s].expand(b - a, -1) for s, (a, b) in enumerate(packed.mode_indices[m]) if b > a]
                if parts:
                    vec[m] = vec[m] + torch.cat(parts)
        early = {m: self.early_stream_modulations[mod_key(m)](vec[m]) for m in MODES}
        single = {m: self.single_stream_modulations[mod_key(m)](vec[m]) for m in MODES}
        x = {m: (self.txt_in if m == "text" else self.emb_in[m])(data[m]) for m in MODES}
        freqs = {m: rope_embed(ids[m][None]) for m in MODES}
        for d in range(DEPTH):
            for m in ("image", "image_cond"):
                x[m] = self.content_mode_blocks[m][d](x[m], freqs[m], packed.mode_indices[m], early[m])
            x["text"] = self.txt_mode_blocks[d](x["text"], freqs["text"], packed.mode_indices["text"], early["text"])

        chunks, id_chunks, mods, bounds, placement, offset = [], [], ([], [], []), [], [], 0
        for s in range(packed.num_samples):
            start = offset
            for m in MODES:
                a, b = packed.mode_indices[m][s]
                if b > a:
                    chunks.append(x[m][a:b])
                    id_chunks.append(ids[m][a:b])
                    for i in range(3):
                        mods[i].append(single[m][i][a:b])
                    placement.append((m, a, b, offset))
                    offset += b - a
            bounds.append((start, offset))
        h = torch.cat(chunks)
        freqs_sm = rope_embed(torch.cat(id_chunks)[None])
        mods = tuple(torch.cat(c) for c in mods)
        for block in self.single_blocks:
            h = block(h, freqs_sm, bounds, mods)

        image = torch.empty_like(x["image"])
        for m, a, b, start in placement:
            if m == "image":
                image[a:b] = h[start : start + b - a]
        return self.final_layer["image"](image, vec["image"])
