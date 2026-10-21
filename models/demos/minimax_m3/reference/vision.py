# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Torch reference for the MiniMax-M3 vision tower: ViT, per-patch projector, 2x2 patch merge.

Mirrors sglang's MiniMaxVLVisionModel: attention is block-diagonal per image, and a video is split into segments of
at most `vision_segment_max_frames` temporal patches. The checkpoint ships no modeling file; transformers'
`MiniMaxM3VLVisionModel` + `MiniMaxM3VLMultiModalProjector` give the same output for one image.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors import safe_open

VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")


@dataclass(frozen=True)
class VisionArgs:
    hidden: int
    layers: int
    heads: int
    intermediate: int
    patch: int
    channels: int
    eps: float
    rope_theta: float
    temporal_patch: int
    merge: int
    segment_max_frames: int | None
    text_hidden: int
    projector_hidden: int

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def axis_dim(self) -> int:
        """Rotary dims per axis (t, h, w); 3 * axis_dim leading head dims rotate, the rest pass through."""
        return 2 * ((2 * (self.head_dim // 2) // 3) // 2)

    @property
    def patch_elems(self) -> int:
        return self.channels * self.temporal_patch * self.patch * self.patch

    @classmethod
    def from_config(cls, cfg: dict) -> "VisionArgs":
        v = cfg["vision_config"]
        comp = cfg.get("img_token_compression_config") or v.get("img_token_compression_config") or {}
        text_hidden = cfg["text_config"]["hidden_size"]
        return cls(
            hidden=v["hidden_size"],
            layers=v["num_hidden_layers"],
            heads=v["num_attention_heads"],
            intermediate=v["intermediate_size"],
            patch=v["patch_size"],
            channels=v.get("num_channels", 3),
            eps=v["layer_norm_eps"],
            rope_theta=v.get("rope_theta", 10000.0),
            temporal_patch=comp.get("temporal_patch_size", 2),
            merge=comp.get("spatial_merge_size", 2),
            segment_max_frames=v.get("vision_segment_max_frames"),
            text_hidden=text_hidden,
            projector_hidden=cfg.get("projector_hidden_size") or text_hidden,
        )

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str) -> "VisionArgs":
        with open(os.path.join(ckpt_dir, "config.json")) as f:
            return cls.from_config(json.load(f))


def load_vision_state_dict(ckpt_dir: str, dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor]:
    """The vision tower, projector and patch-merge weights of the checkpoint, under their checkpoint names."""
    with open(os.path.join(ckpt_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)["weight_map"]
    by_file: dict[str, list[str]] = {}
    for name, file in index.items():
        if name.startswith(VISION_PREFIXES):
            by_file.setdefault(file, []).append(name)
    state = {}
    for file, names in by_file.items():
        with safe_open(os.path.join(ckpt_dir, file), "pt") as f:
            for name in names:
                state[name] = f.get_tensor(name).to(dtype)
    return state


def vision_segments(grid_thw, segment_max_frames: int | None) -> list[tuple[int, int, int]]:
    """Attention blocks: one per image, and a video split into runs of at most `segment_max_frames` temporal patches."""
    out = []
    for t, h, w in (tuple(int(x) for x in g) for g in grid_thw):
        if segment_max_frames is None or t <= segment_max_frames:
            out.append((t, h, w))
        else:
            out.extend((min(segment_max_frames, t - i), h, w) for i in range(0, t, segment_max_frames))
    return out


def rope_frequencies(segments, args: VisionArgs) -> torch.Tensor:
    """[patches, 3 * axis_dim / 2] fp32 rotation angles, t|h|w bands, in the processor's patch order."""
    m = args.merge
    coords = []
    for t, h, w in segments:
        hi = torch.arange(h).unsqueeze(1).expand(-1, w).reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        wi = torch.arange(w).unsqueeze(0).expand(h, -1).reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        ti = torch.arange(t).repeat_interleave(h * w)
        coords.append(torch.stack([ti, hi.repeat(t), wi.repeat(t)], dim=-1))
    coords = torch.cat(coords).float()
    inv_freq = 1.0 / (args.rope_theta ** (torch.arange(0, args.axis_dim, 2, dtype=torch.float32) / args.axis_dim))
    return torch.cat([coords[:, i : i + 1] * inv_freq for i in range(3)], dim=-1)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    rot = cos.shape[-1]
    xr, xp = x[..., :rot].float(), x[..., rot:]
    xr = xr * cos[:, None, :] + _rotate_half(xr) * sin[:, None, :]
    return torch.cat([xr.to(xp.dtype), xp], dim=-1)


class MiniMaxM3VisionReference:
    """Vision tower + projector + patch merge on the host. `compute_dtype` float32 is the accuracy golden."""

    def __init__(self, state: dict[str, torch.Tensor], args: VisionArgs, compute_dtype: torch.dtype = torch.float32):
        self.args = args
        self.dt = compute_dtype
        g = lambda name: state[name].to(compute_dtype)
        pre = "vision_tower.vision_model."
        self.patch = g(pre + "embeddings.patch_embedding.weight").reshape(args.hidden, -1)
        self.pre_ln = (g(pre + "pre_layrnorm.weight"), g(pre + "pre_layrnorm.bias"))
        self.layers = []
        for i in range(args.layers):
            p = f"{pre}encoder.layers.{i}."
            self.layers.append(
                dict(
                    ln1=(g(p + "layer_norm1.weight"), g(p + "layer_norm1.bias")),
                    qkv_w=torch.cat([g(p + f"self_attn.{n}_proj.weight") for n in "qkv"]),
                    qkv_b=torch.cat([g(p + f"self_attn.{n}_proj.bias") for n in "qkv"]),
                    o=(g(p + "self_attn.out_proj.weight"), g(p + "self_attn.out_proj.bias")),
                    ln2=(g(p + "layer_norm2.weight"), g(p + "layer_norm2.bias")),
                    fc1=(g(p + "mlp.fc1.weight"), g(p + "mlp.fc1.bias")),
                    fc2=(g(p + "mlp.fc2.weight"), g(p + "mlp.fc2.bias")),
                )
            )
        self.proj = [
            (g(f"multi_modal_projector.{n}.weight"), g(f"multi_modal_projector.{n}.bias"))
            for n in ("linear_1", "linear_2")
        ]
        self.merge = [
            (g(f"patch_merge_mlp.{n}.weight"), g(f"patch_merge_mlp.{n}.bias")) for n in ("linear_1", "linear_2")
        ]

    @classmethod
    def from_checkpoint(cls, ckpt_dir: str, compute_dtype: torch.dtype = torch.float32):
        return cls(load_vision_state_dict(ckpt_dir), VisionArgs.from_checkpoint(ckpt_dir), compute_dtype)

    @torch.no_grad()
    def vit(self, pixel_values: torch.Tensor, grid_thw) -> torch.Tensor:
        a = self.args
        x = pixel_values.to(self.dt) @ self.patch.T
        x = F.layer_norm(x, (a.hidden,), *self.pre_ln, a.eps)
        segs = vision_segments(grid_thw, a.segment_max_frames)
        freqs = rope_frequencies(segs, a)
        cos, sin = torch.cat([freqs, freqs], -1).cos(), torch.cat([freqs, freqs], -1).sin()
        lens = [t * h * w for t, h, w in segs]
        n = x.shape[0]
        for L in self.layers:
            hn = F.layer_norm(x, (a.hidden,), *L["ln1"], a.eps)
            qkv = F.linear(hn, L["qkv_w"], L["qkv_b"]).view(n, 3, a.heads, a.head_dim)
            q, k, v = _apply_rope(qkv[:, 0], cos, sin), _apply_rope(qkv[:, 1], cos, sin), qkv[:, 2]
            outs, s0 = [], 0
            for ln in lens:
                qs, ks, vs = (t[s0 : s0 + ln].transpose(0, 1) for t in (q, k, v))
                outs.append(F.scaled_dot_product_attention(qs, ks, vs).transpose(0, 1))
                s0 += ln
            x = x + F.linear(torch.cat(outs).reshape(n, a.hidden), *L["o"])
            hn = F.layer_norm(x, (a.hidden,), *L["ln2"], a.eps)
            x = x + F.linear(F.gelu(F.linear(hn, *L["fc1"])), *L["fc2"])
        return x

    @torch.no_grad()
    def project(self, vit_out: torch.Tensor) -> torch.Tensor:
        x = F.linear(F.gelu(F.linear(vit_out, *self.proj[0])), *self.proj[1])
        x = x.reshape(x.shape[0] // self.args.merge**2, -1)
        return F.linear(F.gelu(F.linear(x, *self.merge[0])), *self.merge[1])

    def __call__(self, pixel_values: torch.Tensor, grid_thw) -> torch.Tensor:
        """[patches, patch_elems] in the processor's order -> [patches / merge^2, text_hidden]."""
        return self.project(self.vit(pixel_values, grid_thw))
