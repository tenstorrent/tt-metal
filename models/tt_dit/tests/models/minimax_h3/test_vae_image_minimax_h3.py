# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image VAE decoder (ViT2D) gates: tt_dit's ``decode_clip`` vs a self-contained torch reference.

The reference reads the *raw* release checkpoint (original MiniMax layout: fused per-head-interleaved ``to_qkv``,
gated ``ff.w1`` = ``[gate | value]``), while tt_dit loads the diffusers-layout conversion -- so these gates also check
the conversion. It mirrors the SGLang image VAE: ``post_quant_conv`` (1x1 Conv2d) -> ``x_embedder`` -> + 4 register
tokens + zero cls -> 24 blocks (RMSNorm, parameter-free q/k RMS, NeoX RoPE on 32 of 64 lanes over a 2-axis
length-normalised grid with theta 100, LayerScale) -> LayerNorm -> ``proj_out`` -> unpatchify, tiled in pixel space
(256 px tiles, >= 64 px overlap) with the release's blend (vertical against the *unblended* tile above, then
horizontal against the *unblended* left neighbour). Output is in reference space (before the ImageNet de-norm).

Env: MINIMAX_H3_IMAGE_MODEL_PATH = converted snapshot (vae/config.json + diffusion_pytorch_model.safetensors),
     MINIMAX_H3_IMAGE_RAW_PATH = the original release snapshot.
"""

import math
import os

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

from ....models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3Vae, MiniMaxH3VaeConfig
from ....utils.check import assert_quality
from .common import MESH_4X8_RING

SINGLE_DEVICE = [pytest.param((1, 1), {}, id="single_device")]
GALAXY = [MESH_4X8_RING]
RATIO = 16
TILE_PX = 256


def _paths() -> tuple[str, str]:
    converted, raw = os.environ.get("MINIMAX_H3_IMAGE_MODEL_PATH"), os.environ.get("MINIMAX_H3_IMAGE_RAW_PATH")
    if not converted or not raw:
        pytest.skip("set MINIMAX_H3_IMAGE_MODEL_PATH (converted) and MINIMAX_H3_IMAGE_RAW_PATH (release) snapshots")
    return os.path.join(converted, "vae"), os.path.join(raw, "vae")


def _load(path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(os.path.join(path, "diffusion_pytorch_model.safetensors"))


class ReferenceImageDecoder:
    """The SGLang ``ViT2DDecoder`` + ``post_quant_conv`` in plain fp32 torch, over raw release weights."""

    def __init__(self, raw: dict[str, torch.Tensor], vit: dict) -> None:
        self.w = {k: v.float() for k, v in raw.items() if k.startswith(("decoder.", "post_quant_conv."))}
        self.heads, self.dim_head = vit["heads"], vit["dim_head"]
        self.layers = vit["num_layers"]
        self.rotary = int(self.dim_head * vit["rope_dim_ratio"])
        self.theta = vit["rope_theta"]
        self.eps = 1e-5

    def _rope(self, h: int, w: int, suffix: int) -> tuple[torch.Tensor, torch.Tensor]:
        grids = [2.0 * (torch.arange(0.5, n, dtype=torch.float32) / n) - 1.0 for n in (h, w)]
        ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).reshape(-1, 2)
        ids = torch.cat([ids, ids.new_zeros(suffix, 2)], dim=0)
        inv_freq = 1.0 / self.theta ** torch.arange(0, 1, 2 * 2 / self.rotary, dtype=torch.float32)
        angles = (2.0 * math.pi * ids[:, :, None] * inv_freq[None, None, :]).flatten(1, 2).tile(2)
        return angles.cos()[None, :, None, :], angles.sin()[None, :, None, :]  # [1, S, 1, rotary]

    def _rotate(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        rot, rest = x[..., : self.rotary], x[..., self.rotary :]
        first, second = rot.chunk(2, dim=-1)
        return torch.cat([rot * cos + torch.cat([-second, first], dim=-1) * sin, rest], dim=-1)

    def _rms(self, x: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return out * weight if weight is not None else out

    @torch.no_grad()
    def decode_tile(self, z: torch.Tensor) -> torch.Tensor:
        """``[1, C, h, w]`` latent tile -> ``[1, 3, 16h, 16w]`` pixels."""
        w = self.w
        _, c, h, wd = z.shape
        z = F.conv2d(z.float(), w["post_quant_conv.weight"], w["post_quant_conv.bias"])
        x = (
            z.permute(0, 2, 3, 1).reshape(1, h * wd, c) @ w["decoder.x_embedder.weight"].T
            + w["decoder.x_embedder.bias"]
        )
        register = w["decoder.register_tokens"]
        x = torch.cat([x, register, torch.zeros_like(x[:, :1])], dim=1)
        seq = x.shape[1]
        cos, sin = self._rope(h, wd, register.shape[1] + 1)
        for i in range(self.layers):
            p = f"decoder.transformer_blocks.{i}."
            n = self._rms(x, w[p + "norm1.weight"])
            qkv = (n @ w[p + "attn.to_qkv.weight"].T + w[p + "attn.to_qkv.bias"]).view(
                1, seq, self.heads, 3 * self.dim_head
            )
            q, k, v = qkv.chunk(3, dim=-1)
            q, k = self._rotate(self._rms(q, None), cos, sin), self._rotate(self._rms(k, None), cos, sin)
            a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
            a = a.reshape(1, seq, -1) @ w[p + "attn.to_out.weight"].T + w[p + "attn.to_out.bias"]
            x = x + a * w[p + "scale1"]
            n = self._rms(x, w[p + "norm2.weight"])
            gate, value = (n @ w[p + "ff.w1.weight"].T + w[p + "ff.w1.bias"]).chunk(2, dim=-1)
            f = (F.silu(gate) * value) @ w[p + "ff.w2.weight"].T + w[p + "ff.w2.bias"]
            x = x + f * w[p + "scale2"]
        x = F.layer_norm(x, (x.shape[-1],), w["decoder.norm_out.weight"], w["decoder.norm_out.bias"], self.eps)
        out = (x @ w["decoder.proj_out.weight"].T + w["decoder.proj_out.bias"])[:, : h * wd]
        out = out.view(1, h, wd, 3, RATIO, RATIO).permute(0, 3, 1, 4, 2, 5)
        return out.reshape(1, 3, h * RATIO, wd * RATIO)

    # --- the release's pixel-space tiling (klvae.split_tiles / blend / _assemble_tiles), verbatim semantics ---
    @staticmethod
    def split_tiles(length: int, tile: int = TILE_PX, overlap_min: int = 64):
        if tile >= length:
            return [0], [length], []
        n = math.ceil(length / tile)
        while True:
            overlaps = [overlap_min] * (n - 1)
            remaining = tile * n - sum(overlaps) - length
            if remaining < 0:
                n += 1
            else:
                break
        for i in range(remaining // RATIO):
            overlaps[i % (n - 1)] += RATIO
        starts = [0]
        for i in range(n - 1):
            starts.append(starts[-1] + tile - overlaps[i])
        return starts, [tile] * n, overlaps

    @staticmethod
    def _blend(a: torch.Tensor, b: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
        extent = min(a.shape[dim], b.shape[dim], extent)
        pos = torch.arange(extent, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = extent
        blended = a.narrow(dim, a.shape[dim] - extent, extent) * (1 - pos / extent).view(shape) + b.narrow(
            dim, 0, extent
        ) * (pos / extent).view(shape)
        return torch.cat([blended, b.narrow(dim, extent, b.shape[dim] - extent)], dim=dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """``[1, C, H, W]`` latent -> ``[1, 3, 16H, 16W]``, tiled exactly as the release decodes images."""
        ys, yl, yo = self.split_tiles(z.shape[-2] * RATIO)
        xs, xl, xo = self.split_tiles(z.shape[-1] * RATIO)
        rows = [
            [
                self.decode_tile(z[..., y // RATIO : (y + ly) // RATIO, x // RATIO : (x + lx) // RATIO])
                for x, lx in zip(xs, xl)
            ]
            for y, ly in zip(ys, yl)
        ]
        out_rows = []
        for i, row in enumerate(rows):
            cropped = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(rows[i - 1][j], tile, yo[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, xo[j - 1], dim=-1)
                if i < len(rows) - 1:
                    tile = tile[..., : -yo[i], :]
                if j < len(row) - 1:
                    tile = tile[..., : -xo[j]]
                cropped.append(tile)
            out_rows.append(torch.cat(cropped, dim=-1))
        return torch.cat(out_rows, dim=-2)


def _build(mesh_device):
    import json

    converted_dir, raw_dir = _paths()
    raw_cfg = json.loads(open(os.path.join(raw_dir, "config.json")).read())
    reference = ReferenceImageDecoder(_load(raw_dir), raw_cfg["vit_decoder_kwargs"])
    config = MiniMaxH3VaeConfig.from_pretrained(converted_dir)
    assert config.is_image, "converted vae/config.json must describe the 2D (image) decoder"
    tt_vae = MiniMaxH3Vae(config, task="image", mesh_device=mesh_device)
    tt_vae.load_state(_load(converted_dir))
    return reference, tt_vae, config


@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_image_decoder_one_tile(mesh_device):
    """One 256 px tile (16x16 latents): the ViT2D stack, 2-axis RoPE and the post_quant fold."""
    reference, tt_vae, config = _build(mesh_device)
    torch.manual_seed(0)
    z = torch.randn(1, config.latent_channels, 16, 16)
    expected = reference.decode_tile(z)
    actual = tt_vae.decode_clip(z.unsqueeze(2))[:, :, 0]
    _report("vae_one_tile", expected, actual)
    assert_quality(expected, actual.float(), pcc=0.999, relative_rmse=0.05)


def _report(tag: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
    e, a = expected.float().flatten(), actual.float().flatten()
    pcc = torch.corrcoef(torch.stack([e, a]))[0, 1].item()
    rel = ((e - a).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
    logger.info(f"RESULT {tag}: pcc={pcc:.6f} relative_rmse={rel:.5f} max_abs={float((e - a).abs().max()):.4f}")


@pytest.mark.timeout(3600)  # the fp32 CPU reference decodes 121 tiles x 24 layers at 2K
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("pixels", [pytest.param(2048, id="2K"), pytest.param(1024, id="1K")])
def test_image_decode_tiled(mesh_device, pixels):
    """A whole square canvas: tile geometry, the mesh-wide tile waves and the release's cross-fade."""
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    reference, tt_vae, config = _build(mesh_device)
    torch.manual_seed(1)
    z = torch.randn(1, config.latent_channels, pixels // RATIO, pixels // RATIO)
    expected = reference.decode(z)
    actual = tt_vae.decode_clip(z.unsqueeze(2))[:, :, 0]
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    _report(f"vae_decode_{pixels}", expected, actual)
    assert_quality(expected, actual.float(), pcc=0.999, relative_rmse=0.05)
