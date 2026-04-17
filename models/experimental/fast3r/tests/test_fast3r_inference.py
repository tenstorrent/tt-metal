"""Fast3R benchmark harness.

Prints `inference_speed`, `accuracy`, `peak_dram` on stdout so the autoresearch loop
can `grep` for them after a run.

Current stage: a single encoder block with on-device 2D RoPE, verified against the
torch reference encoder block 0.
"""
from __future__ import annotations

import os
import time
from typing import Dict

import pytest
import torch
from safetensors import safe_open

import ttnn

from models.experimental.fast3r.reference.model import Attention, Fast3RConfig, Mlp
from models.experimental.fast3r.reference.rope import build_rope2d_cos_sin
from models.experimental.fast3r.tt.encoder import TtEncoderBlock, _build_permuted_rope_cache


WEIGHTS = os.environ.get(
    "FAST3R_WEIGHTS",
    "/home/ttuser/.cache/huggingface/hub/models--jedyang97--Fast3R_ViT_Large_512/"
    "snapshots/a2c770b768ceb3a53c36c4f7a3619db0413dc3a1/model.safetensors",
)


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return float(((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12)).item())


def _load_block_weights(block_idx: int, branch: str) -> Dict[str, torch.Tensor]:
    with safe_open(WEIGHTS, framework="pt") as f:
        prefix = f"{branch}.{block_idx}."
        keys = [
            "norm1.weight", "norm1.bias",
            "norm2.weight", "norm2.bias",
            "attn.qkv.weight", "attn.qkv.bias",
            "attn.proj.weight", "attn.proj.bias",
            "mlp.fc1.weight", "mlp.fc1.bias",
            "mlp.fc2.weight", "mlp.fc2.bias",
        ]
        return {k: f.get_tensor(prefix + k) for k in keys}


def _torch_encoder_block(cfg: Fast3RConfig, sd: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    norm1 = torch.nn.LayerNorm(cfg.embed_dim, eps=1e-6)
    norm1.weight.data.copy_(sd["norm1.weight"]); norm1.bias.data.copy_(sd["norm1.bias"])
    norm2 = torch.nn.LayerNorm(cfg.embed_dim, eps=1e-6)
    norm2.weight.data.copy_(sd["norm2.weight"]); norm2.bias.data.copy_(sd["norm2.bias"])
    attn = Attention(cfg).eval()
    attn.qkv.weight.data.copy_(sd["attn.qkv.weight"]); attn.qkv.bias.data.copy_(sd["attn.qkv.bias"])
    attn.proj.weight.data.copy_(sd["attn.proj.weight"]); attn.proj.bias.data.copy_(sd["attn.proj.bias"])
    mlp = Mlp(cfg).eval()
    mlp.fc1.weight.data.copy_(sd["mlp.fc1.weight"]); mlp.fc1.bias.data.copy_(sd["mlp.fc1.bias"])
    mlp.fc2.weight.data.copy_(sd["mlp.fc2.weight"]); mlp.fc2.bias.data.copy_(sd["mlp.fc2.bias"])
    grid = cfg.img_size // cfg.patch_size
    cos, sin = build_rope2d_cos_sin(grid, grid, cfg.embed_dim // cfg.num_heads, base=cfg.rope_base)
    with torch.inference_mode():
        x = x + attn(norm1(x), cos=cos, sin=sin)
        x = x + mlp(norm2(x))
    return x


def _peak_dram_bytes(device) -> int:
    try:
        return int(device.get_memory_statistics(ttnn.BufferType.DRAM)["peak_usage_bytes"])
    except Exception:
        return 0


def test_fast3r_encoder_block_pcc_and_speed(device):
    cfg = Fast3RConfig()
    sd = _load_block_weights(0, branch="encoder.enc_blocks")

    N = (cfg.img_size // cfg.patch_size) ** 2  # 1024
    g = torch.Generator().manual_seed(0)
    x_torch = torch.randn(1, N, cfg.embed_dim, generator=g)

    y_torch = _torch_encoder_block(cfg, sd, x_torch)

    cos_cache, sin_cache = _build_permuted_rope_cache(device, cfg)
    block = TtEncoderBlock(device, cfg, sd, cos_cache, sin_cache)
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    y_warm = block(x_tt)
    ttnn.synchronize_device(device)
    y_warm.deallocate(True)

    iters = int(os.environ.get("FAST3R_BENCH_ITERS", "20"))
    t0 = time.perf_counter()
    for _ in range(iters):
        y_tt = block(x_tt)
    ttnn.synchronize_device(device)
    dt = (time.perf_counter() - t0) / iters
    fps = 1.0 / dt if dt > 0 else 0.0

    y_tt_torch = ttnn.to_torch(y_tt).squeeze(0)
    y_tt.deallocate(True)

    acc = _pcc(y_torch, y_tt_torch) * 100.0

    print(f"inference_speed: {fps:.4f} enc_block_fwd/sec")
    print(f"accuracy: {acc:.2f}")
    print(f"peak_dram: {_peak_dram_bytes(device)}")

    assert acc > 99.0, f"PCC too low: {acc:.3f}%"
