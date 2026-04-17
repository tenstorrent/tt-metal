"""Fast3R benchmark harness.

Prints `inference_speed`, `accuracy`, `peak_dram` on stdout so the autoresearch loop
can `grep` for them after a run.

Current stage: tt-nn (LayerNorm + MLP) fragment of one encoder block.
Subsequent iterations will add attention, stack N blocks, add patch_embed, etc.
"""
from __future__ import annotations

import os
import time
from typing import Dict

import pytest
import torch
from safetensors import safe_open

import ttnn

from models.experimental.fast3r.reference.model import Fast3RConfig, Mlp
from models.experimental.fast3r.tt.block import TtNormMlp


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


def _load_block_weights(block_idx: int = 0, branch: str = "encoder.enc_blocks") -> Dict[str, torch.Tensor]:
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


def _peak_dram_bytes(device) -> int:
    try:
        return int(device.get_memory_statistics(ttnn.BufferType.DRAM)["peak_usage_bytes"])
    except Exception:
        return 0


def _torch_norm_mlp(cfg: Fast3RConfig, sd: Dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    mlp = Mlp(cfg).eval()
    mlp.fc1.weight.data.copy_(sd["mlp.fc1.weight"])
    mlp.fc1.bias.data.copy_(sd["mlp.fc1.bias"])
    mlp.fc2.weight.data.copy_(sd["mlp.fc2.weight"])
    mlp.fc2.bias.data.copy_(sd["mlp.fc2.bias"])
    norm = torch.nn.LayerNorm(cfg.embed_dim, eps=1e-6)
    norm.weight.data.copy_(sd["norm2.weight"])
    norm.bias.data.copy_(sd["norm2.bias"])
    with torch.inference_mode():
        return mlp(norm(x))


def test_fast3r_norm_mlp_pcc_and_speed(device):
    cfg = Fast3RConfig()
    sd = _load_block_weights(0)

    N = (cfg.img_size // cfg.patch_size) ** 2  # 1024
    g = torch.Generator().manual_seed(0)
    x_torch = torch.randn(1, N, cfg.embed_dim, generator=g)

    y_torch = _torch_norm_mlp(cfg, sd, x_torch)

    tt_mod = TtNormMlp(
        device,
        sd["norm2.weight"], sd["norm2.bias"],
        sd["mlp.fc1.weight"], sd["mlp.fc1.bias"],
        sd["mlp.fc2.weight"], sd["mlp.fc2.bias"],
    )
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    # Warm-up (program cache + kernel JIT)
    y_warm = tt_mod(x_tt)
    ttnn.synchronize_device(device)
    y_warm.deallocate(True)

    iters = int(os.environ.get("FAST3R_BENCH_ITERS", "20"))
    t0 = time.perf_counter()
    for _ in range(iters):
        y_tt = tt_mod(x_tt)
    ttnn.synchronize_device(device)
    dt = (time.perf_counter() - t0) / iters
    fps = 1.0 / dt if dt > 0 else 0.0

    y_tt_torch = ttnn.to_torch(y_tt).squeeze(0)
    y_tt.deallocate(True)

    acc = _pcc(y_torch, y_tt_torch) * 100.0

    print(f"inference_speed: {fps:.4f} norm_mlp_fwd/sec")
    print(f"accuracy: {acc:.2f}")
    print(f"peak_dram: {_peak_dram_bytes(device)}")

    assert acc > 99.0, f"PCC too low: {acc:.3f}%"
