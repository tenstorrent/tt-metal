"""Fast3R benchmark harness.

Prints `inference_speed`, `accuracy`, `peak_dram` on stdout so the autoresearch loop
can `grep` for them after a run.

Current stage: end-to-end trunk (encoder + decoder) benchmarked with trace capture
to eliminate host-side op dispatch overhead. PCC is computed against the torch
reference trunk.
"""
from __future__ import annotations

import os
import time

import pytest
import torch
from safetensors import safe_open

import ttnn

from models.experimental.fast3r.reference.model import Fast3RConfig, PatchEmbed
from models.experimental.fast3r.reference.weights import load_fast3r
from models.experimental.fast3r.tt.model import TtFast3RTrunk


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


def _peak_dram_bytes(device) -> int:
    try:
        return int(device.get_memory_statistics(ttnn.BufferType.DRAM)["peak_usage_bytes"])
    except Exception:
        return 0


def _cpu_patch_embed(cfg: Fast3RConfig, img: torch.Tensor) -> torch.Tensor:
    pe = PatchEmbed(cfg).eval()
    with safe_open(WEIGHTS, framework="pt") as f:
        pe.proj.weight.data.copy_(f.get_tensor("encoder.patch_embed.proj.weight"))
        pe.proj.bias.data.copy_(f.get_tensor("encoder.patch_embed.proj.bias"))
    with torch.inference_mode():
        return pe(img)


def _torch_trunk_output(cfg: Fast3RConfig, img: torch.Tensor) -> torch.Tensor:
    model = load_fast3r(device="cpu")
    with torch.inference_mode():
        return model(img)


@pytest.mark.parametrize("device_params", [{"trace_region_size": 30_000_000}], indirect=True)
def test_fast3r_trunk_pcc_and_speed(device):
    cfg = Fast3RConfig()
    g = torch.Generator().manual_seed(0)
    img = torch.randn(1, 3, cfg.img_size, cfg.img_size, generator=g)

    y_torch = _torch_trunk_output(cfg, img)

    tokens = _cpu_patch_embed(cfg, img)
    trunk = TtFast3RTrunk(device, cfg, WEIGHTS)
    x_tt = ttnn.from_torch(tokens.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Warm-up compiles kernels; program cache lets the trace reuse them.
    y_warm = trunk(x_tt)
    ttnn.synchronize_device(device)
    y_warm.deallocate(True)

    # Capture trace. x_tt stays at the same address across iterations so we can replay.
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    y_out = trunk(x_tt)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)

    iters = int(os.environ.get("FAST3R_BENCH_ITERS", "10"))
    t0 = time.perf_counter()
    for _ in range(iters):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    dt = (time.perf_counter() - t0) / iters
    fps = 1.0 / dt if dt > 0 else 0.0

    y_tt_torch = ttnn.to_torch(y_out).squeeze(0)
    ttnn.release_trace(device, trace_id)
    y_out.deallocate(True)

    acc = _pcc(y_torch, y_tt_torch) * 100.0

    print(f"inference_speed: {fps:.4f} trunk_fwd/sec")
    print(f"accuracy: {acc:.2f}")
    print(f"peak_dram: {_peak_dram_bytes(device)}")

    assert acc > 99.0, f"PCC too low: {acc:.3f}%"
