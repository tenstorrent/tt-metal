# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare StagePrefillTP4 K/V on (1,8) parent vs (1,8) row submesh of a (4,8) Galaxy.

The new 16-chip decode_all pipeline carves a (1,8) row from a (4,8) Galaxy and runs
TP=8 prefill in the same orientation as the current production (1,8) parent. This
test verifies that the carved row preserves the known-good prefill numerics before
LIBERO smoke under --backend ttnn_16_decode.

If K-PCC and V-PCC stay >= 0.999 across all 18 expert layers, the row-submesh layout is
safe to reuse the existing prefill code. If any layer drops below threshold, the
16-chip pipeline cannot simply carve the 1x8 row from the larger parent.

Run:
  pytest models/experimental/pi0_5/tests/pcc/test_pcc_prefill_orientation.py -s
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

sys.path.insert(0, str(Path(__file__).parent))
from _fabric_harness import close_parent, open_parent_with_retry, reset_board  # noqa: E402

CKPT = "/home/tt-admin/pi05_cache/pi05_libero_upstream"
PREFIX_LEN = 1024
SEED = 0xDEC0DE
_TRACE_REGION = 134_217_728


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    am, bm = a.mean(), b.mean()
    num = ((a - am) * (b - bm)).sum()
    den = (((a - am).pow(2).sum() * (b - bm).pow(2).sum()).sqrt()).clamp(min=1e-30)
    return float(num / den)


def _chip0(t, mesh) -> torch.Tensor:
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()[:1]


def _run_prefill(prefill_mesh, cfg, weights, prefix_bf16):
    from models.experimental.pi0_5.tt.tt_bh_glx.stage_prefill_tp4 import StagePrefillTP4

    prefill = StagePrefillTP4(cfg, weights, prefill_mesh)
    pin = ttnn.from_torch(
        prefix_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=prefill_mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    _, per_layer_kv = prefill.run(pin, attention_mask=None)
    kv_host = [(_chip0(k, prefill_mesh), _chip0(v, prefill_mesh)) for k, v in per_layer_kv]
    for k, v in per_layer_kv:
        ttnn.deallocate(k)
        ttnn.deallocate(v)
    ttnn.deallocate(pin)
    del prefill
    return kv_host


def _open_48_parent(retries=2, l1_small_size=24576, trace_region_size=_TRACE_REGION):
    """Open a (4,8) FABRIC_1D parent mesh; reset+retry on ethernet-core stalls."""
    last = None
    for _ in range(retries + 1):
        try:
            num_devices = ttnn.get_num_devices()
            if num_devices < 32:
                pytest.skip(f"need >=32 chips, have {num_devices}")
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            return ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(4, 8),
                l1_small_size=l1_small_size,
                trace_region_size=trace_region_size,
            )
        except Exception as e:
            last = e
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass
            reset_board()
    raise last


def test_prefill_orientation_pcc():
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader

    cfg = Pi0_5ModelConfig(action_horizon=10)
    loader = Pi0_5WeightLoader(CKPT)
    weights = loader.categorized_weights

    H = cfg.vlm_config.width
    torch.manual_seed(SEED)
    prefix_t = torch.randn(1, PREFIX_LEN, H, dtype=torch.float32) * 0.05
    prefix_bf16 = prefix_t.to(torch.bfloat16)

    # ── Phase 1: (1, 8) parent (current production shape) ───────────────
    print("\nPhase 1: (1,8) parent prefill")
    parent18 = open_parent_with_retry(8)
    try:
        kv_18 = _run_prefill(parent18, cfg, weights, prefix_bf16)
    finally:
        close_parent(parent18)
    print(f"  (1,8) done: {len(kv_18)} layers, settling 3 s before Phase 2")
    time.sleep(3)

    # ── Phase 2: (1, 8) submesh of (4, 8) parent ────────────────────────
    print("Phase 2: (4,8) parent -> (1,8) row-0 submesh prefill")
    parent48 = _open_48_parent()
    try:
        submesh = parent48.create_submesh(ttnn.MeshShape(1, 8), ttnn.MeshCoordinate(0, 0))
        try:
            kv_18_sub = _run_prefill(submesh, cfg, weights, prefix_bf16)
        finally:
            ttnn.close_mesh_device(submesh)
    finally:
        close_parent(parent48)
    print(f"  carved (1,8) done: {len(kv_18_sub)} layers")

    # ── Compare per-layer ───────────────────────────────────────────────
    print("\nLayer-by-layer K/V PCC ((1,8) parent vs carved (1,8)):")
    print(f"{'L':>3}  {'K-PCC':>9}  {'K-maxΔ':>10}  {'V-PCC':>9}  {'V-maxΔ':>10}")
    worst_kp, worst_vp = 1.0, 1.0
    for i, ((k18, v18), (k18_sub, v18_sub)) in enumerate(zip(kv_18, kv_18_sub)):
        assert k18.shape == k18_sub.shape, f"L{i}: K shape mismatch {k18.shape} vs {k18_sub.shape}"
        assert v18.shape == v18_sub.shape, f"L{i}: V shape mismatch {v18.shape} vs {v18_sub.shape}"
        kp = _pcc(k18, k18_sub)
        vp = _pcc(v18, v18_sub)
        kd = (k18 - k18_sub).abs().max().item()
        vd = (v18 - v18_sub).abs().max().item()
        print(f"{i:>3}  {kp:>9.5f}  {kd:>10.3e}  {vp:>9.5f}  {vd:>10.3e}")
        worst_kp = min(worst_kp, kp)
        worst_vp = min(worst_vp, vp)
    print(f"\nWorst K-PCC: {worst_kp:.5f}    Worst V-PCC: {worst_vp:.5f}")
    assert worst_kp > 0.999, f"K PCC degraded across orientations: worst {worst_kp:.5f}"
    assert worst_vp > 0.999, f"V PCC degraded across orientations: worst {worst_vp:.5f}"
