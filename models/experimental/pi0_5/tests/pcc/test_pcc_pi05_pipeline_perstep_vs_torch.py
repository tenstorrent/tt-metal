# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-step / single-stage drill-down PCC for the tt_pipeline port (plan §11.2).

  * Single-stage drill-down: ``build_single_stage_reference`` (one chip, all 18 layers,
    static-KV) vs the torch golden velocity for one step -- isolates expert-block correctness
    from the multi-mesh socket plumbing AND validates the fill_cache shim path.

Skips cleanly without hardware or a checkpoint. (Synthetic-KV PCC bar follows the project's
0.95 e2e convention; the §11 0.99 target is the real-KV path.)
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

ttnn = pytest.importorskip("ttnn")

CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/ttuser/salnahari/pi05_weights/pi05_base"))
SEED = 42
PCC_THRESHOLD = float(os.environ.get("PI05_PIPELINE_PCC", "0.95"))


def _compute_pcc(a, b):
    t1, t2 = a.flatten().float(), b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return (cov / (s1 * s2)).item()


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(), reason=f"checkpoint not found at {CHECKPOINT_DIR}"
)
def test_single_stage_drilldown_velocity():
    """One-chip all-18-layer static-KV stage velocity PCC >= 0.95 vs torch golden (one step)."""
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig, SuffixConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone as TorchBackbone

    # Single-stage reference uses the DRAM-weight parent block: pinning all 18 layers'
    # projection weights into L1 (the multi-stage <=5-layer perf optimization in
    # TTNNPi05DenoiseExpertBlock) overflows a single chip's L1. The expert-block COMPUTE is
    # identical; only weight residency differs.
    from models.experimental.pi0_5.tt.tt_pipeline.modeling.gemma import TTNNPi05AdaRMSGemmaBlock
    from models.experimental.pi0_5.tt.tt_pipeline.denoise_pipeline import build_single_stage_reference
    from models.experimental.pi0_5.tt.tt_pipeline.stage_denoise import _build_phantom_mask_and_offset
    from models.experimental.pi0_5.tt.tt_pipeline.weight_adapt import (
        expert_reference_blocks,
        final_mod,
        suffix_reference,
    )

    cfg = Pi0_5ModelConfig(action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR), num_denoising_steps=5)
    weights = Pi0_5WeightLoader(str(CHECKPOINT_DIR)).categorized_weights
    ec = cfg.expert_config
    B = 1
    ah = cfg.action_horizon
    ah_pad = ((ah + 31) // 32) * 32
    suffix_len = ah_pad
    prefix_len = 288
    t_scalar = 1.0

    torch.manual_seed(SEED)
    x_t = torch.zeros(B, ah_pad, cfg.action_dim)
    x_t[:, :ah, :] = torch.randn(B, ah, cfg.action_dim)
    prefix_kv_torch = [
        (
            torch.randn(B, ec.num_kv_heads, prefix_len, ec.head_dim) * 0.5,
            torch.randn(B, ec.num_kv_heads, prefix_len, ec.head_dim) * 0.5,
        )
        for _ in range(ec.depth)
    ]
    sc = SuffixConfig(action_dim=cfg.action_dim, action_horizon=ah, expert_width=ec.width, pi05=True)
    suffix_ref = suffix_reference(weights["pi0_projections"], sc)
    cond_torch = suffix_ref.embed_timestep_adarms(torch.tensor([t_scalar]))

    # torch golden velocity for this step
    backbone = TorchBackbone(cfg, weights)
    with torch.no_grad():
        h = suffix_ref.embed_actions(x_t)
        h, _ = backbone.forward_expert(h, adarms_cond=cond_torch, past_key_values=prefix_kv_torch, use_cache=False)
        v_torch = suffix_ref.project_output(h)[:, :ah, :]

    ref_blocks = expert_reference_blocks(weights["action_expert"], ec, depth=ec.depth)
    final_w, final_b = final_mod(weights["action_expert"])
    mask, offset = _build_phantom_mask_and_offset(prefix_len, suffix_len, ah)

    if ttnn.get_num_devices() < 1:
        pytest.skip("need >=1 chip")
    submesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576)
    try:
        stage = build_single_stage_reference(
            ref_blocks,
            final_w,
            final_b,
            suffix_ref,
            cfg,
            sc,
            submesh,
            adarms_cond_torch=cond_torch,
            prefix_kv_cache=prefix_kv_torch,
            prefix_len=prefix_len,
            suffix_len=suffix_len,
            attention_mask_torch=mask,
            position_offset=offset,
            block_cls=TTNNPi05AdaRMSGemmaBlock,
            use_concat_kv=False,  # static-KV path (exercises the fill_cache shim)
        )
        x_dev = ttnn.from_torch(x_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=submesh)
        out = stage.forward(x_dev)
        v_dev = ttnn.to_torch(out)[:, :ah, :]
    finally:
        ttnn.close_mesh_device(submesh)

    pcc = _compute_pcc(v_torch, v_dev)
    print(f"\nsingle-stage drill-down velocity PCC vs torch: {pcc:.6f}")
    assert pcc >= PCC_THRESHOLD, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-xvs"]))
