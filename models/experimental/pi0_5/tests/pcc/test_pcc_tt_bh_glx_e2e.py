# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC test for the BH Galaxy host-bounce pipeline.

Compares Pi0_5GLXPipeline.sample_actions vs the PyTorch reference
Pi0_5Model.sample_actions with matched seeds.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_galaxy_mesh


CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
SEED = 42


def _compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return (cov / (s1 * s2)).item()


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"checkpoint not found at {CHECKPOINT_DIR}",
)
def test_pi0_5_glx_pipeline_e2e_pcc():
    """Full sample_actions pipeline vs torch Pi0_5Model. Target PCC ≥ 0.95."""
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model as TorchPi0_5Model
    from models.experimental.pi0_5.tt.tt_bh_glx.pipeline import Pi0_5GLXPipeline

    num_steps = int(os.environ.get("PI05_NUM_DENOISE_STEPS", "5"))
    num_cams = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
    cfg = Pi0_5ModelConfig(
        action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR),
        num_denoising_steps=num_steps,
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))

    img_h = img_w = cfg.siglip_config.image_size
    lang_len = 256

    # Inputs (matched across both backends).
    torch.manual_seed(SEED)
    images = [torch.randn(1, 3, img_h, img_w) for _ in range(num_cams)]
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(num_cams)]
    lang_tokens = torch.randint(0, 256000, (1, lang_len), dtype=torch.int32)
    lang_masks = torch.ones(1, lang_len, dtype=torch.bool)

    # Torch reference. Set the same seed RIGHT BEFORE sample_actions to match
    # the pipeline's internal noise seed (which is set fresh in sample_actions).
    print(f"\n📋 Torch reference (action_horizon={cfg.action_horizon}, steps={num_steps}, cams={num_cams})")
    torch.manual_seed(SEED)
    ref = TorchPi0_5Model(cfg, loader)
    with torch.no_grad():
        torch.manual_seed(SEED + 1)
        ref_actions = ref.sample_actions(
            images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks, state=None
        )

    print(f"   ref_actions shape: {tuple(ref_actions.shape)}")

    # TTNN sliced pipeline. Set the matching seed for noise inside sample_actions.
    with open_galaxy_mesh(l1_small_size=24576) as h:
        pipe = Pi0_5GLXPipeline(cfg, loader.categorized_weights, h)
        torch.manual_seed(SEED + 1)
        actions, timings = pipe.sample_actions(
            images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks
        )

    print(f"   pipe_actions shape: {tuple(actions.shape)}")
    print(
        f"   timings: vision={timings.vision_ms:.1f}ms  v→p={timings.transport_v2p_ms:.1f}ms  "
        f"prefill={timings.prefill_ms:.1f}ms  kv_mig={timings.kv_migration_ms:.1f}ms  "
        f"denoise_total={timings.denoise_total_ms:.1f}ms  total={timings.total_ms:.1f}ms"
    )

    assert actions.shape == ref_actions.shape, f"shape mismatch: {tuple(actions.shape)} vs {tuple(ref_actions.shape)}"
    pcc = _compute_pcc(ref_actions, actions)
    print(f"\n✅ End-to-end PCC vs torch: {pcc:.6f}")
    assert pcc >= 0.95, f"PCC {pcc:.6f} < 0.95"
