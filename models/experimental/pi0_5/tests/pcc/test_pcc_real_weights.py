# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 reference inference on the real pi05_libero_upstream checkpoint.

Skipped if the checkpoint isn't present locally.
"""

import os
from pathlib import Path

import pytest
import torch

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_libero_upstream"


pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def test_pi0_5_reference_sample_actions_real_weights():
    """Load real pi05_libero_upstream weights, run reference sample_actions, check shapes/finite."""
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model

    torch.manual_seed(0)

    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig.from_checkpoint(CHECKPOINT_DIR, num_denoising_steps=2)  # 2 steps to keep this cheap
    model = Pi0_5Model(cfg, loader)

    batch_size = 1
    seq_len = 16  # short prompt
    # Production pi0.5 LIBERO bs=3 — see [[pi05-siglip-bs3-production]].
    num_cameras = int(os.environ.get("PI0_NUM_CAMERAS", "2"))
    images = [torch.randn(batch_size, 3, 224, 224) for _ in range(num_cameras)]
    img_masks = [torch.ones(batch_size, dtype=torch.bool) for _ in range(num_cameras)]
    lang_tokens = torch.randint(0, 1000, (batch_size, seq_len))
    lang_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)

    with torch.no_grad():
        actions = model.sample_actions(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=None,
        )

    assert actions.shape == (batch_size, cfg.action_horizon, cfg.action_dim)
    assert torch.isfinite(actions).all(), "actions contain NaN/Inf"
