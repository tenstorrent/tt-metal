# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0.5 reference inference on the real lerobot/pi05_base checkpoint.

Skipped if the checkpoint isn't present locally.
"""

from pathlib import Path

import pytest
import torch

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "weights" / "pi05_base"


pytestmark = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)


def test_pi0_5_reference_sample_actions_real_weights():
    """Load real pi05_base weights, run reference sample_actions, check shapes/finite."""
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model

    torch.manual_seed(0)

    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    cfg = Pi0_5ModelConfig(num_denoising_steps=2)  # 2 steps to keep this cheap
    model = Pi0_5Model(cfg, loader)

    batch_size = 1
    seq_len = 16  # short prompt
    # One 224x224 image, 1 valid mask, language tokens, no continuous state.
    image = torch.randn(batch_size, 3, 224, 224)
    img_mask = torch.ones(batch_size, dtype=torch.bool)
    lang_tokens = torch.randint(0, 1000, (batch_size, seq_len))
    lang_masks = torch.ones(batch_size, seq_len, dtype=torch.bool)

    with torch.no_grad():
        actions = model.sample_actions(
            images=[image],
            img_masks=[img_mask],
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=None,
        )

    assert actions.shape == (batch_size, cfg.action_horizon, cfg.action_dim)
    assert torch.isfinite(actions).all(), "actions contain NaN/Inf"
