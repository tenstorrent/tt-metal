# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from models.demos.wormhole.patchtst.config import PatchTSTDemoConfig
from models.demos.wormhole.patchtst.reference.hf_reference import load_reference_model
from models.demos.wormhole.patchtst.tests.helpers import compute_metrics
from models.demos.wormhole.patchtst.tt.patching import patchify


@pytest.mark.timeout(600)
def test_patchify_matches_hf_reference_patchifier():
    cfg = PatchTSTDemoConfig(task="forecast")
    reference = load_reference_model(
        task="forecast",
        checkpoint_id=cfg.checkpoint_for_task(),
        revision=cfg.checkpoint_revision_for_task(),
        config=cfg,
    )
    context_length = int(reference.config.context_length)
    channels = int(reference.config.num_input_channels)

    torch.manual_seed(0)
    past_values = torch.randn(2, context_length, channels, dtype=torch.float32)
    tt_patchified = patchify(
        past_values,
        context_length=context_length,
        patch_length=int(reference.config.patch_length),
        patch_stride=int(reference.config.patch_stride),
    )
    with torch.no_grad():
        ref_patchified = reference.model.model.patchifier(past_values)

    parity = compute_metrics(tt_patchified, ref_patchified)
    assert parity["mse"] <= 1e-8
    assert parity["mae"] <= 1e-6
    assert parity["correlation"] >= 0.999999
