# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch

from models.demos.glm4_moe_lite.tt.reference_layer0 import run_layer0_reference
from models.demos.glm4_moe_lite.tt.weights import resolve_best_effort_snapshot_dir


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
    reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (loads large embedding weights).",
)
def test_layer0_reference_runs_and_is_finite() -> None:
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    out = run_layer0_reference(snap, "Hello")

    assert out.input_ids.ndim == 2
    assert out.x_embed.shape[:-1] == out.input_ids.shape
    assert out.x_attn_out.shape == out.x_embed.shape
    assert out.x_mlp_out.shape == out.x_embed.shape
    assert torch.isfinite(out.x_mlp_out).all()
