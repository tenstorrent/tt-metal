# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch

from models.experimental.glm4_moe_lite.tt.reference_moe import run_layer_moe_reference_from_hidden_states
from models.experimental.glm4_moe_lite.tt.weights import find_missing_shards, resolve_best_effort_snapshot_dir


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
    reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (touches routed expert weights).",
)
def test_layer1_moe_reference_runs_and_is_finite() -> None:
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    torch.manual_seed(0)
    tokens = 32
    hidden = 2048
    # Match TT bring-up behavior: hidden states are bf16 in practice.
    x = torch.randn((tokens, hidden), dtype=torch.float32).to(torch.bfloat16).to(torch.float32)

    out = run_layer_moe_reference_from_hidden_states(snap, layer_idx=1, hidden_states=x)
    assert out.router_logits.shape == (tokens, 64)
    assert out.topk_indices.shape == (tokens, 4)
    assert out.topk_weights.shape == (tokens, 4)
    assert out.shared_out.shape == (tokens, hidden)
    assert out.routed_out.shape == (tokens, hidden)
    assert out.moe_out.shape == (tokens, hidden)
    assert torch.isfinite(out.moe_out).all()
