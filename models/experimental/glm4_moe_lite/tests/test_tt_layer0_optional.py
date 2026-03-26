# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest

import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

from models.demos.glm4_moe_lite.tt.layer0_tt import run_layer0_prefill_tt
from models.demos.glm4_moe_lite.tt.reference_layer0 import run_layer0_reference
from models.demos.glm4_moe_lite.tt.weights import find_missing_shards, resolve_best_effort_snapshot_dir


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
    reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (loads large embedding weights).",
)
def test_layer0_prefill_matches_reference_for_short_prompt() -> None:
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    # CPU oracle
    ref = run_layer0_reference(snap, "Hello.")

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    try:
        tt = run_layer0_prefill_tt(
            device=mesh_device,
            snapshot_dir=Path(snap),
            input_ids=ref.input_ids,
            cache_dir=Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/layer0_tt_cache")),
            seq_pad_multiple=128,
        )
    finally:
        ttnn.close_mesh_device(mesh_device)

    # Embedding should be exact (it's just a gather), but use PCC to be safe.
    ok, msg = comp_pcc(tt.x_embed, ref.x_embed, pcc=0.9999)
    assert ok, f"x_embed mismatch: {msg}"

    ok, msg = comp_pcc(tt.x_attn_out, ref.x_attn_out, pcc=0.99)
    assert ok, f"x_attn_out mismatch: {msg}"

    ok, msg = comp_pcc(tt.x_mlp_out, ref.x_mlp_out, pcc=0.99)
    assert ok, f"x_mlp_out mismatch: {msg}"
