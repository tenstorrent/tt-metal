# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

from models.experimental.glm4_moe_lite.tt.layer0_tt import (
    debug_flash_mla_decode_attention_unpaged_vs_paged_tt,
    debug_fill_kvpe_cache_for_one_step_tt,
    debug_q_for_decode_prefill_vs_single_token_tt,
    run_layer0_decode_one_step_tt,
)
from models.experimental.glm4_moe_lite.tt.reference_layer0 import (
    run_layer0_reference,
    run_layer0_reference_from_input_ids,
)
from models.experimental.glm4_moe_lite.tt.weights import find_missing_shards, resolve_best_effort_snapshot_dir


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
    reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (loads large embedding weights).",
)
def test_layer0_decode_one_step_matches_reference_last_token() -> None:
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    # CPU oracle for prefix.
    ref_prefix = run_layer0_reference(snap, "Hello.")

    # Pick a deterministic next token id (must be within vocab).
    next_token_id = 1
    next_token = torch.tensor([[next_token_id]], dtype=ref_prefix.input_ids.dtype)
    extended_ids = torch.cat([ref_prefix.input_ids, next_token], dim=1)

    # CPU oracle for the extended prompt: compare only the last token's layer0 output.
    ref_extended = run_layer0_reference_from_input_ids(snap, extended_ids)
    ref_last = ref_extended.x_mlp_out[:, -1, :]  # [1, hidden]

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    try:
        kvpe_ref, kvpe_from_cache = debug_fill_kvpe_cache_for_one_step_tt(
            device=mesh_device,
            snapshot_dir=Path(snap),
            prefix_input_ids=ref_prefix.input_ids,
            next_token_id=next_token_id,
            batch=1,
            block_size=64,
            cache_dir=Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/layer0_tt_cache")),
        )
        ok, msg = comp_pcc(kvpe_ref, kvpe_from_cache, pcc=0.9999)
        assert ok, f"paged_fill_cache KVPE mismatch (debug): {msg}"

        q_prefill, q_single = debug_q_for_decode_prefill_vs_single_token_tt(
            device=mesh_device,
            snapshot_dir=Path(snap),
            prefix_input_ids=ref_prefix.input_ids,
            next_token_id=next_token_id,
            block_size=64,
            cache_dir=Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/layer0_tt_cache")),
        )
        ok, msg = comp_pcc(q_prefill, q_single, pcc=0.9999)
        assert ok, f"Decode Q mismatch prefill-vs-single-token (debug): {msg}"

        attn_unpaged, attn_paged = debug_flash_mla_decode_attention_unpaged_vs_paged_tt(
            device=mesh_device,
            snapshot_dir=Path(snap),
            prefix_input_ids=ref_prefix.input_ids,
            next_token_id=next_token_id,
            block_size=64,
            cache_dir=Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/layer0_tt_cache")),
        )
        ok, msg = comp_pcc(attn_unpaged, attn_paged, pcc=0.99)
        assert ok, f"FlashMLA decode mismatch unpaged-vs-paged (debug): {msg}"

        tt_last = run_layer0_decode_one_step_tt(
            device=mesh_device,
            snapshot_dir=Path(snap),
            prefix_input_ids=ref_prefix.input_ids,
            next_token_id=next_token_id,
            batch=1,
            block_size=64,
            cache_dir=Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/layer0_tt_cache")),
        )
    finally:
        ttnn.close_mesh_device(mesh_device)

    ok, msg = comp_pcc(tt_last, ref_last, pcc=0.99)
    assert ok, f"layer0 decode mismatch vs reference last token: {msg}"
