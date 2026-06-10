# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
import torch

import ttnn

from models.experimental.glm4_moe_lite.tt.tt_embedding import (
    convert_embedding_weight_to_tt,
    prefill_embed_memory_config,
    run_tt_embedding,
)
from models.experimental.glm4_moe_lite.tt.weights import load_glm_lazy_state_dict, resolve_best_effort_snapshot_dir


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
def test_tt_embedding_matches_torch_for_one_token() -> None:
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    state = load_glm_lazy_state_dict(snap, num_layers=47)

    embed_w = state["model.embed_tokens.weight"]
    # Keep this test fast by default. For a full-matrix test, set TT_EMBED_TEST_ROWS=154880.
    max_rows = int(os.environ.get("TT_EMBED_TEST_ROWS", "1024"))
    if max_rows <= 0:
        raise ValueError("TT_EMBED_TEST_ROWS must be > 0")
    embed_w = embed_w[:max_rows].contiguous()
    token_ids = torch.tensor([[0]], dtype=torch.int32)  # BOS token id is 0
    ref = torch.nn.functional.embedding(token_ids.long(), embed_w)

    # Default to a single chip for stability. Open a full mesh only when explicitly requested.
    mode = os.environ.get("TT_EMBED_TEST_MODE", "single").strip().lower()
    if mode not in {"single", "mesh"}:
        raise ValueError("TT_EMBED_TEST_MODE must be one of: single, mesh")

    if mode == "mesh":
        # Explicit mesh mode. By default we still restrict to physical device 0
        # to avoid accidentally opening a full cluster in developer environments.
        device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 1),
            physical_device_ids=[0],
            dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
        )
        close_device = ttnn.close_mesh_device
    else:
        device = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 1),
            physical_device_ids=[0],
            dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
        )
        close_device = ttnn.close_mesh_device

    try:
        tt_w = convert_embedding_weight_to_tt(device=device, embed_weight=embed_w)
        tt_out = run_tt_embedding(
            device=device,
            token_ids=token_ids,
            tt_weight=tt_w,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        out = ttnn.to_torch(tt_out)

        # We only care about the last dim matching. Layout/leading dims can vary.
        out = out.reshape(ref.shape)
        assert torch.allclose(out, ref, atol=0, rtol=0), "TT embedding mismatch for token 0"
    finally:
        close_device(device)


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
def test_tt_embedding_l1_matches_dram_for_short_seq() -> None:
    """L1 embedding output must match DRAM for shapes within the prefill L1 threshold."""
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    state = load_glm_lazy_state_dict(snap, num_layers=47)

    embed_w = state["model.embed_tokens.weight"]
    max_rows = int(os.environ.get("TT_EMBED_TEST_ROWS", "1024"))
    if max_rows <= 0:
        raise ValueError("TT_EMBED_TEST_ROWS must be > 0")
    embed_w = embed_w[:max_rows].contiguous()

    seq_len = 128
    hidden = int(embed_w.shape[1])
    token_ids = torch.arange(1, seq_len + 1, dtype=torch.int32).unsqueeze(0)
    ref = torch.nn.functional.embedding(token_ids.long(), embed_w)

    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    try:
        tt_w = convert_embedding_weight_to_tt(device=device, embed_weight=embed_w)
        dram_mc = ttnn.DRAM_MEMORY_CONFIG
        l1_mc = prefill_embed_memory_config(seq_tokens=seq_len, hidden_dim=hidden)
        assert l1_mc == ttnn.L1_MEMORY_CONFIG, "expected L1 threshold to cover seq_len=128"

        tt_dram = run_tt_embedding(device=device, token_ids=token_ids, tt_weight=tt_w, memory_config=dram_mc)
        tt_l1 = run_tt_embedding(device=device, token_ids=token_ids, tt_weight=tt_w, memory_config=l1_mc)

        out_dram = ttnn.to_torch(tt_dram).reshape(ref.shape)
        out_l1 = ttnn.to_torch(tt_l1).reshape(ref.shape)
        assert torch.allclose(out_dram, ref, atol=0, rtol=0), "DRAM embedding mismatch"
        assert torch.allclose(out_l1, ref, atol=0, rtol=0), "L1 embedding mismatch"
        assert torch.allclose(out_l1, out_dram, atol=0, rtol=0), "L1 vs DRAM embedding mismatch"

        ttnn.deallocate(tt_dram, force=False)
        ttnn.deallocate(tt_l1, force=False)
    finally:
        ttnn.close_mesh_device(device)
