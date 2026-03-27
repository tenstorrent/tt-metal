# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn

from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

from models.experimental.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.experimental.glm4_moe_lite.tt.layer_weights import convert_decoder_layer_weights
from models.experimental.glm4_moe_lite.tt.moe_tt import create_moe_runtime, moe_sparse_experts_forward_tt
from models.experimental.glm4_moe_lite.tt.reference_moe import run_layer_moe_reference_from_hidden_states
from models.experimental.glm4_moe_lite.tt.weights import (
    find_missing_shards,
    load_glm_lazy_state_dict,
    resolve_best_effort_snapshot_dir,
)


def _load_hparams(snapshot_dir: Path) -> Glm4MoeLiteHParams:
    cfg = json.loads((Path(snapshot_dir) / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()
    return hparams


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
    reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (loads routed expert weights).",
)
def test_layer1_routed_experts_matches_reference_given_reference_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Validate expert execution + dispatch/combine math independent of router precision.

    We compute top-k routing on CPU (HF semantics) and feed those indices/weights into
    the TT sparse experts path, then compare routed expert output.
    """
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    monkeypatch.setenv("GLM4_MOE_LITE_EXPERTS_TT_DTYPE", "bf16")

    hparams = _load_hparams(Path(snap))
    layer_idx = 1
    tokens = 32  # must be a multiple of 32 for current sparse kernel path
    hidden = int(hparams.hidden_size)

    torch.manual_seed(0)
    x = torch.randn((tokens, hidden), dtype=torch.float32).to(torch.bfloat16).to(torch.float32)

    # CPU oracle (routing + experts + shared)
    ref = run_layer_moe_reference_from_hidden_states(Path(snap), layer_idx=layer_idx, hidden_states=x)

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    try:
        # Load weights and build runtime.
        state = load_glm_lazy_state_dict(Path(snap), num_layers=int(hparams.num_hidden_layers))
        w = convert_decoder_layer_weights(
            device=mesh_device,
            state=state,
            layer_idx=layer_idx,
            hparams=hparams,
            cache_dir=Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/moe_layer1_tt_cache")),
            enable_moe=True,
        )
        assert w.moe is not None, "Expected routed MoE weights for layer 1"

        rt = create_moe_runtime(device=mesh_device, hparams=hparams)

        # TT inputs.
        x_tt = ttnn.from_torch(
            x.to(torch.bfloat16).view(1, 1, tokens, hidden),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        topk_idx = ref.topk_indices.to(dtype=torch.int16).view(1, 1, tokens, -1)
        topk_w = ref.topk_weights.to(dtype=torch.bfloat16).view(1, 1, tokens, -1)

        topk_idx_tt_rm = ttnn.from_torch(
            topk_idx,
            device=mesh_device,
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        topk_w_tt_rm = ttnn.from_torch(
            topk_w,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        topk_idx_tt = ttnn.to_layout(topk_idx_tt_rm, ttnn.TILE_LAYOUT)
        topk_w_tt = ttnn.to_layout(topk_w_tt_rm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(topk_idx_tt_rm)
        ttnn.deallocate(topk_w_tt_rm)

        routed_tt = moe_sparse_experts_forward_tt(
            device=mesh_device,
            hidden_states=x_tt,
            topk_expert_indices=topk_idx_tt,
            topk_expert_weights=topk_w_tt,
            moe_w=w.moe,
            rt=rt,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Readback.
        routed_dev = ttnn.get_device_tensors(routed_tt)[0]
        routed = ttnn.to_torch(routed_dev).reshape(tokens, hidden).to(dtype=torch.float32).cpu()
    finally:
        # Best-effort cleanup.
        try:
            ttnn.deallocate(routed_tt)
        except Exception:
            pass
        try:
            ttnn.deallocate(topk_idx_tt)
            ttnn.deallocate(topk_w_tt)
            ttnn.deallocate(x_tt)
        except Exception:
            pass
        ttnn.close_mesh_device(mesh_device)

    ok, msg = comp_pcc(routed, ref.routed_out, pcc=0.98)
    assert ok, f"layer1 routed experts output mismatch vs reference: {msg}"
