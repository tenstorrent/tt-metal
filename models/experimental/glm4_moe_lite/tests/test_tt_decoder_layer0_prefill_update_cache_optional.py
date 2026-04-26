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
from models.experimental.glm4_moe_lite.tt.decoder_layer_tt import run_decoder_layer_prefill_update_cache_tt
from models.experimental.glm4_moe_lite.tt.layer0_tt import (
    _alloc_contiguous_page_table,
    _alloc_paged_kvpe_cache,
    _round_up,
    convert_layer0_weights,
    make_rope_tensors,
    run_layer0_prefill_tt,
)
from models.experimental.glm4_moe_lite.tt.tt_embedding import run_tt_embedding
from models.experimental.glm4_moe_lite.tt.weights import (
    find_missing_shards,
    load_glm_lazy_state_dict,
    resolve_best_effort_snapshot_dir,
)


@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_HW_TESTS") != "1",
    reason="Enable with TT_ENABLE_HW_TESTS=1 (requires Tenstorrent device access).",
)
@pytest.mark.skipif(
    os.environ.get("TT_ENABLE_LARGE_MODEL_TESTS") != "1",
    reason="Enable with TT_ENABLE_LARGE_MODEL_TESTS=1 (loads large embedding weights).",
)
def test_generic_layer0_prefill_update_cache_matches_existing_harness() -> None:
    """Sanity-check the generic decoder-layer prefill path matches the layer0-specific harness."""
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    # Use deterministic token IDs (must be in vocab).
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32)

    cfg = json.loads((Path(snap) / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**cfg))
    hparams.validate()

    cache_dir = Path(os.path.expanduser("~/.cache/ttnn/models/glm4_moe_lite/layer0_tt_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        physical_device_ids=[0],
        dispatch_core_config=ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER),
    )
    try:
        # Reference from the existing harness (unpaged prefill, returns CPU tensors).
        ref = run_layer0_prefill_tt(
            device=mesh_device,
            snapshot_dir=Path(snap),
            input_ids=input_ids,
            cache_dir=cache_dir,
            seq_pad_multiple=128,
        )

        # -----------------------------
        # Generic decoder-layer harness
        # -----------------------------
        seq_len = int(input_ids.shape[1])
        padded_len = _round_up(seq_len, 128)

        block_size = 64
        blocks_per_seq = max(1, _round_up(seq_len, block_size) // block_size)

        kvpe_cache = _alloc_paged_kvpe_cache(
            device=mesh_device,
            max_num_blocks=int(1 * blocks_per_seq),
            block_size=block_size,
            kvpe_dim=int(hparams.kv_lora_rank + hparams.qk_rope_head_dim),
            dtype=ttnn.bfloat8_b,
        )
        page_table = _alloc_contiguous_page_table(batch=1, blocks_per_seq=blocks_per_seq)
        page_table_tt = ttnn.from_torch(
            page_table,
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        state = load_glm_lazy_state_dict(Path(snap), num_layers=int(hparams.num_hidden_layers))
        w = convert_layer0_weights(device=mesh_device, state=state, cache_dir=cache_dir)
        rope = make_rope_tensors(
            device=mesh_device,
            seq_len=padded_len,
            rope_dim=int(hparams.qk_rope_head_dim),
            rope_theta=float(hparams.rope_theta),
        )

        input_padded = torch.zeros((1, padded_len), dtype=input_ids.dtype)
        input_padded[:, :seq_len] = input_ids
        x_embed = run_tt_embedding(device=mesh_device, token_ids=input_padded.to(torch.int32), tt_weight=w.embed_w)
        if x_embed.layout != ttnn.TILE_LAYOUT:
            x_embed = ttnn.to_layout(x_embed, ttnn.TILE_LAYOUT)
        x_embed = ttnn.reshape(x_embed, (1, 1, padded_len, int(hparams.hidden_size)))

        x_out = run_decoder_layer_prefill_update_cache_tt(
            device=mesh_device,
            x_embed=x_embed,
            page_table_tt=page_table_tt,
            kvpe_cache=kvpe_cache,
            cos_matrix=rope["cos_matrix"],
            sin_matrix=rope["sin_matrix"],
            trans_matrix=rope["trans_matrix"],
            w=w,
            hparams=hparams,
            prompt_len=seq_len,
        )

        x0 = ttnn.slice(x_out, [0, 0, 0, 0], [1, 1, seq_len, int(hparams.hidden_size)])
        out = ttnn.to_torch(x0).reshape(1, seq_len, int(hparams.hidden_size)).cpu()

        # Cleanup.
        ttnn.deallocate(x0)
        ttnn.deallocate(x_out)
        ttnn.deallocate(x_embed)
        ttnn.deallocate(page_table_tt)
        ttnn.deallocate(kvpe_cache)
        ttnn.deallocate(rope["cos_matrix"])
        ttnn.deallocate(rope["sin_matrix"])
        ttnn.deallocate(rope["trans_matrix"])

    finally:
        ttnn.close_mesh_device(mesh_device)

    ok, msg = comp_pcc(out, ref.x_mlp_out, pcc=0.999)
    assert ok, f"generic layer0 prefill mismatch vs specialized harness: {msg}"
