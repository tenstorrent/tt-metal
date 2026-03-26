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

from models.demos.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.demos.glm4_moe_lite.tt.decoder_layer_tt import (
    prepare_decode_rope_and_positions_tt,
    run_decoder_layer_decode_one_step_update_cache_tt,
)
from models.demos.glm4_moe_lite.tt.layer0_tt import (
    _alloc_contiguous_page_table,
    _alloc_paged_kvpe_cache,
    _round_up,
    convert_layer0_weights,
    make_rope_tensors,
    run_layer0_decode_one_step_update_cache_tt,
)
from models.demos.glm4_moe_lite.tt.tt_embedding import run_tt_embedding
from models.demos.glm4_moe_lite.tt.weights import (
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
def test_generic_layer0_decode_update_cache_matches_existing_harness() -> None:
    """Sanity-check the generic decoder layer decode path matches the layer0-specific harness."""
    snap = resolve_best_effort_snapshot_dir("zai-org/GLM-4.7-Flash")
    missing = find_missing_shards(snap)
    if missing:
        pytest.skip(f"GLM snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first.")

    # Minimal deterministic token ids (must be in vocab).
    prefix_input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32)
    next_token_id = 6

    # Build hparams from raw config.json (avoid AutoConfig dependency).
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
        # Reference output from the known-good layer0 harness.
        ref = run_layer0_decode_one_step_update_cache_tt(
            device=mesh_device,
            snapshot_dir=Path(snap),
            prefix_input_ids=prefix_input_ids,
            next_token_id=next_token_id,
            batch=32,
            block_size=64,
            cache_dir=cache_dir,
        )

        # -----------------------------
        # Generic decoder-layer harness
        # -----------------------------
        batch = 32
        block_size = 64
        seq_len = int(prefix_input_ids.shape[1])

        blocks_per_seq = _round_up(seq_len + 1, block_size) // block_size
        min_blocks_per_seq = max(1, (128 + int(block_size) - 1) // int(block_size))
        blocks_per_seq = max(blocks_per_seq, min_blocks_per_seq)
        page_len = int(blocks_per_seq * block_size)

        kvpe_cache = _alloc_paged_kvpe_cache(
            device=mesh_device,
            max_num_blocks=int(batch * blocks_per_seq),
            block_size=block_size,
            kvpe_dim=int(hparams.kv_lora_rank + hparams.qk_rope_head_dim),
            dtype=ttnn.bfloat8_b,
        )
        page_table = _alloc_contiguous_page_table(batch=batch, blocks_per_seq=blocks_per_seq)
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
            device=mesh_device, seq_len=page_len, rope_dim=int(hparams.qk_rope_head_dim), rope_theta=hparams.rope_theta
        )

        # Prefill KVPE cache for slot 0 (prompt only).
        full_padded = torch.zeros((1, page_len), dtype=prefix_input_ids.dtype)
        full_padded[:, :seq_len] = prefix_input_ids

        x_embed = run_tt_embedding(device=mesh_device, token_ids=full_padded.to(torch.int32), tt_weight=w.embed_w)
        if x_embed.layout != ttnn.TILE_LAYOUT:
            x_embed = ttnn.to_layout(x_embed, ttnn.TILE_LAYOUT)
        x_embed = ttnn.reshape(x_embed, (1, 1, page_len, int(hparams.hidden_size)))

        x = w.input_layernorm(x_embed, mode="prefill")
        kv = ttnn.linear(x, w.w_kv_a)  # [1,1,S,kvpe_dim]
        ttnn.deallocate(x)
        kv_nope = ttnn.slice(kv, [0, 0, 0, 0], [1, 1, page_len, int(hparams.kv_lora_rank)])
        kv_rope = ttnn.slice(
            kv,
            [0, 0, 0, int(hparams.kv_lora_rank)],
            [1, 1, page_len, int(hparams.kv_lora_rank + hparams.qk_rope_head_dim)],
        )
        ttnn.deallocate(kv)

        kv_nope = w.kv_a_layernorm(kv_nope, mode="prefill")
        kv_rope = ttnn.experimental.rotary_embedding_llama(
            kv_rope,
            rope["cos_matrix"],
            rope["sin_matrix"],
            rope["trans_matrix"],
            is_decode_mode=False,
        )
        kvpe = ttnn.concat([kv_nope, kv_rope], dim=-1)
        ttnn.deallocate(kv_nope)
        ttnn.deallocate(kv_rope)

        if kvpe.dtype != kvpe_cache.dtype:
            kvpe_cast = ttnn.typecast(kvpe, dtype=kvpe_cache.dtype)
            ttnn.deallocate(kvpe)
            kvpe = kvpe_cast

        ttnn.experimental.paged_fill_cache(kvpe_cache, kvpe, page_table=page_table_tt, batch_idx=0)
        ttnn.deallocate(kvpe)
        ttnn.deallocate(x_embed)

        # Decode step: only slot 0 active.
        tokens = torch.zeros((batch, 1), dtype=torch.int32)
        tokens[0, 0] = int(next_token_id)

        positions = torch.zeros((batch,), dtype=torch.int32)
        positions[0] = seq_len

        tt_positions, cos_batch, sin_batch = prepare_decode_rope_and_positions_tt(
            device=mesh_device, rope=rope, positions=positions
        )

        x_embed_tok = run_tt_embedding(device=mesh_device, token_ids=tokens, tt_weight=w.embed_w)
        if x_embed_tok.layout != ttnn.TILE_LAYOUT:
            x_embed_tok = ttnn.to_layout(x_embed_tok, ttnn.TILE_LAYOUT)
        x_embed_tok = ttnn.reshape(x_embed_tok, (1, batch, 1, int(hparams.hidden_size)))
        x_embed_tok = ttnn.permute(x_embed_tok, (0, 2, 1, 3))  # [1,1,B,D]

        x_out = run_decoder_layer_decode_one_step_update_cache_tt(
            device=mesh_device,
            x_embed_tok=x_embed_tok,
            tt_positions=tt_positions,
            page_table_tt=page_table_tt,
            kvpe_cache=kvpe_cache,
            cos_batch=cos_batch,
            sin_batch=sin_batch,
            trans_matrix=rope["trans_matrix"],
            w=w,
            hparams=hparams,
        )

        x0 = ttnn.slice(x_out, [0, 0, 0, 0], [1, 1, 1, int(hparams.hidden_size)])
        out = ttnn.to_torch(x0).reshape(1, int(hparams.hidden_size)).cpu()

        # Cleanup (explicit to avoid accumulating buffers in optional HW tests).
        ttnn.deallocate(x0)
        ttnn.deallocate(x_out)
        ttnn.deallocate(x_embed_tok)
        ttnn.deallocate(tt_positions)
        ttnn.deallocate(cos_batch)
        ttnn.deallocate(sin_batch)
        ttnn.deallocate(page_table_tt)
        ttnn.deallocate(kvpe_cache)
        ttnn.deallocate(rope["cos_matrix"])
        ttnn.deallocate(rope["sin_matrix"])
        ttnn.deallocate(rope["trans_matrix"])

    finally:
        ttnn.close_mesh_device(mesh_device)

    ok, msg = comp_pcc(out, ref, pcc=0.999)
    assert ok, f"generic layer0 decode mismatch vs specialized harness: {msg}"
