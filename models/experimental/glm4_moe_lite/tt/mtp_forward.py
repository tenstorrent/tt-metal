# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Multi-Token Prediction (MTP) eager forward pass for GLM-4.7-Flash.

Extracted from model_tt.py Glm4MoeLiteDenseOnlyTT._mtp_forward_eager.
This is the MTP layer-47 decode step that runs after the main model's decode
to produce one draft token for speculative decoding.
"""

from __future__ import annotations

from typing import Any

import torch

import ttnn
from models.demos.glm4_moe_lite.tt.config import Glm4MoeLiteHParams
from models.demos.glm4_moe_lite.tt.decoder_layer_tt import (
    prepare_decode_rope_and_positions_tt,
    prepare_decode_rope_inputs_for_rotary_llama_decode_mode_tt,
    run_decoder_layer_decode_one_step_update_cache_tt,
)
from models.demos.glm4_moe_lite.tt.tt_embedding import run_tt_embedding


def _is_mesh_device(device: Any) -> bool:
    return device.__class__.__name__ == "MeshDevice"


def _tt_to_torch_single(*, tensor: ttnn.Tensor, device: Any) -> torch.Tensor:
    """Read a TT tensor to torch (device-0 only for mesh)."""
    if not _is_mesh_device(device):
        return ttnn.to_torch(tensor.cpu())
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0].cpu())


def _mesh_to_torch_selected(*, tensor: ttnn.Tensor, device_ids: list[int]) -> list[torch.Tensor]:
    device_tensors = ttnn.get_device_tensors(tensor)
    return [ttnn.to_torch(device_tensors[i].cpu()) for i in device_ids]


def mtp_forward_eager(
    *,
    device: Any,
    hparams: Glm4MoeLiteHParams,
    main_token_ids: torch.Tensor,
    hidden_state: ttnn.Tensor,
    mtp_positions: torch.Tensor,
    page_table: torch.Tensor,
    kv_cache: list[ttnn.Tensor],
    max_seq_len: int,
    rope: dict[str, ttnn.Tensor],
    embed_w: ttnn.Tensor,
    mtp_enorm: Any,
    mtp_hnorm: Any,
    mtp_eh_proj_w: ttnn.Tensor,
    mtp_decoder_w: Any,
    mtp_shared_head_norm: Any,
    mtp_shared_head_w: ttnn.Tensor,
    moe_runtime: Any | None,
    lm_head_sharded_vocab: bool,
    lm_head_tp_axis: int | None,
    lm_head_vocab_per_shard: int,
) -> torch.Tensor:
    """Run MTP layer 47 eagerly. Returns draft_token_ids [B] int32 on CPU.

    This is a standalone function version of _mtp_forward_eager, taking
    explicit arguments instead of self.* fields.
    """
    batch = int(main_token_ids.shape[0])
    hidden = int(hparams.hidden_size)
    is_mesh = _is_mesh_device(device)

    # 1. Embed main model's predicted tokens
    x_embed = run_tt_embedding(device=device, token_ids=main_token_ids, tt_weight=embed_w)
    if x_embed.layout != ttnn.TILE_LAYOUT:
        x_embed = ttnn.to_layout(x_embed, ttnn.TILE_LAYOUT)
    x_embed = ttnn.reshape(x_embed, (1, batch, 1, hidden))
    x_embed = ttnn.permute(x_embed, (0, 2, 1, 3))
    x_embed_view = ttnn.slice(x_embed, [0, 0, 0, 0], [1, 1, batch, hidden])
    x_embed_tight = ttnn.clone(x_embed_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(x_embed, force=False)
    x_embed = x_embed_tight

    # 2. enorm(embedded), hnorm(hidden_state)
    enorm_out = mtp_enorm(x_embed, mode="decode")
    ttnn.deallocate(x_embed, force=False)
    hnorm_out = mtp_hnorm(hidden_state, mode="decode")

    # 3. concat + project
    concat = ttnn.concat([enorm_out, hnorm_out], dim=3)
    ttnn.deallocate(enorm_out, force=False)
    ttnn.deallocate(hnorm_out, force=False)
    proj = ttnn.linear(concat, mtp_eh_proj_w)
    ttnn.deallocate(concat, force=False)

    # 4. Prepare RoPE and positions
    mtp_positions_clamped = mtp_positions.to(torch.int32).clamp(min=0, max=max(0, int(max_seq_len) - 1))
    tt_positions, cos_batch, sin_batch = prepare_decode_rope_and_positions_tt(
        device=device,
        rope=rope,
        positions=mtp_positions_clamped,
    )
    cos_decode, sin_decode, trans_decode, rope_sharded_cfg = prepare_decode_rope_inputs_for_rotary_llama_decode_mode_tt(
        device=device,
        cos_batch=cos_batch,
        sin_batch=sin_batch,
        trans_matrix=rope["trans_matrix"],
        batch=batch,
        rope_dim=int(hparams.qk_rope_head_dim),
    )

    page_table_tt = ttnn.from_torch(
        page_table[:batch].to(torch.int32).contiguous(),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh else None,
    )

    # 5. Run decoder layer 47
    x = run_decoder_layer_decode_one_step_update_cache_tt(
        device=device,
        x_embed_tok=proj,
        tt_positions=tt_positions,
        page_table_tt=page_table_tt,
        kvpe_cache=kv_cache[47],
        cos_batch=cos_batch,
        sin_batch=sin_batch,
        trans_matrix=rope["trans_matrix"],
        cos_decode=cos_decode,
        sin_decode=sin_decode,
        trans_decode=trans_decode,
        rope_sharded_cfg=rope_sharded_cfg,
        w=mtp_decoder_w,
        hparams=hparams,
        moe_runtime=moe_runtime,
        profile=None,
        use_decode_rope=True,
    )
    ttnn.deallocate(proj, force=False)

    # 6. shared_head: norm + LM head
    x = mtp_shared_head_norm(x, mode="decode")
    logits_tt = ttnn.linear(x, mtp_shared_head_w)
    ttnn.deallocate(x, force=False)

    # 7. Argmax -> draft token
    vocab = int(hparams.vocab_size)

    if lm_head_sharded_vocab and is_mesh:
        logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
        vocab_per_shard = int(lm_head_vocab_per_shard)
        logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, batch, vocab_per_shard])
        max_out = ttnn.max(logits_rm_view, dim=3, keepdim=True)
        if isinstance(max_out, tuple):
            local_max_tt, local_argmax_tt = max_out
        else:
            local_max_tt = max_out
            local_argmax_tt = ttnn.argmax(logits_rm_view, dim=3, keepdim=False, use_multicore=True)

        mesh_rows, mesh_cols = int(device.shape[0]), int(device.shape[1])
        tp_axis = lm_head_tp_axis
        if tp_axis is None:
            tp_size = int(mesh_rows * mesh_cols)
            selected_device_ids = list(range(tp_size))
            shard_indices = list(range(tp_size))
        elif int(tp_axis) == 1:
            tp_size = mesh_cols
            selected_device_ids = list(range(tp_size))
            shard_indices = list(range(tp_size))
        else:
            tp_size = mesh_rows
            selected_device_ids = [r * mesh_cols for r in range(tp_size)]
            shard_indices = list(range(tp_size))

        local_argmax_torch = _mesh_to_torch_selected(tensor=local_argmax_tt, device_ids=selected_device_ids)
        local_max_torch = _mesh_to_torch_selected(tensor=local_max_tt, device_ids=selected_device_ids)
        ttnn.deallocate(local_argmax_tt, force=False)
        ttnn.deallocate(local_max_tt, force=False)
        ttnn.deallocate(logits_rm, force=False)

        draft_token_ids = torch.empty((batch,), dtype=torch.int32)
        for b in range(batch):
            best_val = None
            best_global = None
            for shard_idx, max_tensor, argmax_tensor in zip(shard_indices, local_max_torch, local_argmax_torch):
                max_val = float(max_tensor.reshape(-1)[b].item())
                local_idx = int(argmax_tensor.reshape(-1)[b].item())
                global_idx = int(shard_idx * vocab_per_shard + local_idx)
                if global_idx >= vocab:
                    continue
                if best_val is None or max_val > best_val:
                    best_val = max_val
                    best_global = global_idx
            if best_global is None:
                best_global = max(0, vocab - 1)
            draft_token_ids[b] = int(best_global)
    else:
        logits_rm = ttnn.to_layout(logits_tt, ttnn.ROW_MAJOR_LAYOUT)
        logits_rm_view = ttnn.slice(logits_rm, [0, 0, 0, 0], [1, 1, batch, vocab])
        logits_rm_tight = ttnn.clone(logits_rm_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        max_out = ttnn.max(logits_rm_tight, dim=3, keepdim=True)
        if isinstance(max_out, tuple):
            _, next_ids_tt = max_out
        else:
            next_ids_tt = ttnn.argmax(logits_rm_tight, dim=3, keepdim=False, use_multicore=True)
        draft_token_ids = (
            _tt_to_torch_single(
                tensor=next_ids_tt,
                device=device,
            )
            .reshape(-1)
            .to(dtype=torch.int32)
            .cpu()
        )
        ttnn.deallocate(logits_rm, force=False)
        ttnn.deallocate(logits_rm_tight, force=False)
        ttnn.deallocate(next_ids_tt, force=False)

    ttnn.deallocate(logits_tt, force=False)

    # Cleanup RoPE temporaries
    for t in (cos_decode, sin_decode, trans_decode, cos_batch, sin_batch, tt_positions, page_table_tt):
        ttnn.deallocate(t, force=False)

    return draft_token_ids
