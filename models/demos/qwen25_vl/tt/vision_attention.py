# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
from models.tt_transformers.tt.attention import Attention


class VisionAttention(Attention):
    def __init__(self, *args, **kwargs):
        kwargs["causal_mask"] = False
        # [INFO] disabling kv cache for vision attention needs both of `use_kv_cache` and `use_paged_kv_cache` to be set to False, True
        kwargs["use_kv_cache"] = False
        kwargs["use_paged_kv_cache"] = True
        super().__init__(*args, **kwargs)

    def forward(self, x, cu_seqlens, rot_mats, user_id=0, page_table=None, chunk_page_table=None, chunk_start_idx=None):
        seq_len = x.shape[-2]
        attention_mask = torch.full([1, 1, seq_len, seq_len], -1e9, dtype=torch.float32)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        tt_mask = ttnn.from_torch(
            attention_mask, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=self.mesh_device
        )

        return super().forward_prefill(
            x,
            rot_mats=rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=None,
            mask=tt_mask,
        )
