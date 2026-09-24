# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Allocate DeepSeek-V4-Flash's KV groups on a mesh (device code; the geometry is ``layer_kinds``, the contract is
``kv_contract``)."""

from __future__ import annotations

import ttnn
from models.demos.deepseek_v3_d_p.tt.runners.kv_caches import V4FlashKvCaches
from models.demos.deepseek_v3_d_p.tt.v4.kv_contract import CONTRACT
from models.demos.deepseek_v3_d_p.tt.v4.layer_kinds import V4FlashKvGeometry

_DTYPE_LAYOUT = {
    "bfp8_tile": (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    "bfp4_tile": (ttnn.bfloat4_b, ttnn.TILE_LAYOUT),
    "bf16_rm": (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
}


def allocate_v4_flash_kv_caches(*, mesh_device, hf_config, params) -> V4FlashKvCaches:
    """One DRAM-nd-sharded, zero-filled tensor per KV group for THIS rank's layer slice, allocated the way the
    MLA family allocates its KVPE cache (``init_kvpe_cache``: batch = users * layers_in_group, one 32-row shard
    per DRAM bank, round-robin) but REPLICATED over SP: ``init_kvpe_cache`` divides ``seq_len`` by the SP extent,
    so we pass ``rows * sp_factor`` and every chip holds the same ``rows``. Dtype and layout come from the
    contract (bf16 ROW_MAJOR for the CSA unified cache and the pending groups, tiles otherwise)."""
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

    geom = V4FlashKvGeometry.from_config(
        hf_config,
        max_seq_len=params.max_seq_len,
        sp_factor=params.sp_factor,
        first_layer_idx=params.first_layer_idx,
        num_layers=params.num_layers,
    )

    def _alloc(group: str):
        layers = geom.layers(group)
        if not layers:
            return None
        spec = next(g for g in CONTRACT if g.name == group)
        dtype, layout = _DTYPE_LAYOUT[spec.dtype_tag]
        return init_kvpe_cache(
            kvpe_cache_head_dim=spec.width,
            mesh_device=mesh_device,
            seq_len=geom.rows(group) * geom.sp_factor,
            mesh_shape=list(params.mesh_shape),
            sp_axis=params.sp_axis,
            num_kvpe_cache_layers=len(layers),
            num_users=params.num_users,
            dtype=dtype,
            layout=layout,
        )

    return V4FlashKvCaches(
        geometry=geom,
        swa_window=_alloc("swa_window"),
        hca_unified=_alloc("hca_unified"),
        csa_unified=_alloc("csa_unified"),
        csa_index_k=_alloc("csa_index_k"),
        csa_pending=_alloc("csa_pending"),
        hca_pending=_alloc("hca_pending"),
    )
