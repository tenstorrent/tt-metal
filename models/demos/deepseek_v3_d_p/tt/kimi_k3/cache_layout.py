# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The Kimi-K3 prefill KVPE cache declared as a ``CacheLayoutModel`` (the cache tensor layout spec).

Kimi-K3's merged table has three configs: "00" the MLA KVPE cache, "01" / "02" the KDA recurrent and
convolution states. Only "00" is a cache layout: the KDA configs sit on the contract's synthetic position
axis (a segment per position, ``KDA_VERSIONS`` aliased windows), so they are not declared and stay walked.

"00" is the DeepSeek-V3 KVPE declaration (``deepseek_v3_d_p.tt.runners.cache_layout``) for a compacted
cache: 24 of the 93 layers are MLA, so each stage's cache holds only its MLA layers (slot x those layers on
its batch axis), its stages numbered in compacted MLA-slot space and published at the model layers
``layer_rows`` names. The cache is TP-deduped by default (sequence sharded over both mesh axes).
"""

from models.demos.deepseek_v3_d_p.tt.runners.cache_layout import deepseek_cache_layout_model

# The merged table's KVPE config; the KDA configs "01" / "02" are not declared.
KVPE_CONFIG_NAME = "00"


def kimi_k3_cache_layout_model(
    *,
    kvpe_cache,
    stage_layout,
    mesh_shape,
    sp_axis,
    num_users,
    seq_len,
    chunk_size,
    kv_dedup,
    layer_rows,
):
    """The ``CacheLayoutModel`` of config "00" alone, over the gathered kvpe ``stage_layout``. Raises
    CacheLayoutNotDerivable as ``deepseek_cache_layout_model`` does."""
    return deepseek_cache_layout_model(
        kvpe_cache=kvpe_cache,
        stage_layout=stage_layout,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_users=num_users,
        seq_len=seq_len,
        chunk_size=chunk_size,
        kv_dedup=kv_dedup,
        layer_rows=layer_rows,
        config_name=KVPE_CONFIG_NAME,
    )
