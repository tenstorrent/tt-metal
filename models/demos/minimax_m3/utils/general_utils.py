# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
General utilities for MiniMax-M3.
"""

from models.common.utility_functions import is_blackhole


def get_cache_file_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def cache_file_exists(cache_file_name):
    """True iff a tilized tensor cache file for `cache_file_name` exists on disk. ttnn appends a
    `_dtype_<DT>_layout_<L>.tensorbin` suffix, so match by prefix. Used to decide whether to load an
    OPTIONAL weight (e.g. the MoE gate's correction bias) from cache when the source state_dict is
    absent (cache-only loading) — its presence can't be known from an empty state_dict.

    Deliberately does NOT use glob. The real cache directory name embeds the mesh shape, e.g.
    ``tensor_cache_bfp8_MeshShape([8, 4])``, and glob interprets ``[8, 4]`` as a CHARACTER CLASS
    matching one of {'8', ',', ' ', '4'} — so the pattern never matched the literal path and this
    returned False for every cached tensor whose path contains the mesh shape. Its only caller is the
    MoE gate's OPTIONAL e_score_correction_bias, so the effect was: in cache-only mode (the production
    path, where the tilized cache is complete and state_dict is empty) the bias was silently treated as
    absent and every token routed on unbiased sigmoid scores — even though the bias is cached on disk,
    config.use_routing_bias is true, and reference/model.py:435 applies it. M3's real bias is 128 values
    in 11.27..11.65 per layer, so it is not a no-op buffer.

    listdir + startswith has no metacharacter semantics and is correct for any path.
    """
    if not cache_file_name:
        return False
    import os

    directory, prefix = os.path.split(str(cache_file_name))
    if not directory or not prefix:
        return False
    try:
        return any(e.startswith(prefix) and e.endswith(".tensorbin") for e in os.listdir(directory))
    except OSError:
        return False


def sparse_attention_freq(hf_config):
    """Per-layer MSA flags (``sparse_attention_config.sparse_attention_freq``, 1 = block-sparse layer: M3
    layers 3-59), or None when the config has none. ``sparse_attention_config`` may be a dict (gated by
    ``use_sparse_attention``) or an object. ``Layer``, the weight cache and the KV chunk table's index_k rows
    all read it here, so they agree on which layers carry an index_k."""
    cfg = getattr(hf_config, "sparse_attention_config", None)
    if isinstance(cfg, dict):
        return cfg.get("sparse_attention_freq") if cfg.get("use_sparse_attention") else None
    return getattr(cfg, "sparse_attention_freq", None) if cfg is not None else None


def is_sparse_attention_layer(hf_config, layer_idx: int) -> bool:
    """True iff global layer ``layer_idx`` runs MSA block-sparse attention (see ``sparse_attention_freq``)."""
    freq = sparse_attention_freq(hf_config)
    return bool(freq[layer_idx]) if freq is not None and layer_idx < len(freq) else False


def get_default_num_links(mesh_device):
    """Default number of fabric links for CCL ops on the given mesh.

    Blackhole exposes 2 fabric links per device; Wormhole exposes 4. Single-row meshes
    (shape[0] == 1) only need 1 link regardless of arch.
    """
    if mesh_device.shape[0] == 1:
        return 1
    return 2 if is_blackhole() else 4
