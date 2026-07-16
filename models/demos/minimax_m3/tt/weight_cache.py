# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tilized-weight cache completeness check (DeepSeek's "load from cache, skip the source" trick).

The MiniMax-M3 bf16 text backbone is ~869GB — larger than host RAM here — so reading it via
``ModelArgs.load_state_dict`` on every run thrashes the page cache and takes >1h. But every weight
module already loads from a per-tensor tilized ``.tensorbin`` cache via ``ttnn.as_tensor(cache_file_name=)``;
on a cache hit the passed torch tensor is ignored. So once the cache is populated we can pass an EMPTY
state_dict and never read the source (mirrors deepseek_v3_d_p's ``state_dict={}`` + ``check_cache_complete``).

``weight_cache_is_complete`` verifies the cache holds every tensor the model will build for the layers
this run uses, so the caller can safely skip ``load_state_dict``. It scans the cache dir once and matches
by filename prefix (ttnn appends ``_dtype_<DT>_layout_<L>.tensorbin``); the expert files are checked at the
exact dtype since both bf4 and bf8 variants coexist on disk.
"""

import os
from pathlib import Path

from loguru import logger

import ttnn

# ttnn's cache_file_name suffix uses these dtype tags (e.g. ..._dtype_BFLOAT8_B_layout_TILE.tensorbin).
_DTYPE_TAG = {
    ttnn.bfloat16: "BFLOAT16",
    ttnn.bfloat8_b: "BFLOAT8_B",
    ttnn.bfloat4_b: "BFLOAT4_B",
}


def _is_dense_layer(hf_config, layer_idx: int) -> bool:
    """Dense (plain SwiGLU) MLP layer iff moe_layer_freq[idx]==0. Mirrors tt/layer.py."""
    freq = getattr(hf_config, "moe_layer_freq", None)
    return freq is not None and layer_idx < len(freq) and freq[layer_idx] == 0


def _is_sparse_layer(hf_config, layer_idx: int) -> bool:
    """Block-sparse (MSA index branch) attention layer. Mirrors tt/layer.py."""
    cfg = getattr(hf_config, "sparse_attention_config", None)
    if isinstance(cfg, dict):
        freq = cfg.get("sparse_attention_freq") if cfg.get("use_sparse_attention") else None
    else:
        freq = getattr(cfg, "sparse_attention_freq", None) if cfg is not None else None
    return bool(freq[layer_idx]) if freq is not None and layer_idx < len(freq) else False


def weight_cache_is_complete(weight_cache_path, hf_config, num_layers: int, expert_weight_dtype) -> bool:
    """True iff the tilized cache at ``weight_cache_path`` holds every tensor the model builds for
    layers 0..num_layers-1 (so the caller may pass an empty state_dict and skip the bf16 source read).

    Conservative: a missing file returns False (the caller then loads weights normally — slow but
    correct). A populated cache returns True quickly (one directory walk)."""
    if not weight_cache_path:
        return False
    root = Path(weight_cache_path)
    if not root.is_dir():
        return False

    edt = _DTYPE_TAG.get(expert_weight_dtype)
    if edt is None:
        logger.warning(f"[weight-cache] unknown expert dtype {expert_weight_dtype}; loading weights from source")
        return False

    # One walk: collect every cached file as a path relative to the cache root.
    rels = set()
    for dirpath, _, files in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        for fn in files:
            rels.add(fn if rel_dir == "." else f"{rel_dir}/{fn}")

    def has(prefix: str) -> bool:
        return any(r.startswith(prefix) for r in rels)

    use_qk_norm = bool(getattr(hf_config, "use_qk_norm", True))

    # Top-level (embed / final norm / lm head). The token embedding is now the SHARDED parallel table
    # (see tt/parallel_embedding.py), cached under a distinct key from the old replicated layout.
    from models.demos.minimax_m3.tt.parallel_embedding import EMBED_CACHE_NAME

    required = [EMBED_CACHE_NAME, "lm_head_padded_pow2.weight", "norm/weight"]

    for L in range(num_layers):
        base = f"model.layers.{L}"
        required += [
            f"{base}/input_layernorm/weight",
            f"{base}/post_attention_layernorm/weight",
            f"{base}/self_attn/wqkv",
            f"{base}/self_attn/o_proj",
        ]
        if use_qk_norm:
            required += [f"{base}/self_attn/q_norm", f"{base}/self_attn/k_norm"]
        if _is_sparse_layer(hf_config, L):
            required += [f"{base}/self_attn/index_q_proj", f"{base}/self_attn/index_k_proj"]
        if _is_dense_layer(hf_config, L):
            required += [f"{base}/mlp/gate_proj", f"{base}/mlp/up_proj", f"{base}/mlp/down_proj"]
        else:
            # MoE: top-level router + shared expert, plus the DeepSeek-style EP-MoE's own gate +
            # per-local-expert routed weights (dtype-specific).
            required += [f"{base}/mlp/router/weight", f"{base}/mlp/shared_expert/gate_proj"]
            required += [
                f"{base}/mlp/experts_ep/layer_0.gate.weight",
                f"{base}/mlp/experts_ep/layer_0.routed_expert.local_0_gate_dtype_{edt}",
            ]

    missing = [p for p in required if not has(p)]
    if missing:
        logger.info(
            f"[weight-cache] cache at {root} INCOMPLETE: {len(missing)} of {len(required)} entries missing "
            f"(e.g. {missing[:3]}); will load weights from source."
        )
        return False
    logger.info(f"[weight-cache] cache at {root} complete for {num_layers} layers; skipping bf16 source read.")
    return True
