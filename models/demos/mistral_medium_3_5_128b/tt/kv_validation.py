# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer KV-cache PCC against the CPU golden trace (bring-up / acceptance only).

The trace (``metadata.json`` + ``kv_cache/layer_N.safetensors`` with ``key_cache_layer_N`` /
``value_cache_layer_N``, ``[1, 8, n_tokens, 128]`` bf16) stores post-RoPE K in the HF half-split layout
and raw V. The device cache holds K in the Meta interleaved layout (q/k rows permuted at load, see
``tt/rope.py``), so the golden K's head_dim is permuted the same way before comparing; V is compared as
is. The device cache is read back once per user and un-rotated from block-cyclic to natural order.
"""

import json
from pathlib import Path

import torch
from safetensors import safe_open

from models.common.utility_functions import comp_pcc

from .kv_cache import naturalize
from .rope import hf_to_meta_perm


def trace_token_ids(trace_dir):
    with open(Path(trace_dir) / "metadata.json") as f:
        meta = json.load(f)
    return list(meta["token_ids"]), meta


def golden_layer_kv(trace_dir, layer: int, n_tokens: int):
    with safe_open(str(Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"), framework="pt") as f:
        k = f.get_slice(f"key_cache_layer_{layer}")[:, :, :n_tokens, :]
        v = f.get_slice(f"value_cache_layer_{layer}")[:, :, :n_tokens, :]
    return k[0], v[0]


def layer_kv_pcc(k_blk, v_blk, trace_dir, *, n_tokens, sp, chunk_size, max_seq_len, head_dim, first_layer_idx=0):
    """``k_blk`` / ``v_blk``: host ``[num_layers, heads, max_seq_len, D]`` block-cyclic read-back.
    Returns one ``{"layer", "k", "v"}`` row per layer (global layer index), in order."""
    perm = hf_to_meta_perm(head_dim)
    rows = []
    for local in range(k_blk.shape[0]):
        layer = first_layer_idx + local
        gk, gv = golden_layer_kv(trace_dir, layer, n_tokens)
        dk = naturalize(k_blk[local], n_tokens, sp, chunk_size, max_seq_len)
        dv = naturalize(v_blk[local], n_tokens, sp, chunk_size, max_seq_len)
        pk = float(comp_pcc(gk[..., perm].float(), dk, 0.0)[1])
        pv = float(comp_pcc(gv.float(), dv, 0.0)[1])
        rows.append({"layer": layer, "k": pk, "v": pv})
    return rows


def assert_finite_kv(k_blk, v_blk, n_layers):
    for name, blk in (("k", k_blk), ("v", v_blk)):
        assert blk.shape[0] == n_layers and torch.isfinite(blk).all(), f"non-finite {name} cache"
