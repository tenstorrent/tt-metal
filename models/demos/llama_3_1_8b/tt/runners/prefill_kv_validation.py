# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer KV PCC against the CPU golden trace — the graded comparison (P1/P2).

Three layout conversions stand between the device cache and the golden file, and getting any of them
wrong produces a plausible-looking but wrong PCC, so each is a named function:

1. **block-cyclic -> natural token order.** The device writes SP-sharded block-cyclic rows;
   ``blockcyclic_positions`` (imported from ``models/common/utils.py``) is the inverse of the
   ``update_padded_kv_cache`` writer and says which global position each shard row holds.
2. **Meta -> HF column order for K.** The device rotates in the Meta interleaved layout; the golden
   stores ``k_layout: hf_half_split``. The golden's last dim is gathered through
   ``hf_to_meta_perm`` so the comparison happens in the device's order without touching the device
   tensor.
3. **nothing for V.** V is never rotated (``v_is_raw`` in the trace metadata), so a permutation
   applied to it would be a bug that *reduces* PCC — the trace metadata is asserted rather than
   assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.common.utils import blockcyclic_positions

from ...utils.rope_layout import hf_to_meta_perm


def naturalize(block: torch.Tensor, n_tokens: int, sp: int, chunk_size: int, max_seq_len: int) -> torch.Tensor:
    """``[..., seq_cache, head_dim]`` in on-device block-cyclic order -> natural order, first
    ``n_tokens`` rows. Inverse of the ``update_padded_kv_cache`` writer."""
    p = blockcyclic_positions(sp, chunk_size, max_seq_len)
    nat = torch.empty_like(block)
    nat[..., p, :] = block
    return nat[..., :n_tokens, :]


def load_trace(trace_dir) -> dict:
    """Read a golden trace's metadata and assert the conventions this comparison depends on."""
    meta_path = Path(trace_dir) / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta.get("k_is_post_rope", True), f"{meta_path}: golden K must be post-RoPE"
    assert meta.get("v_is_raw", True), f"{meta_path}: golden V must be raw (un-rotated)"
    assert meta.get("k_layout", "hf_half_split") == "hf_half_split", (
        f"{meta_path}: golden K layout is {meta.get('k_layout')}, but this comparison converts from "
        f"hf_half_split to the device's Meta interleaved order"
    )
    assert not meta.get("reduced_depth", False), f"{meta_path}: golden trace is depth-reduced; not a valid grade"
    return meta


def golden_layer_kv(trace_dir, layer: int, n_tokens: int, head_dim: int):
    """This layer's golden ``(K, V)`` for the first ``n_tokens``, K already in the DEVICE's column
    order. Shapes ``[1, num_kv_heads, n_tokens, head_dim]``."""
    from safetensors import safe_open

    perm = hf_to_meta_perm(head_dim)
    with safe_open(str(Path(trace_dir) / "kv_cache" / f"layer_{layer}.safetensors"), framework="pt") as h:
        g_k = h.get_tensor(f"key_cache_layer_{layer}").float()[:, :, :n_tokens, :][..., perm]
        g_v = h.get_tensor(f"value_cache_layer_{layer}").float()[:, :, :n_tokens, :]
    return g_k, g_v


def per_layer_kv_pcc(runtime, kv_cache, *, trace_dir, n_tokens: int, slot_id: int = 0, log=True):
    """PCC every layer's device K and V against the golden trace.

    Returns ``[{"layer": i, "k": float, "v": float}, ...]``, one row per model layer in order. The
    slot is read back ONCE (one device slice plus one mesh compose per cache) and un-rotated per
    layer on host; reading per layer would re-copy the whole packed cache over PCIe N times.

    This function does not assert — the caller owns the threshold, because the spec's
    ``pcc_lower_bound`` is the thing that decides, not a default buried here.
    """
    cfg = runtime.config
    k_blk, v_blk = runtime.read_slot_kv(kv_cache, slot_id)
    head_dim = runtime.cfg.head_dim
    rows = []
    for L in range(cfg.num_layers):
        dev_k = naturalize(k_blk[L], n_tokens, cfg.sp, cfg.chunk_size, cfg.max_seq_len).unsqueeze(0)
        dev_v = naturalize(v_blk[L], n_tokens, cfg.sp, cfg.chunk_size, cfg.max_seq_len).unsqueeze(0)
        g_k, g_v = golden_layer_kv(trace_dir, L, n_tokens, head_dim)
        assert g_k.shape == dev_k.shape, f"layer {L}: golden K {tuple(g_k.shape)} vs device {tuple(dev_k.shape)}"
        row = {"layer": L, "k": float(comp_pcc(g_k, dev_k, 0.0)[1]), "v": float(comp_pcc(g_v, dev_v, 0.0)[1])}
        rows.append(row)
        if log:
            logger.info(f"  layer {L:>2}: K={row['k']:.6f} V={row['v']:.6f}")
    return rows
