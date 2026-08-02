# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MiniMax-M3 KV-cache golden PCC validation (bring-up / standalone only — never used in serving).

Per-layer K / V / index_k PCC: the device cache (``runtime.gather_layer``) vs the golden trace written
by ``scripts/generate_golden_kv_cache.py`` (keys ``key/value/index_k_cache_layer_N``, HF layout). The
device stores K / index_k Meta-RoPE swizzled over the rotary slice (``ModelArgs.load_state_dict`` ->
``convert_hf_qkv_to_meta_format_partial``), so we permute the golden's rotary slice (HF half-split ->
Meta interleaved; identity tail) before comparing. V is raw (no swizzle).

Ported from ``tests/galaxy_prefill_kv_pcc.py::check_kv_pcc`` so the runner and the standalone test share
one implementation. Called by ``TtPrefillRuntime.kv_cache_pcc_check`` (the runner's PREFILL_STANDALONE_PCC
hook). At <= 2048 tokens MSA == dense so the dense golden matches; above that the device's block-sparse
MSA diverges from the dense golden (see the m3-cpu-reference-model notes).
"""

import os
from pathlib import Path

import torch
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.demos.common.prefill.runners.runner_utils import resolve_trace_dir


def _hf_to_meta_rotary_perm(head_dim: int, rotary_dim: int) -> torch.Tensor:
    """Column permutation mapping the HF golden K's rotary slice (half-split) to the device's Meta
    interleaved layout; identity on the non-rotary tail."""
    half = rotary_dim // 2
    src = list(range(head_dim))
    for m in range(rotary_dim):
        src[m] = half * (m % 2) + (m // 2)
    return torch.tensor(src, dtype=torch.long)


def kv_cache_pcc_check(
    runtime, kv_cache, *, slot_id: int, n_chunks: int, trace_dir=None, first_layer_idx: int = 0, real_len=None
):
    """PCC the populated ``kv_cache`` (slot ``slot_id``) against the golden trace; return the min per-layer
    PCC across K / V / index_k and assert it clears the threshold (unless record-only). ``real_len`` further
    caps the compared extent to the real (non-pad) tokens (used by the migration validators).

    Env:
      PREFILL_STANDALONE_CHUNKED_PCC          min PCC threshold (default 0.88)
      PREFILL_STANDALONE_CHUNKED_RECORD_ONLY  "1" -> log PCC, skip the assert (record a new baseline)
    """
    from safetensors import safe_open

    hf_config = runtime.hf_config
    num_layers = runtime.config.num_layers
    chunk_size = runtime.config.chunk_size

    trace_dir = resolve_trace_dir(trace_dir or os.environ["PREFILL_TRACE_DIR"])
    import json

    with open(Path(trace_dir) / "metadata.json") as f:
        token_ids = list(json.load(f)["token_ids"])
    # Compare only the real prompt tokens this run actually filled (chunk-aligned cap; real_len tightens it).
    n_tokens = min(len(token_ids), n_chunks * chunk_size)
    if real_len:
        n_tokens = min(n_tokens, real_len)

    head_dim = hf_config.head_dim
    rotary_dim = getattr(hf_config, "rotary_dim", head_dim)
    src = _hf_to_meta_rotary_perm(head_dim, rotary_dim)

    kv_dir = Path(trace_dir) / "kv_cache"
    threshold = float(os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.88"))
    record_only = os.environ.get("PREFILL_STANDALONE_CHUNKED_RECORD_ONLY", "0") == "1"

    logger.info(
        f"[kv-pcc] per-layer K / V / index_k vs golden ({trace_dir}) for slot={slot_id}, "
        f"{n_tokens} tokens, layers [{first_layer_idx}, {first_layer_idx + num_layers}):"
    )
    mins = {"k": 1.0, "v": 1.0, "index_k": 1.0}
    for L in range(num_layers):
        gL = first_layer_idx + L  # device layer L == global (golden) layer first_layer_idx + L
        dev_k, dev_v, dev_ik = runtime.gather_layer(kv_cache, slot_id=slot_id, layer_idx=L, n_tokens=n_tokens)
        with safe_open(str(kv_dir / f"layer_{gL}.safetensors"), framework="pt") as h:
            keys = set(h.keys())
            g_k = h.get_tensor(f"key_cache_layer_{gL}").float()[:, :, :n_tokens, :][..., src]  # HF -> Meta
            g_v = h.get_tensor(f"value_cache_layer_{gL}").float()[:, :, :n_tokens, :]
            has_ik = f"index_k_cache_layer_{gL}" in keys
            g_ik = h.get_tensor(f"index_k_cache_layer_{gL}").float()[:, :, :n_tokens, :][..., src] if has_ik else None

        pcc_k = float(comp_pcc(g_k, dev_k, 0.0)[1])
        pcc_v = float(comp_pcc(g_v, dev_v, 0.0)[1])
        mins["k"], mins["v"] = min(mins["k"], pcc_k), min(mins["v"], pcc_v)
        line = f"  layer {gL:>2}: K={pcc_k:.5f} V={pcc_v:.5f}"
        if has_ik:
            pcc_ik = float(comp_pcc(g_ik, dev_ik, 0.0)[1])
            mins["index_k"] = min(mins["index_k"], pcc_ik)
            line += f" index_k={pcc_ik:.5f}"
        logger.info(line)

    min_pcc = min(mins.values())
    logger.info(
        f"[kv-pcc] min PCC across {num_layers} layers: "
        f"K={mins['k']:.5f} V={mins['v']:.5f} index_k={mins['index_k']:.5f} (overall {min_pcc:.5f})"
    )
    if not record_only:
        assert min_pcc >= threshold, f"KV-cache PCC {min_pcc:.5f} < threshold {threshold}"
    return min_pcc
