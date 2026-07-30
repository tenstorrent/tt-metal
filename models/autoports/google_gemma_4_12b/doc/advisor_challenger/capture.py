# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shard-advisor capture target for the shipped batch-32 decoder policy."""

from __future__ import annotations

import os
import sys

import torch
import ttnn

TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    sys.path.append(TT_METAL_ROOT)

from models.autoports.google_gemma_4_12b.doc.advisor_challenger.harness import BATCH, _build

LAYER_KIND = os.environ.get("CHALLENGER_LAYER_KIND", "sliding_attention")
_DECODER = None
_KWARGS = None


def decode(hidden):
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    # The shipped decode has already converted fused QKV to L1 here. Avoid the
    # shared helper's runtime memory_config() inspection, which TracedTensor
    # deliberately does not expose.
    def split_qkv_heads_decode_traced(xqkv_fused, config, is_global, tp=1, kv_replicated=False):
        num_local_heads = config.num_attention_heads // tp
        num_local_kv_heads = 1 if kv_replicated else config.num_key_value_heads // tp
        return ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=num_local_heads,
            num_kv_heads=num_local_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )

    _DECODER, hidden, _KWARGS = _build(LAYER_KIND, device)
    _DECODER.self_attn.decode_forward.__globals__["split_qkv_heads_decode"] = split_qkv_heads_decode_traced
    return (hidden,)
