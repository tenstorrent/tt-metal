# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shard-advisor capture target for one dense GPT-OSS-20B decode block.

The production default uses routed ``sparse_matmul`` experts, which are a
documented terminal path for the advisor tracer.  This target therefore uses
the optimized decoder's dense packed gate/up candidate so the mandatory
attention-plus-MLP layout query covers the same rewritten projections and
boundaries without bypassing them with a toy graph.
"""

from __future__ import annotations

import os
import sys

import torch
import ttnn


TT_METAL_ROOT = os.environ.get("SHARD_ADVISE_TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
if TT_METAL_ROOT not in sys.path:
    # Append: the advisor environment's installed ttnn package must retain
    # precedence over the repository's source-level ttnn namespace.
    sys.path.append(TT_METAL_ROOT)


_DECODER = None
_KEY_CACHE = None
_VALUE_CACHE = None
_CACHE_POSITION = 17
_CACHE_POSITION_TENSOR = None
_ATTENTION_MASK = None


def _build(device):
    from models.autoports.openai_gpt_oss_20b.tests.test_functional_decoder import (
        LAYER_IDX,
        _decode_mask,
        _position_tensor,
        _synthetic_state_dict,
        _to_tt,
    )
    from models.autoports.openai_gpt_oss_20b.tt.optimized_decoder import OptimizedDecoder
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        os.path.join(TT_METAL_ROOT, "models/demos/gpt_oss/configs/gpt-oss-20b"),
        local_files_only=True,
    )
    config._attn_implementation = "eager"
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state_dict(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=device,
        max_cache_len=288,
        candidate="dense_packed_bfp4",
    )
    key_cache, value_cache = decoder.create_kv_cache()
    generator = torch.Generator().manual_seed(20260725)
    hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    return (
        decoder,
        key_cache,
        value_cache,
        _position_tensor(_CACHE_POSITION, device),
        _decode_mask(_CACHE_POSITION, config, decoder.max_cache_len, device),
        _to_tt(hidden, device),
    )


def decode(hidden):
    return _DECODER.decode_forward(
        hidden,
        key_cache=_KEY_CACHE,
        value_cache=_VALUE_CACHE,
        cache_position=_CACHE_POSITION,
        cache_position_tensor=_CACHE_POSITION_TENSOR,
        attention_mask=_ATTENTION_MASK,
    )


def make_inputs(device):
    global _DECODER
    global _KEY_CACHE
    global _VALUE_CACHE
    global _CACHE_POSITION_TENSOR
    global _ATTENTION_MASK

    (
        _DECODER,
        _KEY_CACHE,
        _VALUE_CACHE,
        _CACHE_POSITION_TENSOR,
        _ATTENTION_MASK,
        hidden,
    ) = _build(device)
    return (hidden,)
