# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill PCC of ``DeepSeekV41PrefillAttention`` against the checkpoint's own reference, on real weights.

A case runs a group of layers over a random prompt, each layer on its own random input and in layer order --
the order their shared compressed KV / selection needs. The reference (``tests/reference.py``) runs the same
layers over the whole prompt in one pass at ``start_pos = 0``; the device sees it as chunks of ``chunk`` tokens
(the whole prompt when equal), every layer carrying its own state from chunk to chunk and the group's
``PrefillShared`` reset per chunk. The output of every layer is compared over the whole prompt, and logged per
chunk.

The cases cover every kind of layer and both sides of the indexer: with at most ``index_topk`` (512) compressed
entries the selection is every visible entry, past it each query attends to its top 512 only (ratio 2: 1024
entries at 2048 tokens; ratio 1: 1024 entries at 1024 tokens). Chunked cases exercise the window tail, the
compressed-entry append and the index keys carried between chunks.

Selection is by threshold on bf16 index scores, so a query whose 512th and 513th entries score within bf16
resolution can pick the other one; those cost a little PCC, which is why the floor is below a dense layer's.

The converted ttnn weights are cached under ``CACHE_DIR`` (``$DEEPSEEK_V4_CACHE_DIR``, default ``../cache``, per checkpoint).

    pytest models/experimental/deepseek_v41_flash/tests/prefill/test_attention.py
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tests.reference import load_reference, reference_attentions
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    attention_weights,
    load_config,
)
from models.experimental.deepseek_v41_flash.tt.prefill.attention import DeepSeekV41PrefillAttention, PrefillShared
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

PCC = 0.97


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "layers, seq_len, chunk",
    (
        ((0,), 768, 768),  # window only; the window starts mid-prompt for most queries
        ((0,), 768, 256),  # ... and its tail crosses chunk boundaries
        ((2, 3), 512, 512),  # ratio 2: kv + index source, consumer; 256 entries, selection is the identity
        ((2, 3, 8), 2048, 2048),  # ratio 2, top-k drops entries; a second group (layer 8 is its own source)
        ((2, 3), 1536, 512),  # ratio 2, chunked; selection starts in the last chunk (768 entries)
        ((20, 21, 24, 25), 1024, 1024),  # ratio 1: source, consumer, index-only source and its consumer
        ((14, 15, 20, 21), 1024, 256),  # ratio 2 -> ratio 1 groups, chunked
    ),
    ids=("window", "window_chunked", "ratio2", "ratio2_topk", "ratio2_chunked", "ratio1", "ratio_switch_chunked"),
)
@torch.no_grad()
def test_attention_prefill(device, layers, seq_len, chunk):
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    reference = reference_attentions(load_reference(snapshot), snapshot, loader, layers, seq_len)

    cache = WeightCache(CACHE_DIR)
    attns = {}
    for i in layers:
        # One table per kind of layer, as the model builds it: window-only layers use the plain theta.
        kind = "main" if config.compress_ratios[i] == 0 else "compress"
        rope = {kind: (reference[i].freqs_cis.real, reference[i].freqs_cis.imag)}
        attns[i] = DeepSeekV41PrefillAttention(
            config,
            i,
            attention_weights(loader, i),
            device,
            rope,
            cache=cache.sub(f"layers.{i}.attn"),
            weight_dtype=ttnn.bfloat8_b,
        )

    torch.manual_seed(0)
    hidden = torch.randn(len(layers), seq_len, config.hidden_size).bfloat16().float()
    expected = {i: reference[i](hidden[j : j + 1].clone(), 0)[0] for j, i in enumerate(layers)}

    states = {i: attns[i].new_state() for i in layers}
    actual = {i: [] for i in layers}
    for start in range(0, seq_len, chunk):
        shared = PrefillShared()
        for j, i in enumerate(layers):
            x = ttnn.from_torch(
                hidden[j, start : start + chunk].reshape(1, 1, chunk, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
            out = ttnn.to_torch(attns[i](x, states[i], shared)).reshape(chunk, -1).float()
            actual[i].append(out)
            _, pcc = comp_pcc(expected[i][start : start + chunk], out, PCC)
            logger.info(f"layer {i} (ratio {config.compress_ratios[i]}) chunk at {start}: PCC {pcc}")
        if shared.entry_sel is not None:
            ttnn.deallocate(shared.entry_sel)

    for i in layers:
        passing, pcc = comp_pcc(expected[i], torch.cat(actual[i]), PCC)
        logger.info(f"layer {i} (ratio {config.compress_ratios[i]}): PCC {pcc}")
        assert passing, f"layer {i}: PCC {pcc} < {PCC}"
