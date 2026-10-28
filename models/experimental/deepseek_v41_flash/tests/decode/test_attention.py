# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode PCC of ``DeepSeekV41Attention`` against the checkpoint's own reference, on real weights.

A case decodes a group of layers token by token from position 0, each layer on its own
random input and in layer order -- the order their shared compressed KV / top-k needs.
The reference (``tests/reference.py``) steps the same layers the same way.

``test_attention_decode_s5`` feeds the same positions as ``S``-row ROW_MAJOR blocks, ``hidden``
``[1, S, 1, D]``, through ``decode`` (DSpark's parallel verify), against the same token-by-token reference.

The converted ttnn weights are cached under ``CACHE_DIR`` (``$DEEPSEEK_V4_CACHE_DIR``, default ``../cache``, per checkpoint).

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_attention.py
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
from models.experimental.deepseek_v41_flash.tt.decode.attention import DeepSeekV41Attention, build_decode_caches
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

PCC = 0.95
# DSpark's block: the token plus four drafts.
S = 5


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _build(device, layers, seq_len):
    """The checkpoint's config, the fp32 reference layers, the device layers and their RoPE tables."""
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    reference = reference_attentions(load_reference(snapshot), snapshot, loader, layers, seq_len)
    cache = WeightCache(CACHE_DIR)
    attns = {
        i: DeepSeekV41Attention(
            config,
            i,
            attention_weights(loader, i),
            device,
            cache=cache.sub(f"layers.{i}.attn"),
            weight_dtype=ttnn.bfloat4_b,
        )
        for i in layers
    }
    rope = {i: (reference[i].freqs_cis.real, reference[i].freqs_cis.imag) for i in layers}
    return config, reference, attns, rope


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "layers, seq_len",
    (
        ((0,), 160),  # window only; the ring wraps
        ((2, 3), 160),  # ratio 2: kv + index source, consumer
        ((20, 21, 24), 576),  # ratio 1: source, consumer, index-only source; top-k drops 64 of 576
    ),
    ids=("window", "ratio2", "ratio1"),
)
@torch.no_grad()
def test_attention_decode(device, layers, seq_len):
    config, reference, attns, rope = _build(device, layers, seq_len)
    caches = build_decode_caches(device, config, layers, seq_len)

    torch.manual_seed(0)
    hidden = torch.randn(len(layers), seq_len, config.hidden_size)
    # all_gather_for_matmul takes the decode row width-sharded in L1.
    hidden_config = width_sharded_l1_config(1, config.hidden_size, device, tile_height=1)
    window = config.sliding_window
    checked = {0, 1, window - 2, window - 1, window} | set(range(seq_len - 4, seq_len))
    for pos in range(seq_len):
        for j, i in enumerate(layers):
            x = hidden[j, pos].reshape(1, 1, -1)
            expected = reference[i](x.clone(), pos)
            out = attns[i].decode(
                ttnn.from_torch(
                    x.reshape(1, 1, 1, -1),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=hidden_config,
                ),
                scache=caches[i],
                **attns[i].decode_inputs(pos, *rope[i]),
            )
            if pos not in checked:
                continue
            passing, pcc = comp_pcc(expected.reshape(-1), ttnn.to_torch(out).reshape(-1).float(), PCC)
            logger.info(f"layer {i} (ratio {config.compress_ratios[i]}) pos {pos}: PCC {pcc}")
            assert passing, f"layer {i} pos {pos}: PCC {pcc} < {PCC}"


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize(
    "layers, seq_len",
    (
        # Blocks start at multiples of 5: both ratio-2 parities, one block across the causal
        # switch (125..129 holds window - 1) and the ring wrap.
        ((0,), 160),  # window only
        ((2, 3), 160),  # ratio 2: kv + index source, consumer
        ((20, 21, 24), 580),  # ratio 1: source, consumer, index-only source; top-k drops 68 of 580
    ),
    ids=("window", "ratio2", "ratio1"),
)
@torch.no_grad()
def test_attention_decode_s5(device, layers, seq_len):
    assert seq_len % S == 0
    config, reference, attns, rope = _build(device, layers, seq_len)
    caches = build_decode_caches(device, config, layers, seq_len, verify_rows=S)

    torch.manual_seed(0)
    hidden = torch.randn(len(layers), seq_len, config.hidden_size)
    # The block's rows on every core of one width-sharded L1 row group, as decode's all_gather_for_matmul takes them.
    hidden_config = width_sharded_l1_config(S, config.hidden_size, device, tile_height=1)
    window = config.sliding_window
    checked = {0, 1, window - 2, window - 1, window} | set(range(seq_len - 4, seq_len))
    for start in range(0, seq_len, S):
        positions = range(start, start + S)
        # The reference shares the group's compressed state, so it steps position-major.
        expected = {i: [] for i in layers}
        for pos in positions:
            for j, i in enumerate(layers):
                expected[i].append(reference[i](hidden[j, pos].reshape(1, 1, -1).clone(), pos))
        for j, i in enumerate(layers):
            out = attns[i].decode(
                ttnn.from_torch(
                    hidden[j, start : start + S].reshape(1, S, 1, -1),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=hidden_config,
                ),
                scache=caches[i],
                **attns[i].decode_inputs(start, *rope[i], rows=S),
            )
            assert list(out.shape) == [1, S, 1, config.hidden_size]
            if checked.isdisjoint(positions):
                continue
            got = ttnn.to_torch(out).reshape(S, -1).float()
            for k, pos in enumerate(positions):
                passing, pcc = comp_pcc(expected[i][k].reshape(-1), got[k], PCC)
                logger.info(f"layer {i} (ratio {config.compress_ratios[i]}) pos {pos} (row {k}): PCC {pcc}")
                assert passing, f"layer {i} pos {pos} (row {k} of the block at {start}): PCC {pcc} < {PCC}"
