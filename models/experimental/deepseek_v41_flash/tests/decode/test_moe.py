# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode PCC of ``DeepSeekV41SparseMoeBlock`` against the checkpoint's own reference, on real weights.

Each case decodes random tokens one at a time through one layer's MoE. Layer 39 has the
largest router bias (~57), so it is the one the bf16 ranking is most likely to get wrong.
The routed experts are bfloat4_b, the shared expert bfloat8_b.

Set ``DEEPSEEK_V41_CACHE_DIR`` to keep the converted ttnn weights (~8.5 GB per layer) across runs.

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_moe.py
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tests.reference import load_reference, reference_moe
from models.experimental.deepseek_v41_flash.tt.config import (
    DEFAULT_MODEL_DIR,
    expert_provider,
    load_config,
    moe_weights,
)
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts, DeepSeekV41SparseMoeBlock
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

PCC = 0.95
NUM_TOKENS = 16


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("layer", (2, 39))
@torch.no_grad()
def test_moe_decode(device, layer):
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    reference = reference_moe(load_reference(snapshot), snapshot, loader, layer)

    cache = WeightCache(os.environ.get("DEEPSEEK_V41_CACHE_DIR")).sub(f"layers.{layer}.ffn")
    experts = DeepSeekV41PreloadedExperts(
        config, expert_provider(loader, layer), device, dtype=ttnn.bfloat4_b, cache=cache
    )
    moe = DeepSeekV41SparseMoeBlock(config, moe_weights(loader, layer), device, experts, cache=cache)

    torch.manual_seed(0)
    hidden = torch.randn(NUM_TOKENS, config.hidden_size)
    expected = reference(hidden.reshape(NUM_TOKENS, 1, -1)).reshape(NUM_TOKENS, -1)
    # all_gather_for_matmul takes the decode row width-sharded in L1.
    hidden_config = width_sharded_l1_config(1, config.hidden_size, device, tile_height=1)
    outs = []
    for t in range(NUM_TOKENS):
        out = moe.decode_static(
            ttnn.from_torch(
                hidden[t].reshape(1, 1, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=hidden_config,
            )
        )
        outs.append(ttnn.to_torch(out).reshape(-1).float())
        _, pcc = comp_pcc(expected[t], outs[-1], PCC)
        logger.info(f"layer {layer} token {t}: PCC {pcc}")

    passing, pcc = comp_pcc(expected, torch.stack(outs), PCC)
    logger.info(f"layer {layer}: PCC {pcc}")
    assert passing, f"layer {layer}: PCC {pcc} < {PCC}"
