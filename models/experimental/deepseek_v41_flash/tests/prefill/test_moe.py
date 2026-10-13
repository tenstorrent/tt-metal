# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill PCC of ``DeepSeekV41PrefillMoE`` against the checkpoint's own ``MoE``, on real weights.

* ``test_moe_prefill_router_shared`` -- the block's own parts, which do not touch the routed op: the router
  (``route``: the scores against the reference's, the ranking row's top 6 against the experts the reference
  picks) and the shared expert (clamped like the routed ones). The routed weights are replaced by a stand-in
  that only reports the shape the block checks, so no experts are converted. Layer 39 has the largest router
  bias (~57), the one the bf16 ranking is most likely to get wrong.
* ``test_moe_prefill`` -- the whole block over a chunk, the routed experts included
  (``tests/prefill/test_fused_experts_prefill.py`` covers that op alone at V4.1's shapes).

The converted ttnn weights are cached under ``CACHE_DIR`` (``$DEEPSEEK_V4_CACHE_DIR``, default ``../cache``, per checkpoint).

    pytest models/experimental/deepseek_v41_flash/tests/prefill/test_moe.py
"""

import types

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tests.reference import load_reference, reference_moe
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    DEFAULT_MODEL_DIR,
    expert_provider,
    load_config,
    moe_weights,
)
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts
from models.experimental.deepseek_v41_flash.tt.prefill.moe import DeepSeekV41PrefillMoE
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

SCORES_PCC = 0.99
SHARED_PCC = 0.99
# Mean fraction of each token's 6 experts the bf16 ranking shares with the fp32 reference's.
MIN_ROUTING_OVERLAP = 0.9
PCC = 0.95


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


def _hidden(config, num_tokens: int) -> torch.Tensor:
    torch.manual_seed(num_tokens)
    return torch.randn(num_tokens, config.hidden_size).bfloat16().float()


def _to_device(x: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(x.reshape(1, 1, *x.shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("seq_len", (128, 1000))
@pytest.mark.parametrize("layer", (2, 39))
@torch.no_grad()
def test_moe_prefill_router_shared(device, layer, seq_len):
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    reference = reference_moe(load_reference(snapshot), snapshot, loader, layer)

    # The block checks only these of its routed experts at construction.
    experts = types.SimpleNamespace(
        tp_size=1, top_k=config.num_experts_per_tok, gate_up_weights=[None] * config.n_routed_experts
    )
    moe = DeepSeekV41PrefillMoE(
        config,
        moe_weights(loader, layer),
        device,
        experts=experts,
        cache=WeightCache(CACHE_DIR).sub(f"layers.{layer}.ffn"),
    )

    hidden = _hidden(config, seq_len)
    x = _to_device(hidden, device)

    routing = moe.route(x)
    scores = ttnn.to_torch(routing.scores).reshape(seq_len, -1).float()
    expected_scores = F.softplus(F.linear(hidden, reference.gate.weight) / reference.gate.gate_temp).sqrt()
    passing, pcc = comp_pcc(expected_scores, scores, SCORES_PCC)
    logger.info(f"layer {layer}: router scores PCC {pcc}")
    assert passing, f"layer {layer}: router scores PCC {pcc} < {SCORES_PCC}"

    ranking = ttnn.to_torch(routing.ranking).reshape(seq_len, -1).float()
    picked = ranking.topk(config.num_experts_per_tok, dim=-1).indices
    _, expected_picked = reference.gate(hidden)
    overlap = torch.tensor(
        [len(set(a.tolist()) & set(b.tolist())) / config.num_experts_per_tok for a, b in zip(picked, expected_picked)]
    ).mean()
    logger.info(f"layer {layer}: routing overlap {overlap:.4f}")
    assert overlap >= MIN_ROUTING_OVERLAP, f"layer {layer}: routing overlap {overlap:.4f} < {MIN_ROUTING_OVERLAP}"

    shared = ttnn.to_torch(moe.shared_experts(x)).reshape(seq_len, -1).float()
    passing, pcc = comp_pcc(reference.shared_experts(hidden), shared, SHARED_PCC)
    logger.info(f"layer {layer}: shared expert PCC {pcc}")
    assert passing, f"layer {layer}: shared expert PCC {pcc} < {SHARED_PCC}"


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("seq_len", (128, 512))
@pytest.mark.parametrize("layer", (2, 39))
@torch.no_grad()
def test_moe_prefill(device, layer, seq_len):
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    reference = reference_moe(load_reference(snapshot), snapshot, loader, layer)

    cache = WeightCache(CACHE_DIR).sub(f"layers.{layer}.ffn")
    experts = DeepSeekV41PreloadedExperts(
        config, expert_provider(loader, layer), device, dtype=ttnn.bfloat4_b, cache=cache
    )
    moe = DeepSeekV41PrefillMoE(config, moe_weights(loader, layer), device, experts=experts, cache=cache)

    hidden = _hidden(config, seq_len)
    expected = reference(hidden.reshape(seq_len, 1, -1)).reshape(seq_len, -1)
    out = ttnn.to_torch(moe(_to_device(hidden, device))).reshape(seq_len, -1).float()
    passing, pcc = comp_pcc(expected, out, PCC)
    logger.info(f"layer {layer}: PCC {pcc}")
    assert passing, f"layer {layer}: PCC {pcc} < {PCC}"
