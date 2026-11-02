# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode PCC of ``DeepSeekV41SparseMoeBlock`` against the checkpoint's own reference, on real weights.

Each case decodes random tokens one at a time through one layer's MoE. Layer 39 has the
largest router bias (~57), so it is the one the bf16 ranking is most likely to get wrong.
The routed experts are bfloat4_b, the shared expert bfloat8_b. ``test_moe_decode_s5`` feeds the same
kind of tokens as ``S``-row ROW_MAJOR blocks, ``hidden`` ``[1, S, 1, D]`` (DSpark's parallel verify):
one ``fused_experts`` call over the rows' union of experts, and the shared expert's own projections.

``test_moe_decode_tp`` decodes the same tokens tensor-parallel: TP4 on a 1x4 submesh (the decode stages'
layout) and TP12 on a 3x4 submesh, each chip holding an ``I / TP`` slice of every expert and the partials
all-reduced over both mesh axes. TP12 gives each chip 6 I-tiles, which ``fused_experts``'s 6-expert path runs on
6 of each expert's 16 gate_up cores. ``banked`` cuts the gate_up shards into K-pieces so every expert reads
from all 8 DRAM banks (``whole_k`` is the layout prefill also reads). Tokens after the first
``TP_WARMUP_TOKENS`` are bracketed by ``MOE_TP<tp>_START`` / ``_END`` Tracy signposts:

    python -m tracy -p -r -v -m pytest models/experimental/deepseek_v41_flash/tests/decode/test_moe.py -k "tp12 and banked"

``fused_router`` has ``fused_experts`` run the router on the grid row below its experts (one op for the whole
routed MoE); ``own_router`` runs it as its own ops and hands the op the score and ranking rows.

The converted ttnn weights (~8.5 GB per layer) are cached under ``CACHE_DIR`` (``$DEEPSEEK_V4_CACHE_DIR``,
default ``../cache``, per checkpoint).

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_moe.py
"""

import pytest
import torch
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
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts, DeepSeekV41SparseMoeBlock
from models.experimental.deepseek_v4_flash.tt.common import width_sharded_l1_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader, resolve_snapshot_dir

PCC = 0.95
NUM_TOKENS = 16
# DSpark's block: the token plus four drafts.
S = 5
NUM_BLOCKS = 3
PARENT_MESH = (8, 4)
TP_SUBMESHES = {4: (1, 4), 12: (3, 4)}
# Tokens decoded before the profiled ones, so every program is built.
TP_WARMUP_TOKENS = 4


def _snapshot():
    try:
        return resolve_snapshot_dir(DEFAULT_MODEL_DIR)
    except FileNotFoundError:
        return None


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("layer", (2, 39))
@pytest.mark.parametrize("fused_router", (True, False), ids=("fused_router", "own_router"))
@torch.no_grad()
def test_moe_decode(device, layer, fused_router):
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    reference = reference_moe(load_reference(snapshot), snapshot, loader, layer)

    cache = WeightCache(CACHE_DIR).sub(f"layers.{layer}.ffn")
    experts = DeepSeekV41PreloadedExperts(
        config, expert_provider(loader, layer), device, dtype=ttnn.bfloat4_b, cache=cache
    )
    moe = DeepSeekV41SparseMoeBlock(config, moe_weights(loader, layer), device, experts, cache=cache)
    assert moe.fused_router or not fused_router, "this layer's fused_experts cannot run the router"
    moe.fused_router = fused_router

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


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("layer", (2, 39))
@torch.no_grad()
def test_moe_decode_s5(device, layer):
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    reference = reference_moe(load_reference(snapshot), snapshot, loader, layer)

    cache = WeightCache(CACHE_DIR).sub(f"layers.{layer}.ffn")
    experts = DeepSeekV41PreloadedExperts(
        config, expert_provider(loader, layer), device, dtype=ttnn.bfloat4_b, cache=cache
    )
    moe = DeepSeekV41SparseMoeBlock(config, moe_weights(loader, layer), device, experts, cache=cache, verify_rows=S)

    torch.manual_seed(0)
    tokens = NUM_BLOCKS * S
    hidden = torch.randn(tokens, config.hidden_size)
    # The MoE is per token, so a block's rows match the same rows decoded one at a time.
    expected = reference(hidden.reshape(tokens, 1, -1)).reshape(tokens, -1)
    # The block's rows on every core of one width-sharded L1 row group, as all_gather_for_matmul takes them.
    hidden_config = width_sharded_l1_config(S, config.hidden_size, device, tile_height=1)
    outs = []
    for block in range(NUM_BLOCKS):
        rows = slice(block * S, (block + 1) * S)
        out = moe.decode_static(
            ttnn.from_torch(
                hidden[rows].reshape(1, S, 1, -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=hidden_config,
            )
        )
        assert list(out.shape) == [1, S, 1, config.hidden_size]
        outs.append(ttnn.to_torch(out).reshape(S, -1).float())
        for k in range(S):
            _, pcc = comp_pcc(expected[rows][k], outs[-1][k], PCC)
            logger.info(f"layer {layer} block {block} row {k}: PCC {pcc}")

    passing, pcc = comp_pcc(expected, torch.cat(outs), PCC)
    logger.info(f"layer {layer}: PCC {pcc}")
    assert passing, f"layer {layer}: PCC {pcc} < {PCC}"


@pytest.mark.skipif(_snapshot() is None, reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("tp_size", tuple(TP_SUBMESHES), ids=lambda tp: f"tp{tp}")
@pytest.mark.parametrize("balance_dram_banks", (False, True), ids=("whole_k", "banked"))
@pytest.mark.parametrize("layer", (39,))
@pytest.mark.parametrize("fused_router", (True, False), ids=("fused_router", "own_router"))
@torch.no_grad()
def test_moe_decode_tp(mesh_device, tp_size, balance_dram_banks, layer, fused_router):
    from tracy import signpost

    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"needs an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")
    device = mesh_device.create_submesh(ttnn.MeshShape(*TP_SUBMESHES[tp_size]), ttnn.MeshCoordinate(0, 0))
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    reference = reference_moe(load_reference(snapshot), snapshot, loader, layer)

    cache = WeightCache(CACHE_DIR).sub(f"layers.{layer}.ffn")
    experts = DeepSeekV41PreloadedExperts(
        config,
        expert_provider(loader, layer),
        device,
        dtype=ttnn.bfloat4_b,
        cache=cache,
        tp_size=tp_size,
        balance_dram_banks=balance_dram_banks,
    )
    moe = DeepSeekV41SparseMoeBlock(config, moe_weights(loader, layer), device, experts, cache=cache, tp_size=tp_size)
    assert moe.fused_router or not fused_router, "this layer's fused_experts cannot run the router"
    moe.fused_router = fused_router
    logger.info(f"TP{tp_size}: I per chip {experts.intermediate} (unpadded {config.moe_intermediate_size // tp_size})")

    torch.manual_seed(0)
    hidden = torch.randn(NUM_TOKENS, config.hidden_size)
    expected = reference(hidden.reshape(NUM_TOKENS, 1, -1)).reshape(NUM_TOKENS, -1)
    hidden_config = width_sharded_l1_config(1, config.hidden_size, device, tile_height=1)
    outs = []
    for t in range(NUM_TOKENS):
        x = ttnn.from_torch(
            hidden[t].reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=hidden_config,
        )
        profiled = t >= TP_WARMUP_TOKENS
        if profiled:
            signpost(f"MOE_TP{tp_size}_START")
        out = moe.decode_static(x)
        if profiled:
            signpost(f"MOE_TP{tp_size}_END")
        # Every chip holds the all-reduced row; check that they agree, then compare one.
        copies = [ttnn.to_torch(c).reshape(-1).float() for c in ttnn.get_device_tensors(out)]
        spread = max((c - copies[0]).abs().max().item() for c in copies)
        outs.append(copies[0])
        _, pcc = comp_pcc(expected[t], outs[-1], PCC)
        logger.info(f"TP{tp_size} layer {layer} token {t}: PCC {pcc}, max abs spread over the chips {spread}")

    passing, pcc = comp_pcc(expected, torch.stack(outs), PCC)
    logger.info(f"TP{tp_size} layer {layer}: PCC {pcc}")
    assert passing, f"TP{tp_size} layer {layer}: PCC {pcc} < {PCC}"
