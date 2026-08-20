# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN MoE block vs the HuggingFace reference. Router first, experts after.

The router is tested on its own because it is the one place in this model where
a *discrete* disagreement is possible. Everything else degrades smoothly with
precision; top-k selection does not. HF softmaxes over all 128 experts in fp32
and we run bf16, so two experts with near-equal probability can swap places.
When that happens the token is routed to a genuinely different expert and the
MoE output for that token is unrelated to the reference -- a handful of such
tokens drags whole-block PCC down in a way that looks like a numerics problem
but is actually a selection problem.

``test_router_selection_matches`` separates the two by comparing the chosen
expert *sets* directly, so a later PCC dip can be attributed rather than
guessed at.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc

from ..tt.functional_decoder import (
    MoEConfig,
    build_expert_sparsity,
    moe_prefill,
    router_forward,
    upload_expert_weights,
    upload_router_weight,
)
from ..tt.weight_mapping import convert_moe_weights
from .reference import build_reference_layer, layer_state_dict

LAYER_IDX = 0
PCC_REQUIRED = 0.99


@pytest.fixture(scope="module")
def reference():
    return build_reference_layer(LAYER_IDX)


@pytest.fixture(scope="module")
def torch_weights():
    return convert_moe_weights(layer_state_dict(LAYER_IDX), n_experts=128)


def _hidden(config, seq_len, seed=0):
    torch.manual_seed(seed)
    return torch.randn(1, seq_len, config.hidden_size, dtype=torch.float32) * 0.02


def _reference_router(layer, hidden):
    """Return ``(dense [S, E], indices [S, k])`` from the reference router."""
    flat = hidden.view(-1, hidden.shape[-1])
    with torch.no_grad():
        _, scores, indices = layer.mlp.gate(flat)
    dense = torch.zeros(flat.shape[0], 128, dtype=scores.dtype)
    dense.scatter_(-1, indices, scores)
    return dense, indices


def _run_router(mesh_device, hf_config, torch_weights, hidden):
    w = upload_router_weight(torch_weights["router"], mesh_device)
    tt_in = ttnn.from_torch(
        hidden.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = router_forward(tt_in, w, MoEConfig.from_hf(hf_config))
    return ttnn.to_torch(out).reshape(-1, 128).float()


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 128], ids=["s32", "s128"])
def test_router_dense_weights_vs_reference(mesh_device, reference, torch_weights, seq_len):
    layer, hf_config = reference
    hidden = _hidden(hf_config, seq_len)

    ref_dense, _ = _reference_router(layer, hidden)
    tt_dense = _run_router(mesh_device, hf_config, torch_weights, hidden)

    passing, pcc_message = comp_pcc(ref_dense, tt_dense, PCC_REQUIRED)
    logger.info(comp_allclose(ref_dense, tt_dense))
    logger.info(f"router dense seq={seq_len}: {pcc_message}")
    assert passing, f"router dense weights (seq={seq_len}) below {PCC_REQUIRED}: {pcc_message}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_router_selection_matches(mesh_device, reference, torch_weights):
    """How many tokens pick a different set of 8 experts than the reference.

    Reported explicitly rather than silently folded into PCC, because a
    selection flip and a numerics error need completely different fixes.

    The bound is measured, not guessed. Holding the softmax in fp32 and letting
    only the projection run in bf16 costs 5/128 tokens on this checkpoint (a
    host-side simulation of the same arithmetic agrees), so the floor is ~4%.
    The threshold sits at 10% to leave room for a different activation sample
    while still catching the regression that matters: dropping the softmax to
    bf16 sends this straight to ~65%.
    """
    seq_len = 128
    layer, hf_config = reference
    hidden = _hidden(hf_config, seq_len)

    ref_dense, ref_indices = _reference_router(layer, hidden)
    tt_dense = _run_router(mesh_device, hf_config, torch_weights, hidden)
    tt_indices = tt_dense.topk(8, dim=-1).indices

    mismatched = 0
    worst_missed_weight = 0.0
    for token in range(seq_len):
        ref_set = set(ref_indices[token].tolist())
        tt_set = set(tt_indices[token].tolist())
        if ref_set != tt_set:
            mismatched += 1
            # How much routing weight the reference put on experts we skipped.
            # Near-ties sit at the bottom of the top-8, so this should be small;
            # a large value means a genuinely wrong selection, not rounding.
            for e in ref_set - tt_set:
                worst_missed_weight = max(worst_missed_weight, float(ref_dense[token, e]))

    logger.info(
        f"router selection: {mismatched}/{seq_len} tokens differ from the fp32 reference; "
        f"largest missed routing weight {worst_missed_weight:.4f}"
    )
    assert mismatched <= seq_len * 0.10, (
        f"{mismatched}/{seq_len} tokens routed to a different expert set -- "
        "far above the ~4% bf16-projection floor. Check that the router softmax "
        "and topk are still running in fp32 before suspecting anything else."
    )


def _upload_dense(dense: torch.Tensor, mesh_device):
    return ttnn.from_torch(
        dense.reshape(1, 1, *dense.shape).float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 128], ids=["s32", "s128"])
def test_experts_with_reference_routing(mesh_device, reference, torch_weights, seq_len):
    """Expert math alone, with the router's own selection taken out of play.

    The reference's exact fp32 routing weights are fed straight to our experts,
    so any shortfall here is the sparse_matmul / SwiGLU / reduce path and not
    the ~5/128 near-tie selection differences the router legitimately has.
    Isolating the two is what makes the end-to-end number interpretable.
    """
    layer, hf_config = reference
    config = MoEConfig.from_hf(hf_config)
    hidden = _hidden(hf_config, seq_len)

    with torch.no_grad():
        ref_out = layer.mlp(hidden)

    ref_dense, _ = _reference_router(layer, hidden)
    weights = upload_expert_weights(torch_weights, mesh_device, config)
    sparsity = build_expert_sparsity(mesh_device, config.num_experts)

    tt_in = ttnn.from_torch(
        hidden.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out = moe_prefill(tt_in, _upload_dense(ref_dense, mesh_device), weights, config, sparsity)
    tt_out_torch = ttnn.to_torch(tt_out).squeeze(0)

    passing, pcc_message = comp_pcc(ref_out, tt_out_torch, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_out_torch))
    logger.info(f"experts (reference routing) seq={seq_len}: {pcc_message}")
    assert passing, f"experts (seq={seq_len}) below {PCC_REQUIRED}: {pcc_message}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
@pytest.mark.parametrize("seq_len", [32, 128], ids=["s32", "s128"])
def test_moe_block_end_to_end(mesh_device, reference, torch_weights, seq_len):
    """Our router driving our experts, against the reference MoE block."""
    layer, hf_config = reference
    config = MoEConfig.from_hf(hf_config)
    hidden = _hidden(hf_config, seq_len)

    with torch.no_grad():
        ref_out = layer.mlp(hidden)

    w_router = upload_router_weight(torch_weights["router"], mesh_device)
    weights = upload_expert_weights(torch_weights, mesh_device, config)
    sparsity = build_expert_sparsity(mesh_device, config.num_experts)

    tt_in = ttnn.from_torch(
        hidden.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    routing = router_forward(tt_in, w_router, config)
    tt_out = moe_prefill(tt_in, routing, weights, config, sparsity)
    tt_out_torch = ttnn.to_torch(tt_out).squeeze(0)

    passing, pcc_message = comp_pcc(ref_out, tt_out_torch, PCC_REQUIRED)
    logger.info(comp_allclose(ref_out, tt_out_torch))
    logger.info(f"MoE block end-to-end seq={seq_len}: {pcc_message}")
    assert passing, f"MoE block (seq={seq_len}) below {PCC_REQUIRED}: {pcc_message}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], ids=["1x1"], indirect=True)
def test_router_weights_sum_to_one(mesh_device, reference, torch_weights):
    """norm_topk_prob=True, so each token's 8 weights must renormalise to 1."""
    layer, hf_config = reference
    hidden = _hidden(hf_config, 32)
    tt_dense = _run_router(mesh_device, hf_config, torch_weights, hidden)

    sums = tt_dense.sum(dim=-1)
    nonzero = (tt_dense > 0).sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=2e-2), f"router weights not normalised: {sums[:4]}"
    assert (nonzero == 8).all(), f"expected exactly 8 active experts per token, got {nonzero.unique().tolist()}"
