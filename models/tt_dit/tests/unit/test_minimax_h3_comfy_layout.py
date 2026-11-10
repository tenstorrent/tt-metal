# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for loading MiniMax H3 weights in ComfyUI's layout (full and pruned).

The fixture holds the safetensors headers (name -> shape) of Comfy-Org's pruned and full fl2va files and of
MiniMaxAI's diffusers transformer shards, so the mapping is checked against the real files without weights.
"""

import json
from pathlib import Path

import pytest
import torch

from models.tt_dit.models.transformers.minimax_h3 import convert_comfy_minimax_h3 as cc

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "minimax_h3_headers.json").read_text())
DIFFUSERS = FIXTURE["diffusers"]
TIME_EMBEDDER = {k for k in DIFFUSERS if k.startswith("time_embedder.")}
ADALN_WEIGHTS = {k for k in DIFFUSERS if k.endswith("adaln_proj.linear.weight")} | {"norm_out.linear.weight"}


def test_full_comfy_header_maps_onto_diffusers_exactly():
    mapped = cc.map_comfy_shapes(FIXTURE["comfy_full"])
    assert mapped == DIFFUSERS


def test_pruned_comfy_header_maps_onto_diffusers():
    pruned = FIXTURE["comfy_pruned"]
    assert cc.is_comfy_layout(pruned) and cc.is_pruned_layout(pruned)
    mapped = cc.map_comfy_shapes(pruned)

    # Every pruned tensor lands on a diffusers name, except the curve table that replaces the time embedder.
    assert set(mapped) - set(DIFFUSERS) == {cc.CURVE_TABLE_KEY}
    # The only diffusers tensors with no pruned counterpart are the time embedder's.
    assert set(DIFFUSERS) - set(mapped) == TIME_EMBEDDER
    assert len(TIME_EMBEDDER) == 4

    grid, rank = mapped[cc.CURVE_TABLE_KEY]
    assert (grid, rank) == (1025, 8)
    assert len(ADALN_WEIGHTS) == 51
    for name in set(mapped) & set(DIFFUSERS):
        if name in ADALN_WEIGHTS:
            # The AdaLN linears take the R-dim curve instead of the 2688-dim time embedding.
            assert mapped[name] == [DIFFUSERS[name][0], rank], name
            assert DIFFUSERS[name][1] == 2688, name
        else:
            assert mapped[name] == DIFFUSERS[name], name


def test_convert_splits_stacked_qkv_and_swaps_gated_ff():
    q, k, v = (torch.full((4, 3), float(i)) for i in range(3))
    gate, value = torch.zeros(6, 3), torch.ones(6, 3)
    state = {
        "blocks.7.attn.qkv_proj.weight": torch.cat([q, k, v]),
        "blocks.7.mlp.fc1.weight": torch.cat([gate, value]),
        "token_refiner.blocks.1.attn.out_proj.weight": torch.eye(2),
        "final_layer.adaln_proj.linear.bias": torch.arange(4.0),
        "rope.inv_freq": torch.ones(16),
        "adaln_t_table": torch.zeros(1025, 8),
    }
    out = cc.convert_comfy_state_dict(state)
    assert not state
    assert set(out) == {
        "transformer_blocks.7.attn.to_q.weight",
        "transformer_blocks.7.attn.to_k.weight",
        "transformer_blocks.7.attn.to_v.weight",
        "transformer_blocks.7.ff.net.0.proj.weight",
        "token_refiner.refiner_blocks.1.attn.to_out.0.weight",
        "norm_out.linear.bias",
        "adaln_t_table",
    }
    for name, ref in zip("qkv", (q, k, v)):
        assert torch.equal(out[f"transformer_blocks.7.attn.to_{name}.weight"], ref)
    assert torch.equal(out["transformer_blocks.7.ff.net.0.proj.weight"], torch.cat([value, gate]))


def _comfy_curve(table, t):
    # comfy/ldm/minimax/model.py, written out independently.
    g = table.shape[0]
    pos = t.clamp(0, 1) * (g - 1)
    i0 = pos.floor().long().clamp(max=g - 2)
    w = (pos - i0).unsqueeze(-1)
    return table[i0] * (1 - w) + table[i0 + 1] * w


def test_curve_lookup_matches_comfy_and_hits_grid_points():
    torch.manual_seed(0)
    table = torch.randn(1025, 8)
    t = torch.tensor([0.0, 1e-4, 0.0015, 0.25, 0.5, 0.731, 0.999, 1.0, 1.5, -0.1])
    torch.testing.assert_close(cc.adaln_curve_embedding(table, t), _comfy_curve(table, t), rtol=0, atol=1e-6)
    grid_t = torch.arange(1025) / 1024
    torch.testing.assert_close(cc.adaln_curve_embedding(table, grid_t), table, rtol=0, atol=1e-6)


@pytest.mark.parametrize("dim", [0, -1])
def test_pad_curve_keeps_product(dim):
    torch.manual_seed(0)
    emb, weight = torch.randn(3, 8), torch.randn(5, 8)
    pe, pw = cc.pad_curve(emb), cc.pad_curve(weight)
    assert pe.shape == (3, cc.CURVE_PAD) and pw.shape == (5, cc.CURVE_PAD)
    torch.testing.assert_close(pe @ pw.T, emb @ weight.T)
    assert cc.pad_curve(torch.randn(8, 2), dim=0).shape == (cc.CURVE_PAD, 2)
