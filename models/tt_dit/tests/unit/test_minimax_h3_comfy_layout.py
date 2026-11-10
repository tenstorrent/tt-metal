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


# 10Eros-Max beta7_fix1 header (final layer and metadata, read on f102): 32 stacked heads per projection,
# head 0 full, heads 1..31 offsets from it; 8 forwards use heads 0, 4, ..., 28.
BETA7_ROWS = {"proj_out": 3072, "audio_proj_out": 1024}
BETA7_CHANNELS = {"proj_out": 96, "audio_proj_out": 32}
BETA7_METADATA = {
    "pdd_head_storage": "head0-full;heads1-31-offset-from-head0;fp32",
    "source_inference_contract": json.dumps(
        {
            "num_inference_steps": 8,
            "transformer_forwards": 8,
            "pdd_steps": 32,
            "pdd_step_indices": list(range(0, 33, 4)),
        }
    ),
}
BETA7_HEADS = [0, 4, 8, 12, 16, 20, 24, 28]


def test_beta7_header_has_32_heads_and_an_8_step_contract(expect_error):
    counts = {cc.num_step_heads(BETA7_ROWS[n], BETA7_CHANNELS[n]) for n in cc.STEP_HEAD_KEYS}
    assert counts == {32}
    assert cc.step_head_contract(BETA7_METADATA, 32) == BETA7_HEADS
    assert cc.step_head_contract({}, 32) == BETA7_HEADS
    with expect_error(ValueError, "out of range"):
        cc.step_head_contract({"source_inference_contract": json.dumps({"pdd_step_indices": [0, 40]})}, 32)


def _stacked(heads: int, channels: int, hidden: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    full = torch.randn(heads, channels, hidden, generator=g) * 0.08
    bias = torch.randn(heads, channels, generator=g) * 0.08
    # Stored form: head 0 in full, the rest as offsets from it, in bf16 like the file.
    stored_w = torch.cat([full[:1], full[1:] - full[:1]]).reshape(heads * channels, hidden).bfloat16()
    stored_b = torch.cat([bias[:1], bias[1:] - bias[:1]]).reshape(heads * channels).bfloat16()
    return full, bias, stored_w, stored_b


def test_select_step_heads_rebuilds_offset_heads_in_contract_order():
    hidden = 64
    state, expected = {}, {}
    for i, name in enumerate(cc.STEP_HEAD_KEYS):
        c = BETA7_CHANNELS[name]
        full, bias, w, b = _stacked(32, c, hidden, seed=i)
        state[f"{name}.weight"], state[f"{name}.bias"] = w, b
        expected[name] = (full, bias, w, b)
    assert cc.select_step_heads(state, BETA7_CHANNELS, BETA7_HEADS, BETA7_METADATA) == 8
    for name in cc.STEP_HEAD_KEYS:
        c = BETA7_CHANNELS[name]
        full, bias, w, b = expected[name]
        out_w, out_b = state[f"{name}.weight"], state[f"{name}.bias"]
        assert out_w.shape == (8 * c, hidden) and out_w.dtype == torch.bfloat16
        stored = w.float().reshape(32, c, hidden)
        stored_b = b.float().reshape(32, c)
        for slot, h in enumerate(BETA7_HEADS):
            want_w = stored[0] if h == 0 else stored[0] + stored[h]
            want_b = stored_b[0] if h == 0 else stored_b[0] + stored_b[h]
            assert torch.equal(out_w[slot * c : (slot + 1) * c], want_w.bfloat16())
            assert torch.equal(out_b[slot * c : (slot + 1) * c], want_b.bfloat16())
            # And it is the real head, up to bf16 rounding of the stored parts.
            assert torch.allclose(out_w[slot * c : (slot + 1) * c].float(), full[h], atol=2e-3)


def test_single_head_file_is_untouched():
    state = {
        "proj_out.weight": torch.randn(96, 16),
        "proj_out.bias": torch.randn(96),
        "audio_proj_out.weight": torch.randn(32, 16),
        "audio_proj_out.bias": torch.randn(32),
    }
    before = {k: v.clone() for k, v in state.items()}
    assert cc.select_step_heads(state, BETA7_CHANNELS, BETA7_HEADS, {}) == 1
    assert all(torch.equal(state[k], before[k]) for k in state)


def test_mixed_head_counts_are_refused(expect_error):
    state = {
        "proj_out.weight": torch.randn(3072, 8),
        "proj_out.bias": torch.randn(3072),
        "audio_proj_out.weight": torch.randn(32, 8),
        "audio_proj_out.bias": torch.randn(32),
    }
    with expect_error(ValueError, "disagree"):
        cc.select_step_heads(state, BETA7_CHANNELS, BETA7_HEADS, BETA7_METADATA)


@pytest.mark.parametrize("num_steps", [8, 6, 12])
def test_step_head_selector_picks_one_slot(num_steps):
    c, slots = 96, 8
    stacked = torch.randn(5, slots * c)
    used = []
    for step in range(num_steps):
        sel = cc.step_head_selector(step, num_steps, slots, c)
        assert sel.shape == (slots * c, c)
        picked = stacked @ sel
        slot = next(s for s in range(slots) if torch.equal(picked, stacked[:, s * c : (s + 1) * c]))
        used.append(slot)
    assert used == sorted(used) and used[0] == 0 and used[-1] == slots - 1
    if num_steps == slots:
        assert used == list(range(slots))
