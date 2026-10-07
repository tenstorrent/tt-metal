# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The LTX transformer weight cache is keyed by weight LAYOUT: a Ring-topology model fuses the attention gate
into the QKV projection (``fuse_gate``), which changes the cached tensor shapes/count, so it must not share
``transformer/`` with the Linear layout. No device: the subfolder choice is a pure function of the blocks."""

from types import SimpleNamespace

from models.tt_dit.models.transformers.ltx.transformer_ltx import LTXTransformerModel


def _model(fuse_flags):
    blocks = [SimpleNamespace(attn1=SimpleNamespace(fuse_gate=f)) for f in fuse_flags]
    return SimpleNamespace(transformer_blocks=blocks)


def test_linear_layout_keeps_legacy_subfolder():
    assert LTXTransformerModel.weight_cache_subfolder(_model([False, False])) == "transformer"


def test_fused_gate_layout_gets_its_own_subfolder():
    assert LTXTransformerModel.weight_cache_subfolder(_model([True, True])) == "transformer_fusedgate"


def test_any_fused_block_selects_the_fused_subfolder():
    assert LTXTransformerModel.weight_cache_subfolder(_model([False, True])) == "transformer_fusedgate"


def test_blocks_without_the_attribute_count_as_unfused():
    blocks = [SimpleNamespace(attn1=SimpleNamespace())]
    assert LTXTransformerModel.weight_cache_subfolder(SimpleNamespace(transformer_blocks=blocks)) == "transformer"
