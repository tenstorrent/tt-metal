# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.hf_eager.hunyuanvideo_1_5._stubs import hunyuan_video15_transformer_block as block
from models.demos.hf_eager.hunyuanvideo_1_5.tests.pcc._reference_loader import load_reference_model


def _to_torch(tensor):
    return ttnn.to_torch(tensor).float()


def test_qkv_split_matches_legacy_and_reference(device, monkeypatch):
    """Focused PCC gate for tt_dit's fused projection+split QKV path."""
    torch.manual_seed(0)
    model = load_reference_model("tencent/HunyuanVideo-1.5")
    reference_block = model.transformer_blocks[0]
    width = model.config.num_attention_heads * model.config.attention_head_dim

    hidden = torch.randn(1, 64, width)
    encoder = torch.randn(1, 32, width)
    temb = torch.randn(1, width)
    mask = torch.ones(1, 96, dtype=torch.bool)
    with torch.no_grad():
        expected_hidden, expected_encoder = reference_block(
            hidden_states=hidden,
            encoder_hidden_states=encoder,
            temb=temb,
            attention_mask=mask,
        )

    tt_hidden = ttnn.from_torch(hidden, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    monkeypatch.setenv("HY_DIT_QKV_SPLIT", "0")
    legacy = block.build(device, reference_block)
    legacy_hidden, legacy_encoder = legacy(tt_hidden, encoder_hidden_states=encoder, temb=temb)

    monkeypatch.setenv("HY_DIT_QKV_SPLIT", "1")
    optimized = block.build(device, reference_block)
    optimized_hidden, optimized_encoder = optimized(tt_hidden, encoder_hidden_states=encoder, temb=temb)

    legacy_hidden = _to_torch(legacy_hidden)
    legacy_encoder = _to_torch(legacy_encoder)
    optimized_hidden = _to_torch(optimized_hidden)
    optimized_encoder = _to_torch(optimized_encoder)

    for name, expected, actual in (
        ("hidden/reference", expected_hidden, optimized_hidden),
        ("encoder/reference", expected_encoder, optimized_encoder),
        ("hidden/legacy", legacy_hidden, optimized_hidden),
        ("encoder/legacy", legacy_encoder, optimized_encoder),
    ):
        ok, achieved = comp_pcc(expected, actual, 0.99)
        assert ok, f"{name} PCC {achieved} below 0.99"
