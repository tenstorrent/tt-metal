# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""Relative-position multi-head self-attention (`RelPositionMultiHeadedAttention`)
and the reduced `ConformerEncoderLayer` it feeds -- see tt/flow/encoder.py's module
docstring for the verified real-source scope (macaron-FFN and conv-module both
absent at the real checkpoint's config; `rel_shift` run via a deliberate host
round-trip, not a hidden shortcut).

Same situation as the CFM estimator, HiFT and SineGen2: `matcha-tts`/`diffusers`/
`conformer`/`wenet`/`espnet` are not installed here, so there is no importable real
upstream class to call directly. `*Ref` classes are a line-by-line transcription of
the real source, using genuine `torch.nn`/`torch.nn.functional` primitives
throughout. This file also keeps the one closed-form check the real mechanism
enables permanently: that `rel_shift` actually recovers the relative-offset
alignment it claims to, checked against the mechanism's own mathematical
definition, independent of this port's code on either side.
"""

from __future__ import annotations

import pytest
import torch

from models.common.utility_functions import comp_pcc

GATE_BF16 = 0.99


# --------------------------------------------------------------------------
# host tier -- no device
# --------------------------------------------------------------------------
def test_rel_shift_recovers_relative_offset_alignment():
    """The claim the whole mechanism rests on: after `rel_shift`, `matrix_bd[i,j]`
    must equal `q_with_bias_v[i] . pe[offset = i-j]` -- computed here two
    independent ways (through the shift trick, and via a direct per-(i,j) dot
    product with no shift at all) and checked against each other. This is what
    was verified numerically before porting anything (see tt/flow/encoder.py's
    module docstring); kept here as a permanent regression test, not a one-off
    scratch check.
    """
    from models.demos.audio.cosyvoice2.tt.flow.encoder import rel_shift_torch, sinusoidal_rel_pos_table_torch

    torch.manual_seed(0)
    t_len, d, heads = 6, 8, 1
    pe = sinusoidal_rel_pos_table_torch(t_len, d)  # [1, 2T-1, d]
    q_bias_v = torch.randn(1, heads, t_len, d)

    matrix_bd = torch.matmul(q_bias_v, pe.unsqueeze(1).transpose(-2, -1))  # [1, H, T, 2T-1]
    shifted = rel_shift_torch(matrix_bd)  # [1, H, T, T]

    for i in range(t_len):
        for j in range(t_len):
            pe_idx = (t_len - 1) - (i - j)
            direct = torch.dot(q_bias_v[0, 0, i], pe[0, pe_idx])
            assert torch.allclose(shifted[0, 0, i, j], direct, atol=1e-5), (i, j, shifted[0, 0, i, j], direct)


def test_sinusoidal_rel_pos_table_matches_closed_form():
    """Index `t_len-1` (the table's center) is relative offset 0: `sin(0)=0` in
    the even channels, `cos(0)=1` in the odd channels -- an algebraic fact about
    the formula, independent of its own implementation."""
    from models.demos.audio.cosyvoice2.tt.flow.encoder import sinusoidal_rel_pos_table_torch

    t_len, d = 10, 16
    pe = sinusoidal_rel_pos_table_torch(t_len, d)
    center = pe[0, t_len - 1]
    assert torch.allclose(center[0::2], torch.zeros(d // 2), atol=1e-6)
    assert torch.allclose(center[1::2], torch.ones(d // 2), atol=1e-6)


def test_conformer_layer_torch_reference_shape_and_range():
    from models.demos.audio.cosyvoice2.tt.flow.encoder import ConformerEncoderLayerRef, sinusoidal_rel_pos_table_torch

    torch.manual_seed(1)
    layer = ConformerEncoderLayerRef()
    layer.eval()
    b, t_len, d = 1, 20, 512
    x = torch.randn(b, t_len, d) * 0.1
    mask = torch.ones(b, 1, t_len, dtype=torch.bool)
    pos_emb = sinusoidal_rel_pos_table_torch(t_len, d)
    with torch.no_grad():
        out = layer(x, mask, pos_emb)
    assert out.shape == (b, t_len, d)
    assert torch.isfinite(out).all()


# --------------------------------------------------------------------------
# device tier -- needs silicon
# --------------------------------------------------------------------------
needs_l1_small = pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)


@needs_l1_small
@pytest.mark.parametrize("t_len", [20, 64])
def test_device_rel_position_attention_matches_torch_reference(device, t_len):
    """`TtRelPositionMultiHeadedAttention` -- including the `rel_shift` host
    round-trip and the `(B,H,T,d_k)`-broadcast reformulation of the pos-bias add
    (see tt/flow/encoder.py's docstring for why that reformulation is equivalent,
    not just assumed) -- vs. the real, transcribed reference, isolated from the
    surrounding layer's norms/FFN."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.flow.encoder import (
        RelPositionMultiHeadedAttentionRef,
        TtRelPositionMultiHeadedAttention,
        sinusoidal_rel_pos_table_torch,
    )

    torch.manual_seed(t_len)
    attn = RelPositionMultiHeadedAttentionRef(8, 512)
    attn.eval()
    b, d = 1, 512
    x = torch.randn(b, t_len, d) * 0.1
    mask = torch.ones(b, 1, t_len, dtype=torch.bool)
    pos_emb = sinusoidal_rel_pos_table_torch(t_len, d)
    with torch.no_grad():
        want = attn(x, x, x, pos_emb, mask)

    tt_attn = TtRelPositionMultiHeadedAttention(device, attn)
    x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pos_dev = ttnn.from_torch(pos_emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    bias_dev = ttnn.from_torch(torch.zeros(b, 1, 1, t_len), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    got = ttnn.to_torch(tt_attn(x_dev, pos_dev, bias_dev)).float()
    assert got.shape == want.shape
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device RelPositionMultiHeadedAttention (T={t_len}) PCC {pcc}")
    assert passed, pcc


@needs_l1_small
def test_device_conformer_layer_matches_torch_reference(device):
    """`TtConformerEncoderLayer` (norm -> rel-pos self-attn -> residual -> norm ->
    FFN -> residual, the real checkpoint's reduced form) vs. the torch reference,
    random-init (no CosyVoice2 checkpoint yet)."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.flow.encoder import (
        ConformerEncoderLayerRef,
        TtConformerEncoderLayer,
        sinusoidal_rel_pos_table_torch,
    )

    torch.manual_seed(2)
    layer = ConformerEncoderLayerRef()
    layer.eval()
    b, t_len, d = 1, 32, 512
    x = torch.randn(b, t_len, d) * 0.1
    mask = torch.ones(b, 1, t_len, dtype=torch.bool)
    pos_emb = sinusoidal_rel_pos_table_torch(t_len, d)
    with torch.no_grad():
        want = layer(x, mask, pos_emb)

    tt_layer = TtConformerEncoderLayer(device, layer)
    x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pos_dev = ttnn.from_torch(pos_emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    bias_dev = ttnn.from_torch(torch.zeros(b, 1, 1, t_len), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    got = ttnn.to_torch(tt_layer(x_dev, pos_dev, bias_dev)).float()
    assert got.shape == want.shape
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device ConformerEncoderLayer PCC {pcc}")
    assert passed, pcc


@needs_l1_small
def test_device_conformer_layer_matches_torch_reference_stacked(device):
    """Two layers chained -- confirms weights/state genuinely isolate per
    instance (not a shared-buffer bug that a single-layer test couldn't catch),
    since the real encoder stacks 6 (then 4 more) of these."""
    import ttnn
    from models.demos.audio.cosyvoice2.tt.flow.encoder import (
        ConformerEncoderLayerRef,
        TtConformerEncoderLayer,
        sinusoidal_rel_pos_table_torch,
    )

    torch.manual_seed(3)
    layers = [ConformerEncoderLayerRef() for _ in range(2)]
    for layer in layers:
        layer.eval()
    b, t_len, d = 1, 32, 512
    x = torch.randn(b, t_len, d) * 0.1
    mask = torch.ones(b, 1, t_len, dtype=torch.bool)
    pos_emb = sinusoidal_rel_pos_table_torch(t_len, d)
    h = x
    with torch.no_grad():
        for layer in layers:
            h = layer(h, mask, pos_emb)
    want = h

    tt_layers = [TtConformerEncoderLayer(device, layer) for layer in layers]
    x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pos_dev = ttnn.from_torch(pos_emb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    bias_dev = ttnn.from_torch(torch.zeros(b, 1, 1, t_len), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    h_dev = x_dev
    for tt_layer in tt_layers:
        h_dev = tt_layer(h_dev, pos_dev, bias_dev)
    got = ttnn.to_torch(h_dev).float()

    assert got.shape == want.shape
    passed, pcc = comp_pcc(want, got, GATE_BF16)
    print(f"\n  device 2-layer ConformerEncoderLayer stack PCC {pcc}")
    assert passed, pcc
