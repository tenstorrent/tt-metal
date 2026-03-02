# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
PCC test: Full TTNN Conformer layer vs NeMo ConformerLayer (single layer).

Loads NeMo Parakeet, runs one Conformer layer (reference = NeMo layer 0,
TT = TtConformerLayer with weights copied from NeMo layer 0), compares outputs.
"""

import pytest
import torch
import ttnn
from types import SimpleNamespace

import nemo.collections.asr as nemo_asr

from tests.ttnn.utils_for_testing import check_with_pcc
from models.experimental.parakeet.tt.tt_conformer_layer import TtConformerLayer
from models.experimental.parakeet.tt.ttnn_conf_layer import TtConformerFeedForward, TtConformerConvolution
from models.experimental.parakeet.tt.tt_relposition import RelPositionMultiHeadAttentionTTNN


L1_SMALL_SIZE = 32768


def _copy_one_layer_params(torch_layer, device):
    """Build one layer's TTNN parameters from NeMo ConformerLayer (layer 0)."""
    p = SimpleNamespace()

    # FFN1
    p.ffn1 = SimpleNamespace()
    p.ffn1.layer_norm_weight = ttnn.from_torch(
        torch_layer.norm_feed_forward1.weight.data.clone(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    p.ffn1.layer_norm_bias = ttnn.from_torch(
        torch_layer.norm_feed_forward1.bias.data.clone(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    p.ffn1.linear1 = SimpleNamespace(
        weight=ttnn.from_torch(
            torch_layer.feed_forward1.linear1.weight.data.clone().transpose(-1, -2),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    )
    p.ffn1.linear2 = SimpleNamespace(
        weight=ttnn.from_torch(
            torch_layer.feed_forward1.linear2.weight.data.clone().transpose(-1, -2),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    )

    # Self-attn norm only
    p.self_attn = SimpleNamespace()
    p.self_attn.layer_norm_weight = ttnn.from_torch(
        torch_layer.norm_self_att.weight.data.clone(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    p.self_attn.layer_norm_bias = ttnn.from_torch(
        torch_layer.norm_self_att.bias.data.clone(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Conv
    p.conv = SimpleNamespace()
    p.conv.layer_norm_weight = ttnn.from_torch(
        torch_layer.norm_conv.weight.data.clone(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    p.conv.layer_norm_bias = ttnn.from_torch(
        torch_layer.norm_conv.bias.data.clone(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    # Conv weights: match test_conformer_conv (no TILE_LAYOUT; conv1d expects default layout)
    p.conv.pointwise1 = SimpleNamespace(
        weight=ttnn.from_torch(
            torch_layer.conv.pointwise_conv1.weight.data.clone(),
            dtype=ttnn.bfloat16,
            device=device,
        )
    )
    p.conv.pointwise1.bias = None
    p.conv.depthwise = SimpleNamespace(
        weight=ttnn.from_torch(
            torch_layer.conv.depthwise_conv.weight.data.clone(),
            dtype=ttnn.bfloat16,
            device=device,
        )
    )
    p.conv.depthwise.bias = None
    p.conv.pointwise2 = SimpleNamespace(
        weight=ttnn.from_torch(
            torch_layer.conv.pointwise_conv2.weight.data.clone(),
            dtype=ttnn.bfloat16,
            device=device,
        )
    )
    p.conv.pointwise2.bias = None
    p.conv.bn = SimpleNamespace()
    p.conv.bn.running_mean = ttnn.from_torch(
        torch_layer.conv.batch_norm.running_mean.data.clone(),
        dtype=ttnn.bfloat16,
        device=device,
    )
    p.conv.bn.running_var = ttnn.from_torch(
        torch_layer.conv.batch_norm.running_var.data.clone(),
        dtype=ttnn.bfloat16,
        device=device,
    )
    p.conv.bn.weight = ttnn.from_torch(
        torch_layer.conv.batch_norm.weight.data.clone(),
        dtype=ttnn.bfloat16,
        device=device,
    )
    p.conv.bn.bias = ttnn.from_torch(
        torch_layer.conv.batch_norm.bias.data.clone(),
        dtype=ttnn.bfloat16,
        device=device,
    )

    # FFN2
    p.ffn2 = SimpleNamespace()
    p.ffn2.layer_norm_weight = ttnn.from_torch(
        torch_layer.norm_feed_forward2.weight.data.clone(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    p.ffn2.layer_norm_bias = ttnn.from_torch(
        torch_layer.norm_feed_forward2.bias.data.clone(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    p.ffn2.linear1 = SimpleNamespace(
        weight=ttnn.from_torch(
            torch_layer.feed_forward2.linear1.weight.data.clone().transpose(-1, -2),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    )
    p.ffn2.linear2 = SimpleNamespace(
        weight=ttnn.from_torch(
            torch_layer.feed_forward2.linear2.weight.data.clone().transpose(-1, -2),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    )

    # Final norm
    p.final_layer_norm_weight = ttnn.from_torch(
        torch_layer.norm_out.weight.data.clone(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    p.final_layer_norm_bias = ttnn.from_torch(
        torch_layer.norm_out.bias.data.clone(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return p


@pytest.mark.parametrize("device_params", [{"l1_small_size": L1_SMALL_SIZE}], indirect=True)
def test_full_tt_conformer_encoder_vs_nemo(device):
    """Full 24-layer TTNN Conformer encoder vs NeMo encoder (PCC)."""

    torch.manual_seed(0)

    # --------------------------------------------------
    # Load NeMo Parakeet
    # --------------------------------------------------
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        "nvidia/parakeet-tdt-0.6b-v2",
        map_location="cpu",
    )

    encoder = asr_model.encoder
    encoder.eval()

    num_layers = len(encoder.layers)

    # Extract dimensions from layer 0
    nemo_layer0 = encoder.layers[0].to(torch.bfloat16)
    d_model = nemo_layer0.self_attn.linear_q.weight.shape[1]
    n_heads = nemo_layer0.self_attn.h
    conv_kernel_size = nemo_layer0.conv.depthwise_conv.weight.shape[2]

    class MHAConfig:
        num_heads = n_heads
        dim_head = d_model // n_heads
        context_size = 128

    mha_config = MHAConfig()

    # --------------------------------------------------
    # Build TT layers + parameters
    # --------------------------------------------------
    tt_layers = []
    params_list = []

    for i in range(num_layers):
        nemo_layer = encoder.layers[i].to(torch.bfloat16)
        nemo_layer.eval()

        # Create fresh modules per layer
        ffn = TtConformerFeedForward(device, dtype=ttnn.bfloat16)
        conv = TtConformerConvolution(d_model, conv_kernel_size, device, ttnn.bfloat16)
        mha = RelPositionMultiHeadAttentionTTNN(device, mha_config)

        torch_mha = nemo_layer.self_attn

        mha.prepare_weights(
            torch_mha.linear_q.weight.data.clone(),
            torch_mha.linear_k.weight.data.clone(),
            torch_mha.linear_v.weight.data.clone(),
            torch_mha.linear_q.bias.data.clone()
            if torch_mha.linear_q.bias is not None
            else torch.zeros(d_model, dtype=torch.bfloat16),
            torch_mha.linear_k.bias.data.clone()
            if torch_mha.linear_k.bias is not None
            else torch.zeros(d_model, dtype=torch.bfloat16),
            torch_mha.linear_v.bias.data.clone()
            if torch_mha.linear_v.bias is not None
            else torch.zeros(d_model, dtype=torch.bfloat16),
            torch_mha.linear_out.weight.data.clone(),
            torch_mha.linear_out.bias.data.clone()
            if torch_mha.linear_out.bias is not None
            else torch.zeros(d_model, dtype=torch.bfloat16),
            torch_mha.linear_pos.weight.data.clone(),
            torch_mha.pos_bias_u.data.clone(),
            torch_mha.pos_bias_v.data.clone(),
        )

        tt_layer = TtConformerLayer(device, d_model, ffn, conv, mha)

        tt_layers.append(tt_layer)
        params_list.append(_copy_one_layer_params(nemo_layer, device))

    # --------------------------------------------------
    # Input
    # --------------------------------------------------
    batch_size = 1
    T = 32
    x = torch.randn(batch_size, T, d_model, dtype=torch.bfloat16)

    # --------------------------------------------------
    # Reference forward (NeMo full encoder)
    # --------------------------------------------------
    with torch.no_grad():
        _, pos_emb = encoder.pos_enc(x)
        pos_emb = pos_emb.to(torch.bfloat16)

        ref_out = x
        for layer in encoder.layers:
            ref_out = layer(ref_out, att_mask=None, pos_emb=pos_emb, pad_mask=None)

    # --------------------------------------------------
    # Prepare TT tensors
    # --------------------------------------------------
    pos_emb_tt = pos_emb
    if pos_emb.dim() == 3 and pos_emb.size(0) != 1:
        pos_emb_tt = pos_emb.permute(1, 0, 2)

    tt_x = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_pos_emb = ttnn.from_torch(
        pos_emb_tt,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # --------------------------------------------------
    # TT forward through all layers
    # --------------------------------------------------
    tt_out = tt_x

    for layer, params in zip(tt_layers, params_list):
        tt_out = layer.forward(
            tt_out,
            tt_pos_emb,
            att_mask=None,
            pad_mask=None,
            parameters=params,
        )

    tt_out_torch = ttnn.to_torch(tt_out).float()
    ref_out = ref_out.float()

    if tt_out_torch.shape != ref_out.shape:
        tt_out_torch = tt_out_torch[:, : ref_out.size(1), :]

    # --------------------------------------------------
    # PCC check
    # --------------------------------------------------
    passed, msg = check_with_pcc(ref_out, tt_out_torch, pcc=0.95)
    print(f"Full Encoder (24-layer) PCC: {msg}")

    assert passed, f"PCC failed: {msg}"
    assert ref_out.shape == tt_out_torch.shape, f"Shape mismatch: ref {ref_out.shape} vs tt {tt_out_torch.shape}"
