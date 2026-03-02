# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
PCC test: PyTorch ConformerEncoder reference vs TTNN ConformerEncoder with same weights.

Tests the conformer encoder stack (layers with FF1, Conv, MHA, FF2)
to validate TTNN model architecture correctness.
"""

import pytest
import torch
import ttnn
from loguru import logger
from types import SimpleNamespace

from models.experimental.parakeet.reference.pytorch_conf_layer import (
    ConformerEncoder as ConformerEncoderTorch,
)
from models.experimental.parakeet.tt.ttnn_conf_layer import (
    TtConformerEncoder,
)
from tests.ttnn.utils_for_testing import check_with_pcc


# ============================================================
# Global config
# ============================================================

CONFORMER_L1_SMALL_SIZE = 32768

d_model = 1024
d_ff = 4096
n_heads = 8
conv_kernel_size = 31
time_steps = 128


# ============================================================
# Weight Copy Utility
# ============================================================


def _copy_weights_to_ttnn_params(torch_model, device):
    params = SimpleNamespace()
    params.layers = []

    for torch_layer in torch_model.layers:
        layer_params = SimpleNamespace()

        # ---------------- FeedForward 1 ----------------
        layer_params.ffn1 = SimpleNamespace()
        layer_params.ffn1.layer_norm_weight = ttnn.from_torch(
            torch_layer.norm_feed_forward1.weight.data.clone(),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        layer_params.ffn1.layer_norm_bias = ttnn.from_torch(
            torch_layer.norm_feed_forward1.bias.data.clone(),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        layer_params.ffn1.linear1_weight = ttnn.from_torch(
            torch_layer.feed_forward1.linear1.weight.data.clone().T,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        # Guard against None bias
        if torch_layer.feed_forward1.linear1.bias is not None:
            layer_params.ffn1.linear1_bias = ttnn.from_torch(
                torch_layer.feed_forward1.linear1.bias.data.clone(),
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            layer_params.ffn1.linear1_bias = None

        layer_params.ffn1.linear2_weight = ttnn.from_torch(
            torch_layer.feed_forward1.linear2.weight.data.clone().T,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        if torch_layer.feed_forward1.linear2.bias is not None:
            layer_params.ffn1.linear2_bias = ttnn.from_torch(
                torch_layer.feed_forward1.linear2.bias.data.clone(),
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            layer_params.ffn1.linear2_bias = None

        # ---------------- FeedForward 2 ----------------
        layer_params.ffn2 = SimpleNamespace()
        layer_params.ffn2.layer_norm_weight = ttnn.from_torch(
            torch_layer.norm_feed_forward2.weight.data.clone(),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        layer_params.ffn2.layer_norm_bias = ttnn.from_torch(
            torch_layer.norm_feed_forward2.bias.data.clone(),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        layer_params.ffn2.linear1_weight = ttnn.from_torch(
            torch_layer.feed_forward2.linear1.weight.data.clone().T,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        if torch_layer.feed_forward2.linear1.bias is not None:
            layer_params.ffn2.linear1_bias = ttnn.from_torch(
                torch_layer.feed_forward2.linear1.bias.data.clone(),
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            layer_params.ffn2.linear1_bias = None

        layer_params.ffn2.linear2_weight = ttnn.from_torch(
            torch_layer.feed_forward2.linear2.weight.data.clone().T,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        if torch_layer.feed_forward2.linear2.bias is not None:
            layer_params.ffn2.linear2_bias = ttnn.from_torch(
                torch_layer.feed_forward2.linear2.bias.data.clone(),
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )
        else:
            layer_params.ffn2.linear2_bias = None

        # ---------------- Attention ----------------
        layer_params.self_attn = SimpleNamespace()
        layer_params.self_attn.q = ttnn.from_torch(
            torch_layer.self_attn.linear_q.weight.data.clone().T,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        layer_params.self_attn.k = ttnn.from_torch(
            torch_layer.self_attn.linear_k.weight.data.clone().T,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        layer_params.self_attn.v = ttnn.from_torch(
            torch_layer.self_attn.linear_v.weight.data.clone().T,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        layer_params.self_attn.out = SimpleNamespace()
        layer_params.self_attn.out.weight = ttnn.from_torch(
            torch_layer.self_attn.linear_out.weight.data.clone().T,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        # Optional: pos if used by your attention implementation
        layer_params.self_attn.pos = ttnn.from_torch(
            torch_layer.self_attn.linear_pos.weight.data.clone().T,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        # Apply similar None checks for attention linear weights/biases if needed
        # ...

        # ---------------- Convolution ----------------
        layer_params.conv = SimpleNamespace()
        layer_params.conv.layer_norm_weight = ttnn.from_torch(
            torch_layer.norm_conv.weight.data.clone(),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        layer_params.conv.layer_norm_bias = ttnn.from_torch(
            torch_layer.norm_conv.bias.data.clone(),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        # ... rest of conv weights

        # ---------------- Final LayerNorm ----------------
        layer_params.final_layer_norm_weight = ttnn.from_torch(
            torch_layer.norm_out.weight.data.clone(),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        layer_params.final_layer_norm_bias = ttnn.from_torch(
            torch_layer.norm_out.bias.data.clone(),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )

        params.layers.append(layer_params)

    return params


# ============================================================
# PCC TEST
# ============================================================


@pytest.mark.parametrize("device_params", [{"l1_small_size": CONFORMER_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize("batch_size", [1, 2], ids=["batch1", "batch2"])
@pytest.mark.parametrize("n_layers", [1, 2], ids=["layers1", "layers2"])
def test_ttnn_conformer_encoder_pcc(device, batch_size, n_layers):
    torch.manual_seed(0)

    torch_model = ConformerEncoderTorch(
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        conv_kernel_size=conv_kernel_size,
        n_layers=n_layers,
    ).to(torch.bfloat16)

    torch_model.eval()

    ttnn_model = TtConformerEncoder(
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        conv_kernel_size=conv_kernel_size,
        n_layers=n_layers,
        device=device,
        dtype=ttnn.bfloat16,
    )

    params = _copy_weights_to_ttnn_params(torch_model, device)

    pt_input = torch.randn(batch_size, time_steps, d_model, dtype=torch.bfloat16)

    lengths = torch.tensor(
        [time_steps] + [time_steps // 2] * (batch_size - 1),
        dtype=torch.long,
    )[:batch_size]

    with torch.no_grad():
        ref_out = torch_model(pt_input, lengths)

    tt_input = ttnn.from_torch(
        pt_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    if len(tt_input.shape) == 4 and tt_input.shape[1] == 1:
        tt_input = ttnn.reshape(tt_input, (batch_size, time_steps, d_model))

    tt_out = ttnn_model(tt_input, lengths, params)

    tt_out_torch = ttnn.to_torch(tt_out)

    pcc_threshold = 0.90 if n_layers > 1 else 0.95
    passed, msg = check_with_pcc(ref_out.float(), tt_out_torch.float(), pcc=pcc_threshold)

    logger.info(f"PCC (batch={batch_size}, layers={n_layers}): {passed}, {msg}")

    assert passed, f"PCC failed: {msg}"
    assert ref_out.shape == tt_out_torch.shape
