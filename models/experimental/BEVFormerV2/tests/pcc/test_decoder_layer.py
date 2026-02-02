# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc

from models.experimental.BEVFormerV2.reference.decoder import DetrTransformerDecoderLayer
from models.experimental.BEVFormerV2.tt.ttnn_decoder_layer import TtDecoderLayer
from models.experimental.BEVFormerV2.tt.model_preprocessing import prepare_decoder_layer_parameters_for_test


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_decoder_layer_pcc(
    device,
    reset_seeds,
    model_location_generator,
):
    torch.manual_seed(42)

    embed_dims = 256
    num_heads = 8
    feedforward_channels = 512
    num_query = 900
    num_value = 40000
    bs = 1

    try:
        pytorch_decoder_layer = DetrTransformerDecoderLayer(
            attn_cfgs=[
                {"type": "MultiheadAttention", "embed_dims": embed_dims, "num_heads": num_heads},
                {"type": "CustomMSDeformableAttention", "embed_dims": embed_dims, "num_levels": 1},
            ],
            ffn_cfgs=dict(
                type="FFN",
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=2,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            norm_cfg=dict(type="LN"),
            batch_first=False,
        )
        pytorch_decoder_layer.eval()
    except Exception as e:
        pytest.skip(f"Failed to create PyTorch model: {e}")

    query = torch.randn((num_query, bs, embed_dims), dtype=torch.float32)
    value = torch.randn((num_value, bs, embed_dims), dtype=torch.float32)
    query_pos = torch.randn((num_query, bs, embed_dims), dtype=torch.float32)
    reference_points = torch.rand((bs, num_query, 1, 2), dtype=torch.float32)
    reference_points = reference_points.clamp(0.0, 1.0)
    spatial_shapes = torch.tensor([[200, 200]], dtype=torch.int32)

    with torch.no_grad():
        torch_output = pytorch_decoder_layer(
            query=query,
            key=None,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
        )

    query_ttnn = ttnn.from_torch(
        query.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    query_ttnn = ttnn.to_layout(query_ttnn, ttnn.TILE_LAYOUT)

    value_ttnn = ttnn.from_torch(
        value.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    value_ttnn = ttnn.to_layout(value_ttnn, ttnn.TILE_LAYOUT)

    query_pos_ttnn = ttnn.from_torch(
        query_pos.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    query_pos_ttnn = ttnn.to_layout(query_pos_ttnn, ttnn.TILE_LAYOUT)

    reference_points_ttnn = ttnn.from_torch(
        reference_points.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    reference_points_ttnn = ttnn.to_layout(reference_points_ttnn, ttnn.TILE_LAYOUT)

    spatial_shapes_ttnn = ttnn.from_torch(
        spatial_shapes.to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    model_config = {
        "WEIGHTS_DTYPE": ttnn.bfloat16,
        "ACTIVATIONS_DTYPE": ttnn.bfloat16,
        "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
    }

    decoder_layer_params = prepare_decoder_layer_parameters_for_test(pytorch_decoder_layer, device)
    ttnn_decoder_layer = TtDecoderLayer(
        params=decoder_layer_params,
        device=device,
        attn_cfgs=[
            {"type": "MultiheadAttention", "embed_dims": embed_dims, "num_heads": num_heads},
            {"type": "CustomMSDeformableAttention", "embed_dims": embed_dims, "num_levels": 1},
        ],
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        norm_cfg=dict(type="LN"),
        batch_first=False,
        model_config=model_config,
    )

    ttnn_output = ttnn_decoder_layer(
        query=query_ttnn,
        key=None,
        value=value_ttnn,
        query_pos=query_pos_ttnn,
        reference_points=reference_points_ttnn,
        spatial_shapes=spatial_shapes_ttnn,
    )

    ttnn_output_torch = ttnn.to_torch(ttnn_output)

    if ttnn_output_torch.shape != torch_output.shape:
        raise ValueError(f"Shape mismatch: ttnn={ttnn_output_torch.shape}, ref={torch_output.shape}")

    pcc_result = comp_pcc(torch_output, ttnn_output_torch)
    pcc_value = pcc_result[1] if isinstance(pcc_result, tuple) else pcc_result

    assert pcc_value > 0.96, f"DetrTransformerDecoderLayer PCC {pcc_value:.6f} is below threshold 0.96"
