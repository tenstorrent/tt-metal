# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc

from models.experimental.BEVFormerV2.reference.decoder import DetectionTransformerDecoder
from models.experimental.BEVFormerV2.tt.ttnn_decoder import TtDecoder
from models.experimental.BEVFormerV2.common import download_bevformerv2_weights
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from models.experimental.BEVFormerV2.tests.pcc.custom_preprocessors import custom_preprocessor_decoder


def create_bevformerv2_decoder_parameters(model, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor_decoder,
        device=device,
    )
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_decoder_pcc(
    device,
    reset_seeds,
    model_location_generator,
):
    torch.manual_seed(42)

    try:
        pytorch_decoder = DetectionTransformerDecoder(
            num_layers=1,
            embed_dim=256,
            num_heads=8,
            return_intermediate=True,
        )

        weights_path = download_bevformerv2_weights()
        checkpoint = torch.load(weights_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        decoder_state = {}
        for key, value in state_dict.items():
            if "decoder" in key:
                new_key = key.replace("pts_bbox_head.transformer.decoder.", "")
                decoder_state[new_key] = value

        pytorch_decoder.load_state_dict(decoder_state, strict=False)
        pytorch_decoder.eval()
    except Exception as e:
        pytest.skip(f"Failed to create models: {e}")

    query = torch.rand(300, 1, 256, dtype=torch.float32)
    value = torch.rand(10000, 1, 256, dtype=torch.float32)
    query_pos = torch.rand(300, 1, 256, dtype=torch.float32)
    reference_points = torch.rand(1, 300, 3, dtype=torch.float32)
    spatial_shapes = torch.tensor([[100, 100]], dtype=torch.int32)

    with torch.no_grad():
        torch_output1, torch_output2 = pytorch_decoder(
            query=query,
            key=None,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            reg_branches=None,
        )

    decoder_params = create_bevformerv2_decoder_parameters(pytorch_decoder, device=device)

    ttnn_decoder = TtDecoder(
        num_layers=1,
        embed_dim=256,
        num_heads=8,
        params=decoder_params,
        params_branches=decoder_params,
        device=device,
    )

    query_ttnn = ttnn.from_torch(
        query.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    value_ttnn = ttnn.from_torch(
        value.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    query_pos_ttnn = ttnn.from_torch(
        query_pos.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    reference_points_ttnn = ttnn.from_torch(
        reference_points.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    spatial_shapes_ttnn = ttnn.from_torch(
        spatial_shapes.to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output1, ttnn_output2 = ttnn_decoder(
        query=query_ttnn,
        key=None,
        value=value_ttnn,
        query_pos=query_pos_ttnn,
        reference_points=reference_points_ttnn,
        spatial_shapes=spatial_shapes_ttnn,
        reg_branches=None,
    )

    ttnn_output1_torch = ttnn.to_torch(ttnn_output1)
    ttnn_output2_torch = ttnn.to_torch(ttnn_output2)

    if ttnn_output1_torch.shape != torch_output1.shape:
        raise ValueError(f"Output1 shape mismatch: ttnn={ttnn_output1_torch.shape}, ref={torch_output1.shape}")

    if ttnn_output2_torch.shape != torch_output2.shape:
        raise ValueError(f"Output2 shape mismatch: ttnn={ttnn_output2_torch.shape}, ref={torch_output2.shape}")

    pcc_result1 = comp_pcc(torch_output1, ttnn_output1_torch)
    pcc_value1 = pcc_result1[1] if isinstance(pcc_result1, tuple) else pcc_result1

    pcc_result2 = comp_pcc(torch_output2, ttnn_output2_torch)
    pcc_value2 = pcc_result2[1] if isinstance(pcc_result2, tuple) else pcc_result2

    print(f"DetectionTransformerDecoder Output1 PCC: {pcc_value1:.6f}")
    print(f"DetectionTransformerDecoder Output2 PCC: {pcc_value2:.6f}")

    assert pcc_value1 > 0.97, f"Decoder Output1 PCC {pcc_value1:.6f} is below threshold 0.97"
    assert pcc_value2 > 0.97, f"Decoder Output2 PCC {pcc_value2:.6f} is below threshold 0.97"
