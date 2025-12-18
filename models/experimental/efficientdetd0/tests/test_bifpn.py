# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters

from models.common.utility_functions import comp_pcc
from models.experimental.efficientdetd0.tt.custom_preprocessor import (
    create_custom_mesh_preprocessor,
    infer_torch_module_args,
)
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.experimental.efficientdetd0.common import load_torch_model_state
from models.experimental.efficientdetd0.tt.bifpn import TtBiFPN
from models.experimental.efficientdetd0.reference.modules import BiFPN


torch.manual_seed(0)


@pytest.mark.parametrize(
    "num_channels, conv_channels, first_time, attention, inputs, weight_key",
    [
        (
            64,  # num_channels
            [40, 112, 320],  # conv_channels
            True,  # first_time
            True,  # attention
            (
                torch.randn(1, 40, 64, 64),
                torch.randn(1, 112, 32, 32),
                torch.randn(1, 320, 16, 16),
            ),
            "bifpn.0",
        ),
        (
            64,
            [40, 112, 320],
            False,
            True,
            (
                torch.randn([1, 64, 64, 64]),
                torch.randn([1, 64, 32, 32]),
                torch.randn([1, 64, 16, 16]),
                torch.randn([1, 64, 8, 8]),
                torch.randn([1, 64, 4, 4]),
            ),
            "bifpn.1",
        ),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_bifpn(
    num_channels,
    conv_channels,
    first_time,
    attention,
    inputs,
    weight_key,
    device,
):
    PCC_THRESHOLD = 0.99

    # Create PyTorch BiFPN model
    torch_model = BiFPN(
        num_channels=num_channels,
        conv_channels=conv_channels,
        first_time=first_time,
        epsilon=1e-4,
        onnx_export=False,
        attention=attention,
        use_p8=False,
    ).eval()
    load_torch_model_state(torch_model, weight_key)

    # Run PyTorch forward pass
    with torch.no_grad():
        torch_outputs = torch_model(inputs)

    # Preprocess model parameters
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    # Infer module arguments for all Conv2d layers
    module_args = infer_torch_module_args(model=torch_model, input=inputs)

    # Create TTNN BiFPN model
    ttnn_model = TtBiFPN(
        device=device,
        parameters=parameters,
        module_args=module_args,
        num_channels=num_channels,
        first_time=first_time,
        epsilon=1e-4,
        attention=attention,
        use_p8=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=True,
    )

    # Convert inputs to TTNN format
    ttnn_inputs = [
        ttnn.from_torch(
            x.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for x in inputs
    ]

    # Run TTNN forward pass
    ttnn_outputs = ttnn_model(tuple(ttnn_inputs))

    # Compare each output pyramid level
    output_names = ["P3", "P4", "P5", "P6", "P7"]
    for i, (torch_output, ttnn_output) in enumerate(zip(torch_outputs, ttnn_outputs)):
        B, C, H, W = torch_output.shape
        ttnn_output_torch = ttnn.to_torch(ttnn_output)  # 1, 1, NHW, C
        ttnn_output_torch = torch.reshape(ttnn_output_torch, (B, H, W, C))

        # Permute back to NCHW format
        ttnn_output_torch = torch.permute(ttnn_output_torch, (0, 3, 1, 2))  # N, C, H, W

        passing, pcc_message = comp_pcc(torch_output, ttnn_output_torch, PCC_THRESHOLD)
        logger.info(f"{output_names[i]} PCC: {pcc_message}")

        assert passing, f"{output_names[i]} PCC value is lower than {PCC_THRESHOLD}. {pcc_message}"

    logger.info("BiFPN Test Passed!")
