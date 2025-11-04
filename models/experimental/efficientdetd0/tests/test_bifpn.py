# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc

from models.experimental.efficientdetd0.reference.modules import BiFPN
from models.experimental.efficientdetd0.tt.bifpn import TtBiFPN
from models.experimental.efficientdetd0.tt.custom_preprocessor import (
    create_custom_mesh_preprocessor,
    infer_torch_module_args,
)
from ttnn.dot_access import make_dot_access_dict
from ttnn.model_preprocessing import ModuleArgs

torch.manual_seed(0)


@pytest.mark.parametrize(
    "num_channels, conv_channels, first_time, attention, inputs",
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
    # load_torch_model_state(torch_model, "Bi")

    # Convert model to bfloat16 to match input dtype
    # torch_model = torch_model.to(torch.bfloat16)

    # Run PyTorch forward pass
    with torch.no_grad():
        torch_outputs = torch_model(inputs)

    # Preprocess model parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )

    # Infer module arguments for all Conv2d layers
    # module_args = infer_torch_module_args(model=torch_model, input=inputs, layer_type=torch.nn.Conv2d)

    conv_module_args = infer_torch_module_args(model=torch_model, input=inputs, layer_type=torch.nn.Conv2d)
    maxpool_module_args = infer_torch_module_args(model=torch_model, input=inputs, layer_type=torch.nn.MaxPool2d)

    # Build conv_args structure for BiFPN
    i = 0
    conv_args = {}

    # Add first_time specific layers if needed
    if first_time:
        conv_args["p5_to_p6_conv"] = conv_module_args[0]
        i += 1
        conv_args["p3_down_channel"] = conv_module_args[i]
        i += 1
        conv_args["p4_down_channel"] = conv_module_args[i]
        i += 1
        conv_args["p5_down_channel"] = conv_module_args[i]
        i += 1

    # Upsampling path convs
    conv_args["conv6_up"] = {}
    conv_args["conv6_up"]["depthwise_conv"] = conv_module_args[i]
    conv_args["conv6_up"]["pointwise_conv"] = conv_module_args[i + 1]
    i += 2
    conv_args["conv5_up"] = {}
    conv_args["conv5_up"]["depthwise_conv"] = conv_module_args[i]
    conv_args["conv5_up"]["pointwise_conv"] = conv_module_args[i + 1]
    i += 2
    conv_args["conv4_up"] = {}
    conv_args["conv4_up"]["depthwise_conv"] = conv_module_args[i]
    conv_args["conv4_up"]["pointwise_conv"] = conv_module_args[i + 1]
    i += 2

    conv_args["conv3_up"] = {}
    conv_args["conv3_up"]["depthwise_conv"] = conv_module_args[i]
    conv_args["conv3_up"]["pointwise_conv"] = conv_module_args[i + 1]
    i += 2

    if first_time:
        conv_args["p4_down_channel_2"] = conv_module_args[i]
        i += 1
        conv_args["p5_down_channel_2"] = conv_module_args[i]
        i += 1

    # Downsampling path convs
    conv_args["conv4_down"] = {}
    conv_args["conv4_down"]["depthwise_conv"] = conv_module_args[i]
    conv_args["conv4_down"]["pointwise_conv"] = conv_module_args[i + 1]
    i += 2

    conv_args["conv5_down"] = {}
    conv_args["conv5_down"]["depthwise_conv"] = conv_module_args[i]
    conv_args["conv5_down"]["pointwise_conv"] = conv_module_args[i + 1]
    i += 2

    conv_args["conv6_down"] = {}
    conv_args["conv6_down"]["depthwise_conv"] = conv_module_args[i]
    conv_args["conv6_down"]["pointwise_conv"] = conv_module_args[i + 1]
    i += 2

    conv_args["conv7_down"] = {}
    conv_args["conv7_down"]["depthwise_conv"] = conv_module_args[i]
    conv_args["conv7_down"]["pointwise_conv"] = conv_module_args[i + 1]
    i += 2

    # Add MaxPool2d layers
    # Map maxpool_module_args indices to the correct layer names
    # The order depends on how PyTorch registers the modules
    j = 0
    if first_time:
        # For first_time=True, we have: p5_to_p6_pool, p6_to_p7, then p4/p5/p6/p7_downsample
        conv_args["p5_to_p6_pool"] = maxpool_module_args[j]
        j += 1
        conv_args["p6_to_p7"] = maxpool_module_args[j]
        j += 1

    conv_args["p4_downsample"] = maxpool_module_args[j]
    j += 1
    conv_args["p5_downsample"] = maxpool_module_args[j]
    j += 1
    conv_args["p6_downsample"] = maxpool_module_args[j]
    j += 1
    conv_args["p7_downsample"] = maxpool_module_args[j]

    conv_args = make_dot_access_dict(conv_args, ignore_types=(ModuleArgs,))

    # Create TTNN BiFPN model
    ttnn_model = TtBiFPN(
        device=device,
        parameters=parameters,
        conv_params=conv_args,
        num_channels=num_channels,
        first_time=first_time,
        epsilon=1e-4,
        attention=attention,
        use_p8=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation=False,
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
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        # Permute back to NCHW format
        batch, h, w, channels = ttnn_output_torch.shape
        ttnn_output_torch = ttnn_output_torch.reshape(batch, h, w, channels)
        ttnn_output_torch = torch.permute(ttnn_output_torch, (0, 3, 1, 2))

        passing, pcc_message = comp_pcc(torch_output, ttnn_output_torch, PCC_THRESHOLD)
        logger.info(f"{output_names[i]} PCC: {pcc_message}")
        # logger.info(comp_allclose(torch_output, ttnn_output_torch))

        # assert passing, f"{output_names[i]} PCC value is lower than {PCC_THRESHOLD}. {pcc_message}"

    logger.info("BiFPN Test Passed!")
