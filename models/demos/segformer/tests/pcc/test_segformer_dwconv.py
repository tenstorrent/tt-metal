# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.segformer.common import load_torch_model
from models.demos.segformer.reference.segformer_dwconv import SegformerDWConv
from models.demos.segformer.tt.ttnn_segformer_dwconv import TtSegformerDWConv
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SegformerDWConv):
            parameters["dwconv"] = {}
            parameters["dwconv"]["weight"] = ttnn.from_torch(model.dwconv.weight, dtype=ttnn.bfloat16)
            parameters["dwconv"]["bias"] = ttnn.from_torch(
                torch.reshape(model.dwconv.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

        return parameters

    return custom_preprocessor


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch_size, seq_len, dim, height, width, block_i, dwconv_i",
    [
        (1, 16384, 128, 128, 128, 0, 0),
        (1, 16384, 128, 128, 128, 0, 1),
        (1, 4096, 256, 64, 64, 1, 0),
        (1, 4096, 256, 64, 64, 1, 1),
        (1, 1024, 640, 32, 32, 2, 0),
        (1, 1024, 640, 32, 32, 2, 1),
        (1, 256, 1024, 16, 16, 3, 0),
        (1, 256, 1024, 16, 16, 3, 1),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_segformer_dw_conv(
    device, batch_size, seq_len, dim, height, width, block_i, dwconv_i, model_location_generator
):
    torch_input_tensor = torch.randn(batch_size, seq_len, dim)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    reference_model = SegformerDWConv(dim=dim)
    target_prefix = f"encoder.block.{block_i}.{dwconv_i}.mlp.dwconv."
    reference_model = load_torch_model(
        reference_model, target_prefix, module="semantic_sub", model_location_generator=model_location_generator
    )

    torch_output = reference_model(torch_input_tensor, height, width)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )

    ttnn_model = TtSegformerDWConv(parameters, dim)

    ttnn_output = ttnn_model(
        device,
        ttnn_input_tensor,
        height,
        width,
    )
    ttnn_output = ttnn.from_device(ttnn_output[0])
    ttnn_output = ttnn.to_torch(ttnn_output)[0]

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
