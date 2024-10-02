# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from tests.ttnn.utils_for_testing import assert_with_pcc
from transformers import SegformerModel
from models.experimental.functional_segformer.reference.segformer_dwconv import SegformerDWConv
import pytest
from models.experimental.functional_segformer.tt.ttnn_segformer_dwconv import TtSegformerDWConv
from models.utility_functions import skip_for_grayskull


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
def test_segformer_dw_conv(device, batch_size, seq_len, dim, height, width, block_i, dwconv_i, reset_seeds):
    torch_input_tensor = torch.randn(batch_size, seq_len, dim)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    torch_model = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    torch_model = torch_model.encoder.block[block_i][dwconv_i].mlp.dwconv

    reference_model = SegformerDWConv(dim=dim)
    sd = torch_model.state_dict()
    reference_model.load_state_dict(sd)
    reference_model.eval()

    torch_output = reference_model(torch_input_tensor, height, width)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=None
    )

    ttnn_model = TtSegformerDWConv(parameters, dim)

    ttnn_output = ttnn_model(
        ttnn_input_tensor,
        height,
        width,
        device,
    )
    ttnn_output = ttnn.from_device(ttnn_output)
    ttnn_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_output, ttnn_output, pcc=0.99)
