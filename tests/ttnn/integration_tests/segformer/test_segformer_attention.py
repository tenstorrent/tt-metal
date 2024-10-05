# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)
from transformers import SegformerModel
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull
from models.experimental.functional_segformer.tt.ttnn_segformer_attention import (
    TtSegformerAttention,
)
from models.experimental.functional_segformer.reference.segformer_attention import SegformerAttention
from tests.ttnn.integration_tests.segformer.test_segformer_efficient_selfattention import (
    create_custom_preprocessor as create_customer_preprocessor_selfattention,
)
from tests.ttnn.integration_tests.segformer.test_segformer_selfoutput import (
    create_custom_preprocessor as create_customer_preprocessor_selfoutput,
)


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SegformerAttention):
            parameters["self"] = {}
            self_attention_prepocessor = create_customer_preprocessor_selfattention(device)
            parameters["self"] = self_attention_prepocessor(model.self, None, None)

            parameters["output"] = {}
            self_output_prepocessor = create_customer_preprocessor_selfoutput(device)
            parameters["output"] = self_output_prepocessor(model.output, None, None)

        return parameters

    return custom_preprocessor


@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "hidden_size, num_attention_heads, sequence_reduction_ratio, batch_size, seq_len, height, width, block_i, attention_i",
    [
        (32, 1, 8, 1, 16384, 128, 128, 0, 0),
        (32, 1, 8, 1, 16384, 128, 128, 0, 1),
        (64, 2, 4, 1, 4096, 64, 64, 1, 0),
        (64, 2, 4, 1, 4096, 64, 64, 1, 1),
        (160, 5, 2, 1, 1024, 32, 32, 2, 0),
        (160, 5, 2, 1, 1024, 32, 32, 2, 1),
        (256, 8, 1, 1, 256, 16, 16, 3, 0),
        (256, 8, 1, 1, 256, 16, 16, 3, 1),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_attention(
    device,
    hidden_size,
    num_attention_heads,
    sequence_reduction_ratio,
    batch_size,
    seq_len,
    height,
    width,
    block_i,
    attention_i,
    reset_seeds,
    is_ci_env,
):
    if is_ci_env:
        pytest.skip("Skip in CI, model is WIP, issue# 13357")

    torch_input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    torch_model = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    config = torch_model.config
    torch_model = torch_model.encoder.block[block_i][attention_i].attention

    reference_model = SegformerAttention(
        config,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        sequence_reduction_ratio=sequence_reduction_ratio,
    )

    sd = torch_model.state_dict()
    reference_model.load_state_dict(sd)
    reference_model.eval()

    output = reference_model(torch_input_tensor, height, width)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device), device=device
    )
    if "sr" in parameters["self"]:
        parameters["self"]["sr"]["weight"] = ttnn.from_device(parameters["self"]["sr"]["weight"])
        parameters["self"]["sr"]["bias"] = ttnn.from_device(parameters["self"]["sr"]["bias"])

    ttnn_model = TtSegformerAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        parameters=parameters,
        sequence_reduction_ratio=sequence_reduction_ratio,
    )

    ttnn_output = ttnn_model(ttnn_input_tensor, height, width, parameters=parameters)
    ttnn_final_output = ttnn.to_torch(ttnn_output[0])
    if len(ttnn_final_output.shape) == 4:
        ttnn_final_output = ttnn_final_output[0]

    assert_with_pcc(
        output[0], ttnn_final_output, pcc=0.85
    )  # 0.97 to 0.85 due to adding parameters to linear and layernorm
