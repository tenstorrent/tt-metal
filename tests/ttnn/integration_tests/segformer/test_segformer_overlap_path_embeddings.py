# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d
from tests.ttnn.utils_for_testing import assert_with_pcc
from transformers import SegformerModel
import pytest
from models.experimental.functional_segformer.tt.ttnn_segformer_overlap_patch_embeddings import (
    TtSegformerOverlapPatchEmbeddings,
)

from models.experimental.functional_segformer.reference.segformer_overlap_patch_embeddings import (
    SegformerOverlapPatchEmbeddings,
)


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["deallocate_activation"] = True


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SegformerOverlapPatchEmbeddings):
            parameters["proj"] = {}

            if model.proj.stride[0] != 4:
                update_ttnn_module_args(ttnn_module_args["proj"])
                ttnn_module_args["proj"]["activation"] = None
                ttnn_module_args["proj"]["use_1d_systolic_array"] = True

                parameters["proj"], _ = preprocess_conv2d(
                    model.proj.weight, model.proj.bias, ttnn_module_args["proj"], return_parallel_config=True
                )
            else:
                parameters["proj"]["weight"] = model.proj.weight
                parameters["proj"]["bias"] = model.proj.bias

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "patch_size, stride, num_channels, hidden_size, batch_size, height, width, patch_emb_i",
    [
        (7, 4, 3, 32, 1, 512, 512, 0),
        (3, 2, 32, 64, 1, 128, 128, 1),
        (3, 2, 64, 160, 1, 64, 64, 2),
        (3, 2, 160, 256, 1, 32, 32, 3),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_overlap_patch_embeddings(
    patch_size, stride, num_channels, hidden_size, batch_size, height, width, patch_emb_i, device, reset_seeds
):
    torch_input_tensor = torch.randn(batch_size, num_channels, height, width)
    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    torch_model = SegformerModel.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    torch_model = torch_model.encoder.patch_embeddings[patch_emb_i]
    reference_model = SegformerOverlapPatchEmbeddings(
        patch_size=patch_size, stride=stride, num_channels=num_channels, hidden_size=hidden_size
    )
    sd = torch_model.state_dict()
    reference_model.load_state_dict(sd)
    reference_model.eval()
    torch_output = reference_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: reference_model,
        run_model=lambda model: model(torch_input_tensor),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtSegformerOverlapPatchEmbeddings(parameters, reference_model)

    ttnn_output = ttnn_model(ttnn_input_tensor, parameters=parameters, model=reference_model)
    ttnn_final_output_1 = ttnn.to_torch(ttnn_output[0])
    ttnn_final_output_2 = ttnn.to_torch(ttnn_output[1])

    assert_with_pcc(torch_output[0], ttnn_final_output_1, pcc=0.99)

    assert_with_pcc(torch_output[1], ttnn_final_output_2, pcc=0.99)  # For height,width
