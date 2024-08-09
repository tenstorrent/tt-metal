# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d
from tests.ttnn.utils_for_testing import assert_with_pcc

from transformers import SegformerModel
import pytest
from models.experimental.functional_segformer.tt.ttnn_segformer_model import (
    TtSegformerModel,
)
from models.experimental.functional_segformer.reference.segformer_model import SegformerModelReference


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["activation"] = None


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        parameters["encoder"] = {}
        if isinstance(model, SegformerModelReference):
            for i in range(4):
                parameters["encoder"][f"patch_embeddings_{i}"] = {}
                parameters["encoder"][f"patch_embeddings_{i}"]["proj"] = {}
                if i != 0:
                    update_ttnn_module_args(ttnn_module_args["encoder"][f"patch_embeddings_{i}"]["proj"])
                    parameters["encoder"][f"patch_embeddings_{i}"]["proj"], _ = preprocess_conv2d(
                        getattr(model.encoder, f"patch_embeddings_{i}").proj.weight,
                        getattr(model.encoder, f"patch_embeddings_{i}").proj.bias,
                        ttnn_module_args["encoder"][f"patch_embeddings_{i}"]["proj"],
                        return_parallel_config=True,
                    )
                else:
                    parameters["encoder"][f"patch_embeddings_{i}"]["proj"]["weight"] = getattr(
                        model.encoder, f"patch_embeddings_{i}"
                    ).proj.weight
                    parameters["encoder"][f"patch_embeddings_{i}"]["proj"]["bias"] = getattr(
                        model.encoder, f"patch_embeddings_{i}"
                    ).proj.bias
                parameters["encoder"][f"patch_embeddings_{i}"]["layer_norm"] = {}
                parameters["encoder"][f"patch_embeddings_{i}"]["layer_norm"]["weight"] = ttnn.from_torch(
                    getattr(model.encoder, f"patch_embeddings_{i}").layer_norm.weight,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
                parameters["encoder"][f"patch_embeddings_{i}"]["layer_norm"]["bias"] = ttnn.from_torch(
                    getattr(model.encoder, f"patch_embeddings_{i}").layer_norm.bias,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

            for i in range(4):
                for j in range(2):
                    parameters["encoder"][f"block_{i}_{j}"] = {}
                    parameters["encoder"][f"block_{i}_{j}"]["layer_norm_1"] = {}
                    parameters["encoder"][f"block_{i}_{j}"]["layer_norm_1"]["weight"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").layer_norm_1.weight,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["encoder"][f"block_{i}_{j}"]["layer_norm_1"]["bias"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").layer_norm_1.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["encoder"][f"block_{i}_{j}"]["attention"] = {}
                    parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"] = {}

                    parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["query"] = {}
                    parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["query"]["weight"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").attention.self.query.weight.T,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["query"]["bias"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").attention.self.query.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

                    parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["key"] = {}
                    parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["key"]["weight"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").attention.self.key.weight.T,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["key"]["bias"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").attention.self.key.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

                    parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["value"] = {}
                    parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["value"]["weight"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").attention.self.value.weight.T,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["value"]["bias"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").attention.self.value.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

                    if i != 3:
                        parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["sr"] = {}
                        parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["sr"]["weight"] = ttnn.from_torch(
                            getattr(model.encoder, f"block_{i}_{j}").attention.self.sr.weight,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                        )
                        parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["sr"]["bias"] = ttnn.from_torch(
                            getattr(model.encoder, f"block_{i}_{j}").attention.self.sr.bias,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                        )

                        parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["layer_norm"] = {}
                        parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["layer_norm"][
                            "weight"
                        ] = ttnn.from_torch(
                            getattr(model.encoder, f"block_{i}_{j}").attention.self.layer_norm.weight,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                        )
                        parameters["encoder"][f"block_{i}_{j}"]["attention"]["self"]["layer_norm"][
                            "bias"
                        ] = ttnn.from_torch(
                            getattr(model.encoder, f"block_{i}_{j}").attention.self.layer_norm.bias,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                        )

                    parameters["encoder"][f"block_{i}_{j}"]["attention"]["output"] = {}
                    parameters["encoder"][f"block_{i}_{j}"]["attention"]["output"]["dense"] = {}
                    parameters["encoder"][f"block_{i}_{j}"]["attention"]["output"]["dense"]["weight"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").attention.output.dense.weight.T,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["encoder"][f"block_{i}_{j}"]["attention"]["output"]["dense"]["bias"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").attention.output.dense.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

                    parameters["encoder"][f"block_{i}_{j}"]["layer_norm_2"] = {}
                    parameters["encoder"][f"block_{i}_{j}"]["layer_norm_2"]["weight"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").layer_norm_2.weight,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["encoder"][f"block_{i}_{j}"]["layer_norm_2"]["bias"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").layer_norm_2.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

                    parameters["encoder"][f"block_{i}_{j}"]["mlp"] = {}

                    parameters["encoder"][f"block_{i}_{j}"]["mlp"]["dense1"] = {}
                    parameters["encoder"][f"block_{i}_{j}"]["mlp"]["dense1"]["weight"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").mlp.dense1.weight.T,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["encoder"][f"block_{i}_{j}"]["mlp"]["dense1"]["bias"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").mlp.dense1.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

                    parameters["encoder"][f"block_{i}_{j}"]["mlp"]["dwconv"] = {}
                    ttnn_module_args["encoder"][f"block_{i}_{j}"]["mlp"]["dwconv"] = {}
                    parameters["encoder"][f"block_{i}_{j}"]["mlp"]["dwconv"]["dwconv"] = {}
                    parameters["encoder"][f"block_{i}_{j}"]["mlp"]["dwconv"]["dwconv"]["weight"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").mlp.dwconv.dwconv.weight,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["encoder"][f"block_{i}_{j}"]["mlp"]["dwconv"]["dwconv"]["bias"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").mlp.dwconv.dwconv.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

                    parameters["encoder"][f"block_{i}_{j}"]["mlp"]["dense2"] = {}
                    parameters["encoder"][f"block_{i}_{j}"]["mlp"]["dense2"]["weight"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").mlp.dense2.weight.T,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["encoder"][f"block_{i}_{j}"]["mlp"]["dense2"]["bias"] = ttnn.from_torch(
                        getattr(model.encoder, f"block_{i}_{j}").mlp.dense2.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

            for i in range(4):
                parameters["encoder"][f"layer_norm_{i}"] = {}
                parameters["encoder"][f"layer_norm_{i}"]["weight"] = ttnn.from_torch(
                    getattr(model.encoder, f"layer_norm_{i}").weight,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
                parameters["encoder"][f"layer_norm_{i}"]["bias"] = ttnn.from_torch(
                    getattr(model.encoder, f"layer_norm_{i}").bias,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "batch_size, num_channels, height, width",
    [
        (1, 3, 512, 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_model(
    batch_size,
    num_channels,
    height,
    width,
    device,
    reset_seeds,
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
    config = torch_model.config

    torch_model = torch_model
    reference_model = SegformerModelReference(config)
    state_dict = torch_model.state_dict()

    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    values = [parameter for name, parameter in state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    torch_output = reference_model(torch_input_tensor)

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: reference_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_model = TtSegformerModel(config, parameters, reference_model)

    ttnn_output = ttnn_model(
        ttnn_input_tensor,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        parameters=parameters,
        model=reference_model,
    )
    ttnn_final_output = ttnn.to_torch(ttnn_output[0])

    assert_with_pcc(torch_output[0], ttnn_final_output, pcc=0.89)
