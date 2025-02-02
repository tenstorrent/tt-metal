# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import numpy as np

from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_linear_bias,
)
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_stable_diffusion3_5.reference.ada_layernorm_continuous import AdaLayerNormContinuous
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_ada_layernorm_continuous import (
    ttnn_AdaLayerNormContinuous,
)
from models.experimental.functional_stable_diffusion3_5.ttnn_unopt.ttnn_ada_layernorm_continuous import (
    ttnn_AdaLayerNormContinuous as ttnn_AdaLayerNormContinuous_unopt,
)


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, AdaLayerNormContinuous):
            parameters["linear"] = {}
            parameters["linear"]["weight"] = preprocess_linear_weight(model.linear.weight, dtype=ttnn.bfloat16)
            parameters["linear"]["bias"] = preprocess_linear_bias(model.linear.bias, dtype=ttnn.bfloat16)

            # Its none as elementwise_affine=False
            parameters["norm"] = {}

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@skip_for_grayskull()
@pytest.mark.parametrize(
    "x_shape",
    [
        # ([2, 4096, 1536]),
        # ([2, 333, 1536]),
        ([2, 1024, 1536]),
        # ([2, 154, 1536]),
    ],
)
def test_ada_layernorm_continuous(device, x_shape, reset_seeds):
    reference_model = AdaLayerNormContinuous(
        embedding_dim=1536,
        conditioning_embedding_dim=1536,
        elementwise_affine=False,
        eps=1e-06,
        bias=True,
        norm_type="layer_norm",
    )  # .to(dtype=torch.bfloat16)
    reference_model.eval()

    torch_input_x = torch.randn(x_shape, dtype=torch.float)
    torch_input_conditioning_embedding = torch.randn(2, 1536, dtype=torch.float)

    for i in range(1):
        """
        numpy_array = np.load(
                "models/experimental/functional_stable_diffusion3_5/demo/demo_optim_512x512__TR_in0_layer_"
                + str(i)
                + ".npy"
            )
        torch_input_x = torch.from_numpy(numpy_array) #.to(dtype=torch.float16)

        numpy_array = np.load(
                "models/experimental/functional_stable_diffusion3_5/demo/demo_optim_512x512__TR_in2_layer_"
                + str(i)
                + ".npy"
            )
        torch_input_conditioning_embedding = torch.from_numpy(numpy_array) #.to(dtype=torch.float16)
        """
        parameters = preprocess_model_parameters(
            initialize_model=lambda: reference_model,
            custom_preprocessor=create_custom_preprocessor(device),
            device=device,
        )

        # torch_input_x_unsqueezed = torch_input_x.unsqueeze(1)

        # if torch_input_x_unsqueezed.shape[-2] < 512:
        #     input_memory_config = ttnn.L1_MEMORY_CONFIG
        # else:
        #     mm_a_y = 8
        #     mm_a_x = 8
        #     mm_a_x_strategy = ttnn.ShardStrategy.BLOCK
        #     mm_a_x_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG

        #     input_memory_config = ttnn.create_sharded_memory_config(
        #         torch_input_x_unsqueezed.shape,
        #         core_grid=ttnn.CoreGrid(y=mm_a_y, x=mm_a_x),
        #         strategy=mm_a_x_strategy,
        #         orientation=ttnn.ShardOrientation.ROW_MAJOR,
        #     )

        ttnn_input_conditioning_embedding = ttnn.from_torch(
            torch_input_conditioning_embedding.unsqueeze(0).unsqueeze(
                0
            ),  # .squeeze(1).squeeze(1), #.unsqueeze(1).unsqueeze(1),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ttnn_input_x = ttnn.from_torch(
            torch_input_x.unsqueeze(1),  # .squeeze(1), #_unsqueezed,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,  # input_memory_config
        )

        # torch_output = reference_model(torch_input_x.squeeze(1), torch_input_conditioning_embedding.squeeze(1).squeeze(1))
        torch_output = reference_model(torch_input_x, torch_input_conditioning_embedding)

        ttnn_model = ttnn_AdaLayerNormContinuous(
            embedding_dim=1536,
            conditioning_embedding_dim=1536,
            elementwise_affine=False,
            eps=1e-06,
            bias=True,
            norm_type="layer_norm",
        )

        ttnn_output = ttnn_model(ttnn_input_x, ttnn_input_conditioning_embedding, parameters=parameters)

        print(assert_with_pcc(torch_output.unsqueeze(1), ttnn.to_torch(ttnn_output), pcc=-100))

        #######

        ttnn_input_conditioning_embedding = ttnn.from_torch(
            torch_input_conditioning_embedding.squeeze(1).squeeze(1),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        ttnn_input_x = ttnn.from_torch(
            torch_input_x.squeeze(1),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,  # input_memory_config
        )

        ttnn_model_unopt = ttnn_AdaLayerNormContinuous_unopt(
            embedding_dim=1536,
            conditioning_embedding_dim=1536,
            elementwise_affine=False,
            eps=1e-06,
            bias=True,
            norm_type="layer_norm",
        )

        ttnn_output_unopt = ttnn_model_unopt(ttnn_input_x, ttnn_input_conditioning_embedding, parameters=parameters)

        print("--unopt--", i)
        print(assert_with_pcc(ttnn.to_torch(ttnn_output_unopt).unsqueeze(1), ttnn.to_torch(ttnn_output), pcc=-100))

        # print(ttnn.to_torch(ttnn_output_unopt))
        # print()
        # print(ttnn.to_torch(ttnn_output))
        # print("--")
        # print(ttnn.to_torch(ttnn_output_unopt))
        # print()
        # print(ttnn.to_torch(ttnn_output))
