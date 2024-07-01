# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

from loguru import logger
import pytest
import torch
from models.experimental.functional_segformer.reference.segformer_for_semantic_segmentation import (
    SegformerForSemanticSegmentationReference,
)
from models.experimental.functional_segformer.tt.ttnn_segformer_for_semantic_segmentation import (
    TtSegformerForSemanticSegmentation,
)

from ttnn.model_preprocessing import preprocess_model, preprocess_conv2d, fold_batch_norm2d_into_conv2d

from transformers import SegformerForSemanticSegmentation

from models.utility_functions import (
    skip_for_wormhole_b0,
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)
from models.perf.perf_utils import prep_perf_report

import ttnn


def update_ttnn_module_args(ttnn_module_args):
    ttnn_module_args["use_1d_systolic_array"] = ttnn_module_args.in_channels < 256
    ttnn_module_args["dtype"] = ttnn.bfloat8_b
    ttnn_module_args["math_fidelity"] = ttnn.MathFidelity.LoFi
    ttnn_module_args["weights_dtype"] = ttnn.bfloat8_b
    ttnn_module_args["activation"] = None


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, SegformerForSemanticSegmentationReference):
            parameters["segformer"] = {}
            parameters["segformer"]["encoder"] = {}
            for i in range(4):
                parameters["segformer"]["encoder"][f"patch_embeddings_{i}"] = {}
                parameters["segformer"]["encoder"][f"patch_embeddings_{i}"]["proj"] = {}
                if i != 0:
                    update_ttnn_module_args(ttnn_module_args["segformer"]["encoder"][f"patch_embeddings_{i}"]["proj"])
                    parameters["segformer"]["encoder"][f"patch_embeddings_{i}"]["proj"], _ = preprocess_conv2d(
                        getattr(model.segformer.encoder, f"patch_embeddings_{i}").proj.weight,
                        getattr(model.segformer.encoder, f"patch_embeddings_{i}").proj.bias,
                        ttnn_module_args["segformer"]["encoder"][f"patch_embeddings_{i}"]["proj"],
                        return_parallel_config=True,
                    )
                else:
                    parameters["segformer"]["encoder"][f"patch_embeddings_{i}"]["proj"]["weight"] = getattr(
                        model.segformer.encoder, f"patch_embeddings_{i}"
                    ).proj.weight
                    parameters["segformer"]["encoder"][f"patch_embeddings_{i}"]["proj"]["bias"] = getattr(
                        model.segformer.encoder, f"patch_embeddings_{i}"
                    ).proj.bias
                parameters["segformer"]["encoder"][f"patch_embeddings_{i}"]["layer_norm"] = {}
                parameters["segformer"]["encoder"][f"patch_embeddings_{i}"]["layer_norm"]["weight"] = ttnn.from_torch(
                    getattr(model.segformer.encoder, f"patch_embeddings_{i}").layer_norm.weight,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
                parameters["segformer"]["encoder"][f"patch_embeddings_{i}"]["layer_norm"]["bias"] = ttnn.from_torch(
                    getattr(model.segformer.encoder, f"patch_embeddings_{i}").layer_norm.bias,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

            for i in range(4):
                for j in range(2):
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"] = {}
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["layer_norm_1"] = {}
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["layer_norm_1"]["weight"] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").layer_norm_1.weight,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["layer_norm_1"]["bias"] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").layer_norm_1.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"] = {}
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"] = {}

                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["query"] = {}
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["query"][
                        "weight"
                    ] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").attention.self.query.weight.T,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["query"][
                        "bias"
                    ] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").attention.self.query.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["key"] = {}
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["key"][
                        "weight"
                    ] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").attention.self.key.weight.T,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["key"][
                        "bias"
                    ] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").attention.self.key.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["value"] = {}
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["value"][
                        "weight"
                    ] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").attention.self.value.weight.T,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["value"][
                        "bias"
                    ] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").attention.self.value.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

                    if i != 3:
                        parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["sr"] = {}
                        parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["sr"][
                            "weight"
                        ] = ttnn.from_torch(
                            getattr(model.segformer.encoder, f"block_{i}_{j}").attention.self.sr.weight,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                        )
                        parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["sr"][
                            "bias"
                        ] = ttnn.from_torch(
                            getattr(model.segformer.encoder, f"block_{i}_{j}").attention.self.sr.bias,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                        )

                        parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["layer_norm"] = {}
                        parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["layer_norm"][
                            "weight"
                        ] = ttnn.from_torch(
                            getattr(model.segformer.encoder, f"block_{i}_{j}").attention.self.layer_norm.weight,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                        )
                        parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["self"]["layer_norm"][
                            "bias"
                        ] = ttnn.from_torch(
                            getattr(model.segformer.encoder, f"block_{i}_{j}").attention.self.layer_norm.bias,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                        )

                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["output"] = {}
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["output"]["dense"] = {}
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["output"]["dense"][
                        "weight"
                    ] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").attention.output.dense.weight.T,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["attention"]["output"]["dense"][
                        "bias"
                    ] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").attention.output.dense.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["layer_norm_2"] = {}
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["layer_norm_2"]["weight"] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").layer_norm_2.weight,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["layer_norm_2"]["bias"] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").layer_norm_2.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["mlp"] = {}

                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["mlp"]["dense1"] = {}
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["mlp"]["dense1"]["weight"] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").mlp.dense1.weight.T,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["mlp"]["dense1"]["bias"] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").mlp.dense1.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["mlp"]["dwconv"] = {}
                    ttnn_module_args["segformer"]["encoder"][f"block_{i}_{j}"]["mlp"]["dwconv"] = {}
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["mlp"]["dwconv"]["dwconv"] = {}
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["mlp"]["dwconv"]["dwconv"][
                        "weight"
                    ] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").mlp.dwconv.dwconv.weight,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["mlp"]["dwconv"]["dwconv"][
                        "bias"
                    ] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").mlp.dwconv.dwconv.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["mlp"]["dense2"] = {}
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["mlp"]["dense2"]["weight"] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").mlp.dense2.weight.T,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )
                    parameters["segformer"]["encoder"][f"block_{i}_{j}"]["mlp"]["dense2"]["bias"] = ttnn.from_torch(
                        getattr(model.segformer.encoder, f"block_{i}_{j}").mlp.dense2.bias,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                    )

            for i in range(4):
                parameters["segformer"]["encoder"][f"layer_norm_{i}"] = {}
                parameters["segformer"]["encoder"][f"layer_norm_{i}"]["weight"] = ttnn.from_torch(
                    getattr(model.segformer.encoder, f"layer_norm_{i}").weight,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
                parameters["segformer"]["encoder"][f"layer_norm_{i}"]["bias"] = ttnn.from_torch(
                    getattr(model.segformer.encoder, f"layer_norm_{i}").bias,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

            parameters["decode_head"] = {}
            for i in range(4):
                parameters["decode_head"][f"mlp_{i}"] = {}
                parameters["decode_head"][f"mlp_{i}"]["proj"] = {}
                parameters["decode_head"][f"mlp_{i}"]["proj"]["weight"] = ttnn.from_torch(
                    getattr(model.decode_head, f"mlp_{i}").proj.weight.T,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
                parameters["decode_head"][f"mlp_{i}"]["proj"]["bias"] = ttnn.from_torch(
                    getattr(model.decode_head, f"mlp_{i}").proj.bias,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )

            conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                model.decode_head.linear_fuse, model.decode_head.batch_norm
            )
            update_ttnn_module_args(ttnn_module_args["decode_head"]["linear_fuse"])
            ttnn_module_args["decode_head"]["linear_fuse"]["activation"] = "relu"
            ttnn_module_args["decode_head"]["linear_fuse"]["use_shallow_conv_variant"] = True
            ttnn_module_args["decode_head"]["linear_fuse"]["use_1d_systolic_array"] = True
            ttnn_module_args["decode_head"]["linear_fuse"]["conv_blocking_and_parallelization_config_override"] = {
                "act_block_h": 64
            }
            parameters["decode_head"]["linear_fuse"], _ = preprocess_conv2d(
                conv_weight, conv_bias, ttnn_module_args["decode_head"]["linear_fuse"], return_parallel_config=True
            )

            parameters["decode_head"]["classifier"], _ = preprocess_conv2d(
                model.decode_head.classifier.weight,
                model.decode_head.classifier.bias,
                ttnn_module_args["decode_head"]["classifier"],
                return_parallel_config=True,
            )

        return parameters

    return custom_preprocessor


@skip_for_wormhole_b0()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_segformer_performance(
    device,
    use_program_cache,
    batch_size=1,
    num_channels=3,
    height=512,
    width=512,
    expected_compile_time=20,
    expected_inference_time=20,
):
    disable_persistent_kernel_cache()

    torch_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

    torch_input_tensor = torch.randn(batch_size, num_channels, height, width)
    config = torch_model.config

    torch_model = torch_model
    reference_model = SegformerForSemanticSegmentationReference(config)
    state_dict = torch_model.state_dict()

    new_state_dict = {}
    keys = [name for name, parameter in reference_model.state_dict().items()]
    values = [parameter for name, parameter in state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    reference_model.load_state_dict(new_state_dict)
    reference_model.eval()

    reader_patterns_cache = {}
    parameters = preprocess_model(
        initialize_model=lambda: reference_model,
        run_model=lambda model: model(torch_input_tensor),
        custom_preprocessor=create_custom_preprocessor(device),
        reader_patterns_cache=reader_patterns_cache,
        device=device,
    )

    ttnn_input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    ttnn_model = TtSegformerForSemanticSegmentation(config, parameters, reference_model)

    durations = []
    for _ in range(2):
        start = time.time()
        output = ttnn_model(
            ttnn_input_tensor,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            parameters=parameters,
            model=reference_model,
        )
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    prep_perf_report(
        model_name="nvidia/segformer-b0-finetuned-ade-512-512",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Tokens per second: {1 / inference_time * batch_size}")
