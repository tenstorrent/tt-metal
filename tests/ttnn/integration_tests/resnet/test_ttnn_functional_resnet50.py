# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger
import math
from typing import Union, Tuple, List

import torch
import torchvision

import ttnn
from ttnn.model_preprocessing import (
    preprocess_model,
    preprocess_conv2d,
    fold_batch_norm2d_into_conv2d,
    fold_conv7s2_into_conv4s1,
    preprocess_remaining_children_and_parameters,
    convert_torch_model_to_ttnn_model,
)

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import (
    skip_for_wormhole_b0,
    pad_and_fold_conv_activation_for_unity_stride,
    is_wormhole_b0,
    is_grayskull,
)

from models.experimental.functional_resnet.tt.ttnn_functional_resnet import resnet_basic_block, resnet_bottleneck_block


## copied from ttlib version test:
# golden pcc is ordered fidelity, weight dtype, activation dtype
golden_pcc = {
    8: {
        (
            ttnn.MathFidelity.HiFi4,
            ttnn.bfloat16,
            ttnn.bfloat16,
        ): 0.983301,  # PCC: 0.9833017469734239             TODO: NEED DEBUGGING WHY THIS IS SLIGHTLY LOWER THAN TTLIB
        # ): 0.990804,  # Max ATOL Delta: 1.607335090637207, Max RTOL Delta: 115.62200164794922, PCC: 0.9908042840544742
        (
            ttnn.MathFidelity.HiFi4,
            ttnn.bfloat8_b,
            ttnn.bfloat16,
        ): 0.986301,  # Max ATOL Delta: 1.5697126388549805, Max RTOL Delta: 21.3042049407959, PCC: 0.9863013351442654
        (
            ttnn.MathFidelity.HiFi4,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
        ): 0.973763,  # Max ATOL Delta: 2.455164909362793, Max RTOL Delta: inf, PCC: 0.9737631427307492
        (
            ttnn.MathFidelity.HiFi4,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.978099,  # Max ATOL Delta: 1.955164909362793, Max RTOL Delta: inf, PCC: 0.9780993165966628
        (
            ttnn.MathFidelity.HiFi2,
            ttnn.bfloat16,
            ttnn.bfloat16,
        ): 0.983400,  # Max ATOL Delta: 1.7310011386871338, Max RTOL Delta: 369.5689392089844, PCC: 0.9834004200555363
        (
            ttnn.MathFidelity.HiFi2,
            ttnn.bfloat8_b,
            ttnn.bfloat16,
        ): 0.984828,  # Max ATOL Delta: 1.6054553985595703, Max RTOL Delta: 59.124324798583984, PCC: 0.9848281996919587
        (
            ttnn.MathFidelity.HiFi2,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
        ): 0.934073,  # Max ATOL Delta: 4.330164909362793, Max RTOL Delta: inf, PCC: 0.9340735819578696
        (
            ttnn.MathFidelity.HiFi2,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.944435,  # Max ATOL Delta: 4.705164909362793, Max RTOL Delta: inf, PCC: 0.9444350983635019
        (
            ttnn.MathFidelity.LoFi,
            ttnn.bfloat16,
            ttnn.bfloat16,
        ): 0.938909,  # Max ATOL Delta: 3.861414909362793, Max RTOL Delta: 240.63145446777344, PCC: 0.9389092547575272
        (
            ttnn.MathFidelity.LoFi,
            ttnn.bfloat8_b,
            ttnn.bfloat16,
        ): 0.959609,  # Max ATOL Delta: 3.205164909362793, Max RTOL Delta: 141.7057342529297, PCC: 0.9596095155046113
        (
            ttnn.MathFidelity.LoFi,
            ttnn.bfloat16,
            ttnn.bfloat8_b,
        ): 0.854903,  # Max ATOL Delta: 7.830164909362793, Max RTOL Delta: inf, PCC: 0.8549035869182201
        (
            ttnn.MathFidelity.LoFi,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.884609,  # Max ATOL Delta: 6.455164909362793, Max RTOL Delta: inf, PCC: 0.8846098380419433
    },
    16: {
        (
            ttnn.MathFidelity.HiFi4,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.978099,  # Max ATOL Delta: 1.955164909362793, Max RTOL Delta: inf, PCC: 0.9780993165966632
        (
            ttnn.MathFidelity.HiFi2,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.944435,  # Max ATOL Delta: 4.705164909362793, Max RTOL Delta: inf, PCC: 0.9444350983635021
        (
            ttnn.MathFidelity.LoFi,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.884609,  # Max ATOL Delta: 6.455164909362793, Max RTOL Delta: inf, PCC: 0.8846098380419435
    },
    20: {
        (
            ttnn.MathFidelity.HiFi4,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.978099,  # Max ATOL Delta: 1.955164909362793, Max RTOL Delta: inf, PCC: 0.9780993165966628
        (
            ttnn.MathFidelity.HiFi2,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.944435,  # Max ATOL Delta: 4.705164909362793, Max RTOL Delta: inf, PCC: 0.9444350983635021
        (
            ttnn.MathFidelity.LoFi,
            ttnn.bfloat8_b,
            ttnn.bfloat8_b,
        ): 0.884609,  # Max ATOL Delta: 6.455164909362793, Max RTOL Delta: inf, PCC: 0.8846098380419433
    },
}


def create_core_range_set_from_ncores(ncores: int, bb_ncores_w: int, bb_ncores_h: int):
    bb_ncores = bb_ncores_w * bb_ncores_h  ## total cores in the bounding box grid
    if ncores == bb_ncores:  ## no last partial core row
        return ttnn.experimental.tensor.CoreRangeSet(
            {
                ttnn.experimental.tensor.CoreRange(
                    ttnn.experimental.tensor.CoreCoord(0, 0),
                    ttnn.experimental.tensor.CoreCoord(bb_ncores_w - 1, bb_ncores_h - 1),
                )
            }
        )
    elif ncores < bb_ncores:  ## with last partial core row
        return ttnn.experimental.tensor.CoreRangeSet(
            {
                ttnn.experimental.tensor.CoreRange(
                    ttnn.experimental.tensor.CoreCoord(0, 0),
                    ttnn.experimental.tensor.CoreCoord(bb_ncores_w - 1, bb_ncores_h - 2),
                ),
                ttnn.experimental.tensor.CoreRange(
                    ttnn.experimental.tensor.CoreCoord(0, bb_ncores_h - 1),
                    ttnn.experimental.tensor.CoreCoord(ncores % bb_ncores_w - 1, bb_ncores_h - 1),
                ),
            }
        )
    else:
        assert False, "Invalid bounding box grid size"

    return None


def create_sharded_mem_config_resnet(
    tensor_shape: Union[ttnn.Shape, Tuple[int, ...], List[int]],
    ncores: int,  ## total num cores to use
    ncores_nhw: int,  ## num cores along the tensor nhw dimension
    max_grid_w: int,  ## grid max size
    max_grid_h: int,  ## grid max size
    shard_strategy: ttnn.ShardStrategy,
    shard_orientation: ttnn.ShardOrientation,
    tensor_layout: ttnn.Layout,
    snap_shard_height_to_tile: bool = False,
):
    ncores_w, ncores_h = 0, 0
    ncores_w_cliff = 0
    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        assert ncores <= max_grid_w * max_grid_h
        assert ncores_nhw <= max_grid_h
        assert ncores % ncores_nhw == 0
        ncores_h = ncores_nhw
        ncores_w = ncores // ncores_h
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        assert ncores == ncores_nhw
        assert ncores_nhw <= max_grid_w * max_grid_h
        ncores_w = max_grid_w
        ncores_h = int(math.ceil(ncores_nhw / ncores_w))
        ncores_w_cliff = ncores_nhw % ncores_w

    logger.debug(f"ncores_nhw: {ncores_nhw}")
    logger.debug(f"(bb_ncores_w, bb_ncores_h): {ncores_w}, {ncores_h}")

    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        tensor_memory_layout = ttnn.experimental.tensor.TensorMemoryLayout.BLOCK_SHARDED
    elif shard_strategy == ttnn.ShardStrategy.WIDTH:
        tensor_memory_layout = ttnn.experimental.tensor.TensorMemoryLayout.WIDTH_SHARDED
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        tensor_memory_layout = ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED
    else:
        raise RuntimeError("Invalid sharding strategy")

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        tensor_shard_orientation = ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR
    elif shard_orientation == ttnn.ShardOrientation.COL_MAJOR:
        tensor_shard_orientation = ttnn.experimental.tensor.ShardOrientation.COL_MAJOR
    else:
        raise RuntimeError("Invalid shard orientation")

    shard_grid = create_core_range_set_from_ncores(ncores_nhw, ncores_w, ncores_h)

    tensor_height = tensor_shape[0] * tensor_shape[1] * tensor_shape[2]
    tensor_width = tensor_shape[3]

    if snap_shard_height_to_tile:
        shard_height = int(math.ceil(tensor_height / (ncores_nhw * 32)) * (ncores_nhw * 32)) // ncores_nhw
    else:
        shard_height = int(math.ceil(tensor_height / ncores_nhw))

    if shard_strategy == ttnn.ShardStrategy.BLOCK:
        assert tensor_width % ncores_w == 0
        shard_width = tensor_width // ncores_w
    elif shard_strategy == ttnn.ShardStrategy.HEIGHT:
        if tensor_layout == ttnn.ROW_MAJOR_LAYOUT:
            shard_width = tensor_width
        else:
            shard_width = int(math.ceil(tensor_width / 32) * 32)

    logger.debug(f"tensor_shape: {tensor_shape}")
    logger.debug(f"shard_height: {shard_height}")
    logger.debug(f"shard_width: {shard_width}")

    shard_spec = ttnn.experimental.tensor.ShardSpec(
        shard_grid,
        [shard_height, shard_width],
        tensor_shard_orientation,
        False,
    )
    return ttnn.experimental.tensor.MemoryConfig(
        tensor_memory_layout, ttnn.experimental.tensor.BufferType.L1, shard_spec
    )


@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype, math_fidelity",
    (
        (8, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.HiFi4),  ## pass -- slightly lower pcc than ttlib
        (8, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),  ## pass
        # (16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),  ## L1 clash
        # (16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),   ## L1 clash
        # (20, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2),  ## L1 clash
        # (20, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.MathFidelity.LoFi),   ## L1 clash
    ),
)
def test_resnet_50(device, batch_size, act_dtype, weight_dtype, math_fidelity):
    torch.manual_seed(0)

    torch_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1).eval()

    def update_ttnn_module_args_resnet50(ttnn_module_args):
        ttnn_module_args["use_1d_systolic_array"] = True
        ttnn_module_args["enable_auto_formatting"] = False
        # ttnn_module_args["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 32}
        ttnn_module_args["deallocate_activation"] = True if ttnn_module_args.kernel_size == (3, 3) else False
        ttnn_module_args["weights_dtype"] = weight_dtype
        ttnn_module_args["dtype"] = act_dtype
        ttnn_module_args["math_fidelity"] = math_fidelity

    def custom_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        if isinstance(model, torchvision.models.resnet.Bottleneck):
            ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.conv2["activation"] = "relu"  # Fuse relu with conv1
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)
            conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.conv3, model.bn3)

            update_ttnn_module_args_resnet50(ttnn_module_args.conv1)
            update_ttnn_module_args_resnet50(ttnn_module_args.conv2)
            update_ttnn_module_args_resnet50(ttnn_module_args.conv3)
            if model.downsample is not None:
                update_ttnn_module_args_resnet50(ttnn_module_args.downsample[0])

            ## TODO: Cleanup this atrocity
            if ttnn_module_args.conv1.input_height <= 14 and ttnn_module_args.conv1.input_width <= 14:
                ttnn_module_args.conv1["use_1d_systolic_array"] = False
                if model.downsample is not None:
                    ttnn_module_args.downsample[0]["use_1d_systolic_array"] = False
            else:
                if ttnn_module_args.conv1.input_height == 28 and ttnn_module_args.conv1.input_width == 28:
                    if ttnn_module_args.conv1.stride == (2, 2):
                        ttnn_module_args.conv1["use_1d_systolic_array"] = False
                        if model.downsample is not None:
                            ttnn_module_args.downsample[0]["use_1d_systolic_array"] = False
                    else:
                        ttnn_module_args.conv1["use_1d_systolic_array"] = True
                        if model.downsample is not None:
                            ttnn_module_args.downsample[0]["use_1d_systolic_array"] = True
                else:
                    ttnn_module_args.conv1["use_1d_systolic_array"] = True
                    if model.downsample is not None:
                        ttnn_module_args.downsample[0]["use_1d_systolic_array"] = True

            if ttnn_module_args.conv2.input_height <= 14 and ttnn_module_args.conv2.input_width <= 14:
                ttnn_module_args.conv2["use_1d_systolic_array"] = False
                if model.downsample is not None:
                    ttnn_module_args.downsample[0]["use_1d_systolic_array"] = False
            else:
                if ttnn_module_args.conv2.input_height == 28 and ttnn_module_args.conv2.input_width == 28:
                    if ttnn_module_args.conv2.stride == (2, 2):
                        ttnn_module_args.conv2["use_1d_systolic_array"] = False
                        if model.downsample is not None:
                            ttnn_module_args.downsample[0]["use_1d_systolic_array"] = False
                    else:
                        ttnn_module_args.conv2["use_1d_systolic_array"] = True
                else:
                    ttnn_module_args.conv2["use_1d_systolic_array"] = True

            if ttnn_module_args.conv3.input_height <= 14 and ttnn_module_args.conv3.input_width <= 14:
                ttnn_module_args.conv3["use_1d_systolic_array"] = False
                if model.downsample is not None:
                    ttnn_module_args.downsample[0]["use_1d_systolic_array"] = False
            else:
                if ttnn_module_args.conv3.input_height == 28 and ttnn_module_args.conv3.input_width == 28:
                    if ttnn_module_args.conv3.stride == (2, 2):
                        ttnn_module_args.conv3["use_1d_systolic_array"] = False
                        if model.downsample is not None:
                            ttnn_module_args.downsample[0]["use_1d_systolic_array"] = False
                    else:
                        ttnn_module_args.conv3["use_1d_systolic_array"] = True
                else:
                    ttnn_module_args.conv3["use_1d_systolic_array"] = True

            parameters["conv1"], pconfig1 = preprocess_conv2d(conv1_weight, conv1_bias, ttnn_module_args.conv1, True)
            parameters["conv2"], pconfig2 = preprocess_conv2d(conv2_weight, conv2_bias, ttnn_module_args.conv2, True)
            parameters["conv3"], pconfig3 = preprocess_conv2d(conv3_weight, conv3_bias, ttnn_module_args.conv3, True)

            logger.debug(f"pconfig1: {pconfig1.num_cores_nhw}")
            logger.debug(f"pconfig2: {pconfig2.num_cores_nhw}")
            logger.debug(f"pconfig3: {pconfig3.num_cores_nhw}")

            if model.downsample is not None:
                ttnn_module_args.downsample[0]["use_dram_for_matmul"] = True
                downsample_weight, downsample_bias = fold_batch_norm2d_into_conv2d(
                    model.downsample[0], model.downsample[1]
                )
                parameters["downsample"], pconfig4 = preprocess_conv2d(
                    downsample_weight, downsample_bias, ttnn_module_args.downsample[0], True
                )
                ttnn_module_args["downsample"] = ttnn_module_args.downsample[0]

                logger.debug(f"pconfig4: {pconfig4.num_cores_nhw}")

        elif isinstance(model, torchvision.models.resnet.ResNet):
            ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.conv1["deallocate_activation"] = True
            ttnn_module_args.conv1["reallocate_halo_output"] = True
            ## ttnn_module_args.conv1["use_shallow_conv_variant"] = True
            ttnn_module_args.conv1["padded_input_channels"] = 16
            ttnn_module_args.conv1["math_fidelity"] = math_fidelity
            ttnn_module_args.conv1["weights_dtype"] = weight_dtype
            ttnn_module_args.conv1["dtype"] = act_dtype
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
            parameters["conv1"] = fold_conv7s2_into_conv4s1(conv1_weight, conv1_bias, ttnn_module_args.conv1)

            named_parameters = tuple(
                (name, parameter) for name, parameter in model.named_parameters() if "." not in name
            )
            for child_name, child in tuple(model.named_children()) + named_parameters:
                if child_name in {"conv1", "bn1"}:
                    continue
                parameters[child_name] = convert_torch_model_to_ttnn_model(
                    child,
                    name=name,
                    convert_to_ttnn=convert_to_ttnn,
                    custom_preprocessor=custom_preprocessor,
                    ttnn_module_args=ttnn_module_args.get(child_name, None),
                )
        return parameters

    input_shape = (batch_size, 3, 224, 224)

    reader_patterns_cache = {}
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    parameters = preprocess_model(
        initialize_model=lambda: torch_model,
        run_model=lambda model: model(torch_input_tensor),
        reader_patterns_cache=reader_patterns_cache,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    for module in range(1, 5):
        parameters.layer3[module].conv1.conv.is_1d_systolic = False
        parameters.layer3[module].conv2.conv.is_1d_systolic = False
        parameters.layer3[module].conv3.conv.is_1d_systolic = False
    for module in range(1, 3):
        parameters.layer4[module].conv1.conv.is_1d_systolic = False
        parameters.layer4[module].conv2.conv.is_1d_systolic = False
        parameters.layer4[module].conv3.conv.is_1d_systolic = False

    torch_model.to(torch.bfloat16)
    torch_input_tensor = torch_input_tensor.to(torch.bfloat16)

    ## golden

    torch_output_tensor = torch_model(torch_input_tensor)

    ## ttnn

    input_tensor = pad_and_fold_conv_activation_for_unity_stride(
        torch_input_tensor, *torch_model.conv1.padding, *torch_model.conv1.stride
    )
    input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))

    ## reshape to [1, 1, N*H*W, C]
    input_tensor = torch.reshape(input_tensor, (1, 1, -1, input_tensor.shape[-1]))
    input_shape = input_tensor.shape

    ## create shard spec
    ncores_nhw = parameters.conv1.conv.sliding_window_op_params.num_cores_nhw
    ncores_w = parameters.conv1.conv.sliding_window_op_params.num_cores_w
    ncores_h = parameters.conv1.conv.sliding_window_op_params.num_cores_h

    logger.debug(f"ncores_nhw: {ncores_nhw}")
    logger.debug(f"ncores_w, ncores_h: {ncores_w}, {ncores_h}")

    ## layout is TILE for bfp8_b, otherwise ROW_MAJOR
    # layout = ttnn.TILE_LAYOUT if act_dtype == ttnn.bfloat8_b else ttnn.ROW_MAJOR_LAYOUT
    layout = ttnn.ROW_MAJOR_LAYOUT  ## always start with RM so that UTWH doesn't need to untilize unnecessarily

    ## construct sharding config
    grid_size = device.compute_with_storage_grid_size()
    sharded_mem_config = create_sharded_mem_config_resnet(
        input_shape,
        ncores_nhw,
        ncores_nhw,
        grid_size.x,
        grid_size.y,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        layout,
        True,
    )
    ## NOTE: shards are snapped to tile height, so need to update the act tensor with required padding at end
    input_tensor_height_snapped_to_tile = sharded_mem_config.shard_spec.shape[0] * ncores_nhw
    if layout == ttnn.ROW_MAJOR_LAYOUT:
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, 0, 0, input_tensor_height_snapped_to_tile - input_shape[2], 0, 0)
        )
    else:
        padded_shard_width = sharded_mem_config.shard_spec.shape[1]  ## height sharding only
        TILE_WIDTH = 32
        channels_padding = padded_shard_width - (input_shape[3] % TILE_WIDTH)
        input_tensor = torch.nn.functional.pad(
            input_tensor, (0, channels_padding, 0, input_tensor_height_snapped_to_tile - input_shape[2], 0, 0)
        )

    input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=layout)

    ## copy input to device sharded directly
    output_tensor = ttnn.to_device(input_tensor, device=device, memory_config=sharded_mem_config)

    output_tensor = parameters.conv1(output_tensor)
    output_tensor = parameters.maxpool(output_tensor)

    """
    1st bottleneck layer. all the blocks implemented by ttnn
    """
    output_tensor = ttnn.reshape(output_tensor, (1, 1, 56 * 56 * batch_size, 64))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT, dtype=act_dtype)

    module = 1
    for bottleneck_block_parameters in list(parameters.layer1.values()):
        logger.debug(f"parameters 1st block {bottleneck_block_parameters}")
        output_tensor = resnet_bottleneck_block(
            output_tensor, bottleneck_block_parameters, layer=1, module=module, device=device
        )
        module += 1

    """
    2nd bottleneck layer. 1st block implemented by torch rest by ttnn
    """
    module = 1
    for bottleneck_block_parameters in list(parameters.layer2.values()):
        logger.debug(f"parameters 2nd block {bottleneck_block_parameters}")
        output_tensor = resnet_bottleneck_block(
            output_tensor, bottleneck_block_parameters, layer=2, device=device, module=module
        )
        module += 1

    """
    3rd bottleneck layer. 1st block implemented by torch rest by ttnn
    """
    module = 1
    for bottleneck_block_parameters in list(parameters.layer3.values()):
        logger.debug(f"parameters 3rd block {bottleneck_block_parameters}")
        output_tensor = resnet_bottleneck_block(
            output_tensor, bottleneck_block_parameters, layer=3, module=module, device=device
        )
        module += 1

    """
    4th bottleneck layer. 1st block implemented by torch rest by ttnn
    """
    module = 1
    for bottleneck_block_parameters in list(parameters.layer4.values()):
        logger.debug(f"parameters 4th block {bottleneck_block_parameters}")
        output_tensor = resnet_bottleneck_block(
            output_tensor, bottleneck_block_parameters, layer=4, module=module, device=device
        )
        module += 1

    # """
    # the last layers of the resnet
    # """
    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.reshape(output_tensor, (batch_size, 1, 49, 2048))
    output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
    output_tensor = ttnn.global_avg_pool2d(output_tensor)
    output_tensor = output_tensor @ parameters.fc.weight + parameters.fc.bias

    """
    output verify
    """
    output_tensor = ttnn.to_torch(output_tensor)
    output_tensor = torch.reshape(output_tensor, (batch_size, 1000))

    valid_pcc = 1.0
    if batch_size >= 8:
        valid_pcc = golden_pcc[batch_size][(math_fidelity, weight_dtype, act_dtype)]
    else:
        if act_dtype == ttnn.bfloat8_b:
            if math_fidelity == ttnn.MathFidelity.LoFi:
                valid_pcc = 0.87
            else:
                valid_pcc = 0.94
        else:
            if math_fidelity == ttnn.MathFidelity.LoFi:
                valid_pcc = 0.93
            else:
                valid_pcc = 0.982

    assert_with_pcc(torch_output_tensor, output_tensor, pcc=valid_pcc)
