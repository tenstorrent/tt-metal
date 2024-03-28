# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from typing import Union, Tuple, List
import math

# TODO remove this and pass in parameters instead when building ResNet50
import torchvision
import torch


from ttnn.model_preprocessing import (
    preprocess_model,
    preprocess_model_parameters,
    preprocess_conv2d,
    fold_batch_norm2d_into_conv2d,
    fold_conv7s2_into_conv4s1,
    preprocess_remaining_children_and_parameters,
    convert_torch_model_to_ttnn_model,
)
from models.utility_functions import (
    skip_for_wormhole_b0,
    pad_and_fold_conv_activation_for_unity_stride,
    is_wormhole_b0,
    is_grayskull,
    _nearest_32,
)


def resnet_basic_block(x, *, parameters):
    identity = x

    # Relu and bn1 are fused with conv1
    conv1 = parameters.conv1(x)

    # Relu and bn2 are fused with conv1
    conv2 = parameters.conv2(conv1)
    ttnn.deallocate(conv1)

    if "downsample" in parameters and parameters.downsample is not None:
        identity = parameters.downsample(x)
        ttnn.deallocate(x)

    identity = ttnn.reshape(identity, conv2.shape)
    out = ttnn.add_and_apply_activation(conv2, identity, activation="relu", memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(conv2)
    if x is not identity:
        ttnn.deallocate(identity)

    return out


def create_sharded_mem_config(x, is_1d, core_grid, strategy, orientation, halo, use_height_and_width_as_shard_shape):
    mem_layout = ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED if is_1d else ttnn.types.TensorMemoryLayout.BLOCK_SHARDED
    core_grid = ttnn.CoreGrid(x=12, y=9)
    shard_grid = ttnn.experimental.tensor.CoreRangeSet(
        {
            ttnn.experimental.tensor.CoreRange(
                ttnn.experimental.tensor.CoreCoord(0, 0),
                ttnn.experimental.tensor.CoreCoord(core_grid.x - 1, core_grid.y - 1),
            )
        }
    )
    num_cores_nhw = core_grid.x * core_grid.y
    if is_1d:
        shard_shape = x.shape[0] * x.shape[1] * x.shape[2] // num_cores_nhw
    else:
        shard_shape = x.shape[1] * x.shape[1] * x.shape[2] // core_grid.y, x.shape[3] // core_grid.x
    shard_spec = ttnn.experimental.tensor.ShardSpec(
        shard_grid, shard_shape, ttnn.experimental.tensor.ShardOrientation.COL_MAJOR, False
    )
    return ttnn.types.MemoryConfig(mem_layout, ttnn.types.BufferType.L1, shard_spec)


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


def do_reshard(output_tensor, input_mem_config):
    if ttnn.get_memory_config(output_tensor) != input_mem_config:
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_memory_config(output_tensor, input_mem_config)
    return output_tensor


def resnet_bottleneck_block(x, parameters, layer=None, module=None, device=None):
    conv1 = parameters.conv1(x)
    conv1 = do_reshard(conv1, parameters.conv2.conv.input_sharded_memory_config)

    identity = x

    conv2 = parameters.conv2(conv1)
    if conv1.is_allocated():
        ttnn.deallocate(conv1)

    conv3 = parameters.conv3(conv2)
    ttnn.deallocate(conv2)

    conv3_mem_config = ttnn.get_memory_config(conv3)
    # if layer is not None and layer >= 3:
    #     conv3 = ttnn.to_memory_config(conv3, ttnn.DRAM_MEMORY_CONFIG)

    if "downsample" in parameters and parameters.downsample is not None:
        identity = do_reshard(identity, parameters.downsample.conv.input_sharded_memory_config)
        if layer is not None and layer == 2:
            if x.is_allocated() and x is not identity:
                ttnn.deallocate(x)
            if module >= 2:
                identity = ttnn.experimental.tensor.move_sharded(identity)
        identity = parameters.downsample(identity)

    if layer is not None and layer >= 3:
        conv3 = ttnn.to_memory_config(conv3, conv3_mem_config)
    conv3 = ttnn.reshape(conv3, identity.shape)
    mem_config = ttnn.get_memory_config(conv3)
    out = ttnn.add_and_apply_activation(conv3, identity, activation="relu", memory_config=mem_config)
    ttnn.deallocate(conv3)

    if x is not identity:
        ttnn.deallocate(identity)

    if (layer is not None and module is not None) and (
        (layer == 1 and module == 1)
        or (layer == 1 and module == 2 and is_grayskull())
        or (layer == 1 and module == 3 and is_grayskull())
    ):
        out = ttnn.experimental.tensor.move_sharded(out)

    return out


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


class ResNet50:
    def __init__(self, device, torch_model, input_shape, batch_size, act_dtype, weight_dtype, math_fidelity):
        super().__init__()
        self.device = device
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.act_dtype = act_dtype
        self.weight_dtype = weight_dtype
        self.math_fidelity = math_fidelity
        if is_wormhole_b0():
            self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=math_fidelity,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
        else:
            self.compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
                math_fidelity=self.math_fidelity,
                math_approx_mode=True,
            )

        torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
        self.impl = preprocess_model(
            initialize_model=lambda: torch_model,
            run_model=lambda model: model(torch_input_tensor),
            reader_patterns_cache={},
            custom_preprocessor=self.custom_preprocessor,
            device=device,
        )

        ## create shard spec
        ncores_nhw = self.impl.conv1.conv.sliding_window_op_params.num_cores_nhw
        ncores_w = self.impl.conv1.conv.sliding_window_op_params.num_cores_w
        ncores_h = self.impl.conv1.conv.sliding_window_op_params.num_cores_h

        logger.debug(f"ncores_nhw: {ncores_nhw}")
        logger.debug(f"ncores_w, ncores_h: {ncores_w}, {ncores_h}")

        ## layout is TILE for bfp8_b, otherwise ROW_MAJOR
        # layout = ttnn.TILE_LAYOUT if act_dtype == ttnn.bfloat8_b else ttnn.ROW_MAJOR_LAYOUT
        layout = ttnn.ROW_MAJOR_LAYOUT  ## always start with RM so that UTWH doesn't need to untilize unnecessarily

        ## construct sharding config
        grid_size = device.compute_with_storage_grid_size()
        self.sharded_mem_config = create_sharded_mem_config_resnet(
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

        self.input_tensor_height_snapped_to_tile = self.sharded_mem_config.shard_spec.shape[0] * ncores_nhw

    def update_ttnn_module_args_resnet50(self, ttnn_module_args):
        ttnn_module_args["use_1d_systolic_array"] = True
        ttnn_module_args["enable_auto_formatting"] = False
        ttnn_module_args["deallocate_activation"] = True if ttnn_module_args.kernel_size == (3, 3) else False
        ttnn_module_args["weights_dtype"] = self.weight_dtype
        ttnn_module_args["dtype"] = self.act_dtype
        ttnn_module_args["math_fidelity"] = self.math_fidelity

    def custom_preprocessor(self, model, name, ttnn_module_args, convert_to_ttnn):
        parameters = {}
        if isinstance(model, torchvision.models.resnet.Bottleneck):
            ttnn_module_args.conv1["activation"] = "relu"  # Fuse relu with conv1
            ttnn_module_args.conv2["activation"] = "relu"  # Fuse relu with conv1
            conv1_weight, conv1_bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
            conv2_weight, conv2_bias = fold_batch_norm2d_into_conv2d(model.conv2, model.bn2)
            conv3_weight, conv3_bias = fold_batch_norm2d_into_conv2d(model.conv3, model.bn3)

            self.update_ttnn_module_args_resnet50(ttnn_module_args.conv1)
            self.update_ttnn_module_args_resnet50(ttnn_module_args.conv2)
            self.update_ttnn_module_args_resnet50(ttnn_module_args.conv3)
            if model.downsample is not None:
                self.update_ttnn_module_args_resnet50(ttnn_module_args.downsample[0])

            module_has_a_bypass_path = model.downsample is not None
            if (
                self.batch_size == 20
                and ttnn_module_args.conv1.input_height == 56
                and ttnn_module_args.conv1.in_channels == 256
                and module_has_a_bypass_path
            ):
                ttnn_module_args.conv2["reallocate_halo_output"] = True

            ttnn_module_args.conv1["compute_kernel_config"] = self.compute_kernel_config
            ttnn_module_args.conv2["compute_kernel_config"] = self.compute_kernel_config
            ttnn_module_args.conv3["compute_kernel_config"] = self.compute_kernel_config

            if self.batch_size == 20 and ttnn_module_args.conv3.input_height == 56:
                ttnn_module_args.conv2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 320}
            if self.batch_size == 20 and ttnn_module_args.conv3.input_height == 28:
                ttnn_module_args.conv2["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 160}

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
            if is_grayskull():
                ttnn_module_args.conv1["use_shallow_conv_variant"] = True
            ttnn_module_args.conv1["padded_input_channels"] = 16
            ttnn_module_args.conv1["math_fidelity"] = self.math_fidelity
            ttnn_module_args.conv1["weights_dtype"] = self.weight_dtype
            if self.batch_size == 20:
                ttnn_module_args.conv1["conv_blocking_and_parallelization_config_override"] = {"act_block_h": 1280}
            ttnn_module_args.conv1["dtype"] = self.act_dtype
            ttnn_module_args.conv1["compute_kernel_config"] = self.compute_kernel_config
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
                    custom_preprocessor=self.custom_preprocessor,
                    ttnn_module_args=ttnn_module_args.get(child_name, None),
                )
        return parameters

    def convert_from_torch(self, torch_input_tensor, pad_h, pad_w, stride_h, stride_w):
        input_tensor = pad_and_fold_conv_activation_for_unity_stride(
            torch_input_tensor, pad_h, pad_w, stride_h, stride_w
        )
        input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))

        ## reshape to [1, 1, N*H*W, C]
        input_tensor = torch.reshape(input_tensor, (1, 1, -1, input_tensor.shape[-1]))
        input_shape = input_tensor.shape

        ## create shard spec
        ncores_nhw = self.impl.conv1.conv.sliding_window_op_params.num_cores_nhw
        ncores_w = self.impl.conv1.conv.sliding_window_op_params.num_cores_w
        ncores_h = self.impl.conv1.conv.sliding_window_op_params.num_cores_h

        logger.debug(f"ncores_nhw: {ncores_nhw}")
        logger.debug(f"ncores_w, ncores_h: {ncores_w}, {ncores_h}")

        ## layout is TILE for bfp8_b, otherwise ROW_MAJOR
        # layout = ttnn.TILE_LAYOUT if act_dtype == ttnn.bfloat8_b else ttnn.ROW_MAJOR_LAYOUT
        layout = ttnn.ROW_MAJOR_LAYOUT  ## always start with RM so that UTWH doesn't need to untilize unnecessarily

        ## construct sharding config
        grid_size = self.device.compute_with_storage_grid_size()
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
        input_tensor = ttnn.to_device(input_tensor, device=self.device, memory_config=sharded_mem_config)
        return input_tensor

    def __call__(self, input_tensor):
        output_tensor = self.impl.conv1(input_tensor)
        if self.batch_size == 20:
            output_tensor = ttnn.experimental.tensor.move_sharded(output_tensor)
        output_tensor = self.impl.maxpool(output_tensor)

        """
        1st bottleneck layer. all the blocks implemented by ttnn
        """
        output_tensor = ttnn.reshape(output_tensor, (1, 1, 56 * 56 * self.batch_size, 64))
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT, dtype=self.act_dtype)

        module = 1
        for bottleneck_block_parameters in list(self.impl.layer1.values()):
            logger.debug(f"parameters 1st block {bottleneck_block_parameters}")
            output_tensor = resnet_bottleneck_block(
                output_tensor, bottleneck_block_parameters, layer=1, module=module, device=self.device
            )
            module += 1

        """
        2nd bottleneck layer. 1st block implemented by torch rest by ttnn
        """
        module = 1
        for bottleneck_block_parameters in list(self.impl.layer2.values()):
            logger.debug(f"parameters 2nd block {bottleneck_block_parameters}")
            output_tensor = resnet_bottleneck_block(
                output_tensor, bottleneck_block_parameters, layer=2, device=self.device, module=module
            )
            module += 1

        """
        3rd bottleneck layer. 1st block implemented by torch rest by ttnn
        """
        module = 1
        for bottleneck_block_parameters in list(self.impl.layer3.values()):
            logger.debug(f"parameters 3rd block {bottleneck_block_parameters}")
            output_tensor = resnet_bottleneck_block(
                output_tensor, bottleneck_block_parameters, layer=3, module=module, device=self.device
            )
            module += 1

        """
        4th bottleneck layer. 1st block implemented by torch rest by ttnn
        """
        module = 1
        for bottleneck_block_parameters in list(self.impl.layer4.values()):
            logger.debug(f"parameters 4th block {bottleneck_block_parameters}")
            output_tensor = resnet_bottleneck_block(
                output_tensor, bottleneck_block_parameters, layer=4, module=module, device=self.device
            )
            module += 1

        # """
        # the last layers of the resnet
        # """
        # output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(
            output_tensor, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, use_multicore=True
        )
        output_tensor = ttnn.reshape(output_tensor, (self.batch_size, 1, 49, 2048))

        sharded_mem_config = ttnn.L1_MEMORY_CONFIG
        if self.batch_size == 20:
            grid_size = (8, 4)
            shard_grid = ttnn.experimental.tensor.CoreRangeSet(
                {
                    ttnn.experimental.tensor.CoreRange(
                        ttnn.experimental.tensor.CoreCoord(0, 0),
                        ttnn.experimental.tensor.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                    )
                }
            )
            shard_shape = (980, 64)
            shard_spec = ttnn.experimental.tensor.ShardSpec(
                shard_grid, shard_shape, ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR, False
            )
            sharded_mem_config = ttnn.types.MemoryConfig(
                ttnn.types.TensorMemoryLayout.WIDTH_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )
        output_tensor = ttnn.to_memory_config(output_tensor, sharded_mem_config)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT, memory_config=output_tensor.memory_config())
        output_tensor = ttnn.global_avg_pool2d(output_tensor, memory_config=output_tensor.memory_config())

        matmul_config = ttnn.experimental.operations.primary.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        out_shape = output_tensor.shape
        unpadded_shape_end = [
            out_shape[0] - 1,
            out_shape[1] - 1,
            0,
            out_shape[3] - 1,
        ]
        output_tensor = ttnn.experimental.tensor.untilize_with_unpadding(
            output_tensor, (0, 0, 0, 0), unpadded_shape_end, output_mem_config=sharded_mem_config
        )
        out_shape = output_tensor.shape
        output_tensor = output_tensor.reshape(1, out_shape[1], self.batch_size * out_shape[2], out_shape[3])
        out_shape = output_tensor.shape
        padded_shape = [
            out_shape[0],
            out_shape[1],
            _nearest_32(out_shape[2]),
            _nearest_32(out_shape[3]),
        ]
        output_tensor = ttnn.experimental.tensor.tilize_with_val_padding(
            output_tensor,
            padded_shape,
            [0, 0, 0, 0],
            0,
            output_mem_config=sharded_mem_config,
            output_dtype=self.act_dtype,
        )
        weight_shape = self.impl.fc.weight.get_legacy_shape()
        weight = self.impl.fc.weight.reshape(1, 1, weight_shape[-2], weight_shape[-1])
        bias_shape = self.impl.fc.bias.get_legacy_shape()
        bias = self.impl.fc.bias.reshape(1, 1, bias_shape[-2], bias_shape[-1])
        output_tensor = ttnn.experimental.operations.primary.matmul_1d(
            output_tensor,
            weight,
            bias=bias,
            program_config=matmul_config,
            output_mem_config=sharded_mem_config,
            output_dtype=self.act_dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
        out_shape = list(output_tensor.shape_without_padding())
        out_shape[-1] = 1000
        output_tensor = ttnn.experimental.tensor.untilize_with_unpadding(
            output_tensor, (0, 0, 0, 0), (out_shape[0] - 1, out_shape[1] - 1, out_shape[2] - 1, out_shape[3] - 1)
        )

        return output_tensor
