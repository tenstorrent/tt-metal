# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F

# from models.experimental.oft.reference.resnet import BasicBlock
from models.experimental.oft.tt.model_preprocessing import create_OFT_model_parameters_resnet
from tests.ttnn.unit_tests.base_functionality.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
from tests.ttnn.utils_for_testing import assert_with_pcc

from loguru import logger

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False

# helper funcction for easier enabling/disabling synchronization after each op
SYNC_DEVICE_GLOBAL = False


def enable_sync_device(enable):
    global SYNC_DEVICE_GLOBAL
    SYNC_DEVICE_GLOBAL = enable


def synchronize_device(device, op_name):
    if SYNC_DEVICE_GLOBAL:
        ttnn.synchronize_device(device)
        logger.debug(f"Synchronizing device after {op_name}")


# torch reference basic block
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dtype=torch.float32):
        super(BasicBlock, self).__init__()

        # Update conv functions to pass dtype
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, dtype=dtype)
        self.bn1 = nn.GroupNorm(16, planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, dtype=dtype)
        self.bn2 = nn.GroupNorm(16, planes)

        self.dtype = dtype
        # Convert all parameters to the specified dtype using PyTorch's to() method
        self.to(dtype)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out, inplace=True)

        return out


class Conv:
    def __init__(
        self,
        parameters,
        conv_pt,
        *,
        stride=1,
        padding=1,
        act_block_h=32,
        reshard=False,
        deallocate=False,
        height_sharding=None,
        activation=None,
        width_sharding=False,
        block_sharding=False,
        dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        output_layout=ttnn.TILE_LAYOUT,
        is_sliced=False,
    ) -> None:
        self.weights = parameters.weight
        # logger.debug(f"Conv weights: {self.weights.shape}")
        # logger.debug(f"Conv parameters: {self.weights}")
        self.conv_pt = conv_pt
        # logger.debug(f"Conv: {self.conv_pt}")

        # Automatically detect if bias is present in parameters
        try:
            self.has_bias = hasattr(parameters, "bias") and parameters.bias is not None
        except (KeyError, AttributeError):
            self.has_bias = False

        if self.has_bias:
            logger.debug(f"Conv: bias found in parameters")

        # handle comparison mode that requires bias
        if ttnn.CONFIG.enable_comparison_mode:
            if self.has_bias:
                try:
                    self.bias = parameters.bias
                except (KeyError, AttributeError):
                    self.bias = None
            else:
                # Create bias tensor with proper shape for TTNN conv2d
                bias_tensor = torch.zeros(conv_pt.out_channels)
                self.bias = bias_tensor.view(1, 1, 1, -1)
                # Convert bias to ttnn tensor
                self.bias = ttnn.from_torch(self.bias, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT)
            # In comparison mode, we always have bias (either real or zero)
            self.has_bias = True
        else:
            if self.has_bias:
                try:
                    self.bias = parameters.bias
                except (KeyError, AttributeError):
                    self.bias = None

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.stride = conv_pt.stride  # stride
        self.padding = conv_pt.padding  # padding
        self.out_channels = conv_pt.out_channels
        # logger.debug(f"Conv out channels: {self.out_channels}")
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.dtype = dtype
        self.weights_dtype = weights_dtype
        self.output_layout = output_layout

        if width_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        elif height_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        elif block_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        else:
            self.shard_layout = None

        self.deallocate = deallocate
        self.activation = activation
        self.is_sliced = is_sliced
        self.slice_config = ttnn.Conv2dL1FullSliceConfig
        if is_sliced:
            self.slice_config = ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dDRAMSliceHeight,
                num_slices=2,
            )

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape if self.has_bias else ''} {self.kernel_size}"

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=self.weights_dtype,
            shard_layout=self.shard_layout,
            deallocate_activation=self.deallocate,
            activation=self.activation,
            reshard_if_not_optimal=True,
            output_layout=self.output_layout,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            # TODO(mbezulj): explore fidelity/fp32 settings. affects frontend, latents, scores.
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        [output_tensor, [out_h, out_w], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias if self.has_bias else None,
            in_channels=self.conv_pt.in_channels,
            out_channels=self.conv_pt.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=self.conv_pt.batch_size,
            input_height=self.conv_pt.input_height,
            input_width=self.conv_pt.input_width,
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            slice_config=self.slice_config,
        )
        # logger.debug(f"Output tensor shape: {output_tensor.shape}, out_h: {out_h}, out_w: {out_w}")
        # logger.debug(
        #     f"Output tensor dtype: {output_tensor.dtype}, layout: {output_tensor.layout}, memory config: {output_tensor.memory_config}"
        # )
        return output_tensor, out_h, out_w


def _nearest_32_per_core(x, core):
    return math.ceil(x / core / 32) * 32 * core


class GroupNormDRAM:
    def __init__(self, parameters, num_groups, channels, eps=1e-5, dtype=ttnn.bfloat16, is_sliced=False):
        self.weight = parameters.weight
        self.bias = parameters.bias
        self.num_groups = num_groups
        self.channels = channels
        self.eps = eps
        self.dtype = dtype
        self.is_sliced = is_sliced
        self.num_splited_groups = num_groups
        self.num_splited_channels = channels

    def __call__(self, device, input_tensor, H, W, shard="HS", num_splits=1):
        compute_grid = device.compute_with_storage_grid_size()
        grid_x, grid_y = compute_grid.x, compute_grid.y
        logger.debug(f"DRAM {grid_x=}, {grid_y=}, {shard=}, {num_splits=} {self.is_sliced=}")
        grid_x = 4
        grid_size = ttnn.CoreGrid(y=grid_y, x=grid_x)

        # torch input tensor
        unpadded_shape = input_tensor.shape
        out_shape = [
            unpadded_shape[0],
            unpadded_shape[1],
            _nearest_32_per_core(unpadded_shape[2], grid_x),
            _nearest_32_per_core(unpadded_shape[3], grid_y),
        ]
        # input_tensor_tilized2 = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT) #if you enable this line it will pass without hang
        input_tensor_tilized = ttnn.tilize_with_val_padding(
            input_tensor, output_tensor_shape=out_shape, pad_value=0, use_multicore=True
        )
        logger.debug(
            f"input_tensor_tilized shape: {input_tensor_tilized.shape} padded shape: {input_tensor_tilized.padded_shape}"
        )
        logger.debug(f"ITT {input_tensor_tilized}\n")
        [gamma_t, beta_t], input_mask_tensor = ttnn.dram_group_norm_params_from_torch(
            [self.weight, self.bias], self.channels, self.num_groups, device, core_grid=grid_size, return_mask=True
        )
        logger.debug(f"unpadded_shape: {unpadded_shape} out_shape: {out_shape}")

        # groupnorm
        logger.debug(f"DRAM {grid_size=}")
        output_tensor = ttnn.group_norm(
            input_tensor_tilized,
            num_groups=self.num_groups,
            input_mask=input_mask_tensor,
            weight=gamma_t,
            bias=beta_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
            core_grid=grid_size,
            inplace=False,
            num_out_blocks=num_splits,
            epsilon=1e-5,
        )
        return output_tensor


# ttnn basic block implementation
class TTBasicBlock:
    expansion = 1

    def __init__(self, device, parameters, conv_pt, inplanes, planes, stride=1, scale=1, is_sliced=False):
        self.is_sliced = is_sliced
        logger.debug(f"TTBasicBlock: {inplanes=}, {planes=}, {stride=}, {is_sliced=}")
        self.conv1 = Conv(
            parameters.conv1, conv_pt.conv1, stride=stride, output_layout=ttnn.ROW_MAJOR_LAYOUT, is_sliced=is_sliced
        )
        self.bn1 = GroupNormDRAM(parameters.bn1, num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat16)
        self.conv2 = Conv(
            parameters.conv2, conv_pt.conv2, stride=stride, output_layout=ttnn.ROW_MAJOR_LAYOUT, is_sliced=is_sliced
        )
        self.bn2 = GroupNormDRAM(parameters.bn2, num_groups=16, channels=planes, eps=1e-5, dtype=ttnn.bfloat16)

    def forward(self, device, x, gn_shard="HS", num_splits=1):
        if use_signpost:
            signpost(header="TTBasicBlock forward started")
        out, out_h, out_w = self.conv1(device, x)
        synchronize_device(device, "conv1")
        logger.debug(f"FORWARD X Input shape: {x.shape}, dtype: {x.dtype}, layout: {x.layout}")

        out = self.bn1(device, out, out_h, out_w, shard=gn_shard, num_splits=num_splits)
        logger.debug(f"BN1 output shape: {out.shape}")
        synchronize_device(device, "bn1")
        ttnn.relu(out, output_tensor=out)
        synchronize_device(device, "relu1")

        out, out_h, out_w = self.conv2(device, out)
        synchronize_device(device, "conv2")
        logger.debug(f"Conv2 output shape: {out.shape}")
        out = self.bn2(device, out, out_h, out_w, shard=gn_shard, num_splits=num_splits)
        synchronize_device(device, "bn2")
        logger.debug(f"BN2 output shape: {out.shape}")

        out += x
        synchronize_device(device, "add")
        out = ttnn.relu(out)
        synchronize_device(device, "relu2")
        if use_signpost:
            signpost(header="TTBasicBlock forward finished")
        return out


@pytest.mark.parametrize(
    "n, in_ch, out_ch, h, w, stride, sharding, is_sliced",
    [(1, 256, 256, 159, 159, 1, "HS", True)],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 8 * 1024}], indirect=True)
def test_tt_topdownblock_with_8_basicblocks(device, n, in_ch, out_ch, h, w, stride, sharding, is_sliced):
    skip_if_not_blackhole_20_cores(device)
    # device.disable_and_clear_program_cache()  # test hangs without this line
    enable_sync_device(False)  # True -> if you want to enable synchronization after each op
    torch.manual_seed(42)
    input_tensor = torch.randn(n, in_ch, h, w)
    # Create 2 BasicBlock modules from oft
    block_number = 2
    blocks = []
    params_list = []
    for i in range(block_number):
        block = BasicBlock(inplanes=in_ch, planes=out_ch, stride=stride)
        blocks.append(block)
        params = create_OFT_model_parameters_resnet(block, input_tensor, device)
        params_list.append(params)
    # Reference output using PyTorch blocks sequentially
    out_ref = input_tensor
    for block in blocks:
        out_ref = block(out_ref)
    # Create 8 TTBasicBlock modules
    tt_blocks = [
        TTBasicBlock(
            device,
            params_list[i],
            params_list[i].layer_args,
            inplanes=in_ch,
            planes=out_ch,
            stride=stride,
            is_sliced=is_sliced,
        )
        for i in range(block_number)
    ]
    # Prepare TTNN input
    n, c, h, w = input_tensor.shape
    x_for_ttnn = input_tensor.permute(0, 2, 3, 1).view(1, 1, n * h * w, c)
    ttnn_x = ttnn.from_torch(x_for_ttnn, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    # Forward through TTBasicBlocks sequentially
    ttnn_out = ttnn_x
    for block in tt_blocks:
        ttnn_out = block.forward(device, ttnn_out, gn_shard=sharding, num_splits=2)
    ttnn_out = ttnn.to_torch(ttnn_out)
    # Compare output
    B, C, H, W = out_ref.shape
    out_ref = out_ref.permute(0, 2, 3, 1).reshape(1, 1, B * H * W, C)
    pcc, message = assert_with_pcc(ttnn_out, out_ref, 0.99)
    logger.info(f"PCC for topdown block with 8 BasicBlocks: {pcc}, Message: {message}")
