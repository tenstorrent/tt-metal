"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

from typing import Type, Union, Optional, List, Callable

import tt_lib
import torch
import torch.nn as nn
import math
from utils import conv3x3, conv1x1, fold_bn_to_conv, fold_bn_to_conv_weights_bias
from models.utility_functions import pad_by_zero, tt2torch_tensor
from tt_lib.utils import pad_weight

from tt_lib.fused_ops.average_pool import run_avg_pool_on_device_wrapper as TtAvgPool
from tt_lib.fused_ops.max_pool import run_max_pool_on_device_wrapper as TtMaxPool
from tt_lib.fused_ops.max_pool import compute_max_pool_shape
from tt_lib.fused_ops.linear import Linear as TtLinear
from tt_lib.fused_ops.softmax import softmax as TtSoftmax
from tt_lib.fused_ops.conv import resnet_conv as TtResnetConv
from models.utility_functions import is_conv_supported_on_device, _nearest_32
from tt_lib.fallback_ops import fallback_ops

def _nearest_y(x, y):
    return math.ceil(x / y) * y

def format_tensor(x, target_layout, device, output_mem_config, pad_value=0.0):
    if x.layout() == target_layout:
        return x
    if x.layout() == tt_lib.tensor.Layout.ROW_MAJOR and target_layout == tt_lib.tensor.Layout.TILE:
        x_padded_shape = tt_lib.tensor.pad_to_tile_shape(x.shape(), False, False, True, True)
        return tt_lib.tensor.format_input_tensor(x, device, x_padded_shape, pad_value, target_layout, output_mem_config)
    elif x.layout() == tt_lib.tensor.Layout.TILE and target_layout == tt_lib.tensor.Layout.ROW_MAJOR:
        return tt_lib.tensor.format_output_tensor(x, x.shape_without_padding(), device, target_layout, output_mem_config)
    else:
        assert False

# Local copy of unpad_from_zero to always set output to
def unpad_from_zero(x, desired_shape):
    if x.shape()[-1] == desired_shape[-1] and x.shape()[-2] == desired_shape[-2] :
        x = tt2torch_tensor(x)
    else:
        x = x.cpu()
        if(x.layout() != tt_lib.tensor.Layout.ROW_MAJOR):
            x = x.to(tt_lib.tensor.Layout.ROW_MAJOR)
        x = x.unpad((0, 0, 0, 0), (desired_shape[0] - 1, desired_shape[1] - 1, desired_shape[2] - 1, desired_shape[3] - 1) )
        x = x.to_torch().to(torch.float)
    return x

def compute_conv_output_shape(conv_params, x_shape):
    H = x_shape[1]
    W = x_shape[2]
    K, C, R, S, U, V, P_H, P_W, dilation, groups = [conv_params[i] for i in range(10)]
    OH = ((int) ((H - R + 2 * P_H) / U)) + 1
    OW = ((int) ((W - S + 2 * P_W) / V)) + 1
    return [x_shape[0],OH,OW,K]

# hardcoding matmul config for 1x1 convs
# key: mm act height, mm act width, mm weight width
hardcoded_matmul_config_conv = {
    (3136, 64, 64) : {"compute_with_storage_grid_size" : (2,2),
                            "in0_block_w" : 2,
                            "out_subblock_h" : 1,
                            "out_subblock_w": 1,
                            "per_core_M": 49,
                            "per_core_N": 1,
                        },

    (3136, 64, 256) : {"compute_with_storage_grid_size" : (4,2),
                            "in0_block_w" : 2,
                            "out_subblock_h" : 1,
                            "out_subblock_w": 1,
                            "per_core_M": 49,
                            "per_core_N": 2,
                        },
    (3136, 256, 64) : {"compute_with_storage_grid_size" : (2,7),
                    "in0_block_w" : 8,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 14,
                    "per_core_N": 1,
                },
    (3136, 256, 128) : {"compute_with_storage_grid_size" : (4,7),
                    "in0_block_w" : 8,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 14,
                    "per_core_N": 1,
                },
    (800, 128, 512) : {"compute_with_storage_grid_size" : (4,2),
                    "in0_block_w" : 4,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 13,
                    "per_core_N": 4,
                },
    (800, 512, 128) : {"compute_with_storage_grid_size" : (4,4),
                    "in0_block_w" : 16,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 7,
                    "per_core_N": 1,
                },
    (800, 512, 256) : {"compute_with_storage_grid_size" : (8,4),
                    "in0_block_w" : 16,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 7,
                    "per_core_N": 1,
                },
    (224, 256, 1024) : {"compute_with_storage_grid_size" : (8,7),
                    "in0_block_w" : 8,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 4,
                },
    (224, 1024, 256) : {"compute_with_storage_grid_size" : (8,7),
                    "in0_block_w" : 32,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 1,
                },
    (224, 1024, 512) : {"compute_with_storage_grid_size" : (8,7),
                    "in0_block_w" : 32,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 2,
                },
    (64, 512, 2048) : {"compute_with_storage_grid_size" : (8,2),
                    "in0_block_w" : 16,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 8,
                },
    (64, 2048, 512) : {"compute_with_storage_grid_size" : (8,2),
                    "in0_block_w" : 64,
                    "out_subblock_h" : 1,
                    "out_subblock_w": 1,
                    "per_core_M": 1,
                    "per_core_N": 2,
                },
}

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device = None,
        state_dict = None,
        base_address = None,
        fold_batchnorm = False,
        downsample_conv_on_tt = None,
        norm_layer_after_downsample_conv_on_tt = None,
        downsample_params = [],
        storage_in_dram=True,
        input_shape = []
    ) -> None:
        super().__init__()
        self.device = device
        self.state_dict = state_dict
        self.base_address = base_address
        self.fold_batchnorm = fold_batchnorm
        self.downsample_conv_on_tt = downsample_conv_on_tt
        self.norm_layer_after_downsample_conv_on_tt = norm_layer_after_downsample_conv_on_tt
        self.downsample_params = downsample_params
        self.storage_in_dram = storage_in_dram
        if self.storage_in_dram:
            self.memory_config = tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.DRAM)
        else:
            self.memory_config = tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.L1)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        conv1_weight = state_dict[f"{base_address}.conv1.weight"]
        conv1_bias = None

        self.bn1 = norm_layer(width)
        self.bn1.weight = nn.Parameter(state_dict[f"{self.base_address}.bn1.weight"])
        self.bn1.bias = nn.Parameter(state_dict[f"{self.base_address}.bn1.bias"])
        self.bn1.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_mean"])
        self.bn1.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn1.running_var"])
        self.bn1.num_batches_tracked = nn.Parameter(state_dict[f"{self.base_address}.bn1.num_batches_tracked"], requires_grad=False)
        self.bn1.eval()

        conv2_weight = state_dict[f"{base_address}.conv2.weight"]
        conv2_bias = None

        self.bn2 = norm_layer(width)
        self.bn2.weight = nn.Parameter(state_dict[f"{self.base_address}.bn2.weight"])
        self.bn2.bias = nn.Parameter(state_dict[f"{self.base_address}.bn2.bias"])
        self.bn2.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_mean"])
        self.bn2.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn2.running_var"])
        self.bn2.num_batches_tracked = nn.Parameter(state_dict[f"{self.base_address}.bn2.num_batches_tracked"], requires_grad=False)
        self.bn2.eval()

        conv3_weight = state_dict[f"{base_address}.conv3.weight"]
        conv3_bias = None

        self.bn3 = norm_layer(planes * self.expansion)
        self.bn3.weight = nn.Parameter(state_dict[f"{self.base_address}.bn3.weight"])
        self.bn3.bias = nn.Parameter(state_dict[f"{self.base_address}.bn3.bias"])
        self.bn3.running_mean = nn.Parameter(state_dict[f"{self.base_address}.bn3.running_mean"])
        self.bn3.running_var = nn.Parameter(state_dict[f"{self.base_address}.bn3.running_var"])
        self.bn3.num_batches_tracked = nn.Parameter(state_dict[f"{self.base_address}.bn3.num_batches_tracked"], requires_grad=False)
        self.bn3.eval()

        self.relu = tt_lib.tensor.relu_without_autoformat
        self.downsample = downsample
        self.stride = stride

        if self.fold_batchnorm:
            conv1_weight, conv1_bias = fold_bn_to_conv_weights_bias(conv1_weight, self.bn1)
            conv2_weight, conv2_bias = fold_bn_to_conv_weights_bias(conv2_weight, self.bn2)
            conv3_weight, conv3_bias = fold_bn_to_conv_weights_bias(conv3_weight, self.bn3)
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()
            self.bn3 = nn.Identity()

        self.conv1_params = [width, inplanes, 1, 1, 1, 1, 0, 0, dilation, groups]
        self.conv1_output_shape = compute_conv_output_shape(self.conv1_params, input_shape)
        conv1_as_mm_padded_act_height = _nearest_32(self.conv1_output_shape[1] * self.conv1_output_shape[2])
        matmul_config = None
        if (conv1_as_mm_padded_act_height, inplanes, width) in hardcoded_matmul_config_conv:
            #print("Setting matmul config for 1x1 conv (first conv in module)")
            matmul_config = hardcoded_matmul_config_conv[(conv1_as_mm_padded_act_height, inplanes, width)]
        if is_conv_supported_on_device(self.conv1_params):
            # 1x1 conv with stride 1 padding 0 is run using regular matmul
            self.conv1 = TtResnetConv(conv1_weight.reshape(-1).tolist(), self.conv1_params, self.device, [1, 1], [1, 1], [1, 1], conv1_bias.tolist() if conv1_bias is not None else None, matmul_config=matmul_config, fuse_relu=True)
        else:
            self.conv1 = fallback_ops.Conv2d(conv1_weight, conv1_bias, inplanes, width, kernel_size=1, stride=1, padding=0)

        # With single buffered input CB, these shapes work -
        # hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_conv2 = {
        #     (3136, 64) : [256, 64, 128, 64] ,
        #     (800, 128) : [256, 128, 128, 64] ,
        #     (224, 256) : [128, 128, 128, 64],
        #     (64, 512) : [64, 128, 64, 128] ,
        # }

        # With double buffered input CB, these shapes work -
        # hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_conv2 = {
        #     (3136, 64) : [256, 64, 128, 64] ,
        #     (800, 128) : [128, 128, 128, 64] ,
        #     (224, 256) : [64, 128, 64, 128],
        #     (64, 512) : [32, 64, 32, 64] ,
        # }
        hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_conv2 = {
            (3136, 64) : [128, 64, 128, 64] ,
            (800, 128) : [128, 128, 128, 64] ,
            (224, 256) : [64, 128, 64, 128],
            (64, 512) : [32, 64, 32, 64] ,
        }
        self.conv2_params = [width, width, 3, 3, stride, stride, 1, 1, dilation, groups]
        self.conv2_output_shape = compute_conv_output_shape(self.conv2_params, self.conv1_output_shape)
        conv2_output_padded_face_size = _nearest_32(self.conv2_output_shape[1] * self.conv2_output_shape[2])
        assert (conv2_output_padded_face_size, width) in hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_conv2
        [act_block_h_datums, weight_block_w_datums, out_subblock_h_datums, out_subblock_w_datums] = hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_conv2[(conv2_output_padded_face_size, width)]
        if is_conv_supported_on_device(self.conv2_params):
            self.conv2 = TtResnetConv(conv2_weight.reshape(-1).tolist(), self.conv2_params, self.device, [act_block_h_datums, width*3], [width*3, weight_block_w_datums], [out_subblock_h_datums, out_subblock_w_datums], conv2_bias.tolist() if conv2_bias is not None else None)
        else:
            self.conv2 = fallback_ops.Conv2d(conv2_weight, conv2_bias, width, width, kernel_size=3, stride=1, padding=1)

        self.conv3_params = [planes * self.expansion, width, 1, 1, 1, 1, 0, 0, dilation, groups]
        self.conv3_output_shape = compute_conv_output_shape(self.conv3_params, self.conv2_output_shape)
        conv3_as_mm_padded_act_height = _nearest_32(self.conv3_output_shape[1] * self.conv3_output_shape[2])
        matmul_config = None
        if (conv3_as_mm_padded_act_height, width, planes * self.expansion) in hardcoded_matmul_config_conv:
            #print("Setting matmul config for 1x1 conv (third conv in module)")
            matmul_config = hardcoded_matmul_config_conv[(conv3_as_mm_padded_act_height, width, planes * self.expansion)]
        if is_conv_supported_on_device(self.conv3_params):
            # 1x1 conv with stride 1 padding 0 is run using regular matmul
            self.conv3 = TtResnetConv(conv3_weight.reshape(-1).tolist(), self.conv3_params, self.device, [1, 1], [1, 1], [1, 1], conv3_bias.tolist() if conv3_bias is not None else None, matmul_config=matmul_config)
        else:
            self.conv3 = fallback_ops.Conv2d(conv3_weight, conv3_bias, width, planes * self.expansion, kernel_size=1, stride=1, padding=0)
        self.conv3_output_shape = compute_conv_output_shape(self.conv3_params, self.conv2_output_shape)

    def run_forward(self, x: torch.Tensor, x_actual_shape=[]):
        identity = x
        # conv1 is 1x1 conv
        #print("Running conv1")
        out = self.conv1(x)
        # Relu after conv1 is fused with the 1x1 conv (matmul)
        #out = self.relu(out, self.memory_config)
        out = format_tensor(out, tt_lib.tensor.Layout.ROW_MAJOR, self.device, self.memory_config)
        out = out.reshape(x_actual_shape[0], x_actual_shape[1], x_actual_shape[2], out.shape()[3])
        saved_shape = out.shape()
        #print("Running conv1")
        out = self.conv2(out)
        conv_2_output_shape = compute_conv_output_shape(self.conv2_params, saved_shape)
        out = self.relu(out, self.memory_config)
        # conv3 is 1x1 conv
        #print("Running conv1")
        out = self.conv3(out)

        if self.downsample_conv_on_tt is not None:
            if(self.downsample_params[2] != 1 or self.downsample_params[4] != 1 or self.downsample_params[6] != 0):
                x = format_tensor(x, tt_lib.tensor.Layout.ROW_MAJOR, self.device, self.memory_config)
                x = x.reshape(x_actual_shape[0], x_actual_shape[1], x_actual_shape[2], x_actual_shape[3])
            #print("Running downsample")
            identity = self.downsample_conv_on_tt(x)
            assert self.norm_layer_after_downsample_conv_on_tt is not None
            if not self.fold_batchnorm:
                identity = self.norm_layer_after_downsample_conv_on_tt(identity)
        elif self.downsample is not None:
            identity = self.downsample(x)
        out = tt_lib.tensor.add_without_autoformat(out, identity, output_mem_config=self.memory_config)
        out = self.relu(out, self.memory_config)
        out_actual_shape = [conv_2_output_shape[0], conv_2_output_shape[1], conv_2_output_shape[2], out.shape()[3]]
        return out, out_actual_shape


class ResNet(nn.Module):
    def __init__(
        self,
        block: Bottleneck,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        device = None,
        state_dict = None,
        base_address = None,
        fold_batchnorm = False,
        storage_in_dram = True,
        conv_input_face_shape_hw = [224,224]
    ) -> None:
        super().__init__()
        self.device = device
        self.base_address_with_dot = base_address # this is root layer, no dot is needed
        self.state_dict = state_dict
        self.fold_batchnorm = fold_batchnorm
        self.storage_in_dram = storage_in_dram
        self.conv_input_face_shape_hw = conv_input_face_shape_hw
        if self.storage_in_dram:
            self.memory_config = tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.DRAM)
        else:
            self.memory_config = tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.L1)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        conv1_weight = state_dict[f"{self.base_address_with_dot}conv1.weight"]
        conv1_bias = None

        self.bn1 = norm_layer(self.inplanes) # batch norm
        self.bn1.weight = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.weight"])
        self.bn1.bias = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.bias"])
        self.bn1.running_mean = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.running_mean"])
        self.bn1.running_var = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.running_var"])
        self.bn1.num_batches_tracked = nn.Parameter(state_dict[f"{self.base_address_with_dot}bn1.num_batches_tracked"], requires_grad=False)
        self.bn1.eval()

        if self.fold_batchnorm:
            conv1_weight, conv1_bias = fold_bn_to_conv_weights_bias(conv1_weight, self.bn1)
            self.bn1 = nn.Identity()

        self.conv1_params = [self.inplanes, 3, 7, 7, 2, 2, 3, 3, 1, groups]
        if is_conv_supported_on_device(self.conv1_params):
            self.conv1 = TtResnetConv(conv1_weight.reshape(-1).tolist(), self.conv1_params, self.device, [128, 128], [128, 64], [128, 64], conv1_bias.tolist() if conv1_bias is not None else None, 8, True, enable_fused_bias=False)
        else:
            self.conv1 = fallback_ops.Conv2d(conv1_weight, conv1_bias, 3, self.inplanes, kernel_size=7, stride=2, padding=3)
        self.conv1_output_shape = compute_conv_output_shape(self.conv1_params, [1, self.conv_input_face_shape_hw[0], self.conv_input_face_shape_hw[1], self.inplanes])
        self.relu = tt_lib.tensor.relu_without_autoformat
        # self.maxpool = fallback_ops.MaxPool2d(kernel_size=3, stride=2, padding=1, channels_last=True, reshape_2d=True)
        self.maxpool = TtMaxPool(self.device, kernel_size=3, stride=2, padding=1, output_mem_config=self.memory_config, nblocks=8, channels_last=True, reshape_2d=True)
        self.maxpool_output_shape = compute_max_pool_shape(3, 2, 1, self.conv1_output_shape)
        self.layer1, self.layer1_output_shape = self._make_layer(block, 64, layers[0], name="layer1", state_dict=state_dict, layer_input_shape=self.maxpool_output_shape)
        self.layer2, self.layer2_output_shape = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], name="layer2", state_dict=state_dict, layer_input_shape=self.layer1_output_shape)
        self.layer3, self.layer3_output_shape = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], name="layer3", state_dict=state_dict, layer_input_shape=self.layer2_output_shape)
        self.layer4, self.layer4_output_shape = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], name="layer4", state_dict=state_dict, layer_input_shape=self.layer3_output_shape)
        self.avgpool = TtAvgPool(self.device)

        fc_weight = pad_weight(state_dict[f"{self.base_address_with_dot}fc.weight"])
        fc_weight = tt_lib.tensor.Tensor(fc_weight.reshape(-1).tolist(), fc_weight.shape, tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.ROW_MAJOR).to(tt_lib.tensor.Layout.TILE)
        fc_bias = pad_weight(state_dict[f"{self.base_address_with_dot}fc.bias"])
        fc_bias = tt_lib.tensor.Tensor(fc_bias.reshape(-1).tolist(), fc_bias.shape, tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.ROW_MAJOR).to(tt_lib.tensor.Layout.TILE)

        self.fc = TtLinear(512 * block.expansion, 1024, fc_weight, fc_bias, self.device) # num_classes = 1000
        # self.fc = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(
        self,
        block: Bottleneck,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        name: str = None,
        state_dict = None,
        layer_input_shape = []
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        self.downsample_conv_on_tt = None
        self.norm_layer_after_downsample_conv_on_tt = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            nl = norm_layer(planes * block.expansion)
            nl.weight = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.weight"])
            nl.bias = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.bias"])
            nl.running_mean = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.running_mean"])
            nl.running_var = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.running_var"])
            nl.num_batches_tracked = nn.Parameter(state_dict[f"{self.base_address_with_dot}{name}.0.downsample.1.num_batches_tracked"], requires_grad=False)
            nl.eval()
            downsample_conv_weight = state_dict[f"{self.base_address_with_dot}{name}.0.downsample.0.weight"]
            downsample_conv_bias = None

            if self.fold_batchnorm:
                downsample_conv_weight, downsample_conv_bias = fold_bn_to_conv_weights_bias(downsample_conv_weight, nl)
                nl = nn.Identity()

            # With single buffered input CB, these shapes work -
            # hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_downsample_conv = {
            #     (3136, 256) : [128, 128, 128, 64] ,
            #     (800, 512) : [128, 128, 128, 64] ,
            #     (224, 1024) : [64, 128, 64, 64],
            #     (64, 2048) : [64, 128, 64, 64] ,
            # }

            # With double buffered input CB, these shapes work -
            hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_downsample_conv = {
                (3136, 256) : [128, 64, 128, 64] ,
                (800, 512) : [128, 64, 128, 64] ,
                (224, 1024) : [64, 128, 64, 64],
                (64, 2048) : [64, 128, 64, 64] ,
            }
            downsample_output_channels = planes * block.expansion
            self.downsample_params = [downsample_output_channels, self.inplanes, 1, 1, stride, stride, 0, 0, self.dilation, 1]
            self.downsample_conv_output_shape = compute_conv_output_shape(self.downsample_params, layer_input_shape)
            downsample_output_padded_face_size = _nearest_32(self.downsample_conv_output_shape[1] * self.downsample_conv_output_shape[2])
            assert (downsample_output_padded_face_size, downsample_output_channels) in hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_downsample_conv
            [act_block_h_datums, weight_block_w_datums, out_subblock_h_datums, out_subblock_w_datums] = hardcoded_act_blk_h_weight_blk_w_out_subblk_h_out_subblk_w_for_downsample_conv[(downsample_output_padded_face_size, downsample_output_channels)]

            is_downsample_1x1_conv = stride == 1
            matmul_config = None
            if (is_downsample_1x1_conv and (downsample_output_padded_face_size, self.inplanes, downsample_output_channels) in hardcoded_matmul_config_conv):
                #print("Setting matmul config for 1x1 conv (downsample stride 1 conv in module)")
                matmul_config = hardcoded_matmul_config_conv[(downsample_output_padded_face_size,  self.inplanes, downsample_output_channels)]

            if is_conv_supported_on_device(self.downsample_params):
                self.downsample_conv_on_tt = TtResnetConv(downsample_conv_weight.reshape(-1).tolist(), self.downsample_params, self.device, [act_block_h_datums, self.inplanes], [self.inplanes, weight_block_w_datums], [out_subblock_h_datums, out_subblock_w_datums], downsample_conv_bias.tolist() if downsample_conv_bias is not None else None, matmul_config=matmul_config)
                self.norm_layer_after_downsample_conv_on_tt = nl
            else:
                downsample_conv = fallback_ops.Conv2d(downsample_conv_weight, downsample_conv_bias, self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, padding=0)
                downsample = nn.Sequential(
                    downsample_conv,
                    nl,
                )


        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                device=self.device,
                state_dict=self.state_dict,
                base_address=f"{self.base_address_with_dot}{name}.0",
                fold_batchnorm=self.fold_batchnorm,
                downsample_conv_on_tt=self.downsample_conv_on_tt,
                norm_layer_after_downsample_conv_on_tt=self.norm_layer_after_downsample_conv_on_tt,
                downsample_params=self.downsample_params,
                storage_in_dram=self.storage_in_dram,
                input_shape=layer_input_shape
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            previous_layer = layers[-1]
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    device=self.device,
                    state_dict=self.state_dict,
                    base_address=f"{self.base_address_with_dot}{name}.{_}",
                    fold_batchnorm=self.fold_batchnorm,
                    storage_in_dram=self.storage_in_dram,
                    input_shape=previous_layer.conv3_output_shape
                )
            )
        last_layer_shape = layers[-1].conv3_output_shape
        return layers, last_layer_shape

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        #assert x.shape[3] == 224 and x.shape[2] == 224
        x = torch.permute(x, (0, 2, 3, 1))
        x = tt_lib.tensor.Tensor(
                x.reshape(-1).tolist(),
                x.shape,
                tt_lib.tensor.DataType.BFLOAT16,
                tt_lib.tensor.Layout.ROW_MAJOR)
        # Pre-pad input shape
        act_shape_height_width_channel_padded = [x.shape()[0], x.shape()[1] + 6, x.shape()[2] + 7, _nearest_y(x.shape()[3], 16)]
        x = x.pad(act_shape_height_width_channel_padded, (0, 3, 3, 0), 0)

        x = x.to(self.device, self.memory_config)
        saved_shape = compute_conv_output_shape(self.conv1_params, [1, 224, 224, 16])
        x = self.conv1(x)
        #x = x.reshape(1, 1, x.shape()[0]*x.shape()[1]*x.shape()[2], x.shape()[3]);
        x = self.relu(x, self.memory_config)
        x = format_tensor(x, tt_lib.tensor.Layout.ROW_MAJOR, self.device, self.memory_config)
        x = x.reshape(saved_shape[0], saved_shape[1], saved_shape[2], saved_shape[3])
        x = self.maxpool(x)
        x = format_tensor(x, tt_lib.tensor.Layout.TILE, self.device, self.memory_config)
        saved_shape = compute_max_pool_shape(3, 2, 1, saved_shape)

        for layer in self.layer1:
            x, saved_shape = layer.run_forward(x, saved_shape)
        for layer in self.layer2:
            x, saved_shape = layer.run_forward(x, saved_shape)
        for layer in self.layer3:
            x, saved_shape = layer.run_forward(x, saved_shape)
        for i,layer in enumerate(self.layer4):
            x, saved_shape = layer.run_forward(x, saved_shape)
        x = format_tensor(x, tt_lib.tensor.Layout.ROW_MAJOR, self.device, self.memory_config)
        x = format_tensor(x, tt_lib.tensor.Layout.TILE, self.device, self.memory_config)
        x = self.avgpool(x)
        x = self.fc(x)
        desired_shape = [x.shape()[0], x.shape()[1], 1, 1000]
        x = unpad_from_zero(x, desired_shape)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)
