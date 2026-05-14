# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose_backbone import VitPoseBackbone
from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose_decoder import (
    VitPoseSimpleDecoder,
    preprocess_decoder_parameters,
)
from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose_embeddings import (
    preprocess_embedding_parameters,
)
from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose_layer import preprocess_layer_parameters


class VitPose:
    """
    Full ViTPose-B model: backbone + SimpleDecoder.
    """

    def __init__(self, state_dict, device, *, batch_size=1, num_layers=12, num_heads=12, dtype=ttnn.bfloat16):
        embedding_params = preprocess_embedding_parameters(state_dict, dtype=dtype)

        layer_params_list = []
        for i in range(num_layers):
            lp = preprocess_layer_parameters(state_dict, i, dtype=dtype)
            lp = {k: ttnn.to_device(v, device) for k, v in lp.items()}
            layer_params_list.append(lp)

        layernorm_params = {
            "weight": ttnn.from_torch(state_dict["backbone.layernorm.weight"].reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT),
            "bias": ttnn.from_torch(state_dict["backbone.layernorm.bias"].reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT),
        }

        self.backbone = VitPoseBackbone(
            embedding_params, layer_params_list, layernorm_params, device, batch_size=batch_size, num_heads=num_heads
        )

        decoder_params = preprocess_decoder_parameters(state_dict, dtype=dtype)
        self.decoder = VitPoseSimpleDecoder(decoder_params, device, batch_size=batch_size)
        self.device = device

    def __call__(self, pixel_values):
        """
        Args:
            pixel_values: (batch, 256, 192, 3) NHWC ROW_MAJOR on device

        Returns:
            heatmaps tensor on device
        """
        backbone_output = self.backbone(pixel_values)
        heatmaps = self.decoder(backbone_output)
        return heatmaps

    @staticmethod
    def prepare_input(pixel_values_nchw, device):
        """
        Convert NCHW PyTorch tensor to NHWC ttnn tensor on device.

        Args:
            pixel_values_nchw: (batch, 3, 256, 192) torch tensor
            device: ttnn device

        Returns:
            ttnn tensor (batch, 256, 192, 3) NHWC ROW_MAJOR on device
        """
        pixel_values_nhwc = pixel_values_nchw.permute(0, 2, 3, 1).contiguous()
        tt_input = ttnn.from_torch(pixel_values_nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        tt_input = ttnn.to_device(tt_input, device)
        return tt_input
