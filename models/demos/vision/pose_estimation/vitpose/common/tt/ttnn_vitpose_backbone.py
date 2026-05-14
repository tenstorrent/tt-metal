# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose_embeddings import (
    VitPosePatchEmbeddings,
    vitpose_embeddings,
)
from models.demos.vision.pose_estimation.vitpose.common.tt.ttnn_vitpose_encoder import vitpose_encoder


class VitPoseBackbone:
    def __init__(self, embedding_params, layer_parameters, layernorm_params, device, *, batch_size=1, num_heads=12):
        self.patch_embed = VitPosePatchEmbeddings(embedding_params, device, batch_size=batch_size)
        self.pos_patches = ttnn.to_device(embedding_params["pos_patches"], device)
        self.pos_cls = ttnn.to_device(embedding_params["pos_cls"], device)
        self.layer_parameters = layer_parameters
        self.layernorm_weight = ttnn.to_device(layernorm_params["weight"], device)
        self.layernorm_bias = ttnn.to_device(layernorm_params["bias"], device)
        self.num_heads = num_heads
        self.device = device
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, pixel_values):
        patch_emb = self.patch_embed(pixel_values)
        embeddings = vitpose_embeddings(patch_emb, pos_patches=self.pos_patches, pos_cls=self.pos_cls)
        output = vitpose_encoder(
            embeddings,
            layer_parameters=self.layer_parameters,
            num_heads=self.num_heads,
            compute_kernel_config=self.compute_kernel_config,
        )
        output = ttnn.layer_norm(
            output,
            weight=self.layernorm_weight,
            bias=self.layernorm_bias,
            compute_kernel_config=self.compute_kernel_config,
        )
        return output
