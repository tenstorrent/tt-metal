# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI U.S. Corp., for the TTNN port. Reference code © Meta Platforms, Inc. (Apache-2.0).
# SPDX-License-Identifier: Apache-2.0
import math

import torch
import torch.nn.functional as F

import ttnn


class TtPromptEncoder:
    def __init__(self, parameters, device):
        self.device = device
        self.mask_input_size = (256, 256)
        self.mask_embed = parameters.mask_embed
        self._mask_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.gaussian_matrix = parameters.positional_embedding.detach().to(torch.float32)
        self.point_embeddings = parameters.point_embeddings.to(torch.float32)
        self.not_a_point = parameters.not_a_point_embed.detach().reshape(-1).to(torch.float32)

    @staticmethod
    def _single_object_points(points, labels):
        if points.ndim == 4:
            if points.shape[1] != 1:
                raise ValueError("PR 1 image mode supports one prompted object")
            points = points[:, 0]
        if labels.ndim == 3:
            if labels.shape[1] != 1:
                raise ValueError("PR 1 image mode supports one prompted object")
            labels = labels[:, 0]
        if points.ndim != 3 or labels.ndim != 2 or points.shape[:2] != labels.shape:
            raise ValueError("points and labels must have shapes [1,N,2] and [1,N]")
        return points, labels

    def _embed_labeled_coordinates(self, points, labels):
        projected = (points.to(torch.float32) / 1024.0 * 2.0 - 1.0) @ self.gaussian_matrix
        projected *= 2.0 * math.pi
        embedding = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        for label, weight in enumerate(self.point_embeddings):
            embedding[labels == label] += weight
        embedding[labels == -1] = self.not_a_point
        embedding[labels == -10] = 0
        return ttnn.from_torch(
            embedding,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def _mask_linear(self, x, parameters):
        folded_storage = None
        source = x
        if parameters.kernel_size == 2:
            batch, height, width, channels = x.shape
            grouped = ttnn.reshape(x, (batch, height // 2, 2, width // 2, 2, channels))
            folded_storage = ttnn.permute(grouped, (0, 1, 3, 5, 2, 4))
            x = ttnn.reshape(folded_storage, (batch, height // 2, width // 2, channels * 4))
        tiled = (
            x
            if x.layout == ttnn.TILE_LAYOUT
            else ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        )
        output = ttnn.linear(
            tiled,
            parameters.weight,
            bias=parameters.bias,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._mask_compute_kernel_config,
        )
        if tiled is not x:
            ttnn.deallocate(tiled)
        if folded_storage is not None:
            ttnn.deallocate(folded_storage)
        ttnn.deallocate(source)
        return output

    def embed_dense(self, masks):
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        if masks.ndim != 4 or masks.shape[:2] != (1, 1):
            raise ValueError("masks must have shape [1,1,H,W]")
        if tuple(masks.shape[-2:]) != self.mask_input_size:
            masks = F.interpolate(
                masks.to(torch.float32),
                size=self.mask_input_size,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        hidden = masks.to(torch.float32).permute(0, 2, 3, 1).contiguous()
        hidden = ttnn.from_torch(
            hidden,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for conv, norm in (
            (self.mask_embed.conv1, self.mask_embed.norm1),
            (self.mask_embed.conv2, self.mask_embed.norm2),
        ):
            hidden = self._mask_linear(hidden, conv)
            normalized = ttnn.layer_norm(
                hidden,
                weight=norm.weight,
                bias=norm.bias,
                epsilon=1e-6,
                memory_config=hidden.memory_config(),
                compute_kernel_config=self._mask_compute_kernel_config,
            )
            ttnn.deallocate(hidden)
            hidden = ttnn.gelu(
                normalized,
                fast_and_approximate_mode=False,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(normalized)
        hidden = self._mask_linear(hidden, self.mask_embed.conv3)
        dense = ttnn.permute(hidden, (0, 3, 1, 2))
        ttnn.deallocate(hidden)
        return dense

    def embed_sparse(self, input_points=None, input_labels=None, input_boxes=None):
        sparse = None
        if input_points is not None:
            if input_labels is None:
                raise ValueError("labels are required with points")
            points, labels = self._single_object_points(input_points, input_labels)
            points = points + 0.5
            if input_boxes is None:
                points = F.pad(points, (0, 0, 0, 1), value=0)
                labels = F.pad(labels, (0, 1), value=-1)
            sparse = self._embed_labeled_coordinates(points, labels)
        elif input_labels is not None:
            raise ValueError("points are required with labels")
        if input_boxes is not None:
            if input_boxes.ndim != 3 or input_boxes.shape[1:] != (1, 4):
                raise ValueError("boxes must have shape [1,1,4]")
            corners = (input_boxes + 0.5).reshape(input_boxes.shape[0], 2, 2)
            corners = F.pad(corners, (0, 0, 0, 1), value=0)
            labels = torch.tensor([[2, 3, -1]], dtype=torch.int32, device=input_boxes.device)
            boxes = self._embed_labeled_coordinates(corners, labels)
            if sparse is None:
                sparse = boxes
            else:
                combined = ttnn.concat([sparse, boxes], dim=1)
                ttnn.deallocate(sparse)
                ttnn.deallocate(boxes)
                sparse = combined
        return sparse
