# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from ..utils.tensor import bf16_tensor


class RMSNorm:
    def __init__(
        self, embedding_dim, norm_eps=1e-5, norm_elementwise_affine=True, bias=True, mesh_device=None, init=False
    ):
        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.mesh_device = mesh_device
        self.use_bias = bias
        self.weight = None
        self.bias = None
        if norm_elementwise_affine and init:
            self.weight = bf16_tensor(torch.randn(1, embedding_dim), device=self.mesh_device)
            if bias:
                self.bias = bf16_tensor(torch.randn(1, embedding_dim), device=self.mesh_device)

    def load_state_dict(self, state_dict):
        if self.norm_elementwise_affine:
            self.weight = bf16_tensor(state_dict["weight"].unsqueeze(0), device=self.mesh_device)
            if self.use_bias:
                self.bias = bf16_tensor(state_dict["bias"].unsqueeze(0), device=self.mesh_device)

    def __call__(self, x):
        return ttnn.rms_norm(x, weight=self.weight, bias=self.bias, epsilon=self.norm_eps)


class LayerNorm:
    def __init__(
        self, embedding_dim, norm_eps=1e-5, norm_elementwise_affine=True, bias=True, mesh_device=None, init=False
    ):
        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.mesh_device = mesh_device
        self.use_bias = bias
        self.weight = None
        self.bias = None
        if norm_elementwise_affine and init:
            self.weight = bf16_tensor(torch.randn(1, embedding_dim), device=self.mesh_device)
            if bias:
                self.bias = bf16_tensor(torch.randn(1, embedding_dim), device=self.mesh_device)

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def load_state_dict(self, state_dict):
        if self.norm_elementwise_affine:
            self.weight = bf16_tensor(state_dict["weight"].unsqueeze(0), device=self.mesh_device)
            if self.use_bias:
                self.bias = bf16_tensor(state_dict["bias"].unsqueeze(0), device=self.mesh_device)

    def __call__(self, x):
        return ttnn.layer_norm(
            x,
            weight=self.weight,
            bias=self.bias,
            epsilon=self.norm_eps,
            compute_kernel_config=self.compute_kernel_config,
        )


class DistributedLayerNorm:
    """
    Implements LayerNorm on an activation sharded on the reduction dimension.

    Requires gamma and beta, which will be created if not provided.
    """

    def __init__(
        self,
        embedding_dim,
        norm_eps=1e-5,
        norm_elementwise_affine=True,
        bias=True,
        mesh_axis=0,
        mesh_device=None,
        ccl_manager=None,
        init=False,
    ):
        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.use_bias = bias
        self.mesh_axis = mesh_axis
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.weight = None
        self.bias = None
        self.mesh_width = tuple(mesh_device.shape)[mesh_axis]
        self.TILE_SIZE = 32
        if init or not (norm_elementwise_affine and bias):
            if not (norm_elementwise_affine and bias):
                pass  # TODO: make logging less noisy
                # logger.debug(
                #     "DistributedLayerNorm initialized with norm_elementwise_affine=False. Creating gamma and beta tensors to meet op requirements."
                # )
            weight = torch.ones(1, embedding_dim)
            weight = weight.reshape([-1, self.TILE_SIZE * self.mesh_width])
            bias = torch.zeros(1, embedding_dim)
            bias = bias.reshape([-1, self.TILE_SIZE * self.mesh_width])
            self.weight = bf16_tensor(
                weight, device=self.mesh_device, mesh_axis=mesh_axis, shard_dim=-1, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            self.bias = bf16_tensor(
                bias, device=self.mesh_device, mesh_axis=mesh_axis, shard_dim=-1, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def load_state_dict(self, state_dict):
        if self.norm_elementwise_affine:
            weight = state_dict["weight"]
            weight = (
                weight.reshape(self.mesh_width, -1, self.TILE_SIZE)
                .permute(1, 0, 2)
                .reshape(-1, self.TILE_SIZE * self.mesh_width)
            )
            self.weight = bf16_tensor(
                weight, device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-1, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            if self.use_bias:
                bias = state_dict["bias"]
                bias = (
                    bias.reshape(self.mesh_width, -1, self.TILE_SIZE)
                    .permute(1, 0, 2)
                    .reshape(-1, self.TILE_SIZE * self.mesh_width)
                )
                self.bias = bf16_tensor(
                    bias, device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-1, layout=ttnn.ROW_MAJOR_LAYOUT
                )

    def __call__(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "weight and bias must be initialized before calling __call__"
        stats = ttnn.layer_norm_pre_all_gather(x)

        stats_gathered = ttnn.experimental.all_gather_async(
            stats,
            dim=len(x.shape) - 1,
            cluster_axis=self.mesh_axis,
            mesh_device=x.device(),
            topology=self.ccl_manager.topology,
            multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
            persistent_output_tensor=self.ccl_manager.get_ag_ping_pong_buffer(
                stats.shape, len(stats.shape) - 1, self.mesh_axis
            ),
            num_links=self.ccl_manager.num_links,
        )

        x = ttnn.layer_norm_post_all_gather(
            x,
            stats_gathered,
            weight=self.weight,
            bias=self.bias,
            epsilon=self.norm_eps,
            compute_kernel_config=self.compute_kernel_config,
        )
        return x
