# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

import ttnn

from ..parallel.config import MochiVAEParallelConfig, vae_neighbor_pad
from ..parallel.manager import CCLManager
from .module import Module, Parameter

if TYPE_CHECKING:
    from collections.abc import Sequence


def get_conv3d_config(in_channels, grid_size):
    shape_to_blocking = {
        # (60, 106, 768): (128, 96, 1, 2, 16),
        # (120, 212, 512): (128, 128, 1, 8, 4),
        # (240, 424, 256): (128, 128, 4, 4, 2),
        # (480, 848, 128): (128, 128, 1, 2, 16),
        768: (128, 96, 1, 2, 16),
        512: (128, 128, 1, 8, 4),
        256: (128, 128, 4, 4, 2),
        128: (128, 128, 1, 2, 16),
    }
    blocking = shape_to_blocking.get((in_channels), None)
    if blocking is None:
        C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = 128, 32, 1, 2, 16
        logger.warning(
            f"No blocking found for input shape {in_channels}. Using default blocking: {C_in_block}, {C_out_block}, {T_out_block}, {H_out_block}, {W_out_block}"
        )
    else:
        C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = blocking
    return ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=T_out_block,
        W_out_block=W_out_block,
        H_out_block=H_out_block,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        compute_with_storage_grid_size=grid_size,
    )


class ContextParallelConv3d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: Sequence[int],
        stride: Sequence[int] = (1, 1, 1),
        bias: bool = True,
        causal: bool = True,
        context_parallel: bool = True,
        groups: int = 1,
        padding_mode: str,
        mesh_device: ttnn.MeshDevice,
        parallel_config: MochiVAEParallelConfig,
        ccl_manager: CCLManager,
    ) -> None:
        super().__init__()

        self.halos_chunk_map = {
            768: 30,  # 16
            512: 30,  # 64
            256: 60,  # 96
            128: 120,  # 192,
        }

        assert causal
        assert context_parallel
        assert padding_mode in ["zeros", "replicate"]
        assert groups == 1

        self.mesh_device = mesh_device
        self.grid_size = mesh_device.compute_with_storage_grid_size()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.causal = causal
        self.context_parallel = context_parallel
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        # Conv3d parameters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode
        self.groups = groups

        # Calculate padding
        if self.parallel_config.h_parallel.factor > 1:
            height_pad = 0
        else:
            height_pad = (self.kernel_size[1] - 1) // 2
        if self.parallel_config.w_parallel.factor > 1:
            width_pad = 0
        else:
            width_pad = (self.kernel_size[2] - 1) // 2
        self.padding = (0, height_pad, width_pad)

        d = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
        self.weight = Parameter(total_shape=[d, self.out_channels], device=mesh_device, pad_value=0, on_host=True)
        self.bias = Parameter(total_shape=[1, self.out_channels], device=mesh_device, pad_value=0) if bias else None

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.grid_size,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        alignment = 16

        weight = state.get("weight")
        if weight is not None:
            state["weight"] = weight

        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def _causal_pad_input(self, x_NTHWC, pad_front, pad_back=0):
        """
        Input is either a mult-device tensor (when num_devices == 1) or a single-device tensor (when num_devices > 1).
        """
        T = x_NTHWC.shape[1]
        if self.padding_mode == "zeros":
            # Do front and back padding
            x_pad_NTHWC = ttnn.pad(x_NTHWC, (0, 0, 0, 0, pad_front, pad_back), value=0.0)
            ttnn.deallocate(x_NTHWC)
        elif self.padding_mode == "replicate":
            # Use concat to pad
            front_slice = x_NTHWC[:, 0:1, :, :, :]
            x_pad_NTHWC = ttnn.concat(
                [front_slice] * pad_front + [x_NTHWC],
                dim=1,
            )
            ttnn.deallocate(x_NTHWC)
            T_pad = x_pad_NTHWC.shape[1]
            assert T_pad == T + pad_front + pad_back
        return x_pad_NTHWC

    def forward(self, x_NTHWC: ttnn.Tensor) -> ttnn.Tensor:
        # Compute temporal padding amounts
        context_size = self.kernel_size[0] - 1
        if self.causal:
            pad_front = context_size
            pad_back = 0
        else:
            assert False, "Non-causal padding is not supported"
            # pad_front = context_size // 2
            # pad_back = context_size - pad_front

        if self.parallel_config.time_parallel.factor == 1:
            # Single-device padding
            x_pad_NTHWC = self._causal_pad_input(x_NTHWC, pad_front, pad_back)
        else:
            # Multi-device padding. Input is fractured on dim=1 (T) and must pass frames between devices
            # Pad on first device
            halo_tensor = ttnn.squeeze(x_NTHWC, 0)
            halo_tensor = vae_neighbor_pad(
                self.ccl_manager,
                halo_tensor,
                cluster_axis=self.parallel_config.time_parallel.mesh_axis,
                dim=0,
                padding_left=2,
                padding_right=0,
                padding_mode="replicate",
            )
            x_pad_NTHWC = ttnn.unsqueeze(halo_tensor, 0)

        out_NTHWC = ttnn.experimental.conv3d(
            input_tensor=x_pad_NTHWC,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data if self.bias is not None else None,
            device=self.mesh_device,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            padding_mode=self.padding_mode,
            dtype=ttnn.bfloat16,
            groups=self.groups,
            compute_kernel_config=self.compute_kernel_config,
        )

        return out_NTHWC
