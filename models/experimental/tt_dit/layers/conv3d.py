# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from typing import Tuple

from loguru import logger
from ..parallel.config import vae_neighbor_pad


def get_conv3d_config(in_channels, out_channels, kernel_size, stride, padding, padding_mode, grid_size):
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
    blocking = shape_to_blocking.get(in_channels, None)
    if blocking is None:
        C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = 128, 32, 1, 2, 16
        logger.warning(
            f"No blocking found for input shape {in_channels}. Using default blocking: {C_in_block}, {C_out_block}, {T_out_block}, {H_out_block}, {W_out_block}"
        )
    else:
        C_in_block, C_out_block, T_out_block, H_out_block, W_out_block = blocking
    return ttnn.Conv3dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=T_out_block,
        W_out_block=W_out_block,
        H_out_block=H_out_block,
        C_out_block=C_out_block,
        C_in_block=C_in_block,
        output_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        padding_mode=padding_mode,
        groups=1,
        compute_with_storage_grid_size=grid_size,
    )


def prepare_conv3d_weights(mesh_device, weight, bias, conv_config, ALIGNMENT=16):
    """Prepare weights and bias for TTNN."""
    C_in = weight.shape[1]
    w = weight.permute(2, 3, 4, 1, 0)  # kD, kH, kW, C, out_chan
    ALIGN_PAD = ALIGNMENT - C_in % ALIGNMENT
    if C_in % ALIGNMENT != 0:
        w = torch.nn.functional.pad(w, (0, 0, 0, ALIGN_PAD))

    # Reshape weights so that num_C_in_blocks is the first dimension
    kD, kH, kW, C_in_aligned, out_channels = w.shape

    C_in_block = conv_config.C_in_block
    C_in_block = C_in_aligned if C_in_block == 0 else C_in_block
    num_C_in_blocks = C_in_aligned // C_in_block
    assert num_C_in_blocks * C_in_block == C_in_aligned

    # Kernel expects num_C_in_blocks to be the first dimension to stride over it
    w = w.reshape(kD, kH, kW, num_C_in_blocks, C_in_block, out_channels)
    w = w.permute(3, 0, 1, 2, 4, 5)
    w = w.reshape(-1, out_channels)

    tt_weight = ttnn.from_torch(
        w,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        device=mesh_device,
        mesh_mapper=None,
        pad_value=0,
    )

    if bias is not None:
        tt_bias = ttnn.from_torch(
            bias.reshape(1, -1),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh_device,
            mesh_mapper=None,
            pad_value=0,
        )
    else:
        tt_bias = None
    return tt_weight, tt_bias


class ContextParallelConv3d:
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        in_channels: int = None,
        out_channels: int = None,
        kernel_size: Tuple[int, int, int] = None,
        stride: Tuple[int, int, int] = (1, 1, 1),
        causal: bool = True,
        input_shape=None,
        context_parallel: bool = True,
        groups: int = 1,
        parallel_config=None,
        ccl_manager=None,
        torch_ref=None,
        **kwargs,
    ):
        self.halos_chunk_map = {
            768: 30,  # 16
            512: 30,  # 64
            256: 60,  # 96
            128: 120,  # 192,
        }

        assert causal
        assert context_parallel
        self.mesh_device = mesh_device
        self.grid_size = mesh_device.compute_with_storage_grid_size()
        self.in_channels = in_channels or torch_ref.in_channels
        self.out_channels = out_channels or torch_ref.out_channels
        self.causal = causal
        self.context_parallel = context_parallel
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        # Conv3d parameters
        self.kernel_size = kernel_size or torch_ref.kernel_size
        self.stride = stride or torch_ref.stride
        self.has_bias = kwargs["bias"]
        self.padding_mode = kwargs["padding_mode"]
        assert self.padding_mode in ["zeros", "replicate"]
        assert groups == 1
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

        self.weight = None
        self.bias = None

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        if torch_ref is not None:
            self.load_state_dict(torch_ref.state_dict())
        else:
            # TODO initialize torch_weight and bias
            logger.warning("Torch ref not provided")

        logger.warning("Setting up conv3d config")
        conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.padding_mode,
            self.grid_size,
        )
        self.conv_config = conv_config
        self.weight, self.bias = prepare_conv3d_weights(
            self.mesh_device, self.torch_weight, self.torch_bias, conv_config
        )

    @classmethod
    def from_torch(cls, torch_ref, mesh_device, parallel_config, ccl_manager):
        layer = cls(
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            torch_ref=torch_ref,
        )
        return layer

    def load_state_dict(self, state_dict):
        self.torch_weight = state_dict["weight"]
        self.torch_bias = state_dict.get("bias", None)

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

    def __call__(self, x_NTHWC):
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
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            config=self.conv_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        return out_NTHWC
