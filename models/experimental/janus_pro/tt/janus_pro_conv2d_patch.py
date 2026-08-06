"""
Conv2d patch-embedding for the Janus-Pro-7B vision model.

The convolution weight is folded into a 2D matrix so the patch projection runs
as a single ttnn.linear over the unfolded input. A 4D conv weight is reshaped to
(out_channels, in_channels * kernel_size**2); its inner dimension is zero-padded
to a tile multiple.
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtJanusProConv2dPatch(LightweightModule):
    """Conv2D Patching layer.
    Column parallel over unfolded input.
    Arguments:
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride: Stride for convolution.
        bias: Use bias in Conv2d.
    Input: (bsz, in_channels, height, width)
    Output: (bsz, num_tokens, out_channels)

    The input channel count is not an argument: the folded weight already carries it as the
    matmul's K, and the host unfold reads it off the tensor.
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        dtype,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.dtype = dtype
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.bias = (
            ttnn.as_tensor(
                torch.reshape(state_dict[f"{state_dict_prefix}_linear.bias"], (1, -1)),
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            if bias
            else None
        )

        self._unfold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

        weight = state_dict[f"{state_dict_prefix}_linear.weight"]
        if weight.ndim == 4:
            weight = weight.view(out_channels, -1)
        weight = weight.permute(1, 0).reshape(1, 1, -1, self.out_channels)

        self._linear_weight = ttnn.as_tensor(
            weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def prepare_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Host-side im2col, returning patches ready for transfer.

        Split out of the device compute so a trace can begin at the first device op: this
        is torch on host, and a host call captured inside a trace region would execute once
        at capture and never again on replay.
        """
        return self._unfold(x).permute(0, 2, 1)

    def patches_to_device(self, patches: torch.Tensor) -> ttnn.Tensor:
        """Move prepared patches to device.

        Also outside the device compute: a traced replay reads its inputs from buffers that
        already exist at fixed addresses, so the allocation cannot happen inside the trace.
        """
        return ttnn.as_tensor(
            patches,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def forward_device(self, patches: ttnn.Tensor) -> ttnn.Tensor:
        """Patch projection over already-resident patches; traceable end to end.

        Does not free ``patches``: on the traced path that tensor is the persistent input
        buffer and has to outlive every replay.
        """
        # The bias is a separate add. Folding it in was never evaluated here -- unlike the
        # transformer body, this linear runs once per image, so an op saved is not measurable.
        out = ttnn.linear(
            patches,
            self._linear_weight,
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )

        if self.bias is not None:
            out = ttnn.add(out, self.bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return out

    def forward(self, x: torch.Tensor):
        patches = self.patches_to_device(self.prepare_patches(x))
        out = self.forward_device(patches)
        ttnn.deallocate(patches)
        return out
