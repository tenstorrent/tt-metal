# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Pixtral patch embedding via ttnn.conv2d.

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule


class TtPixtralPatchConv(LightweightModule):
    """Pixtral patch Conv via meta keys ``{prefix}_linear.*``; torch ``[N,C,H,W]`` → ``[N,patches,out]``."""

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        dtype,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self._prepared_hw: tuple[int, int] | None = None

        weight = state_dict[f"{state_dict_prefix}_linear.weight"]
        if weight.ndim == 2:
            weight = weight.T.reshape(out_channels, in_channels, kernel_size, kernel_size)

        self._host_weight_rm = ttnn.from_torch(
            weight,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self.bias = None
        self._host_bias_rm = None
        if bias:
            bias_t = state_dict[f"{state_dict_prefix}_linear.bias"].reshape(1, 1, 1, -1)
            self._host_bias_rm = ttnn.from_torch(bias_t, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

        self._conv_weight = None
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        patch_conv_core_grid = ttnn.CoreGrid(y=8, x=8)
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=dtype,
            output_layout=ttnn.TILE_LAYOUT,
            core_grid=ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(patch_conv_core_grid.x - 1, patch_conv_core_grid.y - 1),
                    )
                }
            ),
            override_sharding_config=True,
        )

    def _prepare_weights_if_needed(self, batch_size: int, height: int, width: int) -> None:
        if self._prepared_hw == (height, width) and self._conv_weight is not None:
            return

        prep_kwargs = dict(
            input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            input_layout=ttnn.ROW_MAJOR_LAYOUT,
            weights_format="OIHW",
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=(self.stride, self.stride),
            padding=(0, 0),
            dilation=(1, 1),
            has_bias=self._host_bias_rm is not None,
            groups=1,
            device=self.mesh_device,
            input_dtype=ttnn.bfloat16,
            conv_config=self.conv_config,
            compute_config=self.compute_kernel_config,
        )

        self._conv_weight = ttnn.prepare_conv_weights(
            weight_tensor=self._host_weight_rm,
            **prep_kwargs,
        )
        if self._host_bias_rm is not None:
            self.bias = ttnn.prepare_conv_bias(
                bias_tensor=self._host_bias_rm,
                **prep_kwargs,
            )
        else:
            self.bias = None

        if not ttnn.is_tensor_storage_on_device(self._conv_weight):
            self._conv_weight = ttnn.to_device(self._conv_weight, self.mesh_device)
        if self.bias is not None and not ttnn.is_tensor_storage_on_device(self.bias):
            self.bias = ttnn.to_device(self.bias, self.mesh_device)

        self._prepared_hw = (height, width)

    def forward(self, x) -> ttnn.Tensor:
        batch_size, _, height, width = x.shape
        self._prepare_weights_if_needed(batch_size, height, width)

        x = ttnn.as_tensor(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        x = ttnn.permute(x, [0, 2, 3, 1])

        output, [out_height, out_width] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self._conv_weight,
            bias_tensor=self.bias,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=height,
            input_width=width,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=(self.stride, self.stride),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            conv_config=self.conv_config,
            compute_config=self.compute_kernel_config,
            return_output_dim=True,
            return_weights_and_bias=False,
        )
        ttnn.deallocate(x)
        return ttnn.reshape(output, (batch_size, out_height * out_width, self.out_channels))


__all__ = ["TtPixtralPatchConv"]
