# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import ttnn

from ..parallel.config import vae_all_gather
from ..parallel.manager import CCLManager
from .module import Module, Parameter

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch
    from typing_extensions import Self


# TODO: Add support for coll and row parallel conv2d
class Conv2d(Module):
    """
    Conv2d with support for tensor parallelism. Data and Seqence Parallelism TBD.

    """

    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    slice_params = {
        (1, 4): {
            (512, 512, 512, 64): 16,
            (128, 128, 16, 512): 8,
            (128, 128, 512, 512): 4,
            (256, 256, 512, 512): 8,
            (512, 512, 512, 512): 16,
            (512, 512, 512, 256): 16,
            (512, 512, 256, 256): 4,
            (1024, 1024, 256, 256): 16,
            (1024, 1024, 256, 128): 16,
            (1024, 1024, 128, 128): 16,
            (1024, 1024, 128, 3): 8,
            (128, 128, 16, 16): 1,
            (128, 128, 16, 384): 1,
            (128, 128, 384, 384): 1,
            (256, 256, 192, 384): 2,
            (256, 256, 384, 192): 2,
            (256, 256, 384, 384): 4,
            (512, 512, 192, 192): 4,
            (512, 512, 384, 192): 8,
            (1024, 1024, 96, 3): 8,
            (1024, 1024, 96, 96): 8,
            (1024, 1024, 192, 96): 16,
        },
        (1, 8): {
            (512, 512, 512, 64): 16,
            (128, 128, 16, 512): 8,
            (128, 128, 512, 512): 4,
            (256, 256, 512, 512): 8,
            (512, 512, 512, 512): 16,
            (512, 512, 512, 256): 16,
            (512, 512, 256, 256): 4,
            (1024, 1024, 256, 256): 16,
            (1024, 1024, 256, 128): 16,
            (1024, 1024, 128, 128): 16,
            (1024, 1024, 128, 3): 8,
        },
        (4, 4): {
            (512, 512, 512, 64): 16,
            (128, 128, 16, 512): 8,
            (128, 128, 512, 512): 4,
            (256, 256, 512, 512): 8,
            (512, 512, 512, 512): 16,
            (512, 512, 512, 256): 16,
            (512, 512, 256, 256): 4,
            (1024, 1024, 256, 256): 16,
            (1024, 1024, 256, 128): 16,
            (1024, 1024, 128, 128): 16,
            (1024, 1024, 128, 3): 8,
        },
        (2, 4): {
            (128, 128, 16, 512): 8,
            (128, 128, 512, 512): 4,
            (256, 256, 512, 512): 8,
            (512, 512, 512, 512): 16,
            (512, 512, 512, 256): 16,
            (512, 512, 256, 256): 4,
            (1024, 1024, 256, 256): 16,
            (1024, 1024, 256, 128): 16,
            (1024, 1024, 128, 128): 16,
            (1024, 1024, 128, 3): 8,
        },
    }
    slice_default = {
        (512, 512, 512, 64): 16,
        (128, 128, 16, 512): 8,
        (128, 128, 512, 512): 4,
        (256, 256, 512, 512): 8,
        (512, 512, 512, 512): 16,
        (512, 512, 512, 256): 16,
        (512, 512, 256, 256): 4,
        (1024, 1024, 256, 256): 16,
        (1024, 1024, 256, 128): 16,
        (1024, 1024, 128, 128): 16,
        (1024, 1024, 128, 3): 8,
    }

    # TODO: Allow weight initilization?
    def __init__(
        self,
        in_channels: int | None,
        out_channels: int | None,
        *,
        kernel_size: Sequence[int] | int | None = None,
        stride: Sequence[int] | int = 1,
        padding: Sequence[int] | int = 0,
        dilation: Sequence[int] | int = 1,
        mesh_device: ttnn.MeshDevice,
        in_mesh_axis: int | None = None,
        out_mesh_axis: int | None = None,
        ccl_manager: CCLManager | None = None,
        torch_ref: torch.nn.Conv2d | None = None,
    ) -> None:
        """
        Initialize the Conv2d layer. Set mesh_axis to None to disable mesh parallelism. Only TP is supported currently.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the kernel.
            stride: Stride of the convolution.
            padding: Padding of the convolution.
            dilation: Dilation of the convolution.
            mesh_device: Mesh device to use.
            in_mesh_axis: Axis to shard input channels across mesh devices.
            out_mesh_axis: Axis to shard output channels across mesh devices.
            ccl_manager: CCL manager to use.
            torch_ref: Reference to the torch layer. Paramaters from this will be used to iniitialize the layer

        The arguments in_mesh_axis and out_mesh_axis control how weights and biases are sharded
        across the mesh devices and how the input and output tensors are sharded.
        """
        super().__init__()

        if in_mesh_axis is not None:
            if out_mesh_axis is not None:
                msg = "only one of in_mesh_axis and out_mesh_axis can be set"
                raise ValueError(msg)

            if ccl_manager is None:
                msg = "ccl_manager must be provided if in_mesh_axis is not None"
                raise ValueError(msg)

        in_mesh_axis_size = mesh_device.shape[in_mesh_axis] if in_mesh_axis is not None else 1
        out_mesh_axis_size = mesh_device.shape[out_mesh_axis] if out_mesh_axis is not None else 1

        if torch_ref is not None:
            assert not isinstance(torch_ref.padding, str)

            in_channels = torch_ref.in_channels
            out_channels = torch_ref.out_channels
            kernel_size = torch_ref.kernel_size
            stride = torch_ref.stride
            padding = torch_ref.padding
            dilation = torch_ref.dilation
        else:
            assert in_channels is not None, "in_channels must be provided if torch_ref is not provided"
            assert out_channels is not None, "out_channels must be provided if torch_ref is not provided"
            assert kernel_size is not None, "kernel_size must be provided if torch_ref is not provided"

        kernel_size = (kernel_size,) * 2 if isinstance(kernel_size, int) else tuple(kernel_size)
        stride = (stride,) * 2 if isinstance(stride, int) else tuple(stride)
        padding = (padding,) * 2 if isinstance(padding, int) else tuple(padding)
        dilation = (dilation,) * 2 if isinstance(dilation, int) else tuple(dilation)

        assert dilation == (1, 1), "dilation other than 1 is not supported"

        self.weight = Parameter(
            total_shape=[out_channels, in_channels, *kernel_size],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_axes=[out_mesh_axis, in_mesh_axis, None, None],
            on_host=True,
        )

        self.bias = Parameter(
            total_shape=[in_mesh_axis_size, 1, 1, out_channels],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_axes=[in_mesh_axis, None, None, out_mesh_axis],
            on_host=True,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mesh_device = mesh_device
        self.out_mesh_axis = out_mesh_axis
        self.in_mesh_axis = in_mesh_axis
        self.in_mesh_axis_size = in_mesh_axis_size
        self.out_mesh_axis_size = out_mesh_axis_size
        self.ccl_manager = ccl_manager

        if torch_ref is not None:
            self.load_torch_state_dict(torch_ref.state_dict())

    @classmethod
    def from_torch(
        cls,
        torch_ref: torch.nn.Conv2d,
        *,
        mesh_device: ttnn.MeshDevice,
        in_mesh_axis: int | None = None,
        out_mesh_axis: int | None = None,
        ccl_manager: CCLManager | None,
    ) -> Self:
        assert not isinstance(torch_ref.padding, str)

        model = cls(
            in_channels=torch_ref.in_channels,
            out_channels=torch_ref.out_channels,
            kernel_size=torch_ref.kernel_size,
            stride=torch_ref.stride,
            padding=torch_ref.padding,
            dilation=torch_ref.dilation,
            mesh_device=mesh_device,
            out_mesh_axis=out_mesh_axis,
            in_mesh_axis=in_mesh_axis,
            ccl_manager=ccl_manager,
        )

        model.load_torch_state_dict(torch_ref.state_dict())

        return model

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        bias = state.pop("bias", None)
        if bias is not None:
            (out_dim,) = bias.shape
            bias = bias.reshape([1, 1, 1, out_dim])
            bias_zeros = torch.zeros([self.in_mesh_axis_size - 1, 1, 1, out_dim])
            state["bias"] = torch.cat([bias, bias_zeros])

    def forward(self, x: ttnn.Tensor, /) -> ttnn.Tensor:
        """Forward pass of the Conv2d layer with support for tensor parallelism.

        Args:
            x: Input tensor with shape (batch, height, width, channels).

        Returns:
            Output tensor with shape (batch, out_height, out_width, out_channels).

        Tensor sharding behavior:

        **No parallelism (in_mesh_axis=None, out_mesh_axis=None):**
            - Input is replicated
            - Output is replicated

        **Input channel parallelism (in_mesh_axis is set, out_mesh_axis=None):**
            - Input: sharded on channel dimension along in_mesh_axis
            - Output: sharded on channel dimension along in_mesh_axis

        **Output channel parallelism (in_mesh_axis=None, out_mesh_axis is set):**
            - Input can be either:
                - replicated, or
                - sharded on channel dimension along out_mesh_axis
            - Output: sharded on channel dimension along out_mesh_axis
        """
        b, h, w, c = x.shape

        # allow sharded input for output parallelism
        if (
            self.in_mesh_axis is None
            and self.out_mesh_axis_size != 1
            and c == self.in_channels // self.out_mesh_axis_size
        ):
            x = vae_all_gather(self.ccl_manager, x, cluster_axis=self.out_mesh_axis)
        else:
            expected_c = self.in_channels // self.in_mesh_axis_size
            if c != expected_c:
                msg = f"expected input channel dimension to be {expected_c}, but got {c}"
                raise ValueError(msg)

        slice_config = ttnn.Conv2dSliceConfig(
            num_slices=self.slice_params.get(tuple(self.mesh_device.shape), self.slice_default)[
                (h, w, self.in_channels, self.out_channels)
            ],
            slice_type=ttnn.Conv2dDRAMSliceWidth,
        )

        try:
            x, [out_height, out_width] = ttnn.conv2d(
                input_tensor=x,
                weight_tensor=self.weight.data,
                bias_tensor=self.bias.data if self.bias is not None else None,
                in_channels=self.in_channels // self.in_mesh_axis_size,
                out_channels=self.weight.data.shape[0],
                device=self.mesh_device,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                batch_size=b,
                input_height=h,
                input_width=w,
                conv_config=None,
                compute_config=self.compute_config,
                slice_config=slice_config,
                return_output_dim=True,
            )
        except RuntimeError as e:
            m = re.search(r"Out of Memory: (.*)", str(e))
            if m is None:
                raise

            msg = (
                f"conv2d out of memory with (height, width, in_channels, out_channels) = "
                f"{(h, w, self.in_channels, self.out_channels)} and mesh_shape = "
                f"{tuple(self.mesh_device.shape)}: {m.group(1)}"
            )
            raise RuntimeError(msg) from e

        x = ttnn.reshape(x, (b, out_height, out_width, -1))

        if self.in_mesh_axis is not None:
            x = self.ccl_manager.reduce_scatter_persistent_buffer(x, dim=-1, mesh_axis=self.in_mesh_axis)

        return x
