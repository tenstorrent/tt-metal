# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from ..parallel.config import vae_all_gather
from ..utils.tensor import bf16_tensor_host


# TODO: Add support for coll and row parallel conv2d
class Conv2d:
    """
    Conv2d with support for tensor parallelism. Data and Seqence Parallelism TBD.

    """

    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # slice_params[mesh_shape][tuple(height, width, in_channels, out_channels)] = num_slices
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

    # TODO: Allow weight initilization?
    def __init__(
        self,
        in_channels=None,
        out_channels=None,
        kernel_size=None,
        stride=None,
        padding=None,
        dilation=None,
        mesh_device=None,
        mesh_axis=None,
        ccl_manager=None,
        torch_ref=None,
    ):
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
            mesh_axis: Axis to use for mesh parallelism.
            ccl_manager: CCL manager to use.
            torch_ref: Reference to the torch layer. Paramaters from this will be used to iniitialize the layer
        Returns:
            Conv2d layer.
        """

        self.in_channels = in_channels or torch_ref.in_channels
        self.out_channels = out_channels or torch_ref.out_channels
        self.kernel_size = kernel_size or torch_ref.kernel_size
        self.stride = stride or torch_ref.stride
        self.padding = padding or torch_ref.padding
        self.dilation = dilation or torch_ref.dilation
        self.bias = None
        self.weight = None
        self.mesh_device = mesh_device
        self.mesh_axis = mesh_axis
        self.ccl_manager = ccl_manager

        if torch_ref is not None:
            self.load_state_dict(torch_ref.state_dict())

    @classmethod
    def from_torch(cls, torch_ref, mesh_device, mesh_axis, ccl_manager):
        layer = cls(
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            ccl_manager=ccl_manager,
            torch_ref=torch_ref,
        )
        return layer

    def load_state_dict(self, state_dict):
        weight = state_dict["weight"]
        bias = state_dict.get("bias", None)
        self.weight = bf16_tensor_host(
            weight,
            device=self.mesh_device,
            mesh_axis=self.mesh_axis,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            shard_dim=0 if self.mesh_axis is not None else None,
        )
        self.bias = (
            bf16_tensor_host(
                bias.reshape((1, 1, 1, -1)),
                device=self.mesh_device,
                mesh_axis=self.mesh_axis,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                shard_dim=-1 if self.mesh_axis is not None else None,
            )
            if bias is not None
            else None
        )

    def is_sharded_tensor(self, x):
        """
        Check if the tensor is sharded.
        Simple heuristic to check if the tensor is sharded.
        """
        return x.shape[3] < self.in_channels

    def __call__(self, x):
        """
        Gather the tensor if it is sharded, since we only support TP. Will be extended to support DP and SP as needed.
        Data is left in the state of the final compute. The burden is on the next layer to prepare its input as needed.
        TODO: Add support for DP and SP
        """
        if self.is_sharded_tensor(x):
            x = vae_all_gather(self.ccl_manager, x)

        b, h, w, c = x.shape
        slice_config = ttnn.Conv2dSliceConfig(
            num_slices=self.slice_params[tuple(self.mesh_device.shape)][(h, w, self.in_channels, self.out_channels)],
            slice_type=ttnn.Conv2dSliceWidth,
        )

        output_tensor, [_out_height, _out_width] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            in_channels=c,
            out_channels=self.weight.shape[0],
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
        output_tensor = ttnn.reshape(output_tensor, (b, _out_height, _out_width, -1))
        return output_tensor
