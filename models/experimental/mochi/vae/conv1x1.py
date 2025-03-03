import ttnn
from ..common import as_replicated_tensor
from models.common.lightweightmodule import LightweightModule
from typing import Callable


class TtConv1x1(LightweightModule):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        state_dict_prefix: str,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        swizzle_weight: Callable = None,
    ):
        """
        A 1x1 convolution implemented as a linear operation for ttnn.

        Can be instantiated from either:
        - nn.Conv3d with kernel_size=(1,1,1)
        - Conv1x1 (which is implemented as nn.Linear)

        Args:
            mesh_device: TTNN mesh device
            state_dict: Dictionary containing weights
            state_dict_prefix: Prefix for loading weights from state_dict
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include bias
            swizzle_weight: Function to swizzle weights, useful for channel expansion
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

        # Configure compute kernel
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        # Load weights - supports both Conv3d(1,1,1) and Conv1x1 format
        weight = state_dict[f"{state_dict_prefix}weight"]
        if weight.ndim == 5:  # Conv3d weight
            # Convert from (out_channels, in_channels, 1, 1, 1) to (out_channels, in_channels)
            weight = weight.squeeze()
        weight = weight.transpose(0, 1)  # (out_channels, in_channels) -> (in_channels, out_channels)
        if swizzle_weight:
            weight = swizzle_weight(weight)
        self.weight = as_replicated_tensor(weight, mesh_device)

        assert bias == (f"{state_dict_prefix}bias" in state_dict)
        if bias:
            bias_weight = state_dict[f"{state_dict_prefix}bias"]
            if swizzle_weight:
                bias_weight = swizzle_weight(bias_weight)
            self.bias = as_replicated_tensor(bias_weight.reshape(1, -1), mesh_device)
        else:
            self.bias = None

    def forward(self, x_NTHWC):
        """
        Forward pass for Conv1x1.

        Args:
            x_NTHWC: Input tensor in NTHWC layout

        Returns:
            Output tensor in NTHWC layout
        """
        # Convert to tile layout for efficient computation
        x_tile_NTHWC = ttnn.to_layout(x_NTHWC, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x_NTHWC)

        # Apply linear transformation
        x_tile_NTHWO = ttnn.linear(
            x_tile_NTHWC,
            self.weight,
            bias=self.bias,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x_tile_NTHWC)

        # Convert back to row major layout
        x_NTHWO = ttnn.to_layout(x_tile_NTHWO, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x_tile_NTHWO)

        return x_NTHWO
