import ttnn
from models.common.lightweightmodule import LightweightModule

from models.experimental.mochi.common import col_parallel_linear, create_linear_layer


class TtFinalLayer(LightweightModule):
    """The final layer of DiT implemented for TensorTorch."""

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        hidden_size: int,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.state_dict = state_dict
        self.state_dict_prefix = state_dict_prefix
        self.weight_cache_path = weight_cache_path
        self.dtype = dtype
        self.hidden_size = hidden_size

        # Create norm_final layer weights
        norm_weight_key = f"{state_dict_prefix}.norm_final.weight"
        norm_bias_key = f"{state_dict_prefix}.norm_final.bias"

        # LayerNorm should not have learnable parameters (elementwise_affine=False)
        if norm_weight_key in state_dict or norm_bias_key in state_dict:
            raise ValueError("norm_final should not have learnable parameters")

        # Create modulation layer
        self.mod, self.mod_bias = col_parallel_linear(
            "mod",
            bias=True,
            weight_cache_path=weight_cache_path,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            mesh_device=mesh_device,
        )

        # Create final linear layer
        self.linear, self.linear_bias = create_linear_layer(
            "linear",
            weight_cache_path=weight_cache_path,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            mesh_device=mesh_device,
        )

    def forward(self, x: ttnn.Tensor, c: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass of the final layer.

        Args:
            x: (1, B, N, hidden_size) tensor of input features
            c: (1, 1, B, hidden_size) tensor of conditioning features

        Returns:
            x: (1, B, N, patch_size * patch_size * out_channels) output tensor
        """
        # Set up compute kernel config for high-fidelity computations
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Apply activation to conditioning
        c = ttnn.silu(c)

        # Apply modulation layer
        mod = ttnn.linear(
            c,
            self.mod,
            bias=self.mod_bias,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.num_devices > 1:
            mod = ttnn.all_gather(mod, dim=3)

        # Split modulation into shift and scale
        shift = mod[:, :, :, : self.hidden_size]
        scale = mod[:, :, :, self.hidden_size :]

        # Apply layer norm with activation-dependent weights
        x = ttnn.layer_norm(x, epsilon=1e-6, weight=(1 + scale), bias=shift)

        # Final linear projection
        x = ttnn.linear(
            x,
            self.linear,
            bias=self.linear_bias,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return x
