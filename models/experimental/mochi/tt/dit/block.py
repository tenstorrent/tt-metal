from typing import Optional, Tuple

import ttnn
from models.common.lightweightmodule import LightweightModule

from models.experimental.mochi.tt.common import create_linear_layer
from models.experimental.mochi.tt.dit.attention import AsymmetricAttention
from models.experimental.mochi.tt.dit.mlp import FeedForward
from models.experimental.mochi.tt.dit.norms import modulated_rmsnorm, residual_tanh_gated_rmsnorm


class AsymmetricJointBlock(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        layer_num,
        dtype,
        hidden_size_x: int,
        hidden_size_y: int,
        num_heads: int,
        *,
        mlp_ratio_x: float = 8.0,  # Ratio of hidden size to d_model for MLP for visual tokens
        mlp_ratio_y: float = 4.0,  # Ratio of hidden size to d_model for MLP for text tokens
        update_y: bool = True,  # Whether to update text tokens in this block
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        **block_kwargs,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.state_dict = state_dict
        self.state_dict_prefix = state_dict_prefix
        self.weight_cache_path = weight_cache_path
        self.layer_num = layer_num
        self.dtype = dtype

        self.update_y = update_y
        self.hidden_size_x = hidden_size_x
        self.hidden_size_y = hidden_size_y

        # Create modulation layers with weights and biases
        self.mod_x, self.mod_x_bias = create_linear_layer(
            "mod_x",
            weight_cache_path=weight_cache_path,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            mesh_device=mesh_device,
        )

        if self.update_y:
            self.mod_y, self.mod_y_bias = create_linear_layer(
                "mod_y",
                weight_cache_path=weight_cache_path,
                state_dict=state_dict,
                state_dict_prefix=state_dict_prefix,
                mesh_device=mesh_device,
            )
        else:
            self.mod_y, self.mod_y_bias = create_linear_layer(
                "mod_y",
                weight_cache_path=weight_cache_path,
                state_dict=state_dict,
                state_dict_prefix=state_dict_prefix,
                mesh_device=mesh_device,
            )

        # Self-attention
        self.attn = AsymmetricAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}.attn",
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            dim_x=hidden_size_x,
            dim_y=hidden_size_y,
            num_heads=num_heads,
            update_y=update_y,
            **block_kwargs,
        )

        # MLP layers using FeedForward
        mlp_hidden_dim_x = int(hidden_size_x * mlp_ratio_x)
        self.mlp_x = FeedForward(
            mesh_device=mesh_device,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            in_features=hidden_size_x,
            hidden_size=mlp_hidden_dim_x,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
            state_dict_prefix=f"{state_dict_prefix}.mlp_x",
        )

        if self.update_y:
            mlp_hidden_dim_y = int(hidden_size_y * mlp_ratio_y)
            self.mlp_y = FeedForward(
                mesh_device=mesh_device,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=layer_num,
                dtype=dtype,
                in_features=hidden_size_y,
                hidden_size=mlp_hidden_dim_y,
                multiple_of=multiple_of,
                ffn_dim_multiplier=ffn_dim_multiplier,
                state_dict_prefix=f"{state_dict_prefix}.mlp_y",
            )

    def ff_block_x(self, x_1BNX: ttnn.Tensor, scale_x_B11X: ttnn.Tensor, gate_x_B11X: ttnn.Tensor) -> ttnn.Tensor:
        """Feed-forward block for visual features.

        Args:
            x_1BNX: Input tensor of shape (1, B, N, X)
            scale_x_B11X: Scale tensor of shape (B, 1, 1, X)
            gate_x_B11X: Gate tensor of shape (B, 1, 1, X)

        Returns:
            Tensor of shape (1, B, N, X)
        """
        x_mod_1BNX = modulated_rmsnorm(x_1BNX, scale_x_B11X)
        x_res_shard_1BNX = self.mlp_x(x_mod_1BNX)
        if self.num_devices > 1:
            # Collect hidden-dim-fractured MLP outputs
            x_res_1BNX = ttnn.all_gather(x_res_shard_1BNX, dim=3)
        else:
            x_res_1BNX = x_res_shard_1BNX
        x_1BNX = residual_tanh_gated_rmsnorm(x_1BNX, x_res_1BNX, gate_x_B11X)
        return x_1BNX

    def ff_block_y(self, y_1BLY: ttnn.Tensor, scale_y_B11Y: ttnn.Tensor, gate_y_B11Y: ttnn.Tensor) -> ttnn.Tensor:
        """Feed-forward block for text features.

        Args:
            y_1BLY: Input tensor of shape (1, B, L, Y)
            scale_y_B11Y: Scale tensor of shape (B, 1, 1, Y)
            gate_y_B11Y: Gate tensor of shape (B, 1, 1, Y)

        Returns:
            Tensor of shape (1, B, L, Y)
        """
        y_mod_1BLY = modulated_rmsnorm(y_1BLY, scale_y_B11Y)
        y_res_shard_1BLY = self.mlp_y(y_mod_1BLY)
        if self.num_devices > 1:
            # Collect hidden-dim-fractured MLP outputs
            y_res_1BLY = ttnn.all_gather(y_res_shard_1BLY, dim=3)
        else:
            y_res_1BLY = y_res_shard_1BLY
        y_1BLY = residual_tanh_gated_rmsnorm(y_1BLY, y_res_1BLY, gate_y_B11Y)
        return y_1BLY

    def forward(
        self,
        x: ttnn.Tensor,
        c: ttnn.Tensor,
        y: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        uncond: bool = False,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Forward pass of a block.
        Shape metadata:
            B: batch
            N: vision sequence length
            L: text sequence length
            H: number of heads
            D: head dim
            X: visual hidden dim
            Y: text hidden dim
            M: 4 * X
            C: 4 * Y
        """
        x_1BNX = x
        c_B11X = c
        y_1BLY = y
        rope_cos_1HND = rope_cos
        rope_sin_1HND = rope_sin

        N = x_1BNX.shape[2]

        # Set up compute kernel config for high-fidelity computations
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Apply modulation
        c_B11X = ttnn.silu(c_B11X)

        # Apply linear layers with bias
        mod_x_B11M = ttnn.linear(
            c_B11X,
            self.mod_x,
            bias=self.mod_x_bias,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        scale_msa_x_B11X = mod_x_B11M[:, :, :, : self.hidden_size_x]
        gate_msa_x_B11X = mod_x_B11M[:, :, :, self.hidden_size_x : 2 * self.hidden_size_x]
        scale_mlp_x_B11X = mod_x_B11M[:, :, :, 2 * self.hidden_size_x : 3 * self.hidden_size_x]
        gate_mlp_x_B11X = mod_x_B11M[:, :, :, 3 * self.hidden_size_x :]
        # scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = ttnn.split(mod_x, 4, dim=1)

        scale_msa_y_B11Y = None
        if not uncond:
            mod_y_B11C = ttnn.linear(
                c_B11X,
                self.mod_y,
                bias=self.mod_y_bias,
                compute_kernel_config=compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=8, x=8),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            if self.update_y:
                scale_msa_y_B11Y = mod_y_B11C[:, :, :, : self.hidden_size_y]
                gate_msa_y_B11Y = mod_y_B11C[:, :, :, self.hidden_size_y : 2 * self.hidden_size_y]
                scale_mlp_y_B11Y = mod_y_B11C[:, :, :, 2 * self.hidden_size_y : 3 * self.hidden_size_y]
                gate_mlp_y_B11Y = mod_y_B11C[:, :, :, 3 * self.hidden_size_y :]
                # scale_msa_y, gate_msa_y, scale_mlp_y, gate_mlp_y = ttnn.split(mod_y, 4, dim=1)
            else:
                scale_msa_y_B11Y = mod_y_B11C

        # Self-attention block
        x_attn_shard_1BNX, y_attn_shard_1BLY = self.attn(
            x_1BNX,
            y_1BLY,
            scale_x=scale_msa_x_B11X,
            scale_y=scale_msa_y_B11Y,
            rope_cos=rope_cos_1HND,
            rope_sin=rope_sin_1HND,
            trans_mat=trans_mat,
            uncond=uncond,
        )

        if self.num_devices > 1:
            # Collect hidden-dim-fractured attention outputs
            x_attn_1BNX = ttnn.all_gather(x_attn_shard_1BNX, dim=3)
        else:
            x_attn_1BNX = x_attn_shard_1BNX

        assert x_attn_1BNX.shape[2] == N
        x_1BNX = residual_tanh_gated_rmsnorm(x_1BNX, x_attn_1BNX, gate_msa_x_B11X)
        # MLP block
        x_1BNX = self.ff_block_x(x_1BNX, scale_mlp_x_B11X, gate_mlp_x_B11X)

        if not uncond:
            if self.num_devices > 1:
                # Collect hidden-dim-fractured attention outputs
                y_attn_1BLY = ttnn.all_gather(y_attn_shard_1BLY, dim=3)
            else:
                y_attn_1BLY = y_attn_shard_1BLY

            if self.update_y:
                y_1BLY = residual_tanh_gated_rmsnorm(y_1BLY, y_attn_1BLY, gate_msa_y_B11Y)
                y_1BLY = self.ff_block_y(y_1BLY, scale_mlp_y_B11Y, gate_mlp_y_B11Y)

        return x_1BNX, y_1BLY
