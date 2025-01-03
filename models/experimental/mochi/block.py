import ttnn
from models.common.lightweightmodule import LightweightModule
import torch
from typing import Optional, Tuple

from models.experimental.mochi.attn import TtAsymmetricAttention
from models.experimental.mochi.ff import TtFeedForward
from models.experimental.mochi.mod_rmsnorm import modulated_rmsnorm
from models.experimental.mochi.residual_tanh_gated_rmsnorm import residual_tanh_gated_rmsnorm


class TtAsymmetricJointBlock(LightweightModule):
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
        self.mod_x, self.mod_x_bias = self._create_linear_layer(
            "mod_x",
            in_features=hidden_size_x,
            out_features=4 * hidden_size_x,
            weight_cache_path=weight_cache_path,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
        )

        if self.update_y:
            self.mod_y, self.mod_y_bias = self._create_linear_layer(
                "mod_y",
                in_features=hidden_size_x,
                out_features=4 * hidden_size_y,
                weight_cache_path=weight_cache_path,
                state_dict=state_dict,
                state_dict_prefix=state_dict_prefix,
            )
        else:
            self.mod_y, self.mod_y_bias = self._create_linear_layer(
                "mod_y",
                in_features=hidden_size_x,
                out_features=hidden_size_y,
                weight_cache_path=weight_cache_path,
                state_dict=state_dict,
                state_dict_prefix=state_dict_prefix,
            )

        # Self-attention
        self.attn = TtAsymmetricAttention(
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

        # MLP layers using TtFeedForward
        mlp_hidden_dim_x = int(hidden_size_x * mlp_ratio_x)
        self.mlp_x = TtFeedForward(
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
            self.mlp_y = TtFeedForward(
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

    def _create_linear_layer(
        self, name, in_features, out_features, weight_cache_path, state_dict, state_dict_prefix, bias=True
    ):
        """Create a linear layer with both weights and biases.

        Args:
            name: Name of the layer (e.g. 'mod_x', 'mod_y')
            in_features: Input dimension
            out_features: Output dimension
            weight_cache_path: Path to cache weights
            state_dict: State dict containing weights and biases
            state_dict_prefix: Prefix for state dict keys

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor]: Weight and bias tensors
        """
        # Get weight and transpose it
        weight_key = f"{state_dict_prefix}.{name}.weight"
        weight = torch.transpose(state_dict[weight_key], -2, -1)

        # Get bias if it exists
        bias_key = f"{state_dict_prefix}.{name}.bias"
        bias_pt = state_dict.get(bias_key)  # Returns None if key doesn't exist
        # Check that bias exists if bias=True is specified
        if bias and bias_pt is None:
            raise ValueError(f"Bias was specified but not found in state dict for {name} layer")
        # Create weight tensor
        weight_tensor = ttnn.as_tensor(
            weight,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            # mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=weight_cache_path / (state_dict_prefix + f".{name}.weight"),
        )

        # Create bias tensor if it exists
        bias_tensor = None
        if bias:
            bias_tensor = ttnn.as_tensor(
                bias_pt,
                dtype=ttnn.bfloat16,
                device=self.mesh_device,
                # mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=weight_cache_path / (state_dict_prefix + f".{name}.bias"),
            )

        return weight_tensor, bias_tensor

    def ff_block_x(self, x: ttnn.Tensor, scale_x: ttnn.Tensor, gate_x: ttnn.Tensor) -> ttnn.Tensor:
        """Feed-forward block for visual features."""
        x_mod = modulated_rmsnorm(x, scale_x)
        x_res_shard = self.mlp_x(x_mod)
        if self.num_devices > 1:
            # Collect hidden-dim-fractured MLP outputs
            x_res = ttnn.all_gather(x_res_shard, dim=3)
        else:
            x_res = x_res_shard
        x = residual_tanh_gated_rmsnorm(x, x_res, gate_x)
        return x

    def ff_block_y(self, y: ttnn.Tensor, scale_y: ttnn.Tensor, gate_y: ttnn.Tensor) -> ttnn.Tensor:
        """Feed-forward block for text features."""
        y_mod = modulated_rmsnorm(y, scale_y)
        y_res_shard = self.mlp_y(y_mod)
        if self.num_devices > 1:
            # Collect hidden-dim-fractured MLP outputs
            y_res = ttnn.all_gather(y_res_shard, dim=3)
        else:
            y_res = y_res_shard
        y = residual_tanh_gated_rmsnorm(y, y_res, gate_y)
        return y

    def forward(
        self,
        x: ttnn.Tensor,
        c: ttnn.Tensor,
        y: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        packed_indices=None,
        attn_mask=None,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Forward pass of a block.

        Args:
            x: (1, B, N, dim) tensor of visual tokens
            c: (1, 1, B, dim) tensor of conditioned features
            y: (1, B, L, dim) tensor of text tokens
            **attn_kwargs: Additional arguments passed to attention layer

        Returns:
            x: (B, N, dim) tensor of visual tokens after block
            y: (B, L, dim) tensor of text tokens after block
        """
        N = x.shape[2]

        # Set up compute kernel config for high-fidelity computations
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Apply modulation
        c = ttnn.silu(c)

        # Apply linear layers with bias
        mod_x = ttnn.linear(
            c,
            self.mod_x,
            bias=self.mod_x_bias,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        print(f"mod_x: {mod_x.shape}")
        scale_msa_x = mod_x[:, :, :, : self.hidden_size_x]
        gate_msa_x = mod_x[:, :, :, self.hidden_size_x : 2 * self.hidden_size_x]
        scale_mlp_x = mod_x[:, :, :, 2 * self.hidden_size_x : 3 * self.hidden_size_x]
        gate_mlp_x = mod_x[:, :, :, 3 * self.hidden_size_x :]
        # scale_msa_x, gate_msa_x, scale_mlp_x, gate_mlp_x = ttnn.split(mod_x, 4, dim=1)

        mod_y = ttnn.linear(
            c,
            self.mod_y,
            bias=self.mod_y_bias,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if self.update_y:
            scale_msa_y = mod_y[:, :, :, : self.hidden_size_y]
            gate_msa_y = mod_y[:, :, :, self.hidden_size_y : 2 * self.hidden_size_y]
            scale_mlp_y = mod_y[:, :, :, 2 * self.hidden_size_y : 3 * self.hidden_size_y]
            gate_mlp_y = mod_y[:, :, :, 3 * self.hidden_size_y :]
            # scale_msa_y, gate_msa_y, scale_mlp_y, gate_mlp_y = ttnn.split(mod_y, 4, dim=1)
        else:
            scale_msa_y = mod_y

        print(f"x: {x.shape}")
        print(f"y: {y.shape}")

        # Self-attention block
        x_attn_shard, y_attn_shard = self.attn(
            x,
            y,
            scale_x=scale_msa_x,
            scale_y=scale_msa_y,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
            packed_indices=packed_indices,
            attn_mask=attn_mask,
        )

        if self.num_devices > 1:
            # Collect hidden-dim-fractured attention outputs
            x_attn = ttnn.all_gather(x_attn_shard, dim=3)
            y_attn = ttnn.all_gather(y_attn_shard, dim=3)
        else:
            x_attn = x_attn_shard
            y_attn = y_attn_shard

        assert x_attn.shape[2] == N
        x = residual_tanh_gated_rmsnorm(x, x_attn, gate_msa_x)

        if self.update_y:
            y = residual_tanh_gated_rmsnorm(y, y_attn, gate_msa_y)

        # MLP block
        x = self.ff_block_x(x, scale_mlp_x, gate_mlp_x)
        if self.update_y:
            y = self.ff_block_y(y, scale_mlp_y, gate_mlp_y)

        return x, y
