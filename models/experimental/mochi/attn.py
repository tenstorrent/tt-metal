import ttnn
from models.common.lightweightmodule import LightweightModule
import torch


class TtAsymmetricAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        dim_x: int,
        dim_y: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        update_y: bool = True,
        out_bias: bool = True,
        attention_mode: str = "flash",
        softmax_scale: Optional[float] = None,
        state_dict_prefix=None,
    ):
        super().__init__()
        assert len(mesh_device.get_devices()) == 1, "Only single-device inference is supported for attention layers"

        self.num_heads = num_heads
        self.head_dim = dim_x // num_heads
        self.update_y = update_y
        self.softmax_scale = softmax_scale

        # Define the qkv and output projections
        self.qkv_x = self._create_linear_layer(
            "qkv_x", dim_x, 3 * dim_x, weight_cache_path, state_dict, state_dict_prefix
        )
        self.qkv_y = self._create_linear_layer(
            "qkv_y", dim_y, 3 * dim_x, weight_cache_path, state_dict, state_dict_prefix
        )
        self.proj_x = self._create_linear_layer(
            "proj_x", dim_x, dim_x, weight_cache_path, state_dict, state_dict_prefix
        )
        self.proj_y = (
            self._create_linear_layer("proj_y", dim_x, dim_y, weight_cache_path, state_dict, state_dict_prefix)
            if update_y
            else ttnn.Identity()
        )

    def _create_linear_layer(self, name, in_features, out_features, weight_cache_path, state_dict, state_dict_prefix):
        torch_weight = lambda name: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
        cache_name = lambda name: weight_cache_path / (state_dict_prefix + f".{name}")

        return ttnn.as_tensor(
            torch_weight(name),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name),
        )

    def _prepare_qkv(
        self,
        x: ttnn.Tensor,
        y: ttnn.Tensor,
        *,
        scale_x: ttnn.Tensor,
        scale_y: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        valid_token_indices: ttnn.Tensor,
        max_seqlen_in_batch: int,
    ):
        # Set up compute kernel config for high-fidelity computations
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Compute QKV projections for both x and y
        qkv_x = ttnn.linear(
            x,
            self.qkv_x,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
            memory_config=x.memory_config(),
        )

        qkv_y = ttnn.linear(
            y,
            self.qkv_y,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
            memory_config=y.memory_config(),
        )

        # Split QKV into separate tensors
        # Original shape: [..., seq_len, 3 * dim_x]
        # After split: 3 tensors of shape [..., seq_len, dim_x]
        q_x, k_x, v_x = ttnn.split(qkv_x, 3, dim=-1)
        q_y, k_y, v_y = ttnn.split(qkv_y, 3, dim=-1)

        # Reshape for multi-head attention
        # New shape: [..., seq_len, num_heads, head_dim]
        batch_size = x.shape[0]
        seq_len_x = x.shape[1]
        seq_len_y = y.shape[1]

        def reshape_for_attention(tensor, seq_len):
            return ttnn.reshape(tensor, (batch_size, seq_len, self.num_heads, self.head_dim))

        # Reshape all tensors
        q_x = reshape_for_attention(q_x, seq_len_x)
        k_x = reshape_for_attention(k_x, seq_len_x)
        v_x = reshape_for_attention(v_x, seq_len_x)
        q_y = reshape_for_attention(q_y, seq_len_y)
        k_y = reshape_for_attention(k_y, seq_len_y)
        v_y = reshape_for_attention(v_y, seq_len_y)

        # Apply scaling and rotary embeddings
        q_x = self.q_norm_x(q_x)
        q_x = apply_rotary_emb_qk_real(q_x, rope_cos, rope_sin)
        k_x = self.k_norm_x(k_x)
        k_x = apply_rotary_emb_qk_real(k_x, rope_cos, rope_sin)

        return q_x, k_x, v_x, q_y, k_y, v_y

    def forward(
        self,
        x: ttnn.Tensor,
        y: ttnn.Tensor,
        *,
        scale_x: ttnn.Tensor,
        scale_y: ttnn.Tensor,
        packed_indices: Dict[str, ttnn.Tensor] = None,
        checkpoint_qkv: bool = False,
        checkpoint_post_attn: bool = False,
        **rope_rotation,
    ) -> ttnn.Tensor:
        # Implement the forward pass logic here
        # This will involve computing q, k, v, and then performing attention
        # Use ttnn operations similar to those in TtFeedForward
        pass
