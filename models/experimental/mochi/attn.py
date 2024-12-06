import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
import torch


class TtAsymmetricAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
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
        softmax_scale=None,
        # Disable LoRA by default ...
        qkv_proj_lora_rank: int = 0,
        qkv_proj_lora_alpha: int = 0,
        qkv_proj_lora_dropout: float = 0.0,
        out_proj_lora_rank: int = 0,
        out_proj_lora_alpha: int = 0,
        out_proj_lora_dropout: float = 0.0,
    ):
        super().__init__()
        assert len(mesh_device.get_devices()) == 1, "Only single-device inference is supported for attention layers"
        # Assert that all LoRA ranks are 0 since we don't support LoRA in this implementation
        assert qkv_proj_lora_rank == 0, "LoRA not supported - qkv_proj_lora_rank must be 0"
        assert qkv_proj_lora_alpha == 0, "LoRA not supported - qkv_proj_lora_alpha must be 0"
        assert qkv_proj_lora_dropout == 0.0, "LoRA not supported - qkv_proj_lora_dropout must be 0.0"
        assert out_proj_lora_rank == 0, "LoRA not supported - out_proj_lora_rank must be 0"
        assert out_proj_lora_alpha == 0, "LoRA not supported - out_proj_lora_alpha must be 0"
        assert out_proj_lora_dropout == 0.0, "LoRA not supported - out_proj_lora_dropout must be 0.0"

        self.mesh_device = mesh_device
        self.state_dict = state_dict
        self.state_dict_prefix = state_dict_prefix
        self.weight_cache_path = weight_cache_path
        self.layer_num = layer_num
        self.dtype = dtype

        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_heads = num_heads
        self.head_dim = dim_x // num_heads
        self.update_y = update_y
        self.softmax_scale = softmax_scale
        if dim_x % num_heads != 0:
            raise ValueError(f"dim_x={dim_x} should be divisible by num_heads={num_heads}")

        # Input layers.
        self.qkv_bias = qkv_bias
        self.out_bias = out_bias

        # Define the qkv and output projections
        self.qkv_x = self._create_linear_layer(
            "qkv_x", dim_x, 3 * dim_x, weight_cache_path, state_dict, state_dict_prefix
        )
        if self.qkv_bias:
            self.qkv_bias_x = self._create_bias_layer("qkv_x", dim_x, weight_cache_path, state_dict, state_dict_prefix)
        self.qkv_y = self._create_linear_layer(
            "qkv_y", dim_y, 3 * dim_x, weight_cache_path, state_dict, state_dict_prefix
        )
        if self.qkv_bias:
            self.qkv_bias_y = self._create_bias_layer("qkv_y", dim_y, weight_cache_path, state_dict, state_dict_prefix)
        self.proj_x = self._create_linear_layer(
            "proj_x", dim_x, dim_x, weight_cache_path, state_dict, state_dict_prefix
        )
        if self.out_bias:
            self.proj_bias_x = self._create_bias_layer(
                "proj_x", dim_x, weight_cache_path, state_dict, state_dict_prefix
            )
        self.proj_y = (
            self._create_linear_layer("proj_y", dim_x, dim_y, weight_cache_path, state_dict, state_dict_prefix)
            if update_y
            else None
        )
        if self.out_bias:
            self.proj_bias_y = self._create_bias_layer(
                "proj_y", dim_y, weight_cache_path, state_dict, state_dict_prefix
            )

        # Query and key normalization for stability.
        assert qk_norm
        self.q_norm_x = RMSNorm(
            device=mesh_device,
            dim=self.head_dim,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key=".q_norm_x",
        )
        self.k_norm_x = RMSNorm(
            device=mesh_device,
            dim=self.head_dim,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key=".k_norm_x",
        )
        self.q_norm_y = RMSNorm(
            device=mesh_device,
            dim=self.head_dim,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key=".q_norm_y",
        )
        self.k_norm_y = RMSNorm(
            device=mesh_device,
            dim=self.head_dim,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            weight_cache_path=weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key=".k_norm_y",
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

    def _create_bias_layer(self, name, dim, weight_cache_path, state_dict, state_dict_prefix):
        torch_bias = lambda name: state_dict[f"{state_dict_prefix}.{name}.bias"]
        cache_name = lambda name: weight_cache_path / (state_dict_prefix + f".{name}")

        return ttnn.as_tensor(
            torch_bias(name),
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_name(name),
        )

    def run_qkv_y(self, y):
        # TODO: Go head parallel
        # Set up compute kernel config for high-fidelity computations
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Compute QKV projection
        qkv_y = ttnn.linear(
            y,
            self.qkv_y,
            bias=self.qkv_bias_y if self.qkv_bias else None,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Reshape and split QKV
        batch_size, seq_len = y.shape[0], y.shape[1]
        # NOTE: This reshape is illegal because it's 5D
        # qkv_y = ttnn.reshape(qkv_y, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        # NOTE: This unpack may not work and may lead to strange error messages
        # q_y, k_y, v_y = ttnn.split(qkv_y, 3, dim=2)
        q_y = ttnn.reshape(
            qkv_y[:, :, : self.num_heads * self.head_dim],
            (qkv_y.shape[0], qkv_y.shape[1], self.num_heads, self.head_dim),
        )
        k_y = ttnn.reshape(
            qkv_y[:, :, self.num_heads * self.head_dim : 2 * self.num_heads * self.head_dim],
            (qkv_y.shape[0], qkv_y.shape[1], self.num_heads, self.head_dim),
        )
        v_y = ttnn.reshape(
            qkv_y[:, :, 2 * self.num_heads * self.head_dim :],
            (qkv_y.shape[0], qkv_y.shape[1], self.num_heads, self.head_dim),
        )

        # Apply normalization
        q_y = self.q_norm_y(q_y, mode="prefill")
        k_y = self.k_norm_y(k_y, mode="prefill")

        return q_y, k_y, v_y

    def prepare_qkv(
        self,
        x: ttnn.Tensor,  # (B, M, dim_x)
        y: ttnn.Tensor,  # (B, L, dim_y)
        *,
        scale_x: ttnn.Tensor,
        scale_y: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        valid_token_indices: ttnn.Tensor,
        max_seqlen_in_batch: int,
    ):
        """Prepare QKV tensors for attention computation.

        Args:
            x: Visual token features
            y: Text token features
            scale_x: Modulation for visual features
            scale_y: Modulation for text features
            rope_cos: Cosine component for rotary position embedding
            rope_sin: Sine component for rotary position embedding
            valid_token_indices: Indices of valid tokens
            max_seqlen_in_batch: Maximum sequence length in batch

        Returns:
            q, k, v: Query, key and value tensors prepared for attention
        """
        # Set up compute kernel config for high-fidelity computations
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Process visual features
        qkv_x = ttnn.linear(
            x,
            self.qkv_x,
            bias=self.qkv_bias_x if self.qkv_bias else None,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Split qkv_x into q, k, v
        batch_size, seq_len = x.shape[0], x.shape[1]
        q_x = ttnn.reshape(
            qkv_x[:, :, : self.num_heads * self.head_dim], (batch_size, seq_len, self.num_heads, self.head_dim)
        )
        k_x = ttnn.reshape(
            qkv_x[:, :, self.num_heads * self.head_dim : 2 * self.num_heads * self.head_dim],
            (batch_size, seq_len, self.num_heads, self.head_dim),
        )
        v_x = ttnn.reshape(
            qkv_x[:, :, 2 * self.num_heads * self.head_dim :], (batch_size, seq_len, self.num_heads, self.head_dim)
        )

        # Apply normalization and rotary embeddings to visual features
        q_x = self.q_norm_x(q_x, mode="prefill")
        k_x = self.k_norm_x(k_x, mode="prefill")

        # Apply rotary embeddings
        q_x = ttnn.apply_rotary_emb(q_x, rope_cos, rope_sin)
        k_x = ttnn.apply_rotary_emb(k_x, rope_cos, rope_sin)

        # Process text features if present
        if batch_size == 1:
            text_seqlen = max_seqlen_in_batch - seq_len
            if text_seqlen > 0:
                # Process text features
                y = y[:, :text_seqlen]
                q_y, k_y, v_y = self.run_qkv_y(y)

                # Concatenate visual and text features
                q = ttnn.concat([q_x, q_y], dim=1)
                k = ttnn.concat([k_x, k_y], dim=1)
                v = ttnn.concat([v_x, v_y], dim=1)
            else:
                q, k, v = q_x, k_x, v_x
        else:
            # Process text features
            q_y, k_y, v_y = self.run_qkv_y(y)

            # Concatenate and gather using indices
            D = self.num_heads * self.head_dim
            indices = valid_token_indices[:, None].expand(-1, D)

            q = ttnn.concat([q_x, q_y], dim=1)
            k = ttnn.concat([k_x, k_y], dim=1)
            v = ttnn.concat([v_x, v_y], dim=1)

            q = ttnn.gather(ttnn.reshape(q, (-1, D)), indices)
            k = ttnn.gather(ttnn.reshape(k, (-1, D)), indices)
            v = ttnn.gather(ttnn.reshape(v, (-1, D)), indices)

        # Reshape for attention
        q = ttnn.reshape(q, (-1, self.num_heads, self.head_dim))
        k = ttnn.reshape(k, (-1, self.num_heads, self.head_dim))
        v = ttnn.reshape(v, (-1, self.num_heads, self.head_dim))

        return q, k, v

    def forward(
        self,
        x: ttnn.Tensor,
        y: ttnn.Tensor,
        *,
        scale_x: ttnn.Tensor,
        scale_y: ttnn.Tensor,
        packed_indices=None,
        checkpoint_qkv: bool = False,
        checkpoint_post_attn: bool = False,
        **rope_rotation,
    ) -> ttnn.Tensor:
        # Implement the forward pass logic here
        # This will involve computing q, k, v, and then performing attention
        # Use ttnn operations similar to those in TtFeedForward
        pass
