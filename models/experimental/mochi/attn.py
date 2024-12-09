import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
import torch
from typing import Tuple, Optional

from models.experimental.mochi.mod_rmsnorm import modulated_rmsnorm


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
        cache_name = lambda name: weight_cache_path / (state_dict_prefix + f".{name}.weight")

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
        cache_name = lambda name: weight_cache_path / (state_dict_prefix + f".{name}.bias")
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
        batch_size, seq_len = y.shape[1], y.shape[2]
        print(f"qkv_y shape: {qkv_y.shape}")
        # NOTE: This reshape is illegal because it's 5D
        # qkv_y = ttnn.reshape(qkv_y, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        # NOTE: This unpack may not work and may lead to strange error messages
        # q_y, k_y, v_y = ttnn.split(qkv_y, 3, dim=2)
        # q_y = ttnn.reshape(
        #     qkv_y[:, :, : self.num_heads * self.head_dim],
        #     (qkv_y.shape[0], qkv_y.shape[1], self.num_heads, self.head_dim),
        # )
        # k_y = ttnn.reshape(
        #     qkv_y[:, :, self.num_heads * self.head_dim : 2 * self.num_heads * self.head_dim],
        #     (qkv_y.shape[0], qkv_y.shape[1], self.num_heads, self.head_dim),
        # )
        # v_y = ttnn.reshape(
        #     qkv_y[:, :, 2 * self.num_heads * self.head_dim :],
        #     (qkv_y.shape[0], qkv_y.shape[1], self.num_heads, self.head_dim),
        # )

        # Use nlp_create_qkv_heads here as well because text will be concatenated on the sequence dimension
        q_y, k_y, v_y = ttnn.experimental.nlp_create_qkv_heads(
            qkv_y,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        max_seqlen_in_batch: int,
        trans_mat: ttnn.Tensor,
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
        B = x.shape[1]
        assert B == 1, f"Batch size must be 1, got {B}"
        assert x.shape[0] == 1, f"x dim0 must be 1, got {x.shape[0]}"
        seq_x = x.shape[2]
        seq_y = y.shape[2]
        # TODO: This assert doesn't do what I want it to do. Should check padded shapes
        # assert seq_x % 1024 == 0, f"Visual sequence length must be divisible by 1024, got {seq_x}"
        assert seq_x + seq_y == max_seqlen_in_batch, f"Padded sequence lengths are not yet supported"
        # Set up compute kernel config for high-fidelity computations
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        x = modulated_rmsnorm(x, scale_x)

        # Process visual features
        # x = ttnn.reshape(x, (1, x.shape[2] // 1024, 1024, x.shape[3]))
        qkv_x = ttnn.linear(
            x,
            self.qkv_x,
            bias=self.qkv_bias_x if self.qkv_bias else None,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # x = ttnn.reshape(x, (1, B, seq_x, x.shape[3]))

        # Split qkv_x into q, k, v
        batch_size, seq_len = x.shape[1], x.shape[2]
        print(f"qkv_x shape: {qkv_x.shape}")
        # Need to get these tensors to shape [B, H, L, D] for rotary embeddings
        q_x, k_x, v_x = ttnn.experimental.nlp_create_qkv_heads(
            qkv_x,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Apply normalization and rotary embeddings to visual features
        q_x = self.q_norm_x(q_x, mode="prefill")
        q_x = ttnn.experimental.rotary_embedding_llama(q_x, rope_cos, rope_sin, trans_mat)
        k_x = self.k_norm_x(k_x, mode="prefill")
        k_x = ttnn.experimental.rotary_embedding_llama(k_x, rope_cos, rope_sin, trans_mat)

        B, num_heads, N, head_dim = q_x.shape
        D = num_heads * head_dim

        # Process text features if present
        if batch_size == 1:
            text_seqlen = max_seqlen_in_batch - N
            if text_seqlen > 0:
                # Process text features
                # TODO: Removing until we support padded sequences
                # y = y[:, :text_seqlen] # Remove padding tokens
                y = modulated_rmsnorm(y, scale_y)
                q_y, k_y, v_y = self.run_qkv_y(y)

                # Concatenate visual and text features on the sequence dimension
                q = ttnn.concat([q_x, q_y], dim=2)
                k = ttnn.concat([k_x, k_y], dim=2)
                v = ttnn.concat([v_x, v_y], dim=2)
            else:
                q, k, v = q_x, k_x, v_x
        else:
            assert False, "Batch size > 1 not supported"

        return q, k, v

    def run_attention(
        self,
        q: ttnn.Tensor,  # (total <= B * (N + L), num_heads, head_dim)
        k: ttnn.Tensor,  # (total <= B * (N + L), num_heads, head_dim)
        v: ttnn.Tensor,  # (total <= B * (N + L), num_heads, head_dim)
        *,
        B: int,
    ) -> ttnn.Tensor:
        """Run attention computation.

        Args:
            q: Query tensor (total <= B * (N + L), num_heads, head_dim)
            k: Key tensor (total <= B * (N + L), num_heads, head_dim)
            v: Value tensor (total <= B * (N + L), num_heads, head_dim)
            B: Batch size (must be 1)

        Returns:
            out: Attention output tensor (total, local_dim)
        """
        assert B == 1, "Only batch size 1 is supported"
        total = q.shape[0]
        local_dim = self.dim_x  # No head parallel support yet
        assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"
        NH = q.shape[1]
        S = q.shape[2]
        DH = q.shape[3]
        assert q.shape[0] == B, "Batch size must be B"

        # Set up compute kernel config for high-fidelity computations
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=128,  # TODO: Make this dynamic
            k_chunk_size=128,
            exp_approx_mode=False,
        )

        # Run attention
        out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )  # (B, num_heads, seq_len, head_dim)

        # Reshape output
        out = ttnn.experimental.nlp_concat_heads(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return out

    def post_attention(
        self,
        out: ttnn.Tensor,  # (total <= B * (N + L), local_dim)
        B: int,
        M: int,
        L: int,
        dtype: ttnn.bfloat16,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Post attention processing to split and project visual and text features.

        Args:
            out: (total <= B * (N + L), local_dim) Combined attention output
            B: Batch size (must be 1)
            M: Number of visual tokens
            L: Number of text tokens
            dtype: Data type of tensors
            valid_token_indices: Indices of valid tokens

        Returns:
            x: (B, M, dim_x) Visual token features
            y: (B, L, dim_y) Text token features
        """
        assert B == 1, "Batch size must be 1"
        assert out.shape[1] == B, "Batch size must be 1"
        N = M  # No context parallel support yet
        local_dim = self.dim_x  # No head parallel support yet

        print(f"B: {B}, M: {M}, L: {L}, N: {N}")

        # Split sequence into visual and text tokens, adding back padding
        if B == 1:
            # out = ttnn.reshape(out, (B, -1, local_dim))
            if out.shape[2] > N:
                # Split into visual and text features
                x = out[:, :, :N, :]  # (B, N, local_dim)
                y = out[:, :, N:, :]  # (B, <=L, local_dim)
                # Pad text features if needed
                print(f"y shape: {y.shape}")
                if y.shape[2] < L:
                    raise ValueError("Not supporting padded text features")
                    # y = ttnn.pad(y, (0, 0, 0, L - y.shape[1], 0, 0))  # (B, L, local_dim)
            else:
                # Empty prompt case
                raise ValueError("Not supporting empty prompt")
                x = out
                y = ttnn.zeros((B, L, local_dim), dtype=dtype, device=self.mesh_device)
        else:
            raise ValueError("Batch size > 1 not supported")

        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Project outputs
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"proj_x shape: {self.proj_x.shape}")
        print(f"proj_bias_x shape: {self.proj_bias_x.shape}")
        x = ttnn.linear(
            x,
            self.proj_x,
            bias=self.proj_bias_x if self.out_bias else None,
            compute_kernel_config=compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if self.update_y:
            y = ttnn.linear(
                y,
                self.proj_y,
                bias=self.proj_bias_y if self.out_bias else None,
                compute_kernel_config=compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=8, x=8),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return x, y

    def forward(
        self,
        x: ttnn.Tensor,
        y: ttnn.Tensor,
        *,
        scale_x: ttnn.Tensor,
        scale_y: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        packed_indices,
    ) -> ttnn.Tensor:
        B, L = y.shape[1], y.shape[2]
        M = x.shape[2]

        q, k, v = self.prepare_qkv(
            x=x,
            y=y,
            scale_x=scale_x,
            scale_y=scale_y,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            max_seqlen_in_batch=packed_indices["max_seqlen_in_batch_kv"],
            trans_mat=trans_mat,
        )

        # Self-attention is expensive, so don't checkpoint it.
        out = self.run_attention(
            q,
            k,
            v,
            B=B,
        )

        x, y = self.post_attention(
            out,
            B=B,
            M=M,
            L=L,
            dtype=out.dtype,
        )

        return x, y
