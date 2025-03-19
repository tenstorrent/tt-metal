import torch
from typing import Tuple
from functools import partial
import math

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm

from models.experimental.mochi.tt.dit.norms import modulated_rmsnorm
from models.experimental.mochi.tt.common import (
    as_sharded_tensor,
    as_replicated_tensor,
    col_parallel_linear,
    load_linear,
    matmul_config,
    get_padded_vision_seq_len,
)
from functools import partial
from ttnn import ConcatMeshToTensor


class AsymmetricAttention(LightweightModule):
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
        # Assert that all LoRA ranks are 0 since we don't support LoRA in this implementation
        assert qkv_proj_lora_rank == 0, "LoRA not supported - qkv_proj_lora_rank must be 0"
        assert qkv_proj_lora_alpha == 0, "LoRA not supported - qkv_proj_lora_alpha must be 0"
        assert qkv_proj_lora_dropout == 0.0, "LoRA not supported - qkv_proj_lora_dropout must be 0.0"
        assert out_proj_lora_rank == 0, "LoRA not supported - out_proj_lora_rank must be 0"
        assert out_proj_lora_alpha == 0, "LoRA not supported - out_proj_lora_alpha must be 0"
        assert out_proj_lora_dropout == 0.0, "LoRA not supported - out_proj_lora_dropout must be 0.0"

        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.state_dict = state_dict
        self.state_dict_prefix = state_dict_prefix
        self.weight_cache_path = weight_cache_path
        self.layer_num = layer_num
        self.dtype = dtype
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_heads = num_heads
        self.n_local_heads = num_heads // self.num_devices
        self.head_dim = dim_x // num_heads
        self.update_y = update_y
        self.softmax_scale = softmax_scale
        if dim_x % num_heads != 0:
            raise ValueError(f"dim_x={dim_x} should be divisible by num_heads={num_heads}")

        # Input layers.
        self.qkv_bias = qkv_bias
        self.out_bias = out_bias

        # Define the qkv and output projections
        self.qkv_x, self.qkv_x_bias = self._col_parallel_qkv(
            "qkv_x", self.qkv_bias, weight_cache_path, state_dict, state_dict_prefix
        )
        self.qkv_y, self.qkv_y_bias = self._col_parallel_qkv(
            "qkv_y", self.qkv_bias, weight_cache_path, state_dict, state_dict_prefix
        )

        self.proj_x, self.proj_x_bias = load_linear(
            "proj_x", self.out_bias, weight_cache_path, state_dict, state_dict_prefix, self.mesh_device
        )
        if update_y:
            self.proj_y, self.proj_y_bias = col_parallel_linear(
                "proj_y", self.out_bias, weight_cache_path, state_dict, state_dict_prefix, self.mesh_device
            )

        # Query and key normalization for stability.
        assert qk_norm
        self.q_norm_x = self._create_rmsmorn(".q_norm_x")
        self.k_norm_x = self._create_rmsmorn(".k_norm_x")
        self.q_norm_y = self._create_rmsmorn(".q_norm_y")
        self.k_norm_y = self._create_rmsmorn(".k_norm_y")

        # TODO: using qkv_x program config leads to worse PCC
        self.qkv_x_config = partial(matmul_config, k=dim_x, n=3 * self.num_heads * self.head_dim, grid_size=(8, 8))
        self.proj_x_config = partial(matmul_config, k=dim_x, n=dim_x, in0_block_w=4, grid_size=(8, 8))
        self.mm_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _create_rmsmorn(self, key):
        return RMSNorm(
            device=self.mesh_device,
            dim=self.head_dim,
            state_dict=self.state_dict,
            state_dict_prefix=self.state_dict_prefix,
            weight_cache_path=self.weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key=key,
        )

    def _col_parallel_qkv(self, name, bias, weight_cache_path, state_dict, state_dict_prefix):
        """
        Shuffle QKV weights to group heads such that they can be column parallel
        """
        torch_weight = lambda name, suffix: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.{suffix}"], -2, -1)
        w = torch_weight(name, "weight")
        b = torch_weight(name, "bias") if bias else None

        def shuffle_heads(tensor):
            # Given torch tensor with output features in the last dimension,
            # shuffle heads to allow for column parallel computation
            in_dim = tensor.shape[0]
            tensor = tensor.reshape(in_dim, 3, self.num_devices, self.n_local_heads, -1)  # [ID, 3, ND, NLH, DH]
            tensor = tensor.permute(0, 2, 1, 3, 4)  # [ID, ND, 3, NLH, DH]
            tensor = tensor.reshape(in_dim, -1)  # [ID, ND*3*NLH*DH]
            return tensor

        w = as_sharded_tensor(
            shuffle_heads(w),
            self.mesh_device,
            dim=-1,
            cache_file_name=weight_cache_path / (state_dict_prefix + f".{name}.weight"),
        )
        if b is not None:
            b = as_sharded_tensor(
                shuffle_heads(b).reshape(1, -1),
                self.mesh_device,
                dim=-1,
                cache_file_name=weight_cache_path / (state_dict_prefix + f".{name}.bias"),
            )
        return w, b

    def _load_qkv(self, name, bias, weight_cache_path, state_dict, state_dict_prefix):
        # Weight is fractured for FSDP, bias is not (it's small)
        torch_weight = lambda name, suffix: torch.transpose(state_dict[f"{state_dict_prefix}.{name}.{suffix}"], -2, -1)
        w = torch_weight(name, "weight")
        b = torch_weight(name, "bias") if bias else None

        w = as_sharded_tensor(
            w,
            mesh_device=self.mesh_device,
            dim=-1,
            cache_file_name=weight_cache_path / (state_dict_prefix + f".{name}.weight"),
        )
        if b is not None:
            b = as_replicated_tensor(
                b.reshape(1, -1),
                mesh_device=self.mesh_device,
                cache_file_name=weight_cache_path / (state_dict_prefix + f".{name}.bias"),
            )
        return w, b

    def _seq_to_col_parallel_tensor(self, seq_parallel_tensor, N):
        dim = seq_parallel_tensor.shape[3]
        local_dim = dim // self.num_devices
        replicated_tensor = ttnn.all_gather(seq_parallel_tensor, dim=2)
        tensors = ttnn.get_device_tensors(replicated_tensor)
        tile_padded_seqlen = math.ceil(N / 32) * 32
        for i in range(len(tensors)):
            # Slice out local head and slice sequence padding
            tensors[i] = tensors[i][:, :, :tile_padded_seqlen, i * local_dim : (i + 1) * local_dim]
            # Add padding information
            tensors[i] = ttnn.reshape(
                tensors[i], [tensors[i].shape[0], tensors[i].shape[1], N, tensors[i].shape[3]], tensors[i].shape
            )
        return ttnn.aggregate_as_tensor(tensors)

    def _col_to_seq_parallel_tensor(self, col_parallel_tensor, N):
        padded_seq_len = get_padded_vision_seq_len(N, self.num_devices)
        if padded_seq_len > N:
            col_parallel_tensor = ttnn.reshape(
                col_parallel_tensor,
                col_parallel_tensor.padded_shape,
            )
            padded_N = col_parallel_tensor.shape[2]

            if padded_seq_len > padded_N:
                col_parallel_tensor = ttnn.pad(col_parallel_tensor, [(0, 0), (0, padded_seq_len - padded_N)], 0.0)
        replicated_tensor = ttnn.all_gather(col_parallel_tensor, dim=3)
        tensors = ttnn.get_device_tensors(replicated_tensor)

        padded_M = padded_seq_len // self.num_devices
        for i in range(len(tensors)):
            tensors[i] = tensors[i][:, :, i * padded_M : (i + 1) * padded_M]
        return ttnn.aggregate_as_tensor(tensors)

    def run_qkv_y(self, y):
        # Compute QKV projection
        qkv_y = ttnn.linear(
            y,
            self.qkv_y,
            bias=self.qkv_y_bias,
            compute_kernel_config=self.mm_compute_kernel_config,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Use nlp_create_qkv_heads here as well because text will be concatenated on the sequence dimension
        q_y, k_y, v_y = ttnn.experimental.nlp_create_qkv_heads(
            qkv_y,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Apply normalization
        q_y = self.q_norm_y(q_y, mode="prefill")
        k_y = self.k_norm_y(k_y, mode="prefill")
        return q_y, k_y, v_y

    def prepare_qkv(
        self,
        x_1BNX: ttnn.Tensor,  # (B, M, dim_x)
        y_1BLY: ttnn.Tensor,  # (B, L, dim_y)
        *,
        N: int,
        scale_x: ttnn.Tensor,  # TODO: add shape metadata
        scale_y: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        uncond: bool = False,
    ):
        """Prepare QKV tensors for attention computation.

        Args:
            x: Visual token features
            y: Text token features
            scale_x: Modulation for visual features
            scale_y: Modulation for text features
            rope_cos: Cosine component for rotary position embedding
            rope_sin: Sine component for rotary position embedding
            trans_mat: Transformation matrix for rotary embeddings
            uncond: Whether to run unconditional attention

        Returns:
            q, k, v: Query, key and value tensors prepared for attention
            joint_q, joint_k, joint_v: Query, key and value tensors prepared for joint attention
        """
        B, M = x_1BNX.shape[1], x_1BNX.shape[2]
        assert B == 1, f"Batch size must be 1, got {B}"
        assert x_1BNX.shape[0] == 1, f"x dim0 must be 1, got {x_1BNX.shape[0]}"
        # NOTE: I removed this check because with unpadded input reshape fails
        # assert (
        #     seq_x % self.X_MM_SEQ_LEN == 0
        # ), f"Visual sequence length must be divisible by {self.X_MM_SEQ_LEN}, got {seq_x}"
        x_1BNX = modulated_rmsnorm(x_1BNX, scale_x)

        # Process visual features
        qkv_x = ttnn.all_gather(self.qkv_x, dim=-1)
        qkv_x_1BNE = ttnn.linear(
            x_1BNX,
            qkv_x,
            bias=self.qkv_x_bias,
            compute_kernel_config=self.mm_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.qkv_x_config(m=M),
        )

        qkv_x_1BNE = self._seq_to_col_parallel_tensor(qkv_x_1BNE, N=N)

        # Split qkv_x into q, k, v
        # Need to get these tensors to shape [B, H, L, D] for rotary embeddings
        q_x_BHND, k_x_BHND, v_x_BHND = ttnn.experimental.nlp_create_qkv_heads(
            qkv_x_1BNE,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Apply normalization and rotary embeddings to visual features
        q_x_BHND = self.q_norm_x(q_x_BHND, mode="prefill")
        q_x_BHND = ttnn.experimental.rotary_embedding_llama(q_x_BHND, rope_cos, rope_sin, trans_mat)
        k_x_BHND = self.k_norm_x(k_x_BHND, mode="prefill")
        k_x_BHND = ttnn.experimental.rotary_embedding_llama(k_x_BHND, rope_cos, rope_sin, trans_mat)

        # Process text features if present
        if B == 1:
            if not uncond:
                # Process text features
                # TODO: Removing until we support padded sequences
                y_1BLY = modulated_rmsnorm(y_1BLY, scale_y)
                q_y_BHND, k_y_BHND, v_y_BHND = self.run_qkv_y(y_1BLY)
            else:
                q_y_BHND, k_y_BHND, v_y_BHND = None, None, None
        else:
            assert False, "Batch size > 1 not supported"

        return q_x_BHND, k_x_BHND, v_x_BHND, q_y_BHND, k_y_BHND, v_y_BHND

    def run_attention(
        self,
        q_x_BHND: ttnn.Tensor,
        k_x_BHND: ttnn.Tensor,
        v_x_BHND: ttnn.Tensor,
        q_y_BHLD: ttnn.Tensor,
        k_y_BHLD: ttnn.Tensor,
        v_y_BHLD: ttnn.Tensor,
        *,
        B: int,
    ) -> ttnn.Tensor:
        """Run attention computation.

        Args:
            q_x_BHND: Query tensor (B, H, N, D)
            k_x_BHND: Key tensor (B, H, N, D)
            v_x_BHND: Value tensor (B, H, N, D)
            q_y_BHLD: Joint query tensor (B, H, L, D)
            k_y_BHLD: Joint key tensor (B, H, L, D)
            v_y_BHLD: Joint value tensor (B, H, L, D)
            B: Batch size (must be 1)

        Returns:
            out_1BNX: Attention output tensor (1, B, N, X)
            out_1BLX: Joint attention output tensor (1, B, L, X)
        """
        assert B == 1, "Only batch size 1 is supported"
        assert q_x_BHND.shape[0] == B, "Batch size must be B"
        assert q_x_BHND.shape == k_x_BHND.shape == v_x_BHND.shape, "q_x, k_x, v_x must have the same shape"
        is_joint = q_y_BHLD is not None
        if is_joint:
            assert q_y_BHLD.shape == k_y_BHLD.shape == v_y_BHLD.shape, "q_y, k_y, v_y must have the same shape"
        # Set up compute kernel config for high-fidelity computations
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=256,  # TODO: Make this dynamic
            k_chunk_size=512,
            exp_approx_mode=False,
        )
        # Run attention
        if is_joint:
            out_BHND, out_joint_BHLD = ttnn.transformer.joint_scaled_dot_product_attention(
                q_x_BHND,
                k_x_BHND,
                v_x_BHND,
                q_y_BHLD,
                k_y_BHLD,
                v_y_BHLD,
                joint_strategy="rear",
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )
            out_joint_1BLX = ttnn.experimental.nlp_concat_heads(out_joint_BHLD, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            out_BHND = ttnn.transformer.scaled_dot_product_attention(
                q_x_BHND,
                k_x_BHND,
                v_x_BHND,
                is_causal=False,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
            )
            out_joint_1BLX = None

        # Reshape output
        out_1BNX = ttnn.experimental.nlp_concat_heads(out_BHND, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        return out_1BNX, out_joint_1BLX

    def post_attention(
        self,
        out_1BNX: ttnn.Tensor,
        out_joint_1BLX: ttnn.Tensor,
        B: int,
        N: int,
        dtype: ttnn.bfloat16,
        uncond: bool = False,
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
        assert out_1BNX.shape[1] == B, "Batch size must be 1"
        assert (out_joint_1BLX is None) == uncond

        if self.num_devices > 1:
            out_1BNX = self._col_to_seq_parallel_tensor(out_1BNX, N=N)

        M = out_1BNX.shape[2]

        # BUG: This linear clobbers `out_joint_1BLX` if padded and certain program configs are used
        proj_x = ttnn.all_gather(self.proj_x, dim=-1)
        out_1BNX = ttnn.linear(
            out_1BNX,
            proj_x,
            bias=self.proj_x_bias,
            compute_kernel_config=self.mm_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.proj_x_config(m=M),
        )

        out_joint = None  # Default None if uncond
        if not uncond:
            if self.update_y:
                if self.num_devices > 1:
                    out_joint_1BLX = ttnn.all_gather(
                        out_joint_1BLX,
                        dim=3,
                    )
                out_joint_1BLY = ttnn.linear(
                    out_joint_1BLX,
                    self.proj_y,
                    bias=self.proj_y_bias,
                    compute_kernel_config=self.mm_compute_kernel_config,
                    core_grid=ttnn.CoreGrid(y=8, x=8),
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                out_joint = out_joint_1BLY
            else:
                out_joint = out_joint_1BLX

        return out_1BNX, out_joint

    def forward(
        self,
        x_1BNX: ttnn.Tensor,
        y_1BLY: ttnn.Tensor,
        *,
        N: int,
        scale_x: ttnn.Tensor,
        scale_y: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        uncond: bool = False,
    ) -> ttnn.Tensor:
        # input is replicated
        B = x_1BNX.shape[1]

        # output is head-sharded q, k, v
        q_x_BHND, k_x_BHND, v_x_BHND, q_y_BHLD, k_y_BHLD, v_y_BHLD = self.prepare_qkv(
            x_1BNX,
            y_1BLY,
            N=N,
            scale_x=scale_x,
            scale_y=scale_y,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
            uncond=uncond,
        )

        # output is col-sharded
        out_1BNX, out_joint_1BLX = self.run_attention(
            q_x_BHND,
            k_x_BHND,
            v_x_BHND,
            q_y_BHLD,
            k_y_BHLD,
            v_y_BHLD,
            B=B,
        )

        # output is col-sharded x, y
        x_1BNX, y_1BLY = self.post_attention(
            out_1BNX,
            out_joint_1BLX,
            B=B,
            N=N,
            dtype=out_1BNX.dtype,
            uncond=uncond,
        )

        return x_1BNX, y_1BLY
