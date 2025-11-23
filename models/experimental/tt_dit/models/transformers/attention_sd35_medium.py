# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
SD3.5 Medium Joint Attention Implementation
Uses fused QKV projections matching MM-DiT reference.
"""

import ttnn
from ...layers.linear import ColParallelLinear, Linear
from ...layers.normalization import RMSNorm


class SD35JointAttention:
    """Joint attention for SD3.5 Medium with spatial and prompt embeddings."""

    def __init__(
        self,
        query_dim,
        head_dim,
        heads,
        out_dim,
        bias=True,
        context_pre_only=False,
        eps=1e-6,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
        padding_config=None,
    ):
        self.query_dim = query_dim
        self.head_dim = head_dim
        self.heads = heads
        self.out_dim = out_dim
        self.context_pre_only = context_pre_only
        self.eps = eps
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.padding_config = padding_config

        self.inner_dim = head_dim * heads
        self.scale = head_dim**-0.5

        # Fused QKV projection for spatial (matching reference)
        self.to_qkv = ColParallelLinear(
            query_dim,
            3 * self.inner_dim,
            bias=bias,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config else None,
        )

        # Fused QKV projection for prompt
        self.add_qkv_proj = ColParallelLinear(
            query_dim,
            3 * self.inner_dim,
            bias=bias,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config else None,
        )

        # RMSNorm for Q and K with learnable scaling
        self.norm_q = RMSNorm(
            embedding_dim=head_dim,
            norm_eps=eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.norm_k = RMSNorm(
            embedding_dim=head_dim,
            norm_eps=eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.norm_added_q = RMSNorm(
            embedding_dim=head_dim,
            norm_eps=eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.norm_added_k = RMSNorm(
            embedding_dim=head_dim,
            norm_eps=eps,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )

        # Output projections
        self.to_out = Linear(
            self.inner_dim,
            out_dim,
            bias=bias,
            mesh_device=mesh_device,
        )

        if not context_pre_only:
            self.to_add_out = Linear(
                self.inner_dim,
                out_dim,
                bias=bias,
                mesh_device=mesh_device,
            )

        # Compute kernel config for N150
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # SDPA config
        self.core_grid = mesh_device.compute_with_storage_grid_size()

    def load_state_dict(self, state_dict):
        from ...utils.substate import substate

        # Load fused QKV projections (no .0 suffix)
        self.to_qkv.load_torch_state_dict(substate(state_dict, "to_qkv"))
        self.add_qkv_proj.load_torch_state_dict(substate(state_dict, "add_qkv_proj"))

        # Load output projections (no .0 suffix)
        self.to_out.load_torch_state_dict(substate(state_dict, "to_out"))

        if not self.context_pre_only:
            self.to_add_out.load_torch_state_dict(substate(state_dict, "to_add_out"))

        # Load RMSNorm layers
        self.norm_q.load_torch_state_dict(substate(state_dict, "norm_q"))
        self.norm_k.load_torch_state_dict(substate(state_dict, "norm_k"))
        self.norm_added_q.load_torch_state_dict(substate(state_dict, "norm_added_q"))
        self.norm_added_k.load_torch_state_dict(substate(state_dict, "norm_added_k"))

    def __call__(self, spatial, prompt, N):
        """
        Args:
            spatial: [1, B, N, D] spatial embeddings
            prompt: [1, B, L, D] prompt embeddings
            N: Sequence length
        Returns:
            spatial_out, prompt_out
        """
        # Fused QKV projection and split for spatial
        qkv = self.to_qkv(spatial)
        qkv = ttnn.reshape(qkv, (1, qkv.shape[1], qkv.shape[2], 3, self.heads, self.head_dim))

        # Extract Q, K, V - shape is [1, B, N, heads, head_dim]
        q = qkv[:, :, :, 0, :, :]
        k = qkv[:, :, :, 1, :, :]
        v = qkv[:, :, :, 2, :, :]

        # Fused QKV projection and split for prompt
        prompt_qkv = self.add_qkv_proj(prompt)
        prompt_qkv = ttnn.reshape(
            prompt_qkv, (1, prompt_qkv.shape[1], prompt_qkv.shape[2], 3, self.heads, self.head_dim)
        )
        prompt_q = prompt_qkv[:, :, :, 0, :, :]
        prompt_k = prompt_qkv[:, :, :, 1, :, :]
        prompt_v = prompt_qkv[:, :, :, 2, :, :]

        # Apply RMSNorm to Q and K (operates on head_dim, last dimension)
        q = self.norm_q(q)
        k = self.norm_k(k)
        prompt_q = self.norm_added_q(prompt_q)
        prompt_k = self.norm_added_k(prompt_k)

        # Now reshape to [B, NH, N/L, DH] format for joint SDPA
        # Remove leading 1 dimension and merge heads dimension
        B = q.shape[1]
        N_spatial = q.shape[2]
        L_prompt = prompt_q.shape[2]

        # Reshape spatial: [1, B, N, heads, head_dim] -> [B, heads, N, head_dim]
        q = ttnn.reshape(q, (B, N_spatial, self.heads, self.head_dim))
        k = ttnn.reshape(k, (B, N_spatial, self.heads, self.head_dim))
        v = ttnn.reshape(v, (B, N_spatial, self.heads, self.head_dim))

        # Reshape prompt: [1, B, L, heads, head_dim] -> [B, heads, L, head_dim]
        prompt_q = ttnn.reshape(prompt_q, (B, L_prompt, self.heads, self.head_dim))
        prompt_k = ttnn.reshape(prompt_k, (B, L_prompt, self.heads, self.head_dim))
        prompt_v = ttnn.reshape(prompt_v, (B, L_prompt, self.heads, self.head_dim))

        # Transpose to [B, heads, N/L, head_dim]
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.permute(v, (0, 2, 1, 3))
        prompt_q = ttnn.permute(prompt_q, (0, 2, 1, 3))
        prompt_k = ttnn.permute(prompt_k, (0, 2, 1, 3))
        prompt_v = ttnn.permute(prompt_v, (0, 2, 1, 3))

        # Joint attention SDPA
        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.core_grid,
            q_chunk_size=64,
            k_chunk_size=64,
        )

        spatial_out, prompt_out = ttnn.transformer.joint_scaled_dot_product_attention(
            q,
            k,
            v,
            prompt_q,
            prompt_k,
            prompt_v,
            joint_strategy="rear",
            program_config=program_config,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Transpose back: [B, heads, N/L, head_dim] -> [B, N/L, heads, head_dim]
        spatial_out = ttnn.permute(spatial_out, (0, 2, 1, 3))
        if prompt_out is not None:
            prompt_out = ttnn.permute(prompt_out, (0, 2, 1, 3))

        # Reshape to [1, B, N/L, inner_dim]
        spatial_out = ttnn.reshape(spatial_out, (1, B, N_spatial, self.inner_dim))
        if prompt_out is not None:
            prompt_out = ttnn.reshape(prompt_out, (1, B, L_prompt, self.inner_dim))

        # Project output
        spatial_out = self.to_out(spatial_out)

        if not self.context_pre_only:
            prompt_out = self.to_add_out(prompt_out)
        else:
            prompt_out = None

        return spatial_out, prompt_out
