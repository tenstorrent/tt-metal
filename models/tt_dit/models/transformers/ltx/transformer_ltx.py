# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 Transformer Block and Model for tt_dit.

Mirrors WanTransformerBlock/WanTransformer3DModel with LTX-2 defaults:
- dim=4096 (32 heads x 128)
- 48 layers
- RMSNorm (not LayerNorm) for all pre-attention norms
- GELU approximate (tanh) activation in feedforward

Reference: LTX-2 transformer.py + model.py + Wan transformer_wan.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn

from ....layers.embeddings import LTXAdaLayerNormSingle
from ....layers.feedforward import ParallelFeedForward
from ....layers.linear import ColParallelLinear, Linear
from ....layers.module import Module, ModuleList, Parameter
from ....layers.normalization import DistributedLayerNorm, DistributedRMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils.substate import pop_substate, rename_substate
from .attention_ltx import LTXAttention

if TYPE_CHECKING:
    pass


class LTXTransformerBlock(Module):
    """
    Single LTX-2 DiT transformer block.

    Structure:
    1. RMSNorm + AdaLN(shift, scale) → self-attention + gate residual
    2. RMSNorm → cross-attention + residual
    3. RMSNorm + AdaLN(shift, scale) → feedforward + gate residual
    """

    def __init__(
        self,
        *,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attention_dim: int | None = None,
        eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.eps = eps

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        # LTX-2 uses RMSNorm for all pre-attention norms (not LayerNorm)
        rms_norm_kwargs = {
            "norm_eps": eps,
            "norm_elementwise_affine": False,
            "bias": False,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
            "mesh_device": mesh_device,
            "ccl_manager": ccl_manager,
        }

        self.norm1 = DistributedRMSNorm(embedding_dim=dim, **rms_norm_kwargs)

        self.attn1 = LTXAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            is_self=True,
        )

        self.attn2 = LTXAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            is_self=False,
            context_dim=cross_attention_dim,
        )

        # Cross-attention norm: functional RMSNorm (no learned weight — LTX-2 uses rms_norm())
        self.norm2 = DistributedRMSNorm(
            embedding_dim=dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.ffn = ParallelFeedForward(
            dim,
            inner_dim=ffn_dim,
            activation_fn="gelu_tanh",
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
        )

        self.norm3 = DistributedRMSNorm(embedding_dim=dim, **rms_norm_kwargs)

        # AdaLN scale_shift_table: 9 modulation params
        # (shift, scale, gate for self-attn) + (shift, scale, gate for FF) + (shift, scale, gate for cross-attn)
        self.scale_shift_table = Parameter(
            total_shape=[1, 1, 9, dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
            device=mesh_device,
            dtype=ttnn.float32,
        )
        # Prompt context modulation (2 params: shift_kv, scale_kv)
        self.prompt_scale_shift_table = Parameter(
            total_shape=[1, 1, 2, dim],
            mesh_axes=[None, None, None, None],
            device=mesh_device,
            dtype=ttnn.float32,
        )

        self.ff_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Rename ff keys: LTX-2 uses "ff.net.0.proj" / "ff.net.2", tt_dit uses "ffn.ff1" / "ffn.ff2"
        rename_substate(state, "ff.net.0.proj", "ffn.ff1")
        rename_substate(state, "ff.net.2", "ffn.ff2")

        for key in ["scale_shift_table", "prompt_scale_shift_table"]:
            if key in state:
                state[key] = state[key].unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        prompt_1BLP: ttnn.Tensor,
        temb_1BTD: ttnn.Tensor,
        N: int,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor | None,
        prompt_temb: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        temb_1BTD: (1, B, 9, D) — 9 modulation params from AdaLayerNormSingle
        prompt_temb: (1, B, 2, D) — prompt context modulation (optional)
        """
        shifted_temb = self.scale_shift_table.data + temb_1BTD
        (shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff, shift_ca, scale_ca, gate_ca) = ttnn.chunk(
            shifted_temb, 9, dim=2
        )

        gate_sa = ttnn.typecast(gate_sa, dtype=ttnn.bfloat16)
        gate_ff = ttnn.typecast(gate_ff, dtype=ttnn.bfloat16)
        gate_ca = ttnn.typecast(gate_ca, dtype=ttnn.bfloat16)

        # Self-attention with AdaLN
        spatial_normed = self.norm1(spatial_1BND)
        spatial_normed = spatial_normed * (1.0 + scale_sa) + shift_sa

        spatial_1BND = self.attn1(
            spatial_1BND=spatial_normed,
            N=N,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
            addcmul_residual=spatial_1BND,
            addcmul_gate=gate_sa,
        )

        # Cross-attention with AdaLN (query modulation + context modulation)
        ca_input = self.norm2(spatial_1BND) * (1.0 + scale_ca) + shift_ca
        if prompt_temb is not None:
            shifted_prompt = self.prompt_scale_shift_table.data + prompt_temb
            kv_shift, kv_scale = ttnn.chunk(shifted_prompt, 2, dim=2)
            prompt_mod = prompt_1BLP * (1.0 + kv_scale) + kv_shift
        else:
            prompt_mod = prompt_1BLP
        ca_output = self.attn2(spatial_1BND=ca_input, N=N, prompt_1BLP=prompt_mod)
        spatial_1BND = spatial_1BND + ca_output * gate_ca

        # Feedforward with AdaLN
        spatial_normed = self.norm3(spatial_1BND)
        spatial_normed = spatial_normed * (1.0 + scale_ff) + shift_ff

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_normed = self.ccl_manager.all_gather_persistent_buffer(
                spatial_normed, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        spatial_ff = self.ffn(spatial_normed, compute_kernel_config=self.ff_compute_kernel_config)
        spatial_1BND = ttnn.addcmul(spatial_1BND, spatial_ff, gate_ff)

        return spatial_1BND


class LTXTransformerModel(Module):
    """
    LTX-2 DiT Transformer model.

    Architecture:
    - patchify_proj: Linear(in_channels, dim)
    - adaln_single: LTXAdaLayerNormSingle for timestep conditioning
    - transformer_blocks: N x LTXTransformerBlock
    - norm_out + proj_out: output projection with AdaLN
    """

    def __init__(
        self,
        *,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-6,
        ffn_mult: int = 4,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        self.inner_dim = num_attention_heads * attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        # Patch embedding: Linear(in_channels -> inner_dim), TP-sharded output
        self.patchify_proj = ColParallelLinear(
            in_channels,
            self.inner_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        # Timestep conditioning (9 params for self-attn + FF + cross-attn)
        self.adaln_single = LTXAdaLayerNormSingle(
            embedding_dim=self.inner_dim,
            embedding_coefficient=9,
            mesh_device=mesh_device,
        )
        # Prompt timestep (2 params for context modulation)
        self.prompt_adaln_single = LTXAdaLayerNormSingle(
            embedding_dim=self.inner_dim,
            embedding_coefficient=2,
            mesh_device=mesh_device,
        )

        # Transformer blocks
        ffn_dim = self.inner_dim * ffn_mult
        self.transformer_blocks = ModuleList()
        for _ in range(num_layers):
            self.transformer_blocks.append(
                LTXTransformerBlock(
                    dim=self.inner_dim,
                    ffn_dim=ffn_dim,
                    num_heads=num_attention_heads,
                    cross_attention_dim=cross_attention_dim,
                    eps=norm_eps,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                    is_fsdp=is_fsdp,
                )
            )

        # Output: LayerNorm (no affine) + AdaLN + proj
        self.norm_out = DistributedLayerNorm(
            self.inner_dim,
            norm_eps=norm_eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.scale_shift_table = Parameter(
            total_shape=[1, 1, 2, self.inner_dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
            device=mesh_device,
            dtype=ttnn.float32,
        )

        self.proj_out = Linear(
            self.inner_dim,
            out_channels,
            bias=True,
            mesh_device=mesh_device,
        )

        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "scale_shift_table" in state:
            sst = state["scale_shift_table"]
            # 22B checkpoint has 2 output modulation params but shape matches, just unsqueeze
            state["scale_shift_table"] = sst.unsqueeze(0).unsqueeze(0)

        # No truncation needed — adaln_single uses coefficient=9 matching 22B

        # Remove 22B-specific keys not in video-only model
        pop_substate(state, "caption_projection")
        pop_substate(state, "video_embeddings_connector")
        pop_substate(state, "audio_embeddings_connector")

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        temb: ttnn.Tensor,
        prompt_1BLP: ttnn.Tensor,
        N: int,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor | None,
    ) -> ttnn.Tensor:
        """
        Args:
            spatial_1BND: Patchified spatial input (1, B, N, D), fractured on SP/TP
            temb: Timestep tensor (B,) or (1,1,B,1) on device
            prompt_1BLP: Encoded text prompt (1, B, L, P), replicated
            N: Logical sequence length
            rope_cos, rope_sin, trans_mat: RoPE tensors
        """
        # Patch embedding
        spatial_1BND = self.patchify_proj(spatial_1BND)

        # Timestep conditioning (9 params)
        modulation_params, _embedded_timestep = self.adaln_single(temb)
        B = modulation_params.shape[2]
        modulation_1B9D = ttnn.reshape(modulation_params, (1, B, 9, self.inner_dim))
        if self.parallel_config.tensor_parallel.factor > 1:
            modulation_1B9D = ttnn.mesh_partition(
                modulation_1B9D, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        # Prompt timestep (2 params, NOT TP-sharded)
        prompt_mod, _ = self.prompt_adaln_single(temb)
        prompt_1B2D = ttnn.reshape(prompt_mod, (1, B, 2, self.inner_dim))

        # Transformer blocks
        for block in self.transformer_blocks:
            spatial_1BND = block(
                spatial_1BND=spatial_1BND,
                prompt_1BLP=prompt_1BLP,
                temb_1BTD=modulation_1B9D,
                N=N,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                trans_mat=trans_mat,
                prompt_temb=prompt_1B2D,
            )

        # Output projection with AdaLN
        # _embedded_timestep: (1, 1, B, dim) -> reshape to (1, B, 1, dim) for broadcasting
        inner_dim_local = _embedded_timestep.shape[-1]
        embedded_1B1D = ttnn.reshape(_embedded_timestep, (1, B, 1, inner_dim_local))
        # TP-shard to match scale_shift_table
        if self.parallel_config.tensor_parallel.factor > 1:
            embedded_1B1D = ttnn.mesh_partition(
                embedded_1B1D, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
            )
        shifted_sst = self.scale_shift_table.data + embedded_1B1D
        shift_out, scale_out = ttnn.chunk(shifted_sst, 2, dim=2)

        spatial_normed = self.norm_out(spatial_1BND)
        spatial_1BND = spatial_normed * (1.0 + scale_out) + shift_out

        # Gather TP shards before output projection
        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        spatial_1BND = self.proj_out(
            spatial_1BND, compute_kernel_config=self.hifi4_compute_kernel_config, dtype=ttnn.float32
        )

        return spatial_1BND

    def inner_step(
        self,
        spatial_1BNI_torch: torch.Tensor,
        prompt_1BLP: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor | None,
        N: int,
        timestep_torch: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Reduced forward for denoising loop. Caches prompt/rope/trans_mat on device
        across steps. Spatial input is torch; output is fp32 ttnn on device.

        Args:
            spatial_1BNI_torch: (1, B, N, in_channels) torch tensor
            prompt_1BLP: Cached text embeddings on device
            rope_cos, rope_sin, trans_mat: Cached RoPE tensors on device
            N: Logical sequence length
            timestep_torch: 1D torch tensor of timestep values
        """
        from ....utils.tensor import bf16_tensor, float32_tensor

        # Push spatial to device (SP-sharded)
        spatial_1BNI = bf16_tensor(
            spatial_1BNI_torch,
            device=self.mesh_device,
            mesh_axis=self.parallel_config.sequence_parallel.mesh_axis,
            shard_dim=-2,
        )

        # Timestep embedding
        B_size = spatial_1BNI_torch.shape[1]
        timestep = float32_tensor(
            timestep_torch.reshape(1, 1, B_size, 1) * 1000.0,
            device=self.mesh_device,
        )
        modulation_params, _embedded_timestep = self.adaln_single(timestep)

        B = modulation_params.shape[2]
        modulation_1B9D = ttnn.reshape(modulation_params, (1, B, 9, self.inner_dim))
        if self.parallel_config.tensor_parallel.factor > 1:
            modulation_1B9D = ttnn.mesh_partition(
                modulation_1B9D, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        # Prompt timestep uses SAME scaled timestep as main adaln
        prompt_mod, _ = self.prompt_adaln_single(timestep)
        prompt_1B2D = ttnn.reshape(prompt_mod, (1, B, 2, self.inner_dim))

        # Patch embedding
        spatial_1BND = self.patchify_proj(spatial_1BNI)

        # Transformer blocks
        for block in self.transformer_blocks:
            spatial_1BND = block(
                spatial_1BND=spatial_1BND,
                prompt_1BLP=prompt_1BLP,
                temb_1BTD=modulation_1B9D,
                N=N,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                trans_mat=trans_mat,
                prompt_temb=prompt_1B2D,
            )

        # Output AdaLN + projection
        inner_dim_local = _embedded_timestep.shape[-1]
        embedded_1B1D = ttnn.reshape(_embedded_timestep, (1, B, 1, inner_dim_local))
        if self.parallel_config.tensor_parallel.factor > 1:
            embedded_1B1D = ttnn.mesh_partition(
                embedded_1B1D, dim=3, cluster_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        shifted_sst = self.scale_shift_table.data + embedded_1B1D
        shift_out, scale_out = ttnn.chunk(shifted_sst, 2, dim=2)

        spatial_normed = self.norm_out(spatial_1BND)
        spatial_1BND = spatial_normed * (1.0 + scale_out) + shift_out

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        proj_out_1BNI = self.proj_out(
            spatial_1BND, compute_kernel_config=self.hifi4_compute_kernel_config, dtype=ttnn.float32
        )

        # Gather SP for final output (stays on device)
        if self.parallel_config.sequence_parallel.factor > 1:
            proj_out_1BNI = self.ccl_manager.all_gather_persistent_buffer(
                proj_out_1BNI, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
            )

        return proj_out_1BNI

    @staticmethod
    def device_to_host(tt_tensor: ttnn.Tensor) -> torch.Tensor:
        """Move a ttnn device tensor to a torch host tensor."""
        return ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0])
