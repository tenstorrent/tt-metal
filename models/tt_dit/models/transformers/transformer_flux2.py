# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn
from models.common.utility_functions import is_blackhole

from ...blocks.attention import Attention
from ...layers.embeddings import Flux2TimestepGuidanceEmbeddings
from ...layers.feedforward import ParallelFeedForward
from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear, prepare_chunked_linear_output
from ...layers.module import Module, ModuleList
from ...layers.normalization import DistributedLayerNorm
from ...utils.substate import rename_substate

if TYPE_CHECKING:
    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager
    from ...utils.padding import PaddingConfig


def _chunk_time3d(t: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    size = t.shape[-1] // count
    if len(t.shape) == 4:
        return [t[:, :, :, i * size : (i + 1) * size] for i in range(count)]
    return [t[:, :, i * size : (i + 1) * size] for i in range(count)]


def _re_fuse_proj_out_weight(
    weight: torch.Tensor,
    *,
    embedding_dim: int,
    device_count: int,
) -> torch.Tensor:
    if device_count == 1:
        return weight

    _, in_dim = weight.shape

    in_dim1 = embedding_dim
    in_dim2 = in_dim - in_dim1

    w1, w2 = weight.split([in_dim1, in_dim2], dim=-1)

    w1 = w1.reshape([-1, device_count, in_dim1 // device_count])
    w2 = w2.reshape([-1, device_count, in_dim2 // device_count])
    return torch.cat([w1, w2], dim=-1).reshape([-1, in_dim])


class Flux2DoubleTransformerBlock(Module):
    """Flux2 double-stream transformer block.

    Mirrors the upstream Flux2TransformerBlock from diffusers but uses TT shared
    layers (Attention, DistributedLayerNorm, ParallelFeedForward).

    Key difference from Flux1's TransformerBlock: modulation parameters are
    passed in externally (pre-computed by the top-level Flux2Transformer) rather
    than computed by an internal AdaLayerNormZero linear.  The block receives
    6-element modulation vectors for both img and txt streams.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 3.0,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
        attention_k_chunk_size: int = 512,
        attention_q_chunk_size: int = 128,
        is_fsdp: bool = False,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        use_fused_ag_matmul: bool = True,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.mesh_device = mesh_device
        self._use_fused_ag_matmul = use_fused_ag_matmul

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.norm1 = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.norm1_context = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.attn = Attention(
            query_dim=dim,
            head_dim=head_dim,
            heads=num_heads,
            out_dim=dim,
            added_kv_proj_dim=dim,
            context_pre_only=False,
            eps=1e-6,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
            k_chunk_size=attention_k_chunk_size,
            q_chunk_size=attention_q_chunk_size,
            is_fsdp=is_fsdp,
            dtype=weights_dtype,
        )

        self.norm2 = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.ff = ParallelFeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="swiglu",
            inner_dim=int(dim * mlp_ratio),
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
            dtype=weights_dtype,
        )

        self.norm2_context = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )
        self.ff_context = ParallelFeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="swiglu",
            inner_dim=int(dim * mlp_ratio),
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
            dtype=weights_dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "ff.linear_in", "ff.ff1")
        rename_substate(state, "ff.linear_out", "ff.ff2")
        rename_substate(state, "ff_context.linear_in", "ff_context.ff1")
        rename_substate(state, "ff_context.linear_out", "ff_context.ff2")

        for prefix in ("ff.ff1", "ff_context.ff1"):
            key = f"{prefix}.weight"
            if key in state:
                w = state[key]
                half = w.shape[0] // 2
                state[key] = torch.cat([w[half:], w[:half]], dim=0)

        for prefix in ("attn.to_q", "attn.to_k", "attn.to_v"):
            wkey = f"{prefix}.weight"
            bkey = f"{prefix}.bias"
            if wkey in state and bkey not in state:
                state[bkey] = torch.zeros(state[wkey].shape[0])
        for prefix in ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"):
            wkey = f"{prefix}.weight"
            bkey = f"{prefix}.bias"
            if wkey in state and bkey not in state:
                state[bkey] = torch.zeros(state[wkey].shape[0])
        if "attn.to_out.0.weight" in state and "attn.to_out.0.bias" not in state:
            state["attn.to_out.0.bias"] = torch.zeros(state["attn.to_out.0.weight"].shape[0])
        if "attn.to_add_out.weight" in state and "attn.to_add_out.bias" not in state:
            state["attn.to_add_out.bias"] = torch.zeros(state["attn.to_add_out.weight"].shape[0])

    def forward(
        self,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        mod_img: ttnn.Tensor,
        mod_txt: ttnn.Tensor,
        *,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        spatial_sequence_length: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = _chunk_time3d(mod_img, 6)
        (c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp) = _chunk_time3d(mod_txt, 6)

        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        attn_pc = self.parallel_config if self._use_fused_ag_matmul else None

        spatial_normed = ttnn.squeeze(
            self.norm1(ttnn.unsqueeze(spatial, 0), dynamic_weight=(1 + scale_msa), dynamic_bias=shift_msa),
            0,
        )
        prompt_normed = ttnn.squeeze(
            self.norm1_context(ttnn.unsqueeze(prompt, 0), dynamic_weight=(1 + c_scale_msa), dynamic_bias=c_shift_msa),
            0,
        )

        if not self._use_fused_ag_matmul:
            spatial_normed = self.ccl_manager.all_gather_persistent_buffer(
                spatial_normed,
                dim=2,
                mesh_axis=tp_axis,
                use_hyperparams=True,
            )
            prompt_normed = self.ccl_manager.all_gather_persistent_buffer(
                prompt_normed,
                dim=2,
                mesh_axis=tp_axis,
                use_hyperparams=True,
            )

        spatial_attn, prompt_attn = self.attn.forward(
            spatial=spatial_normed,
            prompt=prompt_normed,
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            spatial_sequence_length=spatial_sequence_length,
            parallel_config=attn_pc,
        )
        spatial_attn = spatial_attn * gate_msa
        prompt_attn = prompt_attn * c_gate_msa

        spatial += spatial_attn
        prompt += prompt_attn

        spatial_normed = ttnn.squeeze(
            self.norm2(ttnn.unsqueeze(spatial, 0), dynamic_weight=(1 + scale_mlp), dynamic_bias=shift_mlp),
            0,
        )
        if not self._use_fused_ag_matmul:
            spatial_normed = self.ccl_manager.all_gather_persistent_buffer(
                spatial_normed,
                dim=2,
                mesh_axis=tp_axis,
                use_hyperparams=True,
            )
        if self.ccl_manager.topology != ttnn.Topology.Linear:
            spatial = ttnn.squeeze(
                self.ff.forward_fused_addcmul(
                    ttnn.unsqueeze(spatial_normed, 0),
                    ttnn.unsqueeze(spatial, 0),
                    ttnn.unsqueeze(gate_mlp, 0),
                    scalar=1.0,
                    parallel_config=attn_pc,
                ),
                0,
            )
        else:
            spatial_ff = ttnn.squeeze(
                self.ff(ttnn.unsqueeze(spatial_normed, 0), parallel_config=attn_pc),
                0,
            )
            spatial += gate_mlp * spatial_ff

        prompt_normed = ttnn.squeeze(
            self.norm2_context(ttnn.unsqueeze(prompt, 0), dynamic_weight=(1 + c_scale_mlp), dynamic_bias=c_shift_mlp),
            0,
        )
        if not self._use_fused_ag_matmul:
            prompt_normed = self.ccl_manager.all_gather_persistent_buffer(
                prompt_normed,
                dim=2,
                mesh_axis=tp_axis,
                use_hyperparams=True,
            )
        if self.ccl_manager.topology != ttnn.Topology.Linear:
            prompt = ttnn.squeeze(
                self.ff_context.forward_fused_addcmul(
                    ttnn.unsqueeze(prompt_normed, 0),
                    ttnn.unsqueeze(prompt, 0),
                    ttnn.unsqueeze(c_gate_mlp, 0),
                    scalar=1.0,
                    parallel_config=attn_pc,
                ),
                0,
            )
        else:
            prompt_ff = ttnn.squeeze(
                self.ff_context(ttnn.unsqueeze(prompt_normed, 0), parallel_config=attn_pc),
                0,
            )
            prompt += c_gate_mlp * prompt_ff

        return spatial, prompt


class Flux2SingleTransformerBlock(Module):
    """Flux2 single-stream parallel transformer block.

    Unlike Flux1's serial single block (norm -> attn -> MLP -> proj_out), Flux2
    uses a ViT-22B-style parallel block where QKV and MLP-in projections are fused,
    and attention-out and MLP-out projections are fused.

    Modulation (shift/scale/gate) comes from the top-level single_stream_modulation
    and is passed in externally.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        head_dim: int,
        mlp_ratio: float = 3.0,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
        attention_k_chunk_size: int = 512,
        attention_q_chunk_size: int = 128,
        is_fsdp: bool = False,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        use_fused_ag_matmul: bool = True,
    ) -> None:
        super().__init__()

        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.dim = dim
        self._use_fused_ag_matmul = use_fused_ag_matmul

        inner_dim = num_heads * head_dim
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.inner_dim = inner_dim
        self.mlp_hidden_dim = mlp_hidden_dim

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.norm = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.attn = Attention(
            query_dim=dim,
            head_dim=head_dim,
            heads=num_heads,
            out_dim=dim,
            added_kv_proj_dim=0,
            pre_only=True,
            eps=1e-6,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
            use_spatial_weights_for_prompt=True,
            k_chunk_size=attention_k_chunk_size,
            q_chunk_size=attention_q_chunk_size,
            is_fsdp=is_fsdp,
            dtype=weights_dtype,
        )

        self.proj_mlp = ColParallelLinear(
            dim,
            mlp_hidden_dim,
            bias=False,
            activation_fn="swiglu",
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
            dtype=weights_dtype,
        )

        self.proj_out = RowParallelLinear(
            inner_dim + mlp_hidden_dim,
            dim,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
            dtype=weights_dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        tp_factor = self.parallel_config.tensor_parallel.factor

        if "attn.to_qkv_mlp_proj.weight" in state:
            w = state.pop("attn.to_qkv_mlp_proj.weight")
            qkv_dim = self.inner_dim * 3
            mlp_dim = self.mlp_hidden_dim * 2
            w_qkv, w_mlp = w.split([qkv_dim, mlp_dim], dim=0)

            w_q, w_k, w_v = w_qkv.chunk(3, dim=0)
            state["attn.to_q.weight"] = w_q
            state["attn.to_k.weight"] = w_k
            state["attn.to_v.weight"] = w_v

            half = mlp_dim // 2
            state["proj_mlp.weight"] = torch.cat([w_mlp[half:], w_mlp[:half]], dim=0)

        for prefix in ("attn.to_q", "attn.to_k", "attn.to_v"):
            wkey = f"{prefix}.weight"
            bkey = f"{prefix}.bias"
            if wkey in state and bkey not in state:
                state[bkey] = torch.zeros(state[wkey].shape[0])

        if "attn.to_out.weight" in state:
            w = state.pop("attn.to_out.weight")
            state["proj_out.weight"] = w

        if "proj_out.weight" in state:
            state["proj_out.weight"] = _re_fuse_proj_out_weight(
                state["proj_out.weight"],
                embedding_dim=self.inner_dim,
                device_count=tp_factor,
            )

    def forward(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        mod: ttnn.Tensor,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        spatial_sequence_length: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        proj_pc = self.parallel_config if self._use_fused_ag_matmul else None
        shift_msa, scale_msa, gate_msa = _chunk_time3d(mod, 3)

        norm_spatial = ttnn.squeeze(
            self.norm(ttnn.unsqueeze(spatial, 0), dynamic_weight=(1 + scale_msa), dynamic_bias=shift_msa),
            0,
        )
        norm_prompt = ttnn.squeeze(
            self.norm(ttnn.unsqueeze(prompt, 0), dynamic_weight=(1 + scale_msa), dynamic_bias=shift_msa),
            0,
        )

        if not self._use_fused_ag_matmul:
            norm_spatial_full = self.ccl_manager.all_gather_persistent_buffer(
                norm_spatial,
                dim=2,
                mesh_axis=tp_axis,
                use_hyperparams=True,
            )
            norm_prompt_full = self.ccl_manager.all_gather_persistent_buffer(
                norm_prompt,
                dim=2,
                mesh_axis=tp_axis,
                use_hyperparams=True,
            )
        else:
            norm_spatial_full = norm_spatial
            norm_prompt_full = norm_prompt

        mlp_spatial = ttnn.squeeze(
            self.proj_mlp(ttnn.unsqueeze(norm_spatial_full, 0), parallel_config=proj_pc),
            0,
        )
        mlp_prompt = ttnn.squeeze(
            self.proj_mlp(ttnn.unsqueeze(norm_prompt_full, 0), parallel_config=proj_pc),
            0,
        )

        attn_spatial, attn_prompt = self.attn.forward(
            spatial=norm_spatial_full,
            prompt=norm_prompt_full,
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            spatial_sequence_length=spatial_sequence_length,
            parallel_config=proj_pc,
        )
        del norm_spatial, norm_prompt, norm_spatial_full, norm_prompt_full

        additional_spatial = ttnn.concat([attn_spatial, mlp_spatial], dim=-1)
        additional_prompt = ttnn.concat([attn_prompt, mlp_prompt], dim=-1)

        if self.ccl_manager.topology != ttnn.Topology.Linear:
            spatial = ttnn.squeeze(
                self.proj_out.forward_fused_addcmul(
                    ttnn.unsqueeze(additional_spatial, 0),
                    ttnn.unsqueeze(spatial, 0),
                    ttnn.unsqueeze(gate_msa, 0),
                ),
                0,
            )
            prompt = ttnn.squeeze(
                self.proj_out.forward_fused_addcmul(
                    ttnn.unsqueeze(additional_prompt, 0),
                    ttnn.unsqueeze(prompt, 0),
                    ttnn.unsqueeze(gate_msa, 0),
                ),
                0,
            )
        else:
            additional_spatial = ttnn.squeeze(self.proj_out(ttnn.unsqueeze(additional_spatial, 0)), 0)
            additional_prompt = ttnn.squeeze(self.proj_out(ttnn.unsqueeze(additional_prompt, 0)), 0)
            additional_spatial = gate_msa * additional_spatial
            additional_prompt = gate_msa * additional_prompt

            spatial += additional_spatial
            prompt += additional_prompt

        return spatial, prompt


class Flux2Transformer(Module):
    """TT implementation of the Flux2 transformer (Flux2Transformer2DModel).

    Mirrors Flux1Transformer but adapts for Flux2's architecture:
    - Flux2TimestepGuidanceEmbeddings (no pooled projection)
    - External modulation via Flux2Modulation (double_stream_modulation_img/txt, single_stream_modulation)
    - SwiGLU feed-forward
    - Parallel single-stream blocks
    - 4-axis RoPE with theta=2000
    """

    sdpa_chunk_size_map = {
        (False, 1, 2): (128, 512),
        (False, 2, 4): (128, 512),
        (True, 2, 2): (128, 512),
    }
    default_sdpa_chunk_size = (128, 512)

    def __init__(
        self,
        *,
        patch_size: int,
        in_channels: int,
        num_layers: int,
        num_single_layers: int,
        attention_head_dim: int,
        num_attention_heads: int,
        joint_attention_dim: int,
        timestep_guidance_channels: int,
        out_channels: int,
        mlp_ratio: float = 3.0,
        guidance_embeds: bool = True,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
        is_fsdp: bool = False,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        use_fused_ag_matmul: bool = True,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.patch_size = patch_size
        self.inner_dim = inner_dim
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.is_fsdp = is_fsdp
        self._use_fused_ag_matmul = use_fused_ag_matmul
        self.fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        q_chunk_size, k_chunk_size = self.sdpa_chunk_size_map.get(
            (
                is_blackhole(),
                self.parallel_config.sequence_parallel.factor,
                self.parallel_config.tensor_parallel.factor,
            ),
            self.default_sdpa_chunk_size,
        )

        tp_axis = parallel_config.tensor_parallel.mesh_axis

        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            embedding_dim=inner_dim,
            timestep_guidance_channels=timestep_guidance_channels,
            with_guidance=guidance_embeds,
            mesh_device=mesh_device,
        )

        self.double_stream_modulation_img = ColParallelLinear(
            inner_dim,
            inner_dim * 6,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            dtype=weights_dtype,
        )
        self.double_stream_modulation_txt = ColParallelLinear(
            inner_dim,
            inner_dim * 6,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            dtype=weights_dtype,
        )
        self.single_stream_modulation = ColParallelLinear(
            inner_dim,
            inner_dim * 3,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            dtype=weights_dtype,
        )

        self.context_embedder = ColParallelLinear(
            joint_attention_dim,
            inner_dim,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            dtype=weights_dtype,
        )
        self.x_embedder = ColParallelLinear(
            in_channels,
            inner_dim,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            dtype=weights_dtype,
        )

        self.transformer_blocks = ModuleList(
            Flux2DoubleTransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                attention_k_chunk_size=k_chunk_size,
                attention_q_chunk_size=q_chunk_size,
                is_fsdp=is_fsdp,
                weights_dtype=weights_dtype,
                use_fused_ag_matmul=use_fused_ag_matmul,
            )
            for _ in range(num_layers)
        )

        self.single_transformer_blocks = ModuleList(
            Flux2SingleTransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                attention_k_chunk_size=k_chunk_size,
                attention_q_chunk_size=q_chunk_size,
                is_fsdp=is_fsdp,
                weights_dtype=weights_dtype,
                use_fused_ag_matmul=use_fused_ag_matmul,
            )
            for _ in range(num_single_layers)
        )

        self.norm_out = DistributedLayerNorm(
            inner_dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )
        self.norm_out_cond = Linear(
            inner_dim,
            2 * inner_dim,
            bias=False,
            mesh_device=mesh_device,
            dtype=weights_dtype,
        )

        self.proj_out = Linear(
            inner_dim,
            patch_size * patch_size * out_channels,
            bias=False,
            mesh_device=mesh_device,
            dtype=weights_dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "norm_out.linear", "norm_out_cond")
        rename_substate(state, "norm_out.norm", "norm_out")

        rename_substate(state, "double_stream_modulation_img.linear", "double_stream_modulation_img")
        rename_substate(state, "double_stream_modulation_txt.linear", "double_stream_modulation_txt")
        rename_substate(state, "single_stream_modulation.linear", "single_stream_modulation")

        prepare_chunked_linear_output(
            state,
            prefix="double_stream_modulation_img",
            device_count=self.parallel_config.tensor_parallel.factor,
            chunks=6,
        )
        prepare_chunked_linear_output(
            state,
            prefix="double_stream_modulation_txt",
            device_count=self.parallel_config.tensor_parallel.factor,
            chunks=6,
        )
        prepare_chunked_linear_output(
            state,
            prefix="single_stream_modulation",
            device_count=self.parallel_config.tensor_parallel.factor,
            chunks=3,
        )

    def forward(
        self,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
        guidance: ttnn.Tensor,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,
    ) -> ttnn.Tensor:
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        temb = self.time_guidance_embed(timestep=timestep, guidance=guidance)
        ttnn.silu(temb, output_tensor=temb)
        temb = temb.reshape([temb.shape[-2], 1, temb.shape[-1]])

        double_mod_img = self.double_stream_modulation_img(temb)
        double_mod_txt = self.double_stream_modulation_txt(temb)
        single_mod = self.single_stream_modulation(temb)

        spatial = self.x_embedder(spatial)
        prompt = self.context_embedder(prompt)

        for block in self.transformer_blocks:
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                mod_img=double_mod_img,
                mod_txt=double_mod_txt,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                spatial_sequence_length=spatial_sequence_length,
            )

        prompt = ttnn.clone(prompt, dtype=spatial.dtype)

        for block in self.single_transformer_blocks:
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                mod=single_mod,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                spatial_sequence_length=spatial_sequence_length,
            )

        spatial = ttnn.squeeze(self.norm_out(ttnn.unsqueeze(spatial, 0)), 0)

        spatial_cond = self.norm_out_cond(temb)
        [scale, shift] = _chunk_time3d(spatial_cond, 2)

        spatial = self.ccl_manager.all_gather_persistent_buffer(spatial, dim=2, mesh_axis=tp_axis, use_hyperparams=True)

        spatial = spatial * (1 + scale) + shift

        return self.proj_out(spatial)
