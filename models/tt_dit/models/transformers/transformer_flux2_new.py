# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn

from ...layers.embeddings import CombinedTimestepGuidanceTextProjEmbeddings
from ...layers.feedforward import ParallelFeedForward
from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear, prepare_chunked_linear_output
from ...layers.module import Module, ModuleList
from ...layers.normalization import DistributedLayerNorm
from ...utils.substate import rename_substate
from ...utils.tracing import traced_function
from .attention_flux2 import Flux2Attention

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager
    from ...utils.padding import PaddingConfig


class Flux2Modulation(Module):
    def __init__(self, dim: int, *, mod_param_sets: int, tp_axis: int | None, device: ttnn.MeshDevice) -> None:
        super().__init__()

        self.linear = ColParallelLinear(
            dim, dim * 3 * mod_param_sets, bias=False, mesh_axis=tp_axis, mesh_device=device
        )

        self._mod_param_sets = mod_param_sets
        self._tp_factor = device.shape[tp_axis] if tp_axis is not None else 1

    def forward(self, x: torch.Tensor, *, skip_act_fn: bool = False) -> tuple[torch.Tensor, ...]:
        assert len(x.shape) == 3

        if not skip_act_fn:
            x = ttnn.silu(x)

        x = self.linear(x)
        return ttnn.chunk(x, 3 * self._mod_param_sets, dim=-1)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        prepare_chunked_linear_output(
            state, prefix="linear", device_count=self._tp_factor, chunks=3 * self._mod_param_sets
        )


class Flux2DoubleStreamBlock(Module):
    """Optimized double-stream (joint spatial+prompt) transformer block.

    Key optimizations over TransformerBlock:
    - Ring: all-gather is fused with the QKV matmul inside ColParallelLinear.
    - Linear: block gathers before attention (shared gather), then passes gathered input.
    - to_out + gate * result + residual is fused inside attention (_to_out_fused_addcmul).
    - FFN uses forward_fused_addcmul (Ring): fuses reduce-scatter + gate * result + residual.
    - Linear topology fallback: explicit all-gather + separate gate + addcmul.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        head_dim: int,
        context_pre_only: bool,
        attention_proj_bias: bool = False,
        ff_activation_fn: str = "swiglu",
        ff_mult: int = 3,
        ff_bias: bool = False,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.context_pre_only = context_pre_only
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.norm1_norm = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.norm1_context_norm = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.attn = Flux2Attention(
            query_dim=dim,
            head_dim=head_dim,
            heads=num_heads,
            out_dim=dim,
            added_kv_proj_dim=dim,
            context_pre_only=context_pre_only,
            proj_bias=attention_proj_bias,
            eps=1e-6,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
            is_fsdp=is_fsdp,
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
            mult=ff_mult,
            bias=ff_bias,
            activation_fn=ff_activation_fn,
            mesh_device=mesh_device,
            mesh_axis=tp_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        if not context_pre_only:
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
                mult=ff_mult,
                bias=ff_bias,
                activation_fn=ff_activation_fn,
                mesh_device=mesh_device,
                mesh_axis=tp_axis,
                fsdp_mesh_axis=fsdp_mesh_axis,
                ccl_manager=ccl_manager,
            )
        else:
            self.norm2_context = None
            self.ff_context = None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "norm1.norm", "norm1_norm")
        rename_substate(state, "norm1_context.norm", "norm1_context_norm")
        rename_substate(state, "ff.net.0.proj", "ff.ff1")
        rename_substate(state, "ff.net.2", "ff.ff2")
        rename_substate(state, "ff_context.net.0.proj", "ff_context.ff1")
        rename_substate(state, "ff_context.net.2", "ff_context.ff2")

    def forward(
        self,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        spatial_sequence_length: int,
        *,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None = None,
        temb_mod_params_img: tuple[ttnn.Tensor, ...],
        temb_mod_params_txt: tuple[ttnn.Tensor, ...],
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """Optimized double-stream block forward.

        Args:
            spatial: [B, N/sp, D/tp] - fractured on SP and TP. NOT pre-gathered.
            prompt: [B, L, D/tp] - fractured on TP only. NOT pre-gathered.
            temb_mod_params_img: 6-tuple (shift_attn, scale_attn, gate_attn, shift_ff, scale_ff, gate_ff) for spatial.
            temb_mod_params_txt: 6-tuple (same order) for prompt, or 2-tuple if context_pre_only.
        """
        assert len(spatial.shape) == 3
        assert len(prompt.shape) == 3

        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        is_ring = self.ccl_manager.topology == ttnn.Topology.Ring

        (
            spatial_shift_attn,
            spatial_scale_attn,
            spatial_gate_attn,
            spatial_shift_ff,
            spatial_scale_ff,
            spatial_gate_ff,
        ) = temb_mod_params_img

        # NOTE: workaround - addcmul (fused and unfused) is less accurate with fp32 gate input
        spatial_gate_ff = ttnn.typecast(spatial_gate_ff, dtype=ttnn.bfloat16)

        if self.context_pre_only:
            prompt_scale_attn, prompt_shift_attn = temb_mod_params_txt
            prompt_gate_attn = None
            prompt_shift_ff = None
            prompt_scale_ff = None
            prompt_gate_ff = None
        else:
            (
                prompt_shift_attn,
                prompt_scale_attn,
                prompt_gate_attn,
                prompt_shift_ff,
                prompt_scale_ff,
                prompt_gate_ff,
            ) = temb_mod_params_txt
            # NOTE: workaround - addcmul (fused and unfused) is less accurate with fp32 gate input
            prompt_gate_ff = ttnn.typecast(prompt_gate_ff, dtype=ttnn.bfloat16)

        # Norm (fractured output; DistributedLayerNorm handles stats all-gather internally)
        spatial_normed = ttnn.squeeze(
            self.norm1_norm(
                ttnn.unsqueeze(spatial, 0),
                dynamic_weight=(1 + spatial_scale_attn),
                dynamic_bias=spatial_shift_attn,
            ),
            0,
        )
        prompt_normed = ttnn.squeeze(
            self.norm1_context_norm(
                ttnn.unsqueeze(prompt, 0),
                dynamic_weight=(1 + prompt_scale_attn),
                dynamic_bias=prompt_shift_attn,
            ),
            0,
        )

        # For Linear topology, gather before attention (Ring does it inside ColParallelLinear).
        is_ring = self.ccl_manager.topology == ttnn.Topology.Ring
        if not is_ring and self.parallel_config.tensor_parallel.factor > 1:
            tp_axis = self.parallel_config.tensor_parallel.mesh_axis
            spatial_normed = self.ccl_manager.all_gather_persistent_buffer(
                spatial_normed, dim=-1, mesh_axis=tp_axis, use_hyperparams=True
            )
            prompt_normed = self.ccl_manager.all_gather_persistent_buffer(
                prompt_normed, dim=-1, mesh_axis=tp_axis, use_hyperparams=True
            )

        # Fused to_out + gate * result + residual via addcmul params.
        spatial, prompt_out = self.attn.forward(
            spatial=spatial_normed,
            prompt=prompt_normed,
            spatial_sequence_length=spatial_sequence_length,
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            addcmul_spatial_residual=spatial,
            addcmul_spatial_gate=spatial_gate_attn,
            addcmul_prompt_residual=prompt if not self.context_pre_only else None,
            addcmul_prompt_gate=prompt_gate_attn,
        )
        # spatial is now: original_spatial + to_out(attn_spatial) * gate_attn
        # prompt_out: original_prompt + to_add_out(attn_prompt) * gate_attn (or None if pre_only)

        if prompt_out is not None:
            prompt = prompt_out

        # === SPATIAL FFN ===
        spatial_normed = ttnn.squeeze(
            self.norm2(
                ttnn.unsqueeze(spatial, 0),
                dynamic_weight=(1 + spatial_scale_ff),
                dynamic_bias=spatial_shift_ff,
            ),
            0,
        )

        if is_ring:
            # Ring: ff1 uses fused AG+MM (via parallel_config), ff2 fuses RS+addcmul.
            # No explicit all-gather needed here.
            spatial = self.ff.forward_fused_addcmul(
                spatial_normed,
                spatial,
                spatial_gate_ff,
                parallel_config=self.parallel_config,
            )
        else:
            # Linear: explicit all-gather, then FFN, then addcmul.
            spatial_normed = self.ccl_manager.all_gather_persistent_buffer(spatial_normed, dim=-1, mesh_axis=tp_axis)
            spatial_ff = ttnn.squeeze(self.ff(ttnn.unsqueeze(spatial_normed, 0)), 0)
            spatial = ttnn.addcmul(spatial, spatial_ff, spatial_gate_ff)

        if self.context_pre_only:
            return spatial, None

        # === PROMPT FFN ===
        prompt_normed = ttnn.squeeze(
            self.norm2_context(
                ttnn.unsqueeze(prompt, 0),
                dynamic_weight=(1 + prompt_scale_ff),
                dynamic_bias=prompt_shift_ff,
            ),
            0,
        )

        if is_ring:
            prompt = self.ff_context.forward_fused_addcmul(
                prompt_normed,
                prompt,
                prompt_gate_ff,
                parallel_config=self.parallel_config,
            )
        else:
            prompt_normed = self.ccl_manager.all_gather_persistent_buffer(prompt_normed, dim=-1, mesh_axis=tp_axis)
            prompt_ff = ttnn.squeeze(self.ff_context(ttnn.unsqueeze(prompt_normed, 0)), 0)
            prompt = ttnn.addcmul(prompt, prompt_ff, prompt_gate_ff)

        return spatial, prompt


class Flux2SingleTransformerBlock(Module):
    """Optimized single-stream transformer block.

    Key optimizations:
    - QKV and MLP projections use fused AG+MM via parallel_config (Ring) or
      explicit gather (Linear).
    - proj_out uses forward_fused_addcmul (Ring): fuses reduce-scatter + gate * result + residual.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        head_dim: int,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        mlp_hidden_dim = 3 * dim
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None
        # When head padding is applied (e.g., 48 heads padded to 64 for tp=32),
        # the per-device attention output is padded_inner_dim/tp, not dim/tp.
        # proj_out must be sized accordingly so K matches K_w.
        padded_inner_dim = padding_config.target_dim if padding_config is not None else dim

        self.attn = Flux2Attention(
            query_dim=dim,
            head_dim=head_dim,
            heads=num_heads,
            out_dim=dim,
            added_kv_proj_dim=0,
            pre_only=True,
            proj_bias=False,
            eps=1e-6,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
            use_spatial_weights_for_prompt=True,
            is_fsdp=is_fsdp,
        )

        self.norm = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=tp_axis,
            mesh_device=device,
            ccl_manager=ccl_manager,
        )

        self.proj_mlp = ColParallelLinear(
            dim,
            mlp_hidden_dim,
            bias=False,
            activation_fn="swiglu",
            mesh_device=device,
            mesh_axis=tp_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.proj_out = RowParallelLinear(
            padded_inner_dim + mlp_hidden_dim,
            dim,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
        )

        self._dim = dim
        self._mlp_hidden_dim = mlp_hidden_dim
        self._padded_inner_dim = padded_inner_dim
        self._tp_axis = tp_axis
        self._tp_factor = parallel_config.tensor_parallel.factor
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        qkv_mlp_weight = state.pop("attn.to_qkv_mlp_proj.weight", None)
        if qkv_mlp_weight is not None:
            state["attn.to_q.weight"] = qkv_mlp_weight[: self._dim]
            state["attn.to_k.weight"] = qkv_mlp_weight[self._dim : 2 * self._dim]
            state["attn.to_v.weight"] = qkv_mlp_weight[2 * self._dim : 3 * self._dim]

            w = qkv_mlp_weight[3 * self._dim :]
            state["proj_mlp.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)

        proj_out_weight = state.pop("attn.to_out.weight", None)
        if proj_out_weight is not None:
            if self._padded_inner_dim != self._dim:
                # Head padding is active: attn output per device is padded_inner_dim/tp,
                # not dim/tp. Zero-pad the attn input columns of the proj_out weight
                # so that the padding head slots contribute zero to the output.
                attn_w = proj_out_weight[:, : self._dim]
                mlp_w = proj_out_weight[:, self._dim :]
                pad = torch.zeros(attn_w.shape[0], self._padded_inner_dim - self._dim, dtype=attn_w.dtype)
                proj_out_weight = torch.cat([attn_w, pad, mlp_w], dim=1)
            state["proj_out.weight"] = _prepare_weight_for_concatenated_input(
                proj_out_weight,
                [self._padded_inner_dim, self._mlp_hidden_dim],
                device_count=self._tp_factor,
            )

    def forward(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        time_embed: ttnn.Tensor,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        temb_mod_params: tuple[ttnn.Tensor, ...],
        skip_time_embed_activation_fn: bool = False,
        spatial_sequence_length: int,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if not skip_time_embed_activation_fn:
            time_embed = ttnn.silu(time_embed)

        x = ttnn.squeeze(self.norm(ttnn.unsqueeze(spatial, 0)), 0)
        c = ttnn.squeeze(self.norm(ttnn.unsqueeze(prompt, 0)), 0)

        shift_msa, scale_msa, gate_msa = temb_mod_params
        # Workaround: addcmul requires same dtype for all inputs; fp32 gate is also less accurate.
        gate_msa = ttnn.typecast(gate_msa, dtype=ttnn.bfloat16)
        x = x * (1 + scale_msa) + shift_msa
        c = c * (1 + scale_msa) + shift_msa

        is_ring = self._ccl_manager.topology == ttnn.Topology.Ring
        use_nonfused_agmm = not is_ring and (self._tp_factor > 1)

        if use_nonfused_agmm:
            # Linear: one all-gather shared by proj_mlp and attn QKV
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
            c = self._ccl_manager.all_gather_persistent_buffer(c, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        proj_parallel_config = None if use_nonfused_agmm else self._parallel_config

        # MLP branch (ColParallelLinear with fused AG+MM for Ring)
        x_mlp = self.proj_mlp(x, parallel_config=proj_parallel_config)
        c_mlp = self.proj_mlp(c, parallel_config=proj_parallel_config)

        # Attention (pre_only=True: returns raw SDPA output, no to_out projection).
        # x and c are already gathered above when use_nonfused_agmm; attn expects gathered input.
        x, c = self.attn.forward(
            spatial=x,
            prompt=c,
            spatial_rope=spatial_rope,
            prompt_rope=prompt_rope,
            spatial_sequence_length=spatial_sequence_length,
        )

        # Concatenate attention and MLP outputs (both fractured on TP)
        proj_x = ttnn.concat([x, x_mlp], dim=-1)
        proj_c = ttnn.concat([c, c_mlp], dim=-1)

        if is_ring:
            # Fused: RowParallel matmul + reduce-scatter + gate * result + residual
            spatial = self.proj_out.forward_fused_addcmul(proj_x, spatial, gate_msa)
            prompt = self.proj_out.forward_fused_addcmul(proj_c, prompt, gate_msa)
        else:
            # Linear: separate proj_out + gate multiply + add
            x_out = gate_msa * self.proj_out(proj_x)
            c_out = gate_msa * self.proj_out(proj_c)
            spatial = spatial + x_out
            prompt = prompt + c_out

        return spatial, prompt


class Flux2Transformer(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_layers: int,
        num_single_layers: int,
        attention_head_dim: int,
        num_attention_heads: int,
        joint_attention_dim: int,
        out_channels: int,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
        is_fsdp: bool = False,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        tp_axis = parallel_config.tensor_parallel.mesh_axis

        self.time_guidance_embed = CombinedTimestepGuidanceTextProjEmbeddings(
            embedding_dim=inner_dim,
            pooled_projection_dim=0,
            bias=False,
            with_guidance=True,
            mesh_device=device,
        )

        self.context_embedder = ColParallelLinear(
            joint_attention_dim, inner_dim, bias=False, mesh_device=device, mesh_axis=tp_axis
        )

        self.x_embedder = ColParallelLinear(in_channels, inner_dim, bias=False, mesh_device=device, mesh_axis=tp_axis)

        self.transformer_blocks = ModuleList(
            Flux2DoubleStreamBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_pre_only=False,
                attention_proj_bias=False,
                ff_activation_fn="swiglu",
                ff_mult=3,
                ff_bias=False,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                mesh_device=device,
                is_fsdp=is_fsdp,
            )
            for _ in range(num_layers)
        )

        self.single_transformer_blocks = ModuleList(
            Flux2SingleTransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                device=device,
                is_fsdp=is_fsdp,
            )
            for _ in range(num_single_layers)
        )

        self.time_embed_out = Linear(inner_dim, 2 * inner_dim, bias=False, mesh_device=device)

        self.norm_out = DistributedLayerNorm(
            inner_dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )

        self.proj_out = Linear(inner_dim, out_channels, bias=False, mesh_device=device)

        self.double_stream_modulation_img = Flux2Modulation(inner_dim, mod_param_sets=2, tp_axis=tp_axis, device=device)
        self.double_stream_modulation_txt = Flux2Modulation(inner_dim, mod_param_sets=2, tp_axis=tp_axis, device=device)
        self.single_stream_modulation = Flux2Modulation(inner_dim, mod_param_sets=1, tp_axis=tp_axis, device=device)

        self.device = device
        self._ccl_manager = ccl_manager
        self._tp_axis = tp_axis

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "norm_out.linear", "time_embed_out")
        rename_substate(state, "norm_out.norm", "norm_out")

        for i in range(len(self.transformer_blocks)):
            prefix = f"transformer_blocks.{i}."

            w = state.pop(f"{prefix}ff.linear_in.weight", None)
            if w is not None:
                state[f"{prefix}ff.net.0.proj.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)
            w = state.pop(f"{prefix}ff_context.linear_in.weight", None)
            if w is not None:
                state[f"{prefix}ff_context.net.0.proj.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)

            rename_substate(state, f"{prefix}ff.linear_out", f"{prefix}ff.net.2")
            rename_substate(state, f"{prefix}ff_context.linear_out", f"{prefix}ff_context.net.2")

    @traced_function(device=lambda self: self.device, clone_prep_inputs=False)
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
        """Run the model forward.

        Args:
            spatial: [batch_size, spatial_sequence_length / sp_factor, in_channels].
            prompt: [batch_size, prompt_sequence_length, joint_attention_dim].
            timestep: [batch_size, 1].
            guidance: [batch_size, 1].
            spatial_rope: tuple of two tensors [spatial_sequence_length / sp_factor, head_dim].
            prompt_rope: tuple of two tensors [prompt_sequence_length, head_dim].
        """
        time_embed = self.time_guidance_embed(timestep=timestep, guidance=guidance)
        ttnn.silu(time_embed, output_tensor=time_embed)
        time_embed = time_embed.reshape([time_embed.shape[-2], 1, time_embed.shape[-1]])

        double_stream_mod_img = self.double_stream_modulation_img(time_embed, skip_act_fn=True)
        double_stream_mod_txt = self.double_stream_modulation_txt(time_embed, skip_act_fn=True)
        single_stream_mod = self.single_stream_modulation(time_embed, skip_act_fn=True)

        spatial = self.x_embedder(spatial)
        prompt = self.context_embedder(prompt)

        for i, block in enumerate(self.transformer_blocks, start=1):
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                spatial_sequence_length=spatial_sequence_length,
            )

            if i % 6 == 0:
                ttnn.ReadDeviceProfiler(spatial.device())

        prompt = ttnn.clone(prompt, dtype=spatial.dtype)

        for i, block in enumerate(self.single_transformer_blocks, start=1):
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                temb_mod_params=single_stream_mod,
                spatial_sequence_length=spatial_sequence_length,
                skip_time_embed_activation_fn=True,
            )

            if i % 6 == 0:
                ttnn.ReadDeviceProfiler(spatial.device())

        spatial = ttnn.squeeze(self.norm_out(ttnn.unsqueeze(spatial, 0)), 0)

        spatial_time = self.time_embed_out(time_embed)
        [scale, shift] = ttnn.chunk(spatial_time, 2, dim=-1)

        spatial = self._ccl_manager.all_gather_persistent_buffer(
            spatial, dim=2, mesh_axis=self._tp_axis, use_hyperparams=True
        )

        spatial = spatial * (1 + scale) + shift

        return self.proj_out(spatial)

    def release_traces(self):
        tracer = Flux2Transformer.forward._tracers.get(self)
        if tracer is not None:
            tracer.release_trace()


def _prepare_weight_for_concatenated_input(
    weight: torch.Tensor,
    sizes: Sequence[int],
    *,
    device_count: int,
) -> torch.Tensor:
    weights = weight.split(sizes, dim=1)
    weights = [w.unflatten(1, [device_count, -1]) for w in weights]
    return torch.cat(weights, dim=2).flatten(1, 2)
