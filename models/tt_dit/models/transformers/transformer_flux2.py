# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import ttnn

from ...blocks.attention_opt import Attention
from ...blocks.transformer_block_opt import TransformerBlock
from ...layers.embeddings import CombinedTimestepGuidanceTextProjEmbeddings
from ...layers.linear import ColParallelLinear, RowParallelLinear, prepare_chunked_linear_output
from ...layers.module import Module, ModuleList
from ...layers.normalization import DistributedLayerNorm
from ...utils.substate import rename_substate
from ...utils.tensor import prepare_weight_for_concatenated_input

if TYPE_CHECKING:
    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager
    from ...utils.padding import PaddingConfig


# Every time-embedding projection in the transformer consumes the same `time_embed`, so they are
# merged into one chunked matmul. This table is the single source of truth for that merge: the
# checkpoint prefix each group comes from, how many chunks it contributes, and which of its chunks
# are `scale` (those get the consumers' `+1` folded into the bias). Row order here is the chunk
# order of the merged weight, which is also the order `forward` unpacks.
_TIME_MOD_SOURCES = (
    ("double_stream_modulation_img.linear", 6, (1, 4)),  # (shift, scale, gate) x attn, ff
    ("double_stream_modulation_txt.linear", 6, (1, 4)),
    ("single_stream_modulation.linear", 3, (1,)),  # (shift, scale, gate)
    ("norm_out.linear", 2, (0,)),  # (scale, shift)
)
_TIME_MOD_CHUNKS = sum(count for _, count, _ in _TIME_MOD_SOURCES)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/transformers/transformer_flux.py
class Flux2SingleTransformerBlock(Module):
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
        shard_prompt: bool = False,
    ) -> None:
        super().__init__()

        tp_axis = parallel_config.tensor_parallel.mesh_axis
        mlp_hidden_dim = 3 * dim
        fsdp_mesh_axis = parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.attn = Attention(
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
            per_head_norm=True,
            is_fsdp=is_fsdp,
            shard_prompt=shard_prompt,
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

        # Shard output, since size of input dimension << size of output dimension.
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

        # Shard input, since size of input dimension >> size of output dimension.
        self.proj_out = RowParallelLinear(
            dim + mlp_hidden_dim,
            dim,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
            fsdp_mesh_axis=fsdp_mesh_axis,
            # No explicit L1 mm_memory_config: the fused MMRS path now hands the MM output to the
            # RS through the rolling L1 window by default (see Linear.forward_fused_addcmul).
        )

        self._dim = dim
        self._mlp_hidden_dim = mlp_hidden_dim
        self._tp_axis = tp_axis
        self._tp_factor = parallel_config.tensor_parallel.factor
        self._ccl_manager = ccl_manager
        self._parallel_config = parallel_config
        self._mesh_device = device

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        qkv_mlp_weight = state.pop("attn.to_qkv_mlp_proj.weight", None)
        if qkv_mlp_weight is not None:
            state["attn.to_q.weight"] = qkv_mlp_weight[: self._dim]
            state["attn.to_k.weight"] = qkv_mlp_weight[self._dim : 2 * self._dim]
            state["attn.to_v.weight"] = qkv_mlp_weight[2 * self._dim : 3 * self._dim]

            w = qkv_mlp_weight[3 * self._dim :]
            # swiglu implementation uses different ordering
            state["proj_mlp.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)

        proj_out_weight = state.pop("attn.to_out.weight", None)
        if proj_out_weight is not None:
            state["proj_out.weight"] = prepare_weight_for_concatenated_input(
                proj_out_weight,
                [self._dim, self._mlp_hidden_dim],
                device_count=self._tp_factor,
            )

    # Since we do not have operations to concatenate and slice a tensor along a sharded dimension,
    # we keep the spatial and prompt tensors separate for now.
    def forward(
        self,
        *,
        spatial_prompt_concat: ttnn.Tensor,
        spatial_size: int,
        time_embed: ttnn.Tensor,
        combined_rope: tuple[ttnn.Tensor, ttnn.Tensor] | None,
        temb_mod_params: tuple[ttnn.Tensor, ...],
        skip_time_embed_activation_fn: bool = False,
        spatial_sequence_length: int,
        compute_prompt_output: bool = True,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """Run a single-stream block.

        When ``compute_prompt_output`` is False, the prompt's projected output is skipped:
        the prompt still participates in attention (its K/V are context for the spatial
        stream), but ``proj_mlp(c)``, the prompt concat, and ``proj_out(c)`` are not computed
        and ``None`` is returned for the prompt. Used by the final single block, whose prompt
        output is discarded by the transformer.
        """

        shift_msa, scale_msa, gate_msa = temb_mod_params

        x_c = ttnn.squeeze(
            self.norm(ttnn.unsqueeze(spatial_prompt_concat, 0), dynamic_weight=scale_msa, dynamic_bias=shift_msa), 0
        )

        x_c_attn, _ = self.attn.forward(
            sequence_1=x_c,
            sequence_1_rope=combined_rope,
            sequence_1_length=spatial_sequence_length,
        )

        # slice out spatial only. Prompt not needed. The variable names are left to prevent code bloat. But this is just spatial after the slice.
        if not compute_prompt_output:
            x_c = x_c[:, :spatial_size, :]
            x_c_attn = x_c_attn[:, :spatial_size, :]
            spatial_prompt_concat = spatial_prompt_concat[:, :spatial_size, :]

        x_c_mlp = self.proj_mlp(
            x_c,
            parallel_config=self._parallel_config,
            use_heuristic_mmcfg=True,
            core_grid=Attention.get_core_grid(x_c.shape[-2], self._mesh_device.compute_with_storage_grid_size()),
        )

        if self._ccl_manager.topology == ttnn.Topology.Ring:
            return self.proj_out.forward_fused_addcmul([x_c_attn, x_c_mlp], spatial_prompt_concat, gate_msa, scalar=1.0)
        return ttnn.addcmul(spatial_prompt_concat, self.proj_out([x_c_attn, x_c_mlp]), gate_msa)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/transformers/transformer_flux.py
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
        shard_prompt: bool = False,
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
            TransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_pre_only=False,
                attention_proj_bias=False,
                ff_activation_fn="swiglu",
                ff_mult=3,
                ff_bias=False,
                time_norm_affine=False,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                mesh_device=device,
                is_fsdp=is_fsdp,
                shard_prompt=shard_prompt,
            )
            for i in range(num_layers)
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
                shard_prompt=shard_prompt,
            )
            for i in range(num_single_layers)
        )

        # One matmul for every modulation consumer; see _TIME_MOD_SOURCES. The bias carries the
        # consumers' `+1` on each scale chunk (see _prepare_torch_state).
        self.time_mod = ColParallelLinear(
            inner_dim,
            _TIME_MOD_CHUNKS * inner_dim,
            bias=True,
            mesh_device=device,
            mesh_axis=tp_axis,
            chunks=_TIME_MOD_CHUNKS,
        )

        self.norm_out = DistributedLayerNorm(
            inner_dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
        )

        self.proj_out = RowParallelLinear(
            inner_dim,
            out_channels,
            bias=False,
            mesh_device=device,
            mesh_axis=tp_axis,
            ccl_manager=ccl_manager,
            # No explicit L1 mm_memory_config: this layer's fused path uses the rolling-window L1
            # handoff by default, and its plain forward never read mm_memory_config.
        )

        self.device = device
        self._ccl_manager = ccl_manager
        self._tp_axis = tp_axis
        self._tp_factor = parallel_config.tensor_parallel.factor
        self._inner_dim = inner_dim
        self.pc = parallel_config

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Merge the four time-embedding projections into `time_mod` by stacking their rows in
        # _TIME_MOD_SOURCES order. Leave the weight absent (reported as a missing key) rather than
        # build a partial one if any source is missing.
        weights = [state.pop(f"{prefix}.weight", None) for prefix, _, _ in _TIME_MOD_SOURCES]
        assert not any(
            [state.pop(f"{prefix}.bias", None) is not None for prefix, _, _ in _TIME_MOD_SOURCES]
        ), "merged time-embedding projection assumes bias-free sources"
        if all(w is not None for w in weights):
            state["time_mod.weight"] = torch.cat(weights, dim=0)
            # Synthesise the bias that carries the consumers' `+1` on each scale chunk.
            state["time_mod.bias"] = torch.cat(
                [
                    torch.full([self._inner_dim], 1.0 if i in scale_indices else 0.0)
                    for _, count, scale_indices in _TIME_MOD_SOURCES
                    for i in range(count)
                ]
            )
        prepare_chunked_linear_output(state, prefix="time_mod", device_count=self._tp_factor, chunks=_TIME_MOD_CHUNKS)

        rename_substate(state, "norm_out.norm", "norm_out")

        for i in range(len(self.transformer_blocks)):
            prefix = f"transformer_blocks.{i}."

            # swiglu implementation uses different ordering
            w = state.pop(f"{prefix}ff.linear_in.weight", None)
            if w is not None:
                state[f"{prefix}ff.net.0.proj.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)
            w = state.pop(f"{prefix}ff_context.linear_in.weight", None)
            if w is not None:
                state[f"{prefix}ff_context.net.0.proj.weight"] = torch.roll(w, shifts=w.shape[0] // 2, dims=0)

            rename_substate(state, f"{prefix}ff.linear_out", f"{prefix}ff.net.2")
            rename_substate(state, f"{prefix}ff_context.linear_out", f"{prefix}ff_context.net.2")

    def forward(
        self,
        spatial: ttnn.Tensor,
        embedded_prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
        guidance: ttnn.Tensor,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        combined_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,
        compute_prompt_output: bool = False,
    ) -> ttnn.Tensor:
        """Run the model forward.

        Args:
            spatial: Tensor with shape [batch_size, spatial_sequence_length / sp_factor, in_channels].
            embedded_prompt: `context_embedder` applied to the prompt, with shape
                [batch_size, prompt_sequence_length, inner_dim / tp_factor]. Applied by the caller
                rather than here because the prompt is constant across denoising steps.
            timestep: Tensor with shape [batch_size, 1].
            guidance: Optional tensor with shape [batch_size, 1].
            spatial_rope: Tuple of two tensors with shape [spatial_sequence_length / sp_factor, head_dim].
            prompt_rope: Tuple of two tensors with shape [prompt_sequence_length, head_dim] (sequence is not sharded!).
            combined_rope: spatial_rope and prompt_rope concatenated on the sequence dim, for the
                single blocks. Supplied by the caller rather than built here because it is constant
                across denoising steps.
            spatial_sequence_length: int.
            prompt_sequence_length: int.
            compute_prompt_output: bool. Whether to compute the prompt output. Final prompt output is typically unused.
        """
        time_embed = self.time_guidance_embed(timestep=timestep, guidance=guidance)
        ttnn.silu(time_embed, output_tensor=time_embed)
        time_embed = time_embed.reshape([time_embed.shape[-2], 1, time_embed.shape[-1]])

        # One matmul for every modulation consumer, emitting bf16 with the `+1` already folded into
        # each scale chunk, so the tuples below are directly in the (shift, 1 + scale, gate) form
        # the blocks expect — no per-chunk add or typecast. Slices follow _TIME_MOD_SOURCES:
        # img(6), txt(6), single(3), norm_out(2).
        time_mod = self.time_mod(time_embed, dtype=ttnn.bfloat16)
        double_stream_mod_img = tuple(time_mod[0:6])
        double_stream_mod_txt = tuple(time_mod[6:12])
        single_stream_mod = tuple(time_mod[12:15])
        norm_out_scale, norm_out_shift = time_mod[15], time_mod[16]

        spatial = self.x_embedder(spatial)
        prompt = embedded_prompt

        for i, block in enumerate(self.transformer_blocks, start=1):
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                spatial_sequence_length=spatial_sequence_length,
                prompt_sequence_length=prompt_sequence_length,
                skip_time_embed_activation_fn=True,
            )

            # if i % 6 == 0:
            #     ttnn.ReadDeviceProfiler(spatial.device())

        num_single_blocks = len(self.single_transformer_blocks)

        # prep pre concatenated data
        spatial_prompt_concat = ttnn.concat([spatial, prompt], dim=1)
        spatial_size = spatial.shape[1]
        spatial_prompt_sequence_length = spatial_sequence_length + prompt_sequence_length
        for i, block in enumerate(self.single_transformer_blocks, start=1):
            spatial_prompt_concat = block.forward(
                spatial_prompt_concat=spatial_prompt_concat,
                spatial_size=spatial_size,
                time_embed=time_embed,
                combined_rope=combined_rope,
                temb_mod_params=single_stream_mod,
                spatial_sequence_length=spatial_prompt_sequence_length,
                skip_time_embed_activation_fn=True,
                compute_prompt_output=(i < num_single_blocks)
                or compute_prompt_output,  # prompt is unused for the last block.
            )

            # if i % 6 == 0:
            #     ttnn.ReadDeviceProfiler(spatial.device())

        spatial = spatial_prompt_concat if compute_prompt_output else spatial_prompt_concat  # [:, :spatial_size, :]

        spatial = ttnn.squeeze(
            self.norm_out(ttnn.unsqueeze(spatial, 0), dynamic_weight=norm_out_scale, dynamic_bias=norm_out_shift), 0
        )

        return self.proj_out(spatial, gather_output=True)
