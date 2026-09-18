# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel as TorchSD3Transformer2DModel
from loguru import logger

import ttnn

from ...layers.embeddings import PatchEmbed, SD35CombinedTimestepTextProjEmbeddings
from ...layers.feedforward import ParallelFeedForward
from ...layers.linear import ColParallelLinear, Linear, prepare_chunked_linear_output
from ...layers.module import Module, ModuleList
from ...layers.normalization import DistributedLayerNorm, LayerNorm
from ...utils import cache
from ...utils.matmul import get_fabric_agmm_config, register_matmul_configs
from ...utils.padding import PaddingConfig
from ...utils.substate import rename_substate
from .attention_sd35 import SD35JointAttention, flatten_batch, is_fused_tp, unflatten_batch
from .sd35_quant_config import SD35QuantProfile

if TYPE_CHECKING:
    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager


def gate_rows(gate: ttnn.Tensor, spatial_1BND: ttnn.Tensor) -> ttnn.Tensor:
    """Materialize a per-batch adaLN gate at every flattened spatial row: -> [1, 1, B*N, D/tp].

    The fused to_out / ff2 epilogues take the gate either as one broadcast row or as a full
    [M, N] map. With the CFG pair folded into M (uncond rows, then cond rows) the two batch
    elements carry different gates, so the full map is the only layout that keeps them apart.
    ``gate`` holds B rows of D/tp (any placement of B among the leading dims); B=1 stays a row.
    """
    _, b, n, d_local = spatial_1BND.shape
    if gate.dtype != spatial_1BND.dtype:
        # The timestep enters as float32, so the adaLN chunks come out float32. The fused MMRS /
        # strided-AGMM epilogues take the gate's dtype at face value for page sizes, and a float32
        # gate against a bf16 residual produced garbage (2026-09-17); match the residual's dtype.
        gate = ttnn.typecast(gate, spatial_1BND.dtype)
    if b == 1:
        return ttnn.reshape(gate, (1, 1, 1, d_local))
    gate = ttnn.reshape(gate, (1, b, 1, d_local))
    # broadcast_to is ~3x cheaper than ttnn.repeat here (68 us vs 193 us for 2 x 4096 x 608 bf16).
    gate = ttnn.experimental.broadcast_to(gate, ttnn.Shape([1, b, n, d_local]))  # (1, B, N, Dloc)
    return flatten_batch(gate)


# adapted from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/models/attention_processor.py
class SD35TransformerBlock(Module):
    def __init__(
        self,
        dim,
        num_heads,
        head_dim,
        context_pre_only,
        use_dual_attention=False,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
        padding_config=None,
        quant_config=None,
    ):
        super().__init__()

        assert not use_dual_attention, "Expecting not dual attention"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.context_pre_only = context_pre_only
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.quant_config = quant_config
        # LoFi (or profile) matmul compute config, forwarded to the FFN matmuls (the attention linears
        # take theirs at construction). None => the FFN keeps its per-dtype default compute config.
        self._ff_compute_config = (
            quant_config.mm_compute_config(mesh_device.arch()) if quant_config is not None else None
        )
        _ffn_q = quant_config.ffn_kwargs() if quant_config is not None else {}
        # When activations are quantized, narrow the TP activation gathers to bf8 so the collective
        # moves half the bytes (bf8 tiles are 1088 B vs bf16's 2048 B). None => gather stays bf16.
        self._ag_dtype = quant_config.activation_dtype if quant_config is not None else None
        # Ring TP: spatial to_qkv / to_out / ff1 become all-gather-matmuls on the fractured input,
        # ff2 a fused matmul + reduce-scatter, with the gated residuals in the epilogues. The prompt
        # stream (M ~ 160, memory-bound) keeps its explicit gathers on either topology.
        self.fused_tp = is_fused_tp(parallel_config, ccl_manager)

        # adaLN modulation for both streams from ONE chunked matmul (spatial: shift, scale, gate x attn, ff;
        # prompt: the same six, or (scale, shift) for the last block). The consumers' `+1` on every scale
        # chunk is folded into the bias at load (_prepare_torch_state), so the norms take the chunks as
        # they come: no per-block silu, slices or adds. Same weight bytes as the two linears it replaces.
        self._ctx_mod_chunks = 2 if context_pre_only else 6
        self.time_mod = ColParallelLinear(
            dim,
            (6 + self._ctx_mod_chunks) * dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            chunks=6 + self._ctx_mod_chunks,
        )
        self.norm1_norm = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.norm1_context_norm = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.attn = SD35JointAttention(
            query_dim=dim,
            head_dim=head_dim,
            heads=num_heads,
            out_dim=dim,
            bias=True,
            context_pre_only=context_pre_only,
            eps=1e-6,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
            quant_config=quant_config,
        )

        self.norm2 = DistributedLayerNorm(
            dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.ff = ParallelFeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn="gelu_tanh",
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
            **_ffn_q,
        )

        self.norm2_context = None
        self.ff_context = None

        if not context_pre_only:
            self.norm2_context = DistributedLayerNorm(
                dim,
                norm_eps=1e-6,
                norm_elementwise_affine=False,
                bias=False,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
            )
            self.ff_context = ParallelFeedForward(
                dim=dim,
                dim_out=dim,
                activation_fn="gelu_tanh",
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                ccl_manager=ccl_manager,
                **_ffn_q,
            )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Merge norm1.linear (spatial adaLN) and norm1_context.linear (prompt adaLN) into time_mod and add
        # 1 to the bias of every scale chunk. Chunk orders follow diffusers: AdaLayerNormZero emits
        # (shift, scale, gate) x (attn, ff); AdaLayerNormContinuous (last block's prompt) emits (scale, shift).
        w_s, b_s = state.pop("norm1.linear.weight", None), state.pop("norm1.linear.bias", None)
        w_c, b_c = state.pop("norm1_context.linear.weight", None), state.pop("norm1_context.linear.bias", None)
        if w_s is not None and w_c is not None:
            b_s = b_s.clone() if b_s is not None else torch.zeros(w_s.shape[0], dtype=w_s.dtype)
            b_c = b_c.clone() if b_c is not None else torch.zeros(w_c.shape[0], dtype=w_c.dtype)
            for i in (1, 4):
                b_s[i * self.dim : (i + 1) * self.dim] += 1
            for i in (0,) if self.context_pre_only else (1, 4):
                b_c[i * self.dim : (i + 1) * self.dim] += 1
            state["time_mod.weight"] = torch.cat([w_s, w_c], dim=0)
            state["time_mod.bias"] = torch.cat([b_s, b_c], dim=0)
        rename_substate(state, "ff.net.0.proj", "ff.ff1")
        rename_substate(state, "ff.net.2", "ff.ff2")
        rename_substate(state, "ff_context.net.0.proj", "ff_context.ff1")
        rename_substate(state, "ff_context.net.2", "ff_context.ff2")

        prepare_chunked_linear_output(
            state,
            prefix="time_mod",
            device_count=self.parallel_config.tensor_parallel.factor,
            chunks=6 + self._ctx_mod_chunks,
        )

    def _ag_tp(self, x):
        """All-gather ``x`` on the TP feature axis (dim=3).

        If activations are quantized (``self._ag_dtype`` set), narrow ``x`` to bf8 first so the
        collective moves half the bytes; the ping-pong buffer is allocated at the same dtype. A None
        ``_ag_dtype`` reproduces the original bf16 gather exactly.
        """
        axis = self.parallel_config.tensor_parallel.mesh_axis
        if self._ag_dtype is not None:
            x = ttnn.typecast(x, self._ag_dtype)
        return ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                x.shape, 3, axis, dtype=self._ag_dtype or ttnn.bfloat16
            ),
            dim=3,
            multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(axis),
            num_links=self.ccl_manager.num_links,
            topology=self.ccl_manager.topology,
            cluster_axis=axis,
            **self.ccl_manager.get_ag_hyperparams(x.shape),
        )

    def forward(self, spatial_1BND, prompt_1BLD, time_embed_silu_11BE, N):
        """
        spatial_1BND: fractured N on SP, fractured D on TP
        prompt_1BLD: replicated on SP, fractured D on TP
        time_embed_silu_11BE: silu(time embedding), replicated; computed once per step by the model
        """

        # One chunked matmul: bf16 chunks with the `+1` already in every scale (bias), see __init__.
        mods = self.time_mod(time_embed_silu_11BE, dtype=ttnn.bfloat16)
        (
            spatial_shift_attn,
            spatial_scale_attn,
            spatial_gate_attn,
            spatial_shift_ff,
            spatial_scale_ff,
            spatial_gate_ff,
        ) = mods[:6]

        spatial_normed_1BND = self.norm1_norm(
            spatial_1BND, dynamic_weight=spatial_scale_attn, dynamic_bias=spatial_shift_attn
        )

        if self.context_pre_only:
            prompt_scale_attn, prompt_shift_attn = mods[6:8]
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
            ) = mods[6:12]

        prompt_normed_1BLD = self.norm1_context_norm(
            prompt_1BLD, dynamic_weight=prompt_scale_attn, dynamic_bias=prompt_shift_attn
        )

        if self.fused_tp:
            # Spatial stays TP-fractured: to_qkv gathers inside the all-gather-matmul, and to_out
            # applies gate + residual in its epilogue, returning the updated residual stream.
            prompt_normed_1BLD = self._ag_tp(prompt_normed_1BLD)
            spatial_1BND, prompt_attn_1BLD = self.attn(
                spatial_normed_1BND,
                prompt_normed_1BLD,
                N,
                spatial_residual=spatial_1BND,
                spatial_gate=gate_rows(spatial_gate_attn, spatial_1BND),
            )
        else:
            if self.parallel_config.tensor_parallel.factor > 1:
                # Gather spatial, prompt before attention
                spatial_normed_1BND = self._ag_tp(spatial_normed_1BND)
                prompt_normed_1BLD = self._ag_tp(prompt_normed_1BLD)

            spatial_attn_1BLD, prompt_attn_1BLD = self.attn(spatial_normed_1BND, prompt_normed_1BLD, N)
            spatial_attn_1BLD = spatial_attn_1BLD * spatial_gate_attn

            # residual
            spatial_1BND = spatial_1BND + spatial_attn_1BLD

        prompt_attn_1BLD = prompt_attn_1BLD * prompt_gate_attn if prompt_gate_attn is not None else None

        spatial_normed_1BND = self.norm2(spatial_1BND, dynamic_weight=spatial_scale_ff, dynamic_bias=spatial_shift_ff)

        if self.fused_tp:
            # ff1: all_gather_minimal_matmul_async is SLOWER here than an explicit gather followed by
            # the swept plain matmul (1338 us vs ~400 + 439 us on the tp4 column: K/tp = 19 tiles is
            # prime, so the fused kernel is stuck with a 1- or 19-tile K block). Gather + matmul is the
            # default; SD35_FF1_AGMM=1 selects the fused kernel for A/B.
            normed_flat = flatten_batch(spatial_normed_1BND)
            if os.environ.get("SD35_FF1_AGMM", "0") == "1":
                ff1_flat = self.ff.ff1(
                    normed_flat, compute_kernel_config=self._ff_compute_config, parallel_config=self.parallel_config
                )
            else:
                gathered = self.ccl_manager.all_gather_persistent_buffer(
                    normed_flat, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
                )
                ff1_flat = self.ff.ff1(gathered, compute_kernel_config=self._ff_compute_config)
            if os.environ.get("SD35_FF2_MMRS", "1") == "1":
                # ff2 as matmul + reduce-scatter with the gated residual fused at the scatter write.
                # The MMRS config for this shape must keep an 11-column matmul grid (see
                # fused_mmrs_configs): on 12 columns the op deadlocks. SD35_FF2_MMRS=0 falls back to
                # the row-parallel ff2 with a separate reduce-scatter.
                ff_flat = self.ff.ff2.forward_fused_addcmul(
                    ff1_flat,
                    flatten_batch(spatial_1BND),
                    gate_rows(spatial_gate_ff, spatial_1BND),
                    scalar=1.0,
                    compute_kernel_config=self._ff_compute_config,
                )
                spatial_1BND = unflatten_batch(ff_flat, spatial_1BND.shape)
            else:
                ff_flat = self.ff.ff2(ff1_flat, compute_kernel_config=self._ff_compute_config)
                spatial_ff_1BND = unflatten_batch(ff_flat, spatial_1BND.shape)
                spatial_1BND = spatial_1BND + spatial_ff_1BND * spatial_gate_ff
        else:
            if self.parallel_config.tensor_parallel.factor > 1:
                spatial_normed_1BND = self._ag_tp(spatial_normed_1BND)

            spatial_ff_1BND = self.ff(spatial_normed_1BND, compute_kernel_config=self._ff_compute_config)
            spatial_ff_1BND = spatial_ff_1BND * spatial_gate_ff

            spatial_1BND += spatial_ff_1BND

        if self.context_pre_only:
            return spatial_1BND, None

        prompt_1BLD += prompt_attn_1BLD

        prompt_normed_1BLD = self.norm2_context(
            prompt_1BLD, dynamic_weight=prompt_scale_ff, dynamic_bias=prompt_shift_ff
        )

        if self.parallel_config.tensor_parallel.factor > 1:
            prompt_normed_1BLD = self._ag_tp(prompt_normed_1BLD)

        prompt_ff_1BLD = self.ff_context(prompt_normed_1BLD, compute_kernel_config=self._ff_compute_config)
        prompt_ff_1BLD = prompt_ff_1BLD * prompt_gate_ff

        prompt_1BLD += prompt_ff_1BLD

        return spatial_1BND, prompt_1BLD


def chunk_time(t: ttnn.Tensor, count: int) -> list[ttnn.Tensor]:
    size = t.shape[-1] // count
    return [t[:, :, :, i * size : (i + 1) * size] for i in range(count)]


class SD35Transformer2DModel(Module):
    def __init__(
        self,
        sample_size=128,
        patch_size=2,
        in_channels=16,
        num_layers=18,
        attention_head_dim=64,
        num_attention_heads=18,
        joint_attention_dim=4096,
        caption_projection_dim=1152,
        pooled_projection_dim=2048,
        out_channels=16,
        pos_embed_max_size=96,
        dual_attention_layers=(),
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
        padding_config=None,
        quant_config=None,
    ):
        super().__init__()

        self.quant_config = quant_config
        self.sample_size = sample_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.attention_head_dim = attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.joint_attention_dim = joint_attention_dim
        self.caption_projection_dim = caption_projection_dim
        self.pooled_projection_dim = pooled_projection_dim
        self.out_channels = out_channels
        self.pos_embed_max_size = pos_embed_max_size
        self.dual_attention_layers = dual_attention_layers
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # Components
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
            mesh_device=mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            sp_mesh_axis=parallel_config.sequence_parallel.mesh_axis,
        )

        self.time_text_embed = SD35CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=pooled_projection_dim,
            mesh_device=mesh_device,
        )

        self.context_embedder = ColParallelLinear(
            joint_attention_dim,
            caption_projection_dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        # Transformer blocks
        self.transformer_blocks = ModuleList()
        for i in range(num_layers):
            block = SD35TransformerBlock(
                dim=self.inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_pre_only=i == num_layers - 1,
                use_dual_attention=i in dual_attention_layers,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                quant_config=quant_config,
            )
            self.transformer_blocks.append(block)

        # Output normalization and projection. Under fused Ring TP the final norm runs on the
        # TP-fractured stream (distributed LayerNorm with per-device adaLN slices from a
        # column-parallel norm_out_linear) and proj_out gathers D inside a strided all-gather-matmul
        # against its replicated weight, so the standalone D gather disappears. proj_out's weight
        # stays replicated either way: its 64 output columns are too narrow to fracture.
        self.fused_tp = is_fused_tp(parallel_config, ccl_manager)
        if self.fused_tp:
            self.norm_out_linear = ColParallelLinear(
                self.inner_dim,
                2 * self.inner_dim,
                bias=True,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            )
            self.norm_out_norm = DistributedLayerNorm(
                self.inner_dim,
                norm_eps=1e-6,
                norm_elementwise_affine=False,
                bias=False,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
            )
        else:
            self.norm_out_linear = Linear(self.inner_dim, 2 * self.inner_dim, mesh_device=mesh_device)
            self.norm_out_norm = LayerNorm(
                self.inner_dim, norm_elementwise_affine=False, norm_eps=1e-6, mesh_device=mesh_device
            )
        self.proj_out = Linear(self.inner_dim, patch_size * patch_size * self.out_channels, mesh_device=mesh_device)

        self.hifi_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "norm_out.linear", "norm_out_linear")
        if self.fused_tp:
            # [scale | shift] fractured on TP so each device's chunk_time slices are its own D slice.
            prepare_chunked_linear_output(
                state,
                prefix="norm_out_linear",
                device_count=self.parallel_config.tensor_parallel.factor,
                chunks=2,
            )

    def _proj_out_fused(self, spatial_flat_11MD: ttnn.Tensor) -> ttnn.Tensor:
        """proj_out on the TP-fractured stream: strided all-gather-matmul against the replicated
        weight (fabric-bound: N = 64) when the shape is swept, else gather + plain matmul.
        Returns the replicated [1, 1, M, out] projection."""
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        weight = self.proj_out.weight.data
        M, K, N_out = spatial_flat_11MD.padded_shape[-2], weight.padded_shape[-2], weight.padded_shape[-1]
        fabric_cfg = get_fabric_agmm_config(M, K, N_out, 1, self.mesh_device.compute_with_storage_grid_size())
        if fabric_cfg is None:
            gathered = self.ccl_manager.all_gather_persistent_buffer(spatial_flat_11MD, dim=3, mesh_axis=tp_axis)
            return self.proj_out(gathered, compute_kernel_config=self.hifi_compute_kernel_config)

        dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        matmul_config = ttnn.MinimalMatmulConfig(
            M_block_size=fabric_cfg.M_block_size,
            K_block_size=fabric_cfg.K_block_size,
            N_block_size=fabric_cfg.N_block_size,
            subblock_h=fabric_cfg.subblock_h,
            subblock_w=fabric_cfg.subblock_w,
            compute_with_storage_grid_size=fabric_cfg.mm_core_grid,
        )
        outputs = ttnn.experimental.strided_all_gather_minimal_matmul_async(
            spatial_flat_11MD,
            weight,
            persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                spatial_flat_11MD.shape, 3, tp_axis, dtype=spatial_flat_11MD.get_dtype()
            ),
            dim=3,
            multi_device_global_semaphore=self.ccl_manager.get_strided_ag_mm_semaphore(
                tp_axis, fabric_cfg.num_workers_per_link
            ),
            strided_all_gather_core_grid_offset=fabric_cfg.ag_core_grid_offset,
            num_links=self.ccl_manager.num_links,
            memory_config_ag=dram,
            topology=self.ccl_manager.topology,
            cluster_axis=tp_axis,
            bias=self.proj_out.bias.data if self.proj_out.bias is not None else None,
            config=matmul_config,
            memory_config_mm=dram,
            compute_kernel_config=self.hifi_compute_kernel_config,
            num_workers_per_link=fabric_cfg.num_workers_per_link,
            num_buffers_per_channel=fabric_cfg.num_buffers_per_channel,
            read_local_slice_from_input=True,
            chunks=1,
        )
        # Op returns [all_gather_output, matmul_chunk_0]; take the single matmul chunk.
        return outputs[1]

    def forward(self, spatial, prompt_embed, pooled_projections, timestep, N):
        """
        Args:
            spatial: Input spatial tensor (latents) - fractured dim 2 along sp_axis
            prompt_embed: Text prompt embeddings - replicated
            pooled_projections: Pooled text projections - replicated
            timestep: Timestep tensor - replicated
        """
        if not getattr(self, "_logged_shapes", False):
            self._logged_shapes = True
            logger.info(
                f"SD35 DiT input shapes: spatial {tuple(spatial.shape)} prompt {tuple(prompt_embed.shape)} "
                f"pooled {tuple(pooled_projections.shape)} timestep {tuple(timestep.shape)}"
            )
        spatial = self.pos_embed(spatial, already_unfolded=True)

        time_embed = self.time_text_embed(timestep, pooled_projections)
        # silu(temb) once per step; every block's adaLN and norm_out project it.
        time_embed_silu = ttnn.silu(time_embed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        prompt_embed = self.context_embedder(prompt_embed)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            spatial, prompt_embed = block(spatial, prompt_embed, time_embed_silu, N)
        # Final normalization and projection
        spatial_time = self.norm_out_linear(time_embed_silu)
        scale, shift = chunk_time(spatial_time, 2)

        if self.fused_tp:
            spatial = self.norm_out_norm(spatial, dynamic_weight=(1 + scale), dynamic_bias=shift)
            spatial_out = self._proj_out_fused(flatten_batch(spatial))
            return unflatten_batch(spatial_out, spatial.shape)

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial = ttnn.experimental.all_gather_async(
                spatial,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    spatial.shape, 3, self.parallel_config.tensor_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.tensor_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.tensor_parallel.mesh_axis,
                # chunks_per_sync=10,
                # num_workers_per_link=2,
                # num_buffers_per_channel=2,
            )

        spatial = self.norm_out_norm(spatial) * (1 + scale) + shift

        spatial_out = self.proj_out(spatial, compute_kernel_config=self.hifi_compute_kernel_config)

        # NOTE: While we should be able to gather on sequence after norm and proj,
        # it leads to terrible outputs for 2x2sp1tp0. Need to debug.
        # if self.parallel_config.sequence_parallel.factor > 1:
        #     spatial_out = ttnn.experimental.all_gather_async(
        #         spatial_out,
        #         persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
        #             spatial_out.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
        #         ),
        #         dim=2,
        #         multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
        #         num_links=self.ccl_manager.num_links,
        #         topology=self.ccl_manager.topology,
        #         cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
        #         # chunks_per_sync=16,
        #         # num_workers_per_link=3,
        #         # num_buffers_per_channel=2,
        #     )

        return spatial_out

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        # N, H, W, C -> 1, N, (H / P) * (W / P), P * P * C
        batch_size, height, width, channels = latents.shape
        patch = self.patch_size

        if height % patch != 0 or width % patch != 0:
            msg = f"height ({height}) and width ({width}) must be divisible by patch_size ({patch})"
            raise ValueError(msg)

        latents = latents.reshape([batch_size, height // patch, patch, width // patch, patch, channels])
        return latents.transpose(2, 3).flatten(3, 5).flatten(1, 2).unsqueeze(0)

    def unpatchify(self, spatial: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        # 1, N, (H / P) * (W / P), P * P * C -> N, H, W, C
        one, batch_size, _, _ = spatial.shape
        assert one == 1
        patch = self.patch_size

        if height % patch != 0 or width % patch != 0:
            msg = f"height ({height}) and width ({width}) must be divisible by patch_size ({patch})"
            raise ValueError(msg)

        spatial = spatial.reshape([batch_size, height // patch, width // patch, patch, patch, -1])
        return spatial.transpose(2, 3).flatten(3, 4).flatten(1, 2)


class SD35Checkpoint:
    """An SD3.5 checkpoint: fetches weights and builds loaded transformers."""

    def __init__(self, name: str) -> None:
        self._name = name
        torch_transformer = TorchSD3Transformer2DModel.from_pretrained(
            name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        torch_transformer.eval()
        assert isinstance(torch_transformer, TorchSD3Transformer2DModel)
        self._config = torch_transformer.config
        self._state_dict = torch_transformer.state_dict()

        self.num_channels_latents: int = self._config.in_channels
        self.joint_attention_dim: int = self._config.joint_attention_dim
        self.patch_size: int = self._config.patch_size

    def build(
        self,
        *,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
    ) -> SD35Transformer2DModel:
        """Construct an ``SD35Transformer2DModel`` for this checkpoint and load its weights."""
        device = ccl_manager.mesh_device
        c = self._config

        if c.num_attention_heads % parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                c.num_attention_heads,
                c.attention_head_dim,
                parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        quant_config = SD35QuantProfile.from_env()
        if quant_config is not None:
            logger.info(f"SD3.5 DiT quantization enabled: {quant_config}")
            # Device-swept block sizes for the dominant per-device spatial DiT matmuls on the 4-chip
            # 2x2 (11x10 grid) config. These have no swept entry and otherwise fall to the generic
            # 8x8x8 fallback; the swept blockings are ~1.4-1.6x faster in isolation. The 8-tile
            # subblocks require bf16 dest (fp32_dest_acc=False), which every quant profile uses, so
            # they are only registered on a quantized run (the bf16/HiFi path keeps its 4-tile dest).
            register_matmul_configs(
                {
                    "11x10": {
                        (2048, 2432, 3648): (4, 4, 12, (2, 4)),  # to_qkv spatial — 1.64x
                        (2048, 2432, 4864): (4, 8, 16, (1, 8)),  # ff1 spatial    — 1.41x
                        (2048, 4864, 2432): (4, 8, 8, (1, 8)),  # ff2 spatial    — 1.56x
                        (2048, 2432, 1216): (6, 4, 8, (1, 8)),  # to_out spatial — 1.16x
                    },
                }
            )

        model = SD35Transformer2DModel(
            sample_size=c.sample_size,
            patch_size=c.patch_size,
            in_channels=c.in_channels,
            num_layers=c.num_layers,
            attention_head_dim=c.attention_head_dim,
            num_attention_heads=c.num_attention_heads,
            joint_attention_dim=c.joint_attention_dim,
            caption_projection_dim=c.caption_projection_dim,
            pooled_projection_dim=c.pooled_projection_dim,
            out_channels=c.out_channels,
            pos_embed_max_size=c.pos_embed_max_size,
            dual_attention_layers=c.dual_attention_layers,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
            quant_config=quant_config,
        )
        cache.load_model(
            tt_model=model,
            get_torch_state_dict=lambda: self._state_dict,
            model_name=os.path.basename(self._name),
            subfolder="transformer",
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
            mesh_device=device,
        )
        return model
