# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import torch
from diffusers import WanTransformer3DModel as TorchWanTransformer3DModel
from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed as TorchWanRotaryPosEmbed
from loguru import logger

if not os.environ.get("TT_DIT_DEBUG"):
    logger.disable(__name__)


import ttnn

from ....layers.embeddings import WanPatchEmbed, WanTimeTextImageEmbedding
from ....layers.feedforward import ParallelFeedForward
from ....layers.linear import Linear
from ....layers.module import Module, ModuleList, Parameter
from ....layers.normalization import DistributedLayerNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager
from ....utils import cache
from ....utils.mochi import get_rot_transformation_mat
from ....utils.padding import pad_vision_seq_parallel
from ....utils.substate import pop_substate, rename_substate
from ....utils.tensor import bf16_tensor, from_torch, unflatten
from ....utils.tracing import traced_function
from .attention_wan import WanAttention


class WanTransformerBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        cross_attention_norm: bool = True,
        eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = False,
        sdpa_chunk_size_overrides: dict | None = None,
        lora_enabled: bool = False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.cross_attention_norm = cross_attention_norm
        self.eps = eps

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        self.norm1 = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.attn1 = WanAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            is_self=True,
            sdpa_chunk_size_overrides=sdpa_chunk_size_overrides,
            lora_enabled=lora_enabled,
        )

        self.attn2 = WanAttention(
            dim=dim,
            num_heads=num_heads,
            eps=eps,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            is_self=False,
            sdpa_chunk_size_overrides=sdpa_chunk_size_overrides,
            lora_enabled=lora_enabled,
        )

        self.norm2 = (
            DistributedLayerNorm(
                dim,
                norm_eps=eps,
                norm_elementwise_affine=True,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
            )
            if cross_attention_norm
            else None
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
            lora_enabled=lora_enabled,
        )

        self.norm3 = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.scale_shift_table = Parameter(
            total_shape=[1, 1, 6, dim],
            mesh_axes=[None, None, None, parallel_config.tensor_parallel.mesh_axis],
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

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "ffn.net.0.proj", "ffn.ff1")
        rename_substate(state, "ffn.net.2", "ffn.ff2")

        if "scale_shift_table" in state:
            state["scale_shift_table"] = state["scale_shift_table"].unsqueeze(0)

    def forward(
        self,
        spatial_1BND: ttnn.Tensor,
        prompt_1BLP: ttnn.Tensor,
        temb_1BTD: ttnn.Tensor,
        N: int,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        cross_attn_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        spatial_1BND: fractured N on SP, fractured D on TP
        prompt_1BLP: replicated on SP, replicated D on TP
        temb_1BTD: replicated on SP, fractured D on TP
        N: logical sequence length of the spatial input
        rope_cos_BANH: fractured N on SP, A (num_heads) on TP
        rope_sin_BANH: fractured N on SP, A (num_heads) on TP
        trans_mat: replicated on SP, replicated D on TP

        Outputs:
        spatial_1BND: fractured N on SP, fractured D on TP
        """

        # Two timestep layouts are supported, discriminated by width rather than by an extra
        # argument, so `combined_step`'s traced signature is untouched:
        #   scalar   temb (1, B, 6, D/tp) -- one timestep per batch  (T2V, 14B I2V)
        #   per-token temb (1, B, N, 6*D/tp) -- one timestep per token (TI2V-5B I2V)
        # The per-token arm works because time_proj's per-device output is already laid out
        # group-major [g0|g1|...|g5] (see WanTimeTextImageEmbedding._prepare_torch_state), which
        # is exactly the row-major flattening of the (1,1,6,D/tp) table.
        table_width = self.scale_shift_table.data.shape[-1]
        per_token = temb_1BTD.shape[-1] != table_width

        if not per_token:
            assert temb_1BTD.shape[2] == 6, "wan2.2 14b expects 6 chunks in timestep embedding"
            shifted_temb_1BTD = self.scale_shift_table.data + temb_1BTD
            chunk_dim = 2
        else:
            assert (
                temb_1BTD.shape[-1] == 6 * table_width
            ), f"per-token timestep embedding must be 6*{table_width} wide, got {temb_1BTD.shape[-1]}"
            sst_flat = ttnn.reshape(self.scale_shift_table.data, (1, 1, 1, 6 * table_width))
            shifted_temb_1BTD = sst_flat + temb_1BTD
            chunk_dim = 3

        shift_msa_1B1D, scale_msa_1B1D, gate_msa_1B1D, c_shift_msa_1B1D, c_scale_msa_1B1D, c_gate_msa_1B1D = ttnn.chunk(
            shifted_temb_1BTD, 6, dim=chunk_dim
        )

        # NOTE: workaround - addcmul (fused and unfused) is less accurate with fp32 gate input
        gate_msa_1B1D = ttnn.typecast(gate_msa_1B1D, dtype=ttnn.bfloat16)
        c_gate_msa_1B1D = ttnn.typecast(c_gate_msa_1B1D, dtype=ttnn.bfloat16)

        spatial_normed_1BND = self.norm1(
            spatial_1BND, dynamic_weight=(1.0 + scale_msa_1B1D), dynamic_bias=shift_msa_1B1D
        )

        # Self attention on spatial with fused residual addcmul
        # Fuses: spatial_1BND = spatial_1BND + to_out(attn_output) * gate_msa_1B1D
        spatial_1BND = self.attn1(
            spatial_1BND=spatial_normed_1BND,
            N=N,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            trans_mat=trans_mat,
            addcmul_residual=spatial_1BND,
            addcmul_gate=gate_msa_1B1D,
        )

        # Cross attention on prompt
        spatial_normed_1BND = self.norm2(spatial_1BND)

        attn_output_1BND = self.attn2(
            spatial_1BND=spatial_normed_1BND,
            N=N,
            prompt_1BLP=prompt_1BLP,
            cross_attn_mask=cross_attn_mask,
        )
        spatial_1BND = spatial_1BND + attn_output_1BND

        # Feed Forward
        spatial_normed_1BND = self.norm3(
            spatial_1BND, dynamic_weight=(1.0 + c_scale_msa_1B1D), dynamic_bias=c_shift_msa_1B1D
        )

        if self.ccl_manager.topology == ttnn.Topology.Linear:
            if self.parallel_config.tensor_parallel.factor > 1:
                spatial_normed_1BND = self.ccl_manager.all_gather_persistent_buffer(
                    spatial_normed_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
                )

            spatial_ff_1BND = self.ffn(spatial_normed_1BND, compute_kernel_config=self.ff_compute_kernel_config)
            # spatial_1BND = spatial_1BND + spatial_ff_1BND * c_gate_msa_1B1D
            # NOTE: higher precision compute config in addcmul may be needed for correctness
            spatial_1BND = ttnn.addcmul(spatial_1BND, spatial_ff_1BND, c_gate_msa_1B1D)
        else:
            # Fused FFN + addcmul at the RS final write step (Phase 2).
            # Both 'a' (spatial_1BND) and 'b' (c_gate_msa_1B1D) are already at [D/tp] size
            # on each TP device — no AllGather or scatter matmul needed.
            spatial_1BND = self.ffn.forward_fused_addcmul(
                spatial_normed_1BND,
                spatial_1BND,
                c_gate_msa_1B1D,
                scalar=1.0,
                compute_kernel_config=self.ff_compute_kernel_config,
                parallel_config=self.parallel_config,
            )

        return spatial_1BND


class WanTransformer3DModel(Module):
    def __init__(
        self,
        *,
        patch_size: tuple = (1, 2, 2),
        num_heads: int = 40,
        dim: int = 5120,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        rope_max_seq_len: int = 1024,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None = None,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool = True,
        model_type: str = "t2v",
        output_dtype: ttnn.DataType = ttnn.float32,
        lora_enabled: bool = False,
        sdpa_chunk_size_overrides: dict | None = None,
    ) -> None:
        super().__init__()

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.is_fsdp = is_fsdp
        self.lora_enabled = lora_enabled
        self.fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None
        self.model_type = model_type
        self.cached_rope_features = {}
        self.output_dtype = output_dtype

        assert model_type in ["t2v", "i2v", "ti2v"], "model_type must be t2v, i2v, or ti2v"
        if model_type == "i2v":
            in_channels = 36
        elif model_type == "ti2v":
            assert in_channels == 48, "in_channels must be 48 for ti2v (Wan2.2-VAE latents)"
        else:
            assert in_channels == 16, "in_channels must be 16 for t2v"

        self.patch_size = patch_size
        self.dim = dim

        self.out_channels = out_channels

        # NOTE: Fallback
        self.rope = TorchWanRotaryPosEmbed(
            dim // num_heads,
            patch_size,
            rope_max_seq_len,
        )

        self.patch_embedding = WanPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=dim,
            mesh_device=mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=dim,
            time_freq_dim=freq_dim,
            time_proj_dim=dim * 6,
            text_embed_dim=text_dim,
            mesh_device=self.mesh_device,
            tp_mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.blocks = ModuleList(
            WanTransformerBlock(
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                cross_attention_norm=cross_attn_norm,
                eps=eps,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                is_fsdp=is_fsdp,
                lora_enabled=lora_enabled,
                sdpa_chunk_size_overrides=sdpa_chunk_size_overrides,
            )
            for i in range(num_layers)
        )

        self.norm_out = DistributedLayerNorm(
            dim,
            norm_eps=eps,
            norm_elementwise_affine=False,
            bias=False,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
        )

        self.proj_out = Linear(
            dim,
            patch_size[0] * patch_size[1] * patch_size[2] * self.out_channels,
            bias=True,
            mesh_device=mesh_device,
        )

        # Installed by set_per_token_timestep_masks() when a variant drives the transformer with
        # a two-row timestep; None on every scalar-timestep path.
        self._per_token_temb_mask = None
        self._per_token_proj_mask = None

        self.scale_shift_table = Parameter(
            total_shape=[1, 2, dim],
            device=mesh_device,
            mesh_axes=[None, None, parallel_config.tensor_parallel.mesh_axis],
            dtype=ttnn.float32,
        )

        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def save(self, directory: str | Path, /, *, prefix: str = "") -> None:
        super().save(directory, prefix=prefix)

        directory = Path(directory)

        # Torch fallbacks
        torch.save(self.rope.state_dict(), directory / f"{prefix}rope.pt")

    def load(self, directory: str | Path, /, *, prefix: str = "") -> None:
        super().load(directory, prefix=prefix)

        directory = Path(directory)

        # Torch fallbacks
        self.rope.load_state_dict(torch.load(directory / f"{prefix}rope.pt"))

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Torch fallbacks
        self.rope.load_state_dict(pop_substate(state, "rope"))

    def get_rope_features(self, hidden_states):
        if tuple(hidden_states.shape) not in self.cached_rope_features:
            rope_features = self.prepare_rope_features(hidden_states)
            self.cached_rope_features[tuple(hidden_states.shape)] = rope_features
        return self.cached_rope_features[tuple(hidden_states.shape)]

    def prepare_rope_features(self, hidden_states):
        """
        Given video input, compute RoPE features.
        Return tensors on device.
        """
        logger.info(f"Preparing rope features for shape {hidden_states.shape}")
        rope_cos, rope_sin = self.rope(hidden_states)

        # Convert to TT tensors with proper sharding
        rope_cos_1HND = rope_cos.permute(0, 2, 1, 3)
        rope_sin_1HND = rope_sin.permute(0, 2, 1, 3)

        rope_cos_1HND = pad_vision_seq_parallel(
            rope_cos_1HND, num_devices=self.parallel_config.sequence_parallel.factor
        )
        rope_sin_1HND = pad_vision_seq_parallel(
            rope_sin_1HND, num_devices=self.parallel_config.sequence_parallel.factor
        )

        trans_mat = get_rot_transformation_mat()

        sp_axis = self.parallel_config.sequence_parallel.mesh_axis
        tt_rope_cos_1HND = from_torch(
            rope_cos_1HND,
            device=self.mesh_device,
            dtype=ttnn.float32,
            mesh_axes=[..., sp_axis, None],
        )
        tt_rope_sin_1HND = from_torch(
            rope_sin_1HND,
            device=self.mesh_device,
            dtype=ttnn.float32,
            mesh_axes=[..., sp_axis, None],
        )
        tt_trans_mat = bf16_tensor(trans_mat, device=self.mesh_device)

        logger.info(f"TT rope cos shape: {tt_rope_cos_1HND.shape}")
        logger.info(f"TT rope sin shape: {tt_rope_sin_1HND.shape}")
        logger.info(f"TT trans mat shape: {tt_trans_mat.shape}")

        return tt_rope_cos_1HND, tt_rope_sin_1HND, tt_trans_mat

    def prepare_text_conditioning(self, encoder_hidden_states):
        tt_prompt_1BLP = self.condition_embedder.forward_text(encoder_hidden_states)

        logger.info(f"TT prompt shape: {tt_prompt_1BLP.shape}")
        return tt_prompt_1BLP

    def _apply_norm_out_modulation(self, temb_11BD):
        """Add the model-level scale/shift table to `temb` and split it into (shift, scale).

        The table is (1, 2, D/tp) per device -- row-major [shift_row | scale_row]. `temb` is
        D/tp wide in *both* layouts, so the two are told apart by the token axis, not width:

          scalar    temb (1, 1, B, D/tp), token axis 1 -> broadcast-add, chunk on axis -2
          per-token temb (1, B, N, D/tp), token axis N -> the (1,2,D/tp) table cannot
                    broadcast against N tokens, so flatten it to (1, 1, 1, 2*D/tp) and
                    duplicate temb along the feature axis, which reproduces the same
                    [shift | scale] row-major pairing and chunks on the feature axis.
        """
        table_width = self.scale_shift_table.data.shape[-1]
        if temb_11BD.shape[-2] == 1:
            scale_shift_1BSD = self.scale_shift_table.data + temb_11BD
            return ttnn.chunk(scale_shift_1BSD, 2, -2)

        assert (
            temb_11BD.shape[-1] == table_width
        ), f"per-token norm_out temb must be {table_width} wide, got {temb_11BD.shape[-1]}"
        sst_flat = ttnn.reshape(self.scale_shift_table.data, (1, 1, 1, 2 * table_width))
        duplicated = ttnn.concat([temb_11BD, temb_11BD], dim=-1)
        return ttnn.chunk(sst_flat + duplicated, 2, dim=3)

    def set_per_token_timestep_masks(self, temb_mask, proj_mask) -> None:
        """Install the masks that expand a 2-row timestep embedding to one row per token.

        Persistent device tensors, so a captured trace binds them by address the same way the
        pipeline's latent and conditioning buffers are bound. Each is 0 where the token should
        take the first timestep row and 1 where it should take the second, at the temb and
        timestep-projection feature widths respectively.
        """
        self._per_token_temb_mask = temb_mask
        self._per_token_proj_mask = proj_mask

    @staticmethod
    def _expand_two_row(rows, mask):
        """Select per token between two embedding rows: `row0 + mask * (row1 - row0)`.

        `rows` is (..., 2, W); `mask` is (..., N, W) and binary. Rows are repeated to the token
        count rather than relied on to broadcast, so this does not depend on two-axis broadcast
        semantics for the fused ternary.
        """
        n = mask.shape[-2]
        row0 = ttnn.slice(rows, [0, 0, 0, 0], [rows.shape[0], rows.shape[1], 1, rows.shape[3]])
        row1 = ttnn.slice(rows, [0, 0, 1, 0], [rows.shape[0], rows.shape[1], 2, rows.shape[3]])
        row0 = ttnn.repeat(row0, ttnn.Shape([1, 1, n, 1]))
        row1 = ttnn.repeat(row1, ttnn.Shape([1, 1, n, 1]))
        return ttnn.lerp(row0, row1, mask)

    def prepare_timestep_conditioning(self, timestep):
        """Embed the timestep, for either a scalar or a per-token schedule.

        Three layouts reach this, told apart by the token axis so that `combined_step`'s traced
        signature never changes:

          (B, 1, 1, 1)  scalar, one timestep per batch -- T2V and 14B I2V, unchanged
          (1, 1, 2, 1)  two distinct values, expanded per token via the installed masks
          (1, B, N, 1)  fully per-token (kept working, but runs the MLP N/SP times over)

        TI2V-5B image conditioning only ever needs two values -- 0 on the conditioned frame and
        `t` everywhere else -- and the embedder is pointwise in the token axis, so the 2-row
        form is mathematically identical to the per-token form while running the MLP at M=32
        (tile-padded) instead of M=N/SP. That is the same shape the scalar path already uses, so
        it needs no matmul blocking entries of its own and cannot overflow L1 at any resolution.
        """
        tokens = timestep.shape[-2]
        two_row = tokens == 2 and self._per_token_temb_mask is not None
        per_token = tokens != 1 and not two_row

        tt_temb_11BD, tt_timestep_proj_1BTD = self.condition_embedder.forward_timestep(timestep, timestep_seq_len=None)

        if two_row:
            assert tt_temb_11BD.shape[-2] == 2 and tt_timestep_proj_1BTD.shape[-2] == 2, (
                f"expected 2 embedding rows, got temb {tt_temb_11BD.shape} / " f"proj {tt_timestep_proj_1BTD.shape}"
            )
            tt_temb_11BD = self._expand_two_row(tt_temb_11BD, self._per_token_temb_mask)
            tt_timestep_proj_1BTD = self._expand_two_row(tt_timestep_proj_1BTD, self._per_token_proj_mask)
            return tt_temb_11BD, tt_timestep_proj_1BTD

        if per_token:
            # Leave the projection as (1, B, N, 6*D/tp); WanTransformerBlock.forward detects the
            # wider layout and chunks on the feature axis instead of a dedicated chunk axis.
            logger.info(f"TT per-token timestep proj shape: {tt_timestep_proj_1BTD.shape}")
            return tt_temb_11BD, tt_timestep_proj_1BTD
        tt_timestep_proj_1BTD = unflatten(ttnn.squeeze(tt_timestep_proj_1BTD, -2), -1, (6, -1))
        logger.info(f"TT temb shape: {tt_temb_11BD.shape}")
        logger.info(f"TT timestep proj shape: {tt_timestep_proj_1BTD.shape}")
        return tt_temb_11BD, tt_timestep_proj_1BTD

    def prepare_conditioning(self, timestep, encoder_hidden_states):
        """
        Given inputs, execute the combined timestep and text embedding.
        Return tensors on device.
        """
        tt_temb_11BD, tt_timestep_proj_1BTD = self.prepare_timestep_conditioning(timestep)
        tt_prompt_1BLP = self.prepare_text_conditioning(encoder_hidden_states)

        logger.info(f"TT temb shape: {tt_temb_11BD.shape}")
        logger.info(f"TT timestep proj shape: {tt_timestep_proj_1BTD.shape}")
        logger.info(f"TT prompt shape: {tt_prompt_1BLP.shape}")
        return tt_temb_11BD, tt_timestep_proj_1BTD, tt_prompt_1BLP

    def preprocess_spatial_input_host(self, spatial):
        B, C, F, H, W = spatial.shape
        logger.info(f"Preprocessing spatial input with shape {spatial.shape}")
        assert B == 1, "Batch size must be 1"
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        N = patch_F * patch_H * patch_W

        # Patchify video input
        spatial = spatial.reshape(B, C, patch_F, pF, patch_H, pH, patch_W, pW)
        spatial = spatial.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(1, B, N, pF * pH * pW * C)
        logger.info(f"spatial input after patchifying: {spatial.shape}")

        spatial = pad_vision_seq_parallel(spatial, num_devices=self.parallel_config.sequence_parallel.factor)
        logger.info(f"spatial input after padding: {spatial.shape}")

        return spatial, N

    def preprocess_spatial_input(self, spatial):
        spatial, N = self.preprocess_spatial_input_host(spatial)
        spatial = bf16_tensor(
            spatial, device=self.mesh_device, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis, shard_dim=-2
        )
        logger.info(f"TT spatial shape: {spatial.shape}")
        return spatial, N

    def postprocess_spatial_output_host(self, spatial_1BND, F, H, W, N):
        """
        This is the reverse of preprocess_spatial_input
        Input is of shape: 1 B (patch_F patch_H patch_W) (pF pH pW C)
        returns shape: B C F H W
        """
        assert len(spatial_1BND.shape) == 4
        assert spatial_1BND.shape[0] == 1
        B = spatial_1BND.shape[1]
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        logger.info(f"Postprocessing spatial output with shape {spatial_1BND.shape}")
        spatial_BND = spatial_1BND.squeeze(0)

        spatial_BND = spatial_BND[:, :N]  # Slice out sequence-parallel padding tokens
        logger.info(f"Spatial output after slicing: {spatial_BND.shape}")

        spatial_patches = spatial_BND.reshape(B, patch_F, patch_H, patch_W, pF, pH, pW, self.out_channels)
        logger.info(f"Spatial output after reshaping: {spatial_patches.shape}")

        spatial_BCFHW = spatial_patches.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, self.out_channels, F, H, W)
        logger.info(f"Spatial output after permuting: {spatial_BCFHW.shape}")
        return spatial_BCFHW

    def postprocess_spatial_output(self, spatial_1BND, F, H, W, N):
        """
        This is the reverse of preprocess_spatial_input
        Input is of shape: 1 B (patch_F patch_H patch_W) (pF pH pW C)
        returns shape: B C F H W
        """
        assert len(spatial_1BND.shape) == 4
        assert spatial_1BND.shape[0] == 1
        # Gather sequence-parallel output
        spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
            spatial_1BND, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
        )

        spatial_1BND = ttnn.to_torch(ttnn.get_device_tensors(spatial_1BND)[0])

        spatial_BCFHW = self.postprocess_spatial_output_host(spatial_1BND, F, H, W, N)
        return spatial_BCFHW

    def forward(
        self,
        spatial: torch.Tensor,
        prompt: ttnn.Tensor,
        timestep: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Inputs are all torch tensors
            y is an optional argument for image-to-video generation.
            We assume that preprocessing has already been done for y and y has the same shape as spatial.
        Output is torch tensor
        """

        if self.model_type == "i2v":
            assert y is not None, "y must be provided for image-to-video generation"

        B, C, F, H, W = spatial.shape
        pF, pH, pW = self.patch_size
        patch_F, patch_H, patch_W = F // pF, H // pH, W // pW
        N = patch_F * patch_H * patch_W

        rope_cos_1HND, rope_sin_1HND, trans_mat = self.get_rope_features(spatial)

        temb_11BD, timestep_proj_1BTD, prompt_1BLP = self.prepare_conditioning(timestep, prompt)

        # Concatenate spatial and y along the channel dimension
        if self.model_type == "i2v":
            print(spatial.shape, y.shape)
            spatial = torch.cat([spatial, y], dim=1)

        spatial_1BNI, N = self.preprocess_spatial_input(spatial)

        spatial_1BND = self.patch_embedding(spatial_1BNI)

        for idx, block in enumerate(self.blocks):
            spatial_1BND = block(
                spatial_1BND=spatial_1BND,
                prompt_1BLP=prompt_1BLP,
                temb_1BTD=timestep_proj_1BTD,
                N=N,
                rope_cos=rope_cos_1HND,
                rope_sin=rope_sin_1HND,
                trans_mat=trans_mat,
            )

        shift_11BD, scale_11BD = self._apply_norm_out_modulation(temb_11BD)

        spatial_norm_1BND = self.norm_out(
            spatial_1BND, dynamic_weight=(1 + scale_11BD), dynamic_bias=shift_11BD, dtype=ttnn.float32
        )

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_norm_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_norm_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        proj_out_1BNI = self.proj_out(
            spatial_norm_1BND, compute_kernel_config=self.hifi4_compute_kernel_config, dtype=ttnn.float32
        )

        spatial_out = self.postprocess_spatial_output(proj_out_1BNI, F, H, W, N)

        return spatial_out

    def inner_step(
        self, spatial_1BNI, prompt_1BLP, rope_cos_1HND, rope_sin_1HND, trans_mat, N, timestep, gather_output=True
    ):
        """
        Reduced forward function which assumes outer loop has cached certain inputs that are step independent:
            - prompt_1BLP
            - rope_cos_1HND
            - rope_sin_1HND
            - trans_mat
            - N

        Spatial input is a tensor with layout `1 B (patch_F patch_H patch_W) (pF pH pW C)`.
        Spatial output is an fp32 ttnn.Tensor on device with same layout.
        """
        temb_11BD, timestep_proj_1BTD = self.prepare_timestep_conditioning(timestep)

        spatial_1BND = self.patch_embedding(spatial_1BNI)

        for block in self.blocks:
            spatial_1BND = block(
                spatial_1BND=spatial_1BND,
                prompt_1BLP=prompt_1BLP,
                temb_1BTD=timestep_proj_1BTD,
                N=N,
                rope_cos=rope_cos_1HND,
                rope_sin=rope_sin_1HND,
                trans_mat=trans_mat,
            )
        shift_11BD, scale_11BD = self._apply_norm_out_modulation(temb_11BD)

        spatial_norm_1BND = self.norm_out(
            spatial_1BND, dynamic_weight=(1 + scale_11BD), dynamic_bias=shift_11BD, dtype=ttnn.float32
        )

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_norm_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_norm_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        spatial_1BNI = self.proj_out(
            spatial_norm_1BND, compute_kernel_config=self.hifi4_compute_kernel_config, dtype=self.output_dtype
        )

        # Gather fp32 spatial output across sequence parallel devices (remains on device)
        if gather_output:
            spatial_1BNI = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BNI, dim=2, mesh_axis=self.parallel_config.sequence_parallel.mesh_axis
            )

        return spatial_1BNI

    # Prep run is False because we warmup the entire pipeline first. Remove if this is not desired.
    @traced_function(device=lambda self: self.mesh_device, clone_prep_inputs=False, prep_run=False)
    def combined_step(
        self,
        do_classifier_free_guidance: bool,
        spatial_1BNI: ttnn.Tensor,
        prompt_1BLP: ttnn.Tensor,
        negative_prompt_1BLP: ttnn.Tensor,
        N: int,
        rope_cos_1HND: ttnn.Tensor,
        rope_sin_1HND: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        timestep: ttnn.Tensor,
        guidance_scale: ttnn.Tensor,
        *,
        gather_output: bool = True,
    ) -> ttnn.Tensor:
        cond = self.inner_step(
            spatial_1BNI,
            prompt_1BLP,
            rope_cos_1HND,
            rope_sin_1HND,
            trans_mat,
            N,
            timestep,
            gather_output=gather_output,
        )
        if not do_classifier_free_guidance:
            return cond

        uncond = self.inner_step(
            spatial_1BNI,
            negative_prompt_1BLP,
            rope_cos_1HND,
            rope_sin_1HND,
            trans_mat,
            N,
            timestep,
            gather_output=gather_output,
        )

        combined = ttnn.lerp(uncond, cond, guidance_scale)

        return combined


class WanCheckpoint:
    """A Wan transformer-subfolder checkpoint: fetches weights and builds loaded transformers."""

    def __init__(self, name: str, subfolder: str) -> None:
        self._name = name
        self._subfolder = subfolder
        torch_transformer = TorchWanTransformer3DModel.from_pretrained(
            name,
            subfolder=subfolder,
            trust_remote_code=True,
        )
        torch_transformer.eval()
        self._config = torch_transformer.config
        self._state_dict = torch_transformer.state_dict()

    @property
    def subfolder(self) -> str:
        return self._subfolder

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(self._state_dict)

    def build(
        self,
        *,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool,
        model_type: str,
        lora_enabled: bool = False,
        sdpa_chunk_size_overrides: dict | None = None,
    ) -> WanTransformer3DModel:
        """Construct a ``WanTransformer3DModel`` for this checkpoint (weights NOT loaded).

        Loading is deferred so the caller can manage the lifecycle (deallocate / reload).

        ``sdpa_chunk_size_overrides`` layers on top of the ``WanAttention`` chunk table,
        which is keyed only on ``(is_blackhole, sp, tp)`` and is therefore shared with the
        14B at the same parallelism. Retuning a single variant goes through here so the
        other one does not move.
        """
        c = self._config
        return WanTransformer3DModel(
            patch_size=c.patch_size,
            num_heads=c.num_attention_heads,
            dim=c.num_attention_heads * c.attention_head_dim,
            in_channels=c.in_channels,
            out_channels=c.out_channels,
            text_dim=c.text_dim,
            freq_dim=c.freq_dim,
            ffn_dim=c.ffn_dim,
            num_layers=c.num_layers,
            cross_attn_norm=c.cross_attn_norm,
            eps=c.eps,
            rope_max_seq_len=c.rope_max_seq_len,
            mesh_device=ccl_manager.mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            is_fsdp=is_fsdp,
            model_type=model_type,
            lora_enabled=lora_enabled,
            sdpa_chunk_size_overrides=sdpa_chunk_size_overrides,
        )

    def load(
        self,
        model: WanTransformer3DModel,
        *,
        mesh_device: ttnn.MeshDevice,
        parallel_config: DiTParallelConfig,
        is_fsdp: bool,
    ) -> None:
        """Load (or reload) weights for a previously-built transformer."""
        cache.load_model(
            tt_model=model,
            get_torch_state_dict=lambda: self._state_dict,
            model_name=os.path.basename(self._name),
            subfolder=self._subfolder,
            parallel_config=parallel_config,
            mesh_shape=tuple(mesh_device.shape),
            mesh_device=mesh_device,
            is_fsdp=is_fsdp,
        )
