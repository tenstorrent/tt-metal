# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Adapted from models/tt_dit/models/transformers/transformer_flux1.py
#
# Delta vs Flux1Transformer:
#   - No pooled-projection input (LongCat has no CLIP/T5 pooled embed)
#   - No guidance embedding (LongCat checkpoint has guidance_embeds: false)
#   - Time embed uses SD35CombinedTimestepTextProjEmbeddings(pooled_projection_dim=0)
#     instead of CombinedTimestepGuidanceTextProjEmbeddings — functionally identical
#     to LongCat's LongCatImageTimestepEmbeddings (timestep-only sinusoidal → MLP)
#   - No pre-text RMSNorm (QwenImageTransformer adds one; LongCat does not)
#   - Forward signature drops `pooled` and `guidance` args

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from diffusers.models.transformers.transformer_longcat_image import LongCatImageTransformer2DModel

import ttnn

from models.tt_dit.blocks.transformer_block import TransformerBlock, _chunk_time3d
from models.tt_dit.layers.embeddings import SD35CombinedTimestepTextProjEmbeddings
from models.tt_dit.layers.linear import ColParallelLinear, Linear
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.layers.normalization import DistributedLayerNorm
from models.tt_dit.models.transformers.transformer_flux1 import Flux1SingleTransformerBlock
from models.tt_dit.utils import cache
from models.tt_dit.utils.padding import PaddingConfig
from models.tt_dit.utils.substate import rename_substate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from models.tt_dit.parallel.config import DiTParallelConfig
    from models.tt_dit.parallel.manager import CCLManager


class LongCatImageTransformerBlock(TransformerBlock):
    """Dual-stream block for LongCat-Image.

    Identical to TransformerBlock in every way — LongCat's checkpoint key names
    (norm1.linear, norm1.norm, norm1_context.linear, ff.net.0.proj, …) already
    match what TransformerBlock._prepare_torch_state expects, so no renaming is
    needed beyond what the base class already does.
    """


class LongCatImageTransformer(Module):
    """TT port of LongCatImageTransformer2DModel.

    Architecture: dual-stream MMDiT blocks (×num_layers) followed by single-stream
    blocks (×num_single_layers), identical to Flux.1 structurally.  The only
    differences from Flux1Transformer are the time embedding (no pooled projection,
    no guidance) and the absence of a pre-text RMSNorm.
    """

    sdpa_chunk_size_map = {
        (2, 4): (128, 512),
        (8, 4): (128, 256),
        (2, 2): (128, 512),
        (8, 2): (64, 512),
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
        out_channels: int,
        axes_dims_rope: Sequence[int],
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.patch_size = patch_size
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        q_chunk_size, k_chunk_size = self.sdpa_chunk_size_map.get(
            (
                self.parallel_config.sequence_parallel.factor,
                self.parallel_config.tensor_parallel.factor,
            ),
            self.default_sdpa_chunk_size,
        )

        # Timestep-only embedding: sinusoidal(256) → MLP → inner_dim.
        # pooled_projection_dim=0 disables the text_embedder branch, leaving only
        # the timestep path — matching LongCatImageTimestepEmbeddings exactly.
        self.time_text_embed = SD35CombinedTimestepTextProjEmbeddings(
            embedding_dim=inner_dim,
            pooled_projection_dim=0,
            mesh_device=mesh_device,
        )

        # Image token projection: 64 (packed 2×2 latent patches) → inner_dim
        self.x_embedder = ColParallelLinear(
            in_channels,
            inner_dim,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        # Text token projection: joint_attention_dim (3584, Qwen hidden) → inner_dim
        # No pre-norm here — LongCat does not have the RMSNorm that QwenImageTransformer adds.
        self.context_embedder = ColParallelLinear(
            joint_attention_dim,
            inner_dim,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        self.transformer_blocks = ModuleList(
            LongCatImageTransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_pre_only=False,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                attention_k_chunk_size=k_chunk_size,
                attention_q_chunk_size=q_chunk_size,
            )
            for _ in range(num_layers)
        )

        self.single_transformer_blocks = ModuleList(
            Flux1SingleTransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                attention_k_chunk_size=k_chunk_size,
                attention_q_chunk_size=q_chunk_size,
            )
            for _ in range(num_single_layers)
        )

        # Output head: AdaLayerNorm equivalent split into norm + affine-from-temb linear.
        # time_embed_out projects temb → (scale, shift) applied after norm_out.
        self.time_embed_out = Linear(
            inner_dim,
            2 * inner_dim,
            mesh_device=mesh_device,
        )

        self.norm_out = DistributedLayerNorm(
            inner_dim,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.proj_out = Linear(
            inner_dim,
            patch_size * patch_size * out_channels,
            mesh_device=mesh_device,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # LongCat names its timestep embedder `time_embed`; the TT module is
        # SD35CombinedTimestepTextProjEmbeddings, registered as `time_text_embed`.
        rename_substate(state, "time_embed", "time_text_embed")
        # LongCat's norm_out is AdaLayerNormContinuous which stores its linear
        # and norm as sub-modules norm_out.linear and norm_out.norm — same split
        # as Flux1 after this rename.
        rename_substate(state, "norm_out.linear", "time_embed_out")
        rename_substate(state, "norm_out.norm", "norm_out")

    def forward(
        self,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,
    ) -> ttnn.Tensor:
        """Run the model forward.

        Args:
            spatial: [batch, spatial_seq / sp_factor, in_channels] — packed image latents.
                     For the edit pipeline this is cat(noisy_latents, ref_latents) and the
                     caller slices off the noisy portion from the output.
            prompt:  [batch, prompt_seq, joint_attention_dim] — Qwen2.5-VL hidden states.
                     For the edit pipeline this includes the vision tokens as well.
            timestep: [batch, 1] — current denoising timestep / 1000.
            spatial_rope: RoPE (cos, sin) for image tokens.
            prompt_rope:  RoPE (cos, sin) for text tokens.
            spatial_sequence_length: un-sharded spatial sequence length.
            prompt_sequence_length:  un-sharded prompt sequence length.
        """
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        # Timestep embedding (no pooled projection, no guidance).
        time_embed = self.time_text_embed(timestep=timestep)
        ttnn.silu(time_embed, output_tensor=time_embed)
        time_embed = time_embed.reshape([time_embed.shape[-2], 1, time_embed.shape[-1]])

        # Project inputs into inner_dim.
        spatial = self.x_embedder(spatial)
        prompt = self.context_embedder(prompt)

        # 10 dual-stream blocks: spatial and prompt attend jointly, update separately.
        for block in self.transformer_blocks:
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                spatial_sequence_length=spatial_sequence_length,
                skip_time_embed_activation_fn=True,
            )

        prompt = ttnn.clone(prompt, dtype=spatial.dtype)

        # 20 single-stream blocks: spatial and prompt fused into one stream.
        for block in self.single_transformer_blocks:
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                spatial_sequence_length=spatial_sequence_length,
                skip_time_embed_activation_fn=True,
            )

        # Output head: LayerNorm + timestep-conditioned affine + proj_out.
        spatial = ttnn.squeeze(self.norm_out(ttnn.unsqueeze(spatial, 0)), 0)

        spatial_time = self.time_embed_out(time_embed)
        [scale, shift] = _chunk_time3d(spatial_time, 2)

        spatial = self.ccl_manager.all_gather_persistent_buffer(spatial, dim=2, mesh_axis=tp_axis, use_hyperparams=True)

        spatial = spatial * (1 + scale) + shift

        return self.proj_out(spatial)


class LongCatImageCheckpoint:
    """Loads a LongCat-Image (or LongCat-Image-Edit) checkpoint and builds the TT transformer."""

    def __init__(self, name: str) -> None:
        self._name = name
        torch_transformer = LongCatImageTransformer2DModel.from_pretrained(
            name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        torch_transformer.eval()
        self._config = torch_transformer.config
        self._state_dict = torch_transformer.state_dict()

        # RoPE pos-embed runs on CPU as a pre-processing helper; keep the reference.
        self.pos_embed = torch_transformer.pos_embed
        self.patch_size: int = self._config.patch_size
        # in_channels=64 is the packed latent width (16 channels × 2×2 spatial patch).
        self.num_channels_latents: int = self._config.in_channels // 4
        self.joint_attention_dim: int = self._config.joint_attention_dim

    def build(
        self,
        *,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
    ) -> LongCatImageTransformer:
        """Construct and weight-load a LongCatImageTransformer for this checkpoint."""
        device = ccl_manager.mesh_device
        c = self._config

        padding_config = (
            PaddingConfig.from_tensor_parallel_factor(
                c.num_attention_heads,
                c.attention_head_dim,
                parallel_config.tensor_parallel.factor,
            )
            if c.num_attention_heads % parallel_config.tensor_parallel.factor != 0
            else None
        )

        model = LongCatImageTransformer(
            patch_size=c.patch_size,
            in_channels=c.in_channels,
            num_layers=c.num_layers,
            num_single_layers=c.num_single_layers,
            attention_head_dim=c.attention_head_dim,
            num_attention_heads=c.num_attention_heads,
            joint_attention_dim=c.joint_attention_dim,
            out_channels=c.in_channels,
            axes_dims_rope=c.axes_dims_rope,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
        )

        cache.load_model(
            model,
            get_torch_state_dict=lambda: self._state_dict,
            model_name=os.path.basename(self._name),
            subfolder="transformer",
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
        )

        return model
