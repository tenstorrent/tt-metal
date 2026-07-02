# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from diffusers import BriaFiboTransformer2DModel
from diffusers.configuration_utils import FrozenDict

import ttnn
from models.common.utility_functions import is_blackhole
from models.tt_dit.blocks.transformer_block import TransformerBlock, _chunk_time3d
from models.tt_dit.layers.embeddings import TimestepEmbedding, Timesteps
from models.tt_dit.layers.linear import ColParallelLinear, Linear
from models.tt_dit.layers.module import Module, ModuleList
from models.tt_dit.layers.normalization import DistributedLayerNorm
from models.tt_dit.models.transformers.transformer_flux1 import Flux1SingleTransformerBlock
from models.tt_dit.utils import cache
from models.tt_dit.utils.padding import PaddingConfig
from models.tt_dit.utils.substate import rename_substate
from models.tt_dit.utils.tensor import bf16_tensor

if TYPE_CHECKING:
    from models.tt_dit.parallel.config import DiTParallelConfig
    from models.tt_dit.parallel.manager import CCLManager


class FiboTimestepEmbedding(Module):
    """FIBO's combined timestep embedding.

    Sinusoidal timestep projection feeding a two-layer MLP. FIBO does not use pooled text
    projections or distilled guidance, so this is strictly simpler than
    ``CombinedTimestepGuidanceTextProjEmbeddings``.
    """

    def __init__(self, *, embedding_dim: int, mesh_device: ttnn.MeshDevice) -> None:
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, cos_first=True, mesh_device=mesh_device)
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, mesh_device=mesh_device
        )

    def forward(self, *, timestep: ttnn.Tensor) -> ttnn.Tensor:
        return self.timestep_embedder(self.time_proj(timestep))


class FiboCaptionProjection(Module):
    """Per-block projection of a SmolLM3 hidden state into the DiT prompt's upper-half channels.

    Diffusers reference: ``BriaFiboTextProjection`` (a ``Linear(text_encoder_dim, inner_dim // 2)``).

    For TP-correct DimFusion without an extra all_gather / scatter, this widens the linear from
    ``inner_dim // 2`` to ``inner_dim`` output channels and pads the loaded weight with zeros in
    the lower ``inner_dim // 2`` rows. The natural ColParallel shard then places only-zero outputs
    on the lower-half TP devices and the real projection on the upper-half TP devices, so the
    per-block injection reduces to ``prompt * mask + caption_projection(text_layer)`` — a local
    elementwise operation. This mirrors the "re-fuse weights so the natural shard matches the
    expected layout" trick used by ``_re_fuse_proj_out_weight`` in ``transformer_flux1``.
    """

    def __init__(
        self,
        *,
        text_encoder_dim: int,
        inner_dim: int,
        mesh_device: ttnn.MeshDevice,
        mesh_axis: int,
    ) -> None:
        super().__init__()
        self._inner_dim = inner_dim
        self.linear = ColParallelLinear(
            text_encoder_dim,
            inner_dim,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Diffusers stores this as ``Linear(text_encoder_dim, inner_dim // 2)``; pad to
        # ``Linear(text_encoder_dim, inner_dim)`` with zeros in the lower-half output channels.
        # The inner ``ColParallelLinear`` then transposes and TP-shards as usual.
        key = "linear.weight"
        if key in state:
            weight = state[key]
            assert weight.shape[0] == self._inner_dim // 2, (
                f"caption projection weight has out_features {weight.shape[0]}, " f"expected {self._inner_dim // 2}"
            )
            state[key] = torch.cat([torch.zeros_like(weight), weight], dim=0)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.linear(x)


class FiboTransformer(Module):
    sdpa_chunk_size_map = {
        (False, 2, 4): (128, 512),
        (False, 8, 4): (128, 256),
        (True, 2, 2): (128, 512),
        (True, 8, 4): (64, 512),
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
        text_encoder_dim: int,
        out_channels: int,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
        parallel_config: DiTParallelConfig,
        padding_config: PaddingConfig | None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim

        self.patch_size = patch_size
        self.inner_dim = inner_dim
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        q_chunk_size, k_chunk_size = self.sdpa_chunk_size_map.get(
            (
                is_blackhole(),
                self.parallel_config.sequence_parallel.factor,
                self.parallel_config.tensor_parallel.factor,
            ),
            self.default_sdpa_chunk_size,
        )

        self.time_embed = FiboTimestepEmbedding(embedding_dim=inner_dim, mesh_device=mesh_device)

        self.context_embedder = ColParallelLinear(
            joint_attention_dim,
            inner_dim,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        self.x_embedder = ColParallelLinear(
            in_channels,
            inner_dim,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        self.transformer_blocks = ModuleList(
            TransformerBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                context_pre_only=False,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                mesh_device=mesh_device,
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
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                padding_config=padding_config,
                mesh_device=mesh_device,
                attention_k_chunk_size=k_chunk_size,
                attention_q_chunk_size=q_chunk_size,
            )
            for _ in range(num_single_layers)
        )

        self.caption_projection = ModuleList(
            FiboCaptionProjection(
                text_encoder_dim=text_encoder_dim,
                inner_dim=inner_dim,
                mesh_device=mesh_device,
                mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            )
            for _ in range(num_layers + num_single_layers)
        )

        # DimFusion lower-half mask, TP-sharded along the last dim. See ``FiboCaptionProjection``
        # and ``_dimfusion_inject``.
        torch_mask = torch.cat(
            [torch.ones(inner_dim // 2, dtype=torch.float32), torch.zeros(inner_dim // 2, dtype=torch.float32)]
        ).reshape(1, 1, inner_dim)
        self._dimfusion_mask = bf16_tensor(
            torch_mask,
            device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            shard_dim=-1,
        )

        self.time_embed_out = Linear(inner_dim, 2 * inner_dim, mesh_device=mesh_device)

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
        rename_substate(state, "norm_out.linear", "time_embed_out")
        rename_substate(state, "norm_out.norm", "norm_out")

    def forward(
        self,
        *,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        text_encoder_layers: list[ttnn.Tensor],
        timestep: ttnn.Tensor,
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,  # noqa: ARG002 — sized by block via prompt shape
    ) -> ttnn.Tensor:
        """Run the model forward.

        Args:
            spatial: Tensor with shape [batch, spatial_sequence_length / sp_factor, in_channels].
            prompt: Tensor with shape [batch, prompt_sequence_length, joint_attention_dim]. FIBO
                builds this in the pipeline as ``concat(last_hidden_state, second_to_last)``.
            text_encoder_layers: Per-block SmolLM3 hidden states, padded/trimmed to exactly
                ``num_layers + num_single_layers`` entries. Each has shape
                [batch, prompt_sequence_length, text_encoder_dim].
            timestep: Tensor with shape [batch, 1].
            spatial_rope, prompt_rope: Cos/sin tuples for 3-axis RoPE.
        """
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis
        expected_layers = len(self.transformer_blocks) + len(self.single_transformer_blocks)
        assert (
            len(text_encoder_layers) == expected_layers
        ), f"text_encoder_layers must have {expected_layers} entries, got {len(text_encoder_layers)}"

        time_embed = self.time_embed(timestep=timestep)
        ttnn.silu(time_embed, output_tensor=time_embed)
        time_embed = time_embed.reshape([time_embed.shape[-2], 1, time_embed.shape[-1]])

        spatial = self.x_embedder(spatial)
        prompt = self.context_embedder(prompt)

        # Mirrors the diffusers reference's ``new_text_encoder_layers`` pre-pass.
        projected_layers = [self.caption_projection[i](layer) for i, layer in enumerate(text_encoder_layers)]

        block_id = 0
        for block in self.transformer_blocks:
            prompt = self._dimfusion_inject(prompt, projected_layers[block_id])
            block_id += 1
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

        for block in self.single_transformer_blocks:
            prompt = self._dimfusion_inject(prompt, projected_layers[block_id])
            block_id += 1
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                spatial_sequence_length=spatial_sequence_length,
                skip_time_embed_activation_fn=True,
            )

        spatial = ttnn.squeeze(self.norm_out(ttnn.unsqueeze(spatial, 0)), 0)

        spatial_time = self.time_embed_out(time_embed)
        [scale, shift] = _chunk_time3d(spatial_time, 2)

        spatial = self.ccl_manager.all_gather_persistent_buffer(spatial, dim=2, mesh_axis=tp_axis, use_hyperparams=True)

        spatial = spatial * (1 + scale) + shift

        return self.proj_out(spatial)

    def _dimfusion_inject(self, prompt: ttnn.Tensor, projected_text_layer: ttnn.Tensor) -> ttnn.Tensor:
        """Replace the upper half of ``prompt``'s channel axis with a per-block projection of the
        corresponding SmolLM3 hidden state. Diffusers reference:

            encoder_hidden_states = torch.cat(
                [encoder_hidden_states[:, :, : inner_dim // 2], caption_projection[i](layer_i)],
                dim=-1,
            )

        Realized here as a local elementwise op: the caption projection's weights are zero-padded
        so its output is only nonzero in the upper-half TP shards, and a fixed mask zeros out the
        upper-half of ``prompt`` on those same shards. Adding the two yields the concat at no
        communication cost.
        """
        return prompt * self._dimfusion_mask + projected_text_layer

    def patchify(self, latents: torch.Tensor) -> torch.Tensor:
        """Patchify a B,H,W,C latent into B,(H/P)*(W/P),C*P*P tokens."""
        batch_size, height, width, channels = latents.shape
        patch = self.patch_size

        if height % patch != 0 or width % patch != 0:
            msg = f"height ({height}) and width ({width}) must be divisible by patch_size ({patch})"
            raise ValueError(msg)

        latents = latents.reshape([batch_size, height // patch, patch, width // patch, patch, channels])
        return latents.permute(0, 1, 3, 5, 2, 4).flatten(3, 5).flatten(1, 2)

    def unpatchify(self, spatial: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        """Inverse of ``patchify``."""
        batch_size, _, _ = spatial.shape
        patch = self.patch_size

        if height % patch != 0 or width % patch != 0:
            msg = f"height ({height}) and width ({width}) must be divisible by patch_size ({patch})"
            raise ValueError(msg)

        spatial = spatial.reshape([batch_size, height // patch, width // patch, -1, patch, patch])
        return spatial.permute(0, 1, 4, 2, 5, 3).flatten(3, 4).flatten(1, 2)


class FiboCheckpoint:
    """A FIBO transformer checkpoint: fetches weights and builds a loaded ``FiboTransformer``.

    Reads only ``config.json`` in ``__init__``; the actual torch weights are loaded lazily, on
    ``build()`` cache-miss only. Mirrors ``SmolLm3Checkpoint``.
    """

    def __init__(self, name: str) -> None:
        # Internal import: ``BriaFiboEmbedND`` is not re-exported from ``diffusers`` but the file
        # path is stable across versions. The class has no learnable parameters — only the rope
        # axes_dim and theta — so we can construct it from config alone, no torch weights needed.
        from diffusers.models.transformers.transformer_bria_fibo import BriaFiboEmbedND

        config = FrozenDict(BriaFiboTransformer2DModel.load_config(name, subfolder="transformer"))
        self._name = name
        self._config = config

        # CPU-side positional embedding helper. Kept on the checkpoint so the pipeline can prepare
        # ropes once per call and ship them to device, without instantiating the (8B parameter)
        # transformer torch model.
        self.pos_embed = BriaFiboEmbedND(theta=config.rope_theta, axes_dim=config.axes_dims_rope)

        self.num_blocks: int = config.num_layers + config.num_single_layers
        self.latent_channels: int = config.in_channels

    def build(
        self,
        *,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
    ) -> FiboTransformer:
        """Construct a ``FiboTransformer`` for this checkpoint and load its weights."""
        device = ccl_manager.mesh_device
        config = self._config

        if config.num_attention_heads % parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                config.num_attention_heads,
                config.attention_head_dim,
                parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        model = FiboTransformer(
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            num_layers=config.num_layers,
            num_single_layers=config.num_single_layers,
            attention_head_dim=config.attention_head_dim,
            num_attention_heads=config.num_attention_heads,
            joint_attention_dim=config.joint_attention_dim,
            text_encoder_dim=config.text_encoder_dim,
            out_channels=config.in_channels,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
        )
        cache.load_model(
            model,
            get_torch_state_dict=self._load_state_dict,
            model_name=self._name,
            subfolder="transformer",
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
        )
        return model

    def _load_state_dict(self) -> dict[str, torch.Tensor]:
        torch_model = BriaFiboTransformer2DModel.from_pretrained(
            self._name,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        return torch_model.state_dict()
