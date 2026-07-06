# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from diffusers import BriaFiboTransformer2DModel

import ttnn
from models.common.utility_functions import is_blackhole

from ...blocks.transformer_block import TransformerBlock, _chunk_time3d
from ...layers.embeddings import TimestepEmbedding, Timesteps
from ...layers.linear import ColParallelLinear, Linear
from ...layers.module import Module, ModuleList
from ...layers.normalization import DistributedLayerNorm
from ...utils import cache
from ...utils.padding import PaddingConfig
from ...utils.substate import rename_substate
from .transformer_flux1 import Flux1SingleTransformerBlock

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager


class BriaFiboTextProjection(Module):
    """Single linear projection for per-layer caption conditioning.

    Mirrors HF ``BriaFiboTextProjection``:
        linear: Linear(in_features, hidden_size, bias=False)

    State-dict key: ``linear.weight``
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.mesh_device = mesh_device

        self.linear = Linear(in_features, hidden_size, bias=False, mesh_device=mesh_device, dtype=dtype)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.linear(x)


def inject_text(encoder_hidden_states: ttnn.Tensor, projected: ttnn.Tensor) -> ttnn.Tensor:
    """Concat-halves injection for FIBO dual blocks (tp=1).

    Replaces the upper half of the context (prompt) features with the per-block
    projected text embedding:

        out[..., :half] = encoder_hidden_states[..., :half]
        out[..., half:] = projected

    where ``half = inner_dim // 2 = 1536``.

    Args:
        encoder_hidden_states: Context tensor of shape ``[batch, P, inner_dim]``
            (inner_dim = 3072 for FIBO).
        projected: Per-block text projection of shape ``[batch, P, inner_dim // 2]``
            (i.e. ``[batch, P, 1536]``).

    Returns:
        Concatenated tensor of shape ``[batch, P, inner_dim]``.
    """
    half = projected.shape[-1]
    assert (
        encoder_hidden_states.shape[-1] == 2 * half
    ), f"inject_text: expected encoder_hidden_states last dim {2 * half}, got {encoder_hidden_states.shape[-1]}"
    first_half = encoder_hidden_states[:, :, :half]
    return ttnn.concat([first_half, projected], dim=-1)


class BriaFiboTimestepEmbed(Module):
    """Timestep-only embedding for FIBO: sinusoidal → MLP → [batch, inner_dim].

    Mirrors HF ``BriaFiboTimestepProjEmbeddings``:
        time_proj:          BriaFiboTimesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
                            → no learnable parameters
        timestep_embedder:  TimestepEmbedding(256 → inner_dim → inner_dim, act=silu)

    State-dict keys (only from timestep_embedder):
        ``timestep_embedder.linear_1.{weight,bias}``
        ``timestep_embedder.linear_2.{weight,bias}``
    """

    def __init__(
        self,
        inner_dim: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.mesh_device = mesh_device

        # Sinusoidal projection: cos_first=True matches flip_sin_to_cos=True, downscale_freq_shift=0
        self.time_proj = Timesteps(
            num_channels=256,
            cos_first=True,
            downscale_freq_shift=0,
            max_period=10000,
            dtype=dtype,
            mesh_device=mesh_device,
        )

        # Two-layer MLP: 256 → inner_dim → inner_dim
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=inner_dim,
            act_fn="silu",
            dtype=dtype,
            mesh_device=mesh_device,
        )

    def forward(self, timestep: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass.

        Args:
            timestep: [batch, 1] bfloat16 timestep values (raw, not /1000).

        Returns:
            [batch, inner_dim] embedding.
        """
        proj = self.time_proj(timestep)
        return self.timestep_embedder(proj)


# adapted from transformer_flux1.Flux1Transformer for the FIBO (Bria4) architecture.
class BriaFiboTransformer(Module):
    """FIBO MMDiT denoiser (``BriaFiboTransformer2DModel``) in tt_dit.

    Structurally a Flux MMDiT with three FIBO deltas:
      * timestep-only ``time_embed`` (no pooled / no guidance),
      * ``in_channels`` = 48 (``x_embedder``) and ``joint_attention_dim`` = 4096 (``context_embedder``),
      * per-block "concat-halves" text injection via a length-``num_layers+num_single_layers``
        ``caption_projection`` list, indexed by a single ``block_id`` counter spanning both loops.

    The dual (``TransformerBlock``) and single (``Flux1SingleTransformerBlock``) block cores are reused
    unchanged; the injection happens in this forward *before* each block call.
    """

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
        axes_dims_rope: Sequence[int],
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

        # Timestep-only embedding (no pooled / no guidance).
        self.time_embed = BriaFiboTimestepEmbed(inner_dim=inner_dim, mesh_device=mesh_device)

        self.context_embedder = ColParallelLinear(
            joint_attention_dim,
            inner_dim,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        # Shard output, since size of input dimension << size of output dimension.
        self.x_embedder = ColParallelLinear(
            in_channels,
            inner_dim,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
        )

        # Per-block caption projections (2048 -> inner_dim // 2), one per dual + single block.
        self.caption_projection = ModuleList(
            BriaFiboTextProjection(text_encoder_dim, inner_dim // 2, mesh_device=mesh_device)
            for _ in range(num_layers + num_single_layers)
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
        # FIBO's norm_out is AdaLayerNormContinuous: only `norm_out.linear.*` exists (no affine
        # norm params), mapping to `time_embed_out` exactly as in Flux.
        rename_substate(state, "norm_out.linear", "time_embed_out")
        rename_substate(state, "norm_out.norm", "norm_out")

    def forward(
        self,
        spatial: ttnn.Tensor,
        prompt: ttnn.Tensor,
        timestep: ttnn.Tensor,
        text_encoder_layers: Sequence[ttnn.Tensor],
        spatial_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        prompt_rope: tuple[ttnn.Tensor, ttnn.Tensor],
        spatial_sequence_length: int,
        prompt_sequence_length: int,
    ) -> ttnn.Tensor:
        """Run the model forward.

        Args:
            spatial: Tensor [batch, spatial_sequence_length / sp_factor, in_channels].
            prompt: Tensor [batch, prompt_sequence_length, joint_attention_dim] (replicated).
            timestep: Tensor [batch, 1] (raw, not /1000).
            text_encoder_layers: list of ``num_layers + num_single_layers`` tensors,
                each [batch, prompt_sequence_length, text_encoder_dim] (replicated).
            spatial_rope: Tuple of two tensors [spatial_sequence_length / sp_factor, head_dim].
            prompt_rope: Tuple of two tensors [prompt_sequence_length, head_dim] (not sharded!).
        """
        tp_axis = self.parallel_config.tensor_parallel.mesh_axis

        time_embed = self.time_embed(timestep)
        ttnn.silu(time_embed, output_tensor=time_embed)
        time_embed = time_embed.reshape([time_embed.shape[-2], 1, time_embed.shape[-1]])

        spatial = self.x_embedder(spatial)
        prompt = self.context_embedder(prompt)

        block_id = 0

        for block in self.transformer_blocks:
            projected = self.caption_projection[block_id](text_encoder_layers[block_id])
            prompt = inject_text(prompt, projected)
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                spatial_sequence_length=spatial_sequence_length,
                skip_time_embed_activation_fn=True,
            )
            block_id += 1

        prompt = ttnn.clone(prompt, dtype=spatial.dtype)

        for block in self.single_transformer_blocks:
            projected = self.caption_projection[block_id](text_encoder_layers[block_id])
            prompt = inject_text(prompt, projected)
            spatial, prompt = block.forward(
                spatial=spatial,
                prompt=prompt,
                time_embed=time_embed,
                spatial_rope=spatial_rope,
                prompt_rope=prompt_rope,
                spatial_sequence_length=spatial_sequence_length,
                skip_time_embed_activation_fn=True,
            )
            block_id += 1

        spatial = ttnn.squeeze(self.norm_out(ttnn.unsqueeze(spatial, 0)), 0)

        spatial_time = self.time_embed_out(time_embed)
        [scale, shift] = _chunk_time3d(spatial_time, 2)

        spatial = self.ccl_manager.all_gather_persistent_buffer(spatial, dim=2, mesh_axis=tp_axis, use_hyperparams=True)

        spatial = spatial * (1 + scale) + shift

        return self.proj_out(spatial)


class BriaFiboCheckpoint:
    """A FIBO checkpoint: fetches HF ``transformer/`` weights and builds loaded transformers."""

    def __init__(self, name: str) -> None:
        self._name = name

        # Resolve the HF repo id to its local cache path when running offline.
        model_path = name
        if not os.path.isdir(model_path):
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(name, allow_patterns=["transformer/*"], local_files_only=True)

        torch_transformer = BriaFiboTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
        )
        torch_transformer.eval()
        self._config = torch_transformer.config
        self._state_dict = torch_transformer.state_dict()

        # Pos embedding (RoPE) is a CPU-only helper; keep the reference.
        self.pos_embed = torch_transformer.pos_embed
        self.in_channels: int = self._config.in_channels
        self.joint_attention_dim: int = self._config.joint_attention_dim
        self.text_encoder_dim: int = self._config.text_encoder_dim
        self.patch_size: int = self._config.patch_size

    def _filtered_state_dict(self, num_layers: int, num_single_layers: int) -> dict[str, torch.Tensor]:
        """Keep only the blocks / caption projections that the (possibly reduced) model has."""
        if num_layers == self._config.num_layers and num_single_layers == self._config.num_single_layers:
            return self._state_dict

        num_blocks = num_layers + num_single_layers

        def keep(key: str) -> bool:
            for prefix, limit in (
                ("transformer_blocks.", num_layers),
                ("single_transformer_blocks.", num_single_layers),
                ("caption_projection.", num_blocks),
            ):
                if key.startswith(prefix):
                    idx = int(key[len(prefix) :].split(".", 1)[0])
                    return idx < limit
            return True

        return {k: v for k, v in self._state_dict.items() if keep(k)}

    def build(
        self,
        *,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        num_layers: int | None = None,
        num_single_layers: int | None = None,
    ) -> BriaFiboTransformer:
        """Construct a ``BriaFiboTransformer`` for this checkpoint and load its weights.

        ``num_layers`` / ``num_single_layers`` may be reduced below the config values for fast
        iteration; the state dict is filtered to match.
        """
        device = ccl_manager.mesh_device
        c = self._config

        num_layers = c.num_layers if num_layers is None else num_layers
        num_single_layers = c.num_single_layers if num_single_layers is None else num_single_layers

        if c.num_attention_heads % parallel_config.tensor_parallel.factor != 0:
            padding_config = PaddingConfig.from_tensor_parallel_factor(
                c.num_attention_heads,
                c.attention_head_dim,
                parallel_config.tensor_parallel.factor,
            )
        else:
            padding_config = None

        model = BriaFiboTransformer(
            patch_size=c.patch_size,
            in_channels=c.in_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=c.attention_head_dim,
            num_attention_heads=c.num_attention_heads,
            joint_attention_dim=c.joint_attention_dim,
            text_encoder_dim=c.text_encoder_dim,
            out_channels=c.in_channels,  # FIBO out_channels == in_channels == 48.
            axes_dims_rope=c.axes_dims_rope,
            mesh_device=device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            padding_config=padding_config,
        )

        state_dict = self._filtered_state_dict(num_layers, num_single_layers)
        reduced = num_layers != c.num_layers or num_single_layers != c.num_single_layers
        cache.load_model(
            model,
            # Only cache the full-depth model; reduced-depth is dev-only and must not be cached
            # under the same key as the full model.
            get_torch_state_dict=lambda: state_dict,
            model_name=os.path.basename(self._name.rstrip("/")) + ("_reduced" if reduced else ""),
            subfolder="transformer",
            parallel_config=parallel_config,
            mesh_shape=tuple(device.shape),
        )
        return model
