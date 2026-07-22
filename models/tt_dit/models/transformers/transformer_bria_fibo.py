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
from ...utils.matmul import register_matmul_configs
from ...utils.padding import PaddingConfig
from ...utils.substate import rename_substate
from .transformer_flux1 import Flux1SingleTransformerBlock

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ...parallel.config import DiTParallelConfig
    from ...parallel.manager import CCLManager


# FIBO denoise matmul blockings tuned via models/tt_dit/utils/sweep_mm_block_sizes.py on the
# 2x2 Blackhole mesh (sp=2/tp=2), 12x10 compute grid. Every FIBO block matmul takes the
# non-AGMM minimal_matmul path, so a per-shape MinimalMatmulConfig replaces the generic (8,8,8)
# fallback. Keyed by (M, K, N) under the runtime grid; additive (register_matmul_configs merges by
# shape) so it cannot affect other models, whose (M, K, N) differ. Two (M, K, N) collide across use
# cases (proj_mlp "plain" vs ff1 "plain_gelu"); the block is chosen to minimize the op-count-weighted
# total (proj_mlp runs ~38x/forward vs ff1 ~8x). Regenerate with the bh_2x2 sweep if shapes change.
_FIBO_MM_CONFIGS_REGISTERED = False


def _register_fibo_matmul_configs() -> None:
    global _FIBO_MM_CONFIGS_REGISTERED
    if _FIBO_MM_CONFIGS_REGISTERED:
        return
    register_matmul_configs(
        {
            "12x10": {
                # (M, K, N): (M_block, K_block, N_block, (subblock_h, subblock_w))
                (2048, 7680, 3072): (4, 5, 14, (2, 2)),  # single proj_out spatial
                (2048, 3072, 6144): (4, 4, 16, (2, 2)),  # ff1 (gelu) / proj_mlp spatial (weighted pick)
                (2048, 6144, 3072): (4, 4, 10, (2, 2)),  # ff2 spatial
                (2048, 3072, 4608): (4, 4, 15, (4, 1)),  # to_qkv spatial
                (2048, 3072, 1536): (6, 3, 5, (3, 1)),  # attn to_out spatial
                (2048, 3072, 64): (3, 8, 2, (3, 1)),  # final proj_out
                (2048, 64, 1536): (16, 2, 5, (4, 1)),  # x_embedder
                (128, 7680, 3072): (2, 8, 14, (2, 2)),  # single proj_out prompt
                (128, 3072, 6144): (4, 3, 16, (2, 2)),  # proj_mlp / ff1 prompt (weighted pick)
                (128, 6144, 3072): (2, 4, 8, (2, 2)),  # ff2 prompt
                (128, 3072, 4608): (2, 3, 15, (1, 3)),  # to_qkv prompt
                (128, 3072, 1536): (2, 8, 4, (2, 2)),  # attn to_add_out prompt
                (128, 4096, 1536): (2, 8, 4, (2, 2)),  # context_embedder
                (128, 2048, 1536): (2, 8, 4, (2, 2)),  # caption_projection
                (32, 3072, 9216): (4, 4, 8, (2, 2)),  # norm1 modulation
                (32, 3072, 6144): (2, 2, 16, (2, 2)),  # time_embed_out
                (32, 3072, 4608): (2, 3, 14, (2, 2)),  # single time_embed
                (32, 3072, 3072): (2, 4, 14, (2, 2)),  # timestep_embedder linear_2
                (32, 256, 3072): (2, 2, 8, (2, 2)),  # timestep_embedder linear_1
                # --- FIBO denoise on the 4x8 Galaxy (sp=4/tp=8): M=1024 spatial (4096/sp), M=128 prompt,
                # M=32 timestep; N/K are tp=8-sharded so they differ from the 2x2/tp=2 shapes above. Swept
                # 2026-07-15 via sweep_mm_block_sizes.py (bh_4x8_fibo, 12x10). The 4 tp-independent shapes
                # (32,256,3072 / 32,3072,3072 / 32,3072,6144 / 128,2048,1536) already hit the 2x2 12x10
                # entries above, so only the 14 new (tp-dependent) shapes are added here. ns = HiFi2 per-op.
                (1024, 1536, 3072): (8, 3, 10, (2, 2)),  # dual ff.ff2 spatial — 70330 ns
                (1024, 1920, 3072): (10, 3, 12, (2, 2)),  # single proj_out spatial — 79605 ns
                (1024, 3072, 1152): (8, 6, 4, (2, 2)),  # to_qkv spatial — 65895 ns
                (1024, 3072, 1536): (4, 6, 4, (2, 2)),  # dual ff.ff1 / proj_mlp spatial — 101395 ns
                (1024, 3072, 384): (3, 6, 2, (3, 1)),  # attn to_out spatial — 40553 ns
                (1024, 3072, 64): (3, 8, 2, (3, 1)),  # final proj_out — 34773 ns
                (1024, 64, 384): (3, 2, 2, (3, 1)),  # x_embedder — 5864 ns
                (128, 1536, 3072): (2, 4, 14, (2, 2)),  # dual ff_context.ff2 prompt — 47116 ns
                (128, 1920, 3072): (2, 4, 10, (2, 2)),  # single proj_out prompt twin — 56373 ns
                (128, 3072, 1152): (2, 8, 4, (2, 2)),  # to_qkv prompt — 38160 ns
                (128, 3072, 384): (2, 8, 2, (2, 2)),  # attn to_add_out prompt — 21230 ns
                (128, 4096, 384): (2, 8, 2, (2, 2)),  # context_embedder — 25822 ns
                (32, 3072, 1152): (2, 8, 6, (2, 2)),  # single-block time_embed — 37902 ns
                (32, 3072, 2304): (2, 6, 6, (2, 2)),  # norm1 modulation — 64667 ns
                # DiT prompt-branch matmuls at M=864 (long structured-JSON caption -> 833 tokens tile-padded
                # to 864; the committed fibo_vlm_prompt.json). The prompt branch runs UNPADDED at the true
                # token length (encoder unpads at text_encoder.py:160), so a long prompt lands at M=864 --
                # distinct from the M=128 (~128-token) / M=32 (short) twins above. Swept 2026-07-16 via
                # sweep_mm_block_sizes.py (bh_4x8_fibo, 12x10, all 7 measured 0-OOM). ns = HiFi2 per-op.
                (864, 4096, 384): (3, 8, 2, (3, 1)),  # context_embedder — 49220 ns
                (864, 2048, 1536): (3, 4, 4, (1, 4)),  # caption_projection — 46424 ns
                (864, 3072, 1152): (3, 4, 4, (1, 4)),  # to_qkv prompt (chunks=3, approx) — 52626 ns
                (864, 3072, 384): (3, 8, 2, (3, 1)),  # attn to_add_out prompt (addcmul, approx) — 39992 ns
                (864, 3072, 1536): (3, 6, 4, (1, 4)),  # ff_context.ff1 prompt (fused GELU) — 81504 ns
                (864, 1536, 3072): (3, 4, 8, (1, 4)),  # ff_context.ff2 prompt — 61876 ns
                (864, 1920, 3072): (3, 4, 16, (1, 4)),  # single proj_out prompt — 71226 ns
                # SmolLM3 text encoder (tensor-parallel, tp=8) matmuls on the 4x8 Galaxy. M=32 (short
                # prompt, one tile), K=2048=hidden. Swept 2026-07-15 (bh_4x8_fibo). SmolLM3 has no matmul
                # registration of its own and FIBO is its only user, so its configs live here (additive,
                # K=2048 keys distinct from the DiT's). Longer prompts give larger M -> a follow-up.
                (32, 2048, 256): (2, 8, 4, (2, 2)),  # kv proj (grouped-query) — 16107 ns
                (32, 2048, 512): (2, 8, 2, (2, 2)),  # q proj / attn out — 21747 ns
                (32, 2048, 1376): (2, 8, 4, (2, 2)),  # MLP gate/up proj — 33838 ns
                (32, 1376, 2048): (2, 43, 2, (2, 2)),  # MLP down proj (RowParallel) — 49293 ns
                # DiT prompt-branch matmuls at M=32 (short / empty-CFG-uncond prompt; the M=128 twins are
                # for a ~128-token prompt). The 2 small-N shapes swept cleanly; the 4 large-N shapes hit a
                # profiler-buffer failure at M=32, so they reuse their M=128 prompt winners (M=32 is 1 tile
                # -> only M_block differs and it clamps; far better than the generic fallback). 2026-07-15.
                (32, 3072, 384): (2, 8, 2, (2, 2)),  # to_add_out prompt (M=32) — 21061 ns
                (32, 4096, 384): (2, 8, 4, (2, 2)),  # context_embedder (M=32) — 25547 ns
                (32, 1536, 3072): (2, 4, 14, (2, 2)),  # ff_context.ff2 prompt (M=32, reuse M=128)
                (32, 1920, 3072): (2, 4, 10, (2, 2)),  # single proj_out prompt (M=32, reuse M=128)
                (32, 2048, 1536): (2, 8, 4, (2, 2)),  # caption_projection (M=32, reuse M=128)
                (32, 3072, 1536): (2, 8, 4, (2, 2)),  # ff_context.ff1 prompt (M=32, reuse M=128)
            },
            # FIBO denoise on the 4x8 Galaxy at 11x10 (the historical Galaxy grid clamp). FIBO registered
            # nothing at 11x10 before, so all 19 4x8 shapes are added. Kept as a fallback for when the
            # matmul core grid is clamped back to 11x10 (see get_matmul_core_grid in utils/matmul.py).
            # Swept 2026-07-15, same run as the 12x10 block above.
            "11x10": {
                (1024, 1536, 3072): (8, 4, 10, (2, 2)),  # dual ff.ff2 spatial — 78747 ns
                (1024, 1920, 3072): (12, 4, 10, (2, 2)),  # single proj_out spatial — 90564 ns
                (1024, 3072, 1152): (8, 6, 4, (2, 2)),  # to_qkv spatial — 65969 ns
                (1024, 3072, 1536): (4, 3, 5, (4, 1)),  # dual ff.ff1 / proj_mlp spatial — 114275 ns
                (1024, 3072, 384): (3, 6, 2, (3, 1)),  # attn to_out spatial — 40002 ns
                (1024, 3072, 64): (3, 8, 2, (3, 1)),  # final proj_out — 34826 ns
                (1024, 64, 384): (3, 2, 2, (3, 1)),  # x_embedder — 5742 ns
                (128, 1536, 3072): (2, 3, 10, (2, 2)),  # dual ff_context.ff2 prompt — 51235 ns
                (128, 1920, 3072): (2, 4, 10, (2, 2)),  # single proj_out prompt twin — 61632 ns
                (128, 3072, 1152): (2, 8, 4, (2, 2)),  # to_qkv prompt — 44027 ns
                (128, 3072, 384): (2, 8, 2, (2, 2)),  # attn to_add_out prompt — 28460 ns
                (128, 4096, 384): (2, 8, 2, (2, 2)),  # context_embedder — 35620 ns
                (32, 3072, 1152): (2, 8, 6, (2, 2)),  # single-block time_embed — 43811 ns
                (32, 3072, 2304): (2, 6, 8, (2, 2)),  # norm1 modulation — 71665 ns
                (128, 2048, 1536): (2, 4, 6, (2, 2)),  # caption_projection — 39750 ns
                (128, 3072, 1536): (2, 6, 5, (2, 1)),  # dual ff_context.ff1 prompt — 76822 ns
                (32, 256, 3072): (2, 2, 10, (2, 2)),  # timestep_embedder linear_1 — 15618 ns
                (32, 3072, 3072): (2, 4, 16, (2, 2)),  # timestep_embedder linear_2 — 89023 ns
                (32, 3072, 6144): (2, 8, 12, (2, 2)),  # time_embed_out — 164954 ns
                # SmolLM3 encoder at 11x10 (fallback grid). (32,2048,256) produced no OK sweep rows at
                # 11x10, so it reuses its 12x10 winner (same matmul, adjacent grid; beats the generic default).
                (32, 2048, 256): (2, 8, 4, (2, 2)),  # kv proj (reused 12x10 winner)
                (32, 2048, 512): (2, 8, 2, (2, 2)),  # q proj / attn out — 21460 ns
                (32, 2048, 1376): (2, 8, 4, (2, 2)),  # MLP gate/up proj — 33524 ns
                (32, 1376, 2048): (2, 43, 2, (2, 2)),  # MLP down proj — 47616 ns
                # DiT prompt-branch matmuls at M=32 (11x10 fallback grid), same rationale as the 12x10 block.
                (32, 3072, 384): (2, 8, 2, (2, 2)),  # to_add_out prompt (M=32) — 28426 ns
                (32, 4096, 384): (4, 16, 2, (2, 2)),  # context_embedder (M=32) — 35392 ns
                (32, 3072, 1536): (2, 6, 5, (2, 1)),  # ff_context.ff1 prompt (M=32) — 76036 ns
                (32, 1536, 3072): (2, 3, 10, (2, 2)),  # ff_context.ff2 prompt (M=32, reuse M=128)
                (32, 1920, 3072): (2, 4, 10, (2, 2)),  # single proj_out prompt (M=32, reuse M=128)
                (32, 2048, 1536): (2, 4, 6, (2, 2)),  # caption_projection (M=32, reuse M=128)
            },
        }
    )
    _FIBO_MM_CONFIGS_REGISTERED = True


_register_fibo_matmul_configs()


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


def inject_text(
    encoder_hidden_states: ttnn.Tensor,
    projected: ttnn.Tensor,
    *,
    ccl_manager: CCLManager | None = None,
    tp_axis: int | None = None,
) -> ttnn.Tensor:
    """Concat-halves text injection for FIBO blocks, general over the TP factor.

    Replaces the upper half of the context (prompt) features with the per-block projected text
    embedding::

        out[..., :half] = encoder_hidden_states[..., :half]
        out[..., half:] = projected

    where ``half = inner_dim // 2``. ``projected`` comes from a *replicated*
    ``BriaFiboTextProjection`` linear, so it is the full ``half`` on every device.

    Two paths:

    * **tp=1** (``ccl_manager`` / ``tp_axis`` is ``None``): ``encoder_hidden_states`` is the full
      ``[batch, P, inner_dim]`` context; plain feature-dim concat of its first half with ``projected``.
    * **tp>1**: ``encoder_hidden_states`` is the *local* feature shard ``[batch, P, inner_dim / tp]``
      on ``tp_axis``. All-gather it to the full ``inner_dim`` (replicated across ``tp_axis``), do the
      concat-halves in full space against the replicated ``projected``, then ``mesh_partition`` back
      to the local shard on ``tp_axis``. The gather makes this correct for any TP factor (there is no
      longer a tp==2 shard-aligned special case), at the cost of one all-gather + one local reshard
      per injection.

    Args:
        encoder_hidden_states: Context tensor. tp=1: ``[batch, P, inner_dim]``. tp>1: the local
            feature shard ``[batch, P, inner_dim / tp]``.
        projected: Per-block text projection ``[batch, P, inner_dim // 2]`` (replicated on tp).
        ccl_manager: CCL manager for the tp>1 all-gather, or ``None`` for the tp=1 path.
        tp_axis: tensor-parallel mesh axis for the tp>1 all-gather/reshard, or ``None`` for tp=1.

    Returns:
        Injected context, same feature layout as ``encoder_hidden_states``.
    """
    half = projected.shape[-1]

    if ccl_manager is None or tp_axis is None:
        assert (
            encoder_hidden_states.shape[-1] == 2 * half
        ), f"inject_text: expected encoder_hidden_states last dim {2 * half}, got {encoder_hidden_states.shape[-1]}"
        return ttnn.concat([encoder_hidden_states[:, :, :half], projected], dim=-1)

    # tp>1: gather the local feature shard to the full inner_dim (replicated across tp_axis), do the
    # concat-halves in full space against the replicated projection, then reshard on tp_axis.
    # mesh_partition of a replicated tensor is local (no CCL).
    enc_full = ccl_manager.all_gather(encoder_hidden_states, dim=-1, mesh_axis=tp_axis, use_hyperparams=True)
    injected_full = ttnn.concat([enc_full[:, :, :half], projected], dim=-1)
    return ttnn.mesh_partition(injected_full, dim=-1, cluster_axis=tp_axis)


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

        # tp>1: inject_text all-gathers the feature-sharded prompt on tp_axis, does the concat-halves,
        # and reshards. tp=1: plain concat (ccl_manager/tp_axis unused, so pass None).
        tp_factor = self.parallel_config.tensor_parallel.factor
        inject_ccl = self.ccl_manager if tp_factor > 1 else None
        inject_tp_axis = tp_axis if tp_factor > 1 else None

        time_embed = self.time_embed(timestep)
        ttnn.silu(time_embed, output_tensor=time_embed)
        time_embed = time_embed.reshape([time_embed.shape[-2], 1, time_embed.shape[-1]])

        spatial = self.x_embedder(spatial)
        prompt = self.context_embedder(prompt)

        block_id = 0

        for block in self.transformer_blocks:
            projected = self.caption_projection[block_id](text_encoder_layers[block_id])
            prompt = inject_text(prompt, projected, ccl_manager=inject_ccl, tp_axis=inject_tp_axis)
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
            prompt = inject_text(prompt, projected, ccl_manager=inject_ccl, tp_axis=inject_tp_axis)
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
