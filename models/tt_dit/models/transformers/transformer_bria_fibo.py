# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os  # noqa: F401
from typing import TYPE_CHECKING

import torch  # noqa: F401
from diffusers import BriaFiboTransformer2DModel  # noqa: F401

import ttnn
from models.common.utility_functions import is_blackhole  # noqa: F401

from ...blocks.attention import Attention  # noqa: F401
from ...blocks.transformer_block import TransformerBlock, _chunk_time3d  # noqa: F401
from ...layers.embeddings import TimestepEmbedding, Timesteps
from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear, prepare_chunked_linear_output  # noqa: F401
from ...layers.module import Module, ModuleList  # noqa: F401
from ...layers.normalization import DistributedLayerNorm  # noqa: F401
from ...utils import cache  # noqa: F401
from ...utils.padding import PaddingConfig  # noqa: F401
from ...utils.substate import rename_substate  # noqa: F401
from .transformer_flux1 import Flux1SingleTransformerBlock, _re_fuse_proj_out_weight  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Sequence  # noqa: F401

    from ...parallel.config import DiTParallelConfig  # noqa: F401
    from ...parallel.manager import CCLManager  # noqa: F401


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
