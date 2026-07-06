# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os  # noqa: F401
from typing import TYPE_CHECKING

import torch  # noqa: F401
from diffusers import BriaFiboTransformer2DModel  # noqa: F401

import ttnn  # noqa: F401
from models.common.utility_functions import is_blackhole  # noqa: F401

from ...blocks.attention import Attention  # noqa: F401
from ...blocks.transformer_block import TransformerBlock, _chunk_time3d  # noqa: F401
from ...layers.embeddings import CombinedTimestepGuidanceTextProjEmbeddings  # noqa: F401
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
