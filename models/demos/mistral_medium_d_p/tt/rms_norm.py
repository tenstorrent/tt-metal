# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 RMSNorm.

``MistralRMSNorm`` is the plain form — ``out = weight * (x * rsqrt(mean(x^2) + eps))`` — with **no**
Gemma ``(1 + w)`` fold and no per-head variant, and ``rms_norm_eps = 1e-5``. Verified in
``transformers/models/mistral/modeling_mistral.py``.

The activation stays REPLICATED across the TP axis in this design (each chip holds the full hidden
dim of its own sequence shard), so a plain single-device ``ttnn.rms_norm`` is exact — no distributed
two-pass norm and no stats all-gather is needed. That differs from ``llama3_70b_galaxy``, which
shards hidden across 4 chips and therefore must run a distributed RMSNorm.
"""

from torch import nn

import ttnn
from models.demos.mistral_medium_d_p.config import MeshConfig
from models.demos.mistral_medium_d_p.utils.general_utils import get_cache_file_name


class RMSNorm(nn.Module):
    def __init__(self, mesh_device, hf_config, state_dict, tensor_cache_path=None, mesh_config=None):
        super().__init__()
        if getattr(hf_config, "use_gemma_norm", False):
            raise NotImplementedError(
                "use_gemma_norm is set, but Mistral uses a plain RMSNorm (no (1+w) fold). "
                "Folding it silently would shift every norm by one."
            )
        if state_dict:
            # ttnn.rms_norm wants the gain as [1, 1, hidden/TILE, TILE].
            torch_weight = state_dict["weight"].reshape((1, 1, -1, ttnn.TILE_SIZE))
        else:
            torch_weight = None

        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.tt_weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            cache_file_name=get_cache_file_name(tensor_cache_path, "weight"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.eps = hf_config.rms_norm_eps
        self.mesh_device = mesh_device

    def forward(self, x):
        return ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.eps)
