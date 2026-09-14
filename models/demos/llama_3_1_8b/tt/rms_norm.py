# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm on the replicated residual stream.

Llama-3.1 uses the plain form ``x * rsqrt(mean(x^2) + eps) * w`` — there is **no** Gemma ``(1 + w)``
fold, so nothing is folded into the gain at load. That is the one thing in the borrowed M3 norm that
does not transfer (M3 sets ``use_gemma_norm``); the sharding and the weight layout do.

The residual is replicated across the TP columns (``tt/mesh.py``), so every column normalizes the
same full ``hidden_size`` vector and a single-pass ``ttnn.rms_norm`` with a REPLICATED gain is both
correct and collective-free. The three-op distributed form
(``rms_norm_pre_all_gather`` -> gather stats -> ``rms_norm_post_all_gather``) is what an ``emb/tp``
sharded residual needs; it is not implemented here because that residual layout is not.
"""

from __future__ import annotations

from typing import Optional

import ttnn
from models.common.lightweightmodule import LightweightModule


class RMSNorm(LightweightModule):
    def __init__(
        self,
        mesh_device,
        hidden_size: int,
        eps: float,
        state_dict: Optional[dict] = None,
        cache_file_name: Optional[str] = None,
    ):
        """``state_dict`` is the norm's own sub-dict (``{"weight": ...}``); empty/None means
        cache-only, where ``ttnn.as_tensor`` loads the tilized gain straight off disk."""
        super().__init__()
        self.eps = eps
        self.hidden_size = hidden_size

        torch_weight = None
        if state_dict:
            w = state_dict["weight"]
            assert w.shape[-1] == hidden_size, f"norm gain is {tuple(w.shape)}, expected [{hidden_size}]"
            # ROW_MAJOR [1, 1, hidden/32, 32] is the weight layout ttnn.rms_norm expects.
            torch_weight = w.reshape(1, 1, -1, ttnn.TILE_SIZE)

        self.tt_weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache_file_name,
        )

    def forward(self, x):
        return ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.eps)
