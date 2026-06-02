# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""KV migration — one-shot transfer of VLM K/V from prefill stage(s) to expert stage.

Sketch:
- Stage 1 holds VLM layers 0-8; stage 2 holds VLM layers 9-17.
- Each VLM layer maintains a KV cache shaped [B, kv_heads=2, S, head_dim=128] in
  bf16 (current dtype map). Aggregated, ~8.9 MB for 18 layers at S=968 bf16, or
  ~4.5 MB if down-casted to bf8 for the migration.
- The expert stage (stage 3) needs to attend over these K/V at matching layer
  indices during its 10-step denoise loop. So we migrate once at the end of
  prefill, store the resulting KV on stage 3's submesh, and the denoise loop
  reads from local memory.

Implementation choices (in order of preference):
1. Direct D2D copy via parent mesh tensor reshape. Requires that the underlying
   tt-metal exposes a way to view both submeshes as part of the same parent
   distribution; investigation pending.
2. Host bounce: src.to_torch() → dst.from_torch(). Portable, ~2-5 ms wall clock
   for 9 MB at typical PCIe rates. Current default.
3. all_gather over parent mesh: would gather KV onto every chip, but stage 3
   only needs it on stage 3's 8 chips — wasteful.

This module currently implements option 2 as the working fallback. Option 1 is
the target once we confirm the API surface.
"""

from __future__ import annotations

from typing import Dict, List

import ttnn

from .transport import send_activation_via_host


class KVMigration:
    """One-shot VLM-KV → expert-stage KV transfer.

    Usage at end of prefill:
        migrator = KVMigration(expert_submesh=submeshes[3])
        migrator.migrate(vlm_kv_stage_1, vlm_kv_stage_2)

    Call site is `pipeline.py`. Pipeline calls migrate() once between stage 2
    completion and the first denoise step on stage 3.
    """

    def __init__(self, expert_submesh) -> None:
        self.expert_submesh = expert_submesh
        self.migrated_kv: Dict[int, "ttnn.Tensor"] = {}

    def migrate(self, kv_stage_1: List["ttnn.Tensor"], kv_stage_2: List["ttnn.Tensor"]) -> None:
        """Move the KV from prefill stages 1 and 2 to the expert submesh.

        kv_stage_1: list of K∥V tensors for VLM layers 0-8 (or separate K and V).
        kv_stage_2: list of K∥V tensors for VLM layers 9-17.

        Stores them in self.migrated_kv keyed by absolute layer index 0..17.
        """
        for local_idx, src in enumerate(kv_stage_1):
            global_layer = local_idx
            self.migrated_kv[global_layer] = send_activation_via_host(src, self.expert_submesh)
        for local_idx, src in enumerate(kv_stage_2):
            global_layer = 9 + local_idx
            self.migrated_kv[global_layer] = send_activation_via_host(src, self.expert_submesh)

    def get(self, layer_idx: int) -> "ttnn.Tensor":
        return self.migrated_kv[layer_idx]

    def __len__(self) -> int:
        return len(self.migrated_kv)
