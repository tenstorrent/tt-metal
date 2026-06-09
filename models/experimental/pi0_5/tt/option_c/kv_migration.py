# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""KV migration — layer-paired transfer of VLM K/V from prefill to denoise.

Option C uses LAYER-PAIRED routing (deployment plan §3.3.a):

  - Prefill chip i (which owns VLM layer i, i ∈ [0, 18)) pulls its locally-
    cached (K, V) out at the end of prefill.
  - Denoise chip d (which owns expert layers [3d, 3d+3), d ∈ [0, 6)) receives
    K/V for those 3 layers from prefill chips 3d, 3d+1, 3d+2.

Total bytes: 18 × ~500 KB = 9 MB across the mesh, sent in parallel by 18
source chips to 6 destinations. With direct D2D sockets this is ~1-2 ms; via
host bounce (the current default) it's ~120-150 ms at full depth.

Two transport paths:
  - `migrate_layer_paired` (host-bounce) — works for any KV layout. Default.
  - `migrate_layer_paired_d2d` (point_to_point) — requires K/V allocated on
    the galaxy PARENT mesh. Source/dest are on the same row (prefill cols
    0-2, denoise col 3) so 1D linear topology suffices.

The denoise stage's `Pi0_5OptionCExpertSlice.forward` consumes the migrated
KV via the existing `past_key_value` kwarg in AdaRMSGemmaBlockTTNN — no new
op needed on the denoise side.

The denoise stage's `Pi0_5OptionCExpertSlice.forward` consumes the migrated
KV via the existing `past_key_value` kwarg in AdaRMSGemmaBlockTTNN — no new
op needed on the denoise side.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import ttnn

from .stages import (
    DENOISE_SUBMESH_OFFSET,
    EXPERT_LAYERS_PER_DENOISE_CHIP,
    NUM_PREFILL_CHIPS,
    PREFILL_SUBMESH_OFFSET,
    PREFILL_SUBMESH_SHAPE,
)
from .transport import send_per_chip_activation_via_host, send_shard_via_p2p


class KVMigration:
    """One-shot VLM-KV → denoise-stage KV transfer with layer-paired routing.

    Usage at end of prefill:
        migrator = KVMigration(denoise_submesh=submeshes[2])
        migrator.migrate_layer_paired(prefill_kv_per_layer)

    `prefill_kv_per_layer` is a list of length 18 (or whatever vlm_depth is)
    of (K, V) ttnn tensors on the prefill submesh — each layer's KV is held
    on the prefill chip that owns that layer.

    After migrate_layer_paired() returns, the denoise stage can index
    migrated_kv[layer_idx] to get the matching (K, V) on its own submesh.
    """

    def __init__(self, denoise_submesh) -> None:
        self.denoise_submesh = denoise_submesh
        self.migrated_kv: Dict[int, Tuple["ttnn.Tensor", "ttnn.Tensor"]] = {}

    @staticmethod
    def denoise_chip_for_vlm_layer(layer_idx: int) -> int:
        """Map a VLM layer index → the denoise chip that owns the matching
        expert layer.

        With 18 expert layers split 3-per-chip across 6 denoise chips:
            layers 0-2  → chip 0
            layers 3-5  → chip 1
            ...
            layers 15-17→ chip 5
        """
        return layer_idx // EXPERT_LAYERS_PER_DENOISE_CHIP

    def migrate_layer_paired(
        self,
        prefill_kv_per_layer: List[Tuple["ttnn.Tensor", "ttnn.Tensor"]],
        denoise_micro_submeshes: Optional[List] = None,
    ) -> None:
        """Move per-layer KV from prefill chips to denoise chips via layer-paired
        host bounce.

        Two modes, picked automatically by the shard count of each source KV:

        REPLICATED prefill mode (each KV lives on the multi-chip prefill submesh
        with one shard per chip):
            We pluck shard `layer_idx` from each tensor and replicate it onto
            the denoise submesh. This matches the scaffolding pass where the
            VLM slice replicates weights across the whole prefill submesh.

        PAIRED prefill mode (each KV lives on its own 1-chip micro-submesh):
            We pluck shard 0 (the only shard) and replicate to the matching
            denoise micro-submesh from `denoise_micro_submeshes` when given,
            otherwise to the multi-chip denoise submesh.

        Args:
            prefill_kv_per_layer: depth-indexed list of (K, V); None entries
                are skipped.
            denoise_micro_submeshes: optional per-denoise-chip 1-chip submesh
                list (length = NUM_DENOISE_CHIPS). When given, layer i's KV
                lands on submesh `i // EXPERT_LAYERS_PER_DENOISE_CHIP`. Pass
                None to keep the (current) broadcast-to-whole-denoise-submesh
                behaviour.
        """
        if len(prefill_kv_per_layer) > NUM_PREFILL_CHIPS:
            raise ValueError(
                f"Expected ≤{NUM_PREFILL_CHIPS} prefill KV entries (one per chip), " f"got {len(prefill_kv_per_layer)}"
            )
        for layer_idx, kv in enumerate(prefill_kv_per_layer):
            if kv is None:
                continue
            k_src, v_src = kv
            k_shards = ttnn.get_device_tensors(k_src)
            n_shards = len(k_shards)
            # Pick the source chip index:
            # - n_shards == 1   → paired single-chip per layer: shard 0.
            # - n_shards small (<= NUM_DENOISE_CHIPS-ish, typically 2)
            #     → TP sub-mesh per layer: K/V are REPLICATED across the
            #       (tp,1) sub-mesh because num_kv_heads=1 doesn't shard,
            #       so shard 0 carries the full per-layer KV.
            # - n_shards == NUM_PREFILL_CHIPS (replicated mode) → indexed
            #       by global layer_idx.
            if n_shards == 1:
                src_chip_idx = 0
            elif n_shards < NUM_PREFILL_CHIPS:
                src_chip_idx = 0  # TP sub-mesh: replicated KV (num_kv_heads=1)
            else:
                src_chip_idx = layer_idx

            if denoise_micro_submeshes is not None:
                dst = denoise_micro_submeshes[self.denoise_chip_for_vlm_layer(layer_idx)]
            else:
                dst = self.denoise_submesh

            k_on_denoise = send_per_chip_activation_via_host(k_src, src_chip_idx, dst)
            v_on_denoise = send_per_chip_activation_via_host(v_src, src_chip_idx, dst)
            self.migrated_kv[layer_idx] = (k_on_denoise, v_on_denoise)

    # ------------------------------------------------------------------ #
    # D2D path via ttnn.point_to_point                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prefill_parent_coord(layer_idx: int) -> Tuple[int, int]:
        """Galaxy-parent coord of the prefill chip that owns VLM layer `layer_idx`.

        Layer i is placed at prefill submesh's (i // 3, i % 3); the prefill
        submesh sits at parent offset (2, 0). So parent coord =
        (2 + i // 3, i % 3).
        """
        sub_row = layer_idx // PREFILL_SUBMESH_SHAPE[1]
        sub_col = layer_idx % PREFILL_SUBMESH_SHAPE[1]
        return (PREFILL_SUBMESH_OFFSET[0] + sub_row, PREFILL_SUBMESH_OFFSET[1] + sub_col)

    @classmethod
    def _denoise_parent_coord(cls, layer_idx: int) -> Tuple[int, int]:
        """Galaxy-parent coord of the denoise chip that receives layer `layer_idx`'s KV.

        Denoise chip d = layer_idx // EXPERT_LAYERS_PER_DENOISE_CHIP. The denoise
        submesh is (6, 1) at parent offset (2, 3), so chip d sits at parent
        coord (2 + d, 3).
        """
        denoise_chip = cls.denoise_chip_for_vlm_layer(layer_idx)
        return (DENOISE_SUBMESH_OFFSET[0] + denoise_chip, DENOISE_SUBMESH_OFFSET[1])

    def migrate_layer_paired_d2d(
        self,
        prefill_kv_per_layer_on_parent: List[Tuple["ttnn.Tensor", "ttnn.Tensor"]],
    ) -> None:
        """D2D variant of `migrate_layer_paired` using `ttnn.point_to_point`.

        PRECONDITION: each (K, V) tensor in `prefill_kv_per_layer_on_parent`
        MUST be allocated on the galaxy PARENT mesh (full (8,4) shape), with
        the per-layer data residing at the sender's parent coord. The current
        prefill slice (`Pi0_5OptionCVLMSlicePaired`) allocates K/V on 1-chip
        micro-submeshes carved from the prefill submesh — those tensors need
        to be migrated up to the parent before this method can be called.
        See `transport.py` docstring for the integration plan.

        Routing: for VLM layer i, source = prefill chip at parent coord
        (2 + i//3, i%3), destination = denoise chip at parent coord
        (2 + i//3, 3). Same row → 1D linear topology routes via fabric.

        Requires `set_fabric_config(FABRIC_1D)` before parent mesh open.

        Estimated runtime vs the host-bounce path: ~1-2 ms vs ~120-150 ms.
        """
        for layer_idx, kv in enumerate(prefill_kv_per_layer_on_parent):
            if kv is None:
                continue
            k_parent, v_parent = kv
            src_coord = self._prefill_parent_coord(layer_idx)
            dst_coord = self._denoise_parent_coord(layer_idx)
            k_migrated = send_shard_via_p2p(k_parent, src_coord, dst_coord)
            v_migrated = send_shard_via_p2p(v_parent, src_coord, dst_coord)
            self.migrated_kv[layer_idx] = (k_migrated, v_migrated)

    def get(self, layer_idx: int) -> Tuple["ttnn.Tensor", "ttnn.Tensor"]:
        return self.migrated_kv[layer_idx]

    def as_list(self, depth: int) -> List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]]:
        """Return a depth-indexed list of (K, V) suitable for AdaRMSGemmaBlockTTNN's
        past_key_value parameter. Layers not migrated → None.
        """
        return [self.migrated_kv.get(i) for i in range(depth)]

    def __len__(self) -> int:
        return len(self.migrated_kv)
