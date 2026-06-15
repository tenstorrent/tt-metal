# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS prefill pipeline — wraps the (validated) model with the per-layer
KV-migration callback for disaggregated prefill/decode serving.

SCAFFOLD — see PREFILL_PROPOSAL.md §5/§7. Status of the parts:
  * model forward (all 36 layers, prefill + decode paths) — VALIDATED (existing tests pass)
  * on_layer_complete seam — wired into model._forward_layers_and_head / ttnn_prefill_forward
  * _prepare_input_tensor — TIER 1, NOT YET IMPLEMENTED
  * prefill() chunk loop — TIER 1, NOT YET IMPLEMENTED
  * migration transport — NoOp until the migration team's endpoint lands

Key differences from MiniMax-M2 pipeline:
  * 36 layers (not 62); migration fires 36 times per request
  * Separate k_cache + v_cache per layer → endpoint must address both tensors
  * No chunking by default (see PREFILL_PROPOSAL.md §8.4); chunk_size param
    reserved for future tuning

Reference: models/demos/minimax_m2/tt/tt_minimax_prefill_pipeline.py
"""

import math

from loguru import logger

from .runners.migration_setup import MigrationEndpoint, NoOpMigrationEndpoint

# Migration chunk granularity: 32 tokens (DRAM-bank aligned, matches migration team spec).
_MIGRATION_BLOCK = 32


class GptOssPrefillPipeline:
    def __init__(self, mesh_device, hf_config, model):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.model = model  # the validated GptOss Model (tt/model.py)
        self.endpoint: MigrationEndpoint = NoOpMigrationEndpoint()

    def setup_migration(self, endpoint: MigrationEndpoint) -> None:
        """Bind a real migration endpoint (else stays NoOp)."""
        self.endpoint = endpoint

    def _prepare_input_tensor(self, token_ids):
        """Host token_ids -> sharded ttnn input.

        TIER 1 / TODO: shard token_ids across SP=4 rows. SP sharding must happen
        AFTER any chunking (see PREFILL_PROPOSAL.md §5).
        """
        raise NotImplementedError("owner: runner work (Tier 1); shard token_ids per SP after chunking")

    def _build_migration_callback(self, slot_id: int, actual_isl: int, dst_slot: int):
        """Return on_layer_complete(layer_idx) that migrates that layer's KV.

        pos_end is rounded up to the nearest _MIGRATION_BLOCK boundary so every
        migration covers a whole number of 32-token chunks (DRAM-bank aligned).
        NoOp endpoint makes this a cheap no-op until the migration API lands.

        GPT-OSS note: the real endpoint must migrate both k_cache and v_cache for
        each layer (two separate DRAM addresses). See PREFILL_PROPOSAL.md §5.
        """
        pos_end = math.ceil(actual_isl / _MIGRATION_BLOCK) * _MIGRATION_BLOCK

        def on_layer_complete(layer_idx: int) -> None:
            uuid = self.endpoint.migrate_layer(layer_idx, 0, pos_end, slot_id, dst_slot)
            self.endpoint.wait(uuid)

        return on_layer_complete

    def prefill(self, token_ids, slot_id: int, actual_isl: int, dst_slot: int):
        """Full-sequence prefill for one request, migrating KV per layer.

        SCAFFOLD: _prepare_input_tensor and the model call are Tier-1 work.
        Left as NotImplementedError so this never silently runs without the
        SP-sharded input path in place.

        When chunking is needed (see PREFILL_PROPOSAL.md §8.4), add a chunk
        loop here and pass chunk_start_idx to ttnn_prefill_forward.
        """
        logger.info(f"GptOssPrefillPipeline.prefill: isl={actual_isl} slot={slot_id} dst={dst_slot}")
        raise NotImplementedError(
            "Tier 1: _prepare_input_tensor (SP shard) + model.ttnn_prefill_forward(on_layer_complete=...). "
            "See PREFILL_PROPOSAL.md §5."
        )
