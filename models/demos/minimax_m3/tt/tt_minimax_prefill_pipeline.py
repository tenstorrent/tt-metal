# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2 prefill pipeline — wraps the (validated) model with chunked prefill and
the per-layer KV-migration callback.

SCAFFOLD — see PREFILL_PROPOSAL.md §5/§7. Status of the parts:
  * model forward (RMSNorm/attn/MoE/residual per layer) — VALIDATED (single decoder
    layer PCC 0.99993 @ TP=1; full-model assembly not yet run on this card — needs Galaxy)
  * _build_migration_callback / on_layer_complete seam — wired into model.forward
  * chunked prefill — the per-chunk cache-read attention is wired into attention/prefill.py
    (gated on ``cached_len``); the chunk-loop driver here is the remaining runner work
  * migration transport — NoOp until the migration team's endpoint lands

Reference: models/demos/deepseek_v3_d_p/tt/tt_deepseek_prefill_pipeline.py
"""

from loguru import logger

from .runners.migration_setup import MigrationEndpoint, NoOpMigrationEndpoint


class MiniMaxPrefillPipeline:
    def __init__(self, mesh_device, hf_config, model, chunk_size: int = 5120):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.model = model  # the validated MiniMax-M2 Model (tt/model.py)
        self.chunk_size = chunk_size
        self.endpoint: MigrationEndpoint = NoOpMigrationEndpoint()

    def setup_migration(self, endpoint: MigrationEndpoint) -> None:
        """Bind a real migration endpoint (else stays NoOp)."""
        self.endpoint = endpoint

    def _prepare_input_tensor(self, token_ids):
        """Host token_ids -> sharded ttnn input. TIER 1 / TODO: SP sharding must happen
        AFTER chunking (see PREFILL_PROPOSAL.md §11.3)."""
        raise NotImplementedError("owner: runner work (Tier 1); shard token_ids per SP after chunking")

    def _build_migration_callback(self, slot_id: int, actual_isl: int, dst_slot: int):
        """Return on_layer_complete(layer_idx) that migrates that layer's KV. NoOp endpoint
        makes this a cheap no-op until the migration API lands."""
        block = 128

        def on_layer_complete(layer_idx: int) -> None:
            uuid = self.endpoint.migrate_layer(
                layer_idx, 0, ((actual_isl + block - 1) // block) * block, slot_id, dst_slot
            )
            self.endpoint.wait(uuid)

        return on_layer_complete

    def prefill(self, token_ids, slot_id: int, actual_isl: int, dst_slot: int):
        """Chunked prefill for one request, migrating KV per layer.

        SCAFFOLD: the per-chunk cache-read attention is wired into attention/prefill.py (gated on
        ``cached_len``); this is the remaining chunk-loop driver — split token_ids into chunks and call
        ``ttnn_prefill_forward(..., cached_len=c*chunk)`` per chunk + on_layer_complete callback.
        Left as NotImplementedError so this never silently runs a wrong (non-chunked) path.
        """
        logger.info(f"MiniMaxPrefillPipeline.prefill: isl={actual_isl} chunk={self.chunk_size} slot={slot_id}")
        raise NotImplementedError(
            "wire chunk loop -> ttnn_prefill_forward(cached_len=c*chunk) per chunk + on_layer_complete "
            "callback. See PREFILL_PROPOSAL.md §5, §9."
        )
