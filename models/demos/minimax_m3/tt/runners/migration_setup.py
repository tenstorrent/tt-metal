# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
KV-migration seam for MiniMax-M2 prefill/decode disaggregation.

SCAFFOLD — see PREFILL_PROPOSAL.md §7. The fabric transport (sub-context setup,
KvChunkAddressTable, send/recv) is owned by the MIGRATION TEAM. This module only
defines the interface the prefill pipeline binds to, plus a no-op implementation
so the runner can execute end-to-end on one mesh without migration.

Reference (do NOT copy blindly — adapt for head_dim=128, 32-token chunks):
  models/demos/deepseek_v3_d_p/tt/runners/migration_setup.py
  models/demos/deepseek_v3_d_p/tt/tt_deepseek_prefill_pipeline.py  (BoundMigrationEndpoint)
"""

from typing import Protocol


class MigrationEndpoint(Protocol):
    """Interface the prefill pipeline calls per layer. Implemented by the migration team."""

    def migrate_layer(self, layer_idx: int, pos_start: int, pos_end: int, src_slot: int, dst_slot: int) -> int:
        """Send KV for one layer's [pos_start, pos_end) to the decode side. Returns a uuid/handle."""
        ...

    def wait(self, uuid: int) -> None:
        """Block until the migration identified by uuid has been sent + acked."""
        ...


class NoOpMigrationEndpoint:
    """Runs the pipeline with migration disabled (single-mesh / standalone). Real and safe."""

    def migrate_layer(self, layer_idx, pos_start, pos_end, src_slot, dst_slot) -> int:
        return 0

    def wait(self, uuid) -> None:
        return None


def setup_prefill_migration(*args, **kwargs) -> MigrationEndpoint:
    """Connect a real prefill->decode migration endpoint over fabric.

    OWNER: migration team. BLOCKED ON: ttnn disaggregation API + sub-context setup.
    Until that lands, pipelines use NoOpMigrationEndpoint (PREFILL_ENABLE_MIGRATION=0).
    """
    raise NotImplementedError(
        "setup_prefill_migration: owner=migration team; blocked on ttnn disaggregation API. "
        "Use NoOpMigrationEndpoint for standalone bring-up (PREFILL_ENABLE_MIGRATION=0)."
    )
