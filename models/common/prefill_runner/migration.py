# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""KV chunk address table send — model-agnostic half of the disaggregation integration.

Building the table from the device KV layout is model-specific (the cache layout differs per model),
so it lives behind the adapter (`build_and_serialize_kv_chunk_table`). Forwarding the serialized .pb
to the migration_worker is generic and lives here.

NOTE: only the table-publish half of the disaggregation integration lives here. The per-layer
LayerAck channel + scheduler-driven migration are intentionally left out (owned by the
scheduler/worker side).
"""

import os
import sys

from loguru import logger

# Sentinel dst_slot the inference server sends when a request must NOT trigger
# migration (e.g. warmup probes, scheduler-skipped requests).
INVALID_SLOT_ID = 0xFFFFFFFF


def send_kv_chunk_table(table_path: str) -> bool:
    """Forward the serialized table to the migration_worker via SET_TABLE, using
    the tt-llm-engine MigrationLayerClient.

    The client extension (``_migration_client``) lives in tt-llm-engine (the
    superproject), so it is imported LAZILY here — only when migration is actually
    being driven — to avoid a hard tt-metal -> tt-llm-engine dependency. The call
    no-ops gracefully (returns False, logs a warning) if the extension or the
    orchestrator-owned shmem queues are not present, so a standalone runner still
    writes the .pb without requiring the full migration stack.

    Env:
      PREFILL_MIGRATION_CLIENT_DIR  dir holding _migration_client*.so (optional if
                                    already on PYTHONPATH)
      PREFILL_MIGRATION_CMD_QUEUE / _TABLE_QUEUE / _RESP_QUEUE  shmem queue names
                                    (must match the orchestrator/IS that owns them)
    """
    client_dir = os.environ.get("PREFILL_MIGRATION_CLIENT_DIR")
    if client_dir and client_dir not in sys.path:
        sys.path.insert(0, client_dir)
    try:
        import _migration_client
    except ImportError as e:
        logger.warning(
            f"[migration] _migration_client not importable ({e}); table written to {table_path} "
            f"but NOT sent. Set PREFILL_MIGRATION_CLIENT_DIR or add the extension to PYTHONPATH."
        )
        return False

    cmd_q = os.environ.get("PREFILL_MIGRATION_CMD_QUEUE", "/prefill_mig_cmd_1")
    table_q = os.environ.get("PREFILL_MIGRATION_TABLE_QUEUE", "/prefill_mig_tbl_1")
    resp_q = os.environ.get("PREFILL_MIGRATION_RESP_QUEUE", "/prefill_mig_rsp_1")
    try:
        client = _migration_client.MigrationLayerClient(cmd_q, table_q, resp_q)
    except RuntimeError as e:
        logger.warning(
            f"[migration] could not attach MigrationLayerClient to queues "
            f"({cmd_q}, {table_q}, {resp_q}): {e}; table NOT sent. The orchestrator/IS "
            f"must create the shmem queues (and launch the worker) before the runner sends."
        )
        return False

    client.send_kv_chunk_table(table_path)
    logger.info(f"[migration] sent SET_TABLE({table_path}) via MigrationLayerClient on {table_q}")
    return True
