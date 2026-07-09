# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Model-agnostic KV-migration publish for the prefill runner.

The runner owns the migration comms. The model-specific part — building the
KV-chunk address table from the device cache layout — lives on the runtime
(``runtime.build_kv_chunk_table(path)``); this module takes the already-serialized
table and performs the generic handshake with a running migration_endpoint:

    1. client.send_kv_chunk_table(table_path)   # SetTable on table_q
    2. client.send_device_map(entries)          # AssignDevMap + N x DevMapEntry on table_q
    3. client.wait_ready(timeout_ms)            # blocks on RespOpcode::WorkerReady

Without step 2 the worker never opens UMD chips and never asserts WORKER_READY;
without step 3 the runner enters its request loop before the scheduler can
actually issue migrations. The device map is derived from the local mesh topology
(generic). The ``_migration_client`` extension lives in tt-llm-engine and is
imported LAZILY so the runner still works standalone when migration is opted out.
"""

import os
import sys

from loguru import logger

import ttnn

# Default shmem queue names. Overridable via PREFILL_MIGRATION_{CMD,TABLE,RESP}_QUEUE.
_DEFAULT_CMD_QUEUE = "/prefill_mig_cmd_1"
_DEFAULT_TABLE_QUEUE = "/prefill_mig_tbl_1"
_DEFAULT_RESP_QUEUE = "/prefill_mig_rsp_1"


def serialize_kv_chunk_table(
    *,
    table_builder,
    num_layers: int,
    max_seq_len: int,
    num_users: int,
    chunk_n_tokens: int,
    chunk_size_bytes: int,
    path: str,
) -> str:
    """Populate the generic KvChunkAddressTableConfig, invoke a model's `table_builder`, and serialize
    the result to a protobuf file for the worker's SET_TABLE. Returns `path`.

    The KV layout (how natural positions map to storage chips/offsets) is the one model-specific part:
    the caller supplies `table_builder(config, chunk_size_bytes, num_users) -> KvChunkAddressTable`.
    Everything here — config population, serialization, logging — is model-agnostic.
    """
    disagg = ttnn.experimental.disaggregation
    cfg = disagg.KvChunkAddressTableConfig()
    cfg.num_layers = num_layers
    cfg.max_sequence_length = max_seq_len
    cfg.num_slots = num_users
    cfg.chunk_n_tokens = chunk_n_tokens
    cfg.chunk_size_bytes = chunk_size_bytes
    table = table_builder(config=cfg, chunk_size_bytes=chunk_size_bytes, num_users=num_users)
    disagg.export_to_protobuf_file(table, path)
    logger.info(f"[migration] KV chunk address table serialized to {path} (entries={table.total_entries()})")
    return path


def _resolve_queue_names() -> tuple[str, str, str]:
    return (
        os.environ.get("PREFILL_MIGRATION_CMD_QUEUE", _DEFAULT_CMD_QUEUE),
        os.environ.get("PREFILL_MIGRATION_TABLE_QUEUE", _DEFAULT_TABLE_QUEUE),
        os.environ.get("PREFILL_MIGRATION_RESP_QUEUE", _DEFAULT_RESP_QUEUE),
    )


def _import_migration_client():
    """Lazily import the tt-llm-engine ``_migration_client`` extension. Raises
    ImportError if it is not importable."""
    client_dir = os.environ.get("PREFILL_MIGRATION_CLIENT_DIR")
    if client_dir and client_dir not in sys.path:
        sys.path.insert(0, client_dir)
    try:
        import _migration_client  # type: ignore[import-not-found]

        return _migration_client
    except ImportError as e:
        raise ImportError(
            f"[migration] _migration_client not importable ({e}). "
            f"Set PREFILL_MIGRATION_CLIENT_DIR to the dir holding _migration_client*.so, "
            f"or add it to PYTHONPATH."
        ) from e


def _attach_migration_client():
    """Resolve queue names, import ``_migration_client``, and attach. Returns
    (client, cmd_q, table_q, resp_q); raises RuntimeError if the endpoint's shmem
    queues are not reachable (the orchestrator must launch migration_endpoint first)."""
    cmd_q, table_q, resp_q = _resolve_queue_names()
    mod = _import_migration_client()
    try:
        client = mod.MigrationLayerClient(cmd_q, table_q, resp_q)
    except RuntimeError as e:
        raise RuntimeError(
            f"[migration] could not attach MigrationLayerClient to queues "
            f"({cmd_q}, {table_q}, {resp_q}): {e}. The orchestrator / inference server "
            f"must launch migration_endpoint and create the shmem queues before the runner."
        ) from e
    return client, cmd_q, table_q, resp_q


def _enumerate_devices(mesh_device) -> list[tuple[int, int, int]]:
    """Row-major ``(umd_chip_id, fabric_mesh_id, fabric_chip_id)`` for this rank's local mesh.

    ``umd_chip_id`` is the chip's HARDWARE-STABLE 64-bit ASIC unique id, resolved from the
    FabricNodeId via ``ttnn.cluster.get_chip_unique_id_from_fabric_node_id`` — the SAME id the
    migration worker keys ``dram_by_umd`` on (``UmdDevice::unique_id()`` from the cluster
    descriptor). It is NOT the process-local logical id (``get_device_id``, 0..n-1) NOR the
    physical UMD ChipId — those don't match the worker's unique-id-keyed device map, so a
    physical-id device map reaches WORKER_READY but fails to resolve at migrate time.
    """
    rows, cols = mesh_device.shape[0], mesh_device.shape[1]
    out: list[tuple[int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            coord = ttnn.MeshCoordinate(r, c)
            fnid = mesh_device.get_fabric_node_id(coord)
            unique_id = int(ttnn.cluster.get_chip_unique_id_from_fabric_node_id(int(fnid.mesh_id), int(fnid.chip_id)))
            out.append((unique_id, int(fnid.mesh_id), int(fnid.chip_id)))
    return out


def _build_device_map(mesh_device, mesh_shape) -> list[tuple[int, int, int]]:
    """Single-rank device map for the migration endpoint, ordered per the
    ``send_device_map`` binding: ``(fabric_node_mesh_id, fabric_node_chip_id,
    umd_chip_id)``. Sanity-checks mesh size and fabric-node uniqueness so a
    misconfigured PREFILL_SP/PREFILL_TP fails loud instead of producing a
    half-filled device map."""
    raw = _enumerate_devices(mesh_device)
    expected = int(mesh_shape[0]) * int(mesh_shape[1])
    if len(raw) != expected:
        raise RuntimeError(
            f"[migration] mesh enumeration returned {len(raw)} chips but mesh_shape={mesh_shape} "
            f"expects {expected}. Check PREFILL_SP/PREFILL_TP vs the actual mesh device shape."
        )
    device_map = [(mesh, fchip, umd) for (umd, mesh, fchip) in raw]
    unique_fnids = {(m, c) for (m, c, _) in device_map}
    if len(unique_fnids) != len(device_map):
        raise RuntimeError(
            f"[migration] fabric-node collision inside the mesh: {len(device_map)} entries but only "
            f"{len(unique_fnids)} unique (mesh_id, chip_id) pairs. Device map: {device_map}."
        )
    return device_map


def serialize_device_map(mesh_device, path: str) -> str:
    """Write a JSON {"<mesh_id>:<chip_id>": <asic_unique_id>} device map so a device-less consumer
    (the prefill_producer) can resolve a table's FabricNodeIds to the ASIC unique_id that
    read_dram_umd / the migration worker key device reads on. Reuses _enumerate_devices (which calls
    ttnn.cluster.get_chip_unique_id_from_fabric_node_id). Pairs with the mock KV chunk table."""
    import json
    import os

    enumerated = _enumerate_devices(mesh_device)
    device_map = {f"{mesh}:{chip}": unique_id for (unique_id, mesh, chip) in enumerated}
    if len(device_map) != len(enumerated):
        raise RuntimeError(
            f"[migration] device-map fabric-node collision: {len(enumerated)} chips but only "
            f"{len(device_map)} unique (mesh_id, chip_id) keys"
        )
    # Atomic publish: write a temp file then rename, so a concurrent reader never sees partial JSON.
    tmp = f"{path}.tmp"
    with open(tmp, "w") as mp:
        json.dump(device_map, mp)
    os.replace(tmp, path)
    logger.info(f"[migration] device map ({len(device_map)} chips) serialized to {path}")
    return path


def publish_table_and_wait_ready(
    *, mesh_device, mesh_shape, table_path: str, wait_ready_timeout_ms: int = 120_000
) -> None:
    """Publish an already-serialized KV-chunk table to the migration worker and block
    on WORKER_READY (or raise on timeout).

    The runner calls this AFTER the runtime serialized the table
    (``runtime.build_kv_chunk_table(path)``). Builds the device map from the local
    mesh, attaches a ``MigrationLayerClient`` on the env-driven shmem queues, sends
    SET_TABLE and AssignDevMap in the order the worker expects, then waits. The client
    is scope-local to this call; runtime migrations are issued by the C++
    PrefillScheduler over its own adapter.

    Strict-by-default: any failure to import the extension, attach to the queues, or
    reach WORKER_READY raises.
    """
    device_map = _build_device_map(mesh_device, mesh_shape)

    client, cmd_q, table_q, resp_q = _attach_migration_client()

    logger.info(
        f"[migration] publishing: table={table_path} devices={len(device_map)} "
        f"queues=(cmd={cmd_q}, table={table_q}, resp={resp_q}) wait_ready_ms={wait_ready_timeout_ms}"
    )
    client.send_kv_chunk_table(table_path)
    client.send_device_map(device_map)
    client.wait_ready(wait_ready_timeout_ms)
    logger.info(f"[migration] WORKER_READY: devices={len(device_map)} table={table_path}")
