# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""KV chunk address table publish for the prefill runner.

STARTUP step: build the KvChunkAddressTable from the device KV layout, serialize
it to a protobuf file, then publish it (plus the device map) to a running
migration_endpoint over its shmem queues and block until WORKER_READY. The
runner owns the device, so it is the ONLY component that knows the KV cache's
NoC addresses (``kvpe_cache.buffer_address()``) and the local mesh's UMD chip
ids — the worker cannot reach WORKER_READY without both inputs.

Bring-up order is fixed by the worker contract
(``disaggregation/migration/src/worker/control_thread.cpp::maybe_emit_worker_ready``):

    1. send_kv_chunk_table(table_path)       # SetTable on table_q
    2. send_device_map(entries)              # AssignDevMap + N x DevMapEntry on table_q
    3. wait_ready(timeout_ms)                # blocks on RespOpcode::WorkerReady

Without step 2 the worker never opens UMD chips and never asserts WORKER_READY;
without step 3 the runner enters its request loop before the scheduler can
actually issue migrations. ``publish_kv_chunk_table_and_wait_ready()`` is the
one entrypoint that does all three in order; the older split helpers
(``build_and_serialize_kv_chunk_table`` and ``send_kv_chunk_table``) stay
exported so tests can exercise the halves individually.

Serialization uses the ttnn binding
``ttnn.experimental.disaggregation.export_to_protobuf_file`` (no separate
_migration extension needed). The client extension (``_migration_client``)
lives in tt-llm-engine and is imported LAZILY so the runner still works
standalone when migration is opted out.

NOTE: per-layer LayerAck channel + scheduler-driven migration are NOT here
(owned by the scheduler/worker side).
"""

import os
import sys

from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    create_kv_chunk_address_table_deepseek,
)

# bfp8 [1, 1, 32, 576] KV chunk: 18 tiles * 1088 B = 19584 B.
_CHUNK_SIZE_BYTES = 19584

# Sentinel dst_slot the inference server sends when a request must NOT trigger
# migration (e.g. warmup probes, scheduler-skipped requests).
INVALID_SLOT_ID = 0xFFFFFFFF

# Default shmem queue names. Overridable via PREFILL_MIGRATION_{CMD,TABLE,RESP}_QUEUE.
_DEFAULT_CMD_QUEUE = "/prefill_mig_cmd_1"
_DEFAULT_TABLE_QUEUE = "/prefill_mig_tbl_1"
_DEFAULT_RESP_QUEUE = "/prefill_mig_rsp_1"


def _disaggregation():
    """KvChunkAddressTable + serializer come from ttnn.experimental.disaggregation
    (no _migration extension needed)."""
    return ttnn.experimental.disaggregation


def _serialize_table_to_path(table, path: str) -> None:
    """Serialize a KvChunkAddressTable to a protobuf file for the worker's SET_TABLE."""
    _disaggregation().export_to_protobuf_file(table, path)


def _resolve_queue_names() -> tuple[str, str, str]:
    return (
        os.environ.get("PREFILL_MIGRATION_CMD_QUEUE", _DEFAULT_CMD_QUEUE),
        os.environ.get("PREFILL_MIGRATION_TABLE_QUEUE", _DEFAULT_TABLE_QUEUE),
        os.environ.get("PREFILL_MIGRATION_RESP_QUEUE", _DEFAULT_RESP_QUEUE),
    )


def _import_migration_client(strict: bool):
    """Lazily import the tt-llm-engine ``_migration_client`` extension.

    Returns the module on success. On import failure: if ``strict`` raises
    ImportError; otherwise logs a warning and returns ``None`` (back-compat
    path for callers that just want best-effort publish).
    """
    client_dir = os.environ.get("PREFILL_MIGRATION_CLIENT_DIR")
    if client_dir and client_dir not in sys.path:
        sys.path.insert(0, client_dir)
    try:
        import _migration_client  # type: ignore[import-not-found]

        return _migration_client
    except ImportError as e:
        msg = (
            f"[migration] _migration_client not importable ({e}). "
            f"Set PREFILL_MIGRATION_CLIENT_DIR to the dir holding _migration_client*.so, "
            f"or add it to PYTHONPATH."
        )
        if strict:
            raise ImportError(msg) from e
        logger.warning(msg)
        return None


def _attach_migration_client(strict: bool):
    """Resolve queue names, import ``_migration_client``, and attach.

    Returns (client, cmd_q, table_q, resp_q) on success. On failure with
    ``strict=True`` raises RuntimeError (the runner has opted in and the
    endpoint must be reachable). With ``strict=False`` returns ``(None, ...)``
    and logs a warning, matching the legacy best-effort behavior of
    ``send_kv_chunk_table``.
    """
    cmd_q, table_q, resp_q = _resolve_queue_names()
    mod = _import_migration_client(strict=strict)
    if mod is None:
        return None, cmd_q, table_q, resp_q
    try:
        client = mod.MigrationLayerClient(cmd_q, table_q, resp_q)
    except RuntimeError as e:
        msg = (
            f"[migration] could not attach MigrationLayerClient to queues "
            f"({cmd_q}, {table_q}, {resp_q}): {e}. The orchestrator / inference server "
            f"must launch migration_endpoint and create the shmem queues before the runner."
        )
        if strict:
            raise RuntimeError(msg) from e
        logger.warning(msg)
        return None, cmd_q, table_q, resp_q
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


def build_and_serialize_kv_chunk_table(
    *, mesh_device, kvpe_cache, seq_len, num_layers, mesh_shape, sp_axis, num_users, chunk_size_global, path
) -> str:
    """Build the KV chunk address table from the device KV layout and serialize it to ``path``
    for the inference server to forward via SET_TABLE. Returns the path on success.

    Uses the DeepSeek BLOCK-CYCLIC builder (create_kv_chunk_address_table_deepseek): the DeepSeek
    non-balanced prefill cache stores positions block-cyclic across the SP shards (NOT the Kimi
    contiguous-block layout), so each natural position must map to its true block-cyclic storage
    chip + offset — otherwise a partial migration copies the wrong (un-prefilled) storage chunks
    and the migrated KV fails its PCC check. ``chunk_size_global`` is the prefill chunk size (the
    block-cyclic period; same value passed to blockcyclic_positions)."""
    cfg = _disaggregation().KvChunkAddressTableConfig()
    cfg.num_layers = num_layers
    cfg.max_sequence_length = seq_len
    cfg.num_slots = num_users
    cfg.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    cfg.chunk_size_bytes = _CHUNK_SIZE_BYTES
    table = create_kv_chunk_address_table_deepseek(
        config=cfg,
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tt_kvpe_cache=kvpe_cache,
        chunk_size_bytes=_CHUNK_SIZE_BYTES,
        num_users=num_users,
        chunk_size_global=chunk_size_global,
    )
    _serialize_table_to_path(table, path)
    logger.info(f"[migration] KV chunk address table serialized to {path} (entries={table.total_entries()})")
    return path


# def build_device_map(mesh_device, mesh_shape):
#     """Build the migration device map from the runner's live mesh.

#     The migration_endpoint (device mode) opens/keys the model's chips via UMD from
#     these entries — the fnid<->umd mapping is only resolvable by a CreateDevice-capable
#     process (the runner), so it must be streamed to the worker, not guessed. Each entry
#     is ``(fabric_node_mesh_id, fabric_node_chip_id, umd_chip_id)`` (see
#     MigrationLayerClient::send_device_map and worker_context.hpp). Mirrors the device-map
#     construction in disaggregation/tests/test_mla_prefill_migration.py.
#     """
#     rows, cols = int(mesh_shape[0]), int(mesh_shape[1])
#     entries = []
#     for row in range(rows):
#         for col in range(cols):
#             coord = ttnn.MeshCoordinate(row, col)
#             fnid = mesh_device.get_fabric_node_id(coord)
#             umd_chip = mesh_device.get_device_id(coord)
#             entries.append((int(fnid.mesh_id), int(fnid.chip_id), int(umd_chip)))
#     logger.info(f"[migration] built device map: {len(entries)} chips from mesh_shape={rows}x{cols}")
#     return entries


# def send_kv_chunk_table(table_path: str, device_map=None) -> bool:
#     """Forward the serialized table to the migration_worker via SET_TABLE, using
#     the tt-llm-engine MigrationLayerClient. If ``device_map`` is given (a list of
#     ``(mesh_id, fnid_chip, umd_chip)`` tuples from build_device_map), also send it via
#     ``send_device_map`` so a DEVICE-mode endpoint opens the model's chips and reaches
#     WORKER_READY (an empty/None map leaves the worker gated unless it is synthetic).

#     The client extension (``_migration_client``) lives in tt-llm-engine (the
#     superproject), so it is imported LAZILY here — only when migration is actually
#     being driven — to avoid a hard tt-metal -> tt-llm-engine dependency. The call
#     no-ops gracefully (returns False, logs a warning) if the extension or the
#     orchestrator-owned shmem queues are not present, so a standalone runner still
#     writes the .pb without requiring the full migration stack.

#     Env:
#       PREFILL_MIGRATION_CLIENT_DIR  dir holding _migration_client*.so (optional if
#                                     already on PYTHONPATH)
#       PREFILL_MIGRATION_CMD_QUEUE / _TABLE_QUEUE / _RESP_QUEUE  shmem queue names
#                                     (must match the orchestrator/IS that owns them)
#     """
#     client_dir = os.environ.get("PREFILL_MIGRATION_CLIENT_DIR")
#     if client_dir and client_dir not in sys.path:
#         sys.path.insert(0, client_dir)
#     try:
#         import _migration_client
#     except ImportError as e:
#         logger.warning(
#             f"[migration] _migration_client not importable ({e}); table written to {table_path} "
#             f"but NOT sent. Set PREFILL_MIGRATION_CLIENT_DIR or add the extension to PYTHONPATH."
#         )
#         return False


def send_kv_chunk_table(table_path: str) -> bool:
    """Forward the serialized table to the migration_worker via SET_TABLE only.

    Best-effort, back-compat helper. Use ``publish_kv_chunk_table_and_wait_ready``
    in production — by itself this call cannot bring the worker to WORKER_READY
    (the worker also needs the device map; see module docstring).
    """
    client, cmd_q, table_q, resp_q = _attach_migration_client(strict=False)
    if client is None:
        logger.warning(
            f"[migration] table written to {table_path} but NOT sent (no client). "
            f"Use publish_kv_chunk_table_and_wait_ready for the full bring-up."
        )
        return False
    client.send_kv_chunk_table(table_path)
    logger.info(f"[migration] sent SET_TABLE({table_path}) via MigrationLayerClient on {table_q}")
    if device_map is not None:
        client.send_device_map(device_map)
        logger.info(
            f"[migration] sent device map ({len(device_map)} chips) via MigrationLayerClient on {table_q}; "
            f"device-mode endpoint will open these chips and emit WORKER_READY"
        )
    return True


def publish_kv_chunk_table_and_wait_ready(
    *,
    mesh_device,
    kvpe_cache,
    seq_len: int,
    num_layers: int,
    mesh_shape,
    sp_axis: int,
    num_users: int,
    chunk_size_global: int,
    path: str,
    wait_ready_timeout_ms: int = 120_000,
) -> None:
    """Full migration bring-up from the prefill runner side.

    Builds + serializes the KV chunk address table, attaches a
    ``MigrationLayerClient`` on the env-driven shmem queues, sends SET_TABLE and
    AssignDevMap in the order the worker expects, then blocks on WORKER_READY
    (or raises on timeout). The client is scope-local to this call; runtime
    migrations are issued by the C++ PrefillScheduler over its own adapter.

    ``chunk_size_global`` is the prefill chunk size (the block-cyclic period;
    same value passed to blockcyclic_positions). It is required by
    ``build_and_serialize_kv_chunk_table`` to map natural positions to the
    true block-cyclic storage chunks — without it slot-1 migrated reads land
    at the wrong SP shards and PCC collapses to ~0.70 (the half-correct
    pattern we hit before the chunk_size_global plumbing landed).

    Strict-by-default: any failure to import the extension, attach to the
    queues, or reach WORKER_READY raises. Callers that want best-effort
    "publish-if-possible" semantics should use ``build_and_serialize_kv_chunk_table``
    + ``send_kv_chunk_table`` directly.
    """
    build_and_serialize_kv_chunk_table(
        mesh_device=mesh_device,
        kvpe_cache=kvpe_cache,
        seq_len=seq_len,
        num_layers=num_layers,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_users=num_users,
        chunk_size_global=chunk_size_global,
        path=path,
    )

    device_map = _build_device_map(mesh_device, mesh_shape)

    client, cmd_q, table_q, resp_q = _attach_migration_client(strict=True)
    assert client is not None  # strict=True guarantees this

    logger.info(
        f"[migration] publishing: table={path} devices={len(device_map)} "
        f"queues=(cmd={cmd_q}, table={table_q}, resp={resp_q}) wait_ready_ms={wait_ready_timeout_ms}"
    )
    client.send_kv_chunk_table(path)
    client.send_device_map(device_map)
    client.wait_ready(wait_ready_timeout_ms)
    logger.info(
        f"[migration] WORKER_READY: layers={num_layers} slots={num_users} devices={len(device_map)} " f"table={path}"
    )
