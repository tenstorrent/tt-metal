# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Model-agnostic KV-migration comms for the prefill runner.

The RUNNER owns the control flow; this module provides the generic (model-free) pieces and the model
runtime owns the table build. The split for a migration bring-up:

    1. deliver_device_map_and_gather_stage_layout(...)   # ALL RANKS: deliver local FNID->UMD map to
                                                          #   the co-located worker + all-gather barrier
    2. runtime.build_kv_chunk_table(kv_cache, path, ...) # RANK 0: model builds + serializes the table
                                                          #   (via serialize_kv_chunk_table here)
    3. publish_serialized_table_and_wait_ready(path)     # RANK 0: SetTable + wait for WORKER_READY

Step 1 delivers every rank's device map (the worker keys chips on their hardware unique id and never
gathers the map itself) and doubles as the barrier guaranteeing all maps land before rank 0 SET_TABLEs.
Step 2 is the ONLY model-specific step: ``runtime.build_kv_chunk_table`` calls back into
``serialize_kv_chunk_table`` with a model-supplied builder, so this module never imports a model.
Step 3 attaches the ``MigrationLayerClient`` (SetTable on table_q, then blocks on
RespOpcode::WorkerReady) — without WORKER_READY the runner would enter its request loop before the
scheduler can issue migrations. The ``_migration_client`` extension lives in tt-llm-engine and is
imported LAZILY so the runner still works standalone when migration is opted out.
"""

import os
import socket
import sys
import zlib
from ctypes import c_int32

from loguru import logger

import ttnn


def _disaggregation():
    """KvChunkAddressTable + serializer come from ttnn.experimental.disaggregation
    (no _migration extension needed)."""
    return ttnn.experimental.disaggregation


def _serialize_table_to_path(table, path: str) -> None:
    """Serialize a KvChunkAddressTable to a protobuf file for the worker's SET_TABLE."""
    _disaggregation().export_to_protobuf_file(table, path)


def _resolve_queue_names() -> tuple[str, str, str]:
    return (
        # Default shmem queue names. Overridable via PREFILL_MIGRATION_{CMD,TABLE,RESP}_QUEUE.
        os.environ.get("PREFILL_MIGRATION_CMD_QUEUE", "/prefill_mig_cmd_1"),
        os.environ.get("PREFILL_MIGRATION_TABLE_QUEUE", "/prefill_mig_tbl_1"),
        os.environ.get("PREFILL_MIGRATION_RESP_QUEUE", "/prefill_mig_rsp_1"),
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


def _deliver_local_device_map(device_map) -> None:
    """Deliver THIS rank's FNID->UMD device map on the static outward table queue
    (``PREFILL_MIGRATION_*_QUEUE``, passed in by the migration layer); the endpoint orchestrator
    relays the AssignDevMap/DevMapEntry burst to its internal A/B workers."""
    client, _cmd_q, table_q, _resp_q = _attach_migration_client()
    client.send_device_map(device_map)
    logger.info(f"[migration] delivered {len(device_map)} device-map entries -> {table_q}")


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
    """Model-agnostic KvChunkAddressTable serialize. Owns the generic config setup + protobuf
    serialization; the MODEL owns the address math via the ``table_builder`` callback.

    ``table_builder(config, chunk_size_bytes, num_users)`` receives the populated
    ``KvChunkAddressTableConfig`` and returns a built ``KvChunkAddressTable`` for the model's own
    cache layout (e.g. the deepseek/kimi block-cyclic builder). This keeps the common runner module
    free of any model import — the model calls this from ``runtime.build_kv_chunk_table``.

    Returns ``path`` on success."""
    cfg = _disaggregation().KvChunkAddressTableConfig()
    # Initial value only: a multi-stage builder overwrites cfg.num_layers with the gathered global
    # sum of every stage's layer count (== num_layers when a single stage owns the whole model).
    cfg.num_layers = num_layers
    cfg.max_sequence_length = max_seq_len
    cfg.num_slots = num_users
    cfg.chunk_n_tokens = chunk_n_tokens
    cfg.chunk_size_bytes = chunk_size_bytes
    table = table_builder(config=cfg, chunk_size_bytes=chunk_size_bytes, num_users=num_users)
    _serialize_table_to_path(table, path)
    logger.info(f"[migration] KV chunk address table serialized to {path} (entries={table.total_entries()})")
    return path


def deliver_device_map_and_gather_stage_layout(mesh_device, kvpe_cache, mesh_shape, first_layer_idx, num_my_layers):
    """ALL RANKS run this (the runner drives it for every rank). Deliver THIS rank's local FNID->UMD
    device map to its co-located worker, then join the collective all-gather that merges every stage
    into one table. Returns the gathered ``stage_layout`` (the runner passes it to rank 0's
    ``runtime.build_kv_chunk_table``; non-rank-0 callers just needed to join the collective).

    The delivery happens BEFORE the gather so every rank's map is in place before rank 0 SET_TABLEs;
    the all-gather doubles as the barrier that guarantees it. The gather is an MPI collective -- EVERY
    rank must reach it or the communicator deadlocks.
    """
    device_map = _build_device_map(mesh_device, mesh_shape)
    _deliver_local_device_map(device_map)
    return allgather_kv_stage_layout(mesh_device, kvpe_cache, mesh_shape, first_layer_idx, num_my_layers)


def publish_serialized_table_and_wait_ready(*, table_path: str, wait_ready_timeout_ms: int = 120_000):
    """RANK 0 ONLY. Publish an ALREADY-serialized KV chunk table to the migration worker and block on
    WORKER_READY (or raise on timeout). Returns the attached client.

    The table is built + serialized by the model runtime (``runtime.build_kv_chunk_table`` — the model
    owns the cache layout / block-cyclic address math), and the runner runs the all-ranks device-map
    delivery + all-gather barrier (``deliver_device_map_and_gather_stage_layout``) FIRST, so by the
    time this attaches the endpoint ``MigrationLayerClient`` on the master cmd/table/resp queues and
    SET_TABLEs, every rank's local device map has landed and the worker can reach WORKER_READY.
    """
    client, cmd_q, table_q, resp_q = _attach_migration_client()
    logger.info(
        f"[migration] publishing table={table_path} (queues cmd={cmd_q}, table={table_q}, resp={resp_q}) "
        f"wait_ready_ms={wait_ready_timeout_ms}"
    )
    client.send_kv_chunk_table(table_path)
    client.wait_ready(wait_ready_timeout_ms)
    logger.info(f"[migration] WORKER_READY: table={table_path}")

    return client


def _host_tag_int():
    """Per-host stable 31-bit id (crc32 of hostname, masked to fit the signed-int32 allgather).

    Ranks on the same physical host produce the SAME value; different hosts (almost certainly)
    differ. ``allgather_int`` is the only collective primitive exposed and it carries a signed
    32-bit int, so a hostname STRING cannot be gathered directly — we gather this tag instead and
    rebuild a stable per-host string ``host-{tag:08x}`` on every rank (matching tt-blaze's
    migration_table_hook convention). A host owns multiple mesh rows but a given row never spans
    hosts, so one tag per owning rank correctly groups every FNID to its worker.
    """
    return zlib.crc32(socket.gethostname().encode()) & 0x7FFFFFFF


def allgather_kv_stage_layout(mesh_device, tt_kvpe_cache, mesh_shape, first_layer_idx, num_my_layers):
    """COLLECTIVE (all ranks): all-gather each rank's pipeline-STAGE layout so one merged table can
    span every layer across every host -- tt-blaze's layer->mesh merge strategy.

    In this pipeline-parallel deployment each rank owns a contiguous LAYER range
    ``[first_layer_idx, first_layer_idx + num_my_layers)`` and holds the KV for those layers across
    its FULL mesh (all SP rows x TP cols). A migration worker needs ONE table covering every layer,
    but ``mesh_device``/``buffer_address()`` only expose THIS rank's mesh + KV base. So every rank
    contributes, via ``allgather_int``:

      * its layer range ``(first_layer_idx, num_my_layers)`` -- the analog of tt-blaze's my_layer_id
      * KV-cache base address (64-bit, split lo/hi for the 32-bit int gather)
      * usable DRAM bank count (harvested parts differ, so gather it rather than assume)
      * a per-host tag (see :func:`_host_tag_int`)
      * the ``(mesh_id, chip_id)`` of every ``(row, col)`` fabric node in its FULL mesh

    EVERY rank must call this with identical ``mesh_shape`` (the per-(row,col) loop must be symmetric
    for the collective to line up). Returns a per-rank list of stage dicts:
    ``{rank, first_layer, count, base_addr, num_banks, host_tag, fnids[row][col]}``.
    """
    rows = mesh_shape[0]
    cols = mesh_shape[1]
    base_addr = int(tt_kvpe_cache.buffer_address())
    num_banks = get_num_dram_banks(mesh_device)

    all_first = ttnn.distributed_context_allgather_int(int(first_layer_idx))
    all_count = ttnn.distributed_context_allgather_int(int(num_my_layers))
    # allgather_int carries a SIGNED int32, but a KV base word can set bit 31 (a per-rank buffer
    # landing >= 2 GB in a ~3.98 GB DRAM bank), overflowing the binding. c_int32 reinterprets the
    # low 32 bits as two's-complement, keeping the raw bits; the receiver re-masks with 0xFFFFFFFF.
    all_lo = ttnn.distributed_context_allgather_int(c_int32(base_addr).value)
    all_hi = ttnn.distributed_context_allgather_int(c_int32(base_addr >> 32).value)
    all_banks = ttnn.distributed_context_allgather_int(int(num_banks))
    all_host = ttnn.distributed_context_allgather_int(_host_tag_int())

    # Each rank enumerates its FULL mesh (every SP row x TP col) and gathers (mesh_id, chip_id) per
    # coord -- unlike the layers, the whole mesh belongs to this rank's stage, so no row-splitting.
    all_mesh = [[None] * cols for _ in range(rows)]
    all_chip = [[None] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            fid = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(r, c))
            all_mesh[r][c] = ttnn.distributed_context_allgather_int(int(fid.mesh_id))
            all_chip[r][c] = ttnn.distributed_context_allgather_int(int(fid.chip_id))

    size = len(all_lo)
    stages = []
    for rk in range(size):
        base = ((all_hi[rk] & 0xFFFFFFFF) << 32) | (all_lo[rk] & 0xFFFFFFFF)
        fnids = [
            [ttnn.FabricNodeId(ttnn.MeshId(all_mesh[r][c][rk]), all_chip[r][c][rk]) for c in range(cols)]
            for r in range(rows)
        ]
        stages.append(
            {
                "rank": rk,
                "first_layer": all_first[rk],
                "count": all_count[rk],
                "base_addr": base,
                "num_banks": all_banks[rk],
                "host_tag": all_host[rk],
                "fnids": fnids,
            }
        )
    return stages


def get_num_dram_banks(mesh_device):
    """Usable DRAM banks on this device. Full Blackhole = 8; harvested parts expose fewer (e.g. 7).

    The KV cache ND-shards round-robin across these banks and the disaggregation address table replays
    that exact striping (`curr_bank_id = (curr_bank_id + 1) % num_banks`), so both MUST derive the count
    from the same device. dram_grid_size().x is the number of DRAM cores/banks the device exposes."""
    return mesh_device.dram_grid_size().x
