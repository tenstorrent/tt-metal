# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


import glob
import os
import socket
import sys
import time
import zlib
from ctypes import c_int32
from typing import NamedTuple

from loguru import logger

import ttnn

_DEFAULT_DEVICE_MAP_FILE = "/tmp/prefill_device_map.txt"


class KvCacheStage(NamedTuple):
    base_addr: int
    first_layer: int
    count: int


def migration_file_export_enabled() -> bool:
    return os.environ.get("PREFILL_MIGRATION_EXPORT_TO_FILE", "0") == "1"


def migration_device_map_file_path() -> str:
    return os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", _DEFAULT_DEVICE_MAP_FILE)


def _disaggregation():
    return ttnn.experimental.disaggregation


def _serialize_table_to_path(table, path: str) -> None:
    tmp = f"{path}.tmp"
    _disaggregation().export_to_protobuf_file(table, tmp)
    os.replace(tmp, path)


def _resolve_queue_names() -> tuple[str, str, str]:
    return (
        os.environ.get("PREFILL_MIGRATION_CMD_QUEUE", "/prefill_mig_cmd_1"),
        os.environ.get("PREFILL_MIGRATION_TABLE_QUEUE", "/prefill_mig_tbl_1"),
        os.environ.get("PREFILL_MIGRATION_RESP_QUEUE", "/prefill_mig_rsp_1"),
    )


def _import_migration_client():
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


def _deliver_local_device_map(device_map, rank: int, timeout_s: float = 30.0) -> None:
    mod = _import_migration_client()

    def _discover():
        trios = []
        skipped = []
        for side in ("a", "b"):
            candidates = glob.glob(f"/dev/shm/ep_*_{side}_cmd") + glob.glob(f"/dev/shm/ep_*_{side}_cmd_r*")
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for c in candidates:
                name = "/" + os.path.basename(c)
                if not os.access(c, os.R_OK | os.W_OK):
                    st = os.stat(c)
                    import pwd

                    owner = pwd.getpwuid(st.st_uid).pw_name
                    skipped.append(
                        f"{name} (owner={owner}, mtime={time.strftime('%Y-%m-%d %H:%M', time.localtime(st.st_mtime))})"
                    )
                    continue
                trios.append(
                    (
                        name,
                        name.replace(f"_{side}_cmd", f"_{side}_table"),
                        name.replace(f"_{side}_cmd", f"_{side}_resp"),
                    )
                )
        return trios, skipped

    deadline = time.monotonic() + timeout_s
    trios, skipped = _discover()
    while not trios:
        if time.monotonic() >= deadline:
            if skipped:
                details = "\n  ".join(skipped)
                raise RuntimeError(
                    f"[migration] local worker queues (/dev/shm/ep_*_{{a,b}}_cmd*) are present "
                    f"but none are accessible by this user. Skipped {len(skipped)}:\n  {details}\n"
                    f"This is usually caused by stale shm files from another user's previous run."
                )
            raise RuntimeError(
                "[migration] no local worker queues (/dev/shm/ep_*_{a,b}_cmd*) on this host -- is the "
                "migration_endpoint/worker for THIS host running? (The /mig_ep* outward queues are the "
                "master-only control channel, NOT the device-map queues.)"
            )
        time.sleep(0.25)
        trios, skipped = _discover()

    for cmd, table, resp in trios:
        try:
            mod.MigrationLayerClient(cmd, table, resp).send_device_map(device_map)
            logger.info(f"[migration] delivered {len(device_map)} local device-map entries -> {cmd}")
        except RuntimeError as e:
            if "Permission denied" in str(e):
                logger.warning(f"[migration] skipping inaccessible worker queue {cmd}: {e}")
                continue
            raise RuntimeError(f"[migration] could not attach to local worker queue {cmd}: {e}") from e


def _enumerate_devices(mesh_device) -> list[tuple[int, int, int]]:
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


def rank_scoped_device_map_path(path: str, rank: int, num_ranks: int) -> str:
    if num_ranks <= 1:
        return path
    stem, ext = os.path.splitext(path)
    return f"{stem}_r{rank}{ext}"


def validate_stage_layout_contiguous(stage_layout) -> int:
    expected = 0
    for s in sorted(stage_layout, key=lambda s: s["first_layer"]):
        if s["first_layer"] != expected:
            raise RuntimeError(
                f"gathered layer ranges are not contiguous: expected next stage at layer {expected} but got "
                f"first_layer={s['first_layer']} (stages={[(x['first_layer'], x['count']) for x in stage_layout]})"
            )
        expected += s["count"]
    return expected


def remove_stale_device_map_sidecars(path: str) -> None:
    stem, ext = os.path.splitext(path)
    for stale in [path, *glob.glob(f"{stem}_r*{ext}")]:
        try:
            os.remove(stale)
            logger.warning(f"[migration] removed stale device map {stale} from a prior run")
        except FileNotFoundError:
            pass


def serialize_device_map(mesh_device, path: str) -> str:
    import json
    import os

    enumerated = _enumerate_devices(mesh_device)
    device_map = {f"{mesh}:{chip}": unique_id for (unique_id, mesh, chip) in enumerated}
    if len(device_map) != len(enumerated):
        raise RuntimeError(
            f"[migration] device-map fabric-node collision: {len(enumerated)} chips but only "
            f"{len(device_map)} unique (mesh_id, chip_id) keys"
        )
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
    cfg = _disaggregation().KvChunkAddressTableConfig()
    cfg.num_layers = num_layers
    cfg.max_sequence_length = max_seq_len
    cfg.num_slots = num_users
    cfg.chunk_n_tokens = chunk_n_tokens
    cfg.chunk_size_bytes = chunk_size_bytes
    table = table_builder(config=cfg, chunk_size_bytes=chunk_size_bytes, num_users=num_users)
    return serialize_prebuilt_kv_chunk_table(table=table, path=path)


def serialize_prebuilt_kv_chunk_table(*, table, path: str) -> str:
    _serialize_table_to_path(table, path)
    logger.info(
        f"[migration] KV chunk address table serialized to {path} "
        f"(configs={table.num_configs()}, entries={table.total_entries()})"
    )
    return path


def export_device_map_to_file(mesh_device, mesh_shape, path: str) -> str:
    device_map = _build_device_map(mesh_device, mesh_shape)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as map_file:
        map_file.writelines(f"{mesh_id} {chip_id} {umd_id}\n" for mesh_id, chip_id, umd_id in device_map)
    os.replace(tmp, path)
    logger.info(f"[migration] device map ({len(device_map)} chips) exported to {path}")
    return path


def export_device_map_file_and_gather_stage_layouts(mesh_device, stages, mesh_shape, device_map_path: str):
    export_device_map_to_file(mesh_device, mesh_shape, device_map_path)
    return allgather_kv_stage_layouts(mesh_device, stages, mesh_shape)


def deliver_device_map_and_gather_stage_layouts(mesh_device, stages, mesh_shape, rank):
    device_map = _build_device_map(mesh_device, mesh_shape)
    _deliver_local_device_map(device_map, rank)
    return allgather_kv_stage_layouts(mesh_device, stages, mesh_shape)


def publish_serialized_table_and_wait_ready(*, table_path: str, wait_ready_timeout_ms: int = 120_000):
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
    return zlib.crc32(socket.gethostname().encode()) & 0x7FFFFFFF


def allgather_kv_stage_layouts(mesh_device, stages, mesh_shape):
    return [
        allgather_kv_stage_layout(mesh_device, stage.base_addr, mesh_shape, stage.first_layer, stage.count)
        for stage in stages
    ]


def allgather_kv_stage_layout(mesh_device, kv_base_addr, mesh_shape, first_layer_idx, num_my_layers):
    rows = mesh_shape[0]
    cols = mesh_shape[1]
    base_addr = int(kv_base_addr)
    num_banks = get_num_dram_banks(mesh_device)

    all_first = ttnn.distributed_context_allgather_int(int(first_layer_idx))
    all_count = ttnn.distributed_context_allgather_int(int(num_my_layers))
    all_lo = ttnn.distributed_context_allgather_int(c_int32(base_addr).value)
    all_hi = ttnn.distributed_context_allgather_int(c_int32(base_addr >> 32).value)
    all_banks = ttnn.distributed_context_allgather_int(int(num_banks))
    all_host = ttnn.distributed_context_allgather_int(_host_tag_int())

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
    return mesh_device.dram_grid_size().x
