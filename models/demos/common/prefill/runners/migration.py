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


_PER_HOST_FS_PREFIXES = ("/tmp", "/dev/shm", "/run", "/var/tmp")


def migration_table_path_is_explicit() -> bool:
    return bool(os.environ.get("PREFILL_MIGRATION_TABLE_PATH"))


def migration_table_path() -> str:
    if migration_table_path_is_explicit():
        return os.environ["PREFILL_MIGRATION_TABLE_PATH"]
    return f"/tmp/prefill_kv_chunk_table_{os.environ.get('PREFILL_H2D_SERVICE_ID', 'ds_prefill')}.pb"


def is_per_host_storage(path: str) -> bool:
    abs_path = os.path.abspath(path)
    return any(abs_path == p or abs_path.startswith(p + "/") for p in _PER_HOST_FS_PREFIXES)


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


def _attach_migration_client(timeout_s: float | None = None):
    cmd_q, table_q, resp_q = _resolve_queue_names()
    mod = _import_migration_client()
    # Runs at table-publish time, so a ctor race here kills rank 0: wait the transient window out.
    client = _attach_with_retry(
        lambda: mod.MigrationLayerClient(cmd_q, table_q, resp_q),
        f"endpoint queues ({cmd_q})",
        f"[migration] endpoint queues never became attachable (cmd={cmd_q}, table={table_q}, "
        f"resp={resp_q}) — is migration_endpoint running and past queue init?",
        timeout_s,
    )
    return client, cmd_q, table_q, resp_q


# Wait budget for the migration layer's shm queues: PREFILL_MIGRATION_ATTACH_WAIT_S seconds, 0 (default) = forever.
_DEFAULT_MIGRATION_ATTACH_WAIT_S = 0.0
_MIGRATION_ATTACH_HEARTBEAT_S = 15.0


def _migration_attach_wait_s() -> float:
    raw = os.environ.get("PREFILL_MIGRATION_ATTACH_WAIT_S")
    if raw is None or raw.strip() == "":
        return _DEFAULT_MIGRATION_ATTACH_WAIT_S
    try:
        budget = float(raw)
    except ValueError:
        budget = None
    # Negatives and NaN both fall straight through `budget > 0` into an unbounded wait, so reject them
    # here: a typo'd sign must not silently remove the only bound the operator asked for.
    if budget is None or not budget >= 0.0:
        logger.warning(
            f"[migration] PREFILL_MIGRATION_ATTACH_WAIT_S={raw!r} is not a non-negative number; "
            f"waiting indefinitely instead"
        )
        return _DEFAULT_MIGRATION_ATTACH_WAIT_S
    return budget


def _producer_not_ready(err: BaseException) -> bool:
    """True for ctor throws a retry clears: shm_open ENOENT, header not published yet, or the table
    lock file not there yet (migration_client.cpp:175/195/121 -- self-created rank queues init all
    three headers before the lock file). EACCES/mmap never clear, hence the explicit ENOENT match.
    """
    msg = str(err)
    if "header not initialized yet" in msg:
        return True
    return ("shm_open(" in msg or "table lock file" in msg) and "No such file or directory" in msg


def _attach_with_retry(make_client, what: str, on_timeout: str, timeout_s: float | None = None):
    """Construct a MigrationLayerClient, retrying only while its producer is still coming up.
    Finding a queue is not the same as being able to use it; anything else re-raises on attempt one.
    """
    budget = _migration_attach_wait_s() if timeout_s is None else timeout_s
    start = last_log = time.monotonic()
    deadline = start + budget if budget > 0 else None
    while True:
        try:
            return make_client()
        except RuntimeError as e:
            if not _producer_not_ready(e):
                raise
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            raise RuntimeError(on_timeout)
        if now - last_log >= _MIGRATION_ATTACH_HEARTBEAT_S:
            last_log = now
            bound = "no timeout" if deadline is None else f"{budget:.0f}s budget"
            logger.info(
                f"[migration] still waiting for {what} after {now - start:.0f}s ({bound}) — "
                f"is the migration layer up on this host yet?"
            )
        time.sleep(0.25)


def _deliver_local_device_map(device_map, rank: int, timeout_s: float | None = None) -> None:
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

    budget = _migration_attach_wait_s() if timeout_s is None else timeout_s
    start = time.monotonic()
    # None => no deadline: poll forever, but log so the wait is never mistaken for a hang.
    deadline = start + budget if budget > 0 else None
    last_log = start
    trios, skipped = _discover()
    # Queues that are not ours NEVER become usable, and with no deadline the raise below is dead code.
    if skipped and not trios:
        logger.warning(
            f"[migration] {len(skipped)} local worker queue(s) present but NOT accessible by this user — "
            f"likely stale shm from another user's run, which waiting will not clear:\n  " + "\n  ".join(skipped)
        )
    while not trios:
        if deadline is not None and time.monotonic() >= deadline:
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
        now = time.monotonic()
        if now - last_log >= _MIGRATION_ATTACH_HEARTBEAT_S:
            last_log = now
            bound = "no timeout" if deadline is None else f"{budget:.0f}s budget"
            inaccessible = f", {len(skipped)} present but not ours" if skipped else ""
            logger.info(
                f"[migration] still waiting for local worker queues (/dev/shm/ep_*_{{a,b}}_cmd*) after "
                f"{now - start:.0f}s ({bound}{inaccessible}) — is the migration layer up on this host yet?"
            )
        time.sleep(0.25)
        trios, skipped = _discover()

    # _discover() only proves the cmd file is ours, not that the worker is past init: same wait applies.
    for cmd, table, resp in trios:
        try:
            client = _attach_with_retry(
                lambda cmd=cmd, table=table, resp=resp: mod.MigrationLayerClient(cmd, table, resp),
                f"worker queue header init ({resp})",
                f"[migration] worker queues found but never became attachable ({resp}) — is the "
                f"migration_worker for this host healthy?",
                timeout_s,
            )
            client.send_device_map(device_map)
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
