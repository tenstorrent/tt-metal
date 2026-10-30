# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


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


def _enumerate_devices(mesh_device) -> list[tuple[int, int, int]]:
    rows, cols = mesh_device.shape[0], mesh_device.shape[1]
    out: list[tuple[int, int, int]] = []
    for r in range(rows):
        for c in range(cols):
            coord = ttnn.MeshCoordinate(r, c)
            fnid = mesh_device.get_fabric_node_id(coord)
            unique_id = int(
                ttnn.cluster.get_chip_unique_id_from_fabric_node_id(mesh_device, int(fnid.mesh_id), int(fnid.chip_id))
            )
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


def _write_device_map_lines(device_map, path: str, rank: int) -> None:
    tmp = f"{path}.tmp.{rank}"
    with open(tmp, "w", encoding="utf-8") as map_file:
        map_file.writelines(f"{mesh_id} {chip_id} {umd_id}\n" for mesh_id, chip_id, umd_id in device_map)
    os.replace(tmp, path)


def device_map_rank_sidecar_path(path: str, rank: int) -> str:
    return f"{path}.r{rank}"


def write_device_map_rank_sidecar(device_map, path: str, rank: int) -> str:
    """Write this rank's chips to its own sidecar and drop any merged map a prior run left at ``path``.

    Ranks sharing a host share ``path`` (one file per node); writing it directly was last-writer-wins.
    Called before a collective, so every rank's removal precedes every rank's merge."""
    try:
        os.remove(path)
        logger.warning(f"[migration] removed stale device map {path} from a prior run")
    except FileNotFoundError:
        pass
    sidecar = device_map_rank_sidecar_path(path, rank)
    _write_device_map_lines(device_map, sidecar, rank)
    return sidecar


def merge_device_map_rank_sidecars(path: str, host_ranks, rank: int) -> int:
    """Union the sidecars of ``host_ranks`` (every rank on this host) into ``path``.

    Every rank on the host writes the same content, so the concurrent replaces are idempotent and the
    merged file never appears with only some ranks' chips."""
    merged = {}
    for host_rank in sorted(host_ranks):
        sidecar = device_map_rank_sidecar_path(path, host_rank)
        with open(sidecar, encoding="utf-8") as map_file:
            for line in map_file:
                if not line.strip():
                    continue
                mesh_id, chip_id, umd_id = (int(field) for field in line.split())
                if merged.setdefault((mesh_id, chip_id), umd_id) != umd_id:
                    raise RuntimeError(
                        f"[migration] device map sidecars disagree on fabric node ({mesh_id}, {chip_id}): "
                        f"{merged[(mesh_id, chip_id)]} vs {umd_id} in {sidecar}"
                    )
    _write_device_map_lines([(m, c, u) for (m, c), u in sorted(merged.items())], path, rank)
    return len(merged)


def export_device_map_file_and_gather_stage_layouts(mesh_device, stages, mesh_shape, device_map_path: str):
    rank = int(ttnn.distributed_context_get_rank())
    write_device_map_rank_sidecar(_build_device_map(mesh_device, mesh_shape), device_map_path, rank)
    # A collective: past it, every rank's sidecar exists.
    stage_layouts = allgather_kv_stage_layouts(mesh_device, stages, mesh_shape)
    host_tag = _host_tag_int()
    host_ranks = [s["rank"] for s in stage_layouts[0] if s["host_tag"] == host_tag]
    num_chips = merge_device_map_rank_sidecars(device_map_path, host_ranks, rank)
    logger.info(f"[migration] device map ({num_chips} chips, ranks {host_ranks}) exported to {device_map_path}")
    return stage_layouts


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


def setup_kv_chunk_table(
    runtime, kv_caches, mesh_device, mesh_shape, *, rank: int, num_ranks: int, first_layer_idx: int, num_my_layers: int
):
    """Gather every rank's KV stage layouts, export each host's device map, and have rank 0 write the merged
    chunk table.

    The table goes to ``PREFILL_MIGRATION_TABLE_PATH`` (shared storage past one rank) and the device map to
    ``PREFILL_MIGRATION_DEVICE_MAP_PATH`` (per-host storage past one rank), the layout the KV migration
    service reads in deployment.

    A runtime may report no stage (GPT-OSS with a bounded sliding cache); that serves without a table, and
    only single-rank, since every rank must join the same stage all-gathers."""
    table_path = migration_table_path()
    device_map_path = migration_device_map_file_path()

    if num_ranks > 1 and is_per_host_storage(table_path):
        if migration_table_path_is_explicit():
            raise ValueError(
                f"PREFILL_MIGRATION_TABLE_PATH={os.path.abspath(table_path)} is on per-host storage; "
                f"with num_ranks={num_ranks} the table rank 0 writes is invisible to the other hosts' "
                "readers. Point it at shared/NFS storage (e.g. /data/...)."
            )
        logger.warning(
            f"[migration] KV chunk table defaults to per-host {table_path} at num_ranks={num_ranks}; "
            "readers on other hosts cannot see it. Set PREFILL_MIGRATION_TABLE_PATH to shared storage "
            "if one is expected."
        )

    if num_ranks > 1 and not is_per_host_storage(device_map_path):
        raise ValueError(
            f"PREFILL_MIGRATION_DEVICE_MAP_PATH={os.path.abspath(device_map_path)} is not on per-host storage; "
            f"with num_ranks={num_ranks} every host writes its own chips to that one path. Point it at a "
            f"host-local path (one of {', '.join(_PER_HOST_FS_PREFIXES)})."
        )

    if rank == 0 and os.path.exists(table_path):
        logger.warning(f"[migration] removing stale KV chunk table {table_path} from a prior run")
        os.remove(table_path)

    if not hasattr(runtime, "kv_migration_stages"):
        raise RuntimeError(
            f"runtime {type(runtime).__name__} does not implement kv_migration_stages, so its KV cache "
            "layout cannot be described and no chunk table can be built (see docs/ADDING_A_PREFILL_MODEL.md §2)."
        )
    kv_stages = runtime.kv_migration_stages(kv_caches, first_layer_idx, num_my_layers)

    if not kv_stages:
        if num_ranks > 1:
            raise RuntimeError(
                f"[migration] rank {rank}: {type(runtime).__name__} reported no KV cache stage, so no chunk "
                f"table can be built, but num_ranks={num_ranks} needs one. Serve single-rank, or give the "
                "cache a stage."
            )
        logger.warning(
            f"[migration] {type(runtime).__name__} reported no KV cache stage (GPT-OSS does this for a "
            "bounded sliding cache), so no chunk table is built and this cache cannot be migrated."
        )
        return

    stage_layouts = export_device_map_file_and_gather_stage_layouts(mesh_device, kv_stages, mesh_shape, device_map_path)

    if rank != 0:
        logger.info(
            f"[migration] rank {rank}: contributed stage (first_layer={first_layer_idx}, "
            f"count={num_my_layers}); rank 0 owns the merged table."
        )
        return

    table_path = runtime.build_kv_chunk_table(
        kv_caches,
        table_path,
        first_layer_idx=first_layer_idx,
        num_my_layers=num_my_layers,
        stage_layouts=stage_layouts,
    )
    logger.info(f"[migration] merged KV chunk table -> {table_path}")
