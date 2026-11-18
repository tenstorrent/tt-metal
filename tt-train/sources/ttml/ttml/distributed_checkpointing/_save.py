# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Save: read each owned shard off its device once, untilize on the host, write it from its file's own thread.

No tensor is ever gathered, composed or pickled. The main thread issues the device reads (one command-queue batch per
tensor for the coordinates this host owns after replica deduplication) and the host untilize; writer threads only
touch numpy views of host buffers and file descriptors. A byte budget bounds the host memory in flight.
"""

from __future__ import annotations

import os
import secrets
import shutil
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import prod

import numpy as np
import ttnn

from ._errors import CheckpointError
from ._index import (
    FORMAT,
    VERSION,
    FileRecord,
    Index,
    begin_staging,
    check_json,
    data_file_name,
    finalize_staging,
    write_header,
    write_index,
)
from ._planner import SavePlan, create_save_plan
from ._safetensors import create_file, pwrite_all
from ._state_dict import (
    NATIVE,
    LiveTensor,
    collect_model,
    check_threads,
    collect_optimizer,
    fsdp_gate,
    world_record,
)
from ._topology import Chunk, owner


@dataclass
class SaveResult:
    path: str
    bytes_written: int
    files: list[str]
    keys: int
    stats: dict[str, float] = field(default_factory=dict)


class _ByteBudget:
    """Blocks the producer while more than ``limit`` bytes of host buffers are waiting to be written."""

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.used = 0
        self.cv = threading.Condition()

    def acquire(self, nbytes: int) -> None:
        with self.cv:
            while self.used > 0 and self.used + nbytes > self.limit:
                self.cv.wait()
            self.used += nbytes

    def release(self, nbytes: int) -> None:
        with self.cv:
            self.used -= nbytes
            self.cv.notify_all()


def _logical_bytes(view: np.ndarray, sizes: tuple[int, ...], itemsize: int) -> np.ndarray:
    """The row-major logical bytes of a shard from its host buffer.

    A row-major host tensor holds exactly its logical bytes: converting to row-major on the host strips the tile
    padding. A buffer of any other size has a padding or sharding this code does not know the layout of, so it is
    refused rather than sliced by a guessed stride."""
    expected = prod(sizes) * itemsize
    if view.nbytes != expected:
        raise CheckpointError(
            f"host buffer of {view.nbytes} bytes is not the {expected} bytes of a {list(sizes)} shard of itemsize "
            f"{itemsize}; its host layout is padded or sharded in a way save does not handle"
        )
    return view


def _one_process_owner(key: str, group: list[Chunk]) -> Chunk:
    """The writer of a replica group in a single-process save: one process holds every coordinate, so it writes the
    chunk at the coordinate ``owner`` picks."""
    coord = owner(key, [c.mesh_coord for c in group])
    return next(c for c in group if c.mesh_coord == coord)


def _write_chunk(fd: int, offset: int, view: np.ndarray, budget: _ByteBudget, charged: int) -> int:
    try:
        pwrite_all(fd, view, offset)
        return zlib.crc32(view)
    finally:
        budget.release(charged)


def _write_plan(
    staging: str,
    plan: SavePlan,
    by_key: dict[str, LiveTensor],
    threads: int | None,
    max_inflight_bytes: int,
    stats: dict,
) -> dict[tuple[str, int], int]:
    """Create this process's files in ``staging`` and write every planned chunk into them; returns the CRC32 per
    (key, file). Files are synced to the filesystem before they are closed, the first failed write stops the readback
    of the keys after it, and every descriptor opened is closed whatever fails."""
    fds: dict[int, int] = {}
    data_start: dict[int, int] = {}
    budget = _ByteBudget(max_inflight_bytes)
    crcs: dict[tuple[str, int], int] = {}
    t2 = time.perf_counter()
    readback_s = 0.0
    synced = False
    try:
        for fplan in plan.files:
            path = os.path.join(staging, fplan.path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data_start[fplan.index] = create_file(path, fplan.header, fplan.data_bytes)
            fds[fplan.index] = os.open(path, os.O_RDWR)
        workers = max(1, len(plan.files)) if threads is None else threads
        failed: list[BaseException] = []

        def record_failure(future) -> None:
            if not future.cancelled() and future.exception() is not None:
                failed.append(future.exception())

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            order = sorted(plan.items, key=lambda k: (-plan.infos[k].local_nbytes * len(plan.items[k]), k))
            for key in order:
                if failed:
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise failed[0]
                items = plan.items[key]
                info = plan.infos[key]
                live_tensor = by_key[key]
                charged = info.local_nbytes * len(items)
                budget.acquire(charged)
                value = live_tensor.tensor.get_value(NATIVE)
                t_read = time.perf_counter()
                # One read for the owned coordinates; each shard comes back as a single-device host tensor.
                shards = value.cpu_shards([ttnn.MeshCoordinate(list(it.chunk.mesh_coord)) for it in items])
                shards = [shard.to(ttnn.ROW_MAJOR_LAYOUT) for shard in shards]
                readback_s += time.perf_counter() - t_read
                per_item = charged // len(items)
                for it, shard in zip(items, shards):
                    data = shard.host_buffer().get_shard(ttnn.MeshCoordinate(0, 0))  # its one shard
                    view = _logical_bytes(np.from_dlpack(data), it.chunk.box.sizes, info.itemsize)
                    future = pool.submit(
                        _write_chunk, fds[it.file], data_start[it.file] + it.data_offset, view, budget, per_item
                    )
                    future.add_done_callback(record_failure)
                    futures.append((key, it.file, future))
                del shards
            for key, file_index, future in futures:
                crcs[(key, file_index)] = future.result()
        for fd in fds.values():
            os.fdatasync(fd)
        synced = True
    finally:
        for fd in fds.values():
            try:
                os.close(fd)
            except OSError:
                if synced:
                    raise
    stats["readback_s"] = readback_s
    stats["write_s"] = time.perf_counter() - t2
    return crcs


def save(
    ckpt_dir: str,
    *,
    model=None,
    optimizer=None,
    header: dict | None = None,
    threads: int | None = None,
    max_inflight_bytes: int = 4 << 30,
    overwrite: bool = False,
    reshard_fsdp: bool = True,
) -> SaveResult:
    """Write ``model`` and ``optimizer`` to the directory ``ckpt_dir``.

    ``header`` is any JSON-encodable dict (the step, dataloader state, ...), returned by ``load``. ``threads`` caps
    the writer threads (default: one per device data file); ``max_inflight_bytes`` bounds the host buffers waiting
    to be written. An existing
    ``ckpt_dir`` is refused unless ``overwrite=True``. FSDP modules must be in their sharded state; with
    ``reshard_fsdp`` (default) they are resharded first.

    A replicated shard is written once, from one device of its replica group. Save does not check that the replicas
    hold the same values, as ``ttml.checkpointing.save_checkpoint`` does by default (``verify_replicas=True``):
    replicas that drifted apart, say through a gradient that was not all-reduced, load back as that one copy on every
    device. ``ttml.Sharding.from_tensor(t).gather(t, verify_replicas=True)`` checks a tensor when in doubt.
    """
    if model is None and optimizer is None:
        raise TypeError("save needs a model and/or an optimizer")
    if optimizer is not None and model is None:
        raise TypeError("optimizer state is keyed by the model's parameters; pass model= as well")
    if header is not None and not isinstance(header, dict):
        raise TypeError(f"header must be a dict, not {type(header).__name__}")
    check_threads(threads)
    if int(max_inflight_bytes) <= 0:
        raise ValueError(f"max_inflight_bytes must be positive, got {max_inflight_bytes}")
    header = check_json("header", header or {})  # before any device read or file write
    t0 = time.perf_counter()
    stats: dict[str, float] = {}
    fsdp_gate(model, reshard_fsdp)
    live: list[LiveTensor] = collect_model(model)
    scalars: dict = {}
    if optimizer is not None:
        state_live, scalars = collect_optimizer(optimizer, live)
        live.extend(state_live)
    by_key = {lt.key: lt for lt in live}
    world = world_record()
    stats["collect_s"] = time.perf_counter() - t0

    ckpt_dir = os.path.abspath(ckpt_dir)
    if os.path.exists(ckpt_dir) and not overwrite:
        raise CheckpointError(f"{ckpt_dir} already exists; pass overwrite=True to replace it")
    os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
    staging = begin_staging(ckpt_dir, secrets.randbits(31))
    try:
        t1 = time.perf_counter()
        plan: SavePlan = create_save_plan(
            [lt.info for lt in live],
            file_for_coord=lambda c: data_file_name(0, 0, c),
            owner_of=_one_process_owner,
        )
        stats["plan_s"] = time.perf_counter() - t1

        crcs = _write_plan(staging, plan, by_key, threads, max_inflight_bytes, stats)

        write_header(
            staging,
            {"format": FORMAT, "version": VERSION, "checkpoint_id": os.path.basename(ckpt_dir), "user": header or {}},
        )
        index = Index(
            checkpoint_id=os.path.basename(ckpt_dir),
            world=world,
            files=[FileRecord(f.path, len(f.header) + f.data_bytes, 0, 0, f.coord) for f in plan.files],
            entries=plan.entries(crcs),
            scalars=scalars,
            writer_software={"ttml_distributed_checkpointing": "0.1"},
        )
        write_index(staging, index)
        finalize_staging(staging, ckpt_dir, overwrite=overwrite)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    stats["total_s"] = time.perf_counter() - t0
    return SaveResult(ckpt_dir, plan.bytes_written, [f.path for f in plan.files], len(live), stats)
