# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Save: read each owned shard off its device once, untilize it on the host and write it from its file's own thread.

No tensor is ever gathered, composed or pickled. The main thread issues the device reads (one command-queue batch per
tensor for the coordinates this host owns after replica deduplication); each file's writer thread untilizes the
shards queued for it on the host (ttnn releases the GIL there, so the files' shards convert in parallel) and writes
them. A byte budget bounds the host memory in flight.
"""

from __future__ import annotations

import os
import queue
import secrets
import shutil
import threading
import time
import zlib
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
    check_threads,
    collect_model,
    collect_optimizer,
    reshard_fsdp_modules,
    world_record,
)
from ._topology import Chunk, owner
from ._train_state import TRAIN_STATE_KEY, capture_train_state


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


def _writer(jobs: queue.SimpleQueue, crcs: dict, failed: list, stop: threading.Event, budget: _ByteBudget) -> None:
    """A writer thread: untilizes the shards queued for its files on the host, writes them in queue order and records
    each one's CRC32 per (key, file), until it gets None. Once a write or the readback has failed it only releases the
    budget of what is still queued."""
    while (job := jobs.get()) is not None:
        key, file_index, fd, offset, shard, sizes, itemsize, charged = job
        job = None
        try:
            if not stop.is_set():
                row_major = shard.to(ttnn.ROW_MAJOR_LAYOUT)
                shard = None
                data = row_major.host_buffer().get_shard(ttnn.MeshCoordinate(0, 0))  # its one shard
                view = _logical_bytes(np.from_dlpack(data), sizes, itemsize)
                pwrite_all(fd, view, offset)
                crcs[(key, file_index)] = zlib.crc32(view)
        except BaseException as err:
            failed.append(err)
            stop.set()
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
    (key, file). Each file's chunks go through one queue to one writer thread (``threads`` caps the threads, which then
    share the files round-robin), so nothing per chunk outlives its write. Files are synced to the filesystem before
    they are closed, the first failed write stops the readback of the keys after it, and every descriptor opened is
    closed whatever fails."""
    fds: dict[int, int] = {}
    data_start: dict[int, int] = {}
    budget = _ByteBudget(max_inflight_bytes)
    crcs: dict[tuple[str, int], int] = {}
    failed: list[BaseException] = []
    stop = threading.Event()
    queues: list[queue.SimpleQueue] = []
    writers: list[threading.Thread] = []
    t2 = time.perf_counter()
    readback_s = 0.0
    synced = False
    try:
        for fplan in plan.files:
            path = os.path.join(staging, fplan.path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data_start[fplan.index] = create_file(path, fplan.header, fplan.data_bytes)
            fds[fplan.index] = os.open(path, os.O_RDWR)
        workers = len(plan.files) if threads is None else min(threads, len(plan.files))
        queues = [queue.SimpleQueue() for _ in range(workers)]
        queue_of = {fplan.index: queues[i % workers] for i, fplan in enumerate(plan.files)}
        for i, jobs in enumerate(queues):
            writer = threading.Thread(
                target=_writer, args=(jobs, crcs, failed, stop, budget), name=f"dcp-writer-{i}", daemon=True
            )
            writer.start()
            writers.append(writer)
        try:
            order = sorted(plan.items, key=lambda k: (-plan.infos[k].local_nbytes * len(plan.items[k]), k))
            for key in order:
                if failed:
                    break
                items = plan.items[key]
                info = plan.infos[key]
                live_tensor = by_key[key]
                charged = info.local_nbytes * len(items)
                budget.acquire(charged)
                value = live_tensor.tensor.get_value(NATIVE)
                t_read = time.perf_counter()
                # One read for the owned coordinates; each shard comes back as a single-device host tensor, which its
                # file's writer untilizes.
                shards = value.cpu_shards([ttnn.MeshCoordinate(list(it.chunk.mesh_coord)) for it in items])
                readback_s += time.perf_counter() - t_read
                per_item = charged // len(items)
                for it, shard in zip(items, shards):
                    offset = data_start[it.file] + it.data_offset
                    job = (key, it.file, fds[it.file], offset, shard, it.chunk.box.sizes, info.itemsize, per_item)
                    queue_of[it.file].put(job)
                del shards
        except BaseException:
            stop.set()  # the writers drop what is still queued
            raise
        finally:
            for jobs in queues:
                jobs.put(None)
            for writer in writers:
                writer.join()
        if failed:
            raise failed[0]
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
    scheduler=None,
    dataloader=None,
    rng: bool = True,
    threads: int | None = None,
    max_inflight_bytes: int = 4 << 30,
    overwrite: bool = False,
) -> SaveResult:
    """Write ``model`` and ``optimizer`` to the directory ``ckpt_dir``.

    ``header`` is any JSON-encodable dict (the step, the config, ...), returned by ``load``. ``scheduler`` and
    ``dataloader`` are saved through their ``get_state_dict`` (JSON only; ``load`` hands their ``set_state_dict`` the
    state as JSON types, so a tuple comes back as a list) and, with ``rng`` (default), so is the state of every random
    generator training draws from: the ttml generator (an mt19937 in ``AutoContext``), ``ttml.init``'s PCG64, python's,
    numpy's, and torch's when torch is imported; ``load`` restores them with a scheduler or dataloader, or on request.
    ``threads`` caps the writer threads (default: one per device data file); ``max_inflight_bytes`` bounds the host
    buffers waiting to be written. An existing ``ckpt_dir`` is refused unless ``overwrite=True``. FSDP modules are
    resharded first (nothing to do for one already sharded).

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
    # The JSON state is checked before any device read (the FSDP modules may be resharded) or file write.
    header = check_json("header", header or {})
    world = world_record()
    train_state = capture_train_state(
        scheduler=scheduler, dataloader=dataloader, rng=rng, world_rank=world["ranks"][0]["world_rank"]
    )
    t0 = time.perf_counter()
    stats: dict[str, float] = {}
    reshard_fsdp_modules(model)
    live: list[LiveTensor] = collect_model(model)
    scalars: dict = {}
    if optimizer is not None:
        state_live, scalars = collect_optimizer(optimizer, live)
        live.extend(state_live)
    by_key = {lt.key: lt for lt in live}
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
            {
                "format": FORMAT,
                "version": VERSION,
                "checkpoint_id": os.path.basename(ckpt_dir),
                "user": header or {},
                TRAIN_STATE_KEY: train_state,
            },
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
