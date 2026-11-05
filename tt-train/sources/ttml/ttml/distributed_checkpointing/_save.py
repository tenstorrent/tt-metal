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
from typing import Callable

import numpy as np
import ttnn

from ._coordinator import (
    Group,
    PlanPart,
    RankRecord,
    ResultPart,
    UnionPlan,
    current_transport,
    merge_plan_parts,
    merge_results,
    name_hash,
)
from ._errors import CheckpointError, CheckpointFatal
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
    check_label_unchanged,
    collect_model,
    collect_optimizer,
    fsdp_gate,
    rank_record,
    world_record,
)
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
        self.limit = max(int(limit), 1)
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
        workers = len(plan.files) if threads is None else threads
        failed: list[BaseException] = []

        def record_failure(future) -> None:
            if not future.cancelled() and future.exception() is not None:
                failed.append(future.exception())

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = []
            order = sorted(plan.items, key=lambda k: (-plan.infos[k].local_nbytes * len(plan.items[k]), k))
            for key in order:
                if failed:
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise failed[0]
                items = plan.items[key]
                if not items:
                    continue
                info = plan.infos[key]
                live_tensor = by_key[key]
                charged = info.local_nbytes * len(items)
                budget.acquire(charged)
                value = live_tensor.tensor.get_value(NATIVE)
                t_read = time.perf_counter()
                host = value.cpu(coords=[ttnn.MeshCoordinate(list(it.chunk.mesh_coord)) for it in items])
                host_rm = host.to(ttnn.ROW_MAJOR_LAYOUT)
                del host
                readback_s += time.perf_counter() - t_read
                check_label_unchanged(live_tensor, "during readback")
                buffer = host_rm.host_buffer()
                per_item = charged // len(items)
                for it in items:
                    shard = buffer.get_shard(ttnn.MeshCoordinate(list(it.chunk.mesh_coord)))
                    if shard is None:
                        raise CheckpointError(f"{key}: no shard came back for coordinate {it.chunk.mesh_coord}")
                    view = _logical_bytes(np.from_dlpack(shard), it.chunk.box.sizes, info.itemsize)
                    if view.nbytes != it.nbytes:
                        raise CheckpointError(
                            f"{key}: shard at {it.chunk.mesh_coord} is {view.nbytes} bytes, planned {it.nbytes}"
                        )
                    future = pool.submit(
                        _write_chunk, fds[it.file], data_start[it.file] + it.data_offset, view, budget, per_item
                    )
                    future.add_done_callback(record_failure)
                    futures.append((key, it.file, future))
                del buffer, host_rm
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


def _file_metadata(world_rank: int, mesh: int) -> Callable[[tuple[int, ...]], dict]:
    return lambda c: {
        "ttml.format": f"{FORMAT}/{VERSION}",
        "ttml.writer": {"world_rank": world_rank, "mesh_id": mesh, "coord": list(c)},
    }


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
    reshard_fsdp: bool = True,
) -> SaveResult:
    """Write ``model`` and ``optimizer`` to the directory ``ckpt_dir``.

    ``header`` is any JSON-encodable dict (the step, the config, ...), returned by ``load``. ``scheduler`` and
    ``dataloader`` are saved through their ``get_state_dict`` (JSON only) and, with ``rng`` (default), so is the
    state of every random generator training draws from: the ttml generator, python's, numpy's and torch's when
    torch is imported; ``load`` restores them on request. ``threads`` caps the writer threads (default: one per
    device data file); ``max_inflight_bytes`` bounds the host buffers waiting to be written. An existing
    ``ckpt_dir`` is refused unless ``overwrite=True``. FSDP modules must be in their sharded state; with
    ``reshard_fsdp`` (default) they are resharded first.

    Under ``tt-run`` with several ranks the call is collective: every rank calls it at the same point with the same
    ``ckpt_dir``; each rank writes the shards of the coordinates it holds into its own files in a shared directory,
    and rank 0 merges the ranks' parts into one index (``_coordinator``). The result on each rank lists that rank's
    files and bytes; ``keys`` counts the whole checkpoint.
    """
    if model is None and optimizer is None:
        raise CheckpointError("save needs a model and/or an optimizer")
    if optimizer is not None and model is None:
        raise CheckpointError("optimizer state is keyed by the model's parameters; pass model= as well")
    if header is not None and not isinstance(header, dict):
        raise CheckpointError(f"header must be a dict, not {type(header).__name__}")
    header = check_json("header", header or {})  # before any device read or file write
    t0 = time.perf_counter()
    stats: dict[str, float] = {}
    transport = current_transport()
    group = Group(transport) if transport is not None else None

    def collect() -> tuple[list[LiveTensor], dict]:
        fsdp_gate(model, reshard_fsdp)
        live: list[LiveTensor] = collect_model(model)
        scalars: dict = {}
        if optimizer is not None:
            state_live, scalars = collect_optimizer(optimizer, live)
            live.extend(state_live)
        return live, scalars

    # On a multi-rank world even the local collection runs inside a phase, so a rank that fails here cannot leave
    # the others waiting in the first collective.
    live, scalars = collect() if group is None else group.phase("collecting the live tensors", collect)
    by_key = {lt.key: lt for lt in live}
    stats["collect_s"] = time.perf_counter() - t0
    ckpt_dir = os.path.abspath(ckpt_dir)
    args = (ckpt_dir, live, by_key, scalars, header or {}, scheduler, dataloader, rng, threads, max_inflight_bytes)
    if group is None:
        result = _save_single(*args, overwrite=overwrite, stats=stats)
    else:
        result = _save_multi(group, *args, overwrite=overwrite, stats=stats)
    stats["total_s"] = time.perf_counter() - t0
    return result


def _save_single(
    ckpt_dir,
    live,
    by_key,
    scalars,
    header,
    scheduler,
    dataloader,
    rng,
    threads,
    max_inflight_bytes,
    *,
    overwrite,
    stats,
) -> SaveResult:
    world = world_record()
    train_state = capture_train_state(scheduler=scheduler, dataloader=dataloader, rng=rng, world_rank=0)
    if os.path.exists(ckpt_dir) and not overwrite:
        raise CheckpointError(f"{ckpt_dir} already exists; pass overwrite=True to replace it")
    os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
    staging, _ = begin_staging(ckpt_dir)
    try:
        t1 = time.perf_counter()
        mesh = world["ranks"][0]["mesh_id"]
        plan: SavePlan = create_save_plan(
            [lt.info for lt in live],
            file_for_coord=lambda c: data_file_name(0, mesh, c),
            file_metadata=_file_metadata(0, mesh),
        )
        stats["plan_s"] = time.perf_counter() - t1
        crcs = _write_plan(staging, plan, by_key, threads, max_inflight_bytes, stats)
        write_header(
            staging,
            {
                "format": FORMAT,
                "version": VERSION,
                "checkpoint_id": os.path.basename(ckpt_dir),
                "user": header,
                TRAIN_STATE_KEY: train_state,
            },
        )
        index = Index(
            checkpoint_id=os.path.basename(ckpt_dir),
            world=world,
            files=[FileRecord(f.path, len(f.header) + f.data_bytes, 0, mesh, f.coord) for f in plan.files],
            entries=plan.entries(crcs),
            scalars=scalars,
            writer_software={"ttml_distributed_checkpointing": "0.2"},
        )
        write_index(staging, index)
        finalize_staging(staging, ckpt_dir, overwrite=overwrite)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return SaveResult(ckpt_dir, plan.bytes_written, [f.path for f in plan.files], len(live), stats)


def _save_multi(
    group: Group,
    ckpt_dir,
    live,
    by_key,
    scalars,
    header,
    scheduler,
    dataloader,
    rng,
    threads,
    max_inflight_bytes,
    *,
    overwrite,
    stats,
) -> SaveResult:
    """The collective save: see ``_coordinator``. Every phase is followed by a status exchange."""
    rank, world_rank = group.rank, group.world_rank
    record = RankRecord.from_json(rank_record(rank, world_rank))
    name = os.path.basename(ckpt_dir)
    group.agree("the checkpoint directory", name_hash(ckpt_dir))
    group.agree("overwrite", int(bool(overwrite)))
    nonce = group.allgather(secrets.randbits(31))[0]

    def check_target() -> None:
        if group.is_coordinator and os.path.exists(ckpt_dir) and not overwrite:
            raise CheckpointError(f"{ckpt_dir} already exists; pass overwrite=True to replace it")
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)

    group.phase("checking the target directory", check_target)
    staging, _ = begin_staging(ckpt_dir, nonce, shared=True)
    try:
        t1 = time.perf_counter()
        gathered = group.gather(PlanPart(record, [lt.info for lt in live]).to_bytes())

        def merge_parts() -> bytes | None:
            if not group.is_coordinator:
                return None
            return merge_plan_parts([PlanPart.from_bytes(b) for b in gathered]).to_bytes()

        union = UnionPlan.from_bytes(group.broadcast(group.phase("merging the plan parts", merge_parts)))
        stats["plan_s"] = time.perf_counter() - t1

        def write() -> tuple[SavePlan, ResultPart]:
            plan = create_save_plan(
                [lt.info for lt in live],
                file_for_coord=lambda c: data_file_name(world_rank, record.mesh_id, c),
                file_metadata=_file_metadata(world_rank, record.mesh_id),
                owner_of=union.owner_of(rank),
            )
            crcs = _write_plan(staging, plan, by_key, threads, max_inflight_bytes, stats)
            train_state = capture_train_state(
                scheduler=scheduler, dataloader=dataloader, rng=rng, world_rank=world_rank
            )
            files = [
                FileRecord(f.path, len(f.header) + f.data_bytes, world_rank, record.mesh_id, f.coord)
                for f in plan.files
            ]
            result = ResultPart(
                record,
                files,
                plan.entries(crcs),
                train_state.pop("rng", {}),
                plan.bytes_written,
                scalars,
                header,
                train_state,
            )
            return plan, result

        plan, result = group.phase("writing the shards", write)
        gathered = group.gather(result.to_bytes())

        def finalize() -> None:
            if not group.is_coordinator:
                return
            index, header_json = merge_results([ResultPart.from_bytes(b) for b in gathered], union, name)
            for record_ in index.files:
                fd = os.open(os.path.join(staging, record_.path), os.O_RDONLY)  # by exact name: fresh attributes
                try:
                    size = os.fstat(fd).st_size
                finally:
                    os.close(fd)
                if size != record_.nbytes:
                    raise CheckpointError(
                        f"{record_.path}: {size} bytes on the shared filesystem, {record_.nbytes} reported by world rank "
                        f"{record_.world_rank}"
                    )
            write_header(staging, header_json)
            write_index(staging, index)
            finalize_staging(staging, ckpt_dir, overwrite=overwrite)

        group.phase("merging the index and finalizing", finalize)
    except CheckpointFatal:
        raise  # the staging directory is left in place; nothing else can be agreed on
    except BaseException:
        if group.is_coordinator:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    return SaveResult(ckpt_dir, plan.bytes_written, [f.path for f in plan.files], len(union.infos), stats)
