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

from ._coordinator import (
    Election,
    PlanPart,
    RankRecord,
    ResultPart,
    checked_payload,
    current_group,
    merge_plan_parts,
    merge_results,
    state_hash,
)
from ._errors import CheckpointError, CheckpointFatalError
from ._index import (
    FileRecord,
    begin_staging,
    check_json,
    data_file_name,
    finalize_staging,
    staging_dir,
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
    is_structural_scalar,
    rank_record,
    reshard_fsdp_modules,
)
from ._topology import crc31
from ._train_state import capture_train_state


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
                if not items:
                    continue
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

    Under ``tt-run`` with several ranks the call is collective: every rank calls it at the same point with the same
    ``ckpt_dir``; each rank writes the shards of the coordinates it holds into its own files in a shared directory, and
    rank 0 merges the ranks' parts into one index (``_coordinator``). The ranks' scheduler and dataloader states must be
    equal, and so must their optimizers' scalars (steps, learning rates; every rank passes an optimizer or none), since
    the checkpoint stores one copy; their generators are stored per rank. The user header stored is rank 0's. A failure
    on any rank, the local work included, makes every rank raise the same error class, and no point-to-point MPI traffic
    between the ranks may be outstanding when save is called. No other code may use the MPI tag ``DCP_TAG`` (``0x7DC0``)
    on the world communicator while it runs: the ranks' metadata travels under it. Every rank of the world must call
    save (#65436 tracks a subset of the ranks; #65666 a communicator of its own, without the reserved tag). The result
    on each rank lists that rank's files and bytes; ``keys`` counts the whole checkpoint. A single process is the group
    of one and takes the same path.
    """
    t0 = time.perf_counter()
    stats: dict[str, float] = {}
    group = current_group()
    rank, world_rank = group.rank, group.world_rank

    def collect():
        # The arguments are checked here, inside the first phase: a rank called with a bad one (a None directory from
        # a find_latest that saw no checkpoint) then stops every rank instead of leaving them in the first collective.
        if model is None and optimizer is None:
            raise TypeError("save needs a model and/or an optimizer")
        if optimizer is not None and model is None:
            raise TypeError("optimizer state is keyed by the model's parameters; pass model= as well")
        if header is not None and not isinstance(header, dict):
            raise TypeError(f"header must be a dict, not {type(header).__name__}")
        check_threads(threads)
        if int(max_inflight_bytes) <= 0:
            raise ValueError(f"max_inflight_bytes must be positive, got {max_inflight_bytes}")
        target = os.path.abspath(ckpt_dir)
        # The JSON state is checked before any device read (the FSDP modules may be resharded) or file write.
        user_header = check_json("header", header or {})
        train_state = capture_train_state(scheduler=scheduler, dataloader=dataloader, rng=rng, world_rank=world_rank)
        reshard_fsdp_modules(model)
        live: list[LiveTensor] = collect_model(model)
        scalars: dict = {}
        if optimizer is not None:
            state_live, scalars = collect_optimizer(optimizer, live)
            live.extend(state_live)
        shared_state = {k: v for k, v in train_state.items() if k != "rng"}
        record = RankRecord.from_json(rank_record(rank, world_rank))
        part = PlanPart(record, [lt.info for lt in live], state_hash(shared_state)).to_bytes()
        progress = {k: v for k, v in scalars.items() if not is_structural_scalar(k)}
        payload = checked_payload(part, "this rank's plan part")
        return target, live, scalars, progress, user_header, train_state, record, payload

    # Even the local work runs inside a phase: on a multi-rank world a rank that fails here stops every rank instead
    # of leaving the others waiting in the first collective. On a single process the phase just runs it.
    ckpt_dir, live, scalars, progress, user_header, train_state, record, part = group.phase(
        "collecting the live tensors", collect
    )
    name = os.path.basename(ckpt_dir)
    by_key = {lt.key: lt for lt in live}
    stats["collect_s"] = time.perf_counter() - t0
    group.agree("the checkpoint directory", crc31(ckpt_dir), shown=ckpt_dir)
    group.agree("overwrite", int(bool(overwrite)), shown=bool(overwrite))
    # The index holds one copy of the optimizer's scalars, rank 0's; a rank whose optimizer differs (or that has none)
    # would load rank 0's steps and learning rates without a word. A param group's member list is per rank (a rank of
    # a multi-mesh world holds its own parameters) and is merged instead.
    group.agree("whether an optimizer is saved", int(optimizer is not None), shown=optimizer is not None)
    group.agree("the optimizer's scalars (steps, learning rates, ...)", state_hash(progress), shown=progress)
    nonce = group.allgather(secrets.randbits(31))[0]
    staging = staging_dir(ckpt_dir, nonce)
    created_staging = False  # whether the coordinator made the staging directory, and so may remove it

    def check_target() -> None:
        # The coordinator alone creates the staging directory, and only a new one: an existing one is another save's
        # (of the same step, with the same random nonce) or one a dead save left, and writing into it would mix two
        # saves' files. The other ranks join it in the next phase.
        nonlocal created_staging
        if not group.is_coordinator:
            return
        if os.path.exists(ckpt_dir) and not overwrite:
            raise CheckpointError(f"{ckpt_dir} already exists; pass overwrite=True to replace it")
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        try:
            os.mkdir(staging)
        except FileExistsError:
            raise CheckpointError(
                f"{staging} already exists: another save of this step drew the same nonce, or a save that died left "
                f"it behind. Nothing was written, and the directory is left as it is; a new save draws a new nonce."
            ) from None
        created_staging = True

    try:
        group.phase("checking the target directory", check_target)
        group.phase("creating the staging directory", lambda: begin_staging(ckpt_dir, nonce))
        t1 = time.perf_counter()
        gathered = group.gather(part)

        def merge_parts() -> bytes | None:
            if not group.is_coordinator:
                return None
            parts = [PlanPart.from_bytes(b) for b in gathered]
            return checked_payload(merge_plan_parts(parts).to_bytes(), "the owner election")

        election = Election.from_bytes(group.broadcast(group.phase("merging the plan parts", merge_parts)))
        stats["plan_s"] = time.perf_counter() - t1

        def write() -> tuple[SavePlan, bytes]:
            plan = create_save_plan(
                [lt.info for lt in live],
                file_for_coord=lambda c: data_file_name(world_rank, record.mesh_id, c),
                owner_of=election.owner_of(rank),
            )
            crcs = _write_plan(staging, plan, by_key, threads, max_inflight_bytes, stats)
            files = [
                FileRecord(f.path, len(f.header) + f.data_bytes, world_rank, record.mesh_id, f.coord)
                for f in plan.files
            ]
            shared_state = dict(train_state)
            generators = shared_state.pop("rng", {})
            # The header and the scheduler and dataloader state stored are rank 0's (the latter two agree, by hash).
            first = group.is_coordinator
            result = ResultPart(
                record,
                files,
                plan.entries(crcs),
                generators,
                scalars,
                user_header if first else {},
                shared_state if first else {},
            )
            return plan, checked_payload(result.to_bytes(), "this rank's result part")

        plan, result_bytes = group.phase("writing the shards", write)
        gathered = group.gather(result_bytes)

        def finalize() -> None:
            if not group.is_coordinator:
                return
            index, header_json = merge_results([ResultPart.from_bytes(b) for b in gathered], election, name)
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
    except CheckpointFatalError:
        raise  # the staging directory is left in place; nothing else can be agreed on
    except BaseException:
        if group.is_coordinator and created_staging:  # never another save's directory
            shutil.rmtree(staging, ignore_errors=True)
        raise
    stats["total_s"] = time.perf_counter() - t0
    return SaveResult(ckpt_dir, plan.bytes_written, [f.path for f in plan.files], election.keys, stats)
