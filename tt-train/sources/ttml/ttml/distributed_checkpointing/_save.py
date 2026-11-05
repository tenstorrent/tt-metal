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
    world_record,
)


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
    """The row-major logical bytes of a shard from its host buffer, which may pad the last dim to an alignment."""
    expected = prod(sizes) * itemsize
    if view.nbytes == expected:
        return view
    rows = prod(sizes[:-1])
    if rows == 0 or view.nbytes % rows:
        raise CheckpointError(
            f"host buffer of {view.nbytes} bytes does not hold a {list(sizes)} shard of itemsize {itemsize}"
        )
    physical_width = view.nbytes // rows
    return np.ascontiguousarray(view.reshape(rows, physical_width)[:, : sizes[-1] * itemsize]).reshape(-1)


def _write_chunk(fd: int, offset: int, view: np.ndarray, budget: _ByteBudget, charged: int) -> int:
    try:
        pwrite_all(fd, view, offset)
        return zlib.crc32(view)
    finally:
        budget.release(charged)


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
    """
    if model is None and optimizer is None:
        raise CheckpointError("save needs a model and/or an optimizer")
    if optimizer is not None and model is None:
        raise CheckpointError("optimizer state is keyed by the model's parameters; pass model= as well")
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
    staging, _ = begin_staging(ckpt_dir)
    try:
        t1 = time.perf_counter()
        plan: SavePlan = create_save_plan(
            [lt.info for lt in live],
            file_for_coord=lambda c: data_file_name(0, 0, c),
            file_metadata=lambda c: {
                "ttml.format": f"{FORMAT}/{VERSION}",
                "ttml.writer": {"world_rank": 0, "mesh_id": 0, "coord": list(c)},
            },
        )
        stats["plan_s"] = time.perf_counter() - t1

        fds: dict[int, int] = {}
        data_start: dict[int, int] = {}
        for fplan in plan.files:
            path = os.path.join(staging, fplan.path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data_start[fplan.index] = create_file(path, fplan.header, fplan.data_bytes)
            fds[fplan.index] = os.open(path, os.O_RDWR)

        budget = _ByteBudget(max_inflight_bytes)
        crcs: dict[tuple[str, int], int] = {}
        t2 = time.perf_counter()
        readback_s = 0.0
        try:
            workers = len(plan.files) if threads is None else threads
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = []
                order = sorted(plan.items, key=lambda k: (-plan.infos[k].local_nbytes * len(plan.items[k]), k))
                for key in order:
                    items = plan.items[key]
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
                        futures.append(((key, it.file), future))
                    del buffer, host_rm
                for key_file, future in futures:
                    crcs[key_file] = future.result()
        finally:
            for fd in fds.values():
                os.fdatasync(fd)
                os.close(fd)
        stats["readback_s"] = readback_s
        stats["write_s"] = time.perf_counter() - t2

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
