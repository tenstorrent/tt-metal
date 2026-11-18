# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Load: read only the bytes each destination shard needs, place them on the device in place.

Every live tensor's label gives its destination boxes; the planner turns them into reads of sub-boxes of saved
chunks. A reader pool fills one host array per unique destination shard, each array becomes a zero-copy host
tensor, the shards are assembled under the live tensor's own label, moved to the device, tilized, checked against
the live TensorSpec and installed with ``set_value``. The live label is asserted unchanged afterwards, so a load can
never corrupt the next save.
"""

from __future__ import annotations

import os
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import prod

import numpy as np
import ttnn

from ._errors import CheckpointError, IncompleteCheckpointError, MissingKeyError, UnexpectedKeyError
from ._index import Index, is_complete, read_checked_header, read_index
from ._planner import KeyPlan, LoadPlan, ReadItem, create_load_plan
from ._safetensors import pread_all, read_file_header
from ._state_dict import (
    NATIVE,
    NUMPY_CARRIERS,
    TTNN_DTYPES,
    LiveTensor,
    check_label_unchanged,
    check_param_groups,
    check_threads,
    check_undeclared_fusion,
    collect_model,
    collect_optimizer,
    fsdp_gate,
    mesh_device,
    restore_scalars,
    world_record,
)
from ._topology import chunk_boxes
from ._train_state import check_train_state, missing_train_state, restore_train_state, train_state_of


@dataclass
class LoadResult:
    path: str
    header: dict
    scalars: dict
    missing: list[str]
    unexpected: list[str]
    fast_path_keys: list[str]
    stats: dict[str, float] = field(default_factory=dict)
    train_state: dict = field(default_factory=dict)
    restored: list[str] = field(default_factory=list)


def _read_into(
    fd: int, data_start: int, plan: KeyPlan, item: ReadItem, itemsize: int, carrier, dest: np.ndarray
) -> int:
    """Copy ``item.src`` of the chunk into ``item.dst`` of ``dest`` with the fewest contiguous reads; returns bytes read."""
    record = plan.entry.chunks[item.chunk]
    chunk = record.box.sizes
    src, dst = item.src, item.dst
    ndim = len(chunk)
    # dims >= k are read in full, so dim k-1 can span a contiguous run of rows.
    k = ndim
    while k > 0 and src.sizes[k - 1] == chunk[k - 1]:
        k -= 1
    strides = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        strides[d] = strides[d + 1] * chunk[d + 1]
    base = data_start + record.data_offset
    if k == 0:
        raw = pread_all(fd, record.nbytes, base)
        dest[dst.slices()] = np.frombuffer(raw, dtype=carrier).reshape(chunk)
        return record.nbytes
    run_rows = src.sizes[k - 1]
    inner = prod(chunk[k:])
    run_bytes = run_rows * inner * itemsize
    total = 0
    for prefix in np.ndindex(*src.sizes[: k - 1]):
        flat = (
            sum((src.offsets[d] + prefix[d]) * strides[d] for d in range(k - 1)) + src.offsets[k - 1] * strides[k - 1]
        )
        raw = pread_all(fd, run_bytes, base + flat * itemsize)
        run = np.frombuffer(raw, dtype=carrier).reshape((run_rows,) + tuple(chunk[k:]))
        index = tuple(dst.offsets[d] + prefix[d] for d in range(k - 1))
        index += (slice(dst.offsets[k - 1], dst.offsets[k - 1] + run_rows),)
        index += tuple(slice(dst.offsets[d], dst.offsets[d] + dst.sizes[d]) for d in range(k, ndim))
        dest[index] = run
        total += run_bytes
    return total


def _read_chunk(
    fd: int, data_start: int, plan: KeyPlan, items: list[ReadItem], itemsize: int, carrier, arrays: dict, path: str
) -> int:
    """The reads of one chunk: one ``pread`` of the whole chunk, sliced in memory, when the items need at least half
    of it (on a single host the destination shards together cover every chunk), else one read per contiguous run.
    A whole chunk is checked against the CRC32 that save recorded for it; a partial read cannot be, since the checksum
    covers the whole chunk. Returns bytes read."""
    record = plan.entry.chunks[items[0].chunk]
    needed = sum(it.src.volume for it in items) * itemsize
    if 2 * needed >= record.nbytes:
        raw = pread_all(fd, record.nbytes, data_start + record.data_offset)
        crc = zlib.crc32(raw)
        if crc != record.crc32:
            raise CheckpointError(
                f"{plan.key}: chunk {items[0].chunk} ({record.box}) in {path} fails its CRC32 check (recorded "
                f"{record.crc32:08x}, read {crc:08x}); the checkpoint is corrupt"
            )
        chunk = np.frombuffer(raw, dtype=carrier).reshape(record.box.sizes)
        for it in items:
            arrays[it.group][it.dst.slices()] = chunk[it.src.slices()]
        return record.nbytes
    return sum(_read_into(fd, data_start, plan, it, itemsize, carrier, arrays[it.group]) for it in items)


def _host_shard(array: np.ndarray, dtype_name: str):
    spec = ttnn.TensorSpec(
        ttnn.Shape(list(array.shape)), TTNN_DTYPES[dtype_name], ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG
    )
    return ttnn.host_tensor_from_bytes(array.reshape(-1).view(np.uint8), spec)


def _same_spec(a, b) -> bool:
    return (
        a.dtype == b.dtype
        and a.layout == b.layout
        and list(a.shape) == list(b.shape)
        and str(a.memory_config()) == str(b.memory_config())
    )


def _place(live: LiveTensor, arrays: dict[tuple, np.ndarray], device) -> None:
    """Assemble the destination shards under the live label, move them to the device and install them in place."""
    value = live.tensor.get_value(NATIVE)
    shards = {}
    hosts = {group: _host_shard(arr, live.info.dtype) for group, arr in arrays.items()}
    for chunk in chunk_boxes(live.info.label, live.info.local_shape):
        shards[ttnn.MeshCoordinate(list(chunk.mesh_coord))] = hosts[chunk.group]
    assembled = ttnn.from_local_host_shards(shards, device, value.tensor_topology())
    on_device = assembled.to(device)
    if value.layout == ttnn.TILE_LAYOUT:
        on_device = ttnn.tilize_with_zero_padding(on_device)
    if str(on_device.memory_config()) != str(value.memory_config()):
        on_device = ttnn.to_memory_config(on_device, value.memory_config())
    if not _same_spec(on_device, value):
        raise CheckpointError(
            f"{live.key}: the restored tensor ({on_device.dtype}, {on_device.layout}, {list(on_device.shape)}) does not match "
            f"the live tensor ({value.dtype}, {value.layout}, {list(value.shape)})"
        )
    live.tensor.set_value(on_device)
    check_label_unchanged(live, "after set_value")


class _DataFiles:
    """The data files a load reads, opened and checked while the load is planned, so that a missing or truncated file
    stops the load before it places a tensor; open until the load ends."""

    def __init__(self) -> None:
        self.fds: dict[int, int] = {}
        self.data_start: dict[int, int] = {}
        self.paths: dict[int, str] = {}

    def open(self, ckpt_dir: str, index: Index, plan: LoadPlan) -> None:
        for file_index in sorted({r.file for kp in plan.keys.values() for r in kp.reads}):
            record = index.files[file_index]
            path = self.paths[file_index] = os.path.join(ckpt_dir, record.path)
            if not os.path.isfile(path):
                raise CheckpointError(f"data file {path} listed in the index is missing")
            self.fds[file_index] = os.open(path, os.O_RDONLY)
            size = os.fstat(self.fds[file_index]).st_size
            if size != record.nbytes:
                raise IncompleteCheckpointError(
                    f"data file {path} is {size} bytes but the index records {record.nbytes}; the checkpoint is "
                    "truncated or corrupt"
                )
            _, self.data_start[file_index] = read_file_header(path)

    def close(self) -> None:
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


def _raise_partly_restored(err: Exception, at: str, placed: list[str], total: int) -> None:
    """Re-raise ``err``, as a ``CheckpointError`` that says the live state is partly restored once a tensor was
    placed."""
    if not placed:
        raise err
    raise CheckpointError(
        f"load failed at {at} ({type(err).__name__}: {err}) after {len(placed)} of {total} tensors were placed (the "
        f"first: {placed[0]}); the model and optimizer are partly restored: load again or discard them"
    ) from err


def _read_and_place(
    plan: LoadPlan, files: _DataFiles, by_key: dict[str, LiveTensor], threads: int | None
) -> tuple[int, int, float, float, list[str]]:
    """Read every planned key into host arrays and place it on the device; returns the bytes read, the number of reads,
    the seconds spent waiting for reads and placing, and the keys placed. A failure once a tensor was placed raises a
    ``CheckpointError`` that says the live state is partly restored."""
    fds, data_start, paths = files.fds, files.data_start, files.paths
    placed: list[str] = []
    device = mesh_device()
    bytes_read = 0
    reads = 0
    read_s = place_s = 0.0
    workers = 32 if threads is None else threads
    with ThreadPoolExecutor(max_workers=workers) as pool:

        def submit(kplan: KeyPlan):
            """Allocate a key's destination shards and submit its reads; returns (arrays, futures)."""
            info = kplan.info
            carrier = NUMPY_CARRIERS[info.dtype]
            arrays = {group: np.zeros(box.sizes, dtype=carrier) for group, box in kplan.boxes.items()}
            by_chunk: dict[int, list[ReadItem]] = {}
            for r in kplan.reads:
                by_chunk.setdefault(r.chunk, []).append(r)
            futures = [
                pool.submit(
                    _read_chunk,
                    fds[its[0].file],
                    data_start[its[0].file],
                    kplan,
                    its,
                    info.itemsize,
                    carrier,
                    arrays,
                    paths[its[0].file],
                )
                for its in by_chunk.values()
            ]
            return arrays, futures

        # The reads of the next key run while the current key is placed on the device, so file I/O and placement
        # overlap; at most two keys' host arrays are alive at a time.
        keys = list(plan.keys.items())
        pending = submit(keys[0][1]) if keys else None
        for position, (key, kplan) in enumerate(keys):
            try:
                arrays, futures = pending
                t_key = time.perf_counter()
                for future in futures:
                    bytes_read += future.result()
                t_read_done = time.perf_counter()
                pending = submit(keys[position + 1][1]) if position + 1 < len(keys) else None
                _place(by_key[key], arrays, device)
            except Exception as err:
                _raise_partly_restored(err, key, placed, len(keys))
            placed.append(key)
            read_s += t_read_done - t_key
            place_s += time.perf_counter() - t_read_done
            reads += len(futures)
    return bytes_read, reads, read_s, place_s, placed


def load(
    ckpt_dir: str,
    *,
    model=None,
    optimizer=None,
    scheduler=None,
    dataloader=None,
    rng: bool = False,
    strict: bool = True,
    threads: int | None = None,
    reshard_fsdp: bool = True,
) -> LoadResult:
    """Restore ``model`` and ``optimizer`` in place from the checkpoint directory ``ckpt_dir``.

    Works on any single-host mesh shape and parallelism: each destination shard reads only the bytes it needs. With
    ``strict`` (default), a live tensor without an entry, or an entry of a requested role (model, and optimizer state
    when ``optimizer`` is passed) without a live tensor, is an error; otherwise they are listed in the result.
    Optimizer scalars (steps, learning rates) are restored through ``set_state_dict``. ``scheduler`` and
    ``dataloader`` are restored through their ``set_state_dict`` from the state ``save`` stored, and with
    ``rng=True`` so is every random generator ``save`` captured (ttml's and ttml.init's for this world rank, python,
    numpy, torch when imported). A component asked for that the checkpoint lacks entirely is an error under ``strict``
    and listed in ``missing`` otherwise; within the RNG block, a generator or a world rank the checkpoint holds no state
    for (torch imported only now, a world with more hosts than the save) keeps its current state with a warning, and
    ``restored`` lists exactly what was restored. ``threads`` caps the reader threads (default 32; a read is one
    contiguous byte range of one file). Returns the header passed to ``save``, the train state stored and the names
    restored.

    Everything that can be checked without reading tensor data is checked before the first tensor is placed: keys,
    shapes, dtypes, layouts, param groups, the sizes of the data files and the stored RNG state. A failure after that
    (a read error, the device out of memory) leaves the model and optimizer partly restored; the ``CheckpointError``
    raised then says how many tensors were placed, and the model should be loaded again or discarded. A scheduler or
    dataloader without a ``set_state_dict`` is refused before anything is placed; one that refuses its state can only
    say so after, and raises a ``CheckpointError`` naming it, with the tensors, the optimizer scalars and the scheduler
    (when the dataloader refuses) already restored.
    """
    if model is None and optimizer is None:
        raise TypeError("load needs a model and/or an optimizer")
    if optimizer is not None and model is None:
        raise TypeError("optimizer state is keyed by the model's parameters; pass model= as well")
    check_threads(threads)
    t0 = time.perf_counter()
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not is_complete(ckpt_dir):
        raise IncompleteCheckpointError(
            f"{ckpt_dir} is not a complete checkpoint (staging directory or missing index.json)"
        )
    index: Index = read_index(ckpt_dir)
    header = read_checked_header(ckpt_dir)
    world_rank = world_record()["ranks"][0]["world_rank"]
    train_state = train_state_of(header, ckpt_dir)
    missing_state = missing_train_state(train_state, scheduler=scheduler, dataloader=dataloader, rng=rng)
    if strict and missing_state:
        raise MissingKeyError(f"the checkpoint holds no state for {missing_state}")
    check_train_state(train_state, scheduler=scheduler, dataloader=dataloader, rng=rng, world_rank=world_rank)

    fsdp_states = fsdp_gate(model, reshard_fsdp)
    # A sharded FSDP module's cached shards are its current values: drop them now, so every value the load replaces is
    # freed as it is replaced instead of at the end (the next forward rebuilds the cache).
    for state in fsdp_states:
        state._cached_shards.clear()
    live: list[LiveTensor] = collect_model(model)
    if optimizer is not None:
        state_live, _ = collect_optimizer(optimizer, live)
        live.extend(state_live)
    by_key = {lt.key: lt for lt in live}
    roles = {"param", "buffer"} | ({"optimizer_state"} if optimizer is not None else set())
    plan: LoadPlan = create_load_plan(index, [lt.info for lt in live], roles=roles)
    if strict and plan.missing:
        raise MissingKeyError(f"{len(plan.missing)} live tensor(s) have no entry in the checkpoint: {plan.missing[:5]}")
    if strict and plan.unexpected:
        raise UnexpectedKeyError(
            f"{len(plan.unexpected)} checkpoint entr(ies) have no live tensor: {plan.unexpected[:5]}"
        )
    for key, kplan in plan.keys.items():
        check_undeclared_fusion(by_key[key], kplan.entry)
    if optimizer is not None:
        check_param_groups(optimizer, index.scalars, strict)  # before anything is placed
    files = _DataFiles()
    try:
        files.open(ckpt_dir, index, plan)  # before anything is placed
        stats: dict[str, float] = {"plan_s": time.perf_counter() - t0}
        t1 = time.perf_counter()
        bytes_read, reads, read_s, place_s, placed = _read_and_place(plan, files, by_key, threads)
        stats["read_and_place_s"] = time.perf_counter() - t1
    finally:
        files.close()  # every descriptor opened, whatever ended the load
    stats["read_s"] = read_s
    stats["place_s"] = place_s
    stats["reads"] = float(reads)
    stats["bytes_read"] = float(bytes_read)

    applied: list[str] = []
    if optimizer is not None:
        try:
            applied = restore_scalars(optimizer, index.scalars)
        except Exception as err:
            _raise_partly_restored(err, "the optimizer scalars", placed, len(placed))
    # After the optimizer scalars: a scheduler writes its last LRs back into the optimizer and has the last word.
    restored = restore_train_state(
        train_state, scheduler=scheduler, dataloader=dataloader, rng=rng, world_rank=world_rank
    )
    stats["total_s"] = time.perf_counter() - t0
    return LoadResult(
        ckpt_dir,
        header["user"],
        {k: index.scalars[k] for k in applied} if optimizer is not None else dict(index.scalars),
        plan.missing + missing_state,
        plan.unexpected,
        plan.fast_path_keys,
        stats,
        train_state=train_state,
        restored=restored,
    )
