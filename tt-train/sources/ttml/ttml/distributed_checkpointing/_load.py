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

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import prod

import numpy as np
import ttnn

from ._coordinator import checked_payload, current_group, name_hash
from ._errors import CheckpointError, IncompleteCheckpoint, MissingKeyError, UnexpectedKeyError
from ._index import Index, is_complete, read_index
from ._index import read_header as _read_header_json
from ._planner import KeyPlan, LoadPlan, ReadItem, create_load_plan, unexpected_keys
from ._safetensors import pread_all
from ._safetensors import read_header as read_safetensors_header
from ._state_dict import (
    NATIVE,
    NUMPY_CARRIERS,
    TTNN_DTYPES,
    LiveTensor,
    check_label_unchanged,
    check_param_groups,
    collect_model,
    collect_optimizer,
    fsdp_gate,
    local_coords,
    mesh_device,
    restore_scalars,
)
from ._train_state import TRAIN_STATE_KEY, check_train_state, missing_train_state, restore_train_state


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
    fd: int, data_start: int, plan: KeyPlan, items: list[ReadItem], itemsize: int, carrier, arrays: dict
) -> int:
    """The reads of one chunk: one ``pread`` of the whole chunk, sliced in memory, when the items need at least half
    of it (on a single host the destination shards together cover every chunk), else one read per contiguous run.
    Returns bytes read."""
    record = plan.entry.chunks[items[0].chunk]
    needed = sum(it.src.volume for it in items) * itemsize
    if 2 * needed >= record.nbytes:
        raw = pread_all(fd, record.nbytes, data_start + record.data_offset)
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


def _place(
    live: LiveTensor, plan: KeyPlan, arrays: dict[tuple, np.ndarray], device, local: set[tuple[int, ...]]
) -> None:
    """Assemble the destination shards this process holds under the live label, move them to the device and install
    them in place."""
    value = live.tensor.get_value(NATIVE)
    shards = {}
    hosts = {group: _host_shard(arr, live.info.dtype) for group, arr in arrays.items()}
    from ._topology import chunk_boxes

    for chunk in chunk_boxes(live.info.label, live.info.local_shape):
        if chunk.mesh_coord not in local:
            continue
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

    Works on any mesh shape and parallelism: each destination shard reads only the bytes it needs. With ``strict``
    (default), a live tensor without an entry, or an entry of a requested role (model, and optimizer state when
    ``optimizer`` is passed) without a live tensor, is an error; otherwise they are listed in the result. Optimizer
    scalars (steps, learning rates) are restored through ``set_state_dict``. ``scheduler`` and ``dataloader`` are
    restored through their ``set_state_dict`` from the state ``save`` stored, and with ``rng=True`` so is every random
    generator ``save`` captured for this world rank (ttml, ttml.init, python, numpy, torch when imported). A component
    asked for that the checkpoint lacks entirely is an error under ``strict`` and listed in ``missing`` otherwise;
    within the RNG block, a generator or a world rank the checkpoint holds no state for (torch imported only now, a
    world with more hosts than the save) keeps its current state with a warning, and ``restored`` lists exactly what was
    restored. Everything that can be checked is checked before anything is placed: the optimizer's param groups, the
    data files' sizes and the stored RNG state; a scheduler or dataloader that refuses its state raises a
    ``CheckpointError`` naming it. ``threads`` caps the reader threads (default 32; a read is one contiguous byte
    range of one file). Returns the header passed to ``save``, the train state stored and the names restored.

    Under ``tt-run`` with several ranks the call is collective: every rank calls it with the same ``ckpt_dir``,
    ``strict`` and optimizer-or-not (checked before anything is read), restores the shards of the coordinates it
    holds, and ``strict`` judges unexpected entries over the union of the ranks' live tensors. A failure on any rank
    makes every rank raise the same error class, and no point-to-point MPI traffic between the ranks may be
    outstanding when load is called. A single process is the group of one and takes the same path.
    """
    if model is None and optimizer is None:
        raise CheckpointError("load needs a model and/or an optimizer")
    if optimizer is not None and model is None:
        raise CheckpointError("optimizer state is keyed by the model's parameters; pass model= as well")
    t0 = time.perf_counter()
    ckpt_dir = os.path.abspath(ckpt_dir)
    group = current_group()
    world_rank = group.world_rank
    # Ranks that resolved different directories (a stale NFS listing behind find_latest) or asked for different
    # things would restore different checkpoints, or wait for each other in different collectives.
    group.agree("the checkpoint directory", name_hash(ckpt_dir))
    group.agree("strict and whether an optimizer is loaded", int(bool(strict)) + 2 * int(optimizer is not None))

    def open_and_collect():
        if not is_complete(ckpt_dir):
            raise IncompleteCheckpoint(
                f"{ckpt_dir} is not a complete checkpoint (staging directory or missing index.json)"
            )
        index: Index = read_index(ckpt_dir)
        header = _read_header_json(ckpt_dir)
        train_state = header.get(TRAIN_STATE_KEY, {})
        missing_state = missing_train_state(train_state, scheduler=scheduler, dataloader=dataloader, rng=rng)
        if strict and missing_state:
            raise MissingKeyError(f"the checkpoint holds no state for {missing_state}")
        check_train_state(train_state, rng=rng, world_rank=world_rank)
        fsdp_states = fsdp_gate(model, reshard_fsdp)
        # A sharded FSDP module's cached shards are its current values: drop them now, so every value the load
        # replaces is freed as it is replaced instead of at the end (the next forward rebuilds the cache).
        for state in fsdp_states:
            state._cached_shards.clear()
        live: list[LiveTensor] = collect_model(model)
        if optimizer is not None:
            state_live, _ = collect_optimizer(optimizer, live)
            live.extend(state_live)
        return index, header, train_state, missing_state, fsdp_states, live, set(local_coords())

    # Even the local work runs inside a phase, so a rank that fails here cannot leave the others waiting in the next
    # collective; on a single process the phase just runs it.
    index, header, train_state, missing_state, fsdp_states, live, local = group.phase(
        "reading the index and collecting the live tensors", open_and_collect
    )
    by_key = {lt.key: lt for lt in live}
    roles = {"param", "buffer"} | ({"optimizer_state"} if optimizer is not None else set())

    def plan_locally() -> LoadPlan:
        plan = create_load_plan(index, [lt.info for lt in live], strict=False, roles=roles, local_coords=local)
        if strict and plan.missing:
            raise MissingKeyError(
                f"{len(plan.missing)} live tensor(s) have no entry in the checkpoint: {plan.missing[:5]}"
            )
        if optimizer is not None:
            check_param_groups(optimizer, index.scalars)  # before anything is placed
        return plan

    def plan_and_report() -> tuple[LoadPlan, bytes]:
        plan = plan_locally()
        used = sorted({kp.entry.key for kp in plan.keys.values()})
        return plan, checked_payload(json.dumps(used).encode(), "this rank's planned keys")

    plan, used_bytes = group.phase("planning the load", plan_and_report)
    gathered = group.gather(used_bytes)

    def judge() -> bytes | None:
        if not group.is_coordinator:
            return None
        union_used = {key for part in gathered for key in json.loads(part.decode("utf-8"))}
        return checked_payload(json.dumps(unexpected_keys(index, union_used, roles)).encode(), "the unexpected keys")

    plan.unexpected = json.loads(group.broadcast(group.phase("judging the unexpected entries", judge)))
    if strict and plan.unexpected:
        raise UnexpectedKeyError(
            f"{len(plan.unexpected)} checkpoint entr(ies) have no live tensor: {plan.unexpected[:5]}"
        )
    stats: dict[str, float] = {"plan_s": time.perf_counter() - t0}

    def read_and_place() -> tuple[int, int, float, float]:
        fds: dict[int, int] = {}
        data_start: dict[int, int] = {}
        device = mesh_device()
        bytes_read = 0
        reads = 0
        read_s = place_s = 0.0
        try:
            needed = {r.file for kp in plan.keys.values() for r in kp.reads}
            for file_index in sorted(needed):
                record = index.files[file_index]
                path = os.path.join(ckpt_dir, record.path)
                if not os.path.isfile(path):
                    raise CheckpointError(f"data file {path} listed in the index is missing")
                fds[file_index] = os.open(path, os.O_RDONLY)
                size = os.fstat(fds[file_index]).st_size
                if size != record.nbytes:
                    raise IncompleteCheckpoint(
                        f"data file {path} is {size} bytes but the index records {record.nbytes}; the checkpoint is "
                        "truncated or corrupt"
                    )
                _, data_start[file_index] = read_safetensors_header(path)
            workers = 32 if threads is None else threads
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:

                def submit(kplan: KeyPlan):
                    """Allocate a key's destination shards and submit its reads; returns (arrays, futures)."""
                    info = kplan.info
                    carrier = NUMPY_CARRIERS[info.dtype]
                    arrays = {group_: np.zeros(box.sizes, dtype=carrier) for group_, box in kplan.boxes.items()}
                    for fill in kplan.fills:
                        if fill.fill != "zeros":
                            raise CheckpointError(f"{kplan.key}: unsupported padding fill {fill.fill!r}")
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
                        )
                        for its in by_chunk.values()
                    ]
                    return arrays, futures

                # The reads of the next key run while the current key is placed on the device, so file I/O and
                # placement overlap; at most two keys' host arrays are alive at a time.
                keys = list(plan.keys.items())
                pending = submit(keys[0][1]) if keys else None
                for position, (key, kplan) in enumerate(keys):
                    arrays, futures = pending
                    t_key = time.perf_counter()
                    for future in futures:
                        bytes_read += future.result()
                    t_read_done = time.perf_counter()
                    pending = submit(keys[position + 1][1]) if position + 1 < len(keys) else None
                    _place(by_key[key], kplan, arrays, device, local)
                    read_s += t_read_done - t_key
                    place_s += time.perf_counter() - t_read_done
                    reads += len(futures)
        finally:
            for fd in fds.values():
                os.close(fd)
        return bytes_read, reads, read_s, place_s

    t1 = time.perf_counter()
    bytes_read, reads, read_s, place_s = group.phase("reading and placing the shards", read_and_place)
    stats["read_and_place_s"] = time.perf_counter() - t1
    stats["read_s"] = read_s
    stats["place_s"] = place_s
    stats["reads"] = float(reads)
    stats["bytes_read"] = float(bytes_read)

    def restore_rest() -> tuple[list[str], list[str]]:
        applied: list[str] = []
        if optimizer is not None:
            applied = restore_scalars(optimizer, index.scalars)
        # After the optimizer scalars: a scheduler writes its last LRs back into the optimizer and has the last word.
        restored = restore_train_state(
            train_state, scheduler=scheduler, dataloader=dataloader, rng=rng, world_rank=world_rank
        )
        return applied, restored

    applied, restored = group.phase("restoring the scalars and the train state", restore_rest)
    stats["total_s"] = time.perf_counter() - t0
    return LoadResult(
        ckpt_dir,
        header.get("user", {}),
        {k: index.scalars[k] for k in applied} if optimizer is not None else dict(index.scalars),
        plan.missing + missing_state,
        plan.unexpected,
        plan.fast_path_keys,
        stats,
        train_state=train_state,
        restored=restored,
    )
