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
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from math import prod

import numpy as np
import ttnn

from ._coordinator import checked_payload, current_group
from ._errors import CheckpointError, IncompleteCheckpointError, MissingKeyError, UnexpectedKeyError
from ._index import Index, is_complete, read_checked_header, read_index
from ._planner import KeyPlan, LoadPlan, ReadItem, create_load_plan, unexpected_keys
from ._safetensors import pread_all
from ._state_dict import (
    NATIVE,
    NUMPY_CARRIERS,
    TTNN_DTYPES,
    LiveTensor,
    check_label_unchanged,
    check_member_union,
    check_param_groups,
    check_threads,
    check_undeclared_fusion,
    collect_model,
    collect_optimizer,
    live_group_members,
    local_coords,
    mesh_device,
    reshard_fsdp_modules,
    restore_scalars,
)
from ._topology import chunk_boxes, crc31
from ._train_state import (
    check_train_state,
    missing_train_state,
    missing_train_state_error,
    restore_train_state,
    train_state_of,
)


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
    """Copy ``item.src`` of the chunk into ``item.dst`` of ``dest`` with the fewest contiguous reads; returns bytes read.
    ``_read_chunk`` calls it only for items that need under half of the chunk, so some dim is read in part."""
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


def _pread_into(fd: int, buffer: memoryview, offset: int) -> None:
    """Fill ``buffer`` from ``fd`` at ``offset`` (``os.preadv`` may return short reads on some filesystems)."""
    done = 0
    while done < len(buffer):
        n = os.preadv(fd, [buffer[done:]], offset + done)
        if not n:
            raise EOFError(f"safetensors: read past the end of the file at offset {offset + done}")
        done += n


def _check_crc(plan: KeyPlan, chunk_index: int, crc: int, path: str) -> None:
    record = plan.entry.chunks[chunk_index]
    if crc != record.crc32:
        raise CheckpointError(
            f"{plan.key}: chunk {chunk_index} ({record.box}) in {path} fails its CRC32 check (recorded "
            f"{record.crc32:08x}, read {crc:08x}); the checkpoint is corrupt"
        )


# A rank reads a chunk with one pread, and checks it against the CRC32 save recorded for it, when its destination
# shards need at least this share of the chunk; otherwise it reads only the runs they need, which the checksum cannot
# cover. On one host the destination shards together cover every chunk, and of two ranks one needs at least half.
WHOLE_READ_SHARE = 0.5


def _needed_bytes(plan: KeyPlan) -> dict[int, int]:
    """Per chunk of the entry, the bytes this rank's destination shards need of it."""
    needed: dict[int, int] = {}
    for item in plan.reads:
        needed[item.chunk] = needed.get(item.chunk, 0) + item.src.volume * plan.info.itemsize
    return needed


def chunk_reads(plan: LoadPlan) -> tuple[list[list], list[list]]:
    """``(read in part, read whole although only part is needed)`` over this rank's plan, as ``[key, chunk]`` pairs.
    A chunk split over three or more loading ranks can be read in part by every one of them, and then no rank checks
    its CRC32; ``elect_checkers`` uses both lists to pick one reader of each such chunk to read it whole."""
    in_part: list[list] = []
    whole_of_part: list[list] = []
    for kplan in plan.keys.values():
        for chunk, needed in sorted(_needed_bytes(kplan).items()):
            nbytes = kplan.entry.chunks[chunk].nbytes
            if needed < WHOLE_READ_SHARE * nbytes:
                in_part.append([kplan.entry.key, chunk])
            elif needed < nbytes:
                whole_of_part.append([kplan.entry.key, chunk])
    return in_part, whole_of_part


def elect_checkers(index: Index, reports: list[tuple[list[list], list[list]]]) -> list[list[list]]:
    """Rank 0's pick, over every rank's ``chunk_reads``: for each chunk read only in part, by every rank that reads it,
    the reader that reads it whole and checks its CRC32, so no chunk the load reads goes unchecked. Largest chunks
    first, each to the reader with the fewest such bytes so far (ties to the lower rank). Returns per rank the
    ``[key, chunk]`` pairs it reads whole."""
    readers: dict[tuple[str, int], list[int]] = {}
    for rank, (in_part, _) in enumerate(reports):
        for key, chunk in in_part:
            readers.setdefault((key, int(chunk)), []).append(rank)
    for _, whole_of_part in reports:
        for key, chunk in whole_of_part:
            readers.pop((key, int(chunk)), None)  # another rank reads it whole and checks it

    def size(pair: tuple[str, int]) -> int:
        return index.entries[pair[0]].chunks[pair[1]].nbytes

    extra = [0] * len(reports)
    checkers: list[list[list]] = [[] for _ in reports]
    for pair in sorted(readers, key=lambda p: (-size(p), p)):
        rank = min(readers[pair], key=lambda r: (extra[r], r))
        checkers[rank].append(list(pair))
        extra[rank] += size(pair)
    return checkers


def _read_chunk(
    fd: int,
    data_start: int,
    plan: KeyPlan,
    items: list[ReadItem],
    itemsize: int,
    carrier,
    arrays: dict,
    path: str,
    whole: bool = False,
) -> int:
    """The reads of one chunk: one ``pread`` of the whole chunk, sliced in memory, when the items need at least
    ``WHOLE_READ_SHARE`` of it or ``whole`` says so (this rank was elected to check it), else one read per contiguous
    run. A chunk that is its one destination shard, as in a same-world load, is read into the shard's array itself. A
    whole chunk is checked against the CRC32 that save recorded for it; a partial read cannot be, since the checksum
    covers the whole chunk. Returns bytes read."""
    record = plan.entry.chunks[items[0].chunk]
    sizes = tuple(record.box.sizes)
    dest = arrays[items[0].group]
    if len(items) == 1 and tuple(items[0].src.sizes) == sizes and dest.shape == sizes and dest.flags.c_contiguous:
        _pread_into(fd, memoryview(dest).cast("B"), data_start + record.data_offset)
        _check_crc(plan, items[0].chunk, zlib.crc32(dest), path)
        return record.nbytes
    needed = sum(it.src.volume for it in items) * itemsize
    if whole or needed >= WHOLE_READ_SHARE * record.nbytes:
        raw = pread_all(fd, record.nbytes, data_start + record.data_offset)
        _check_crc(plan, items[0].chunk, zlib.crc32(raw), path)
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
    """Whether two tensors have the same ``TensorSpec``: dtype, layout, shape, tile, memory config and alignment."""
    return a.spec == b.spec


def _describe_spec(t) -> str:
    return (
        f"{t.dtype}, {t.layout}, shape {list(t.shape)} padded to {list(t.padded_shape)}, {t.spec.tile}, "
        f"{t.memory_config()}"
    )


def _place(live: LiveTensor, arrays: dict[tuple, np.ndarray], device, local: set[tuple[int, ...]]) -> None:
    """Assemble the destination shards this process holds under the live label, move them to the device and install
    them in place."""
    value = live.tensor.get_value(NATIVE)
    shards = {}
    hosts = {group: _host_shard(arr, live.info.dtype) for group, arr in arrays.items()}
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
            f"{live.key}: the restored tensor ({_describe_spec(on_device)}) does not match the live tensor "
            f"({_describe_spec(value)})"
        )
    live.tensor.set_value(on_device)
    check_label_unchanged(live, "after set_value")


class _DataFiles:
    """The data files a load reads, opened and checked while the load is planned, so that a missing or truncated file
    stops every rank before any rank places a tensor; open until the load ends."""

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
            # The data section starts after the 8-byte header length and the header, which load does not parse.
            (length,) = struct.unpack("<Q", pread_all(self.fds[file_index], 8, 0))
            if 8 + length > size:
                raise IncompleteCheckpointError(
                    f"data file {path} declares a {length}-byte header but is {size} bytes; the checkpoint is corrupt"
                )
            self.data_start[file_index] = 8 + length

    def close(self) -> None:
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()


def _partly_restored_advice(shared_mesh: bool) -> str:
    """What a load that stopped after placing tensors tells its caller to do. Ranks that share a mesh must make the
    same device allocations in the same order, and ranks that stopped at different tensors no longer have, so on such
    a mesh the job has to restart; elsewhere the model can be loaded again."""
    if shared_mesh:
        return (
            "the model and optimizer are partly restored, and the ranks sharing this mesh may now hold different "
            "device allocations: end the job and restart it instead of loading again"
        )
    return "the model and optimizer are partly restored: load again or discard them"


def _shares_mesh(local: set[tuple[int, ...]]) -> bool:
    """Whether other ranks hold devices of this process's mesh."""
    return len(local) < prod(list(mesh_device().shape))


def _raise_partly_restored(err: Exception, at: str, placed: list[str], total: int, shared_mesh: bool = False) -> None:
    """Re-raise ``err``, as a ``CheckpointError`` that says the live state is partly restored once a tensor was
    placed, and what to do about it (``_partly_restored_advice``)."""
    if not placed:
        raise err
    raise CheckpointError(
        f"load failed at {at} ({type(err).__name__}: {err}) after {len(placed)} of {total} tensors were placed (the "
        f"first: {placed[0]}); {_partly_restored_advice(shared_mesh)}"
    ) from err


def _read_and_place(
    plan: LoadPlan,
    files: _DataFiles,
    by_key: dict[str, LiveTensor],
    threads: int | None,
    local: set[tuple[int, ...]],
    placed: list[str],
    check: set[tuple[str, int]],
) -> tuple[int, int, float, float]:
    """Read every planned key into host arrays and place the shards of the coordinates in ``local`` on the device,
    appending each key placed to ``placed``; the ``(key, chunk)`` pairs in ``check`` are read whole and checked
    (``elect_checkers``). Returns the bytes read, the number of reads and the seconds spent waiting for reads and
    placing. A failure once a tensor was placed raises a ``CheckpointError`` that says the live state is partly
    restored."""
    fds, data_start, paths = files.fds, files.data_start, files.paths
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
                    (kplan.entry.key, its[0].chunk) in check,
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
                _place(by_key[key], arrays, device, local)
            except Exception as err:
                _raise_partly_restored(err, key, placed, len(keys), _shares_mesh(local))
            placed.append(key)
            read_s += t_read_done - t_key
            place_s += time.perf_counter() - t_read_done
            reads += len(futures)
    return bytes_read, reads, read_s, place_s


def load(
    ckpt_dir: str,
    *,
    model=None,
    optimizer=None,
    scheduler=None,
    dataloader=None,
    rng: bool | None = None,
    strict: bool = True,
    threads: int | None = None,
) -> LoadResult:
    """Restore ``model`` and ``optimizer`` in place from the checkpoint directory ``ckpt_dir``.

    Works on any mesh shape and parallelism: each destination shard reads only the bytes it needs. With ``strict``
    (default), a live tensor without an entry, or an entry of a requested role (model, and optimizer state when
    ``optimizer`` is passed) without a live tensor, is an error; otherwise they are listed in the result. Optimizer
    scalars (steps, learning rates) are restored through ``set_state_dict``. ``scheduler`` and ``dataloader`` are
    restored through their ``set_state_dict`` from the state ``save`` stored (a dataloader from this world rank's own
    when the saving ranks' states differed). With ``rng`` so is every random generator
    ``save`` captured for this world rank (ttml, ttml.init, python, numpy, torch when imported). By default (``None``)
    the generators are restored whenever a scheduler or a dataloader is, since a resume that restores those but keeps
    the current generators takes a different step; ``rng=False`` keeps them (a deliberate reseed), ``rng=True``
    restores them with only a model. A component asked for that the checkpoint lacks entirely is an error under
    ``strict`` and listed in ``missing`` otherwise; within the RNG block, a generator or a world rank the checkpoint
    holds no state for (torch imported only now, a world with more hosts than the save) keeps its current state with a
    warning, and ``restored`` lists exactly what was restored. ``threads`` caps the reader threads (default 32; a read
    is one contiguous byte range of one file). FSDP modules are resharded first. Returns the header passed to ``save``,
    the train state stored and the names restored.

    Everything that can be checked without reading tensor data is checked before the first tensor is placed: keys,
    shapes, dtypes, layouts, param groups, the sizes of the data files and the stored RNG state. A failure after that (a
    read error, the device out of memory) leaves the model and optimizer partly restored; the ``CheckpointError`` raised
    then says how many tensors were placed, and the model should be loaded again or discarded. On a mesh several ranks
    share, the ranks may then have placed different tensors, and such a mesh needs every rank to make the same device
    allocations in the same order: the error then says to end the job and restart it instead of loading again. A
    scheduler or dataloader without a ``set_state_dict`` is refused before anything is placed; one that refuses its
    state can only say so after, and raises a ``CheckpointError`` naming it, with the tensors, the optimizer scalars and
    the scheduler (when the dataloader refuses) already restored.

    Under ``tt-run`` with several ranks the call is collective (the package docstring has the rules): every rank
    restores the shards of the coordinates it holds, and ``strict`` judges unexpected entries over the union of the
    ranks' live tensors.
    """
    t0 = time.perf_counter()
    rng_by_default = rng is None
    if rng_by_default:  # a resume: the generators come back with the scheduler or the dataloader
        rng = scheduler is not None or dataloader is not None
    group = current_group()
    world_rank = group.world_rank

    def check_arguments() -> str:
        if model is None and optimizer is None:
            raise TypeError("load needs a model and/or an optimizer")
        if optimizer is not None and model is None:
            raise TypeError("optimizer state is keyed by the model's parameters; pass model= as well")
        check_threads(threads)
        return os.path.abspath(ckpt_dir)

    # In a phase of its own: a rank called with a bad argument (a None directory from a find_latest that saw no
    # checkpoint) then stops every rank instead of leaving them waiting in the first collective.
    ckpt_dir = group.phase("checking the arguments", check_arguments)
    # Ranks that resolved different directories (a stale NFS listing behind find_latest) or asked for different
    # things would restore different checkpoints, or wait for each other in different collectives.
    group.agree("the checkpoint directory", crc31(ckpt_dir), shown=ckpt_dir)
    group.agree(
        "strict and whether an optimizer is loaded",
        int(bool(strict)) + 2 * int(optimizer is not None),
        shown=f"strict={bool(strict)}, optimizer={'passed' if optimizer is not None else 'none'}",
    )

    def open_and_collect():
        if not is_complete(ckpt_dir):
            raise IncompleteCheckpointError(
                f"{ckpt_dir} is not a complete checkpoint (staging directory or missing index.json)"
            )
        index: Index = read_index(ckpt_dir)
        header = read_checked_header(ckpt_dir)
        train_state = train_state_of(header, ckpt_dir)
        missing_state = missing_train_state(
            train_state, scheduler=scheduler, dataloader=dataloader, rng=rng, world_rank=world_rank
        )
        if strict and missing_state:
            raise missing_train_state_error(missing_state, rng_by_default=rng_by_default)
        check_train_state(train_state, scheduler=scheduler, dataloader=dataloader, rng=rng, world_rank=world_rank)
        fsdp_states = reshard_fsdp_modules(model)
        # A sharded FSDP module's cached shards are its current values: drop them now, so every value the load
        # replaces is freed as it is replaced instead of at the end (the next forward rebuilds the cache).
        for state in fsdp_states:
            state._cached_shards.clear()
        live: list[LiveTensor] = collect_model(model)
        if optimizer is not None:
            state_live, _ = collect_optimizer(optimizer, live)
            live.extend(state_live)
        return index, header, train_state, missing_state, live, set(local_coords())

    # Even the local work runs inside a phase, so a rank that fails here cannot leave the others waiting in the next
    # collective; on a single process the phase just runs it.
    index, header, train_state, missing_state, live, local = group.phase(
        "reading the index and collecting the live tensors", open_and_collect
    )
    # Hosts holding their own copies at one path (a relative path into per-host clones, say) agree on the directory
    # string but read different checkpoints: the ranks also agree on the index's id and creation time.
    group.agree(
        "the checkpoint (its id and creation time)",
        crc31(f"{index.checkpoint_id}\n{index.created}"),
        shown=f"{index.checkpoint_id}, created {index.created}",
    )
    by_key = {lt.key: lt for lt in live}
    roles = {"param", "buffer"} | ({"optimizer_state"} if optimizer is not None else set())

    def plan_locally() -> LoadPlan:
        plan = create_load_plan(index, [lt.info for lt in live], roles=roles, local_coords=local)
        if strict and plan.missing:
            raise MissingKeyError(
                f"{len(plan.missing)} live tensor(s) have no entry in the checkpoint: {plan.missing[:5]}"
            )
        for key, kplan in plan.keys.items():
            check_undeclared_fusion(by_key[key], kplan.entry)
        if optimizer is not None:
            check_param_groups(optimizer, index.scalars, strict)  # before anything is placed
        return plan

    files = _DataFiles()

    def plan_and_report() -> tuple[LoadPlan, bytes]:
        plan = plan_locally()
        files.open(ckpt_dir, index, plan)  # before any rank places anything
        in_part, whole_of_part = chunk_reads(plan)
        report = {
            "keys": sorted({kp.entry.key for kp in plan.keys.values()}),
            "members": live_group_members(optimizer) if optimizer is not None else {},
            "in_part": in_part,
            "whole_of_part": whole_of_part,
        }
        return plan, checked_payload(json.dumps(report).encode(), "this rank's planned keys")

    try:
        plan, used_bytes = group.phase("planning the load", plan_and_report)
        gathered = group.gather(used_bytes)

        def judge() -> bytes | None:
            if not group.is_coordinator:
                return None
            reports = [json.loads(part.decode("utf-8")) for part in gathered]
            union_used = {key for report in reports for key in report["keys"]}
            if strict and optimizer is not None:
                check_member_union(index.scalars, [report["members"] for report in reports])
            checkers = elect_checkers(index, [(report["in_part"], report["whole_of_part"]) for report in reports])
            verdict = {"unexpected": unexpected_keys(index, union_used, roles), "checkers": checkers}
            return checked_payload(json.dumps(verdict).encode(), "the unexpected keys and the elected checkers")

        verdict = json.loads(group.broadcast(group.phase("judging the unexpected entries", judge)))
        plan.unexpected = verdict["unexpected"]
        check = {(key, int(chunk)) for key, chunk in verdict["checkers"][group.rank]}
        if strict and plan.unexpected:
            raise UnexpectedKeyError(
                f"{len(plan.unexpected)} checkpoint entr(ies) have no live tensor: {plan.unexpected[:5]}"
            )
        stats: dict[str, float] = {"plan_s": time.perf_counter() - t0}

        t1 = time.perf_counter()
        placed: list[str] = []
        try:
            bytes_read, reads, read_s, place_s = group.phase(
                "reading and placing the shards",
                lambda: _read_and_place(plan, files, by_key, threads, local, placed, check),
            )
        except CheckpointError as err:
            if placed and err.__cause__ is None:
                # Another rank's failure stopped the load after this rank placed tensors; a failure of this rank's
                # own (the cause) already says how many it placed.
                raise type(err)(
                    f"{err}; rank {group.rank} had placed {len(placed)} of {len(plan.keys)} tensors, so "
                    f"{_partly_restored_advice(_shares_mesh(local))}"
                ) from err
            raise
        stats["read_and_place_s"] = time.perf_counter() - t1
        stats["read_s"] = read_s
        stats["place_s"] = place_s
        stats["reads"] = float(reads)
        stats["bytes_read"] = float(bytes_read)

        def restore_rest() -> tuple[list[str], list[str]]:
            applied: list[str] = []
            if optimizer is not None:
                try:
                    applied = restore_scalars(optimizer, index.scalars)
                except Exception as err:
                    _raise_partly_restored(err, "the optimizer scalars", placed, len(placed))
            # After the optimizer scalars: a scheduler writes its last LRs back into the optimizer, having the last
            # word.
            restored = restore_train_state(
                train_state, scheduler=scheduler, dataloader=dataloader, rng=rng, world_rank=world_rank
            )
            return applied, restored

        applied, restored = group.phase("restoring the scalars and the train state", restore_rest)
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
    finally:
        files.close()  # every descriptor opened while planning, whichever phase ended the load
