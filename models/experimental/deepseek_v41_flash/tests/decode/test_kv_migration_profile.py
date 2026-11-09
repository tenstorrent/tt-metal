# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Profile V4.1's host KV-cache migration -- ``export_session`` (chip -> host) and ``import_session``
(host -> chip) -- in both directions over sequence lengths in powers of two, reporting each one's
effective bandwidth and how far the pipeline stages overlap.

The host tier of ``demo/server.py`` parks a conversation's whole decode state in host RAM when its
device slot is taken by another request, and copies it back when the conversation continues. Both
directions move the same tensors: every layer's ring slot, each ratio>1 source's compressor window,
and each source's ``pos // ratio`` compressed entries with their index keys pulled out of the paged
pools. A session's Engram history is already host-side and moves no bytes.

The stages are measured one at a time, from the host's side. ``export_session`` walks
``submeshes_io`` in order and readbacks only through ``_replica_rows`` (``ttnn.to_torch`` on the
first chip of the stage's ``1 x TP`` submesh, which is blocking), so this test wraps that funnel and
times each stage's transfer. ``import_session`` is wrapped the same way through the two primitives it
uploads with, ``ttnn.from_torch(device=...)`` and ``ttnn.copy_host_to_device_tensor``. Each direction
reports, per stage, the bytes it moved with and the seconds it took, and then the overlap
``sum(stage seconds) / union(stage seconds)``: ~1.0 means the stages are serialized (one stage's DMA
starts only once the previous one's data is home), while a value near the stage count means they run
concurrently.

Nothing is decoded: a position is reached by allocating the blocks a step there would write
(``ensure_session_capacity``) and placing the session at it, so the transfer sizes are the real ones
at a fraction of the cost. The round trip is checked bit-exact at the largest length -- export ->
import -> export must reproduce every ring, window and compressed entry.

Run (32-chip Galaxy, ttnn venv, from the repo root)::

    pytest models/experimental/deepseek_v41_flash/tests/decode/test_kv_migration_profile.py -s

``DEEPSEEK_V41_KV_MIGRATION_SEQ_LENS`` (default ``1024,2048,...,131072``) sets the sweep.
"""

from __future__ import annotations

import contextlib
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
import models.experimental.deepseek_v41_flash.tt.model as model_module
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _PAGE_BLOCK_SIZE,
    _build_rope,
    _checkpoint_available,
    _construct_model,
    _traced_max_seq,
)
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.decode.dspark import host_to_device
from models.experimental.deepseek_v41_flash.tt.model import _WIN_SLOT_ROWS
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

# Powers of two, all multiples of the sliding window so a ring slot is whole.
_SEQ_LENS = sorted(
    int(n)
    for n in os.environ.get("DEEPSEEK_V41_KV_MIGRATION_SEQ_LENS", "1024,2048,4096,8192,16384,32768,65536,131072").split(
        ","
    )
)
# Below this the stages are not overlapping: they run one after another.
_SERIAL_OVERLAP = 1.1
# Two directions' byte counts should agree this closely; the import also rewrites the (tiny) page tables.
_DIRECTION_TOLERANCE = 0.05

_ITEMSIZE = {
    ttnn.bfloat16: 2,
    ttnn.float32: 4,
    ttnn.int32: 4,
    ttnn.uint32: 4,
    ttnn.uint16: 2,
    ttnn.uint8: 1,
    ttnn.int8: 1,
}


def _device_nbytes(tensor: ttnn.Tensor) -> int:
    """Bytes a device ``tensor`` moves in a copy: its padded shape (what a TILE transfer sends) times its
    element size."""
    shape = tensor.padded_shape if hasattr(tensor, "padded_shape") else tensor.shape
    return math.prod(int(d) for d in shape) * _ITEMSIZE.get(tensor.dtype, 2)


def _stage_of_device(model, device) -> int | None:
    """The pipeline stage a ``1 x TP`` submesh is, or ``None`` if it is not one of the model's."""
    if device is None:
        return None
    ids = tuple(device.get_device_ids())
    return {tuple(sm["device"].get_device_ids()): sm["index"] for sm in model.submeshes_io}.get(ids)


def _paged_sources(model) -> dict[int, dict]:
    """Per KV source: the stages holding its compressed pools, and the bytes one stage's slice takes.

    This is where the two directions differ. ``export_session`` reads a group's entries from **one**
    stage (``if src is None or src in entries`` skips the rest), while ``import_session`` writes them
    to **every** stage holding the group (its ``written`` set is per stage). KV source 20 is ratio 1
    and lives on stages 4-7, so its entries cross the bus once on the way out and four times on the
    way back -- real, and worth seeing in the numbers rather than hidden.
    """
    sources: dict[int, dict] = {}
    for sm in model.submeshes_io:
        for li in sm["layers"]:
            src = model._kv_source_of.get(li)
            scache = sm["scaches"].get(li)
            if src is None or scache is None or scache.comp_kv is None:
                continue
            entry = sources.setdefault(src, {"stages": set(), "rows": 0, "width": 0})
            entry["stages"].add(sm["index"])
            entry["rows"] = scache.comp_kv.shape[2]
            entry["width"] = scache.comp_kv.shape[3] + scache.idx_key_cache.shape[3]
    return sources


def _expected_import_bytes(model, sid: int, export_nbytes: int) -> int:
    """``export_nbytes`` with each source's entries counted once per stage holding them instead of once.

    Rings and compressor windows are already per-stage on both sides, so only the paged part changes.
    """
    sources = _paged_sources(model)
    if not sources:
        return export_nbytes
    pos = model._session_pos[sid]
    one_replica = replicated = 0
    for src, entry in sources.items():
        ratio = int(model.config.compress_ratios[src])
        block = entry["rows"]
        n = -(-(pos // ratio) // block) * block
        nbytes = n * entry["width"] * 2  # bf16
        one_replica += nbytes
        replicated += nbytes * len(entry["stages"])
    return export_nbytes - one_replica + replicated


def _stage_of_tensor(model, tensor) -> int | None:
    try:
        return _stage_of_device(model, tensor.device())
    except Exception:  # noqa: BLE001 - a host tensor has no device; attribution is all this is for
        return None


class _Transfers:
    """Every host <-> device byte movement seen inside a timed window, as ``(direction, stage, nbytes,
    start, end)`` samples."""

    def __init__(self) -> None:
        self.samples: list[tuple[str, int | None, int, float, float]] = []

    def record(self, direction: str, stage: int | None, nbytes: int, start: float, end: float) -> None:
        self.samples.append((direction, stage, nbytes, start, end))

    def clear(self) -> None:
        self.samples.clear()

    def summary(self, direction: str) -> dict:
        """Bytes and seconds of ``direction`` in total and per stage. ``overlap`` is the summed stage
        time over the wall time the stages actually occupied: 1.0 if they never ran together."""
        picked = [s for s in self.samples if s[0] == direction]
        stages: dict[int | None, dict] = {}
        for _, stage, nbytes, start, end in picked:
            entry = stages.setdefault(stage, {"nbytes": 0, "seconds": 0.0, "calls": 0})
            entry["nbytes"] += nbytes
            entry["seconds"] += end - start
            entry["calls"] += 1
        summed = sum(e["seconds"] for e in stages.values())
        return {
            "nbytes": sum(s[2] for s in picked),
            "calls": len(picked),
            "summed": summed,
            "union": _union_seconds([(s[3], s[4]) for s in picked]),
            "overlap": summed / _union_seconds([(s[3], s[4]) for s in picked]) if picked else 0.0,
            "slowest": max(picked, key=lambda s: s[4] - s[3]) if picked else None,
            "stages": stages,
        }


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    """Wall time the (possibly overlapping) ``intervals`` cover."""
    if not intervals:
        return 0.0
    total, current_start, current_end = 0.0, *min(intervals)
    for start, end in sorted(intervals)[1:]:
        if start > current_end:
            total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    return total + current_end - current_start


@contextlib.contextmanager
def _instrument(monkeypatch, model, transfers: _Transfers):
    """Route every readback and upload of the migration through ``transfers``.

    Both funnels are patched on the modules the code under test resolves them on, so nothing in
    ``tt/model.py`` changes: ``_replica_rows`` in the model module (the one readback primitive
    ``export_session`` has) and ``ttnn.from_torch`` / ``ttnn.copy_host_to_device_tensor`` (the two
    upload primitives ``import_session`` has).
    """
    replica_rows = model_module._replica_rows

    def timed_replica_rows(tensor):
        stage, nbytes = _stage_of_tensor(model, tensor), _device_nbytes(tensor)
        start = time.perf_counter()
        try:
            return replica_rows(tensor)
        finally:
            transfers.record("export", stage, nbytes, start, time.perf_counter())

    monkeypatch.setattr(model_module, "_replica_rows", timed_replica_rows)

    from_torch = ttnn.from_torch

    def timed_from_torch(value, *args, **kwargs):
        device = kwargs.get("device")
        if device is None:  # a host-side tensor: no transfer to time
            return from_torch(value, *args, **kwargs)
        stage = _stage_of_device(model, device)
        start = time.perf_counter()
        try:
            result = from_torch(value, *args, **kwargs)
        finally:
            # The device tensor's own size, not the host buffer's: the import hands ``from_torch`` an
            # fp32 view of bf16 data, and only the bf16 bytes cross the bus.
            transfers.record("import", stage, _device_nbytes(result), start, time.perf_counter())
        return result

    monkeypatch.setattr(ttnn, "from_torch", timed_from_torch)

    copy_host_to_device_tensor = ttnn.copy_host_to_device_tensor

    def timed_copy_host_to_device_tensor(host_tensor, device_tensor, *args, **kwargs):
        stage, nbytes = _stage_of_tensor(model, device_tensor), _device_nbytes(device_tensor)
        start = time.perf_counter()
        try:
            return copy_host_to_device_tensor(host_tensor, device_tensor, *args, **kwargs)
        finally:
            transfers.record("import", stage, nbytes, start, time.perf_counter())

    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", timed_copy_host_to_device_tensor)
    try:
        yield
    finally:
        transfers.clear()


def _fill(model, sid: int, generator: torch.Generator) -> None:
    """Put recognisable bytes in every tensor a snapshot carries, so a round trip can be checked.

    A session's blocks and rings are allocated, not decoded, so they hold whatever the pool had. The
    writes mirror :meth:`import_session` (the same ``host_to_device`` / ``fill_cache`` /
    ``_commit_paged_entries`` calls), and happen before the timed window.
    """
    pos, slot, w = model._session_pos[sid], model._slot_of[sid], model.sliding_window

    def noise(shape) -> torch.Tensor:
        return torch.rand(shape, generator=generator, dtype=torch.float32) * 2 - 1

    for sm in model.submeshes_io:
        device = sm["device"]
        mapper = ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None

        def write_rows(dst: ttnn.Tensor, rows: torch.Tensor, start: int) -> None:
            if rows.shape[0] == dst.shape[2]:
                host_to_device(rows.reshape(tuple(dst.shape)), dst)
                return
            src_rows = ttnn.from_torch(
                rows.reshape(1, 1, *rows.shape),
                dtype=dst.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            ttnn.fill_cache(dst, src_rows, 0, update_idx=start)
            ttnn.deallocate(src_rows)

        written = set()
        for li in sm["layers"]:
            scache = sm["scaches"][li]
            write_rows(scache.kv, noise((w, scache.kv.shape[3])), slot * w)
            src = model._kv_source_of.get(li)
            if li == src and scache.win_kv is not None:
                win_rows = _WIN_SLOT_ROWS if model._slotted else scache.win_kv.shape[2]
                for dst in (scache.win_kv, scache.win_gate):
                    write_rows(dst, noise((win_rows, dst.shape[3])), slot * win_rows)
            if src is None or src in written or scache.comp_kv is None:
                continue
            written.add(src)
            ratio = int(model.config.compress_ratios[src])
            n = pos // ratio
            if not n:
                continue
            block = scache.comp_kv.shape[2]
            blocks = -(-n // block)
            kv, keys = (
                ttnn.from_torch(
                    noise((1, 1, blocks * block, shape[3])),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mapper,
                )
                for shape in (scache.comp_kv.shape, scache.idx_key_cache.shape)
            )
            model._commit_paged_entries(scache, model._paged_ratios[ratio].layer_type, sid, kv, keys)
            ttnn.deallocate(kv)
            ttnn.deallocate(keys)
        ttnn.synchronize_device(device)


def _measure(model, sid: int, length: int, transfers: _Transfers, fill: bool = False) -> dict:
    """Migrate a session of ``length`` tokens to host RAM and back, timing each direction.

    ``fill`` also puts recognisable bytes in the caches first and reads the state back out afterwards,
    for the round-trip check; the sweep does neither.
    """
    model.reset_session(sid)
    model.ensure_session_capacity(length - 1, sid)
    # The server's slot sits at the position its last decode reached; the blocks above were allocated
    # rather than decoded, so the position is placed here instead.
    model._session_pos[sid] = length
    if fill:
        _fill(model, sid, torch.Generator().manual_seed(length))

    transfers.clear()
    start = time.perf_counter()
    snapshot = model.export_session(sid)
    export_seconds = time.perf_counter() - start
    export = transfers.summary("export")

    model.reset_session(sid)
    transfers.clear()
    start = time.perf_counter()
    model.import_session(sid, snapshot)
    import_seconds = time.perf_counter() - start
    restore = transfers.summary("import")

    # Only a round trip needs the state read back out of the device it was just written to.
    again = model.export_session(sid) if fill else None

    return {
        "length": length,
        "snapshot": snapshot,
        "again": again,
        "export": export,
        "export_seconds": export_seconds,
        "import": restore,
        "import_seconds": import_seconds,
    }


def _log_direction(measured: dict, direction: str) -> None:
    """One line for the direction, then one per stage that moved bytes."""
    summary = measured[direction]
    seconds = measured[f"{direction}_seconds"]
    nbytes = summary["nbytes"]
    populated = sum(1 for e in summary["stages"].values() if e["nbytes"])
    logger.info(
        f"  {direction:6s} {nbytes / 1e6:8.1f} MB in {seconds * 1e3:7.2f} ms = "
        f"{nbytes / 1e9 / max(seconds, 1e-9):6.2f} GB/s ({nbytes / 2**20 / max(seconds, 1e-9):6.1f} MiB/s)"
        f" | {populated} stage(s), sum {summary['summed'] * 1e3:7.2f} ms, "
        f"wall {summary['union'] * 1e3:7.2f} ms, overlap {summary['overlap']:4.2f}x"
    )
    for stage in sorted(summary["stages"], key=lambda s: (s is None, s)):
        entry = summary["stages"][stage]
        if not entry["nbytes"]:
            continue
        logger.info(
            f"    stage {stage if stage is not None else '?'} {entry['nbytes'] / 1e6:8.1f} MB in "
            f"{entry['seconds'] * 1e3:7.2f} ms = {entry['nbytes'] / 1e9 / max(entry['seconds'], 1e-9):6.2f} GB/s "
            f"({entry['calls']} call(s))"
        )
    # One stage far slower than its peers means a per-call stall, not a bandwidth limit: the slowest
    # call of the slowest stage is named so the next step is obvious.
    slowest = summary["slowest"]
    stages_with_bytes = [e for e in summary["stages"].values() if e["nbytes"]]
    if slowest is not None and len(stages_with_bytes) > 1:
        median = sorted(e["seconds"] for e in stages_with_bytes)[len(stages_with_bytes) // 2]
        if slowest[4] - slowest[3] > 3 * median:
            # A sample is (direction, stage, nbytes, start, end).
            _, slow_stage, nbytes, start, end = slowest
            logger.warning(
                f"  {direction}: one call on stage {slow_stage} took {(end - start) * 1e3:.1f} ms for "
                f"{nbytes / 1e6:.1f} MB ({(end - start) / max(median, 1e-9):.0f}x the median stage time) -- "
                f"that single tensor dominates the {seconds * 1e3:.1f} ms total, and it is not a bandwidth limit"
            )
    if populated and summary["overlap"] < _SERIAL_OVERLAP:
        slowest_stage = max((e["seconds"] for e in stages_with_bytes), default=0.0)
        logger.warning(
            f"  {direction}: the stages do not overlap (overlap {summary['overlap']:.2f}x) -- the "
            f"{populated} stage(s) transfer one after another, so the host waits for each in turn. "
            f"Staged, the wall time would be the slowest stage's, {slowest_stage * 1e3:.2f} ms, "
            f"not {summary['summed'] * 1e3:.2f} ms."
        )
    elif summary["overlap"] < populated - 0.05:
        logger.log(
            "INFO",
            f"  {direction}: the stages overlap {summary['overlap']:.2f}x of {populated}; "
            f"fully staged they would take {summary['summed'] * 1e3 / populated:.2f} ms, not "
            f"{seconds * 1e3:.2f} ms",
        )


def _assert_round_trip(snapshot, again, length: int) -> None:
    """Every byte the snapshot carried must come back out of the device unchanged."""
    assert again.pos == snapshot.pos, f"{length} tokens: round trip is at {again.pos}, not {snapshot.pos}"
    assert set(again.rings) == set(snapshot.rings)
    for li, want in snapshot.rings.items():
        assert torch.equal(want, again.rings[li]), f"{length} tokens: layer {li} ring differs after the round trip"
    assert set(again.windows) == set(snapshot.windows)
    for li, want in snapshot.windows.items():
        for name, w, got in zip(("win_kv", "win_gate"), want, again.windows[li]):
            assert torch.equal(w, got), f"{length} tokens: layer {li} {name} differs after the round trip"
    assert set(again.entries) == set(snapshot.entries)
    for src, want in snapshot.entries.items():
        for name, w, got in zip(("entries", "index keys"), want, again.entries[src]):
            assert torch.equal(w, got), f"{length} tokens: source {src} {name} differ after the round trip"
    if snapshot.engram is not None:
        assert torch.equal(snapshot.engram, again.engram), f"{length} tokens: the Engram history differs"


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "num_command_queues": 2,
            "l1_small_size": int(os.environ.get("DEEPSEEK_V4_L1_SMALL_SIZE", "8192")),
        }
    ],
    indirect=["device_params"],
    ids=["fabric_1d"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True, ids=["tp4_8stage"])
def test_kv_cache_migration_bandwidth(mesh_device, reset_seeds, monkeypatch) -> None:
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    with contextlib.ExitStack() as prefetcher:
        model, lm_head, loader, config = _construct_model(mesh_device, prefetcher, loader=loader, config=config)
        # One session, so a snapshot is exactly one conversation's state; the pools are sized for one
        # session's `max_seq`, which only has to clear the longest length in the sweep.
        max_seq = _traced_max_seq(config, max(_SEQ_LENS) + 1, model.num_layers)
        model.prepare_static_decode(
            _build_rope(config, loader.snapshot_dir, max_seq),
            max_seq,
            lm_head=lm_head,
            num_sessions=1,
            total_tokens=max_seq,
            block_size=_PAGE_BLOCK_SIZE,
        )
        sid = model.open_session()
        logger.info(
            f"kv migration profile: {len(_SEQ_LENS)} length(s) {_SEQ_LENS}, max_seq {max_seq}, "
            f"{model.pipeline_stages} pipeline stages of TP{model.tp_size}"
        )

        transfers = _Transfers()
        with _instrument(monkeypatch, model, transfers):
            # A first pass pays the one-off allocation of every slice, so it is not timed.
            _measure(model, sid, _SEQ_LENS[0], transfers)

            measured = [_measure(model, sid, length, transfers) for length in _SEQ_LENS]

            # The round trip must carry bytes no step wrote, or a zeroed pool would pass it either way.
            filled = _measure(model, sid, _SEQ_LENS[-1], transfers, fill=True)
            _assert_round_trip(filled["snapshot"], filled["again"], _SEQ_LENS[-1])

        for run in measured:
            logger.info(f"kv migration {run['length']} tokens: session state {run['snapshot'].nbytes / 1e6:.1f} MB")
            _log_direction(run, "export")
            _log_direction(run, "import")
            # The bytes are only part of the cost: the rest is device slicing, page-table walks and
            # host-side tensor building, which no copy primitive accounts for.
            for direction in ("export", "import"):
                inside = run[direction]["summed"]
                wall = run[f"{direction}_seconds"]
                if wall > 1.2 * inside:
                    logger.info(
                        f"  {direction:6s} {wall * 1e3:7.2f} ms wall but only {inside * 1e3:7.2f} ms in "
                        f"transfers: {100 * (1 - inside / wall):.0f}% is host work and device ops around them "
                        f"({wall / max(inside, 1e-9):.2f}x)"
                    )
        logger.info(
            f"best bandwidth: export "
            f"{max(r['export']['nbytes'] / r['export_seconds'] for r in measured) / 1e9:.2f} GB/s, "
            f"import {max(r['import']['nbytes'] / r['import_seconds'] for r in measured) / 1e9:.2f} GB/s"
        )

        # The session state must grow with the context, and both directions must have moved it.
        sizes = [run["snapshot"].nbytes for run in measured]
        assert sizes == sorted(sizes) and sizes[0] < sizes[-1], f"snapshot sizes do not grow with the context: {sizes}"
        for run in measured:
            length, export, restore = run["length"], run["export"], run["import"]
            assert export["nbytes"] and export["calls"], f"{length} tokens: export moved nothing"
            assert restore["nbytes"] and restore["calls"], f"{length} tokens: import moved nothing"
            assert run["export_seconds"] > 0 and run["import_seconds"] > 0
            # The import legitimately moves more: a shared group's entries go out once but back to every
            # stage holding them. Check it against that expectation rather than against the export.
            model._session_pos[sid] = length
            expected = _expected_import_bytes(model, sid, export["nbytes"])
            assert restore["nbytes"] == pytest.approx(expected, rel=_DIRECTION_TOLERANCE), (
                f"{length} tokens: import moved {restore['nbytes']} B, expected ~{expected} B "
                f"(export {export['nbytes']} B, replicated across the stages holding each group)"
            )
