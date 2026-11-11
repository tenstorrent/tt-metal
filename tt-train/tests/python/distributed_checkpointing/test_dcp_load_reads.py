# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The load path's file reads, device-free. ``_read_chunk`` reads a chunk with one ``pread`` when the destination
shards together need at least half of it (on one host they tile the whole tensor, so this is the usual case) and
falls back to one read per contiguous run otherwise; both must put the same bytes in the same places."""

import os

import numpy as np
from dcp_oracle import R, S, FakeCheckpoint, arange, assert_shards_equal, expected_shards, world

from ttml.distributed_checkpointing import _load
from ttml.distributed_checkpointing._layouts import Identity
from ttml.distributed_checkpointing._load import _read_chunk
from ttml.distributed_checkpointing._safetensors import ITEMSIZE

TP2_COLS = world([R(), S(3)], (1, 2))  # a row-parallel weight: columns split over two devices
FSDP4_ROWS = world([S(2), R()], (4, 1))  # rows split over four devices: every shard needs rows of both chunks
TP4_COLS = world([R(), S(3)], (1, 4))  # columns split over four devices
SHAPE = (1, 1, 8, 16)


def _plan_and_files(tmp_path, saved=TP2_COLS, loaded=FSDP4_ROWS):
    ckpt = FakeCheckpoint()
    src = arange(SHAPE)
    ckpt.save({"w": (src, saved, Identity())})
    plan, _ = ckpt.load({"w": (loaded, Identity(), SHAPE)})
    fds = {}
    for index, buf in ckpt.files.items():  # the fake files hold just the data region, so data starts at 0
        path = tmp_path / f"file{index}"
        path.write_bytes(bytes(buf))
        fds[index] = os.open(path, os.O_RDONLY)
    return plan.keys["w"], fds, src


def _by_chunk(kplan):
    groups = {}
    for read in kplan.reads:
        groups.setdefault(read.chunk, []).append(read)
    return groups


def test_whole_chunk_reads_and_per_run_reads_agree(tmp_path):
    kplan, fds, src = _plan_and_files(tmp_path)
    try:
        expected = expected_shards(src, FSDP4_ROWS)
        by_chunk = _by_chunk(kplan)
        assert len(by_chunk) == 2 and all(len(items) == 4 for items in by_chunk.values())

        # All four destination shards: each chunk is needed in full, so it is read once.
        whole = {group: np.zeros(box.sizes, np.float32) for group, box in kplan.boxes.items()}
        n = sum(
            _read_chunk(fds[items[0].file], 0, kplan, items, ITEMSIZE["F32"], np.float32, whole)
            for items in by_chunk.values()
        )
        assert n == src.nbytes
        assert_shards_equal(whole, expected)

        # One destination shard needs a quarter of each chunk: below the threshold, so one read per run of rows.
        group = next(iter(kplan.boxes))
        one = {group: np.zeros(kplan.boxes[group].sizes, np.float32)}
        n = sum(
            _read_chunk(
                fds[items[0].file], 0, kplan, [i for i in items if i.group == group], ITEMSIZE["F32"], np.float32, one
            )
            for items in by_chunk.values()
        )
        assert n == expected[group].nbytes
        np.testing.assert_array_equal(one[group], expected[group])
    finally:
        for fd in fds.values():
            os.close(fd)


def test_a_column_slice_of_a_chunk_is_read_row_by_row(tmp_path, monkeypatch):
    """A destination needing some columns of a chunk gets them with one read per row: on disk a chunk's rows are
    contiguous, but a column slice of them is not."""
    kplan, fds, src = _plan_and_files(tmp_path, saved=FSDP4_ROWS, loaded=TP4_COLS)
    reads = []
    real_pread = _load.pread_all

    def counting_pread(fd, nbytes, offset):
        reads.append((nbytes, offset))
        return real_pread(fd, nbytes, offset)

    monkeypatch.setattr(_load, "pread_all", counting_pread)
    try:
        group = sorted(kplan.boxes)[1]  # columns 4-7, so every read starts inside its row
        box = kplan.boxes[group]
        assert box.offsets[3] == 4 and box.sizes[3] == 4, "precondition: columns 4-7"
        one = {group: np.zeros(box.sizes, np.float32)}
        for items in _by_chunk(kplan).values():
            mine = [i for i in items if i.group == group]
            _read_chunk(fds[items[0].file], 0, kplan, mine, ITEMSIZE["F32"], np.float32, one)

        np.testing.assert_array_equal(one[group], expected_shards(src, TP4_COLS)[group])
        # Each of the four 2x16 chunks is read as two runs: columns 4-7 of its row 0, then of its row 1.
        itemsize = ITEMSIZE["F32"]
        run, start, row = 4 * itemsize, 4 * itemsize, SHAPE[3] * itemsize
        assert sorted(reads) == [(run, start)] * 4 + [(run, row + start)] * 4, reads
    finally:
        for fd in fds.values():
            os.close(fd)
