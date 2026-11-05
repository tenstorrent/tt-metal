# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The load path's file reads, device-free. ``_read_chunk`` reads a chunk with one ``pread`` when the destination
shards together need at least half of it (on one host they tile the whole tensor, so this is the usual case) and
falls back to one read per contiguous run otherwise; both must put the same bytes in the same places."""

import os

import numpy as np
from dcp_oracle import R, S, FakeCheckpoint, arange, assert_shards_equal, expected_shards, world

from ttml.distributed_checkpointing._layouts import Identity
from ttml.distributed_checkpointing._load import _read_chunk
from ttml.distributed_checkpointing._safetensors import ITEMSIZE

TP2_COLS = world([R(), S(3)], (1, 2))  # a row-parallel weight: columns split over two devices
FSDP4_ROWS = world([S(2), R()], (4, 1))  # rows split over four devices: every shard needs rows of both chunks
SHAPE = (1, 1, 8, 16)


def _plan_and_files(tmp_path):
    ckpt = FakeCheckpoint()
    src = arange(SHAPE)
    ckpt.save({"w": (src, TP2_COLS, Identity())})
    plan, _ = ckpt.load({"w": (FSDP4_ROWS, Identity(), SHAPE)})
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
