# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The load path's file reads, device-free. ``_read_chunk`` reads a chunk with one ``pread`` when the destination
shards together need at least half of it (on one host they tile the whole tensor, so this is the usual case) and
falls back to one read per contiguous run otherwise; both must put the same bytes in the same places. A data file's
data section starts where its 8-byte header length says."""

import os
import struct
from types import SimpleNamespace

import numpy as np
from dcp_oracle import R, S, FakeCheckpoint, arange, assert_shards_equal, expected_shards, info, world

from ttml.distributed_checkpointing import _load
from ttml.distributed_checkpointing._errors import CheckpointError, IncompleteCheckpointError
from ttml.distributed_checkpointing._layouts import Identity
from ttml.distributed_checkpointing._load import _read_chunk, chunk_reads, elect_checkers
from ttml.distributed_checkpointing._planner import create_load_plan
from ttml.distributed_checkpointing._safetensors import ITEMSIZE, create_file

TP2_COLS = world([R(), S(3)], (1, 2))  # a row-parallel weight: columns split over two devices
FSDP4_ROWS = world([S(2), R()], (4, 1))  # rows split over four devices: every shard needs rows of both chunks
TP4_COLS = world([R(), S(3)], (1, 4))  # columns split over four devices
FSDP2_ROWS = world([S(2), R()], (2, 1))  # rows split over two devices
ROWS4_COLS4 = world([S(2), S(3)], (4, 4))  # rows split over four devices and columns over four
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


def _rank_plans(ckpt, loaded, coords):
    """One load plan per rank, each rank holding one of ``coords``, as a multi-rank load plans."""
    return [create_load_plan(ckpt.index, [info("w", loaded, SHAPE, Identity())], local_coords={c}) for c in coords]


def test_a_chunk_split_over_four_ranks_is_read_whole_and_checked_by_one_of_them(tmp_path, expect_error):
    """Two column-half chunks loaded as four row quarters, one per rank: every rank needs a quarter of each chunk and
    reads it in part, which the chunk's CRC32 cannot cover. One reader of each chunk is elected to read it whole and
    check it, the two chunks going to different ranks; a flipped byte then fails that rank's read."""
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (arange(SHAPE), TP2_COLS, Identity())})
    coords = [(r, 0) for r in range(4)]
    plans = _rank_plans(ckpt, FSDP4_ROWS, coords)
    reports = [chunk_reads(plan) for plan in plans]
    assert all(r == ([["w", 0], ["w", 1]], []) for r in reports), "every rank reads both chunks in part"
    checkers = elect_checkers(ckpt.index, reports)
    assert sorted(pair for pairs in checkers for pair in pairs) == [["w", 0], ["w", 1]], "each chunk once"
    assert sum(1 for pairs in checkers if pairs) == 2, "the two chunks go to two ranks"

    record = ckpt.index.entries["w"].chunks[0]
    ckpt.files[record.file][record.data_offset + record.nbytes - 1] ^= 0xFF
    fds = {}
    for index, buf in ckpt.files.items():
        path = tmp_path / f"file{index}"
        path.write_bytes(bytes(buf))
        fds[index] = os.open(path, os.O_RDONLY)
    try:
        for rank, plan in enumerate(plans):
            kplan = plan.keys["w"]
            arrays = {group: np.zeros(box.sizes, np.float32) for group, box in kplan.boxes.items()}
            for chunk, items in sorted(_by_chunk(kplan).items()):

                def read(whole, items=items):
                    _read_chunk(fds[items[0].file], 0, kplan, items, ITEMSIZE["F32"], np.float32, arrays, "file", whole)

                if chunk == 0 and ["w", 0] in checkers[rank]:
                    with expect_error(CheckpointError, "chunk 0 .* fails its CRC32 check"):
                        read(True)
                read(False)  # read in part, the flipped byte goes unnoticed: what the election is for
    finally:
        for fd in fds.values():
            os.close(fd)


def test_a_chunk_another_rank_reads_whole_gets_no_checker():
    """A chunk split unevenly: a rank needing at least half of it reads it whole and checks it, so the ranks reading
    it in part are not asked to read it again."""
    ckpt = FakeCheckpoint()
    ckpt.save({"w": (arange(SHAPE), FSDP2_ROWS, Identity())})
    reports = [([["w", 0]], []), ([["w", 0]], []), ([], [["w", 0]])]
    assert elect_checkers(ckpt.index, reports) == [[], [], []]
    assert elect_checkers(ckpt.index, reports[:2]) == [[["w", 0]], []]


def test_whole_chunk_reads_and_per_run_reads_agree(tmp_path, monkeypatch):
    kplan, fds, src = _plan_and_files(tmp_path)
    reads = []
    real_pread = _load.pread_all

    def counting_pread(fd, nbytes, offset):
        reads.append(nbytes)
        return real_pread(fd, nbytes, offset)

    monkeypatch.setattr(_load, "pread_all", counting_pread)
    try:
        expected = expected_shards(src, FSDP4_ROWS)
        by_chunk = _by_chunk(kplan)
        assert len(by_chunk) == 2 and all(len(items) == 4 for items in by_chunk.values())

        # All four destination shards: each chunk is needed in full, so it is read once.
        whole = {group: np.zeros(box.sizes, np.float32) for group, box in kplan.boxes.items()}
        n = sum(
            _read_chunk(fds[items[0].file], 0, kplan, items, ITEMSIZE["F32"], np.float32, whole, "file")
            for items in by_chunk.values()
        )
        assert n == src.nbytes
        assert reads == [src.nbytes // 2] * 2, "one pread per chunk, the speed-up a same-world load relies on"
        assert_shards_equal(whole, expected)
        reads.clear()

        # One destination shard needs a quarter of each chunk: below the threshold, so one read per run of rows.
        group = next(iter(kplan.boxes))
        one = {group: np.zeros(kplan.boxes[group].sizes, np.float32)}
        n = sum(
            _read_chunk(
                fds[items[0].file],
                0,
                kplan,
                [i for i in items if i.group == group],
                ITEMSIZE["F32"],
                np.float32,
                one,
                "file",
            )
            for items in by_chunk.values()
        )
        assert n == expected[group].nbytes
        assert reads == [expected[group].nbytes // 2] * 2, "the quarter of each chunk the shard needs, nothing more"
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
            _read_chunk(fds[items[0].file], 0, kplan, mine, ITEMSIZE["F32"], np.float32, one, "file")

        np.testing.assert_array_equal(one[group], expected_shards(src, TP4_COLS)[group])
        # Each of the four 2x16 chunks is read as two runs: columns 4-7 of its row 0, then of its row 1.
        itemsize = ITEMSIZE["F32"]
        run, start, row = 4 * itemsize, 4 * itemsize, SHAPE[3] * itemsize
        assert sorted(reads) == [(run, start)] * 4 + [(run, row + start)] * 4, reads
    finally:
        for fd in fds.values():
            os.close(fd)


def test_a_block_from_the_middle_of_a_chunk_reads_its_own_rows(tmp_path, monkeypatch):
    """A destination needing rows 2-3 of a 4-row chunk and some of its columns reads two runs that start at chunk rows
    2 and 3: the block's offset into the chunk counts on every dim before the run's, not just on the run's own."""
    kplan, fds, src = _plan_and_files(tmp_path, saved=FSDP2_ROWS, loaded=ROWS4_COLS4)
    reads = []
    real_pread = _load.pread_all

    def counting_pread(fd, nbytes, offset):
        reads.append((nbytes, offset))
        return real_pread(fd, nbytes, offset)

    monkeypatch.setattr(_load, "pread_all", counting_pread)
    try:
        group = next(g for g, box in kplan.boxes.items() if tuple(box.offsets) == (0, 0, 2, 4))  # rows 2-3, cols 4-7
        one = {group: np.zeros(kplan.boxes[group].sizes, np.float32)}
        for items in _by_chunk(kplan).values():
            mine = [i for i in items if i.group == group]
            if mine:
                _read_chunk(fds[items[0].file], 0, kplan, mine, ITEMSIZE["F32"], np.float32, one, "file")

        np.testing.assert_array_equal(one[group], expected_shards(src, ROWS4_COLS4)[group])
        # Chunk 0 holds rows 0-3; the block is columns 4-7 of its rows 2 and 3.
        itemsize = ITEMSIZE["F32"]
        run, start, row = 4 * itemsize, 4 * itemsize, SHAPE[3] * itemsize
        assert sorted(reads) == [(run, 2 * row + start), (run, 3 * row + start)], reads
    finally:
        for fd in fds.values():
            os.close(fd)


def test_a_same_layout_chunk_is_read_into_its_destination(tmp_path, monkeypatch, expect_error):
    """A chunk that is its one destination shard, as in a same-world load, is read into the shard's array itself, with
    no intermediate bytes, and still checked against its CRC32."""
    kplan, fds, src = _plan_and_files(tmp_path, saved=TP2_COLS, loaded=TP2_COLS)

    def no_intermediate_copy(fd, nbytes, offset):
        raise AssertionError("a same-layout chunk was read into intermediate bytes")

    monkeypatch.setattr(_load, "pread_all", no_intermediate_copy)
    try:
        arrays = {group: np.zeros(box.sizes, np.float32) for group, box in kplan.boxes.items()}
        chunks = _by_chunk(kplan)
        for items in chunks.values():
            assert len(items) == 1, "precondition: each chunk is one destination shard"
            n = _read_chunk(fds[items[0].file], 0, kplan, items, ITEMSIZE["F32"], np.float32, arrays, "file")
            assert n == src.nbytes // 2
        assert_shards_equal(arrays, expected_shards(src, TP2_COLS))

        items = chunks[max(chunks)]
        with open(tmp_path / f"file{items[0].file}", "r+b") as f:  # one flipped bit; the descriptor sees it
            f.seek(5)
            byte = f.read(1)[0]
            f.seek(5)
            f.write(bytes([byte ^ 0x40]))
        with expect_error(CheckpointError, r"w: chunk 1 \(.*\) in file fails its CRC32 check"):
            _read_chunk(fds[items[0].file], 0, kplan, items, ITEMSIZE["F32"], np.float32, arrays, "file")
    finally:
        for fd in fds.values():
            os.close(fd)


def test_a_whole_chunk_read_checks_the_chunk_crc(tmp_path, expect_error):
    ckpt = FakeCheckpoint()
    src = arange(SHAPE)
    ckpt.save({"w": (src, TP2_COLS, Identity())})  # records every chunk's CRC32, as save does
    plan, _ = ckpt.load({"w": (FSDP4_ROWS, Identity(), SHAPE)})
    kplan = plan.keys["w"]
    ckpt.files[1][5] ^= 0x40  # one flipped bit in the second chunk; the file size is unchanged
    fds = {}
    for index, buf in ckpt.files.items():
        (tmp_path / f"file{index}").write_bytes(bytes(buf))
        fds[index] = os.open(tmp_path / f"file{index}", os.O_RDONLY)
    try:
        chunks = _by_chunk(kplan)
        whole = {group: np.zeros(box.sizes, np.float32) for group, box in kplan.boxes.items()}
        first, second = (chunks[c] for c in sorted(chunks))
        _read_chunk(fds[first[0].file], 0, kplan, first, ITEMSIZE["F32"], np.float32, whole, "file0")
        with expect_error(CheckpointError, r"w: chunk 1 \(.*\) in file1 fails its CRC32 check"):
            _read_chunk(fds[second[0].file], 0, kplan, second, ITEMSIZE["F32"], np.float32, whole, "file1")
    finally:
        for fd in fds.values():
            os.close(fd)


def test_data_files_take_their_data_offset_from_the_header_length(tmp_path, expect_error):
    """Load takes each data file's data offset from its 8-byte header length, without parsing the header, and refuses
    a length that runs past the end of the file."""
    body = b'{"__metadata__":{}}   '
    header = struct.pack("<Q", len(body)) + body
    good = create_file(str(tmp_path / "good.safetensors"), header, 64)
    (tmp_path / "bad.safetensors").write_bytes(struct.pack("<Q", 1 << 20) + bytes(64))
    index = SimpleNamespace(
        files=[
            SimpleNamespace(path="good.safetensors", nbytes=good + 64),
            SimpleNamespace(path="bad.safetensors", nbytes=72),
        ]
    )

    def plan(*files):
        return SimpleNamespace(keys={"w": SimpleNamespace(reads=[SimpleNamespace(file=f) for f in files])})

    data_files = _load._DataFiles()
    try:
        data_files.open(str(tmp_path), index, plan(0))
        assert data_files.data_start == {0: len(header)}
    finally:
        data_files.close()
    data_files = _load._DataFiles()
    try:
        with expect_error(IncompleteCheckpointError, r"bad.safetensors declares a 1048576-byte header but is 72 bytes"):
            data_files.open(str(tmp_path), index, plan(1))
    finally:
        data_files.close()


def test_a_partly_restored_load_says_what_to_do(expect_error):
    """Once a tensor was placed, a failed load says that the model is partly restored. On a mesh of one rank's own the
    model can be loaded again; on a mesh several ranks share, ranks that stopped at different tensors no longer make
    the same device allocations, so the job has to restart. A failure before any tensor passes as raised."""
    err = RuntimeError("simulated read error")
    with expect_error(
        CheckpointError,
        r"after 1 of 2 tensors were placed \(the first: a\); the model and optimizer are partly restored: load again "
        r"or discard them$",
    ):
        _load._raise_partly_restored(err, "b", ["a"], 2)
    with expect_error(
        CheckpointError,
        r"partly restored, and the ranks sharing this mesh may now hold different device allocations: end the job and "
        r"restart it instead of loading again$",
    ):
        _load._raise_partly_restored(err, "b", ["a"], 2, shared_mesh=True)
    with expect_error(RuntimeError, r"^simulated read error$"):
        _load._raise_partly_restored(err, "a", [], 2, shared_mesh=True)
