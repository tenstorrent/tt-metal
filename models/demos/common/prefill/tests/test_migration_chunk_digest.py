# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-free checks of migration_chunk_digest. Compare: per-host digests merge per side, and any chunk whose
bytes differ, that only one side hashed, or that a side's table names but no host hashed, fails the check, as do
positions left unchecked. Digest: which bytes of a chunk hold its first positions, per storage layout."""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "migration_chunk_digest", Path(__file__).parents[1] / "runners" / "migration_chunk_digest.py"
)
mcd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mcd)


def _write(path, chunks, expected, positions=64, unchecked=None, slot=0):
    scope = {
        "slot": slot,
        "positions": positions,
        "configs": ["00", "01"],
        "layouts": {},
        "unchecked_positions": unchecked or {},
    }
    path.write_text(json.dumps({"host": path.stem, "scope": scope, "expected_chunks": expected, "chunks": chunks}))


def _compare(tmp_path, src_hosts, dst_hosts, expected=None):
    """``expected`` is the chunk count each side's table names; by default, what its hosts hashed between them."""
    for side, hosts in (("prefill", src_hosts), ("decode", dst_hosts)):
        total = len(set().union(*hosts)) if expected is None else expected
        for i, chunks in enumerate(hosts):
            _write(tmp_path / f"{side}_h{i}.json", chunks, total)

    class Args:
        src = str(tmp_path / "prefill_*.json")
        dst = str(tmp_path / "decode_*.json")

    return mcd.compare(Args)


def test_identical_chunks_across_differently_split_hosts_pass(tmp_path):
    src = [{"00/3/0": "a", "00/3/32": "b"}, {"01/3/0": "c"}]
    dst = [{"00/3/0": "a"}, {"00/3/32": "b", "01/3/0": "c"}]
    assert _compare(tmp_path, src, dst) == 0


@pytest.mark.parametrize(
    "dst",
    [
        [{"00/3/0": "a", "00/3/32": "X", "01/3/0": "c"}],  # a byte difference
        [{"00/3/0": "a", "01/3/0": "c"}],  # a chunk decode never received
    ],
)
def test_a_differing_or_missing_chunk_fails(tmp_path, dst):
    src = [{"00/3/0": "a", "00/3/32": "b", "01/3/0": "c"}]
    assert _compare(tmp_path, src, dst) == 1


def test_a_host_missing_from_both_sides_fails(tmp_path):
    # Each side's table names three chunks, and the host holding 01/3/0 was digested on neither: the two sides
    # still agree key for key.
    left = [{"00/3/0": "a", "00/3/32": "b"}]
    assert _compare(tmp_path, left, left, expected=3) == 1


def test_digests_of_one_side_taken_with_different_arguments_are_an_error(tmp_path, expect_error):
    _write(tmp_path / "prefill_h0.json", {"00/3/0": "a"}, 2)
    _write(tmp_path / "prefill_h1.json", {"00/3/32": "b"}, 2, positions=128)
    _write(tmp_path / "decode_h0.json", {"00/3/0": "a", "00/3/32": "b"}, 2)

    class Args:
        src = str(tmp_path / "prefill_*.json")
        dst = str(tmp_path / "decode_*.json")

    with expect_error(SystemExit, "differ in scope or table"):
        mcd.compare(Args)


def test_two_captures_of_one_chunk_that_disagree_are_an_error(tmp_path, expect_error):
    src = [{"00/3/0": "CORRUPT"}, {"00/3/0": "a"}]
    with expect_error(SystemExit, "00/3/0"):
        _compare(tmp_path, src, [{"00/3/0": "a"}])


def _compare_files(tmp_path):
    class Args:
        src = str(tmp_path / "prefill_*.json")
        dst = str(tmp_path / "decode_*.json")

    return mcd.compare(Args)


def test_the_two_sides_may_differ_in_slot_only(tmp_path):
    chunks = {"00/3/0": "a"}
    _write(tmp_path / "prefill_h0.json", chunks, 1, slot=0)
    _write(tmp_path / "decode_h0.json", chunks, 1, slot=1)
    assert _compare_files(tmp_path) == 0
    _write(tmp_path / "decode_h0.json", chunks, 1, slot=1, positions=96)
    assert _compare_files(tmp_path) == 1


def test_positions_left_unchecked_fail(tmp_path, capsys):
    chunks = {"00/3/0": "a"}
    for side in ("prefill", "decode"):
        _write(tmp_path / f"{side}_h0.json", chunks, 1, positions=63, unchecked={"01": 31})
    assert _compare_files(tmp_path) == 1
    assert "config 01: last 31 position(s) of each layer NOT checked" in capsys.readouterr().out


def test_rows_layout_keeps_the_leading_pages():
    raw = bytes(range(64)) * 2  # 32 positions, 4 bytes each
    assert mcd._bytes_below(raw, "rows", 31, 32) == raw[:124]
    assert mcd._bytes_below(raw, "rows", 1, 32) == raw[:4]


def test_bfp8_tile_rows_partition_the_tile():
    # Every byte of a tile belongs to exactly one tile row: 2 exponent bytes and 32 mantissa bytes each.
    owner = {}
    for row in range(32):
        spans = mcd._bfp8_tile_row_spans(row)
        assert sum(length for _, length in spans) == 34
        for off, length in spans:
            for byte in range(off, off + length):
                assert byte not in owner
                owner[byte] = row
    assert sorted(owner) == list(range(1088))
    # Tile row 17 is face row 1 of the two bottom faces: exponents 33 and 49, mantissas at 64 + 2*256 + 16.
    assert mcd._bfp8_tile_row_spans(17) == [(33, 1), (592, 16), (49, 1), (848, 16)]


def test_bfp8_layout_ignores_bytes_of_later_positions():
    def chunk(fill):
        """Two tiles; rows below 31 hold their row number, row 31 holds ``fill``."""
        tile = bytearray(1088)
        for row in range(32):
            for off, length in mcd._bfp8_tile_row_spans(row):
                tile[off : off + length] = bytes([fill if row == 31 else row]) * length
        return bytes(tile) * 2

    assert mcd._bytes_below(chunk(200), "bfp8", 31, 32) == mcd._bytes_below(chunk(201), "bfp8", 31, 32)
    assert len(mcd._bytes_below(chunk(200), "bfp8", 31, 32)) == 2 * 31 * 34
    assert mcd._bytes_below(chunk(200), "bfp8", 30, 32) != mcd._bytes_below(chunk(200), "bfp8", 31, 32)
    assert mcd._bytes_below(chunk(200), "bfp8", 32, 32) != mcd._bytes_below(chunk(201), "bfp8", 32, 32)


def test_a_chunk_that_does_not_fit_its_layout_is_an_error(expect_error):
    with expect_error(SystemExit, "not whole"):
        mcd._bytes_below(bytes(1000), "bfp8", 5, 32)
    with expect_error(SystemExit, "equal pages"):
        mcd._bytes_below(bytes(1001), "rows", 5, 32)
    with expect_error(SystemExit, "unknown layout"):
        mcd._parse_layouts("00=tiles")


def test_bfp8_tile_rows_match_the_producer_decoder():
    # The producer's decoder reads the same DRAM bytes for the PCC gate, so it is the reference for which bytes
    # hold which position.
    import numpy as np

    from models.demos.common.prefill.runners.prefill_producer import _decode_bfp8_chunk

    n, tiles = 31, 2
    rng = np.random.default_rng(0)
    raw = rng.integers(1, 128, size=tiles * 1088, dtype=np.uint8)
    below = np.zeros(raw.size, dtype=bool)
    for tile in range(tiles):
        for row in range(n):
            for off, length in mcd._bfp8_tile_row_spans(row):
                below[tile * 1088 + off : tile * 1088 + off + length] = True
    assert mcd._bytes_below(raw.tobytes(), "bfp8", n, 32) == raw[below].tobytes()

    reference = _decode_bfp8_chunk(raw.tobytes(), tiles * 32)
    later_changed = _decode_bfp8_chunk(np.where(below, raw, raw ^ 1).tobytes(), tiles * 32)
    assert (later_changed[:n] == reference[:n]).all() and (later_changed[n:] != reference[n:]).all()
    below_changed = _decode_bfp8_chunk(np.where(below, raw ^ 1, raw).tobytes(), tiles * 32)
    assert (below_changed[:n] != reference[:n]).all() and (below_changed[n:] == reference[n:]).all()
