# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-free checks of migration_chunk_digest's compare step: per-host digests merge per side, and any chunk
whose bytes differ, that only one side hashed, or that a side's table names but no host hashed, fails the check."""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "migration_chunk_digest", Path(__file__).parents[1] / "runners" / "migration_chunk_digest.py"
)
mcd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mcd)


def _write(path, chunks, expected, positions=64):
    scope = {"slot": 0, "positions": positions, "configs": ["00", "01"]}
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
