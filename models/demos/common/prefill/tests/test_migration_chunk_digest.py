# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-free checks of migration_chunk_digest's compare step: per-host digests merge per side, and any chunk
whose bytes differ, or that only one side hashed, fails the check."""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "migration_chunk_digest", Path(__file__).parents[1] / "runners" / "migration_chunk_digest.py"
)
mcd = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mcd)


def _write(path, chunks):
    path.write_text(json.dumps({"host": path.stem, "chunks": chunks}))


def _compare(tmp_path, src_hosts, dst_hosts):
    for i, chunks in enumerate(src_hosts):
        _write(tmp_path / f"prefill_h{i}.json", chunks)
    for i, chunks in enumerate(dst_hosts):
        _write(tmp_path / f"decode_h{i}.json", chunks)

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


def test_two_captures_of_one_chunk_that_disagree_are_an_error(tmp_path):
    src = [{"00/3/0": "CORRUPT"}, {"00/3/0": "a"}]
    with pytest.raises(SystemExit, match="00/3/0"):
        _compare(tmp_path, src, [{"00/3/0": "a"}])
