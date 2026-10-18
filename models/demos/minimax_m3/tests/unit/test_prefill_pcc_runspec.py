# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the multi-run plumbing of tests/galaxy_prefill_kv_pcc.py: spec parsing, the batch /
stdin / fifo spec iterator, isl tiling, chunk planning, and the bounded block-cyclic slot read that
check_kv_pcc relies on (read_slot_kv(n_tokens=...) + naturalize_kv_block). No device."""

import json
import os
from pathlib import Path

import pytest
import torch

from models.common.utils import blockcyclic_positions
from models.demos.minimax_m3.tests import galaxy_prefill_kv_pcc as h
from models.demos.minimax_m3.tt.runners.prefill_kv_validation import naturalize_kv_block


@pytest.fixture
def golden_root(tmp_path):
    for name, n in (("t_small", 181), ("t_5k", 5120), ("t_55k", 55218)):
        d = tmp_path / name
        d.mkdir()
        (d / "metadata.json").write_text(json.dumps({"token_ids": list(range(1, n + 1))}))
    (tmp_path / "not_a_trace").mkdir()
    return str(tmp_path)


def test_runspec_parse_keys_and_defaults(golden_root, monkeypatch):
    monkeypatch.setenv("PREFILL_TPS_ITERS", "5")
    monkeypatch.delenv("PREFILL_MAX_SEQ_LEN", raising=False)
    s = h.RunSpec.parse("trace=t_5k", golden_root)
    assert s.trace_dir == os.path.join(golden_root, "t_5k")
    assert (s.tps_iters, s.skip_pcc, s.expected_tps, s.capacity, s.isl, s.label) == (5, False, None, None, None, "")
    assert s.name == "t_5k"
    s = h.RunSpec.parse(
        f"trace={golden_root}/t_55k iters=2 skip_pcc=1 expected_tps=8225 perf_margin=0.1 pcc_threshold=0.9 "
        "capacity=100000 isl=1000000 label=big",
        None,
    )
    assert (s.tps_iters, s.skip_pcc, s.expected_tps, s.perf_margin) == (2, True, 8225.0, 0.1)
    assert (s.pcc_threshold, s.capacity, s.isl, s.name) == (0.9, 100000, 1000000, "big")
    assert h.RunSpec.parse("trace=t_5k isl=640", golden_root).name == "t_5k@640"
    monkeypatch.setenv("PREFILL_MAX_SEQ_LEN", "56320")
    assert h.RunSpec.parse("trace=t_5k", golden_root).capacity == 56320
    assert h.RunSpec.parse("trace=t_5k capacity=10240", golden_root).capacity == 10240


@pytest.mark.parametrize(
    "line, message",
    [
        ("iters=3", "needs trace="),
        ("trace=nope", "no metadata.json"),
        ("trace=not_a_trace", "no metadata.json"),
        ("trace=t_5k foo=1", "unknown key"),
        ("trace=t_5k iters", "bad token"),
        ("trace=t_5k iters=x", "invalid literal"),
        ("trace=t_5k iters=0", "need >= 1"),
        ("trace=t_5k isl=0", "need >= 1"),
        ("trace=t_5k capacity=-1", "need >= 1"),
    ],
)
def test_runspec_parse_rejects(golden_root, line, message, expect_error):
    with expect_error(ValueError, message):
        h.RunSpec.parse(line, golden_root)


def test_iter_run_specs_batch_file(golden_root, tmp_path):
    f = tmp_path / "runs.txt"
    f.write_text("# header\ntrace=t_5k iters=2\n\ntrace=t_55k # trailing comment\nbogus\nquit\ntrace=t_small\n")
    items = list(h.iter_run_specs(str(f), golden_root))
    assert [type(i).__name__ for _, i in items] == ["RunSpec", "RunSpec", "ValueError", "NoneType"]
    assert items[1][1].trace_dir.endswith("t_55k")  # the comment was stripped before resolving


def test_iter_run_specs_stdin(golden_root, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("trace=t_5k\nexit\n"))
    items = list(h.iter_run_specs("-", golden_root))
    assert len(items) == 2 and items[-1][1] is None


def test_ensure_fifo(tmp_path, expect_error):
    p = tmp_path / "specs.fifo"
    assert h.ensure_fifo(f"fifo:{p}") == str(p)
    assert Path(p).is_fifo()
    h.ensure_fifo(f"fifo:{p}")  # idempotent
    reg = tmp_path / "regular"
    reg.write_text("")
    with expect_error(SystemExit, "not a FIFO"):
        h.ensure_fifo(f"fifo:{reg}")


def test_tile_tokens():
    ids = [1, 2, 3]
    assert h.tile_tokens(ids, 7) == [1, 2, 3, 1, 2, 3, 1]
    assert h.tile_tokens(ids, 2) == [1, 2]
    assert h.tile_tokens(ids, 3) == ids
    big = h.tile_tokens(list(range(55218)), 1_000_000)
    assert len(big) == 1_000_000 and big[55218:55221] == [0, 1, 2] and big[-1] == (1_000_000 - 1) % 55218


def test_plan():
    assert h.plan(55218, 5120, chunked=True) == (11, 5120, 56320)
    assert h.plan(1_000_000, 5120, chunked=True) == (196, 5120, 1_003_520)
    assert h.plan(181, 5120, chunked=False) == (1, h.MSA_MIN_TOKENS, h.MSA_MIN_TOKENS)  # one-shot: MSA floor
    assert h.plan(5000, 5120, chunked=False) == (1, 5120, 5120)  # one-shot: multiple of 1024


@pytest.mark.parametrize(
    "sp,chunk,cap_chunks,n_tokens", [(8, 5120, 11, 55218), (8, 5120, 196, 55218), (4, 1024, 7, 1500)]
)
def test_bounded_slot_read_matches_full_read(sp, chunk, cap_chunks, n_tokens):
    """The first ceil(n/chunk) chunks of a block-cyclic cache live in the first k*chunk_local rows of every
    chip, so slicing each chip there and composing gives exactly the layout of a k*chunk cache — which is
    what read_slot_kv(n_tokens=...) returns and naturalize_kv_block(max_seq_len=k*chunk) un-rotates."""
    capacity, chunk_local = cap_chunks * chunk, chunk // sp
    k = -(-n_tokens // chunk)
    seq_read = k * chunk
    pos = blockcyclic_positions(sp, chunk, capacity)  # composed row r -> natural position
    natural = torch.arange(capacity, dtype=torch.float32).view(1, 1, capacity, 1).expand(1, 1, capacity, 4).clone()
    natural[:, :, n_tokens:] = -1.0  # never written this run
    full_block = natural[:, :, pos]  # what a FULL slot read composes (device layout)
    # bounded read: per-chip rows [0, k*chunk_local) of each chip, concatenated in chip order
    per_chip = full_block.view(1, 1, sp, capacity // sp, 4)[:, :, :, : k * chunk_local]
    bounded_block = per_chip.reshape(1, 1, seq_read, 4)
    got_full = naturalize_kv_block(full_block[0, 0], n_tokens, sp, chunk, capacity)
    got_bounded = naturalize_kv_block(bounded_block[0, 0], n_tokens, sp, chunk, seq_read)
    expect = natural[0, 0, :n_tokens]
    assert torch.equal(got_full, expect) and torch.equal(got_bounded, expect)
