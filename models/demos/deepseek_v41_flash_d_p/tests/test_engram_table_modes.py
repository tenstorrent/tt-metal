# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU: tt/engram.py's rows are EngramTable.rows' bytes in every ENGRAM_TABLE mode and thread count (no device).

resident copies a whole table into RAM (94.5 GiB, ~2 min): it runs only with DS41_TEST_ENGRAM_RESIDENT=1; its copy is
also checked on a small tensor in every run."""

import os

import pytest
import torch

from models.demos.common.bringup.reference.golden import hf_path
from models.demos.common.bringup.testing.harness import spec
from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
from models.demos.deepseek_v41_flash_d_p.reference.ngram import DEAD
from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint, EngramTable
from models.demos.deepseek_v41_flash_d_p.tt.engram import TtEngramRows, engram_hasher, resident_copy

LAYER, TOKENS = 1, 5120
MODES = ["mmap", "random"] + (["resident"] if os.environ.get("DS41_TEST_ENGRAM_RESIDENT") == "1" else [])


@pytest.fixture(scope="module")
def setup():
    path = str(hf_path(spec()))
    cfg = Config.from_dir(path)
    return cfg, Checkpoint(path), engram_hasher(path, cfg)


@pytest.mark.parametrize("threads", [1, 8])
@pytest.mark.parametrize("mode", MODES)
def test_rows_bit_identical(setup, mode, threads):
    cfg, ck, hasher = setup
    table = EngramTable(ck, LAYER)
    rows = TtEngramRows(None, hasher, table, LAYER, mode=mode, threads=threads)
    ids = torch.randint(0, cfg.vocab_size, (TOKENS,), generator=torch.Generator().manual_seed(threads))
    ids[-7:] = -1  # engine pad past actual_end
    tail = torch.full((hasher.max_ngram_size - 1,), DEAD, dtype=torch.int64)
    got, got_tail = rows.host_rows(ids, tail)
    h, ref_tail = hasher.hashes(ids[:-7], LAYER, tail)
    ref = torch.zeros_like(got)
    ref[:-7] = table.rows(h).flatten(1).to(torch.bfloat16)
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))
    assert torch.equal(got_tail, ref_tail)


def test_resident_copy_is_the_stored_bytes(setup):
    _, ck, _ = setup
    name = f"layers.{LAYER}.engram.wkv.scale"
    got = resident_copy(ck, name)
    assert torch.equal(got.reshape(-1), ck.view(name).reshape(-1).view(torch.uint8))
