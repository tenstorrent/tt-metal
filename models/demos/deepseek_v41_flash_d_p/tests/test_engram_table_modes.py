# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU: tt/engram.py's rows are EngramTable.rows' bytes in every ENGRAM_TABLE mode and thread count (no device).

resident copies a whole table into RAM (94.5 GiB raw / 104.5 GB as bfp8, ~2 min): it runs only with
DS41_TEST_ENGRAM_RESIDENT=1; its copy is also checked on a small tensor, and the packed-row tiling on a sample of rows,
in every run."""

import os

import pytest
import torch

from models.demos.common.bringup.reference.golden import hf_path
from models.demos.common.bringup.testing.harness import spec
from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
from models.demos.deepseek_v41_flash_d_p.reference.ngram import DEAD
from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint, EngramTable
from models.demos.deepseek_v41_flash_d_p.tt.engram import TtEngramRows, engram_hasher, resident_bfp8, resident_copy
from models.demos.deepseek_v41_flash_d_p.tt.engram_bfp8 import pack_into, pack_rows, tile_bytes, tile_rows

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


def test_packed_rows_tile_as_the_packed_chunk(setup):
    """tile_rows (gather from packed bfp8 rows straight into TILE bytes) == tile_bytes of pack_rows of the same rows,
    rows past valid 0 (a sample of real table rows: the packed table is those rows)."""
    _, ck, _ = setup
    table = EngramTable(ck, LAYER)
    g = torch.Generator().manual_seed(3)
    rows = torch.randint(0, table.num_rows, (4096,), generator=g)
    w, s = table.weight[rows].contiguous(), table.scale[rows].contiguous()
    packed = torch.empty(4096, 272, dtype=torch.uint8).numpy()
    pack_into(w, s, packed, threads=8)
    n, valid = 256, 249
    ids = torch.randint(0, 4096, (valid, 24), generator=g)
    got = tile_rows(packed, ids, n, threads=8)
    e, m = pack_rows(w[ids.reshape(-1)], s[ids.reshape(-1)])
    exps, mags = torch.zeros(n, 384, dtype=torch.uint8), torch.zeros(n, 6144, dtype=torch.uint8)
    exps[:valid], mags[:valid] = e.view(valid, -1), m.view(valid, -1)
    assert torch.equal(got, tile_bytes(exps, mags))


@pytest.mark.skipif(os.environ.get("DS41_TEST_ENGRAM_RESIDENT") != "1", reason="loads a whole table (~2 min)")
def test_resident_bfp8_is_the_packed_table(setup):
    """The resident bfp8 table's rows == pack_rows of the stored rows, and a chunk's tiles from it == the per-chunk
    pack's (ENGRAM_TABLE=resident vs mmap with ENGRAM_ROWS=bfp8)."""
    cfg, ck, hasher = setup
    table = EngramTable(ck, LAYER)
    packed = resident_bfp8(ck, LAYER)
    rows = torch.cat([torch.arange(4096), torch.randint(0, table.num_rows, (1 << 16,)), torch.arange(-4096, 0)])
    rows = rows % table.num_rows
    e, m = pack_rows(table.weight[rows].contiguous(), table.scale[rows].contiguous())
    assert torch.equal(packed[rows], torch.cat([e, m], dim=1))
    res = TtEngramRows(None, hasher, table, LAYER, mode="resident", threads=8, rows_format="bfp8")
    ref = TtEngramRows(None, hasher, table, LAYER, mode="mmap", threads=8, rows_format="bfp8")
    ids = torch.randint(0, cfg.vocab_size, (TOKENS,), generator=torch.Generator().manual_seed(5))
    ids[-7:] = -1
    tail = torch.full((hasher.max_ngram_size - 1,), DEAD, dtype=torch.int64)
    (tiles,), t1 = res.tiles_from_packed(ids, tail)
    exps, mags, t2 = ref.host_bfp8(ids, tail)
    assert torch.equal(tiles, tile_bytes(exps, mags)) and torch.equal(t1, t2)
