# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_QSA_MT_ROW without a device: on by default, the dispatch of main_tail's staging core kernels (flag off ->
the default kernels; flag on with one lane -> the one-row kernels; more lanes, paged or a KV stage -> the default
kernels), and the one-row kernels' text."""

from __future__ import annotations

import inspect
import re

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block

KERNELS = qsa_block.fp.REPO_ROOT / "models/demos/blackhole/qwen38_flash_next/ttnn/fused/qsa_block/kernels"
SOURCE = inspect.getsource(qsa_block.main_tail)


def test_on_by_default():
    assert qsa_block.MT_ROW_ENV == "QWEN38_QSA_MT_ROW"
    assert qsa_block.mt_row_enabled({})
    assert not qsa_block.mt_row_enabled({qsa_block.MT_ROW_ENV: "0"})
    assert not qsa_block.mt_row_enabled({qsa_block.MT_ROW_ENV: "true"})
    assert qsa_block.mt_row_enabled({qsa_block.MT_ROW_ENV: "1"})


def test_admits_only_the_single_lane_unpaged_decode():
    assert qsa_block.mt_row_admits(1, False, None)
    for rows in (2, 8, 32):
        assert not qsa_block.mt_row_admits(rows, False, None)
    assert not qsa_block.mt_row_admits(1, True, None)
    assert not qsa_block.mt_row_admits(1, False, object())


def test_dispatch_in_main_tail():
    # the kernel sets: the flag and the admission pick the one-row pair, everything else the default three kernels
    assert "mt_row = mt_row_enabled() and mt_row_admits(rows, paged, kv_stage)" in SOURCE
    assert "kernels += staging_row_kernels() if mt_row else staging_kernels()" in SOURCE
    assert 'MAIN_TAIL["reader_staging"]' in SOURCE and 'MAIN_TAIL["compute_staging"]' in SOURCE
    assert 'MAIN_TAIL_ROW["reader"]' in SOURCE and 'MAIN_TAIL_ROW["writer"]' in SOURCE
    row_kernels = SOURCE[
        SOURCE.index("def staging_row_kernels") : SOURCE.index("if kv_stage is None:\n        kernels +=")
    ]
    assert "compute_kernel" not in row_kernels and "MAIN_TAIL[" not in row_kernels
    assert row_kernels.count("fp.reader_kernel(") == 1 and row_kernels.count("fp.writer_kernel(") == 1
    assert qsa_block.MAIN_TAIL_ROW["reader"].endswith("main_tail_reader_staging_row.cpp")
    assert qsa_block.MAIN_TAIL_ROW["writer"].endswith("main_tail_writer_staging_row.cpp")
    assert qsa_block.MAIN_TAIL["reader_staging"].endswith("main_tail_reader_staging.cpp")
    assert qsa_block.CB_MT_ROW == 30


def test_default_kernels_are_untouched_by_the_flag():
    for name in ("main_tail_reader_staging.cpp", "main_tail_compute_staging.cpp", "main_tail_writer_staging.cpp"):
        assert "CB_ROW" not in (KERNELS / name).read_text()
    assert "CB_ROW = 30" in (KERNELS / "main_tail_cbs.h").read_text()


def _code(text):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)


def test_one_row_kernels_text():
    reader = _code((KERNELS / "main_tail_reader_staging_row.cpp").read_text())
    writer = _code((KERNELS / "main_tail_writer_staging_row.cpp").read_text())
    assert "sem_krot.wait_min(1)" in reader and "sem_knorm.wait_min(1)" in reader
    header = _code((KERNELS / "main_tail_row.h").read_text())
    assert "sem_done.wait(1)" in reader and "SEM_ROW_DONE" in writer and "HEAD_TILES / 2" in reader
    assert "MT_ROW_SPLIT" not in reader + writer
    assert reader.count("cb_push_back(CB_ROW, 1)") == 2 and "canonical_word(s[k])" in header
    assert reader.index("build_tiles(pack_l1, 0, HEAD_TILES, lane") < reader.index(
        "sem_krot.wait_min(1)"
    )  # the V half first
    assert "copy_tile_row" not in reader and "CB_STG" not in reader and "CB_ONES" not in reader
    # per half one 512 B cache write and the 32-byte staging chunks, one barrier, no 32-row loop
    assert writer.count("noc_async_write(") == 2 and writer.count("noc_async_write_barrier()") == 1
    assert "cache.get_noc_addr(cache_row, h * HALF_BYTES)" in writer and "cb_wait_front(CB_ROW, h + 1)" in writer
    assert "chunk_offset(slot, half)" in writer and "HALF_BYTES = QUERY_ROW_BYTES / 2" in writer
    assert "TILE_ROWS" not in writer and "CB_RM" not in writer and "CB_STGW" not in writer
