# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device contract of QWEN38_GR_GATE_TINY: default off, rows == 1 only, the compute kernels untouched, the readers and
writers of the tiny program move the two valid-row chunks at 64 B aligned DRAM offsets, the up weight stays whole tiles.
"""

from pathlib import Path

import pytest

import ttnn
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gr_read
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.gr_read import gate3

KERNELS = Path(gr_read.__file__).parent / "kernels"
GATE3_PY = Path(gate3.__file__).read_text()


def test_default_on_and_rows_one_only(monkeypatch):
    monkeypatch.delenv(gate3.TINY_ENV, raising=False)
    assert gate3.tiny_enabled(1)
    for rows in (2, 4, 8, 16):
        assert not gate3.tiny_enabled(rows)
    monkeypatch.setenv(gate3.TINY_ENV, "1")
    assert gate3.tiny_enabled(1)
    for rows in (2, 4, 8, 16):
        assert not gate3.tiny_enabled(rows)
    monkeypatch.setenv(gate3.TINY_ENV, "0")
    assert not gate3.tiny_enabled(1)


def test_tile_sizes_and_face_offsets():
    assert (gate3.TINY_BF16, gate3.TINY_FP32) == (64, 128)  # 1 x 32: 32 values
    assert gate3.FACE_BF16_BYTES == 512 and gate3.FACE_FP32_BYTES == 1024  # face 1 in a 32 x 32 DRAM page
    for offset in (gate3.FACE_BF16_BYTES, gate3.FACE_FP32_BYTES):
        assert offset % 64 == 0  # a DRAM read/write chunk must start 64 B aligned


def test_cb_descriptor_tile_rows_is_optional():
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    full = fp.cb_descriptor(3, ttnn.bfloat16, 2048, 2, cores)
    tiny = fp.cb_descriptor(3, ttnn.bfloat16, 64, 2, cores, 1)
    assert full.total_size == 4096 and tiny.total_size == 128
    assert tiny.format_descriptors[0].page_size == 64


def test_compute_kernels_are_the_untouched_files():
    # the tiny program runs gate_compute3.cpp / lowrank_compute3.cpp as they are: the tile size comes from the CBs
    assert "GT_" not in (KERNELS / "gate_compute3.cpp").read_text()
    assert "GT_" not in (KERNELS / "lowrank_compute3.cpp").read_text()
    assert "LOWRANK3, plain_set" in GATE3_PY and "GATE3,\n        w_set" in GATE3_PY


@pytest.mark.parametrize(
    "name,defines",
    [
        ("reader_t.cpp", ("GT_CHUNK", "GT_SRC_FACE")),
        ("mcast_reader_t.cpp", ("GT_SRC_FACE", "GT_SCRATCH_CB")),
        ("writer_t.cpp", ("GT_DST_FACE",)),
        ("mcast_writer_t.cpp", ("GT_DST_FACE",)),
        ("mcast_phase_t.h", ("GT_DST_FACE",)),
    ],
)
def test_tiny_kernels_exist_and_use_their_defines(name, defines):
    text = (KERNELS / name).read_text()
    for define in defines:
        assert define in text
        assert f'("{define}"' in GATE3_PY  # the host passes it


def test_shared_kernels_are_not_edited_for_the_flag():
    # the tiny variants are copies: the kernels other programs use carry no GT_ define
    for name in ("reader.cpp", "mcast_reader.cpp", "writer.cpp", "mcast_writer.cpp", "mcast_phase.h"):
        assert "GT_" not in (KERNELS / name).read_text()


def test_up_weight_cb_stays_whole_tiles_and_worker_cbs_are_tiny():
    assert "fp.cb_descriptor(9, BF16, TILE_BF16, BRANCHES * PARTIAL_TILES, w_set)" in GATE3_PY
    for cb in (0, 1, 2, 16, 17, 8, 10, 11, 12, 13, 14, 15, 18):
        assert f"fp.cb_descriptor({cb}, " in GATE3_PY
    assert GATE3_PY.count("shape_rows)") == 13  # every other CB takes the tile height
