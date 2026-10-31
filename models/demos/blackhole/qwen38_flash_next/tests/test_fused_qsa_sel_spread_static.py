# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_QSA_SEL_SPREAD without a device: off by default, the core count knob, the segment cut (whole 16-column blocks,
every column once, in order), and the kernel and program wiring."""

from __future__ import annotations

import inspect

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import program as fp
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block


def test_flag_default_on_and_cores():
    assert qsa_block.SEL_SPREAD_ENV == "QWEN38_QSA_SEL_SPREAD"
    assert qsa_block.sel_spread_enabled({}) and qsa_block.sel_spread_enabled({qsa_block.SEL_SPREAD_ENV: "1"})
    assert not qsa_block.sel_spread_enabled({qsa_block.SEL_SPREAD_ENV: "0"})
    assert qsa_block.sel_spread_cores({}) == qsa_block.SEL_SPREAD_CORES
    assert qsa_block.sel_spread_cores({qsa_block.SEL_SPREAD_CORES_ENV: "65"}) == 65
    assert qsa_block.sel_spread_cores({qsa_block.SEL_SPREAD_CORES_ENV: "500"}) == 65
    assert qsa_block.sel_spread_cores({qsa_block.SEL_SPREAD_CORES_ENV: "0"}) == 1


def test_segments_cover_every_column_once():
    for cores in (1, 8, 16, 32, 65):
        segments = qsa_block.sel_spread_segments(cores)
        assert len(segments) == 2 * cores
        position = 0
        for first, count in segments:
            assert first == position and count > 0 and first % 16 == 0 and count % 16 == 0
            position += count
        assert position == qsa_block.SELECTION_WIDTH
        widths = {count for _, count in segments}
        assert max(widths) - min(widths) <= 16


def test_wiring():
    source = inspect.getsource(qsa_block.selection_row)
    assert "rows == 1 and page_table is None and sel_spread_enabled()" in source
    assert source.count("SEL_HALF") == 1 and "fp.reader_kernel, 0" in source and "fp.writer_kernel, 1" in source
    kernel = (fp.REPO_ROOT / qsa_block.SELECTION_ROW_SPREAD).read_text()
    assert "SEL_HALF" in kernel and "SEL_MAXC" in kernel and "PAGED" not in kernel
    assert "(ids_l1[i >> 2] << 2)" not in kernel and "const uint32_t id4 = ids_l1[i >> 2] << 2;" in kernel
