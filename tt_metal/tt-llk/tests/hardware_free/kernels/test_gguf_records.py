# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GGUF records are their weights' GGUF bytes, regrouped and never widened.

The Blackhole GGUF decoders read pages laid out by ``experimental/ckernel_gguf_page.h``; ``helpers/gguf_records.py``
mirrors that layout to build the records a reader turns into pages. A group's record is the codes of its pages and the
group's scale fields, so it holds the bits of 32 columns by ``32 * group_tiles`` rows of K, at the GGUF file's bits per
weight. The codebook formats add one level table each.

No hardware needed: the tests read the header and run the Python packer.
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest
from helpers.gguf_records import FORMATS, level_table, random_blocks, records

# tests/hardware_free/kernels -> hardware_free -> tests -> tt-llk
_PAGE_H = (
    Path(__file__).resolve().parents[3]
    / "tt_llk_blackhole/common/inc/experimental/ckernel_gguf_page.h"
)

LEVEL_TABLE_BYTES = {
    "IQ2_XXS": 512,
    "IQ2_XS": 1024,
    "IQ2_S": 2048,
    "IQ3_XXS": 512,
    "IQ3_S": 1024,
    "IQ1_S": 4096,
    "IQ1_M": 4096,
}


def test_layouts_match_header():
    text = _PAGE_H.read_text()
    ids = {name: int(v) for name, v in re.findall(r"^\s+(\w+)\s+= (\d+),$", text, re.M)}
    layouts = {
        name: tuple(int(a) for a in args.split(","))
        for name, args in re.findall(
            r"case (\w+):\s+return make_layout\(([^)]*)\);", text
        )
    }
    assert set(layouts) == set(FORMATS) == set(ids)
    for name, f in FORMATS.items():
        args = layouts[name] + (0,) * (7 - len(layouts[name]))
        python = (
            f.group_tiles,
            f.page_tiles,
            f.code_bytes,
            f.scale_rows,
            f.fp16_rows,
            f.code_offset,
            f.table_entries,
        )
        assert (ids[name], args) == (f.id, python), name


@pytest.mark.parametrize("name", list(FORMATS))
def test_record_is_gguf_bytes(name):
    f = FORMATS[name]
    group_weights = 32 * 32 * f.group_tiles
    gguf_bytes = group_weights * f.block_bytes // f.block_elems
    assert f.pages_per_group * f.code_bytes + 64 * f.scale_rows == gguf_bytes

    k = 2 * math.lcm(32 * f.group_tiles, f.block_elems)
    blocks = random_blocks(f, 64, k, np.random.default_rng(f.id))
    recs = records(f, blocks)
    assert {sum(map(len, group)) for tile in recs for group in tile} == {gguf_bytes}
    assert (
        sum(len(part) for tile in recs for group in tile for part in group)
        == blocks.nbytes
    )


def test_level_table_bytes():
    assert {name for name, f in FORMATS.items() if f.codebook} == set(LEVEL_TABLE_BYTES)
    for name, size in LEVEL_TABLE_BYTES.items():
        assert (
            level_table(FORMATS[name]).nbytes == size == 2 * FORMATS[name].table_entries
        ), name
