# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GGUF records are their weights' GGUF bytes, regrouped and never widened.

The Blackhole GGUF decoders read pages laid out by ``experimental/ckernel_gguf_page.h``; ``helpers/gguf_records.py``
reads that layout to build the records a reader turns into pages. A group's record is the codes of its pages and the
group's scale fields, so it holds the bits of 32 columns by ``32 * group_tiles`` rows of K, at the GGUF file's bits per
weight. The codebook formats add one level table each.

No hardware needed: the tests run the Python packer.
"""

import math

import numpy as np
import pytest
from helpers.gguf_records import FORMATS, level_table, random_blocks, records

LEVEL_TABLE_BYTES = {
    "IQ2_XXS": 512,
    "IQ2_XS": 1024,
    "IQ2_S": 2048,
    "IQ3_XXS": 512,
    "IQ3_S": 1024,
    "IQ1_S": 4096,
    "IQ1_M": 4096,
}


def test_formats_from_header():
    """FORMATS takes its ids and layouts from the header: one entry per format, ids 1 .. 25."""
    assert sorted(f.id for f in FORMATS.values()) == list(range(1, 26))


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
