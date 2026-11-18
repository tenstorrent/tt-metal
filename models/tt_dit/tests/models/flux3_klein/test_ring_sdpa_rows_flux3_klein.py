# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only check of the ring SDPA chunk table's DP4 rows (transformer_block_flux3_klein.RING_SDPA_CHUNKS_BY_LOCAL_ROWS).
No device.

A DP4 group (SP4 x TP2) has no key in the shared ring SDPA table, so a row count missing from Klein's table gets a 256 k
chunk where SP8 x TP4 gets 512. Every image / joined row count a DP4 group runs at 4096 (each text and reference bucket)
must have an entry with a 512 k chunk, and none of them may be a row count SP8 x TP4 runs (the table is keyed by rows alone).
"""

from __future__ import annotations

import pytest

from ....models.transformers.flux3_klein.transformer_block_flux3_klein import RING_SDPA_CHUNKS_BY_LOCAL_ROWS
from ....models.transformers.flux3_klein.transformer_flux3_klein import Flux3KleinSequenceLayout
from ....pipelines.flux3_klein import pipeline_flux3_klein as pl


def _ring_rows(sp: int, size: int) -> set[int]:
    """Rows per device of the image early blocks and the joined single blocks, every reference and text bucket."""
    h, w = pl.latent_shape(size, size)
    rows = set()
    for bucket in (None, *pl.REFERENCE_TOKEN_BUCKETS):
        for text in pl.TEXT_BUCKETS_BY_LAYOUT.get((size, bucket or 0), pl.TEXT_BUCKETS):
            layout = Flux3KleinSequenceLayout(sp, {"image": h * w, "image_cond": bucket or 0, "text": text}, 1, 0)
            rows |= {layout.rows["image"], layout.total_padded // sp}
    return rows


def test_dp4_4096_rows_have_a_512_k_chunk():
    rows = _ring_rows(4, 4096)
    assert rows == {16384, 16640, 16896, 17408, 17920, 18432, 20736, 20992, 21504}
    for r in rows:
        assert RING_SDPA_CHUNKS_BY_LOCAL_ROWS.get(r, (0, 0))[1] == 512, r


@pytest.mark.parametrize("size", pl.SIZES)
def test_dp4_4096_entries_are_no_sp8_row_count(size):
    assert not _ring_rows(8, size) & _ring_rows(4, 4096)
