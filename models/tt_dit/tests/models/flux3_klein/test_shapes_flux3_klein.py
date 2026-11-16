# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the served output shapes, square and not, on one Galaxy (SP8) and on a quad (SP32). No device.

A non-square output is the same DiT sequence with an h x w token grid: row-major ids, an even split of the rows over
the SP devices, a swept matmul blocking for every per-device row count the heuristic would block too large for L1, and
ring SDPA chunks for every per-device row count of the quad.
"""

from __future__ import annotations

import pytest
import torch

from ....models.transformers.flux3_klein.transformer_block_flux3_klein import (
    HEADS,
    RING_SDPA_CHUNKS_BY_LOCAL_ROWS,
    RING_SDPA_CHUNKS_SP32_BY_LOCAL_ROWS,
)
from ....models.transformers.flux3_klein.transformer_flux3_klein import HIDDEN, Flux3KleinSequenceLayout
from ....pipelines.flux3_klein.packing import (
    IMAGE_ROPE_SCALE,
    TEXT_BUCKETS,
    TEXT_BUCKETS_BY_LAYOUT,
    image_ids,
    latent_shape,
    output_shape,
)
from ....pipelines.flux3_klein.pipeline_flux3_klein import REFERENCE_TOKEN_BUCKETS, SHAPES, SIZES
from ....utils import matmul

TP = 4
SP_FACTORS = (8, 32)
AGMM_GRID = (12, 9)
AGMM_N = (2304, 4608)
HEURISTIC_M_BLOCK_FITS = 8
SDPA_CORES = 12 * 9
SDPA_K_MAX = 736  # the largest k chunk the L1 model fits at q 128

NON_SQUARE = tuple(shape for shape in SHAPES if shape[0] != shape[1])


def test_shapes() -> None:
    assert SHAPES[: len(SIZES)] == tuple(output_shape(size) for size in SIZES)
    assert NON_SQUARE == ((4096, 2304), (2304, 4096))
    assert output_shape((4096, 2304)) == (4096, 2304)


@pytest.mark.parametrize("width, height", SHAPES)
def test_image_ids_are_row_major(width: int, height: int) -> None:
    h, w = latent_shape(height, width)
    ids = image_ids(h, w)
    assert ids.shape == (h * w, 4)
    rows, cols = torch.arange(h * w) // w, torch.arange(h * w) % w
    assert torch.equal(ids[:, 1], rows * IMAGE_ROPE_SCALE) and torch.equal(ids[:, 2], cols * IMAGE_ROPE_SCALE)
    assert not ids[:, 0].any() and not ids[:, 3].any()


def _layouts(sp: int, shapes=SHAPES):
    for width, height in shapes:
        grid = latent_shape(height, width)
        # every shape takes a reference of either token bucket
        for cond in (0, *REFERENCE_TOKEN_BUCKETS):
            for text in TEXT_BUCKETS_BY_LAYOUT.get(((width, height), int(cond > 0)), TEXT_BUCKETS):
                lengths = {"image": grid[0] * grid[1], "image_cond": cond, "text": text}
                yield pytest.param(
                    Flux3KleinSequenceLayout(sp, lengths, text), id=f"sp{sp}_{width}x{height}_r{cond}_t{text}"
                )


def _row_counts(layout: Flux3KleinSequenceLayout) -> set[int]:
    return {layout.total_padded // layout.sp_factor, *(layout.rows[m] for m in layout.modes)}


@pytest.mark.parametrize("layout", [p for sp in SP_FACTORS for p in _layouts(sp)])
def test_no_pad_rows_inside_the_image(layout: Flux3KleinSequenceLayout) -> None:
    assert layout.padded["image"] == layout.lengths["image"]
    assert layout.total_padded == layout.total


def test_quad_rows_of_the_non_square_shapes() -> None:
    rows = set().union(*(_row_counts(p.values[0]) for p in _layouts(32, NON_SQUARE)))
    assert rows == {32, 64, 128, 512, 1152, 1184, 1216, 1280, 1312, 1344, 1408, 1696, 1728, 1792}


@pytest.mark.parametrize("layout", [p for sp in SP_FACTORS for p in _layouts(sp)])
def test_matmul_blocking_is_swept_where_the_heuristic_is_too_large(layout: Flux3KleinSequenceLayout) -> None:
    table = matmul._grid_config_lookup[AGMM_GRID]
    for rows in _row_counts(layout):
        for n in AGMM_N:
            m_block = matmul._compute_heuristic_blocking(rows, HIDDEN, n, *AGMM_GRID, TP)[0]
            swept = (rows, HIDDEN, n) in table or matmul._same_m_per_core_match(table, rows, HIDDEN, n, AGMM_GRID[0])
            assert swept or m_block <= HEURISTIC_M_BLOCK_FITS, f"({rows}, {HIDDEN}, {n}): heuristic M block {m_block}"


@pytest.mark.parametrize("layout", list(_layouts(32)))
def test_quad_ring_sdpa_chunks_cover_every_row_count(layout: Flux3KleinSequenceLayout) -> None:
    for rows in _row_counts(layout):
        if rows < 1152:
            continue  # the 1024 and 2048 squares and the short early stacks: the small entries or the shared rule
        if rows in (2624, 2688):
            continue  # 4096x4096 + a 4 MP reference: more q units than one round; the base config (c80's quad demo)
        q, k = RING_SDPA_CHUNKS_SP32_BY_LOCAL_ROWS[rows]
        assert q == 128 and k % 32 == 0 and k <= SDPA_K_MAX
        assert (HEADS // TP) * -(-rows // q) <= SDPA_CORES  # one round of (head, q chunk) units
        padded = -(-rows // k) * k
        assert padded - rows <= 0.1 * rows, f"{rows} rows: k {k} pads {padded - rows} rows"


def test_galaxy_ring_sdpa_chunks_keep_the_4096_square() -> None:
    assert {8192, 8320, 8960} <= set(RING_SDPA_CHUNKS_BY_LOCAL_ROWS)


def test_ring_sdpa_device_test_knows_every_table_row() -> None:
    from . import test_ring_sdpa_flux3_klein as device_test

    rows = {(sp, r) for sp, table in device_test.CHUNKS.items() for r in table}
    assert rows <= set(device_test.TEXT_BUCKET_BY_ROWS)
    for sp, r in rows:
        bucket = device_test.TEXT_BUCKET_BY_ROWS[(sp, r)]
        assert r * sp - (bucket - device_test.VALID_TEXT if bucket else 0) > 0
