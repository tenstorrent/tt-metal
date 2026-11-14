# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the single-reference rule and the reference token layout. No device needed.

The canvas sizes are BFL's own (reference/src/flux3/pipeline.py reference_canvas, evaluated on these inputs). A reference
padded to its token bucket must put every masked row (text past text_valid, the bucket's pad, alignment) after every
attended row, and a reference that fills its bucket must keep the unpadded layout.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from ....models.transformers.flux3_klein.transformer_flux3_klein import MODES, Flux3KleinSequenceLayout
from ....pipelines.flux3_klein.pipeline_flux3_klein import (
    REFERENCE_TOKEN_BUCKETS,
    prepare_reference,
    reference_bucket,
    reference_canvas,
)

BFL_CANVAS = {
    (1024, 1024): ((1024, 1024), (1024, 1024)),
    (4000, 3000): ((2341, 1756), (2352, 1760)),
    (1600, 900): ((1600, 900), (1600, 912)),
    (900, 1600): ((900, 1600), (912, 1600)),
    (5000, 5000): ((2048, 2048), (2048, 2048)),
    (4096, 1024): ((4096, 1024), (4096, 1024)),
    (8000, 1000): ((4096, 512), (4096, 512)),
    (300, 200): ((300, 200), (304, 208)),
    (50, 40): ((50, 40), (64, 64)),
    (2048, 2048): ((2048, 2048), (2048, 2048)),
    (1366, 768): ((1366, 768), (1376, 768)),
    (3840, 2160): ((2717, 1528), (2720, 1536)),
}


@pytest.mark.parametrize("size", list(BFL_CANVAS))
def test_reference_canvas_matches_bfl(size):
    assert reference_canvas(*size) == BFL_CANVAS[size]


@pytest.mark.parametrize("size", [(1600, 900), (300, 200), (2048, 2048), (3840, 2160)])
def test_prepare_reference(size):
    torch.manual_seed(0)
    pixels = torch.randint(0, 256, (size[1], size[0], 3), dtype=torch.uint8).numpy()
    out = prepare_reference(Image.fromarray(pixels))
    fit, canvas = BFL_CANVAS[size]
    assert out.size == canvas
    tokens = (canvas[0] // 16) * (canvas[1] // 16)
    assert tokens <= REFERENCE_TOKEN_BUCKETS[-1]
    if fit == size and canvas != size:
        left, top = (canvas[0] - size[0]) // 2, (canvas[1] - size[1]) // 2
        inner = torch.from_numpy(np.asarray(out))[top : top + size[1], left : left + size[0]]
        assert torch.equal(inner, torch.from_numpy(pixels))


def test_reference_bucket(expect_error):
    assert reference_bucket(1) == REFERENCE_TOKEN_BUCKETS[0]
    assert reference_bucket(4096) == 4096
    assert reference_bucket(4097) == REFERENCE_TOKEN_BUCKETS[1]
    assert reference_bucket(16384) == 16384
    with expect_error(StopIteration, ""):
        reference_bucket(16385)


def _layout(cond_valid: int, cond_rows: int, text: int = 512, text_valid: int = 300, image: int = 4096, sp: int = 8):
    return Flux3KleinSequenceLayout(sp, {"image": image, "image_cond": cond_rows, "text": text}, text_valid, cond_valid)


@pytest.mark.parametrize("cond_valid, cond_rows", [(4096, 4096), (2584, 4096), (10625, 16384), (16384, 16384), (0, 0)])
def test_masked_rows_after_attended(cond_valid, cond_rows):
    layout = _layout(cond_valid, cond_rows)
    rows = layout.single_rows()
    assert len(rows) == layout.total
    attended = layout.attended()
    assert attended == 4096 + cond_valid + 300
    head = rows[:attended]
    assert sum(m == "image" for m, _ in head) == 4096
    assert sorted(i for m, i in head if m == "image_cond") == list(range(cond_valid))
    assert sorted(i for m, i in head if m == "text") == list(range(300))
    ids = layout.mode_ids()
    assert len(ids) == layout.total_padded
    assert ids[: layout.total].tolist() == [MODES.index(m) for m, _ in rows]
    assert len(layout.assembly_indices()) == layout.total_padded


def test_full_bucket_keeps_unpadded_layout():
    full = _layout(4096, 4096)
    legacy = Flux3KleinSequenceLayout(8, {"image": 4096, "image_cond": 4096, "text": 512}, 300)
    assert full.single_rows() == [(m, i) for m in legacy.modes for i in range(legacy.lengths[m])]
    assert torch.equal(full.assembly_indices(), legacy.assembly_indices())
    assert torch.equal(full.mode_ids(), legacy.mode_ids())
