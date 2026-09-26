# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lock in the host-seam invariants the vision port is built on.

The tower reuses ``qwen36``'s kernels unchanged, which is only sound because of
three facts about PaddleOCR-VL specifically. Each is cheap to check on CPU and
expensive to discover the hard way at PCC time, so they are asserted here rather
than assumed:

1. The image processor emits patches in **raster** order, and the rotary
   positions are plain ``(row, col)`` to match. PaddleOCR merges 2x2 blocks in
   the projector rather than inside the encoder, so unlike Qwen it passes
   ``merge_size=1`` when building position ids. If a future checkpoint moved the
   merge into the encoder, this test fails instead of the model quietly
   regressing.

2. Our rotary tables reproduce ``PaddleOCRVisionRotaryEmbedding`` exactly.

3. Permuting tokens into merge-block order and then reshaping contiguously is
   *identical* to the projector's ``reshape(...).transpose(2, 3)`` gather. This
   is what lets ``PatchMerger`` be reused with no device-side gather.

No device required.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from models.demos.blackhole.paddleocr_vl.tt.vision.functional import (
    block_permutation,
    preprocess,
    raster_position_ids,
    vision_rope_tables,
)

MODEL_ID = "PaddlePaddle/PaddleOCR-VL-1.6"
HEAD_DIM = 72  # 1152 / 16
MERGE = 2


@pytest.fixture(scope="module")
def grid():
    return torch.tensor([[1, 32, 32]])


def test_processor_emits_raster_order():
    """A marked image must come back patch-by-patch in reading order."""
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    p, h, w = 14, 4, 4
    arr = np.zeros((h * p, w * p, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            arr[r * p : (r + 1) * p, c * p : (c + 1) * p, :] = (r * w + c) * 15 + 5

    out = proc.image_processor(
        images=[Image.fromarray(arr)], return_tensors="pt", do_resize=False, do_normalize=False, do_rescale=False
    )
    ids = [int(round((out["pixel_values"][i].float().mean().item() - 5) / 15)) for i in range(h * w)]
    assert ids == list(range(h * w)), f"processor order changed: {ids}"


def test_position_ids_match_transformers(grid):
    from transformers.vision_utils import get_vision_position_ids

    # merge_size=1 is the PaddleOCR convention; see modeling_paddleocr_vl.py:849.
    assert torch.equal(raster_position_ids(grid), get_vision_position_ids(grid, 1, kwargs={}))


def test_rope_tables_match_reference(grid):
    from transformers.models.paddleocr_vl.modeling_paddleocr_vl import PaddleOCRVisionRotaryEmbedding

    pos = raster_position_ids(grid)
    ref = PaddleOCRVisionRotaryEmbedding(HEAD_DIM // 2)(pos).repeat(1, 2)
    cos, sin = vision_rope_tables(pos, HEAD_DIM)

    assert cos.shape == (pos.shape[0], HEAD_DIM)
    torch.testing.assert_close(cos, ref.cos(), atol=0, rtol=0)
    torch.testing.assert_close(sin, ref.sin(), atol=0, rtol=0)


def test_block_permutation_is_a_bijection(grid):
    perm = block_permutation(grid, MERGE)
    n = int(grid.prod(dim=-1).sum())
    assert torch.equal(perm.sort().values, torch.arange(n))


def test_first_block_gathers_the_2x2_neighbourhood(grid):
    _, h, w = grid[0].tolist()
    perm = block_permutation(grid, MERGE)
    assert perm[:4].tolist() == [0, 1, w, w + 1]


def test_permute_then_reshape_equals_projector_merge(grid):
    """The load-bearing identity: no device-side gather is needed."""
    t, h, w = grid[0].tolist()
    d = 1152
    torch.manual_seed(0)
    tower = torch.randn(t * h * w, d)

    projector_order = (
        tower.reshape(t, h // MERGE, MERGE, w // MERGE, MERGE, d).transpose(2, 3).reshape(-1, MERGE**2 * d)
    )
    ours = tower[block_permutation(grid, MERGE)].reshape(-1, MERGE**2 * d)

    assert torch.equal(projector_order, ours)


def test_preprocess_pads_rotations_to_identity(grid):
    n = int(grid.prod(dim=-1).sum())
    bucket = 2048
    out = preprocess(grid, HEAD_DIM, MERGE, bucket=bucket)

    assert out["unpadded_len"] == n and out["seq_len"] == bucket
    assert out["cos"].shape == (bucket, HEAD_DIM)
    # Padded rows must be an identity rotation so they cannot perturb real tokens.
    assert (out["cos"][n:] == 1.0).all()
    assert (out["sin"][n:] == 0.0).all()
