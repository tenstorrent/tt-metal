# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU only: the torch mirror of the TT DINOv3 decomposition (dinov3_host.functional_forward: unfolded patch matmul,
RoPE as a matrix over all tokens with identity prefix and padding rows, tokens padded to 512 with block-diagonal
attention) against the fp32 golden cond, from the golden's preprocessed image. Tight bounds: a gap here is a
decomposition bug, not numerics.
"""

from __future__ import annotations

import pytest
import torch
from PIL import Image

from models.experimental.trellis2.reference import dinov3_host as dino
from models.experimental.trellis2.tests import goldens

pytestmark = pytest.mark.skipif(goldens.missing_reason() is not None, reason=goldens.missing_reason() or "")


def test_rotate_half_matrix():
    x = torch.randn(3, dino.HEAD_DIM)
    h = dino.HEAD_DIM // 2
    assert torch.equal(x @ dino.rotate_half_matrix(), torch.cat([-x[:, h:], x[:, :h]], dim=-1))


def test_functional_matches_golden():
    state = dino.load_state_dict(f"{goldens.weights_dir()}/{dino.WEIGHTS}")
    pixels = dino.preprocess(Image.open(f"{goldens.golden_dir()}/preprocessed.png"), 512)
    gold = goldens.conditions()["cond"]
    with torch.no_grad():
        out = dino.functional_forward(state, pixels)
    assert out.shape == gold.shape
    assert goldens.pcc(out, gold) > 0.99999
    assert (out - gold).abs().max() / gold.abs().max() < 1e-4
