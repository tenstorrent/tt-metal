# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU only: the channels-last torch mirror of the TT structure decoder (ss_decoder_host.functional_forward) against
the fp32 golden decoder call. Any gap here is a decomposition bug (channel order of the shuffle, padding, norm), not
device numerics, so the bounds are tight: the logits match to fp32 rounding and the voxel set is identical.
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.trellis2.reference import ss_decoder_host as dec
from models.experimental.trellis2.tests import goldens

pytestmark = pytest.mark.skipif(goldens.missing_reason() is not None, reason=goldens.missing_reason() or "")


def test_shuffle_matches_reference():
    """shuffle_weight_order + shuffle_ndhwc == the reference's pixel_shuffle_3d on a channels-first conv output."""
    torch.manual_seed(0)
    c_in, c, n = 6, 5, 3
    x = torch.randn(1, c_in, n, n + 1, n + 2)
    w, b = torch.randn(c * 8, c_in, 3, 3, 3), torch.randn(c * 8)
    y = torch.nn.functional.conv3d(x, w, b, padding=1)
    ref = (
        y.reshape(1, c, 2, 2, 2, *y.shape[2:])
        .permute(0, 1, 5, 2, 6, 3, 7, 4)
        .reshape(1, c, 2 * n, 2 * n + 2, 2 * n + 4)
    )
    w2, b2 = dec.shuffle_weight_order(w, b)
    y2 = torch.nn.functional.conv3d(x, w2, b2, padding=1).permute(0, 2, 3, 4, 1)
    out = dec.shuffle_ndhwc(y2).permute(0, 4, 1, 2, 3)
    assert torch.equal(out, ref)


def test_functional_matches_golden():
    state = dec.load_state_dict(f"{goldens.weights_dir()}/{dec.CKPT}")
    z, gold, coords = goldens.ss_decoder_call()
    with torch.no_grad():
        out = dec.functional_forward(state, z)
    assert goldens.pcc(out, gold) > 0.99999
    assert (out - gold).abs().max() / gold.abs().max() < 1e-4
    assert torch.equal(dec.occupancy_coords(out), coords)
