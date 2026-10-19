# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU only: the torch mirror of the TT shape SLat DiT decomposition (slat_flow_host.functional_forward) against
the fp32 golden calls, with tokens padded to a bucket and block-diagonal self-attention. Any gap here is a
decomposition, padding or host-table bug, not device numerics, so the bound is tight.
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.trellis2.reference import slat_flow_host as slat
from models.experimental.trellis2.reference import structure_flow_host as host
from models.experimental.trellis2.tests import goldens

CALLS = [0, 1, 18, 20]

pytestmark = pytest.mark.skipif(goldens.missing_reason() is not None, reason=goldens.missing_reason() or "")


@pytest.mark.parametrize("call", CALLS)
def test_functional_matches_golden(call):
    state = host.load_state_dict(f"{goldens.weights_dir()}/{slat.SHAPE_CKPT}")
    feats, coords, t, cond, gold, _ = goldens.shape_call_inputs(call)
    with torch.no_grad():
        out = slat.functional_forward(state, feats, coords, t, cond)
    assert goldens.pcc(out, gold) > 0.99999
    assert (out - gold).abs().max() / gold.abs().max() < 1e-4


def test_padding_does_not_leak():
    state = host.load_state_dict(f"{goldens.weights_dir()}/{slat.SHAPE_CKPT}")
    feats, coords, t, cond, _, _ = goldens.shape_call_inputs(0)
    n = coords.shape[0]
    with torch.no_grad():
        a = slat.functional_forward(state, feats, coords, t, cond, slat.bucket_len(n))
        b = slat.functional_forward(state, feats, coords, t, cond, slat.bucket_len(n) + 512)
    assert (a - b).abs().max() / a.abs().max() < 1e-5
