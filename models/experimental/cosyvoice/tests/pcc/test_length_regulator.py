# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""InterpolateRegulator: arbitrary-length linear interpolation as a matmul."""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from models.experimental.cosyvoice.tt.common import GOLDEN_DIR, as_torch, load_golden, pcc
from models.experimental.cosyvoice.tt.flow.length_regulator import (
    TtInterpolateRegulator,
    linear_resample_matrix,
    torch_resample,
)
from models.experimental.cosyvoice.tt.weights import default_weights_path

needs_golden = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "flow.length_regulator.npz")),
    reason="run scripts/gen_golden.py in the CosyVoice venv first",
)


@pytest.mark.parametrize("t_in,t_out", [(20, 17), (134, 226), (174, 348), (164, 260), (50, 50), (7, 100)])
def test_resample_matrix_matches_f_interpolate(t_in, t_out):
    """The matrix form must equal F.interpolate(mode='linear', align_corners=False).

    ttnn.upsample cannot do this: it takes a scale factor, and these ratios are
    not integers (174->348 is, 164->260 is not). Expressing it as a constant
    [t_out, t_in] matrix with two non-zeros per row turns it into one matmul.
    """
    torch.manual_seed(0)
    x = torch.randn(1, t_in, 8)
    want = F.interpolate(x.transpose(1, 2).contiguous(), size=t_out, mode="linear").transpose(1, 2)
    got = torch_resample(x, t_out)
    assert got.shape == want.shape, (got.shape, want.shape)
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()


def test_resample_matrix_is_sparse_and_normalised():
    """Two non-zeros per row summing to 1 -- interpolation, not an arbitrary mix.
    A matrix that failed to sum to 1 would scale the signal invisibly."""
    m = linear_resample_matrix(37, 91)
    assert m.shape == (91, 37)
    assert torch.allclose(m.sum(dim=1), torch.ones(91), atol=1e-6)
    assert int((m > 0).sum(dim=1).max()) <= 2


def test_identity_when_lengths_match():
    m = linear_resample_matrix(64, 64)
    assert torch.equal(m, torch.eye(64))


@needs_golden
def test_head_mid_tail_split_reproduces_reference_length():
    """The reference resamples the first and last 20 tokens separately so a
    streaming chunk seam falls cleanly. Collapsing that into one resample would
    change the boundaries subtly and only in streaming -- exactly what the streaming
    test exists to catch -- so the split is preserved and its output length pinned here."""
    g = load_golden("flow.length_regulator")
    x1, x2 = as_torch(g["call0.in_x1"]), as_torch(g["call0.in_x2"])
    ml1, ml2 = int(g["call0.in_mel_len1"]), int(g["call0.in_mel_len2"])
    want = as_torch(g["call0.out_y"])

    got = TtInterpolateRegulator.torch_reference_resample(x1, x2, ml1, ml2)
    assert got.shape == want.shape, (got.shape, want.shape)
    assert got.shape[1] == ml1 + ml2


FLOW_WEIGHTS = default_weights_path().replace("hift_", "flow_")
needs_weights = pytest.mark.skipif(
    not os.path.exists(FLOW_WEIGHTS),
    reason="run scripts/export_weights.py --module flow in the CosyVoice venv first",
)


@needs_weights
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("t", [891, 1717, 1895])
def test_device_conv_stack_matches_torch(device, t):
    """The conv stack on device against torch, at flow lengths the goldens do not reach.

    The flow length includes the prompt, and cross-lingual's prompt is 1289 frames, so its
    utterances run this stack at 1700-2300 frames; zero-shot's run near 900 and the golden
    at 282. Prepared conv weights make the stack wrong on Blackhole at 1717 and 1895 and
    leave it exact at 891 (`docs/VALIDATION.md`), so it takes the op's own weight
    preparation.
    """
    import ttnn
    from models.experimental.cosyvoice.tt.weights import WeightBag

    bag = WeightBag.load(FLOW_WEIGHTS).sub("length_regulator")
    lr = TtInterpolateRegulator(device, bag, 80)
    x = torch.randn(1, t, 80, generator=torch.Generator().manual_seed(t)) * 5.0
    xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    got = ttnn.to_torch(lr(xd, t)).float()
    want = TtInterpolateRegulator.torch_reference_convs(bag, ttnn.to_torch(xd).float())
    p = pcc(got, want)
    print(f"\n  T={t}: conv stack PCC {p:.6f}")
    assert got.shape == want.shape, (got.shape, want.shape)
    assert p >= 0.999, p
