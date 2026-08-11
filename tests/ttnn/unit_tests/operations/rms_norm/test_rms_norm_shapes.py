# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shape-resilience sweep for rms_norm — DO NOT DELETE.

The golden suite's `resilience` and `pad_poison` loose cases are pinned to
`fp32_dest_acc_en=False`, which is outside the Phase 0 SUPPORTED rectangle, so
they all xfail and never exercise the kernel.  Those shapes are precisely the
adversarial ones for this op's work split — prime tile counts that never divide
the grid, widths far past one core's L1, extreme aspect ratios, poisoned tile
padding — so this file replays them at the Phase 0 corner
(`fp32_dest_acc_en=True`), where they DO run.

What each block stresses:
  * `_RESILIENCE_SHAPES`  — the regime-selection function, the ragged hidden
    split (core_w < cb_w_tiles), the ragged row split, and num_blocks > 1.
  * `_PAD_POISON_SHAPES`  — the ragged-hidden-tile mask: a small W where one
    tile of padding is 11-38% of the row, filled with a loud finite value.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.rms_norm import rms_norm


PCC = 0.995
EPS = 1e-6


def _torch_rms_norm(x, gamma):
    x = x.float()
    out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return out * gamma.float().reshape(-1)


def _run(device, shape, layout, poison=None, pcc=PCC):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1,) * (len(shape) - 1) + (shape[-1],), dtype=torch.float32).to(torch.bfloat16)
    expected = _torch_rms_norm(torch_input, torch_gamma)

    tt_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device)
    if poison is not None:
        tt_input = ttnn.fill_implicit_tile_padding(tt_input, poison)
    tt_gamma = ttnn.from_torch(torch_gamma, dtype=ttnn.bfloat16, layout=layout, device=device)
    if poison is not None and layout == ttnn.TILE_LAYOUT:
        tt_gamma = ttnn.fill_implicit_tile_padding(tt_gamma, poison)

    tt_out = rms_norm(tt_input, gamma=tt_gamma, epsilon=EPS)
    assert tt_out.layout == tt_input.layout
    assert tuple(tt_out.shape) == tuple(shape)
    assert_with_pcc(expected, ttnn.to_torch(tt_out).float(), pcc)


# Mirrors eval/golden_tests/rms_norm/feature_spec.py::_RESILIENCE_SHAPES.
# Rt = prod(shape[:-1]) / 32 (tile-rows), Wt = ceil(W / 32).
_RESILIENCE_SHAPES = [
    # 4D tile-aligned, PRIME tile counts (grid never divides evenly)
    (1, 1, 224, 3072),  # Rt=7,   Wt=96
    (1, 1, 352, 2560),  # Rt=11,  Wt=80
    (1, 1, 416, 1184),  # Rt=13,  Wt=37   both prime
    (1, 1, 544, 736),  # Rt=17,  Wt=23
    (1, 1, 928, 1696),  # Rt=29,  Wt=53
    (1, 1, 1504, 544),  # Rt=47,  Wt=17
    (1, 1, 3104, 416),  # Rt=97,  Wt=13
    (1, 1, 3232, 96),  # Rt=101, Wt=3
    (1, 1, 4064, 160),  # Rt=127, Wt=5
    # few rows -> forced W-split / cross-core combine
    (1, 1, 32, 2848),  # Rt=1,  Wt=89 (prime)
    (1, 1, 32, 4064),  # Rt=1,  Wt=127
    (1, 1, 96, 6144),  # Rt=3,  Wt=192
    (1, 1, 160, 11008),  # Rt=5,  Wt=344
    # multi-batch -> composite Rt coprime-ish with the grid
    (3, 5, 224, 736),  # Rt=105, Wt=23
    (7, 1, 352, 1184),  # Rt=77,  Wt=37
    (5, 3, 928, 544),  # Rt=435, Wt=17
    # W non-aligned (large, ugly)
    (1, 1, 224, 1000),
    (1, 1, 352, 1023),
    (1, 1, 416, 1025),
    (1, 1, 544, 2047),
    (1, 1, 992, 3000),
    (3, 1, 736, 5119),
    (1, 1, 32, 4095),  # W-split + W-mask together
    # H non-aligned (large, ugly)
    (1, 1, 100, 736),
    (1, 1, 333, 544),
    (1, 1, 777, 1184),
    (1, 1, 1023, 416),
    (5, 1, 555, 736),
    # both non-aligned
    (1, 1, 333, 1000),
    (1, 1, 777, 2047),
    (3, 1, 555, 1025),
    # 3D
    (7, 224, 3072),
    (11, 352, 736),
    (3, 3104, 544),
    (5, 333, 1184),
    (13, 777, 1023),
    (1, 224, 11008),
    # 2D
    (7136, 736),
    (3104, 4064),
    (1500, 1000),
    (2047, 2047),
    (100, 5120),
    (99991, 64),  # extreme aspect: Rt=3125
]


@pytest.mark.parametrize("shape", _RESILIENCE_SHAPES, ids=lambda s: "x".join(str(d) for d in s))
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
def test_rms_norm_resilience_shapes(device, shape, layout):
    _run(device, shape, layout)


# Mirrors feature_spec.py::_PAD_POISON_SHAPES. TILE only — a ROW_MAJOR tensor
# has no implicit tile padding to poison.
_PAD_POISON_SHAPES = [
    (1, 1, 32, 40),  # Wt=2,  24 pad, 37.5% of the row
    (1, 1, 32, 72),  # Wt=3,  24 pad, 25.0%
    (1, 1, 32, 136),  # Wt=5,  24 pad, 15.0%
    (1, 1, 32, 200),  # Wt=7,  24 pad, 10.7%
    (1, 1, 224, 72),  # many rows x tiny W
    (1, 1, 40, 40),  # H and W both padded
]


@pytest.mark.parametrize("shape", _PAD_POISON_SHAPES, ids=lambda s: "x".join(str(d) for d in s))
def test_rms_norm_pad_poison(device, shape):
    _run(device, shape, ttnn.TILE_LAYOUT, poison=1000.0)
