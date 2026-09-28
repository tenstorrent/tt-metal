# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The flat expert's on-device weights vs ttnn's bfp4 round trip of W: reads the down weights back, undoes the per-core
bank regions (flat_weight_regions' layout) and compares. From fp32 weights the upload rounds through bf16 first (~0.75%
of the elements one bfp4 step off the direct fp32 -> bfp4 rounding; equally close to fp32): a precision reference must
quantize bf16-valued weights (test_expert_precision.py does) or it charges that rounding to the op."""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.mimo_v2_d_p.tt.flat_expert import FlatRoutedExpert, _kd_of

H, I, M = 4096, 2048, 512


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_weight_quant(device):
    torch.manual_seed(0)
    E = 1
    W = [(torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02)]
    fe = FlatRoutedExpert(device, [W], m=M, H=H, I=I, gids=[[1]], n_global=4, pin=1)
    lay = fe.plan
    banks, nd = lay["banks"], lay["nd"]
    host = ttnn.to_torch(fe.w_d).float()  # [per * tiles * 32, banks * 32]: bank b = column block b
    it, ht = I // 32, H // 32
    per = -(-nd // banks)
    region_tiles = host.shape[0] // 32 // per
    got = torch.zeros(I, H)
    for d in range(nd):
        b, h_ = d % banks, d // banks
        col = host[:, b * 32 : (b + 1) * 32].reshape(-1, 32, 32)[h_ * region_tiles : (h_ + 1) * region_tiles]
        p_, c0 = lay["pcds"][d], lay["col0s"][d]
        k_ = _kd_of(p_, it)
        t = 0
        for c in range(it // k_):
            for kk in range(k_):
                for j in range(p_):
                    r0, cc = (c * k_ + kk) * 32, (c0 + j) * 32
                    got[r0 : r0 + 32, cc : cc + 32] = col[t]
                    t += 1
    qd = ttnn.to_torch(ttnn.from_torch(W[0][2], dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
    diff = (got - qd).abs()
    n_bad = int((diff > 0).sum())
    logger.info(
        f"down weights: {n_bad} of {got.numel()} elements differ from bfp4(Wd); max |diff| {float(diff.max()):.3g}, "
        f"rel {float((got - qd).norm() / qd.norm()):.4g}; vs fp32 Wd rel {float((got - W[0][2]).norm() / W[0][2].norm()):.4g}"
    )
    wd = W[0][2]
    q_bf16 = ttnn.to_torch(ttnn.from_torch(wd.bfloat16(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
    logger.info(
        f"vs fp32 Wd: reference bfp4 rel {float((qd - wd).norm() / wd.norm()):.4g}, flat's {float((got - wd).norm() / wd.norm()):.4g};"
        f" bfp4 via bf16 differs from the reference in {int(((q_bf16 - qd).abs() > 0).sum())} elements,"
        f" from flat's in {int(((q_bf16 - got).abs() > 0).sum())}"
    )
    assert torch.equal(got, q_bf16), "flat's weights are not the bfp4 of the bf16-rounded weights"
    assert (
        abs(float((got - wd).norm() - (qd - wd).norm()) / float(wd.norm())) < 1e-3
    ), "flat's rounding is less accurate"
