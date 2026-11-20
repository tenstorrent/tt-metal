# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 debug: layer_norm precision with 16-bit DEST (bf16, fp32_dest_acc_en=False).

No gamma / beta, so the output is (x - mu) / sigma directly. Each output row is fitted as a * ref + b; the report
splits the error into a per-row SCALE (a - 1: the variance estimate) and a per-row OFFSET (b: the mean estimate),
plus the residual (element rounding). DO NOT DELETE — documents the Refinement 2 investigation.
"""

from __future__ import annotations

import pytest
import torch
import ttnn
from loguru import logger

from eval.golden_tests.all_gather_rms_norm.helpers import TOLERANCES, create_ttnn_input_tensor
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

from .test_all_gather_rms_norm import _DEVICE_PARAMS, EPS, make_input, torch_all_gather_norm

SHAPES = [
    (1, 1, 32, 384),
    (1, 1, 32, 1280),  # golden failing (k > 1 symmetric)
    (1, 1, 640, 32),  # k = 1
    (1, 1, 32, 4096),  # owner combine (k > 8)
    (1, 1, 64, 1000),  # w_non_aligned (dense assembly)
    (1, 1, 2048, 384),  # several tile-rows per member
]


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["mesh1x4"], indirect=True)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("row_offset", [0.0, 2.0], ids=["noofs", "ofs2"])
def test_ln16_decompose(mesh_device, shape, row_offset):
    cluster_axis = 1
    x = make_input(tuple(mesh_device.shape), shape, ttnn.bfloat16, seed=11, row_offset_scale=row_offset)
    expected = torch_all_gather_norm(x, cluster_axis, None, None, "layer_norm", EPS)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ckc = ttnn.ComputeConfigDescriptor()
    ckc.math_fidelity = ttnn.MathFidelity.HiFi4
    ckc.fp32_dest_acc_en = False
    ckc.math_approx_mode = False
    out = all_gather_rms_norm(
        tx,
        cluster_axis=cluster_axis,
        epsilon=EPS,
        topology=ttnn.Topology.Ring,
        num_links=2,
        norm_type="layer_norm",
        compute_kernel_config=ckc,
    )
    _, rms_t = TOLERANCES[(ttnn.bfloat16, False)]
    worst = 0.0
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        a = ttnn.to_torch(t).to(torch.float64).reshape(-1, expected.shape[-1])
        e = expected[0, idx].to(torch.float64).reshape(-1, expected.shape[-1])
        bad_rows = (~torch.isfinite(a)).any(-1)
        if bad_rows.any():
            tr = bad_rows.reshape(-1, 32).any(-1).nonzero().flatten().tolist()
            logger.info(f"LN16 dev{idx}: non-finite tile-rows {tr[:24]} ({len(tr)} of {a.shape[0] // 32})")
        rel_rms = float(((a - e) ** 2).mean().sqrt() / (e**2).mean().sqrt())
        em, am = e.mean(-1, keepdim=True), a.mean(-1, keepdim=True)
        scale = ((a - am) * (e - em)).sum(-1) / ((e - em) ** 2).sum(-1)
        offset = (am - scale[:, None] * em).squeeze(-1)
        resid = a - (scale[:, None] * e + offset[:, None])
        logger.info(
            f"LN16 {'x'.join(map(str, shape))} ofs={row_offset} dev{idx}: rel_rms={rel_rms:.5f} | "
            f"scale-1 mean={float(scale.mean() - 1):+.5f} std={float(scale.std()):.5f} | "
            f"offset mean={float(offset.mean()):+.5f} rms={float(offset.pow(2).mean().sqrt()):.5f} | "
            f"resid rms={float(resid.pow(2).mean().sqrt()):.5f}"
        )
        worst = max(worst, rel_rms)
    assert worst <= rms_t, f"rel_rms {worst} > {rms_t}"
