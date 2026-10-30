# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic debug tests for distributed_rms_norm — DO NOT DELETE (documents the debugging process).

1. test_repeat_rows: 640x1792 (R3-i, one 2-row block per core, 2 waves of 1 tile-row): repeated calls must be
   bit-identical; on a mismatch, report which devices / tile-rows / tile-columns differ (wave 0 rows = even tile-rows
   of each row-group's pair, wave 1 rows = odd ones).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import _fabric_mesh_cache, mesh_device  # noqa: F401 (fixtures)

_spec = importlib.util.spec_from_file_location(
    "drms_acceptance", Path(__file__).with_name("test_distributed_rms_norm.py")
)
_acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acc)


@pytest.mark.parametrize("mesh_shape", [None])
@pytest.mark.parametrize("fp32", [True, False], ids=["fp32acc", "bf16acc"])
def test_repeat_rows(mesh_shape, fp32, mesh_device):
    axis = _acc.LINEAR_AXES[0]
    shape = (1, 1, 640, 1792)
    torch.manual_seed(0)
    rows, cols = tuple(mesh_device.shape)
    x = torch.randn((rows, cols, *shape)).to(torch.bfloat16)
    g = torch.randn((rows, cols, shape[-1])).to(torch.bfloat16)
    expected = _acc.reference(x, g, axis, 1e-6)
    tx = _acc.to_mesh(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tg = _acc.make_gamma(g, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    outs = []
    for i in range(4):
        o = _acc.distributed_rms_norm(
            tx, cluster_axis=axis, gamma=tg, epsilon=1e-6, compute_kernel_config=_acc.config(fp32)
        )
        outs.append(_acc.device_outputs(o, mesh_device))
    bad = []
    for i, got in enumerate(outs):
        for r in range(rows):
            for c in range(cols):
                a = got[r][c].float()
                e = expected[r, c].float()
                err = (a - e).abs().reshape(20, 32, 56, 32).amax(dim=(1, 3))  # [tile_row, tile_col]
                badtiles = (err > 0.25) | torch.isnan(err)
                if badtiles.any():
                    trs = sorted(set(badtiles.nonzero()[:, 0].tolist()))
                    tcs = sorted(set(badtiles.nonzero()[:, 1].tolist()))
                    bad.append(f"call {i} dev ({r},{c}): tile-rows {trs} tile-cols {tcs[:12]}...")
    assert not bad, "\n".join(bad)
