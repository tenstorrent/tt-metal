# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 2 debug (DO NOT DELETE): per-tile-row error map of a sharded case under forced knobs."""

import importlib.util
from pathlib import Path

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import create_ttnn_gamma, create_ttnn_input_tensor
from ttnn.operations.distributed_rms_norm import distributed_rms_norm
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

_spec = importlib.util.spec_from_file_location("_drn_acc", Path(__file__).with_name("test_distributed_rms_norm.py"))
_acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acc)


@pytest.mark.parametrize("use_gamma", [True, False])
def test_rowmap_width_pre_rounds(use_gamma, mesh_device, monkeypatch):
    monkeypatch.setattr(pd, "PRE_COMBINE", True)
    monkeypatch.setattr(pd, "MIN_ROUNDS", 3)
    shape = (1, 1, 200, 1312)
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(3)
    x = torch.randn((rows, cols, *shape)).to(torch.bfloat16)
    gamma = torch.ones((rows, cols, 1, 1, 1, shape[-1])).to(torch.bfloat16) if use_gamma else None
    tg = create_ttnn_gamma(gamma, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) if use_gamma else None
    tx = create_ttnn_input_tensor(
        x,
        mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    )
    out = distributed_rms_norm(tx, cluster_axis=_acc.LINEAR_AXES[0], gamma=tg, memory_config=tx.memory_config())
    exp = _acc.reference(x, None if gamma is None else gamma.reshape(rows, cols, shape[-1]), _acc.LINEAR_AXES[0], 1e-6)
    a = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(shape)
    e = exp[0, 0].reshape(shape)
    for t in range(-(-shape[-2] // 32)):
        sl = slice(t * 32, min(shape[-2], (t + 1) * 32))
        ratio = (a[..., sl, :] / e[..., sl, :]).median().item()
        err = (a[..., sl, :] - e[..., sl, :]).abs().max().item()
        print(f"tile row {t}: max err {err:.4f} median ratio {ratio:.4f}")
    assert _acc.pcc(a, e) > 0.995
