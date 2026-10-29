# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 1 debug (DO NOT DELETE): the golden fp32 WIDTH_SHARDED 4096x1024 cell, per-device output inspection.

Run after test_distributed_rms_norm_l1_fallbacks.py in one session (precompile on) to reproduce an order-dependent
all-zero output. Finding: in that combination the DEVICE INPUT built by the golden helper (create_ttnn_input_tensor,
fp32 WIDTH_SHARDED) is already all zeros (host data nonzero) — the op is handed zeros and returns zeros. The same cell
passes alone, after any two of that file's three test groups, with --no-precompile, and inside the golden suite.
Not an op defect; kept to document the investigation."""

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.distributed_rms_norm import helpers as H
from ttnn.operations.distributed_rms_norm import distributed_rms_norm


@pytest.fixture(scope="module")
def mesh_cache():
    cache = _FabricMeshCache()
    yield cache
    cache.close()


@pytest.mark.parametrize("clear_resources", [False, True], ids=["keep_cache", "clear_cache"])
def test_ws_4096x1024_fp32_per_device(mesh_cache, clear_resources):
    import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd

    if clear_resources:
        print("R1DBG resources before clear:", [(k[0], k[1][:2], k[2], v[1].words) for k, v in pd._RESOURCES.items()])
        pd._RESOURCES.clear()
    mesh = mesh_cache.get()
    shape = (1, 1, 4096, 1024)
    x = H.quantize_like_device(H.random_stacked(mesh, shape, torch.float32, seed=0), ttnn.float32)
    gamma = H.random_stacked(mesh, (1, 1, 1, shape[-1]), torch.float32, seed=1)
    tg = H.create_ttnn_gamma(gamma, mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    expected = H.pytorch_distributed_rms_norm(x, gamma, 0, 1e-6)
    tx = H.create_ttnn_input_tensor(
        x, mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED
    )
    for idx, d in enumerate(ttnn.get_device_tensors(tx)):
        print(
            f"R1DBG input dev {idx} absmax={float(ttnn.to_torch(d).float().abs().max()):.4g} host_absmax={float(x.abs().max()):.4g}"
        )
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    bad = []
    for call in range(2):
        out = distributed_rms_norm(
            tx,
            cluster_axis=0,
            gamma=tg,
            epsilon=1e-6,
            topology=ttnn.Topology.Linear,
            num_links=1,
            compute_kernel_config=cfg,
            memory_config=tx.memory_config(),
        )
        bad += _inspect(out, expected, shape, call)
        out.deallocate()
    assert not bad, bad


def _inspect(out, expected, shape, call):
    rows, cols = expected.shape[0], expected.shape[1]
    bad = []
    for idx, d in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        got = ttnn.to_torch(d).float()
        exp = expected[r, c].float()
        zero_rows = int((got.reshape(-1, shape[-1]).abs().sum(-1) == 0).sum())
        err = float((got - exp).abs().max())
        print(f"R1DBG call {call} dev ({r},{c}) zero_rows={zero_rows}/{got.numel() // shape[-1]} max_err={err:.4g}")
        if err > 0.05:
            bad.append((call, r, c, zero_rows, err))
    return bad
