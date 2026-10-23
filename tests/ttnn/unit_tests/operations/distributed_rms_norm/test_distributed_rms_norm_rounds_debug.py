# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 3 debug (DO NOT DELETE): per-(tile row, tile column) error map of interleaved multi-round plans with a
ragged last W-chunk (the plans the auto-rounds solve picks for 2048x2048 / 2048x1280 1 link / 4096x1024 1 link)."""

import importlib.util
from pathlib import Path

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import create_ttnn_gamma, create_ttnn_input_tensor
from eval.golden_tests.distributed_rms_norm.topology import ring_mock_meshes
from ttnn.operations.distributed_rms_norm import distributed_rms_norm
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

_spec = importlib.util.spec_from_file_location("_drn_acc_rd", Path(__file__).with_name("test_distributed_rms_norm.py"))
_acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acc)

_RING = [(s, a) for s, a in ring_mock_meshes() if s == (1, 4)]


@pytest.mark.skipif(not _RING, reason="needs a 1x4 ring stand-in")
@pytest.mark.parametrize("mesh_shape,cluster_axis", [pytest.param(s, a, id="ring1x4") for s, a in _RING])
@pytest.mark.parametrize("min_rounds", [1, 2, 4, 8])
@pytest.mark.parametrize("shape,num_links", [((1, 1, 2048, 2048), 2), ((1, 1, 256, 224), 1)])
def test_rounds_map(shape, num_links, min_rounds, mesh_shape, cluster_axis, mesh_device, monkeypatch):
    monkeypatch.setattr(pd, "AUTO_ROUNDS", False)
    monkeypatch.setattr(pd, "MIN_ROUNDS", min_rounds)
    monkeypatch.setattr(pd, "MIN_BLOCK_W", 2)
    th, tw = shape[-2] // 32, shape[-1] // 32
    try:
        plan = pd.plan_blocks(th, tw, 106 if num_links == 2 else 108, 4, num_links, 2048, 2048, True)
    except RuntimeError:
        pytest.skip("no plan")
    print("PLAN", plan)
    rows, cols = tuple(mesh_device.shape)
    torch.manual_seed(5)
    x = torch.randn((rows, cols, *shape)).to(torch.bfloat16)
    gamma = torch.ones((rows, cols, 1, 1, 1, shape[-1])).to(torch.bfloat16)
    tg = create_ttnn_gamma(gamma, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tx = create_ttnn_input_tensor(x, mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    exp = _acc.reference(x, gamma.reshape(rows, cols, shape[-1]), cluster_axis, 1e-6)
    total = 0
    for call in range(3):
        out = distributed_rms_norm(
            tx, cluster_axis=cluster_axis, gamma=tg, topology=ttnn.Topology.Ring, num_links=num_links
        )
        for d in range(rows * cols):
            a = ttnn.to_torch(ttnn.get_device_tensors(out)[d]).float().reshape(th, 32, tw, 32)
            e = exp[d // cols, d % cols].reshape(th, 32, tw, 32)
            bad = ((a - e).abs().amax(dim=(1, 3)) > 0.1).int()  # (th, tw) tile map
            total += int(bad.sum())
            if int(bad.sum()):
                print(f"call {call} dev {d} BAD tiles", int(bad.sum()), "of", th * tw)
                print("BAD rows", bad.sum(1).tolist())
                print("BAD cols", bad.sum(0).tolist())
                if d == 0 and call == 0:
                    # which expected tile (row, col) does each bad output tile best match (scale-free: correlation)?
                    ef = e.permute(0, 2, 1, 3).reshape(th * tw, 32 * 32)
                    ef = (ef - ef.mean(1, keepdim=True)) / (ef.std(1, keepdim=True) + 1e-9)
                    for t, c in bad.nonzero().tolist()[:12]:
                        g = a[t, :, c, :].reshape(-1)
                        g = (g - g.mean()) / (g.std() + 1e-9)
                        corr = ef @ g / g.numel()
                        k = int(corr.argmax())
                        print(
                            f"  bad ({t},{c}) best match expected ({k // tw},{k % tw}) corr {float(corr[k]):.3f} "
                            f"ratio {float((a[t, :, c, :] / e[k // tw, :, k % tw, :]).median()):.3f}"
                        )
    assert total == 0
