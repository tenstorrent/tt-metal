# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 6: the gamma broadcast (regime R4, knob GAMMA_MCAST, parked off by default — measured flat).

With the knob forced on, one worker per W-chunk column reads the chunk's gamma slivers and multicasts its cb_gamma
pages down its grid column (2-D placement: worker (rg, wc) on core (1 + wc, rg), ports in column 0). Pins the path for
TILE and ROW_MAJOR gamma, a ragged last chunk with non-aligned W, a 2-round plan, and back-to-back calls (the
Counter data-ready semaphore is re-armed by each receiver).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import ttnn

from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_acc = _load("_drn_acceptance_gmc", "test_distributed_rms_norm.py")
_na = _load("_drn_nonaligned_gmc", "test_distributed_rms_norm_nonaligned.py")
LINEAR_AXES = _acc.LINEAR_AXES

pytestmark = pytest.mark.skipif(not LINEAR_AXES, reason="no mesh axis of size >= 2")


@pytest.fixture
def gamma_mcast_on(monkeypatch):
    """Force the knob on and count the calls that really built the multicast family."""
    monkeypatch.setattr(pd, "GAMMA_MCAST", True)
    built = []
    real = pd.gamma_mcast_family

    def spy(*args):
        fam = real(*args)
        built.append(fam is not None)
        return fam

    monkeypatch.setattr(pd, "gamma_mcast_family", spy)
    return built


@pytest.mark.parametrize("gamma_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["g_tile", "g_rm"])
@pytest.mark.parametrize(
    "shape",
    # plans that fit the column box (<= grid.x - 1 chunks, <= grid.y row-slots): 640x1792 = 2 rounds over 10 row-slots;
    # 320x1000 = 7 chunks of 5 tiles, the last holding 2 (ragged) with W_local % 32 = 8 (non-aligned)
    [(1, 1, 640, 1792), (1, 1, 320, 1000)],
    ids=["640x1792_rounds", "320x1000_ragged_nonaligned"],
)
def test_gamma_mcast(shape, gamma_layout, gamma_mcast_on, mesh_device):
    _na.run(mesh_device, shape, ttnn.bfloat16, gamma_layout=gamma_layout)
    assert gamma_mcast_on and all(gamma_mcast_on), "the gamma multicast was not built for this plan"


def test_gamma_mcast_back_to_back(gamma_mcast_on, mesh_device):
    outs = [
        ttnn.to_torch(
            ttnn.get_device_tensors(
                _acc.run_case(mesh_device, (1, 1, 640, 1792), ttnn.bfloat16, LINEAR_AXES[0], use_gamma=True)
            )[0]
        )
        for _ in range(3)
    ]
    assert all(gamma_mcast_on)
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])
