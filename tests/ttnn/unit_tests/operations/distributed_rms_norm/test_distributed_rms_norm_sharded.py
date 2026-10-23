# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2: HEIGHT_SHARDED / WIDTH_SHARDED input + output, consumed natively (the shard backs cb_x_block,
the output shard backs cb_out_tiles). Shard specs come from the golden suite's auto_shard_config (row-major from
core (0, 0), Split.PAD), so these cover the same geometries as the golden cells:

* HEIGHT: one row-slot per shard, multi-tile-row shards, ragged last shard, wide rows (streamed gamma);
* WIDTH: one chunk per shard, tall shards (pre-combine on the leader), a padded last shard (8192 wide);
* every mode knob forced on small shapes (PRE_COMBINE, GAMMA_STREAM, MIN_ROUNDS), plus back-to-back bit identity.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import create_ttnn_gamma, create_ttnn_input_tensor
from ttnn.operations.distributed_rms_norm import distributed_rms_norm
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

_spec = importlib.util.spec_from_file_location(
    "_drn_acceptance", Path(__file__).with_name("test_distributed_rms_norm.py")
)
_acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acc)
LINEAR_AXES, reference, pcc, PCC, TORCH_DTYPE = _acc.LINEAR_AXES, _acc.reference, _acc.pcc, _acc.PCC, _acc.TORCH_DTYPE

pytestmark = pytest.mark.skipif(not LINEAR_AXES, reason="no mesh axis of size >= 2")

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED


def run_sharded(mesh, shape, dtype, layout, cluster_axis, *, use_gamma, gamma_dtype=ttnn.bfloat16, eps=1e-6):
    torch.manual_seed(7)
    rows, cols = tuple(mesh.shape)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(TORCH_DTYPE[dtype])
    gamma = tg = None
    if use_gamma:
        gamma = torch.randn((rows, cols, 1, 1, 1, shape[-1]), dtype=torch.float32).to(TORCH_DTYPE[gamma_dtype])
        tg = create_ttnn_gamma(gamma, mesh, dtype=gamma_dtype, layout=ttnn.TILE_LAYOUT)
    tx = create_ttnn_input_tensor(x, mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_layout=layout)
    out = distributed_rms_norm(tx, cluster_axis=cluster_axis, gamma=tg, epsilon=eps, memory_config=tx.memory_config())
    assert out.memory_config() == tx.memory_config()
    assert list(out.shape) == list(shape) and out.dtype == dtype
    expected = reference(x, None if gamma is None else gamma.reshape(rows, cols, shape[-1]), cluster_axis, eps)
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        actual = ttnn.to_torch(t).to(torch.float32).reshape(shape)
        assert not torch.isnan(actual).any(), f"device ({r},{c}) produced NaN"
        got = pcc(actual, expected[r, c])
        assert got >= PCC[dtype], f"device ({r},{c}): pcc {got:.6f} < {PCC[dtype]}"
    return out


CASES = [
    # HEIGHT: 20 one-row shards; 64 two-row shards; 4-row ragged (100 rows); rank 4 batched; wide rows -> streamed gamma
    (HS, (1, 1, 640, 1792)),
    (HS, (1, 1, 4096, 1024)),
    (HS, (1, 1, 100, 512)),
    (HS, (2, 4, 128, 512)),
    (HS, (1, 1, 128, 4096)),
    (HS, (1, 1, 32, 8192)),
    # WIDTH: decode (56 one-tile shards); tall (board needs the pre-combine); padded last shard; batched
    (WS, (1, 1, 32, 1792)),
    (WS, (1, 1, 640, 1792)),
    (WS, (1, 1, 4096, 1024)),
    (WS, (1, 1, 32, 8192)),
    (WS, (2, 4, 128, 512)),
]


@pytest.mark.parametrize("use_gamma", [True, False], ids=["gamma", "no_gamma"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize(
    "layout,shape", CASES, ids=[f"{'H' if l == HS else 'W'}-{'x'.join(map(str, s))}" for l, s in CASES]
)
def test_sharded(layout, shape, dtype, use_gamma, mesh_device):
    if dtype == ttnn.float32 and layout == HS and shape[-1] >= 8192:
        pytest.skip("fp32 x + out shards (2 x 1 MiB) do not fit one core's L1 (harness-side infeasible)")
    run_sharded(mesh_device, shape, dtype, layout, LINEAR_AXES[0], use_gamma=use_gamma, gamma_dtype=ttnn.float32)


@pytest.mark.parametrize("use_gamma", [True, False], ids=["gamma", "no_gamma"])
@pytest.mark.parametrize(
    "layout,shape,knobs",
    [
        (WS, (1, 1, 64, 1792), dict(PRE_COMBINE=True)),
        (WS, (1, 1, 200, 1312), dict(PRE_COMBINE=True, MIN_ROUNDS=3)),
        (WS, (1, 1, 200, 1312), dict(PRE_COMBINE=False, MIN_ROUNDS=3)),
        (HS, (1, 1, 640, 1792), dict(GAMMA_STREAM=True)),
        (HS, (1, 1, 4096, 1024), dict(MIN_ROUNDS=2)),
        (HS, (1, 1, 4096, 1024), dict(MIN_ROUNDS=2, GAMMA_STREAM=True)),
        (HS, (1, 1, 4000, 512), dict(MIN_ROUNDS=2)),
    ],
    ids=["W-pre", "W-pre-3r", "W-nopre-3r", "H-stream", "H-2r", "H-2r-stream", "H-2r-ragged"],
)
def test_sharded_forced_modes(layout, shape, knobs, use_gamma, mesh_device, monkeypatch):
    for k, v in knobs.items():
        monkeypatch.setattr(pd, k, v)
    run_sharded(mesh_device, shape, ttnn.bfloat16, layout, LINEAR_AXES[0], use_gamma=use_gamma)


@pytest.mark.parametrize("layout", [HS, WS], ids=["H", "W"])
def test_sharded_back_to_back(layout, mesh_device):
    """The pre-combine / board counters are re-armed per call: three calls agree bit for bit."""
    outs = [
        ttnn.to_torch(
            ttnn.get_device_tensors(
                run_sharded(mesh_device, (1, 1, 640, 1792), ttnn.bfloat16, layout, LINEAR_AXES[0], use_gamma=True)
            )[0]
        )
        for _ in range(3)
    ]
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])
