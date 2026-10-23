# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4: non-tile-aligned W_local (the reader zeroes the last tile column's padding before the sum of
squares) and ROW_MAJOR gamma sticks (1, 1, ceil(W_local/32), 32).

The input's implicit tile padding is POISONED (ttnn.fill_implicit_tile_padding) so a pad column that leaks into the
sum of squares fails the check instead of silently adding zeros.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import create_ttnn_gamma, create_ttnn_input_tensor
from ttnn.operations.distributed_rms_norm import distributed_rms_norm

_spec = importlib.util.spec_from_file_location(
    "_drn_acceptance", Path(__file__).with_name("test_distributed_rms_norm.py")
)
_acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acc)
LINEAR_AXES, reference, pcc, PCC = _acc.LINEAR_AXES, _acc.reference, _acc.pcc, _acc.PCC

pytestmark = pytest.mark.skipif(not LINEAR_AXES, reason="no mesh axis of size >= 2")

PAD_POISON = 1000.0
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}


def run(mesh, shape, dtype, *, gamma_layout, gamma_dtype=ttnn.bfloat16, memory_config=None, poison=True):
    cluster_axis = LINEAR_AXES[0]
    torch.manual_seed(11)
    rows, cols = tuple(mesh.shape)
    x = torch.randn((rows, cols, *shape), dtype=torch.float32).to(TORCH_DTYPE[dtype])
    tx = create_ttnn_input_tensor(x, mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)
    if dtype == ttnn.bfloat8_b:
        x = ttnn.to_torch(ttnn.from_torch(x.reshape(-1, shape[-1]), dtype=dtype, layout=ttnn.TILE_LAYOUT))
        x = x.reshape(rows, cols, *shape)
    # bfloat8_b: a poisoned pad shares its face row's exponent with valid values and would crush their mantissas in
    # the INPUT itself (not an op error), so block-float inputs keep from_torch's zero padding.
    if poison and dtype != ttnn.bfloat8_b:
        tx = ttnn.fill_implicit_tile_padding(tx, PAD_POISON)
    gamma = tg = None
    if gamma_layout is not None:
        gamma = torch.randn((rows, cols, 1, 1, 1, shape[-1]), dtype=torch.float32).to(TORCH_DTYPE[gamma_dtype])
        tg = create_ttnn_gamma(gamma, mesh, dtype=gamma_dtype, layout=gamma_layout)
    out = distributed_rms_norm(tx, cluster_axis=cluster_axis, gamma=tg, epsilon=1e-6, memory_config=memory_config)
    expected = reference(x, None if gamma is None else gamma.reshape(rows, cols, shape[-1]), cluster_axis, 1e-6)
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        actual = ttnn.to_torch(t).to(torch.float32).reshape(shape)
        assert not torch.isnan(actual).any()
        got = pcc(actual, expected[r, c])
        assert got >= PCC[dtype], f"device ({r},{c}): pcc {got:.6f} < {PCC[dtype]}"


W_SHAPES = [(32, 17), (1, 1, 17, 50), (4, 128, 47), (1, 1, 64, 1000), (2, 1, 128, 100), (1, 1, 32, 120)]


@pytest.mark.parametrize(
    "gamma_layout", [None, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["none", "g_tile", "g_rm"]
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bf8b"])
@pytest.mark.parametrize("shape", W_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_w_non_aligned(shape, dtype, gamma_layout, mesh_device):
    run(mesh_device, shape, dtype, gamma_layout=gamma_layout)


@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, ttnn.float32], ids=["g_bf16", "g_fp32"])
@pytest.mark.parametrize(
    "shape", [(1, 1, 32, 1792), (1, 1, 640, 1792), (2, 4, 64, 256)], ids=lambda s: "x".join(map(str, s))
)
def test_row_major_gamma_aligned(shape, gamma_dtype, mesh_device):
    run(mesh_device, shape, ttnn.bfloat16, gamma_layout=ttnn.ROW_MAJOR_LAYOUT, gamma_dtype=gamma_dtype, poison=False)


@pytest.mark.parametrize("w,cores", [(120, (2, 1)), (240, (2, 1)), (120, (2, 2))])
@pytest.mark.parametrize("gamma_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["g_tile", "g_rm"])
def test_width_sharded_w_non_aligned(w, cores, gamma_layout, mesh_device):
    n = cores[0] * cores[1]
    shard_w = -(-w // (n * 32)) * 32
    mc = ttnn.create_sharded_memory_config(
        shape=(32, shard_w),
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores[0] - 1, cores[1] - 1))}),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )
    run(mesh_device, (1, 1, 32, w), ttnn.bfloat16, gamma_layout=gamma_layout, memory_config=mc)
