# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

# Refinement 5a debug: ROW_MAJOR + WIDTH/BLOCK_SHARDED cross-core reduction.
# DO NOT DELETE — documents the R5a debugging progression (cleanest case first).

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}
_TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED


def pytorch_rms_norm(x, gamma=None, epsilon=1e-6):
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + epsilon)
    out = xf / rms
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out.to(x.dtype)


def _cfg():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


def _run(shape, memory_layout, dtype, with_gamma, seed=0):
    torch.manual_seed(seed)
    tdt = _TORCH_DTYPE[dtype]
    torch_input = torch.randn(shape, dtype=tdt)
    gamma_t = torch.randn(shape[-1], dtype=tdt) if with_gamma else None
    mem_cfg = auto_shard_config(list(shape), memory_layout, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=DEVICE)
    ttnn_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEVICE, memory_config=mem_cfg
    )
    ttnn_gamma = (
        ttnn.from_torch(gamma_t.reshape(1, 1, 1, shape[-1]), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=DEVICE)
        if with_gamma
        else None
    )
    out = rms_norm(ttnn_input, gamma=ttnn_gamma, compute_kernel_config=_cfg(), memory_config=ttnn_input.memory_config())
    assert out.memory_config().memory_layout == memory_layout
    actual = ttnn.to_torch(out).reshape(shape)
    expected = pytorch_rms_norm(torch_input, gamma=gamma_t)
    assert_with_pcc(expected.to(torch.float32), actual.to(torch.float32), PCC[dtype])


DEVICE = None


@pytest.fixture(autouse=True)
def _capture_device(device):
    global DEVICE
    DEVICE = device
    yield


# Progression from cleanest (tile-aligned Ws, 1 tile-row, no gamma) to harder.
def test_width_clean_tilealigned(device):
    # (1,1,32,2048) RM WIDTH bf16: shard[32,32] -> Ws=32 (1 tile), Hs=32 (1 tile-row).
    _run((1, 1, 32, 2048), WIDTH, ttnn.bfloat16, with_gamma=False)


def test_width_subtile_w(device):
    # (1,1,32,64) RM WIDTH bf16: shard[32,8] -> sub-tile Ws=8.
    _run((1, 1, 32, 64), WIDTH, ttnn.bfloat16, with_gamma=False)


def test_block_subtile_hw(device):
    # (1,1,32,64) RM BLOCK bf16: shard[4,8] -> sub-tile H=4 AND sub-tile W=8.
    _run((1, 1, 32, 64), BLOCK, ttnn.bfloat16, with_gamma=False)


# Broad matrix over the golden RM WIDTH/BLOCK surface (excluding {WIDTH, w_non}
# and TILE gamma — both op-side EXCLUDED). RM gamma + no_gamma, bf16 + fp32.
WIDTH_SHAPES = [
    (1, 1, 32, 64),  # sub-tile Ws, 1 tile-row
    (2, 4, 128, 512),  # multi-tile-row (32), multi-image
    (1, 1, 17, 64),  # h_non (Hs=17)
    (1, 1, 50, 128),  # h_non (Hs=50 -> 2 tile-rows)
    (4, 8, 47, 256),  # h_non, larger flattened H
    (1, 1, 32, 4096),  # wide, tile-aligned Ws=64 (2 tiles)
    (1024, 1024),  # 2D, 32 tile-rows
    (1, 32, 128),  # 3D
]
BLOCK_SHAPES = WIDTH_SHAPES + [
    (1, 1, 64, 17),  # w_non (BLOCK only; rectangular)
    (128, 100),  # w_non (BLOCK only)
    (1, 1, 32, 8192),  # wide, sub-tile H=4, large local_Wt
]


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("with_gamma", [False, True])
@pytest.mark.parametrize("shape", WIDTH_SHAPES)
def test_width_matrix(device, shape, dtype, with_gamma):
    _run(shape, WIDTH, dtype, with_gamma)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("with_gamma", [False, True])
@pytest.mark.parametrize("shape", BLOCK_SHAPES)
def test_block_matrix(device, shape, dtype, with_gamma):
    _run(shape, BLOCK, dtype, with_gamma)
