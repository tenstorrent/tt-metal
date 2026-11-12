# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 1: TILE-layout gamma (interleaved) + HEIGHT_SHARDED placement.

HEIGHT_SHARDED covers the three dataflow paths of the program descriptor:
  - TILE x, resident (cb_x / cb_out backed by the shards; ragged last shard via auto_shard_config),
  - RM x, resident (W % 32 == 0 and 32-stick multiple shards: cb_x_rm / cb_out_rm backed by the shards),
  - RM x, local-sticks (arbitrary stick counts / non-aligned W: restrided L1->L1 copies).
"""

import pytest
import torch
import ttnn

from eval.metrics import check_output
from eval.golden_tests.rms_norm.helpers import TOLERANCES
from eval.oom import oom_guard
from eval.sharding import auto_shard_config, shard_config

from ttnn.operations.rms_norm import rms_norm

_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def _ref(x, gamma, eps=1e-6):
    xf = x.float()
    out = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    if gamma is not None:
        out = out * gamma.float().reshape(-1)
    return out.to(x.dtype)


def _cfg():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    return cfg


def _run(device, shape, dtype, layout, gamma_layout, memory_config, gamma_dtype=None, poison=None):
    torch.manual_seed(0)
    td = _TORCH_DTYPE[dtype]
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.float32).to(td)
    with oom_guard("input shard"):  # infeasible shard geometry -> skip (same rule as the golden harness)
        tt_x = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    g = tt_g = None
    if gamma_layout is not None:
        gd = gamma_dtype or dtype
        g = torch.randn(W, dtype=torch.float32).to(_TORCH_DTYPE[gd])
        tt_g = ttnn.from_torch(
            g.reshape(1, 1, 1, W), dtype=gd, layout=gamma_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
    if poison is not None:
        if layout == ttnn.TILE_LAYOUT:
            tt_x = ttnn.fill_implicit_tile_padding(tt_x, poison)
        if tt_g is not None and gamma_layout == ttnn.TILE_LAYOUT:
            tt_g = ttnn.fill_implicit_tile_padding(tt_g, poison)
    kwargs = {}
    if memory_config.memory_layout == HS:
        kwargs["memory_config"] = tt_x.memory_config()
    with oom_guard("op"):
        out = rms_norm(tt_x, gamma=tt_g, compute_kernel_config=_cfg(), **kwargs)
    assert out.memory_config().memory_layout == memory_config.memory_layout
    check_output(out, _ref(x, g), shape=list(shape), dtype=dtype, expected_layout=layout, tolerance=TOLERANCES[dtype])


@pytest.mark.parametrize(
    "shape", [(1, 1, 32, 64), (1, 1, 17, 50), (2, 4, 64, 256), (1, 32, 8192), (1, 1, 64, 12288)], ids=str
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_tile_gamma_interleaved(device, shape, layout, dtype):
    if dtype == ttnn.float32 and shape[-1] > 8192:
        pytest.skip("fp32 W=12288 needs the W-split regime (Refinement 4)")
    _run(device, shape, dtype, layout, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)


def test_tile_gamma_poisoned_padding(device):
    _run(device, (1, 1, 40, 50), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG, poison=1e4)


def test_tile_gamma_mixed_dtype(device):
    _run(
        device,
        (1, 1, 64, 1024),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.TILE_LAYOUT,
        ttnn.DRAM_MEMORY_CONFIG,
        gamma_dtype=ttnn.float32,
    )


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 64), (1, 1, 17, 50), (4, 8, 47, 256), (1, 1, 2048, 256), (2, 1, 100, 47), (128, 8192), (1, 7 * 32, 96)],
    ids=str,
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("gamma_layout", [None, ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["nog", "rmg", "tg"])
def test_height_sharded_auto(device, shape, layout, dtype, gamma_layout):
    if layout == ttnn.ROW_MAJOR_LAYOUT and gamma_layout == ttnn.TILE_LAYOUT:
        pytest.skip("INVALID in feature_spec (RM x + sharded + TILE gamma)")
    mc = auto_shard_config(shape, HS, layout=layout, dtype=dtype, device=device)
    _run(device, shape, dtype, layout, gamma_layout, mc)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("gamma_layout", [None, ttnn.ROW_MAJOR_LAYOUT], ids=["nog", "rmg"])
def test_height_sharded_rm_resident(device, dtype, gamma_layout):
    # 32-stick-multiple shards with W % 32 == 0: cb_x_rm / cb_out_rm are the shards themselves.
    shape = (1, 1, 7 * 64 - 32, 512)  # 13 tile-rows over 7 cores, 64-stick shards, ragged last shard
    mc = shard_config([64, 512], (7, 1), HS, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype, device=device)
    _run(device, shape, dtype, ttnn.ROW_MAJOR_LAYOUT, gamma_layout, mc)


def test_height_sharded_default_output_inherits_spec(device):
    shape = (1, 1, 256, 128)
    mc = auto_shard_config(shape, HS, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    x = torch.randn(shape).bfloat16()
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    out = rms_norm(tt_x, compute_kernel_config=_cfg())
    assert out.memory_config().memory_layout == HS
    check_output(
        out,
        _ref(x, None),
        shape=list(shape),
        dtype=ttnn.bfloat16,
        expected_layout=ttnn.TILE_LAYOUT,
        tolerance=TOLERANCES[ttnn.bfloat16],
    )
