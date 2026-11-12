# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Refinement 4: cross-core W-split (regime R3).

Covers:
  - WIDTH_SHARDED / BLOCK_SHARDED (TILE resident shards, RM restrided shards with granule-wide slices),
  - interleaved logical W-split: the pinned predicate (R3 iff one whole row does not fit one core) on
    the `(1,1,32,32768)` case, and FORCE_W_SPLIT-pinned runs on small shapes (TILE / RM, partial W,
    TILE / RM gamma) so the interleaved R3 path is covered independent of the device's L1 size.
"""

import pytest
import torch
import ttnn

from eval.metrics import check_output
from eval.golden_tests.rms_norm.helpers import TOLERANCES
from eval.oom import oom_guard
from eval.sharding import auto_shard_config, shard_config

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
ML = ttnn.TensorMemoryLayout
TL, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT


def _ref(x, gamma, eps=1e-6):
    xf = x.float()
    out = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    if gamma is not None:
        out = out * gamma.float().reshape(-1)
    return out.to(x.dtype)


def _cfg(fp32_dest_acc_en=True):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = fp32_dest_acc_en
    return cfg


def _run(device, shape, dtype, layout, gamma_layout, memory_config, poison=None, fp32_dest_acc_en=True):
    torch.manual_seed(0)
    td = _TORCH_DTYPE[dtype]
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.float32).to(td)
    tt_x = tt_g = out = g = None
    try:
        with oom_guard("input shard"):
            tt_x = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
        if gamma_layout is not None:
            g = torch.randn(W, dtype=torch.float32).to(td)
            tt_g = ttnn.from_torch(
                g.reshape(1, 1, 1, W), dtype=dtype, layout=gamma_layout, device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )  # fmt: skip
        if poison is not None:
            tt_x = ttnn.fill_implicit_tile_padding(tt_x, poison)
            if tt_g is not None and gamma_layout == TL:
                tt_g = ttnn.fill_implicit_tile_padding(tt_g, poison)
        kwargs = {}
        if memory_config.memory_layout != ML.INTERLEAVED:
            kwargs["memory_config"] = tt_x.memory_config()
        with oom_guard("op"):
            out = rms_norm(tt_x, gamma=tt_g, compute_kernel_config=_cfg(fp32_dest_acc_en), **kwargs)
        assert out.memory_config().memory_layout == memory_config.memory_layout
        check_output(
            out, _ref(x, g), shape=list(shape), dtype=dtype, expected_layout=layout, tolerance=TOLERANCES[dtype]
        )
    finally:
        for t in (tt_x, tt_g, out):
            if t is not None:
                ttnn.deallocate(t)


# ---------------- sharded (WIDTH / BLOCK) ----------------

SHARDED_CASES = [
    # (shape, layout, gamma_layout, memory_layout)
    ((1, 1, 32, 2048), TL, TL, ML.WIDTH_SHARDED),  # golden loose: 8 slices... (auto)
    ((1, 1, 256, 512), TL, TL, ML.BLOCK_SHARDED),  # golden loose: 2-D grid
    ((1, 1, 32, 2848), TL, TL, ML.WIDTH_SHARDED),  # Wt=89 prime -> ragged last slice
    ((1, 1, 333, 1000), TL, None, ML.BLOCK_SHARDED),  # both non-aligned, partial last tile
    ((2, 1, 128, 100), TL, TL, ML.WIDTH_SHARDED),  # partial W, Wt=4
    ((1, 1, 32, 2848), RM, RM, ML.WIDTH_SHARDED),  # RM granule-wide slices (24 elems)
    ((1, 1, 333, 1000), RM, RM, ML.BLOCK_SHARDED),  # RM ragged sticks + slices
    ((4, 8, 32, 47), RM, None, ML.WIDTH_SHARDED),  # RM tiny slices, no gamma
]


@pytest.mark.parametrize("shape,layout,gamma_layout,memory_layout", SHARDED_CASES)
def test_w_split_sharded(device, shape, layout, gamma_layout, memory_layout):
    mc = auto_shard_config(list(shape), memory_layout, layout=layout, dtype=ttnn.bfloat16, device=device)
    _run(device, shape, ttnn.bfloat16, layout, gamma_layout, mc, fp32_dest_acc_en=False)


@pytest.mark.parametrize(
    "shape,shard,grid,memory_layout",
    [
        ((1, 1, 32, 7168), [32, 256], (7, 4), ML.WIDTH_SHARDED),  # perf: 28 cores, multi-row group
        ((1, 1, 640, 6144), [64, 704], (9, 10), ML.BLOCK_SHARDED),  # perf: ragged last slice
    ],
)
def test_w_split_pinned_geometry(device, shape, shard, grid, memory_layout):
    mc = shard_config(shard, grid, memory_layout, layout=TL, dtype=ttnn.bfloat16, device=device)
    _run(device, shape, ttnn.bfloat16, TL, TL, mc, fp32_dest_acc_en=False)


@pytest.mark.parametrize("memory_layout", [ML.WIDTH_SHARDED, ML.BLOCK_SHARDED])
def test_w_split_poisoned_padding(device, memory_layout):
    shape = (1, 1, 40, 72)
    mc = auto_shard_config(list(shape), memory_layout, layout=TL, dtype=ttnn.bfloat16, device=device)
    _run(device, shape, ttnn.bfloat16, TL, TL, mc, poison=1000.0, fp32_dest_acc_en=False)


# ---------------- interleaved logical W-split ----------------


def test_r3_predicate_pinned(device):
    """R1 residency (_r1_fits): one whole (1,1,32,32768) bf16 TILE + TILE gamma row does not fit a core,
    (1,1,32,16384) does. (Refinement 5 additionally takes R3 for few-row shapes on occupancy — see
    test_rms_norm_decode_perf.py::test_r3_occupancy_predicate.)"""

    def geom(W):
        x = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 32, W]), ttnn.bfloat16, TL, device, ttnn.DRAM_MEMORY_CONFIG
        )
        gm = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, 1, W]), ttnn.bfloat16, TL, device, ttnn.DRAM_MEMORY_CONFIG
        )
        g = pd._Geometry(x, gm, x, False)
        fits = pd._r1_fits(g, pd._cb_budget(device))
        ttnn.deallocate(x)
        ttnn.deallocate(gm)
        return fits

    assert not geom(32768)
    assert geom(16384)


def test_wide_interleaved_32768(device):
    _run(device, (1, 1, 32, 32768), ttnn.bfloat16, TL, TL, ttnn.DRAM_MEMORY_CONFIG, fp32_dest_acc_en=True)


FORCED_CASES = [
    ((1, 1, 32, 2048), TL, TL),
    ((1, 1, 96, 6144), TL, TL),  # 3 row-groups
    ((2, 1, 100, 1000), TL, RM),  # partial W + RM gamma, h non-aligned
    ((1, 1, 64, 4095), RM, RM),  # RM, partial last slice
    ((1, 1, 32, 200), TL, None),  # tiny W: Wt=7 slices of 1 tile
]


@pytest.mark.parametrize("shape,layout,gamma_layout", FORCED_CASES)
def test_w_split_interleaved_forced(device, monkeypatch, shape, layout, gamma_layout):
    monkeypatch.setattr(pd, "FORCE_W_SPLIT", True)
    _run(device, shape, ttnn.bfloat16, layout, gamma_layout, ttnn.DRAM_MEMORY_CONFIG, fp32_dest_acc_en=False)
