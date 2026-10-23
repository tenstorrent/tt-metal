# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — sharded placements (HEIGHT -> R1, WIDTH / BLOCK -> R2 over the shard grid), zero-copy.

Covers TILE (x / out CBs backed on the shards) and ROW_MAJOR (local restride of the shard sticks), ragged shards,
non-aligned W / H (including poisoned tile padding), the pinned perf geometries of feature_spec._perf_case, the
multi-grid-row WIDTH W-group, and the placement-mismatch refusals.
"""

import pytest
import torch
import ttnn

from eval.sharding import auto_shard_config, shard_config
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.rms_norm import rms_norm

ML = ttnn.TensorMemoryLayout
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}
PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995, ttnn.bfloat8_b: 0.99}


def _ref(x, gamma, eps):
    xf = x.float()
    out = xf / torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return out * gamma.float().reshape(-1) if gamma is not None else out


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    assert torch.isfinite(b).all(), "non-finite output"
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def _run(
    device,
    shape,
    memory_layout,
    *,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    gamma_layout=ttnn.TILE_LAYOUT,
    gamma_dtype=ttnn.bfloat16,
    with_gamma=True,
    fp32_acc=True,
    fidelity=ttnn.MathFidelity.HiFi4,
    shard=None,
    poison=None,
    pass_memory_config=True,
    pcc=None,
):
    torch.manual_seed(0)
    x = torch.randn(shape).to(TORCH_DTYPE[dtype])
    if shard is not None:
        mc = shard_config(shard[0], shard[1], memory_layout, layout=layout, dtype=dtype, device=device)
    else:
        mc = auto_shard_config(list(shape), memory_layout, layout=layout, dtype=dtype, device=device)
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=mc)
    gamma = tg = None
    if with_gamma:
        gamma = torch.randn(shape[-1]).to(TORCH_DTYPE[gamma_dtype])
        tg = ttnn.from_torch(
            gamma.reshape(1, 1, 1, -1),
            dtype=gamma_dtype,
            layout=gamma_layout,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if poison is not None:
        tx = ttnn.fill_implicit_tile_padding(tx, poison)
        if tg is not None and gamma_layout == ttnn.TILE_LAYOUT:
            tg = ttnn.fill_implicit_tile_padding(tg, poison)
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = fidelity
    cfg.fp32_dest_acc_en = fp32_acc
    cfg.math_approx_mode = False
    kwargs = {"memory_config": tx.memory_config()} if pass_memory_config else {}
    out = rms_norm(tx, gamma=tg, epsilon=1e-6, compute_kernel_config=cfg, **kwargs)
    assert out.memory_config().memory_layout == memory_layout
    assert out.layout == layout
    got = ttnn.to_torch(out)
    assert list(got.shape) == list(shape)
    # reference from the device-visible (quantized) input
    xin = ttnn.to_torch(tx).float() if poison is None else x.float()
    expected = _ref(xin, gamma, 1e-6)
    p = _pcc(expected, got)
    thr = pcc if pcc is not None else PCC[dtype]
    assert p >= thr, f"PCC {p} < {thr}"
    rel = float((got.float() - expected).pow(2).mean().sqrt() / expected.std())
    assert rel < 0.05, f"rel rms err {rel}"


# --------------------------------------------------------------------------- TILE, auto (golden-style) geometry

TILE_SHAPES = [
    (1, 1, 256, 512),  # HEIGHT 8 cores; WIDTH; BLOCK
    (1, 1, 32, 2048),
    (4, 8, 47, 256),  # h_non_aligned, multi-batch
    (2, 1, 100, 47),  # both non-aligned
    (1, 1, 224, 3072),  # prime Rt
    (1, 1, 416, 1184),  # both prime tile counts
    (1, 1, 32, 4095),  # single row, W-mask
    (3104, 4064),  # Rt=97, Wt=127
]


@pytest.mark.parametrize("memory_layout", [ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED, ML.BLOCK_SHARDED], ids=["H", "W", "B"])
@pytest.mark.parametrize("shape", TILE_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_tile_auto(device, shape, memory_layout):
    _run(device, shape, memory_layout)


@pytest.mark.parametrize("memory_layout", [ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED, ML.BLOCK_SHARDED], ids=["H", "W", "B"])
@pytest.mark.parametrize(
    "shape", [(1, 1, 32, 72), (1, 1, 224, 72), (1, 1, 40, 40)], ids=lambda s: "x".join(map(str, s))
)
def test_tile_poisoned_padding(device, shape, memory_layout):
    _run(device, shape, memory_layout, poison=1000.0, fp32_acc=False)


@pytest.mark.parametrize("memory_layout", [ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED, ML.BLOCK_SHARDED], ids=["H", "W", "B"])
@pytest.mark.parametrize(
    "dtype,gamma_dtype,gamma_layout,fp32_acc",
    [
        (ttnn.float32, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, True),
        (ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, False),
        (ttnn.bfloat16, None, None, False),
    ],
    ids=["fp32_rmg", "bf8b_bf8bg", "bf16_nog"],
)
def test_tile_formats(device, memory_layout, dtype, gamma_dtype, gamma_layout, fp32_acc):
    _run(
        device,
        (1, 1, 256, 1024),
        memory_layout,
        dtype=dtype,
        gamma_dtype=gamma_dtype or ttnn.bfloat16,
        gamma_layout=gamma_layout or ttnn.TILE_LAYOUT,
        with_gamma=gamma_dtype is not None,
        fp32_acc=fp32_acc,
    )


# --------------------------------------------------------------------------- pinned perf geometries

PERF = [
    ((1, 1, 32, 1024), ML.WIDTH_SHARDED, ([32, 128], (8, 1))),
    ((1, 1, 32, 2304), ML.WIDTH_SHARDED, ([32, 256], (9, 1))),
    ((1, 1, 32, 5120), ML.WIDTH_SHARDED, ([32, 160], (8, 4))),
    ((1, 1, 32, 7168), ML.WIDTH_SHARDED, ([32, 256], (7, 4))),
    ((1, 1, 640, 6144), ML.BLOCK_SHARDED, ([64, 704], (9, 10))),
    ((1, 1, 640, 7168), ML.BLOCK_SHARDED, ([64, 1024], (7, 10))),
    ((1, 1, 7168, 1024), ML.BLOCK_SHARDED, ([736, 128], (8, 10))),
    ((1, 1, 7168, 2048), ML.BLOCK_SHARDED, ([736, 192], (11, 10))),
    ((1, 1, 8192, 1024), ML.BLOCK_SHARDED, ([1024, 128], (8, 8))),
]


@pytest.mark.parametrize(
    "shape,memory_layout,shard",
    PERF,
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) and isinstance(v[0], int) else None,
)
def test_perf_geometry(device, shape, memory_layout, shard):
    _run(device, shape, memory_layout, shard=shard, fp32_acc=False, fidelity=ttnn.MathFidelity.HiFi2, pcc=0.9995)


# --------------------------------------------------------------------------- ROW_MAJOR (gamma RM or absent)

RM_SHAPES = [
    (1, 1, 32, 50),
    (1, 1, 256, 512),
    (1, 1, 416, 1184),  # WIDTH shard 16 elements: half-tile slices
    (2, 1, 100, 47),
    (1, 1, 333, 1000),
    (1, 1, 32, 4064),
]


@pytest.mark.parametrize("with_gamma", [True, False], ids=["rmg", "nog"])
@pytest.mark.parametrize("memory_layout", [ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED, ML.BLOCK_SHARDED], ids=["H", "W", "B"])
@pytest.mark.parametrize("shape", RM_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rm_auto(device, shape, memory_layout, with_gamma):
    _run(
        device,
        shape,
        memory_layout,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
        with_gamma=with_gamma,
        fp32_acc=False,
    )


@pytest.mark.parametrize("memory_layout", [ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED, ML.BLOCK_SHARDED], ids=["H", "W", "B"])
@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16g", "fp32g"])
def test_rm_fp32_input(device, memory_layout, gamma_dtype):
    # fp32 RM shards are 4-element granular: a bf16 gamma slice starts 8 B into an alignment unit.
    _run(
        device,
        (1, 1, 64, 50),
        memory_layout,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.float32,
        gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
        gamma_dtype=gamma_dtype,
    )


# --------------------------------------------------------------------------- contract


def test_default_memory_config_inherits_shard(device):
    _run(device, (1, 1, 256, 512), ML.BLOCK_SHARDED, pass_memory_config=False)


def test_rejects_placement_mismatch(device, expect_error):
    x = torch.randn(1, 1, 256, 512).bfloat16()
    mc = auto_shard_config(
        [1, 1, 256, 512], ML.HEIGHT_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    with expect_error(UnsupportedAxisValue, "placement"):
        rms_norm(tx, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(UnsupportedAxisValue, "placement"):
        rms_norm(ti, memory_config=mc)


def test_rejects_rm_sharded_tile_gamma(device, expect_error):
    x = torch.randn(1, 1, 64, 64).bfloat16()
    mc = auto_shard_config(
        [1, 1, 64, 64], ML.HEIGHT_SHARDED, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    tg = ttnn.from_torch(
        torch.ones(1, 1, 1, 64).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with expect_error(ExcludedCell, "unsupported combination"):
        rms_norm(tx, gamma=tg)


# --------------------------------------------------------------------------- streamed gamma (sharded R1, wide W)


@pytest.mark.parametrize(
    "layout,gamma_layout",
    [
        (ttnn.TILE_LAYOUT, ttnn.TILE_LAYOUT),
        (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    ],
    ids=["tile_tg", "tile_rmg", "rm_rmg"],
)
@pytest.mark.parametrize("shape", [(1, 1, 160, 11008), (1, 1, 32, 8192)], ids=lambda s: "x".join(map(str, s)))
def test_height_wide_streamed_gamma(device, shape, layout, gamma_layout):
    # A HEIGHT shard pins the whole W row on one core; the resident gamma slice no longer fits next to the shards
    # (TILE) / the resident x block (RM), so gamma streams per chunk.
    _run(device, shape, ML.HEIGHT_SHARDED, layout=layout, gamma_layout=gamma_layout, fp32_acc=False)


# --------------------------------------------------------------------------- RM zero-copy (whole-tile shards)

RM_ZERO_COPY = [
    ((1, 1, 256, 512), ML.HEIGHT_SHARDED, ([32, 512], (8, 1))),
    ((1, 1, 100, 512), ML.HEIGHT_SHARDED, ([64, 512], (2, 1))),  # ragged last shard (36 of 64 sticks valid)
    ((1, 1, 64, 1024), ML.WIDTH_SHARDED, ([64, 128], (8, 1))),
    ((1, 1, 32, 5120), ML.WIDTH_SHARDED, ([32, 160], (8, 4))),  # shard 160 = 5 tiles; W-group spans 4 grid rows
    ((1, 1, 256, 512), ML.BLOCK_SHARDED, ([64, 128], (4, 4))),
    ((1, 1, 640, 6144), ML.BLOCK_SHARDED, ([64, 704], (9, 10))),  # ragged last W shard
]


@pytest.mark.parametrize("with_gamma", [True, False], ids=["rmg", "nog"])
@pytest.mark.parametrize(
    "shape,memory_layout,shard",
    RM_ZERO_COPY,
    ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) and isinstance(v[0], int) else None,
)
def test_rm_zero_copy(device, shape, memory_layout, shard, with_gamma):
    from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

    _run(
        device,
        shape,
        memory_layout,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
        with_gamma=with_gamma,
        fp32_acc=False,
        shard=shard,
    )
    # the selector must have taken the zero-copy stick CBs for these whole-tile shards
    mc = shard_config(
        shard[0], shard[1], memory_layout, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    tx = ttnn.from_torch(
        torch.zeros(shape).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mc,
    )
    geo = pd._geometry(tx, tx, None, False)
    sp = pd._shard_placement(tx, geo)
    assert pd._rm_shard_zero_copy_ok(geo, sp)
    geo2, _ = pd._select_sharded_blocking(geo, sp, pd._l1_budget_bytes(device))
    assert geo2.zero_copy
