# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""WHOLE-OP A/B for the scoped+fused finalize, on a private COPY of rms_norm.

`whole_op/plan.py` + `whole_op/kernels/` are a copy of the shipped op with two
knobs added (see plan.py's header): `FINALIZE_VARIANT` selects the finalize
implementation, `ABLATE_BITS` can stub stage payloads.  The real op's files are
never touched.

WHAT THIS ADDS OVER micro.py
    micro.py prices the finalize in isolation (pure SFPU ns on the MATH thread).
    This measures what that is worth on the WALL, at the focus case's exact
    geometry, and -- via the finalize-only ablation -- splits the owner combine's
    measured cost into "reduce math" vs "finalize".

    It is also the correctness proof for the SCOPE: the whole-op PCC vs a torch
    fp32 golden is what establishes that the ~31/32 lanes the candidate no longer
    computes are genuinely never read (by the `BroadcastDim::Col` consumer, and by
    the funnel `noc_async_write` that ships the whole 4 KB finalized tile to the
    root).

Metric: `DEVICE KERNEL DURATION [ns]` from the Tracy per-op CSV.  ONE fresh run
per point -- device kernel time has no warm-up transient.

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/finalize_sfpu_scope/test_whole_op.py

MEASURED — Blackhole p150b @1350 MHz, device kernel ns, one fresh run per point.
The focus row was measured twice (34485 / 34579, i.e. 0.27% run-to-run).

  PINNED SHARDED GEOMETRIES, target config (bf16, HiFi2, fp32_dest_acc_en=False)
    case                     stock   cskip  nonneg   fusRC   noFin  noComb    win      x
    focus_bshard_8192x1024   34579   31517   31508   34149   30586   29044   3062  1.097
    wshard_32x5120            6448    5671    5640    6324    5431    4667    777  1.137
    wshard_32x7168            6416    5654    5651    6345    5427    4688    762  1.135

  THE SPLIT the coordinator asked for, from the two ablations on the focus case:
    owner combine total (stock - no_combine) = 5535 ns
      of which FINALIZE  (stock - no_finalize) = 3993 ns   (72%)
      of which reduce math                     = 1542 ns   (28%)
    The finalize -- not the cross-core reduce -- IS the owner combine.
    After the candidate only 931 ns of that 3993 remains.

  The isolated SFPU number maps 1:1 onto the wall, which is what "serial critical
  path" means quantitatively: the focus case runs 4 finalizes per owner core
  (2 blocks x own_rows 2); 4 x 989.7 ns = 3959 predicted vs 3993 measured, and
  4 x (989.7 - 240.1) = 2998 predicted saving vs 3062 measured.

  s == 1 `cp_collapse` (HEIGHT-sharded (1,1,32768,256), block_rows forced).  Here
  EVERY core finalizes BLOCK_ROWS tiles per block, so the finalize is a much
  larger share -- and this is the ReduceTile datapath, whose `reduce_init` is
  hoisted ONCE outside the per-output loop:
    B= 1  stock=41110  cskip=29799  win=11311  1.380x
    B= 2  stock=33553  cskip=22713  win=10840  1.477x
    B= 4  stock=30648  cskip=18547  win=12101  1.652x
    B= 8  stock=29198  cskip=17113  win=12085  1.706x
    B=16  stock=28444  cskip=16379  win=12065  1.737x
  (16 finalizes either way -- 16 blocks x 1 or 1 block x 16 -- and the saving is
  flat at ~12.0 us = 16 x 749.6 ns.  The model holds exactly.)

  fp32_dest_acc_en=True + HiFi4 (the op's own Phase-0 DEFAULT config):
    focus_fp32acc             stock=48038  cskip=45088  noFin=44110  win= 2950 1.065x
    collapse_s1_B16_fp32acc   stock=42154  cskip=30367  noFin=26784  win=11787 1.388x

  INTERLEAVED domain sweep, (1,1,2048,*) -- DRAM-bandwidth-bound, so the finalize
  is diluted and the wins are small but never negative beyond noise:
    il_bf16_tile           29580 -> 28412  1.041x
    il_bf16_fp32acc_hifi4  29841 -> 29904  0.998x   (flat: -0.2%, inside noise)
    il_fp32_fp32acc        51001 -> 50929  1.001x   (flat)
    il_bfp8_fp32acc        24162 -> 23546  1.026x
    il_bf16_rowmajor       35831 -> 35444  1.011x
    il_bf16_w_nonaligned   29893 -> 28716  1.041x
    il_bf16_no_gamma       27120 -> 26856  1.010x
    il_bf16_h_nonaligned   28730 -> 27907  1.029x

CORRECTNESS (the real deliverable of the sweep): every case above passes the
0.9995 gate, and the candidate's PCC is IDENTICAL to the baseline's on every
sharded geometry and every domain cell but two, where it differs in the 6th
decimal (wshard_32x7168 0.9999889 -> 0.9999871; il_bf16_w_nonaligned 0.9999772 ->
0.9999726) -- both shapes where 1/W is not a power of two, so the fp32 intermediate
is the only thing that moved.  `fused_rc` (fusion WITHOUT the scope) reports the
same PCC as `fused_cskip` everywhere, which isolates that difference to the fusion
and proves the SCOPE is bit-neutral: the ~31/32 lanes the candidate stops
computing are genuinely never read, neither by the `BroadcastDim::Col` consumer
nor through the funnel `noc_async_write` that ships the whole 4 KB tile to the root.
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from eval.sharding import shard_config

from tests.ttnn.unit_tests.operations.rms_norm.perf_experiments.finalize_sfpu_scope.whole_op.plan import (
    create_program_descriptor,
)

PLAN_GLOBALS = create_program_descriptor.__globals__

_ML = ttnn.TensorMemoryLayout

# The perf group's FIXED user config.  Never a lever -- identical for every variant.
TARGET_FIDELITY = ttnn.MathFidelity.HiFi2
TARGET_FP32_ACC = False
EPSILON = 1e-6
PCC_GATE = 0.9995  # the focus case's soft precision gate

# finalize variants (-DRMS_FINALIZE_VARIANT); see whole_op/kernels/finalize_scoped.hpp
V_STOCK = 0  # BASELINE: 3 full-tile SFPU walks + 2 inits   [96 vectors]
V_FUSED_CSKIP = 1  # CANDIDATE: 1 fused even-parity walk + 1 init  [8 vectors]
V_FUSED_CSKIP_NONNEG = 2  # + negative-input NaN guard dropped           [8 vectors]
V_FUSED_RC = 3  # fusion without the scope                     [32 vectors]

ABL_FINALIZE = 8  # stub ONLY the finalize (wrong answers; perf split only)
ABL_COMBINE = 1  # stub the whole owner combine (reduce + finalize)

VARIANTS = {
    "stock": (V_STOCK, 0),
    "fused_cskip": (V_FUSED_CSKIP, 0),
    "fused_cskip_nonneg": (V_FUSED_CSKIP_NONNEG, 0),
    "fused_rc": (V_FUSED_RC, 0),
    "no_finalize": (V_STOCK, ABL_FINALIZE),  # ABLATION -- wrong answers
    "no_combine": (V_STOCK, ABL_COMBINE),  # ABLATION -- wrong answers
}
ABLATIONS = ("no_finalize", "no_combine")

# ---------------------------------------------------------------------------
# Geometries.  The focus case first, then the regimes where the finalize is a
# LARGER share of the wall (latency-bound single-tile-row decode) and the s == 1
# `cp_collapse` path, which finalizes BLOCK_ROWS tiles per reduce call.
#   (id, shape, memory_layout, shard_shape, core_grid, block_rows_force)
# ---------------------------------------------------------------------------
GEOMETRIES = [
    ("focus_bshard_8192x1024", (1, 1, 8192, 1024), _ML.BLOCK_SHARDED, [1024, 128], (8, 8), 0),
    ("wshard_32x5120", (1, 1, 32, 5120), _ML.WIDTH_SHARDED, [32, 160], (8, 4), 0),
    ("wshard_32x7168", (1, 1, 32, 7168), _ML.WIDTH_SHARDED, [32, 256], (7, 4), 0),
]

# s == 1 (`cp_collapse`) sweep: one HEIGHT-sharded tensor, block_rows forced.  The
# shard spans the whole hidden axis, so num_hidden_slices == 1 and every core runs
# the collapse reduce over (BLOCK_ROWS, 1) with BLOCK_ROWS finalizes per call.
COLLAPSE_SHAPE = (1, 1, 32768, 256)
COLLAPSE_SHARD = [512, 256]  # 64 shards of 16 tile-rows over an 8x8 grid
COLLAPSE_GRID = (8, 8)
COLLAPSE_BLOCK_ROWS = [1, 2, 4, 8, 16]


def _compute_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=TARGET_FIDELITY,
        fp32_dest_acc_en=TARGET_FP32_ACC,
        math_approx_mode=False,
    )


def _run(device, torch_x, torch_gamma, memory_config):
    """The copied op, invoked exactly the way rms_norm() invokes the shipped one."""
    x = ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = ttnn.allocate_tensor_on_device(
        x.shape, x.dtype, x.layout, x.device(), x.memory_config()
    )
    descriptor = create_program_descriptor(
        x, out, gamma=gamma, epsilon=EPSILON, compute_kernel_config=_compute_config()
    )
    return ttnn.to_torch(ttnn.generic_op([x, gamma, out], descriptor)).to(torch.float32)


def _pcc(got, torch_x, torch_gamma):
    xf = torch_x.to(torch.float32)
    expected = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPSILON)
    expected = expected * torch_gamma.to(torch.float32).reshape(-1)
    a, b = got.flatten(), expected.flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _cases():
    """(case_id, shape, memory_layout, shard_shape, core_grid, block_rows) x variant."""
    out = []
    for gid, shape, ml, shard, grid, brf in GEOMETRIES:
        for v in VARIANTS:
            out.append(pytest.param(gid, shape, ml, shard, grid, brf, v, id=f"{gid}-{v}"))
    for b in COLLAPSE_BLOCK_ROWS:
        for v in ("stock", "fused_cskip"):
            out.append(
                pytest.param(
                    f"collapse_s1_B{b}",
                    COLLAPSE_SHAPE,
                    _ML.HEIGHT_SHARDED,
                    COLLAPSE_SHARD,
                    COLLAPSE_GRID,
                    b,
                    v,
                    id=f"collapse_s1_B{b}-{v}",
                )
            )
    return out


@pytest.mark.parametrize("case_id,shape,memory_layout,shard_shape,core_grid,block_rows,variant", _cases())
def test_finalize_whole_op(device, case_id, shape, memory_layout, shard_shape, core_grid, block_rows, variant):
    fin_variant, ablate = VARIANTS[variant]
    PLAN_GLOBALS["FINALIZE_VARIANT"] = fin_variant
    PLAN_GLOBALS["ABLATE_BITS"] = ablate
    PLAN_GLOBALS["BLOCK_ROWS_FORCE"] = block_rows

    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)

    memory_config = shard_config(
        shard_shape, core_grid, memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    try:
        out = _run(device, torch_x, torch_gamma, memory_config)
    finally:
        PLAN_GLOBALS["FINALIZE_VARIANT"] = 0
        PLAN_GLOBALS["ABLATE_BITS"] = 0
        PLAN_GLOBALS["BLOCK_ROWS_FORCE"] = 0

    if variant in ABLATIONS:
        return  # payload stubbed: only the ns are meaningful

    pcc = _pcc(out, torch_x, torch_gamma)
    print(f"\nPCC[{case_id}/{variant}] = {pcc:.7f}")
    assert pcc > PCC_GATE, f"{case_id}/{variant}: PCC {pcc} <= {PCC_GATE}"


# ===========================================================================
# fp32_dest_acc_en = True (+ HiFi4) on the two FINALIZE-SENSITIVE geometries.
#
# The interleaved domain cases below are DRAM-bandwidth-bound, so they cannot
# tell a 0.2% fp32-dest result apart from noise.  These two can: the focus shard
# and the s==1 collapse are the geometries where the finalize is a measurable
# share of the wall.  This is the case that matters most for the domain, because
# fp32 dest + HiFi4 IS the op's Phase-0 default config.
# ===========================================================================
FP32ACC_GEOMETRIES = [
    ("focus_fp32acc", (1, 1, 8192, 1024), ttnn.TensorMemoryLayout.BLOCK_SHARDED, [1024, 128], (8, 8), 0),
    (
        "collapse_s1_B16_fp32acc",
        COLLAPSE_SHAPE,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        COLLAPSE_SHARD,
        COLLAPSE_GRID,
        16,
    ),
]


@pytest.mark.parametrize(
    "case_id,shape,memory_layout,shard_shape,core_grid,block_rows",
    [pytest.param(*c, id=c[0]) for c in FP32ACC_GEOMETRIES],
)
@pytest.mark.parametrize("variant", ["stock", "fused_cskip", "no_finalize"])
def test_finalize_fp32acc(device, case_id, shape, memory_layout, shard_shape, core_grid, block_rows, variant):
    PLAN_GLOBALS["FINALIZE_VARIANT"] = VARIANTS[variant][0]
    PLAN_GLOBALS["ABLATE_BITS"] = VARIANTS[variant][1]
    PLAN_GLOBALS["BLOCK_ROWS_FORCE"] = block_rows

    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)
    memory_config = shard_config(
        shard_shape, core_grid, memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    try:
        x = ttnn.from_torch(
            torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
        )
        gamma = ttnn.from_torch(
            torch_gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_t = ttnn.allocate_tensor_on_device(x.shape, x.dtype, x.layout, x.device(), x.memory_config())
        descriptor = create_program_descriptor(
            x, out_t, gamma=gamma, epsilon=EPSILON, compute_kernel_config=cfg
        )
        out = ttnn.to_torch(ttnn.generic_op([x, gamma, out_t], descriptor)).to(torch.float32)
    finally:
        PLAN_GLOBALS["FINALIZE_VARIANT"] = 0
        PLAN_GLOBALS["ABLATE_BITS"] = 0
        PLAN_GLOBALS["BLOCK_ROWS_FORCE"] = 0

    if variant in ABLATIONS:
        return
    pcc = _pcc(out, torch_x, torch_gamma)
    print(f"\nPCC[{case_id}/{variant}] = {pcc:.7f}")
    assert pcc > PCC_GATE, f"{case_id}/{variant}: PCC {pcc} <= {PCC_GATE}"


# ===========================================================================
# DOMAIN sweep — the axes the finalize could plausibly be sensitive to.
#
# The finalize only ever sees the reduce's DEST tile, so most op axes (layout,
# placement, gamma, W-alignment) cannot reach it.  The ONE that can is
# `fp32_dest_acc_en`: it selects the DEST width, and the even-parity stride is an
# assertion about how a DEST tile is addressed.  It is therefore the first case
# here, at the op's own Phase-0 default config (HiFi4 + fp32 dest), which is what
# every golden cell runs.  dtype is swept for the same reason (bfloat8_b and
# float32 activations change what the reduce accumulates, not the walk).
#
# (case_id, shape, dtype, layout, gamma?, fp32_dest_acc, fidelity, W-alignment)
# ===========================================================================
_HiFi4, _HiFi2 = ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2

DOMAIN_CASES = [
    # id                     shape              dtype             layout               gamma  fp32acc fidelity
    ("il_bf16_tile", (1, 1, 2048, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, True, False, _HiFi2),
    ("il_bf16_fp32acc_hifi4", (1, 1, 2048, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, True, True, _HiFi4),
    ("il_fp32_fp32acc", (1, 1, 2048, 1024), ttnn.float32, ttnn.TILE_LAYOUT, True, True, _HiFi4),
    ("il_bfp8_fp32acc", (1, 1, 2048, 1024), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, True, True, _HiFi4),
    ("il_bf16_rowmajor", (1, 1, 2048, 1024), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, True, False, _HiFi2),
    ("il_bf16_w_nonaligned", (1, 1, 2048, 1000), ttnn.bfloat16, ttnn.TILE_LAYOUT, True, False, _HiFi2),
    ("il_bf16_no_gamma", (1, 1, 2048, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, False, False, _HiFi2),
    ("il_bf16_h_nonaligned", (1, 1, 2000, 1024), ttnn.bfloat16, ttnn.TILE_LAYOUT, True, False, _HiFi2),
]


@pytest.mark.parametrize(
    "case_id,shape,dtype,layout,use_gamma,fp32_acc,fidelity",
    [pytest.param(*c, id=c[0]) for c in DOMAIN_CASES],
)
@pytest.mark.parametrize("variant", ["stock", "fused_cskip"])
def test_finalize_domain(device, case_id, shape, dtype, layout, use_gamma, fp32_acc, fidelity, variant):
    PLAN_GLOBALS["FINALIZE_VARIANT"] = VARIANTS[variant][0]
    PLAN_GLOBALS["ABLATE_BITS"] = 0
    PLAN_GLOBALS["BLOCK_ROWS_FORCE"] = 0

    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.float32)
    torch_gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32)

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=fidelity, fp32_dest_acc_en=fp32_acc, math_approx_mode=False
    )
    try:
        x = ttnn.from_torch(
            torch_x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gamma = (
            ttnn.from_torch(
                torch_gamma,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            if use_gamma
            else None
        )
        out_t = ttnn.allocate_tensor_on_device(x.shape, x.dtype, x.layout, x.device(), x.memory_config())
        descriptor = create_program_descriptor(
            x, out_t, gamma=gamma, epsilon=EPSILON, compute_kernel_config=cfg
        )
        io = [x] + ([gamma] if gamma is not None else []) + [out_t]
        out = ttnn.to_torch(ttnn.generic_op(io, descriptor)).to(torch.float32)
    finally:
        PLAN_GLOBALS["FINALIZE_VARIANT"] = 0

    # golden against the dtype the device actually saw
    xf = ttnn.to_torch(ttnn.from_torch(torch_x, dtype=dtype, layout=layout)).to(torch.float32)
    expected = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPSILON)
    if use_gamma:
        gf = ttnn.to_torch(ttnn.from_torch(torch_gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
        expected = expected * gf.to(torch.float32).reshape(-1)
    a, b = out.flatten(), expected.flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    print(f"\nPCC[{case_id}/{variant}] = {pcc:.7f}")
    assert pcc > PCC_GATE, f"{case_id}/{variant}: PCC {pcc} <= {PCC_GATE}"
