# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device perf harness for rms_norm. Correctness-checked, timing-measured.

Run under the Tracy profiler to get per-op device kernel time:

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf.py

Each test is a single fresh-cache dispatch of one shape so the emitted
ops_perf_results CSV has one row per case (DEVICE KERNEL DURATION [ns] +
CORE COUNT). Correctness is still asserted — a perf number from a wrong kernel
is worthless.

Shapes mirror `eval/golden_tests/rms_norm/feature_spec.py`'s LLM-derived perf
cases: (1, 1, rows, hidden) with rows = 32 (decode, latency-bound, ht_total = 1
tile-row) and rows = 8192 (prefill, bandwidth-bound, ht_total = 256 tile-rows).

`RMS_NORM_BLOCK_BUDGET_KB` overrides the block-factor knob
(`L1_BLOCK_BUDGET_BYTES`) so the block size can be A/B-measured without editing
the op:

    RMS_NORM_BLOCK_BUDGET_KB=128 scripts/run_safe_pytest.sh --profile ...

`RMS_NORM_X_RES_DEPTH` overrides the resident-strip depth knob, and
`RMS_NORM_FORCE_STREAMING=1` disables the residency fast path entirely, so the
1-read-pass vs 2-read-pass lever can be measured as an ablation.

Measured — blackhole_p150b, 110-core grid, AICLK 1350 MHz, bf16 / HiFi4 /
fp32_dest_acc_en=True, DEVICE KERNEL DURATION [ns], fresh JIT cache per run.
"first commit" is the correctness-only baseline (commit 56cbc4f8).

    shape            cores   first commit      tuned    speedup
    (1,1,32,1024)        1         13_559     12_172      1.11x
    (1,1,32,2304)        1         25_404     21_382      1.19x
    (1,1,32,5120)        1         51_434     42_439      1.21x
    (1,1,32,7168)        1         63_313     57_474      1.10x
    (1,1,8192,1024)    110        103_238    102_114      1.01x
    (1,1,8192,2304)    110        214_633    215_181      1.00x
    (1,1,8192,5120)    110        482_655    483_074      1.00x
    (1,1,8192,7168)    110        835_407    831_973      1.00x

Two independent 512 KB runs agreed to within 1-2%, which sets the noise band;
everything reported as 1.00x is inside it.

Bottleneck classification (from the per-RISC columns of the same CSV): the
prefill rows are READER-bound — NCRISC occupies 90-99% of the kernel
(8192x5120: 442_221 of 482_655 ns) — so they sit at the interleaved-DRAM read
floor and the levers that pay off there are the ones that reduce bytes or
transactions, not the ones that overlap stages. The decode rows are
LATENCY-bound on one core (ht_total == 1 tile-row leaves the independent row
axis with a single unit of work), which is why the overlap levers win 1.10-1.21x
there and are explicitly disabled when the grid is full.

Block-factor knob A/B (L1_BLOCK_BUDGET_BYTES): 512 KB beats 1024 KB
everywhere — the wide prefill case degrades badly at 1024 KB
((1,1,8192,7168): 831_973 -> 1_058_535 ns, 0.79x), so the default is measured,
not assumed.

Refinement 2 built op_design.md's Lamp L1 — the cross-core W-split: each core
reduces its own W slice to a raw partial, the group root combines them and
multicasts 1/rms back. `RMS_NORM_W_SPLIT=0` forces the phase-0 row-only split,
so the two schemes A/B directly. Measured, same box/config, DEVICE KERNEL
DURATION [ns], fresh JIT cache per variant:

    shape             row-split  cores | W-split  cores | speedup
    (1,1,32,1024)        12_271      1 |   6_999     32 |  1.75x
    (1,1,32,2304)        21_429      1 |   7_663     36 |  2.80x
    (1,1,32,5120)        42_484      1 |   9_611     40 |  4.42x
    (1,1,32,7168)        57_608      1 |  11_279     56 |  5.11x
    (1,1,8192,1024)     103_052    110 | 101_922    110 |  1.01x
    (1,1,8192,2304)     217_543    110 | 215_814    110 |  1.01x
    (1,1,8192,5120)     493_253    110 | 487_581    110 |  1.01x
    (1,1,8192,7168)     830_471    110 | 834_799    110 |  0.99x

The decode column is the whole win, and it is grid occupancy: the split turns
1 busy core into 32-56. Prefill is untouched (the split is not engaged there —
its row axis already fills the grid), every prefill row inside the 1-2% noise
band.

Refinement 3 tuned that split's own knobs, at the PINNED perf config every
`achievable_ns` reference was taken at (bf16 / TILE / HiFi2 /
fp32_dest_acc_en=False / bf16 TILE gamma — `perf_compute_kernel_config()`, used
by `test_rms_norm_perf_decode_pinned`). Measured AICLK 1349.98 MHz, i.e. the
reference clock, so no scaling applies.

    shape          R2 flat   R3 staged   speedup   ceiling   margin
    (1,1,32,1024)    6_938       5_709     1.22x     9_149    1.60x inside
    (1,1,32,2304)    7_555       6_219     1.21x    17_003    2.73x inside
    (1,1,32,5120)    9_309       7_933     1.17x    75_825    9.56x inside
    (1,1,32,7168)   10_929       8_917     1.23x    14_894    1.67x inside

The `(1,1,32,7168)` ceiling is `104_259 / minimum_expected_speedup 7.0`; the op
delivers 104_259 / 8_917 = **11.7x** against that 7.0x requirement.

Where the time goes, by ABLATION (`RMS_NORM_ABLATE=combine[,gamma]`, which keeps
the placement and every DRAM read/write and removes only the named stage):

    shape          full   no-combine   combine   combine share
    (1,1,32,1024)  6_938       3_524     3_414        49%
    (1,1,32,2304)  7_555       3_827     3_728        49%
    (1,1,32,5120)  9_309       5_152     4_157        45%
    (1,1,32,7168) 10_929       5_759     5_170        47%

so the combine, not the data movement, was the decode bottleneck (NCRISC is only
1.0-2.2 us of it). gamma costs 117-468 ns total — noise by comparison.

Two knobs came out of that. `RMS_NORM_MAX_FLAT_FANIN` picks the combine topology
at fixed CW (see COMBINE_MAX_FLAT_FANIN); `RMS_NORM_GATHER_BUDGET_KB` caps CW.
CW re-swept under the staged topology — widest-that-fits is confirmed optimal,
where under the flat one it had been actively harmful:

    shape          CW=8    CW=16    CW=32    CW=36-56 (widest)
    (1,1,32,7168) 12_016      —    10_116*      8_917
    (1,1,32,1024)  5_739      —         —       5_709
    (* flat-topology measurement; the staged one is not CW-limited there)

Refinement 4 added `RMS_NORM_L1_HEADROOM_KB`, which sets L1_ALLOC_HEADROOM_BYTES
(see the op). A zero-copy sharded CB is aliased onto the tensor's own buffer, so
it is NOT program-allocated; charging it against L1_CB_BUDGET_BYTES double-counts
it and makes the halving loop pay for the shard by shrinking the block. Set the
headroom wider than the L1 bank to get that single-budget model back for an A/B.
Measured on the five pinned sharded geometries (test_rms_norm_perf_sharded_pinned,
pinned perf config), only the one whose blocking actually moves changes:

    geometry                                single-budget   two-budget
    (1,1,32,1024)   WIDTH [32,128]  (8,1)           5_007        5_016
    (1,1,32,2304)   WIDTH [32,256]  (9,1)           5_684        5_667
    (1,1,32,5120)   WIDTH [32,160]  (8,4)           5_878        5_890
    (1,1,32,7168)   WIDTH [32,256]  (7,4)           6_295        6_284
    (1,1,8192,1024) BLOCK [1024,128](8,8)          89_413       85_107   HT_BLOCK 4->8
"""

import os

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd


PERF_SHAPES = [
    # decode (ht_total == 1 tile-row -> the row axis offers ONE core; wider W is
    # in-core chunked. This is the regime op_design.md Lamp L1 unlocks).
    (1, 1, 32, 1024),
    (1, 1, 32, 2304),
    (1, 1, 32, 5120),
    (1, 1, 32, 7168),
    # prefill (ht_total == 256 tile-rows -> fills the whole compute grid).
    (1, 1, 8192, 1024),
    (1, 1, 8192, 2304),
    (1, 1, 8192, 5120),
    (1, 1, 8192, 7168),
]

# The four perf-flagged DECODE profiles of eval/golden_tests/rms_norm/feature_spec.py,
# with their clock-scalable achievable_ns reference (and the required speedup
# where feature_spec pins one). Refinement 3 optimizes exactly these, at the
# PINNED perf config below — never at the op's default HiFi4 / fp32-on corner.
DECODE_REFERENCE_NS = {
    (1, 1, 32, 1024): (9149, 1.0),
    (1, 1, 32, 2304): (17003, 1.0),
    (1, 1, 32, 5120): (75825, 1.0),
    (1, 1, 32, 7168): (104259, 7.0),
}
REFERENCE_AICLK_MHZ = 1350


def perf_compute_kernel_config():
    """feature_spec._PERF_BASE's fixed precision: HiFi2 + bf16 DEST.

    Single source for every perf measurement in this file, so a knob A/B can
    never accidentally be taken on the default (HiFi4 / fp32_dest_acc_en=True)
    datapath the achievable_ns references were NOT measured at.
    """
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


@pytest.fixture(autouse=True)
def _knob_overrides():
    """Apply the env-var knob overrides for one test, then restore."""
    saved = (
        pd.L1_BLOCK_BUDGET_BYTES,
        pd.L1_CB_BUDGET_BYTES,
        pd.X_RESIDENT_DEPTH,
        pd.L1_GATHER_BUDGET_BYTES,
        pd.COMBINE_MAX_FLAT_FANIN,
    )
    kb = os.environ.get("RMS_NORM_BLOCK_BUDGET_KB")
    if kb:
        pd.L1_BLOCK_BUDGET_BYTES = int(kb) * 1024
    depth = os.environ.get("RMS_NORM_X_RES_DEPTH")
    if depth:
        pd.X_RESIDENT_DEPTH = int(depth)
    if os.environ.get("RMS_NORM_FORCE_STREAMING") == "1":
        # A CB budget of 0 makes both residency predicates false, selecting the
        # bounded streaming fallback (2 reader passes) without touching the op.
        pd.L1_CB_BUDGET_BYTES = 0
    # Refinement 3: the combine's own two knobs. The gather budget caps the group
    # width CW (cap = budget // fp32_tile_bytes), so sweeping it sweeps how many
    # cores share one reduce group; the flat-fan-in cap selects the combine
    # TOPOLOGY at a fixed CW (raise it past the group area to force the flat root
    # gather, i.e. the Refinement 2 behaviour).
    gkb = os.environ.get("RMS_NORM_GATHER_BUDGET_KB")
    if gkb:
        pd.L1_GATHER_BUDGET_BYTES = int(gkb) * 1024
    flat = os.environ.get("RMS_NORM_MAX_FLAT_FANIN")
    if flat:
        pd.COMBINE_MAX_FLAT_FANIN = int(flat)
    yield
    (
        pd.L1_BLOCK_BUDGET_BYTES,
        pd.L1_CB_BUDGET_BYTES,
        pd.X_RESIDENT_DEPTH,
        pd.L1_GATHER_BUDGET_BYTES,
        pd.COMBINE_MAX_FLAT_FANIN,
    ) = saved


@pytest.mark.parametrize("shape", PERF_SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rms_norm_perf(device, shape):
    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(shape[-1], dtype=torch.bfloat16)

    tt_x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, shape[-1]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_out = rms_norm(tt_x, gamma=tt_gamma, epsilon=1e-6)

    xf = torch_x.to(torch.float32)
    expected = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
    expected = expected * torch_gamma.to(torch.float32).reshape(-1)

    actual = ttnn.to_torch(tt_out).to(torch.float32)
    passed, message = check_with_pcc(expected, actual, 0.995)
    assert passed, message


@pytest.mark.skipif(
    not os.environ.get("RMS_NORM_ABLATE"),
    reason="ablation variant — opt in with RMS_NORM_ABLATE=<stage[,stage]> (output is WRONG by design)",
)
@pytest.mark.parametrize("shape", list(DECODE_REFERENCE_NS), ids=lambda s: "x".join(map(str, s)))
def test_rms_norm_ablate(device, shape, monkeypatch):
    """ABLATION (perf only, output is wrong by design): peel one stage, keep the rest.

    ``RMS_NORM_ABLATE`` is a comma list of stages to remove:

      ``combine`` — keeps the cross-core placement byte-for-byte (same core
        count, same per-core W slice, same DRAM reads and writes, same per-core
        square/reduce/scale) and removes only the gather + root-fold +
        multicast legs, by handing the kernels ``cw = 1`` after the placement is
        already laid out. Each core then finalizes ``rsqrt`` over its OWN slice.
      ``gamma`` — drops the gamma tensor, removing its DRAM read and the phase-6
        multiply.

    Deliberately asserts nothing (perf-measure: never PCC-gate an ablated
    kernel). Opt-in only, so a normal suite run never executes it.
    """
    stages = set(os.environ.get("RMS_NORM_ABLATE", "").split(","))

    if "combine" in stages:
        real_select = pd._select_placement

        def _no_combine(*args, **kwargs):
            p = real_select(*args, **kwargs)
            p.cw = p.cw1 = p.cw2 = 1  # kernels compile with W_SPLIT == 0; slices unchanged
            p.groups = []
            return p

        monkeypatch.setattr(pd, "_select_placement", _no_combine)

    torch.manual_seed(42)
    tt_x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt_gamma = None
    if "gamma" not in stages:
        tt_gamma = ttnn.from_torch(
            torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
    rms_norm(tt_x, gamma=tt_gamma, epsilon=1e-6, compute_kernel_config=perf_compute_kernel_config())


@pytest.mark.parametrize("shape", list(DECODE_REFERENCE_NS), ids=lambda s: "x".join(map(str, s)))
def test_rms_norm_perf_decode_pinned(device, shape):
    """Decode column at the PINNED perf config (bf16 / TILE / HiFi2 / fp32-off).

    Refinement 3's measurement surface. Same single dispatch per shape as
    test_rms_norm_perf, but with the compute config feature_spec's perf loose
    cases actually run at, and the tighter pcc_threshold = 0.9995 soft gate
    those cases carry.
    """
    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(shape[-1], dtype=torch.bfloat16)

    tt_x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, shape[-1]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_out = rms_norm(tt_x, gamma=tt_gamma, epsilon=1e-6, compute_kernel_config=perf_compute_kernel_config())

    xf = torch_x.to(torch.float32)
    expected = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
    expected = expected * torch_gamma.to(torch.float32).reshape(-1)

    actual = ttnn.to_torch(tt_out).to(torch.float32)
    passed, message = check_with_pcc(expected, actual, 0.9995)
    assert passed, message


# The five measured-fastest SHARDED geometries feature_spec pins (shard_shape +
# core_grid come straight from its extras, so the geometry is reproduced exactly
# rather than left to auto_shard_config). Refinement 5 owns optimizing these;
# Refinement 4 added them as its no-regression guard, because splitting the L1
# budget into "program CBs" + "resident shard" (L1_ALLOC_HEADROOM_BYTES) is the
# one change that can move a SHARDED cell's blocking. Measured, exactly one of
# them moves: (1,1,8192,1024) BLOCK_SHARDED, HT_BLOCK 4 -> 8.
SHARDED_REFERENCE_NS = {
    ((1, 1, 32, 1024), "WIDTH", (32, 128), (8, 1)): 4110,
    ((1, 1, 32, 2304), "WIDTH", (32, 256), (9, 1)): 4617,
    ((1, 1, 32, 5120), "WIDTH", (32, 160), (8, 4)): 5267,
    ((1, 1, 32, 7168), "WIDTH", (32, 256), (7, 4)): 5481,
    ((1, 1, 8192, 1024), "BLOCK", (1024, 128), (8, 8)): 25640,
}


@pytest.mark.parametrize("case", list(SHARDED_REFERENCE_NS), ids=lambda c: f"{'x'.join(map(str, c[0]))}-{c[1].lower()}")
def test_rms_norm_perf_sharded_pinned(device, case):
    """The pinned sharded geometries at the PINNED perf config, one dispatch each.

    Same shape as test_rms_norm_perf_decode_pinned: no timing assertion here (the
    number comes from `--profile`'s CSV), just the 0.9995 soft gate those loose
    cases carry, so an A/B can never be taken on a wrong-but-fast kernel.
    """
    from eval.sharding import shard_config

    shape, kind, shard_shape, core_grid = case
    memory_layout = getattr(ttnn.TensorMemoryLayout, f"{kind}_SHARDED")
    mc = shard_config(
        list(shard_shape), core_grid, memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )

    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(shape[-1], dtype=torch.bfloat16)

    tt_x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tt_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    tt_out = rms_norm(
        tt_x, gamma=tt_gamma, epsilon=1e-6, compute_kernel_config=perf_compute_kernel_config(), memory_config=mc
    )

    xf = torch_x.to(torch.float32)
    expected = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
    expected = expected * torch_gamma.to(torch.float32).reshape(-1)

    actual = ttnn.to_torch(tt_out).to(torch.float32)
    passed, message = check_with_pcc(expected, actual, 0.9995)
    assert passed, message


def test_report_blocking(device):
    """Print the derived blocking for every perf shape (no device work).

    Cheap visibility into what the knobs actually resolved to, so a perf number
    can be attributed to a block size / residency regime / core count.
    """
    grid = device.compute_with_storage_grid_size()
    print(f"\ncompute grid: {grid.x} x {grid.y} = {grid.x * grid.y} cores")
    header = (
        f"{'shape':22s} {'Wt/core':>7} {'C':>4} {'NW':>4} {'H':>3} "
        f"{'ht_tot':>7} {'CW':>4} {'CW1':>4} {'CW2':>4} {'cores':>6} "
        f"{'xdep':>5} {'gres':>5} {'rdbat':>6} {'CB KB':>7}"
    )
    print(header)
    for shape in PERF_SHAPES:
        tt_x = ttnn.from_torch(
            torch.zeros(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        tt_g = ttnn.from_torch(
            torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        ht_total, wt_global = pd._tile_geometry(tt_x)
        placement = pd._select_placement(device, grid, tt_x, ht_total, wt_global, False)
        blk = pd._derive_blocking(tt_x, tt_g, grid.x * grid.y, placement)
        print(
            f"{str(shape):22s} {blk.Wt:7d} {blk.wt_chunk:4d} {blk.nw:4d} {blk.ht_block:3d} "
            f"{blk.ht_total:7d} {placement.cw:4d} {placement.cw1:4d} {placement.cw2:4d} "
            f"{placement.num_cores:6d} {blk.x_res_depth:5d} {int(blk.gamma_resident):5d} "
            f"{pd._x_read_chunks(blk):6d} {blk.cb_total_bytes // 1024:7d}"
        )
        ttnn.deallocate(tt_x)
        ttnn.deallocate(tt_g)
