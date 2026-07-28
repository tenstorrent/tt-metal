# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: `gather_spread_topology` — spread the combine's DESTINATION.

ONE idea, three honest variants of the cross-core combine's gather TREE. Every
other byte of the op is held constant (same core set, same per-core W slice, same
tile-row assignment, same multicast rectangles + ack counts, same precision
contract: bf16 / HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False / bf16
TILE gamma), so the measured device delta is attributable to the topology alone.

    flat          the op's CURRENT approach (honest baseline). CW workers write
                  ht fp32 tiles each into ONE root; the root folds CW * ht tiles
                  and multicasts the answer back.
    two_stage_1d  SCHEME 1. Factorize CW itself in SLOT space (the op's
                  _two_stage_extents refuses a group that is one grid row, which
                  is exactly BLOCK_SHARDED). CW = A x CW/A: sub-leaders fold A,
                  the root folds CW/A. Busiest receiver: (A + CW/A) * ht tiles.
    row_rotate    SCHEME 2. Core j owns the fold for tile-rows h with h % CW == j,
                  so every core receives CW tiles and NO core receives CW * ht.
                  The owner's fold is the FINAL one, so there is no second
                  REDUCE_ROW anywhere and the at/at_last double-count trap cannot
                  arise. Owners then unicast their finalized tiles into the root's
                  cb_rms_mean and the root broadcasts the block exactly as `flat`.

Run (one fresh-cache dispatch per case; DEVICE KERNEL DURATION [ns] = col 19):

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/gather_spread_topology/test_gather_spread.py

Correctness is the ONLY pass/fail. Perf is measured, never asserted.

MEASURED — blackhole_p150b, 11x10 grid, AICLK 1349.99 MHz, bf16 / TILE / HiFi2 /
fp32_dest_acc_en=False / math_approx_mode=False / bf16 TILE gamma (identical for
every variant). DEVICE KERNEL DURATION [ns], ONE dispatch per cell, no trial loop.
`op_current` vs `flat` are byte-identical programs on every 1-D group, so their
spread IS the noise band: 0.15-0.9% across the nine geometries.

  geometry           cw  htb nhc | op_current    flat  2stage_A4  2stage_A2  row_rotate
  focus 8192x1024 B   8    8   4 |     75_573  75_280     73_558     73_505      64_364
  width 32x1024 W     8    1   1 |      5_167   5_216      5_531      5_436       5_421
  width 32x7168 W    28    1   1 |      6_595   7_384      6_619      6_872       7_672
  width 32x5120 W    32    1   1 |      5_941   7_154      6_129      6_469       7_304
  block 2048x1024 B   8    8   1 |     20_621  20_652     20_141     20_135      17_952
  block 1024x1024 B   8    4   1 |     11_750  11_771     11_773     11_796      10_912
  block  512x1024 B   8    2   1 |      7_529   7_560      7_630      7_566       7_450
  block  256x1024 B   8    1   1 |      5_441   5_397      5_705      5_755       5_644
  focus @1 MB budget  8  8/16  4 |     75_575  75_490     73_650     73_702      61_373

  speedup vs op_current:  row_rotate 1.174x (focus) / 1.231x (focus @1 MB budget)
                          1.149x (htb 8) 1.077x (htb 4) 1.011x (htb 2) 0.964x (htb 1)
                          0.813-0.953x on the ht = 1 WIDTH geometries
                          two_stage_1d 1.027-1.028x (focus), null/regression elsewhere

Focus-shape per-stage zones (zone_report.py), baseline -> row_rotate, ns:

  wtr_gather_hop   59_076 avg / 60_342 max  ->  52_189 avg / 52_582 max
  rdr_gather_wait   5_464 avg / 43_401 max  ->  43_420 avg / 46_425 max
  cmp_combine(M)    3_207 avg / 25_367 max  ->  13_126 avg / 14_419 max
  rdr_mcast        56_444 avg               ->   7_466 avg
  cmp_rsqrt(M)     56_063 avg               ->  35_093 avg   (idle it was absorbing)

Both halves of the imbalance close at once: the gather-wait max/avg spread
collapses from 8:1 to 1.07:1 (every core now absorbs CW tiles, not CW x ht), and
the root's serial fold max falls 1.76x because min(ht, CW) cores fold in parallel.
The broadcast wait nearly disappears — there is no single core left to wait for.

L1: row_rotate SHRINKS the gather CB from HT_BLOCK*CW to ceil(HT_BLOCK/CW)*CW
(focus: 256 KB -> 32 KB; program CBs 536 KB -> 312 KB). That is what buys the
second-order win: at a 1 MB block budget the flat gather (512 KB) forces the
halving loop back to HT_BLOCK 8, while row_rotate keeps HT_BLOCK 16 and halves the
number of combine round trips (nh_core 4 -> 2).

NOTE on imports: this file lives under ttnn/ttnn/operations/, whose __init__.py
exec_module()s every .py it walks at `import ttnn` time. Keep module-level work
to `import ttnn` + pytest decorators — torch and the lab descriptor are imported
lazily inside the tests.
"""

import importlib.util
import pathlib

import pytest

import ttnn

# Root `device` fixture is function-scoped; this marker switches it to module
# scope so the 30 cases share one device open. Applied via `pytestmark` rather
# than a conftest.py — a conftest INSIDE the ttnn package makes pytest import
# ttnn twice ("Operation with name bernoulli is already registered").
pytestmark = pytest.mark.use_module_device

_LAB_DIR = pathlib.Path(__file__).resolve().parent
_lab = None


def lab():
    """Load lab_descriptor.py by path (the dir is deliberately not a package)."""
    global _lab
    if _lab is None:
        spec = importlib.util.spec_from_file_location("gst_lab_descriptor", _LAB_DIR / "lab_descriptor.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _lab = mod
    return _lab


def perf_compute_kernel_config():
    """The PINNED precision contract. Identical for every variant — never a lever."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


# ---------------------------------------------------------------------------
# Geometries. The FOCUS case first, then the combine shapes whose (cw, ht_block)
# differ — that pair is the predicate the win is expected to key on.
# ---------------------------------------------------------------------------
#
# (id, shape, kind, shard_shape, core_grid, block_budget_kb or None)
GEOMETRIES = [
    # FOCUS: 64 cores, cw=8 cw1=8 cw2=1 flat, per core 32 tile-rows x Wt=4,
    # ht_block=8, nh_core=4 -> the gather is 8 cores x 8 fp32 tiles = 256 KB per
    # row-block into ONE root.
    ("focus-block-8192x1024", (1, 1, 8192, 1024), "BLOCK", [1024, 128], (8, 8), None),
    # ht = 1 geometries: SCHEME 2's ht-fold has nothing to fold. Predicate boundary.
    ("width-32x1024", (1, 1, 32, 1024), "WIDTH", [32, 128], (8, 1), None),
    ("width-32x7168", (1, 1, 32, 7168), "WIDTH", [32, 256], (7, 4), None),
    ("width-32x5120", (1, 1, 32, 5120), "WIDTH", [32, 160], (8, 4), None),
    # ht sweep at fixed cw=8 and nh_core=1 (BLOCK, one grid row per group): where
    # does the ht-fold start paying? ht_block is derived from the shard height.
    ("block-2048x1024", (1, 1, 2048, 1024), "BLOCK", [256, 128], (8, 8), None),  # ht 8
    ("block-1024x1024", (1, 1, 1024, 1024), "BLOCK", [128, 128], (8, 8), None),  # ht 4
    ("block-512x1024", (1, 1, 512, 1024), "BLOCK", [64, 128], (8, 8), None),  # ht 2
    ("block-256x1024", (1, 1, 256, 1024), "BLOCK", [32, 128], (8, 8), None),  # ht 1
    # SECOND-ORDER: does the L1 the smaller gather CB frees buy a BIGGER block?
    # At a 1 MB block budget the derivation wants ht_block = 16; the flat gather
    # (16 * 8 fp32 tiles = 512 KB) will not fit, so the halving loop takes the
    # block back — but row_rotate's gather is 16 tiles = 64 KB and should keep it.
    ("focus-budget1m", (1, 1, 8192, 1024), "BLOCK", [1024, 128], (8, 8), 1024),
]

VARIANTS = [
    # THE HONEST BASELINE: whatever the op picks today for this geometry (flat on
    # the 1-D BLOCK groups, R3 grid two-stage on the wide WIDTH ones).
    ("op_current", "op_current", None),
    # Forced flat at the same CW — equals op_current on the 1-D groups; on the
    # already-staged WIDTH groups it is the R2 topology, kept as a reference.
    ("flat", "flat", None),
    ("two_stage_1d-A4", "two_stage_1d", 4),
    ("two_stage_1d-A2", "two_stage_1d", 2),
    ("row_rotate", "row_rotate", None),
]

_CASES = [(g[0], v[0]) for g in GEOMETRIES for v in VARIANTS]
_GEO = {g[0]: g for g in GEOMETRIES}
_VAR = {v[0]: v for v in VARIANTS}


class _budget:
    """Scoped override of the lab descriptor's block-factor knob (default = op's)."""

    def __init__(self, kb):
        self.kb = kb

    def __enter__(self):
        self.saved = lab().L1_BLOCK_BUDGET_BYTES
        if self.kb:
            lab().L1_BLOCK_BUDGET_BYTES = self.kb * 1024
        return self

    def __exit__(self, *a):
        lab().L1_BLOCK_BUDGET_BYTES = self.saved
        return False


def _memcfg(device, geo):
    from eval.sharding import shard_config

    _, _shape, kind, shard_shape, core_grid, _budget_kb = geo
    return shard_config(
        list(shard_shape),
        core_grid,
        getattr(ttnn.TensorMemoryLayout, f"{kind}_SHARDED"),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )


def _inputs(device, geo, *, ones=False):
    import torch

    shape = geo[1]
    mc = _memcfg(device, geo)
    torch.manual_seed(42)
    torch_x = torch.ones(shape, dtype=torch.bfloat16) if ones else torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.ones(shape[-1], dtype=torch.bfloat16) if ones else torch.randn(shape[-1], dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tt_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    return torch_x, torch_gamma, tt_x, tt_gamma, mc


def _reference(torch_x, torch_gamma):
    import torch

    xf = torch_x.to(torch.float32)
    out = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
    return out * torch_gamma.to(torch.float32).reshape(-1)


def _pcc(a, b):
    import torch

    af, bf = a.flatten().to(torch.float32), b.flatten().to(torch.float32)
    return torch.corrcoef(torch.stack([af, bf]))[0, 1].item()


def _run(device, geo_id, variant_id, *, ones=False):
    geo = _GEO[geo_id]
    _, topology, stage1 = _VAR[variant_id]
    torch_x, torch_gamma, tt_x, tt_gamma, mc = _inputs(device, geo, ones=ones)
    with _budget(geo[5]):
        tt_out = lab().lab_rms_norm(
            tt_x,
            gamma=tt_gamma,
            epsilon=1e-6,
            compute_kernel_config=perf_compute_kernel_config(),
            memory_config=mc,
            topology=topology,
            stage1_width=stage1,
        )
    out = ttnn.to_torch(tt_out)
    # The inputs are L1-SHARDED, so a leaked tensor is a quarter-MB per core that
    # the next case cannot use. Free them explicitly, not on GC.
    ttnn.deallocate(tt_out)
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_gamma)
    return torch_x, torch_gamma, out


# ---------------------------------------------------------------------------
# 0. host-only: what each topology actually derives (no device work, no dispatch)
# ---------------------------------------------------------------------------


def test_report_topologies(device):
    import torch

    print()
    hdr = f"{'geometry':24s} {'variant':16s} {'cores':>5s} {'cw':>3s} {'cw1':>4s} {'cw2':>4s} {'htb':>4s} {'nhc':>4s} {'gatherKB':>9s} {'progCB_KB':>10s}"
    print(hdr)
    for geo in GEOMETRIES:
        mc = _memcfg(device, geo)
        x = torch.zeros(geo[1], dtype=torch.bfloat16)
        g = torch.zeros(1, 1, 1, geo[1][-1], dtype=torch.bfloat16)
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        tt_g = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for vid, topology, stage1 in VARIANTS:
            with _budget(geo[5]):
                r = lab().lab_report(device, tt_x, tt_g, topology, stage1)
            print(
                f"{geo[0]:24s} {vid:16s} {r['cores']:5d} {r['cw']:3d} {r['cw1']:4d} {r['cw2']:4d} "
                f"{r['ht_block']:4d} {r['nh_core']:4d} {r['gather_kb']:9d} {r['program_cb_kb']:10d}"
            )
        ttnn.deallocate(tt_x)
        ttnn.deallocate(tt_g)


# ---------------------------------------------------------------------------
# 1. correctness gates (mandatory, all three)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("geo_id,variant_id", _CASES, ids=[f"{g}-{v}" for g, v in _CASES])
def test_pcc(device, geo_id, variant_id):
    """PCC >= 0.9995 vs torch — the focus case's soft threshold."""
    torch_x, torch_gamma, actual = _run(device, geo_id, variant_id)
    import torch

    pcc = _pcc(_reference(torch_x, torch_gamma), actual.to(torch.float32))
    print(f"\nPCC {geo_id} {variant_id}: {pcc:.6f}")
    assert pcc >= 0.9995, f"{geo_id} {variant_id}: pcc {pcc}"


@pytest.mark.parametrize("geo_id,variant_id", _CASES, ids=[f"{g}-{v}" for g, v in _CASES])
def test_all_ones_absolute(device, geo_id, variant_id):
    """ABSOLUTE element-count check: all-ones must recover mean(x^2) = 1.0 exactly.

    PCC is scale-invariant and scored the historical 8.75-instead-of-1.0 bug at
    0.9999, so this — not PCC — is the gate that catches a premature within-tile
    fold or a dropped/double-counted contributor.
    """
    import torch

    _, _, actual = _run(device, geo_id, variant_id, ones=True)
    out = actual.to(torch.float32)
    implied = 1.0 / (out.flatten()[0].item() ** 2)
    print(f"\nall-ones {geo_id} {variant_id}: implied mean(x^2) = {implied:.6f}")
    assert torch.allclose(
        out, torch.ones_like(out), rtol=2e-3, atol=2e-3
    ), f"{geo_id} {variant_id}: implied mean(x^2) = {implied:.4f}, expected 1.0"


@pytest.mark.parametrize("geo_id", [g[0] for g in GEOMETRIES])
def test_topologies_agree(device, geo_id):
    """Every topology folds the SAME raw slice accumulators over the SAME cores.

    Only the fan-in tree differs, so the answers must agree. Association order of
    an fp32 add tree changing is numerically benign; a premature within-tile fold
    is not, and this is where it would show.
    """
    import torch

    outs = {}
    for vid, _, _ in VARIANTS:
        _, _, outs[vid] = _run(device, geo_id, vid)
    ref = outs["flat"].to(torch.float32)
    scale = ref.abs().max().item()
    for vid, out in outs.items():
        if vid == "flat":
            continue
        o = out.to(torch.float32)
        # The OUTPUT is bf16 (relative spacing 2^-8), and the mean lands in a bf16
        # DEST before rsqrt (fp32_dest_acc_en=False, the pinned contract). So a
        # different fp32 ADD ORDER legitimately flips a few output ULPs. Gate on
        # correlation, which a structural error (a dropped or double-counted
        # contributor) destroys, and report the worst element as data.
        pcc = _pcc(o, ref)
        worst = (o - ref).abs().max().item()
        print(f"\nagree {geo_id} {vid}: pcc-vs-flat {pcc:.7f} max|diff| {worst:.6f} ({worst / scale:.2%} of peak)")
        assert pcc >= 0.9999, f"{geo_id}: {vid} disagrees with flat (pcc {pcc})"


# ---------------------------------------------------------------------------
# 2. the measurement — ONE fresh dispatch per (geometry, variant), no trial loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("geo_id,variant_id", _CASES, ids=[f"{g}-{v}" for g, v in _CASES])
def test_measure(device, geo_id, variant_id):
    """One dispatch. The number comes from --profile's CSV, never from a timer.

    Correctness still gated (PCC) so a perf number can never be taken off a
    wrong-but-fast kernel.
    """
    import torch

    torch_x, torch_gamma, actual = _run(device, geo_id, variant_id)
    pcc = _pcc(_reference(torch_x, torch_gamma), actual.to(torch.float32))
    assert pcc >= 0.9995, f"{geo_id} {variant_id}: pcc {pcc}"
