# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: `combine_parallel_fold` — take the group's fold off ONE core.

PERF-2 IDEA. Under the cross-core W-split every group's combine folds onto a
SINGLE root core: on the focus shape that is 256 fp32 tile-reduces (HT_BLOCK=8 x
CW1=8) on one core while the other 7 sit idle. Round 1 measured two pure ends of
a spectrum:

  row_rotate (this dir's sibling `gather_spread_topology`)
      Core j of the group owns the fold for tile-rows h with h % CW == j: CW
      owners, ho = ceil(HT_BLOCK/CW) rows each, full CW-way parallelism, but the
      GATHER PAYLOAD is unchanged (`ht` full fp32 tiles per worker).
  colpack (this round's sibling `gather_payload_shrink`)
      Every worker column-packs all HT_BLOCK row-sums into ONE tile (8x smaller
      payload on the focus shape), but there is exactly ONE owner (the root) —
      zero fold parallelism.

THIS bake-off's own idea, `pack_rotate`, is the K-way hybrid in between: each
worker column-packs its `HT_BLOCK/K` row-sums for pack-lane `p = h % K` into ONE
tile, so it ships `K` tiles per row-block (down from `HT_BLOCK` at K=HT_BLOCK, or
1 at K=1) — AND each of the K packed streams gets its OWN owner (K distinct
cores), each folding its lane's CW contributions in parallel. K = HT_BLOCK
degenerates to row_rotate's shape (ho=1, no packing, full tile-index rotation);
K = 1 degenerates to colpack's shape (one owner, full packing, no rotation).
1 < K < HT_BLOCK is the new interior point this file exists to measure.

Every variant is a topology change ONLY — same core set, same per-core W slice,
same tile-row assignment, same multicast rectangles/ack counts, same precision
contract (bf16 / TILE / HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False /
bf16 TILE gamma). The baseline (`op_current`) carries round 1's Perf-1 phase-4
`RsqrtAddUnaryColZero` fix — this lab's kernels are NOT the pre-Perf-1 snapshot
`gather_spread_topology` forked from; see the "BASELINE REFRESH" block comment in
kernels/lab_compute.cpp.

Run (one fresh-cache dispatch per case; DEVICE KERNEL DURATION [ns] = col 19):

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/combine_parallel_fold/test_combine_parallel_fold.py

Correctness is the ONLY pass/fail. Perf is measured, never asserted.
"""

import importlib.util
import pathlib

import pytest

import ttnn

# Root `device` fixture is function-scoped; this marker switches it to module
# scope so all cases share one device open (same convention as the sibling
# gather_spread_topology bench).
pytestmark = pytest.mark.use_module_device

_LAB_DIR = pathlib.Path(__file__).resolve().parent
_lab = None


def lab():
    """Load lab_descriptor.py by path (the dir is deliberately not a package)."""
    global _lab
    if _lab is None:
        spec = importlib.util.spec_from_file_location("cpf_lab_descriptor", _LAB_DIR / "lab_descriptor.py")
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
# Geometries. (id, shape, kind, shard_shape_or_None, core_grid_or_None)
# kind == "INTERLEAVED" skips shard_config entirely (plain DRAM tensor); the
# cross-core W-split placement is derived the same way test_rms_norm_perf_decode
# _pinned's interleaved decode cases are.
# ---------------------------------------------------------------------------
GEOMETRIES = [
    # FOCUS (mandatory primary target): 64 cores, cw=8 cw1=8 cw2=1 flat, per
    # core 32 tile-rows x Wt=4, ht_block=8, nh_core=4 -> HT_BLOCK/K in {8,4,2,1}.
    ("focus", (1, 1, 8192, 1024), "BLOCK", [1024, 128], (8, 8)),
    # Secondary BLOCK_SHARDED cell at ht_block=4 (nh_core=1): K in {1,2,4} only.
    ("block_ht4", (1, 1, 1024, 1024), "BLOCK", [128, 128], (8, 8)),
    # The four pinned WIDTH_SHARDED decode geometries (shard shape + core grid
    # reproduced exactly from test_rms_norm_perf.py's SHARDED_REFERENCE_NS /
    # gather_payload_shrink's CASES — NOT left to auto_shard_config). ht_block==1
    # here: colpack has nothing to pack; row_rotate/pack_rotate(K=1) can only
    # rotate the single tile-row's owner, or (K=1) degenerate to a single root.
    ("w32x1024", (1, 1, 32, 1024), "WIDTH", [32, 128], (8, 1)),
    ("w32x2304", (1, 1, 32, 2304), "WIDTH", [32, 256], (9, 1)),
    ("w32x5120", (1, 1, 32, 5120), "WIDTH", [32, 160], (8, 4)),
    ("w32x7168", (1, 1, 32, 7168), "WIDTH", [32, 256], (7, 4)),
    # The two interleaved W-split decode representatives (test_rms_norm_perf_
    # decode_pinned). kind="INTERLEAVED" -> plain DRAM tensor, no shard_config.
    ("i32x5120", (1, 1, 32, 5120), "INTERLEAVED", None, None),
    ("i32x7168", (1, 1, 32, 7168), "INTERLEAVED", None, None),
]
_GEO = {g[0]: g for g in GEOMETRIES}

# ---------------------------------------------------------------------------
# Variants per geometry. ("variant_id", topology, pack_k_or_None)
# THE HONEST BASELINE is "op_current" everywhere — whatever the op picks today
# (flat on the 1-D BLOCK groups; possibly two-stage on the widest WIDTH groups),
# never overridden. row_rotate/pack_rotate force a flat (CW2==1) fan-in by
# construction (no leader tree), so on a geometry op_current already stages this
# is an honest apples-to-the-op comparison, not apples-to-apples-topology.
# ---------------------------------------------------------------------------
VARIANTS_BY_GEO = {
    "focus": [
        ("baseline", "op_current", None),
        ("row_rotate", "row_rotate", None),
        ("pack_k1", "pack_rotate", 1),
        ("pack_k2", "pack_rotate", 2),
        ("pack_k4", "pack_rotate", 4),
        ("pack_k8", "pack_rotate", 8),
    ],
    "block_ht4": [
        ("baseline", "op_current", None),
        ("row_rotate", "row_rotate", None),
        ("pack_k1", "pack_rotate", 1),
        ("pack_k2", "pack_rotate", 2),
        ("pack_k4", "pack_rotate", 4),
    ],
}
_DEFAULT_VARIANTS = [
    ("baseline", "op_current", None),
    ("row_rotate", "row_rotate", None),
    ("pack_k1", "pack_rotate", 1),
]


def _variants_for(geo_id):
    return VARIANTS_BY_GEO.get(geo_id, _DEFAULT_VARIANTS)


_CASES = [(g_id, v[0]) for g_id in _GEO for v in _variants_for(g_id)]
_VAR_BY_GEO = {g_id: {v[0]: v for v in _variants_for(g_id)} for g_id in _GEO}


def _memcfg(device, geo):
    from eval.sharding import shard_config

    _, _shape, kind, shard_shape, core_grid = geo
    if kind == "INTERLEAVED":
        return ttnn.DRAM_MEMORY_CONFIG
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
    _, topology, pack_k = _VAR_BY_GEO[geo_id][variant_id]
    torch_x, torch_gamma, tt_x, tt_gamma, mc = _inputs(device, geo, ones=ones)
    tt_out = lab().lab_rms_norm(
        tt_x,
        gamma=tt_gamma,
        epsilon=1e-6,
        compute_kernel_config=perf_compute_kernel_config(),
        memory_config=mc,
        topology=topology,
        pack_k=pack_k,
    )
    out = ttnn.to_torch(tt_out)
    # Sharded inputs are L1-resident; free explicitly so the next case has room.
    ttnn.deallocate(tt_out)
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_gamma)
    return torch_x, torch_gamma, out


# ---------------------------------------------------------------------------
# 0. host-only: what each (geometry, variant) actually derives (no device work)
# ---------------------------------------------------------------------------


def test_report_topologies(device):
    import torch

    print()
    hdr = (
        f"{'geometry':12s} {'variant':10s} {'cores':>5s} {'cw':>3s} {'cw1':>4s} {'cw2':>4s} "
        f"{'htb':>4s} {'nhc':>4s} {'K':>3s} {'gatherKB':>9s} {'progCB_KB':>10s}"
    )
    print(hdr)
    for geo_id, geo in _GEO.items():
        mc = _memcfg(device, geo)
        shape = geo[1]
        x = torch.zeros(shape, dtype=torch.bfloat16)
        g = torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16)
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        tt_g = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for vid, topology, pack_k in _variants_for(geo_id):
            r = lab().lab_report(device, tt_x, tt_g, topology, pack_k=pack_k)
            print(
                f"{geo_id:12s} {vid:10s} {r['cores']:5d} {r['cw']:3d} {r['cw1']:4d} {r['cw2']:4d} "
                f"{r['ht_block']:4d} {r['nh_core']:4d} {r['pack_k']:3d} {r['gather_kb']:9d} {r['program_cb_kb']:10d}"
            )
        ttnn.deallocate(tt_x)
        ttnn.deallocate(tt_g)


# ---------------------------------------------------------------------------
# 1. correctness gates (mandatory, both)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("geo_id,variant_id", _CASES, ids=[f"{g}-{v}" for g, v in _CASES])
def test_pcc(device, geo_id, variant_id):
    """PCC >= 0.9995 vs torch — the focus case's soft threshold."""
    torch_x, torch_gamma, actual = _run(device, geo_id, variant_id)
    pcc = _pcc(_reference(torch_x, torch_gamma), actual)
    print(f"\nPCC {geo_id} {variant_id}: {pcc:.7f}")
    assert pcc >= 0.9995, f"{geo_id} {variant_id}: pcc {pcc}"


@pytest.mark.parametrize("geo_id,variant_id", _CASES, ids=[f"{g}-{v}" for g, v in _CASES])
def test_all_ones_absolute(device, geo_id, variant_id):
    """ABSOLUTE element-count check: all-ones must recover mean(x^2) = 1.0 exactly.

    PCC is scale-invariant and would score a rescaled-row bug (a wrong owner map,
    a dropped slot, a double-counted lane) >= 0.9998 — this op has shipped four
    such bugs. Gate copied from gather_payload_shrink/bench.py.
    """
    import torch

    _, _, actual = _run(device, geo_id, variant_id, ones=True)
    out = actual.to(torch.float32)
    want = 1.0 / (1.0 + 1e-6) ** 0.5
    err = (out - want).abs().max().item()
    implied = (out.flatten()[0].item() ** -2) - 1e-6
    print(f"\nall-ones {geo_id} {variant_id}: implied mean(x^2) = {implied:.6f}  max|err| = {err:.6f}")
    assert err < 5e-3, f"{geo_id} {variant_id}: implied mean(x^2) = {implied:.4f}, expected 1.0 (max err {err})"


# ---------------------------------------------------------------------------
# 2. the measurement — ONE fresh dispatch per (geometry, variant), no trial loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("geo_id,variant_id", _CASES, ids=[f"{g}-{v}" for g, v in _CASES])
def test_measure(device, geo_id, variant_id):
    """One dispatch. The number comes from --profile's CSV, never from a timer.

    Correctness still gated (PCC) so a perf number can never be taken off a
    wrong-but-fast kernel.
    """
    torch_x, torch_gamma, actual = _run(device, geo_id, variant_id)
    pcc = _pcc(_reference(torch_x, torch_gamma), actual)
    assert pcc >= 0.9995, f"{geo_id} {variant_id}: pcc {pcc}"


# ---------------------------------------------------------------------------
# 3. SECOND-ORDER: does a bigger L1 block budget (-> a bigger HT_BLOCK) move the
# K-curve's optimum? Round 1 found row_rotate's smaller gather CB let the focus
# shape's block grow from HT_BLOCK 8 to 16 at a 1 MB budget. Same override here,
# swept over the SAME K menu, to see whether an interior K ever beats BOTH pure
# ends (K=1 and K=HT_BLOCK) once there is more to pack per lane.
# ---------------------------------------------------------------------------


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


_BUDGET_KB = 1024  # round 1's second-order point: focus @ 1 MB -> HT_BLOCK 16
_BUDGET_VARIANTS = [
    ("baseline", "op_current", None),
    ("row_rotate", "row_rotate", None),
    ("pack_k1", "pack_rotate", 1),
    ("pack_k2", "pack_rotate", 2),
    ("pack_k4", "pack_rotate", 4),
    ("pack_k8", "pack_rotate", 8),
    ("pack_k16", "pack_rotate", 16),
]
_BUDGET_VAR = {v[0]: v for v in _BUDGET_VARIANTS}


def _run_budget(device, variant_id, *, ones=False):
    geo = _GEO["focus"]
    _, topology, pack_k = _BUDGET_VAR[variant_id]
    torch_x, torch_gamma, tt_x, tt_gamma, mc = _inputs(device, geo, ones=ones)
    with _budget(_BUDGET_KB):
        tt_out = lab().lab_rms_norm(
            tt_x,
            gamma=tt_gamma,
            epsilon=1e-6,
            compute_kernel_config=perf_compute_kernel_config(),
            memory_config=mc,
            topology=topology,
            pack_k=pack_k,
        )
    out = ttnn.to_torch(tt_out)
    ttnn.deallocate(tt_out)
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_gamma)
    return torch_x, torch_gamma, out


@pytest.mark.parametrize("variant_id", [v[0] for v in _BUDGET_VARIANTS])
def test_pcc_budget(device, variant_id):
    torch_x, torch_gamma, actual = _run_budget(device, variant_id)
    pcc = _pcc(_reference(torch_x, torch_gamma), actual)
    print(f"\nPCC focus@{_BUDGET_KB}KB {variant_id}: {pcc:.7f}")
    assert pcc >= 0.9995, f"focus@{_BUDGET_KB}KB {variant_id}: pcc {pcc}"


@pytest.mark.parametrize("variant_id", [v[0] for v in _BUDGET_VARIANTS])
def test_all_ones_budget(device, variant_id):
    import torch

    _, _, actual = _run_budget(device, variant_id, ones=True)
    out = actual.to(torch.float32)
    want = 1.0 / (1.0 + 1e-6) ** 0.5
    err = (out - want).abs().max().item()
    print(f"\nall-ones focus@{_BUDGET_KB}KB {variant_id}: max|err| = {err:.6f}")
    assert err < 5e-3, f"focus@{_BUDGET_KB}KB {variant_id}: max err {err}"


@pytest.mark.parametrize("variant_id", [v[0] for v in _BUDGET_VARIANTS])
def test_measure_budget(device, variant_id):
    torch_x, torch_gamma, actual = _run_budget(device, variant_id)
    pcc = _pcc(_reference(torch_x, torch_gamma), actual)
    assert pcc >= 0.9995, f"focus@{_BUDGET_KB}KB {variant_id}: pcc {pcc}"


def test_report_budget(device):
    geo = _GEO["focus"]
    mc = _memcfg(device, geo)
    shape = geo[1]
    tt_x = ttnn.from_torch(
        __import__("torch").zeros(shape, dtype=__import__("torch").bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    tt_g = ttnn.from_torch(
        __import__("torch").zeros(1, 1, 1, shape[-1], dtype=__import__("torch").bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    print()
    for vid, topology, pack_k in _BUDGET_VARIANTS:
        with _budget(_BUDGET_KB):
            r = lab().lab_report(device, tt_x, tt_g, topology, pack_k=pack_k)
        print(
            f"focus@{_BUDGET_KB}KB {vid:10s} cw={r['cw']:3d} htb={r['ht_block']:3d} nhc={r['nh_core']:3d} "
            f"K={r['pack_k']:3d} gatherKB={r['gather_kb']:4d} progCB_KB={r['program_cb_kb']:4d}"
        )
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_g)
