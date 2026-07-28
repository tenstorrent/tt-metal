# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: `combine_allgather_no_root` — delete the combine round trip.

ONE idea: replace the op's two-hop cross-core combine (CW workers write their
raw partial into ONE root's cb_group_partials, the root folds CW tiles and
Mcast2D-broadcasts mean(x^2) back) with a ONE-hop all-gather: every core PULLS
all CW peers' raw partials directly out of their L1 (unicast NoC reads, no
multicast, no root, no leader) and then folds all CW tiles LOCALLY, so every
core already holds the final answer -- no broadcast-back at all. Precision
contract held fixed for every variant: bf16 / TILE / fp32_dest_acc_en=False /
HiFi2 / math_approx_mode=False / bf16 TILE gamma.

    baseline         THE HONEST BASELINE — byte-identical to the op's current
                     gather-to-root + Mcast2D-back.
    allgather_pull   THE IDEA — see lab_descriptor.py's LAB FORK docstring.

Run (one fresh-cache dispatch per case; DEVICE KERNEL DURATION [ns] = col 19):

    scripts/run_safe_pytest.sh --profile --run-all \
        tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/combine_allgather_no_root/test_combine_allgather.py

Correctness is the ONLY pass/fail. Perf is measured, never asserted.

NOTE on imports: this file lives under tests/, not ttnn/ttnn/operations/, so the
`exec_module()`-on-import hazard the sibling labs warn about does not apply
here — but the lazy-import discipline (torch imported inside tests, not at
module scope) is kept anyway for consistency with the established pattern.
"""

import importlib.util
import pathlib

import pytest

import ttnn

# Root `device` fixture is function-scoped; this marker switches it to module
# scope so all cases in this file share one device open (same precedent as
# gather_spread_topology / gather_payload_shrink).
pytestmark = pytest.mark.use_module_device

_LAB_DIR = pathlib.Path(__file__).resolve().parent
_lab = None


def lab():
    """Load lab_descriptor.py by path (the dir is deliberately not a package)."""
    global _lab
    if _lab is None:
        spec = importlib.util.spec_from_file_location("cagr_lab_descriptor", _LAB_DIR / "lab_descriptor.py")
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
# kind in {"BLOCK", "WIDTH", "INTERLEAVED"}. shard_shape/core_grid are None for
# INTERLEAVED (plain DRAM tensor, no sharding).
# ---------------------------------------------------------------------------
GEOMETRIES = [
    # MANDATORY FOCUS: 64 cores, cw=8 cw1=8 cw2=1 (flat, one grid row per group),
    # per-core 32 tile-rows x Wt=4, ht_block=8, nh_core=4. achievable_ns=25_640.
    ("focus-block-8192x1024", (1, 1, 8192, 1024), "BLOCK", [1024, 128], (8, 8)),
    # BLOCK ht-sweep at fixed cw=8, nh_core=1 (one grid row per group): where does
    # the replicated local fold stop paying vs. the two-hop root gather?
    ("block-2048x1024", (1, 1, 2048, 1024), "BLOCK", [256, 128], (8, 8)),  # ht_block 8
    ("block-1024x1024", (1, 1, 1024, 1024), "BLOCK", [128, 128], (8, 8)),  # ht_block 4
    ("block-512x1024", (1, 1, 512, 1024), "BLOCK", [64, 128], (8, 8)),  # ht_block 2
    ("block-256x1024", (1, 1, 256, 1024), "BLOCK", [32, 128], (8, 8)),  # ht_block 1
    # WIDTH decode geometries — SHARDED. All-gather at wide CW (32-56) replicates
    # the fold and multiplies traffic by CW: this IS the crossover to look for.
    ("width-32x1024", (1, 1, 32, 1024), "WIDTH", [32, 128], (8, 1)),  # CW=32
    ("width-32x2304", (1, 1, 32, 2304), "WIDTH", [32, 256], (9, 1)),  # CW=36
    ("width-32x5120", (1, 1, 32, 5120), "WIDTH", [32, 160], (8, 4)),  # CW=40
    ("width-32x7168", (1, 1, 32, 7168), "WIDTH", [32, 256], (7, 4)),  # CW=56
    # Interleaved W-split decode reps (test_rms_norm_perf_decode_pinned's shapes,
    # currently 7_804 / 8_774 ns). No shard: the W-split placement engages on the
    # interleaved tensor directly (row axis under-fills the grid at ht_total=1).
    ("interleaved-32x5120", (1, 1, 32, 5120), "INTERLEAVED", None, None),
    ("interleaved-32x7168", (1, 1, 32, 7168), "INTERLEAVED", None, None),
]

VARIANTS = [
    ("baseline", "baseline"),
    ("allgather_pull", "allgather_pull"),
]

_CASES = [(g[0], v[0]) for g in GEOMETRIES for v in VARIANTS]
_GEO = {g[0]: g for g in GEOMETRIES}
_VAR = {v[0]: v for v in VARIANTS}


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
    _, combine_variant = _VAR[variant_id]
    torch_x, torch_gamma, tt_x, tt_gamma, mc = _inputs(device, geo, ones=ones)
    out_mc = mc if geo[2] != "INTERLEAVED" else ttnn.DRAM_MEMORY_CONFIG
    tt_out = lab().lab_rms_norm(
        tt_x,
        gamma=tt_gamma,
        epsilon=1e-6,
        compute_kernel_config=perf_compute_kernel_config(),
        memory_config=out_mc,
        combine_variant=combine_variant,
    )
    out = ttnn.to_torch(tt_out)
    # Sharded inputs are L1-resident — a leaked tensor costs the next case its
    # shard. Free explicitly, not on GC.
    ttnn.deallocate(tt_out)
    ttnn.deallocate(tt_x)
    ttnn.deallocate(tt_gamma)
    return torch_x, torch_gamma, out


# ---------------------------------------------------------------------------
# 0. host-only: what each variant actually derives (no device work, no dispatch)
# ---------------------------------------------------------------------------


def test_report_variants(device):
    import torch

    print()
    hdr = f"{'geometry':22s} {'variant':16s} {'cores':>5s} {'cw':>3s} {'cw1':>4s} {'cw2':>4s} {'htb':>4s} {'nhc':>4s} {'gatherKB':>9s} {'progCB_KB':>10s}"
    print(hdr)
    for geo in GEOMETRIES:
        mc = _memcfg(device, geo)
        shape = geo[1]
        x = torch.zeros(shape, dtype=torch.bfloat16)
        g = torch.zeros(1, 1, 1, shape[-1], dtype=torch.bfloat16)
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
        tt_g = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for vid, combine_variant in VARIANTS:
            r = lab().lab_report(device, tt_x, tt_g, combine_variant)
            print(
                f"{geo[0]:22s} {vid:16s} {r['cores']:5d} {r['cw']:3d} {r['cw1']:4d} {r['cw2']:4d} "
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

    PCC is scale-invariant and would score a rescale-only bug (a missed or
    double-counted contributor in the all-gather) as passing; this is the gate
    that catches it.
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
def test_variants_agree(device, geo_id):
    """Every variant folds the SAME raw slice accumulators. Only the transport
    (root gather vs. all-gather) and the fold's association order differ, so the
    answers must agree — a structural error (dropped/double-counted contributor)
    is exactly what this catches; a differing fp32 add order is numerically
    benign and shows up only as sub-ULP bf16 noise.
    """
    import torch

    outs = {}
    for vid, _ in VARIANTS:
        _, _, outs[vid] = _run(device, geo_id, vid)
    ref = outs["baseline"].to(torch.float32)
    scale = ref.abs().max().item()
    for vid, out in outs.items():
        if vid == "baseline":
            continue
        o = out.to(torch.float32)
        pcc = _pcc(o, ref)
        worst = (o - ref).abs().max().item()
        print(f"\nagree {geo_id} {vid}: pcc-vs-baseline {pcc:.7f} max|diff| {worst:.6f} ({worst / scale:.2%} of peak)")
        assert pcc >= 0.9999, f"{geo_id}: {vid} disagrees with baseline (pcc {pcc})"


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
