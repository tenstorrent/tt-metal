# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""`combine_latency_hiding` — isolated A/B bench for ONE idea (Perf 2 tournament).

IDEA. Per row-block the cross-core W-split's combine is a hard serialization:
publish partial -> STALL (writer gather -> root fold -> mcast) -> rsqrt ->
scale -> gamma -> write. The STALL is a period where this core's TRISCs have
NO work queued at all, while the NEXT row-block's pass A (square+accumulate)
is fully independent and already has its input resident in L1. Software-
pipeline the row-block loop so that work fills the stall instead of it being
dead time. See clh_program_descriptor.py's module docstring and
kernels/clh_compute.cpp's kernel-head comment for the exact mechanism/schedule.

VARIANTS (`CLH_VARIANT`), all under the IDENTICAL precision contract
(bf16 / HiFi2 / fp32_dest_acc_en=False / math_approx_mode=False / bf16 TILE
gamma) and the identical placement, core count and per-core slice:

  baseline      the op as it stands (byte-identical fork).
  prefetch_a    A(0); for hb {{ if hb+1 exists: A(hb+1); stall+rsqrt(hb);
                passB(hb) }}.
  defer_passb   A(0); stall+rsqrt(0); for hb {{ if hb+1 exists: A(hb+1);
                passB(hb); if hb+1 exists: stall+rsqrt(hb+1) }}.

Both are pure REORDERINGS at the fixed precision contract -- no math changes,
so both should be bit-exact vs baseline (verified below, not assumed).

`CLH_STALL_WAIT=n` is a SENSITIVITY-STUDY ABLATION (output WRONG by design):
overrides how many of the CW1 stage-1 partials the combine root actually
waits for before folding (0 = real wait = CW1). Shrinking it models "a sibling
idea (payload-shrink / rootless all-gather) shrunk the round trip" without
implementing that sibling here -- it exercises the SAME mcast/gather code
path, just satisfied earlier, so the resulting ns delta isolates how much of
THIS idea's win is latency-proportional vs fixed.

CORRECTNESS GATE (pass/fail; perf is measured, never asserted):
  * PCC >= 0.9995 vs torch fp32, and
  * an ABSOLUTE all-ones check: mean(x^2) must come back EXACTLY 1.0, so
    out == gamma / sqrt(1 + eps). A pipelined loop that mixes up which
    accumulator belongs to which row-block is exactly the rescale-only bug
    class PCC hides (this op has shipped four of them).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch

import ttnn

_HERE = Path(__file__).resolve().parent


def _load_pd():
    spec = importlib.util.spec_from_file_location("_clh_pd", _HERE / "clh_program_descriptor.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pd = _load_pd()


# --------------------------------------------------------------------------
# geometry cases: (shape, kind, shard_shape, core_grid)
# kind == "INTERLEAVED" skips shard_config; shard_shape/core_grid are unused.
# --------------------------------------------------------------------------
CASES = {
    # THE FOCUS SHAPE — feature_spec's perf-flagged BLOCK_SHARDED prefill cell.
    # 64 cores, per core 32 tile-rows x Wt=4, nw=1, ht_block=8, nh_core=4,
    # cw=cw1=8 (flat fold). achievable_ns reference: 25_640.
    "focus": ((1, 1, 8192, 1024), "BLOCK", (1024, 128), (8, 8)),
    # Secondary WIDTH_SHARDED decode geometries: ht_total == 1 => nh_core == 1,
    # NO next row-block to prefetch -- structurally inert predicate boundary.
    "w32x1024": ((1, 1, 32, 1024), "WIDTH", (32, 128), (8, 1)),
    "w32x2304": ((1, 1, 32, 2304), "WIDTH", (32, 256), (9, 1)),
    "w32x5120": ((1, 1, 32, 5120), "WIDTH", (32, 160), (8, 4)),
    "w32x7168": ((1, 1, 32, 7168), "WIDTH", (32, 256), (7, 4)),
    # Interleaved W-split DECODE reps (test_rms_norm_perf_decode_pinned).
    "i32x5120": ((1, 1, 32, 5120), "INTERLEAVED", None, None),
    "i32x7168": ((1, 1, 32, 7168), "INTERLEAVED", None, None),
    # Interleaved PREFILL reps (test_rms_norm_perf_prefill_pinned): cw == 1 (no
    # combine at all), DRAM-read-bound. Must not regress -- CLH_ELIGIBLE
    # requires sharded_in, so these always take the byte-identical fallback
    # regardless of CLH_VARIANT; measuring them anyway proves that fallback is
    # actually inert (no accidental cost from the new CT args / branch).
    "i8192x1024": ((1, 1, 8192, 1024), "INTERLEAVED", None, None),
    "i8192x7168": ((1, 1, 8192, 7168), "INTERLEAVED", None, None),
}


def perf_compute_kernel_config():
    """The PINNED precision contract. Identical for every variant — never a lever."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


def _dispatch(device, shape, mc, torch_x, torch_gamma):
    tt_x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tt_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    prog = pd.create_program_descriptor(
        tt_x,
        out,
        gamma=tt_gamma,
        epsilon=1e-6,
        compute_kernel_config=perf_compute_kernel_config(),
        device=device,
    )
    return ttnn.generic_op([tt_x, tt_gamma, out], prog)


def run_case(device, case, variant, mode="random"):
    """One fresh dispatch of one variant on one geometry. Returns nothing; the
    ns comes from --profile's ops_perf_results CSV."""
    from eval.sharding import shard_config

    shape, kind, shard_shape, core_grid = CASES[case]
    os.environ["CLH_VARIANT"] = variant

    if kind == "INTERLEAVED":
        mc = ttnn.DRAM_MEMORY_CONFIG
    else:
        memory_layout = getattr(ttnn.TensorMemoryLayout, f"{kind}_SHARDED")
        mc = shard_config(
            list(shard_shape), core_grid, memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
        )

    torch.manual_seed(42)
    if mode == "ones":
        torch_x = torch.ones(shape, dtype=torch.bfloat16)
        torch_gamma = torch.ones(shape[-1], dtype=torch.bfloat16)
    else:
        torch_x = torch.randn(shape, dtype=torch.bfloat16)
        torch_gamma = torch.randn(shape[-1], dtype=torch.bfloat16)

    if os.environ.get("CLH_REPORT"):
        print(f"[clh] {case:16s} {_report_blocking(device, shape, mc)}")
        return

    tt_out = _dispatch(device, shape, mc, torch_x, torch_gamma)
    actual = ttnn.to_torch(tt_out).to(torch.float32)

    if os.environ.get("CLH_STALL_WAIT"):
        print(f"\n[clh] {case} {variant} ABLATION (CLH_STALL_WAIT) — output WRONG by design, no gate\n")
        return

    xf = torch_x.to(torch.float32)
    if mode == "ones":
        # ABSOLUTE gate. mean(x^2) == 1.0 exactly for an all-ones input, so
        # every output element is 1/sqrt(1+eps). A pipelined loop that mixes up
        # which accumulator belongs to which row-block shows up here as a
        # per-tile-row scale error even though PCC would stay near 1.0.
        want = 1.0 / (1.0 + 1e-6) ** 0.5
        err = (actual - want).abs().max().item()
        rec = (actual.to(torch.float32) ** 2).mean().item() * (1.0 + 1e-6)
        print(
            f"\n[clh] {case} {variant} ALL-ONES: max|out - 1/sqrt(1+eps)| = {err:.6f}  (recovered mean(x^2) = {rec:.6f})"
        )
        if err >= 5e-3:
            a = actual.reshape(-1, actual.shape[-1])
            m = (1.0 / a[:, 0] ** 2) - 1e-6
            uniq = sorted(set(round(float(v), 5) for v in m))
            print(f"[clh]   per-row recovered mean(x^2): {len(uniq)} distinct -> {uniq[:8]}")
            bad = (m - 1.0).abs() > 1e-3
            idx = bad.nonzero().flatten()[:24].tolist()
            print(f"[clh]   first bad rows: {idx}   (bad fraction {bad.float().mean().item():.4f})")
        assert err < 5e-3, f"{case}/{variant}: all-ones absolute check FAILED, max err {err}"
    else:
        expected = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
        expected = expected * torch_gamma.to(torch.float32).reshape(-1)
        pcc = _pcc(expected, actual)
        print(f"\n[clh] {case} {variant} PCC = {pcc:.8f}\n")
        assert pcc >= 0.9995, f"{case}/{variant}: PCC {pcc} < 0.9995"


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()))


def _report_blocking(device, shape, mc):
    """Derived blocking + CLH knobs for one case (no perf claim; predicate bookkeeping)."""
    torch.manual_seed(0)
    torch_x = torch.randn(shape, dtype=torch.bfloat16)
    torch_gamma = torch.randn(shape[-1], dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    tt_g = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, shape[-1]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    grid = device.compute_with_storage_grid_size()
    ht_total, wt_global = pd._tile_geometry(tt_x)
    in_sharded = mc.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    placement = pd._select_placement(device, grid, tt_x, ht_total, wt_global, in_sharded)
    blk = pd._derive_blocking(
        tt_x,
        tt_g,
        grid.x * grid.y,
        placement,
        sharded_in=in_sharded,
        sharded_out=in_sharded,
        l1_total_budget=pd._l1_total_budget(device),
    )
    nh_core = -(-blk._rows_core_max // blk.ht_block)
    return (
        f"cores={placement.num_cores:3d} cw={placement.cw} cw1={placement.cw1} cw2={placement.cw2} "
        f"Wt/core={blk.Wt} nw={blk.nw} HT_BLOCK={blk.ht_block} rows/core={blk._rows_core_max} "
        f"nh_core={nh_core} fuse_sq={int(blk.fuse_sq)} clh_eligible={int(blk.clh_eligible)} "
        f"clh_variant={blk.clh_variant} clh_pipeline_depth={blk.clh_pipeline_depth} "
        f"cb_partial_out_pages={blk.clh_pipeline_depth * blk.ht_block if blk.w_split else 0} "
        f"program_cb_bytes={blk.program_cb_bytes} resident_shard_bytes={blk.resident_shard_bytes} "
        f"cb_total_bytes={blk.cb_total_bytes}"
    )
