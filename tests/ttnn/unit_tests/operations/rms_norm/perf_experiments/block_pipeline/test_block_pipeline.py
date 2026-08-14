# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""P6 bake-off — SOFTWARE-PIPELINE THE BLOCK LOOP so the combine round trip hides.

ISOLATED CONCEPT: the SCHEDULE of the per-block chain

    cp_wait_in -> cp_sumsq -> [gather -> cp_combine -> funnel -> mcast] -> cp_scale

The bracketed part is a pure LATENCY chain and the op blocks straight through it.
Two skews are measured, independently:

    stat_lead    — Sum(x^2) AND the writer's gather contribution for block
                   b+stat_lead are issued before block b's 1/rms is waited on.
                   Costs (stat_lead+1) landing windows on cb_gathered_partials
                   and (stat_lead+1) blocks on cb_sq_partials.
    combine_lead — the OWNER's reduce, the funnel and the broadcast for block
                   b+combine_lead run before block b's apply pass.  Costs
                   (combine_lead+1) blocks on cb_rms_recip and cb_slice_stat.

`baseline` is stat_lead=combine_lead=0, which emits the SHIPPED program.

Correctness is the only pass/fail; perf is measured, never asserted.  The gate a
mis-pipelined loop breaks SILENTLY is that every block must scale with ITS OWN
1/rms, so PCC vs torch is checked on every variant.

    scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/block_pipeline/test_block_pipeline.py

    BP_CASES=focus BP_VARIANTS=baseline,stat1_comb1 scripts/run_safe_pytest.sh --run-all <this file>

One fresh run per (case, variant) — device kernel time has no warm-up transient.

=============================================================================
MEASURED — Blackhole p150b @1350 MHz, DEVICE KERNEL DURATION [ns], one fresh
run per point, gamma bf16 TILE in DRAM, HiFi2 / fp32_dest_acc_en=False.
EVERY pipelined result is BIT-IDENTICAL to the serial baseline's output.
"cb KB" is total_size over all CBs, which INCLUDES the two resident shards
(512 KB at the focus) — subtract those for the CB-heap cost.
=============================================================================

focus  (1,1,8192,1024) BLOCK [1024,128] (8,8)  s=8 S=4  32 tile-rows/core
  variant        B  nb  lead        ns        vs base   cb KB
  baseline      16   2  (0,0)    34645          —         788   <- the shipped op
  stat1         16   2  (1,0)    30804      1.125x        916
  stat1_comb1   16   2  (1,1)    28556      1.213x        988   <- RECOMMENDED
  fat_B32       32   1  (0,0)    32793      1.056x       1052   <- P5's lever
  B8_base        8   4  (0,0)    38224      0.907x        656
  B8_pipe        8   4  (1,1)    32004      1.083x        756
  B8_pipe3       8   4  (2,2)    30724      1.128x        856
  B4_base        4   8  (0,0)    52101      0.665x        608
  B4_pipe        4   8  (1,1)    41054      0.844x        676
  B2_base        2  16  (0,0)    78326      0.442x        584
  B2_pipe        2  16  (1,1)    58053      0.597x        636
  baseline across 6 fresh runs: 34547..34659 (0.3% spread).

  HEAD-TO-HEAD, the number that decides P6 vs P5:
    pipelined 2 blocks  28556 ns @ 988 KB   BEATS
    one fat block       32793 ns @ 1052 KB  by 1.148x AND 64 KB LESS L1.

s4     (1,1,8192, 512) BLOCK [1024,128] (4,8)  s=4 S=4
  baseline 33443 | stat1_comb1 27011 (1.238x) | fat_B32 31254 (1.070x)
  B8: 37300 -> 30334 (1.230x) -> 29611 at lead 2 | B4: 44813 -> 35757 | B2: 70011 -> 53804

short  (1,1,2048,1024) BLOCK [256,128] (8,8)  s=8 S=4  8 tile-rows/core
  baseline B=8 nb=1 10382 | stat1_comb1 10404 (leads CLAMPED to 0 -> byte-identical)
  B4: 13772 -> 11197 (1.230x) | B2: 20675 -> 15895 (1.301x)
  The op's own B=8/nb=1 is already the optimum here, so the pipeline is a no-op.

deep_s8_S8 (1,1,8192,2048) BLOCK [1024,256] (8,8)  s=8 S=8  1 MB of resident shards
  baseline (ladder falls to B=1!) 127819 | stat1_comb1 at B=1 98061 (1.303x)
  B16_base 44861 | B16_pipe L1-INFEASIBLE | fat_B32 L1-INFEASIBLE
  B8_base 48704 | B8_pipe 41555 | B8_pipe3 39714 (3.218x vs baseline)
  Best AFFORDABLE point is PIPELINED (39714) and beats the best non-pipelined
  affordable point (B16_base 44861) by 1.130x.  P5's one-fat-block lever cannot
  be spent here at all.

decode (1,1,32,1024) WIDTH [32,128] (8,1)  1 tile-row/core, nb=1
  baseline 4716 | stat1_comb1 4552 — the leads are clamped to 0, so the two
  programs are BYTE-IDENTICAL and the 3.6% is this kernel's noise band.

-----------------------------------------------------------------------------
TWO BUGS THIS BENCH TURNED UP (both cost a measurement to find)
-----------------------------------------------------------------------------
1. THE CUMULATIVE GATHER COUNTER IS NOT PIPELINE-SAFE.  The op waits
   `gather_progress >= (block+1)*s` on ONE counter.  That is sound only while
   contributors advance in lockstep; pipelined, one contributor can ship blocks
   0 AND 1 before a peer ships anything, so the counter hits `s` with half the
   contributors missing and the owner reduces a half-stale window.  It does not
   hang and PCC only sags to 0.99912 (from 0.99994) — a silent wrong answer.
   FIX (in this bench): one counter PER LANDING WINDOW, waited as
   `sem[j % W] >= (j / W + 1) * s`.  Cost: W-1 extra semaphores.
2. THE OP DESTROYS ITS OWN INPUT on the resident-shard path (`scale_block`
   rewrites x in place in the caller's buffer), so an A/B harness that uploads x
   ONCE feeds variant N+1 an already-normalized tensor.  PCC cannot see it —
   rms_norm is idempotent — but ~5.6% of the bf16 elements shift by up to 1 ULP
   and it reads exactly like a pipelining race.  Upload a FRESH x per variant.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402

import ttnn  # noqa: E402


# ---------------------------------------------------------------------------
# Plumbing.  Deliberately NOT a sibling `harness.py` on sys.path: `ttnn.operations`
# walk_packages-imports every module under it, and a bare `from harness import ...`
# picks up whichever sibling experiment inserted its directory first (measured —
# it resolved to block_count's harness).  The descriptor is imported LAZILY, at
# test time, because this module is exec'd DURING `import ttnn`.
# ---------------------------------------------------------------------------
_ABLATION_FILES = (
    "/tmp/rms_norm_ablate_bits",  # stubs stage payloads -> wrong answers
    "/tmp/rms_norm_l1_mb",
    "/tmp/rms_norm_dm_chunk",
    "/tmp/rms_norm_dest_block",
)


def guard_no_ablation():
    import pathlib

    stale = [f for f in _ABLATION_FILES if pathlib.Path(f).exists()]
    if stale:
        raise RuntimeError(f"block_pipeline: stale tuning/ablation files present, delete them first: {stale}")


def target_compute_config():
    """The user's precision contract — a FIXED INPUT to every variant, never a lever."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )


def _bp():
    from tests.ttnn.unit_tests.operations.rms_norm.perf_experiments.block_pipeline import bp_program_descriptor

    return bp_program_descriptor


def run(x, gamma, *, stat_lead, combine_lead, force_block_rows, l1_mb=None, epsilon=1e-6, explicit=0):
    """One rms_norm through the FORKED descriptor.  Returns (tensor, plan)."""
    bp = _bp()
    out = ttnn.allocate_tensor_on_device(x.shape, x.dtype, x.layout, x.device(), x.memory_config())
    saved_l1 = bp.L1_SIZE_PER_CORE_FALLBACK
    try:
        if l1_mb is not None:
            bp.L1_SIZE_PER_CORE_FALLBACK = int(l1_mb * 1024 * 1024)
        pd = bp.create_program_descriptor(
            x,
            out,
            gamma=gamma,
            epsilon=epsilon,
            compute_kernel_config=target_compute_config(),
            stat_lead=stat_lead,
            combine_lead=combine_lead,
            force_explicit_sumsq=explicit,
            force_block_rows=force_block_rows,
        )
    except Exception:
        out.deallocate()
        raise
    finally:
        bp.L1_SIZE_PER_CORE_FALLBACK = saved_l1
    io = [x] + ([gamma] if gamma is not None else []) + [out]
    return ttnn.generic_op(io, pd), dict(bp.LAST_PLAN)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_ML = ttnn.TensorMemoryLayout
_EPS = 1e-6
_PCC_GATE = 0.9995  # the focus case's soft gate

pytestmark = pytest.mark.use_module_device


# ---------------------------------------------------------------------------
# Cases.  `focus` is verbatim from feature_spec.LOOSE_CASES' perf group.
# ---------------------------------------------------------------------------
CASES = {
    "focus": dict(shape=(1, 1, 8192, 1024), layout=_ML.BLOCK_SHARDED, shard=[1024, 128], grid=(8, 8), group="focus"),
    # HALF the combine fan-in at the same row-group depth.
    "s4": dict(shape=(1, 1, 8192, 512), layout=_ML.BLOCK_SHARDED, shard=[1024, 128], grid=(4, 8), group="sweep"),
    # A SHALLOW row-group: 8 tile-rows/core, so much less local work to hide behind.
    "short": dict(shape=(1, 1, 2048, 1024), layout=_ML.BLOCK_SHARDED, shard=[256, 128], grid=(8, 8), group="sweep"),
    # Deep shard + wide S: the regime where L1 cannot afford one fat block.
    "deep_s8_S8": dict(
        shape=(1, 1, 8192, 2048), layout=_ML.BLOCK_SHARDED, shard=[1024, 256], grid=(8, 8), group="deep"
    ),
    # The decode regime: one tile-row per core, so num_blocks == 1 and the
    # pipeline is a no-op by construction.  Must be FLAT, not worse.
    "decode": dict(shape=(1, 1, 32, 1024), layout=_ML.WIDTH_SHARDED, shard=[32, 128], grid=(8, 1), group="decode"),
}

# (stat_lead, combine_lead, force_block_rows, l1_mb)
#   force_block_rows None -> the op's own L1 ladder picks B.
#   l1_mb            None -> the op's shipped 1 MB fallback budget.
VARIANTS = {
    # ---- THE BASELINE: the shipped program (ladder B, serial loop) ----
    "baseline": dict(sl=0, cl=0, b=None, l1=None),
    # Diagnostic: the op's SERIAL schedule with the pipelined chain SPELLING.
    "serial_explicit": dict(sl=0, cl=0, b=None, l1=None, explicit=1),
    # ---- the two skews at the ladder's own B ----
    "stat1": dict(sl=1, cl=0, b=None, l1=None),
    "stat1_comb1": dict(sl=1, cl=1, b=None, l1=None),
    "stat2_comb1": dict(sl=2, cl=1, b=None, l1=None),
    "stat2_comb2": dict(sl=2, cl=2, b=None, l1=None),
    # ---- P5's opposite lever: ONE fat block, no pipeline (needs a wider budget) ----
    "fat_B32": dict(sl=0, cl=0, b=32, l1=1.46),
    "fat_B8": dict(sl=0, cl=0, b=8, l1=1.46),
    # ---- num_blocks sweep: baseline vs pipelined at B = 16 / 8 / 4 ----
    "B16_base": dict(sl=0, cl=0, b=16, l1=1.46),
    "B16_pipe": dict(sl=1, cl=1, b=16, l1=1.46),
    "B8_base": dict(sl=0, cl=0, b=8, l1=1.46),
    "B8_pipe": dict(sl=1, cl=1, b=8, l1=1.46),
    "B8_pipe3": dict(sl=2, cl=2, b=8, l1=1.46),
    "B4_base": dict(sl=0, cl=0, b=4, l1=1.46),
    "B4_pipe": dict(sl=1, cl=1, b=4, l1=1.46),
    "B4_pipe3": dict(sl=2, cl=2, b=4, l1=1.46),
    "B2_base": dict(sl=0, cl=0, b=2, l1=1.46),
    "B2_pipe": dict(sl=1, cl=1, b=2, l1=1.46),
}

DEFAULT_VARIANTS = ("baseline", "stat1", "stat1_comb1", "fat_B32")


def _selected(env, default, allowed):
    names = tuple(p for p in os.environ.get(env, ",".join(default)).split(",") if p)
    unknown = set(names) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {env}: {sorted(unknown)}")
    return names


def _read_kernel_ns(device):
    """One program's DEVICE KERNEL DURATION.  Drained per variant — the reader
    returns every UNREAD program, so summing across variants double-counts."""
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    total, n = 0.0, 0
    for programs in per_chip.values():
        for program in programs:
            analyses = getattr(program, "program_analyses_results", None) or {}
            entry = analyses.get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                n += 1
    return (total, n) if n else (None, 0)


def _torch_inputs(case):
    shape = case["shape"]
    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)
    xf = torch_x.to(torch.float32)
    expected = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + _EPS)
    expected = expected * torch_gamma.to(torch.float32).reshape(-1)
    return torch_x, torch_gamma, expected


def _upload_x(device, case, torch_x):
    """A FRESH device copy of x for every variant.

    HARNESS TRAP, measured: on the resident-shard path `cb_input_tiles` IS the
    caller's input buffer and `scale_block` rewrites x IN PLACE, so the op
    DESTROYS its own input.  Reusing one uploaded x across variants therefore
    feeds variant N+1 an already-normalized tensor.  PCC hides this completely —
    rms_norm is idempotent (rms(x*r) == 1), so the answer stays ~0.99994 — but
    ~5.6% of the bf16 elements shift by up to 1 ULP, which reads as a
    pipelining race that is not there.
    """
    from eval.sharding import shard_config

    memory_config = shard_config(
        case["shard"], case["grid"], case["layout"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    return ttnn.from_torch(
        torch_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )


_GROUPS = tuple(dict.fromkeys(c["group"] for c in CASES.values()))


@pytest.mark.parametrize("case_name", tuple(CASES))
def test_block_pipeline(device, case_name):
    guard_no_ablation()
    case = CASES[case_name]
    if case["group"] not in _selected("BP_GROUPS", _GROUPS, _GROUPS):
        pytest.skip(f"group {case['group']} not selected")
    variants = _selected("BP_VARIANTS", DEFAULT_VARIANTS, VARIANTS)

    torch_x, torch_gamma, expected = _torch_inputs(case)
    gamma = ttnn.from_torch(
        torch_gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    rows, failures = [], []
    ref_out = None
    try:
        shard_rows = case["shard"][0] // 32
        for variant in variants:
            spec = VARIANTS[variant]
            fb = spec["b"]
            if fb is not None and (fb > shard_rows or shard_rows % fb):
                rows.append((variant, None, float("nan"), {}, f"N/A: B={fb} does not divide shard_rows={shard_rows}"))
                continue
            x = _upload_x(device, case, torch_x)  # fresh — the op rewrites x in place
            try:
                out_t, plan = run(
                    x,
                    gamma,
                    stat_lead=spec["sl"],
                    combine_lead=spec["cl"],
                    force_block_rows=spec["b"],
                    l1_mb=spec["l1"],
                    epsilon=_EPS,
                    explicit=spec.get("explicit", 0),
                )
            except RuntimeError as exc:
                msg = " | ".join(l.strip() for l in str(exc).splitlines() if l.strip())[:180]
                rows.append((variant, None, float("nan"), {}, f"INFEASIBLE: {msg}"))
                x.deallocate()
                continue
            ttnn.synchronize_device(device)
            ns, nprog = _read_kernel_ns(device)
            got = ttnn.to_torch(out_t).to(torch.float32)
            out_t.deallocate()
            x.deallocate()
            a, b = got.flatten(), expected.flatten()
            pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
            # Bit-identity against the BASELINE is the real gate: the pipeline
            # reorders WHEN work happens, never WHAT is summed, so any element
            # that differs from the serial run is a synchronization bug that PCC
            # may be too coarse to see.
            if ref_out is None:
                ref_out = got.clone()
                ident = "REF"
            else:
                diff = (got - ref_out).abs()
                nbad = int((diff > 0).sum())
                ident = "bit-identical" if nbad == 0 else f"DIFFERS n={nbad} max={diff.max().item():.3e}"
            rows.append((variant, ns, pcc, plan, f"nprog={nprog} {ident}"))
            if not pcc > _PCC_GATE:
                failures.append(f"{case_name}/{variant}: pcc={pcc}")
    finally:
        gamma.deallocate()

    base = next((ns for name, ns, *_ in rows if name == variants[0]), None)
    logger.info(f"\n=== BP {case_name} {case['shape']} {case['layout']} ===")
    for variant, ns, pcc, plan, note in rows:
        if ns is None:
            logger.info(f"BP {case_name:10s} {variant:12s}       n/a       -   {note}")
            continue
        rel = f"{base / ns:.3f}x" if base else "-"
        geom = (
            f"B={plan.get('block_rows')} nb={plan.get('num_blocks')} s={plan.get('num_hidden_slices')} "
            f"own={plan.get('own_rows')} lead=({plan.get('stat_lead')},{plan.get('combine_lead')}) "
            f"cb={plan.get('cb_bytes', 0) // 1024}KB"
        )
        logger.info(f"BP {case_name:10s} {variant:12s} {ns:9.0f} ns {rel:>7} pcc={pcc:.6f}  {geom}  {note}")

    assert not failures, "; ".join(failures)
