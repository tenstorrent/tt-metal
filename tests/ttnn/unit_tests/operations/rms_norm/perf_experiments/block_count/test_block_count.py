# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""P5 bake-off — ONE BLOCK PER CORE: how many combine round trips does a core pay?

ISOLATED CONCEPT: `block_rows` (B), and therefore `num_blocks = ceil(rows/B)`.
Everything else — kernels, grid, shard spec, precision contract (bf16 / HiFi2 /
fp32_dest_acc_en=False / math_approx_mode=False), gamma, DM chunk, DEST window —
is held byte-identical across variants.  `baseline` is the SHIPPED plan (the L1
ladder's answer at the 1 MB fallback budget that actually runs on this box);
every candidate overrides exactly the one plan field.

Correctness is the only pass/fail; perf is measured, never asserted.

    scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/block_count/test_block_count.py

    BC_GROUPS=focus  scripts/run_safe_pytest.sh --run-all <this file>

One fresh run per (case, variant) — device kernel time has no warm-up transient.

MEASURED (Blackhole p150b @1350 MHz, bf16 / TILE / bf16 TILE gamma / HiFi2 /
fp32_dest_acc_en=False, device kernel ns, one fresh run per point).

1. `block_rows` sweep at a FIXED budget, focus (1,1,8192,1024) BLOCK [1024,128]
   (8,8) — 32 tile-rows per core, s=8, S=4:

       B          1       2       4       8      16(shipped)   32
       ns    113250   78129   51881   38549       34619     32635
       blocks    32      16       8       4           2         1

   Monotone: coarser is better and it saturates at one block.  ~2.1 us of fixed
   cost per block.  The ladder stops at 16 only because `_l1_working_budget`
   runs on the 1 MB fallback (`device.l1_size_per_core` is NOT bound on this
   box); B=32 needs a 1.12 MB budget and the part has 1,461,376 B per bank.

2. The same knob, HEIGHT-sharded (1,1,8192,256) [128,256] (8,8) — s == 1, so
   there is NO cross-core combine at all:

       B          1       2       4(shipped)
       ns     11708    9108    8384

   ~1.1 us per block survives with the combine removed, so roughly HALF the
   per-block cost is the combine round trip and half is the local pipeline
   fill/drain + per-block init/reconfig.

3. Where the win lands (zone diff, focus, B16 -> B32, ns/core):
       rd_load_total   4174 ->   29   (the 2nd block's wait to re-publish the
                                       resident-shard CB window: pure pipeline
                                       serialization, no payload)
       cp_rms_wait     9280 -> 6134   cp_scale_total(T2) 11248 -> 8968
       wr_store_wait  20726 ->19223   rd_gather_wait     13291 ->12717
       rd_bcast_send   2384 -> 1951   rd_bcast_recv       2966 -> 2513
                                      rd_bcast_wait_stat  1674 -> 1458
   i.e. one combine round trip fewer AND one pipeline fill/drain fewer.

4. THE SIGN FLIPS on ROW_MAJOR.  Same geometry, ROW_MAJOR BLOCK shard (the core
   tilizes its own sticks into a B*S-tile `cb_input_tiles` before the reduce can
   start), 2 reps, identical to <0.2%:

       B          1       2       4       8(shipped)  16      32
       ns    152885  119401   99630    96401     100469  101807

   U-shaped, minimum at the shipped B=8: past it a coarser block is a LONGER
   serial tilize fill.  Same mechanism as the interleaved DRAM read.

5. Interleaved: widening the budget WHOLESALE regresses (1,1,8192,5120) by 5.7 %
   (median of 5: 401839 -> 424912) — but NOT via `block_rows`, which stays 1.
   It moves the PARTITION SEARCH (G=55,s=2,S=80,depth2 -> G=110,s=1,S=160,
   depth1), because the admission filter `_footprint_bytes(1, ...) <= budget` is
   what bounds how fat a hidden slice may be.  Widening only the LADDER
   (`split_*` variants) leaves the interleaved profile flat: w1024 87104 ->
   86963, w5120 401839 -> 406349 (+1.1 %, inside that shape's 2.7 % run-to-run
   spread), w7168 559654 -> 555915.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bc_harness import (  # noqa: E402
    PLAN_GLOBALS,
    guard_no_ablation,
    make_hook,
    make_split_budget_hook,
    target_compute_config,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_ML = ttnn.TensorMemoryLayout

pytestmark = pytest.mark.use_module_device


# ---------------------------------------------------------------------------
# Cases.  Every sharded row is verbatim from feature_spec.LOOSE_CASES' `perf`
# group (shape, layout, shard_shape, core_grid); the interleaved rows are the
# prefill/decode perf profiles.
# ---------------------------------------------------------------------------
CASES = {
    # focus
    "focus_block_8192x1024": dict(
        shape=(1, 1, 8192, 1024), layout=_ML.BLOCK_SHARDED, shard=[1024, 128], grid=(8, 8), group="focus"
    ),
    # WIDTH-sharded decode guard set (shard_rows == 1 -> B is already 1)
    "wshard_w1024": dict(shape=(1, 1, 32, 1024), layout=_ML.WIDTH_SHARDED, shard=[32, 128], grid=(8, 1), group="wshard"),
    "wshard_w2304": dict(shape=(1, 1, 32, 2304), layout=_ML.WIDTH_SHARDED, shard=[32, 256], grid=(9, 1), group="wshard"),
    "wshard_w5120": dict(shape=(1, 1, 32, 5120), layout=_ML.WIDTH_SHARDED, shard=[32, 160], grid=(8, 4), group="wshard"),
    "wshard_w7168": dict(shape=(1, 1, 32, 7168), layout=_ML.WIDTH_SHARDED, shard=[32, 256], grid=(7, 4), group="wshard"),
    # interleaved prefill (where Refinement 4 measured the budget-widening regression)
    "ileaved_prefill_w1024": dict(shape=(1, 1, 8192, 1024), layout=None, group="ileaved_prefill"),
    "ileaved_prefill_w5120": dict(shape=(1, 1, 8192, 5120), layout=None, group="ileaved_prefill"),
    "ileaved_prefill_w7168": dict(shape=(1, 1, 8192, 7168), layout=None, group="ileaved_prefill"),
    # interleaved decode
    "ileaved_decode_w1024": dict(shape=(1, 1, 32, 1024), layout=None, group="ileaved_decode"),
    "ileaved_decode_w7168": dict(shape=(1, 1, 32, 7168), layout=None, group="ileaved_decode"),
    # sharded BLOCK geometries AROUND the focus shape (domain sweep: does the
    # answer depend on how many tile-rows a core owns / how wide the slice is?)
    "block_4096x1024": dict(
        shape=(1, 1, 4096, 1024), layout=_ML.BLOCK_SHARDED, shard=[512, 128], grid=(8, 8), group="block_sweep"
    ),
    "block_8192x2048": dict(
        shape=(1, 1, 8192, 2048), layout=_ML.BLOCK_SHARDED, shard=[1024, 256], grid=(8, 8), group="block_sweep"
    ),
    "block_16384x1024": dict(
        shape=(1, 1, 16384, 1024), layout=_ML.BLOCK_SHARDED, shard=[2048, 128], grid=(8, 8), group="block_sweep"
    ),
    # HEIGHT-sharded: s == 1, so there is NO cross-core combine at all.  This is
    # the control that separates "fewer combine round trips" from "fewer
    # pipeline fill/drains + fewer init passes".
    "hshard_8192x256": dict(
        shape=(1, 1, 8192, 256), layout=_ML.HEIGHT_SHARDED, shard=[128, 256], grid=(8, 8), group="hshard"
    ),
    # ROW_MAJOR — a DIFFERENT ladder (`_plan_sharded`'s rm-depth rungs, and on the
    # interleaved side a depth-1-only block ladder), so the budget reaches it too.
    "rm_block_8192x1024": dict(
        shape=(1, 1, 8192, 1024), layout=_ML.BLOCK_SHARDED, shard=[1024, 128], grid=(8, 8), group="rm", rm=True
    ),
    "rm_wshard_32x1024": dict(
        shape=(1, 1, 32, 1024), layout=_ML.WIDTH_SHARDED, shard=[32, 128], grid=(8, 1), group="rm", rm=True
    ),
    "rm_ileaved_prefill_w1024": dict(shape=(1, 1, 8192, 1024), layout=None, group="rm", rm=True),
    "rm_ileaved_decode_w1024": dict(shape=(1, 1, 32, 1024), layout=None, group="rm", rm=True),
}

# variant -> (force_block_rows, l1_mb, note).  `None` = leave the shipped ladder
# alone.  The budget variants are the SHIPPABLE forms; the forced-B variants are
# the mechanism sweep (same budget, only B moves).
VARIANTS = {
    "baseline": dict(force_b=None, l1_mb=None),
    "budget_1.2mb": dict(force_b=None, l1_mb=1.2),
    "budget_1.46mb": dict(force_b=None, l1_mb=1.46),
    "B1": dict(force_b=1, l1_mb=None),
    "B2": dict(force_b=2, l1_mb=None),
    "B4": dict(force_b=4, l1_mb=None),
    "B8": dict(force_b=8, l1_mb=None),
    "B16": dict(force_b=16, l1_mb=None),
    "B32": dict(force_b=32, l1_mb=None),
    "B64": dict(force_b=64, l1_mb=None),
    # The NARROW form: the real L1 reaches only the block ladder; the
    # interleaved partition-search admission filter keeps the conservative
    # budget (which is what actually moved on (1,1,8192,5120)).
    "split_1.46mb": dict(force_b=None, l1_mb=None, split=(1.0, 1.46)),
    "split_1.2mb": dict(force_b=None, l1_mb=None, split=(1.0, 1.2)),
    # SHIPPABLE forms of the same thing, at budgets that are actually legal:
    #   1464 KB = min(MEM_L1_SIZE) over wormhole (1464K) and blackhole (1536K)
    #   1427 KB = the per-bank size this BH p150b's allocator reports (1461376 B)
    "split_1464kb": dict(force_b=None, l1_mb=None, split=(1.0, 1464 / 1024.0)),
    "split_1427kb": dict(force_b=None, l1_mb=None, split=(1.0, 1461376 / 1048576.0)),
}

DEFAULT_VARIANTS = ("baseline", "budget_1.2mb", "budget_1.46mb")


def _selected(env, default, allowed):
    names = tuple(p for p in os.environ.get(env, ",".join(default)).split(",") if p)
    unknown = set(names) - set(allowed)
    if unknown:
        raise ValueError(f"unknown {env}: {sorted(unknown)}")
    return names


def _read_kernel_ns(device):
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


def _make_tensors(device, case):
    shape = case["shape"]
    tt_layout = ttnn.ROW_MAJOR_LAYOUT if case.get("rm") else ttnn.TILE_LAYOUT
    torch.manual_seed(42)
    torch_x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    torch_gamma = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)

    if case["layout"] is None:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        from eval.sharding import shard_config

        memory_config = shard_config(
            case["shard"],
            case["grid"],
            case["layout"],
            layout=tt_layout,
            dtype=ttnn.bfloat16,
            device=device,
        )
    x = ttnn.from_torch(torch_x, dtype=ttnn.bfloat16, layout=tt_layout, device=device, memory_config=memory_config)
    gamma = ttnn.from_torch(
        torch_gamma, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    xf = torch_x.to(torch.float32)
    expected = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    expected = expected * torch_gamma.to(torch.float32).reshape(-1)
    return x, gamma, expected


_GROUPS = tuple(dict.fromkeys(c["group"] for c in CASES.values()))


@pytest.mark.parametrize("case_name", tuple(CASES))
def test_block_count(device, case_name):
    guard_no_ablation()
    case = CASES[case_name]
    groups = _selected("BC_GROUPS", _GROUPS, _GROUPS)
    if case["group"] not in groups:
        pytest.skip(f"group {case['group']} not selected")
    variants = _selected("BC_VARIANTS", DEFAULT_VARIANTS, VARIANTS)

    from ttnn.operations.rms_norm import rms_norm

    x, gamma, expected = _make_tensors(device, case)
    rows = []
    failures = []
    try:
        for variant in variants:
            spec = VARIANTS[variant]
            saved_plan = PLAN_GLOBALS["_plan"]
            saved_l1 = PLAN_GLOBALS["L1_SIZE_PER_CORE_FALLBACK"]
            try:
                if spec["l1_mb"] is not None:
                    PLAN_GLOBALS["L1_SIZE_PER_CORE_FALLBACK"] = int(spec["l1_mb"] * 1024 * 1024)
                if spec.get("split") is not None:
                    search_mb, ladder_mb = spec["split"]
                    PLAN_GLOBALS["_plan"] = make_split_budget_hook(
                        f"{case_name}/{variant}",
                        search_mb=search_mb,
                        ladder_mb=ladder_mb,
                        force_block_rows=spec["force_b"],
                    )
                else:
                    PLAN_GLOBALS["_plan"] = make_hook(f"{case_name}/{variant}", force_block_rows=spec["force_b"])
                try:
                    out_t = rms_norm(
                        x,
                        gamma=gamma,
                        compute_kernel_config=target_compute_config(),
                        memory_config=x.memory_config(),
                    )
                except RuntimeError as exc:
                    msg = " | ".join(l.strip() for l in str(exc).splitlines() if l.strip())[:200]
                    rows.append((variant, None, float("nan"), f"INFEASIBLE: {msg}"))
                    continue
                ttnn.synchronize_device(device)
                ns, nprog = _read_kernel_ns(device)
                got = ttnn.to_torch(out_t).to(torch.float32)
                out_t.deallocate()
            finally:
                PLAN_GLOBALS["_plan"] = saved_plan
                PLAN_GLOBALS["L1_SIZE_PER_CORE_FALLBACK"] = saved_l1
            a, b = got.flatten(), expected.flatten()
            pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
            rows.append((variant, ns, pcc, f"nprog={nprog}"))
            # The perf group's soft gate.  Block size does not change the
            # arithmetic, so a miss here is a real bug, not a precision trade.
            if not pcc > 0.9995:
                failures.append(f"{case_name}/{variant}: pcc={pcc}")
    finally:
        x.deallocate()
        gamma.deallocate()

    base = next((ns for name, ns, *_ in rows if name == variants[0]), None)
    logger.info(f"\n=== BC {case_name} {case['shape']} {case['layout']} ===")
    for variant, ns, pcc, note in rows:
        if ns is None:
            logger.info(f"BC {case_name:24s} {variant:14s}       n/a        -   {note}")
            continue
        rel = f"{base / ns:.3f}x" if base else "-"
        logger.info(f"BC {case_name:24s} {variant:14s} {ns:10.0f} ns  {rel:>7}  pcc={pcc:.6f}  {note}")

    assert not failures, "; ".join(failures)
