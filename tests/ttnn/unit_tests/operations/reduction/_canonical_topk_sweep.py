# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Canonical device-kernel-runtime sweep for the top-k / sorting op family.

WHY THIS EXISTS
---------------
The branch has kernel-level (MATH_ISOLATE, cyc/vector) numbers from the tt-llk
perf drivers and a single op-level Tracy baseline for ttnn.topk. Neither is a
table you can put in front of a reviewer: the LLK numbers exclude unpack/pack/
dispatch, and the op-level baseline covers one op at a handful of shapes with
no A/B arm. This script produces ONE table across ops x (N, K, dtype) x arms,
at both layers:

  - ttnn layer : DEVICE KERNEL DURATION [ns] from Tracy, per (op, N, K, dtype).
  - llk layer  : cyc/vector from tt_metal/tt-llk/perf_data/*.post.csv
                 (perf_topk_rebuild_xl.py carries baseline and ours as paired
                 arms in one run, so it needs no header flip).

MEASUREMENT DISCIPLINE (inherited from _topk_sort_bench.py, and extended)
-------------------------------------------------------------------------
- Device Kernel Duration from Tracy, never time.perf_counter.
- Every cell runs in a FRESH subprocess under a watchdog: a hang in one
  (shape, k) cannot kill the sweep, and each Tracy report contains exactly
  one cell, which makes CSV attribution unambiguous.
- Per-config attribution joins on OP CODE and takes the LAST `iters` rows by
  GLOBAL CALL COUNT, dropping the cache-miss/warmup invocations.
- mean +/- std across `--trials` independent subprocess runs; a speedup is
  printed only when |delta| > 2 * pooled_std, otherwise "~1.00 (noise)".
- Unsupported configurations are RECORDED WITH THEIR ERROR, not silently
  dropped. Cells the constraint model predicts as unsupported are still
  attempted once in the baseline arm: the real error message is the datum,
  and a prediction/reality mismatch is itself a finding.
- The A/B arms flip `#define TOPK_REPLAY_STEP_{LOAD,STORE}` at the top of
  tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h. There is no
  env-var -> kernel-define mechanism in tt-metal (checked: the topk program
  factories pass no compute defines; jit_build injects defines only from
  rtoptions), so the header edit IS the mechanism. It is guarded behind
  --allow-header-edit, asserted against `git diff`, restored on exit, and the
  JIT kernel cache is cleared between arms as the hard guarantee (the dephash
  sidecar would catch the header change anyway, but a guarantee beats a
  mechanism you have to trust).

USAGE
-----
  # (a) baseline-only, ttnn layer
  python tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py \
      --arms baseline --layers ttnn --out generated/canonical_sweep/run1

  # (b) full three-arm A/B, both layers
  python tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py \
      --arms baseline,replay_load,replay_store --layers ttnn,llk \
      --allow-header-edit --out generated/canonical_sweep/run1

  # (c) report-only (rebuild CSV+markdown from an existing out dir)
  python tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py \
      --report --out generated/canonical_sweep/run1

  # resume an interrupted run (skips cells already MEASURED/UNSUPPORTED)
  ... --resume --out generated/canonical_sweep/run1

The child measurement process is this same file: the orchestrator launches
`python -m tracy -r -v <this file>` with CANONICAL_SWEEP_CHILD_SPEC pointing
at a JSON cell spec. Args ride an env var, not the command line, because
tracy -r re-invokes via shell=True and mangles anything with spaces
(docs/profiling.md).

Underscore-prefixed so routine `pytest tests/...` does not collect it.
"""

import argparse
import csv
import glob
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time

REPO = os.environ.get("TT_METAL_HOME", "/home/nachiket/tt-metal")
SCRIPT_PATH = os.path.abspath(__file__)

CHILD_SPEC_ENV = "CANONICAL_SWEEP_CHILD_SPEC"

# ---------------------------------------------------------------------------
# A/B arms: the LLK header edit.
#
# The #ifdef sites live inside _bitonic_topk_phases_steps, which is the
# function ttnn.topk's topk_local_sort actually executes; a JIT header change
# needs no host rebuild (tt_metal/tt-llk/common is on the JIT include path,
# jit_build/build.cpp:348). TOPK_REPLAY_STEP_STORE implies _LOAD inside the
# header (ckernel_sfpu_topk.h:64-65), so the arms are strictly ordered:
# baseline < replay_load < replay_store.
# ---------------------------------------------------------------------------
HEADER_RELPATH = "tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_topk.h"
MARKER_BEGIN = "// BEGIN CANONICAL_SWEEP_ARM (auto-managed by _canonical_topk_sweep.py;"
MARKER_BEGIN_FULL = MARKER_BEGIN + " a stray copy means a sweep died mid-arm -- safe to delete this block)"
MARKER_END = "// END CANONICAL_SWEEP_ARM"
ARM_DEFINES = {
    "baseline": None,
    "replay_load": "#define TOPK_REPLAY_STEP_LOAD 1",
    "replay_store": "#define TOPK_REPLAY_STEP_STORE 1",
}
KERNEL_CACHE_DIR = os.path.expanduser("~/.cache/tt-metal-cache")

# Blackhole user-visible L1 per core; feeds the a-priori single-core cost model.
BH_L1_BYTES = 1_572_864
TILE_BYTES = {"bf16": 2048, "fp32": 4096, "uint16": 2048, "uint32": 4096}


# Tracy OP CODE is the device-operation struct name. Verified against the
# 2026_08_16 baseline CSV (TopKDeviceOperation) and the C++ struct names
# (SortDeviceOperation, TopkLargeIndicesDeviceOperation,
# GeneralizedMoeGateDeviceOperation). Matched loosely so a rename does not
# silently zero the table; FillPad/Pad rows must never match.
def _op_code_matches(op, op_code):
    s = op_code.lower().replace("_", "")
    if "pad" in s:
        return False
    if op == "topk":
        return s.startswith("topk") and "large" not in s
    if op == "sort":
        return "sort" in s
    if op == "topk_large_indices":
        return "topklargeindices" in s
    if op == "moe_gate":
        return "moegate" in s
    return False


# ---------------------------------------------------------------------------
# Grid + a-priori constraint model.
#
# Sources (audited on this tree, quoted by file:line so drift is checkable):
#   ttnn.topk   ttnn/cpp/ttnn/operations/reduction/topk/device/topk_device_operation.cpp
#               multi-core iff W>=8192 (:66) AND W<65535 strict (:70) AND
#               pow2(W) (:72) AND k<=64 (:75) AND find_topk_core_config fits.
#               Single-core L1 wall: verify_single_core_cost, topk_utils.cpp:220.
#   ttnn.sort   ttnn/cpp/ttnn/operations/data_movement/sort/device/sort_device_operation.cpp
#               host pads W to next pow2 (min 64); Wt<=64 -> single core,
#               else hybrid/DRAM multi-core. dtype in {bf16, uint16, fp32}.
#   topk_large_indices  .../experimental/topk_large_indices/device/..._device_operation.cpp
#               Blackhole only, ROW_MAJOR BFLOAT16 interleaved,
#               k in [16, 2048] multiples of 16, N in [k, 2^30].
#   moe_gate    .../experimental/deepseek/moe/generalized_moe_gate/device/...
#               topk in {4, 6, 8}, bf16, sharded, 256 experts fixed geometry.
# ---------------------------------------------------------------------------
DEFAULT_NS = "1024,8192,32768,65534,65536,131072"
DEFAULT_KS = "8,16,32,64,128,256,512,1024,2048"
DEFAULT_DTYPES = "bf16,fp32"

# The two shipped production perf anchors for topk_large_indices
# (tests/ttnn/nightly/.../test_topk_large_indices.py::TOPK_LARGE_INDICES_PRODUCTION_PERF_CONFIGS).
LARGE_INDICES_ANCHORS = [
    # (tag, rows, n, valid_length, k)
    ("prod_prefill", 640, 51200, None, 1536),
    ("prod_bounded_cache", 2, 102400, 56320, 1536),
]


def _is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0


def _adjusted_k(k):
    # ttnn.topk host wrapper rounds k up to a tile multiple (topk.cpp:39-41);
    # the device (and the multi-core k<=64 gate) sees this value.
    return 32 * math.ceil(k / 32)


def topk_single_core_l1_cost(n, k, dtype):
    """Mirror of verify_single_core_cost (topk_utils.cpp:220-262). Approximate
    (raw tile payload sizes, no CB page headers) -- used only for a-priori
    labeling; the baseline arm still attempts the cell to get the real answer."""
    ktiles = math.ceil(_adjusted_k(k) / 32)
    value = TILE_BYTES[dtype]
    compute = value  # bfp8/bfp4 would upcast; bf16/fp32 stay as-is
    uint16_output = (n <= 65535) and dtype != "fp32"
    index = TILE_BYTES["uint16" if uint16_output else "uint32"]
    return 4 * (value + index) + (4 + 2 * ktiles) * (compute + index) + ktiles * (value + index)


def topk_predicted_factory(n, k, dtype):
    ak = _adjusted_k(k)
    if n >= 8192 and n < 65535 and _is_pow2(n) and ak <= 64:
        # find_topk_core_config can still demote (exact-division + core
        # rectangle + L1 fit), so this is a prediction, not a promise.
        return "multi_core(pred)"
    return "single_core(pred)"


def sort_predicted_factory(n):
    w = max(n, 64)
    if not _is_pow2(w):
        w = 1 << (w - 1).bit_length()  # host pads to next pow2 (sort.cpp:110-129)
    wt = w // 32
    if wt <= 64:
        return "single_row_single_core(pred)"
    # Hybrid capacity depends on the live core grid; don't guess the grid here.
    return "hybrid_or_dram_multi_core(pred)"


def build_grid(args):
    """Every cell gets status PLANNED or UNSUPPORTED_APRIORI(reason). A-priori
    unsupported cells are attempted once in the baseline arm anyway."""
    ops = args.ops.split(",")
    ns = [int(x) for x in args.ns.split(",")]
    ks = [int(x) for x in args.ks.split(",")]
    dtypes = args.dtypes.split(",")
    cells = []

    def add(op, batch, n, k, dtype, anchor="", valid_length=None, apriori=None, factory=""):
        kpart = f"k{k}" if k is not None else "knone"
        cid = f"{op}_b{batch}xN{n}_{kpart}_{dtype}" + (f"_{anchor}" if anchor else "")
        cells.append(
            {
                "id": cid,
                "op": op,
                "batch": batch,
                "n": n,
                "k": k,
                "dtype": dtype,
                "dim": -1,
                "anchor": anchor,
                "valid_length": valid_length,
                "apriori": apriori or "",
                "expected_factory": factory,
            }
        )

    if "topk" in ops:
        for n in ns:
            for k in ks:
                for dt in dtypes:
                    apriori = None
                    if k > n:
                        apriori = f"k={k} > dim size N={n} (topk.cpp:222-226)"
                    elif (
                        topk_predicted_factory(n, k, dt) == "single_core(pred)"
                        and topk_single_core_l1_cost(n, k, dt) >= BH_L1_BYTES
                    ):
                        apriori = (
                            f"predicted single-core L1 overflow: "
                            f"cost {topk_single_core_l1_cost(n, k, dt)} >= {BH_L1_BYTES} "
                            f"(verify_single_core_cost, topk_utils.cpp:220)"
                        )
                    add("topk", 1, n, k, dt, apriori=apriori, factory=topk_predicted_factory(n, k, dt))

    if "sort" in ops:
        for n in ns:
            for dt in dtypes:
                add("sort", 1, n, None, dt, factory=sort_predicted_factory(n))

    if "topk_large_indices" in ops:
        for n in ns:
            for k in ks:
                for dt in dtypes:
                    apriori = None
                    if dt != "bf16":
                        apriori = "input must be BFLOAT16 (topk_large_indices_device_operation.cpp:33)"
                    elif k < 16 or k > 2048 or k % 16 != 0:
                        apriori = f"k must be in [16, 2048] in multiples of 16, got {k} (:24-28)"
                    elif n < k:
                        apriori = f"N={n} < k={k} (:49-50)"
                    add("topk_large_indices", 1, n, k, dt, apriori=apriori, factory="row_major_multi_core")
        for tag, rows, n, vl, k in LARGE_INDICES_ANCHORS:
            add(
                "topk_large_indices",
                rows,
                n,
                k,
                "bf16",
                anchor=tag,
                valid_length=vl,
                factory="row_major_multi_core",
            )

    if "moe_gate" in ops:
        # Fixed 256-expert DeepSeek geometry (8 groups x 32); k in {4,6,8} is the
        # kernel's entire supported range -- these are anchor cells, not a grid axis.
        for k in (4, 6, 8):
            add("moe_gate", 32, 256, k, "bf16", anchor="moe_anchor", factory="single_tile_sharded")

    return cells


# ---------------------------------------------------------------------------
# Child mode: runs INSIDE `python -m tracy -r -v`. One process per cell.
# Philosophy from _topk_sort_bench.py: warm the cache with a correctness-checked
# call, warmup iters, measured iters, sync between phases; every exception is a
# recorded datum. This child additionally checks sort values (the old bench
# never did) and covers topk_large_indices and the MoE gate.
# ---------------------------------------------------------------------------
def run_child(spec_path):
    import torch

    import ttnn

    with open(spec_path) as f:
        spec = json.load(f)
    iters = spec["iters"]
    warmup = spec["warmup"]

    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    arch = ttnn.get_arch_name()
    manifest = {"arch": arch, "iters": iters, "cells": []}
    torch.manual_seed(0)

    for cell in spec["cells"]:
        entry = dict(cell)
        entry.update({"status": "", "error": "", "phase": ""})
        try:
            entry["phase"] = "setup"
            call, correctness = _build_cell_callable(ttnn, torch, device, cell)

            # First call warms the program cache AND is correctness-checked:
            # a table row that silently reports garbage timing is worse than
            # no row. An exception here (validate() fires before dispatch)
            # means UNSUPPORTED; an exception later means FAILED.
            entry["phase"] = "first_call"
            out = call()
            entry.update(correctness(out))
            ttnn.synchronize_device(device)

            entry["phase"] = "warmup"
            for _ in range(warmup):
                call()
            ttnn.synchronize_device(device)

            entry["phase"] = "measure"
            for _ in range(iters):
                call()
            ttnn.synchronize_device(device)

            entry["status"] = "RAN"
            print(f"SWEEP_OK   {cell['id']}", flush=True)
        except Exception as e:  # noqa: BLE001 - the message IS the result
            entry["status"] = "UNSUPPORTED" if entry["phase"] in ("setup", "first_call") else "FAILED"
            entry["error"] = f"{type(e).__name__}: {e}".split("\n")[0][:400]
            print(f"SWEEP_FAIL {cell['id']} [{entry['phase']}] :: {entry['error']}", flush=True)
        manifest["cells"].append(entry)

    ttnn.close_device(device)
    with open(spec["manifest"], "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"SWEEP: manifest -> {spec['manifest']}", flush=True)


def _build_cell_callable(ttnn, torch, device, cell):
    """Return (call, correctness) for one cell. correctness(out) -> dict of
    extra manifest fields; it runs on the first call's output only."""
    op, n, k, batch, dtype = cell["op"], cell["n"], cell["k"], cell["batch"], cell["dtype"]
    torch_dt = torch.bfloat16 if dtype == "bf16" else torch.float32
    ttnn_dt = ttnn.bfloat16 if dtype == "bf16" else ttnn.float32

    if op == "topk":
        t = torch.randn((1, 1, batch, n), dtype=torch_dt)
        x = ttnn.from_torch(
            t, dtype=ttnn_dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        def call():
            return ttnn.topk(x, k=k, dim=-1, largest=True, sorted=True)

        def correctness(out):
            vals = ttnn.to_torch(out[0])[..., :k].float()
            ref = torch.topk(t.float(), k=k, dim=-1).values
            return {"max_abs_err": (vals - ref).abs().max().item()}

        return call, correctness

    if op == "sort":
        t = torch.randn((1, 1, batch, n), dtype=torch_dt)
        x = ttnn.from_torch(
            t, dtype=ttnn_dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        def call():
            return ttnn.sort(x, dim=-1, descending=True)

        def correctness(out):
            # Output carries the host's pow2 padding; descending puts the -inf
            # pads last, so the first n columns are the real sorted values.
            vals = ttnn.to_torch(out[0]).float()[..., :n]
            ref = torch.sort(t.float(), dim=-1, descending=True).values
            return {"max_abs_err": (vals - ref).abs().max().item()}

        return call, correctness

    if op == "topk_large_indices":
        vl = cell.get("valid_length")
        t = torch.randn((batch, n), dtype=torch.bfloat16)
        x = ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def call():
            if vl is not None:
                return ttnn.experimental.topk_large_indices(x, k=k, valid_length=vl)
            return ttnn.experimental.topk_large_indices(x, k=k)

        def correctness(out):
            # The op returns uint32 INDICES; validate them by gathering the
            # input at those indices and comparing (order-insensitively) to
            # torch.topk over the valid_length-masked input.
            idx = ttnn.to_torch(out).to(torch.int64)[..., :k]
            gathered = t.float().gather(-1, idx)
            masked = t.float().clone()
            if vl is not None:
                masked[..., vl:] = float("-inf")
            ref = torch.topk(masked, k=k, dim=-1).values
            g_sorted = torch.sort(gathered, dim=-1, descending=True).values
            return {"max_abs_err": (g_sorted - ref).abs().max().item()}

        return call, correctness

    if op == "moe_gate":
        # Fixed DeepSeek geometry mirrored from
        # models/common/tests/modules/moe/test_generalized_moe_gate.py:
        # (batch, 8, 32) logits reshaped to (batch, 16, 16), one 32x32 shard
        # per batch row, transposed bias and arange indices, preallocated
        # outputs. Linear-renorm path (enable_sigmoid=True, output_softmax=False).
        input_shape = (batch, 8, 32)
        reshaped = (batch, 16, 16)
        out_shape = (batch, 1, 16)
        tile32 = ttnn.Tile((32, 32))
        t_in = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1
        t_bias = (2 * torch.rand(input_shape, dtype=torch.bfloat16)) - 1

        grid = device.compute_with_storage_grid_size()
        core_grid = ttnn.num_cores_to_corerangeset(batch, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(core_grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR),
        )

        def up(torch_t, dt):
            return ttnn.from_torch(
                torch_t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem, tile=tile32
            )

        x = up(t_in.reshape(reshaped), ttnn.bfloat16)
        bias = up(t_bias.reshape(reshaped).transpose(-2, -1), ttnn.bfloat16)
        t_idx = torch.arange(256, dtype=torch.int32).unsqueeze(0).expand(batch, -1).reshape(reshaped)
        idx = up(t_idx.transpose(-2, -1).to(torch.uint16), ttnn.uint16)
        out_v = up(torch.zeros(out_shape, dtype=torch.bfloat16), ttnn.bfloat16)
        out_i = up(torch.zeros(out_shape, dtype=torch.uint16), ttnn.uint16)

        def call():
            return ttnn.experimental.deepseek.moe.generalized_moe_gate(
                x,
                bias_tensor=bias,
                input_indices_tensor=idx,
                output_tensor=out_v,
                output_indices_tensor=out_i,
                eps=1e-20,
                scaling_factor=2.5,
                enable_sigmoid=True,
                topk=k,
                output_softmax=False,
            )

        def correctness(out):
            # Selection golden: the gate ranks by sigmoid(logit) + bias
            # (DeepSeek noaux). Ties in bf16 make exact index equality too
            # strict for a perf harness, so record the tie-tolerant index
            # match fraction; the full golden lives in
            # test_generalized_moe_gate.py and is not re-derived here.
            dev_idx = ttnn.to_torch(out[1])[:, 0, :k].to(torch.int64)
            sel = torch.sigmoid(t_in.float()) + t_bias.float()
            ref_idx = torch.topk(sel.reshape(batch, 256), k=k, dim=-1).indices
            match = 0
            for b in range(batch):
                match += len(set(dev_idx[b].tolist()) & set(ref_idx[b].tolist()))
            frac = match / (batch * k)
            return {"max_abs_err": 1.0 - frac, "index_match_frac": frac}

        return call, correctness

    raise ValueError(f"unknown op {op}")


# ---------------------------------------------------------------------------
# Header arm management (orchestrator side).
# ---------------------------------------------------------------------------
def _header_path():
    return os.path.join(REPO, HEADER_RELPATH)


# Recognize BOTH marker dialects: ours, and the "SWEEP_ARM_BEGIN/END" variant
# found on this tree (inserted by a parallel session). Stripping only our own
# markers would let a foreign block survive into the "baseline" arm and
# silently turn every baseline number into a replay_load number.
_MARKER_BEGINS = (MARKER_BEGIN, "// SWEEP_ARM_BEGIN")
_MARKER_ENDS = (MARKER_END, "// SWEEP_ARM_END")


def _strip_arm_block(text):
    if not any(m in text for m in _MARKER_BEGINS):
        return text, False
    lines = text.splitlines(keepends=True)
    out, skipping, stripped = [], False, False
    for line in lines:
        if any(m in line for m in _MARKER_BEGINS):
            skipping, stripped = True, True
            continue
        if skipping and any(m in line for m in _MARKER_ENDS):
            skipping = False
            continue
        if not skipping:
            out.append(line)
    return "".join(out), stripped


def _verify_baseline_header(text):
    """The pristine header contains exactly ONE `#define TOPK_REPLAY_STEP_LOAD 1`
    (the STORE->LOAD implication at ckernel_sfpu_topk.h:64-66) and no STORE
    define. Anything else means some edit outside our markers is arming the
    replay path -- a 'baseline' measured on top of that is a lie."""
    loads = text.count("#define TOPK_REPLAY_STEP_LOAD 1")
    stores = text.count("#define TOPK_REPLAY_STEP_STORE 1")
    return loads == 1 and stores == 0


def checkout_arm(arm, allow_edit, git_baseline_dirty):
    """Flip the header to `arm`. baseline strips the marker block; the other
    arms insert it at the very top of the file. Asserts via git that nothing
    else changed. Refuses to write anything without --allow-header-edit."""
    path = _header_path()
    with open(path) as f:
        text = f.read()
    stripped, had_block = _strip_arm_block(text)

    if not _verify_baseline_header(stripped):
        sys.exit(
            f"REFUSING to run: after stripping the marker block, {HEADER_RELPATH} still "
            "arms TOPK_REPLAY_STEP_* somewhere outside the markers. A 'baseline' on top "
            "of that would be a lie. Diff the header against git and clean it by hand."
        )

    define = ARM_DEFINES[arm]
    if define is None:
        new_text = stripped
    else:
        new_text = f"{MARKER_BEGIN_FULL}\n{define}\n{MARKER_END}\n{stripped}"

    if new_text == text:
        return  # already on this arm; no write, no mtime churn

    if not allow_edit:
        if had_block:
            sys.exit(
                f"REFUSING to run: {HEADER_RELPATH} carries a leftover CANONICAL_SWEEP_ARM "
                "block from a previous sweep, so 'baseline' would not be baseline. "
                "Re-run with --allow-header-edit to let the sweep clean it, or remove it by hand."
            )
        sys.exit(
            f"arm '{arm}' requires editing {HEADER_RELPATH}; pass --allow-header-edit "
            "to permit the (git-verified, auto-restored) edit."
        )

    with open(path, "w") as f:
        f.write(new_text)

    # The only file allowed to differ from the pre-sweep state is the header.
    dirty = set(
        subprocess.run(
            ["git", "-C", REPO, "diff", "--name-only"], capture_output=True, text=True, check=True
        ).stdout.split()
    )
    unexpected = dirty - git_baseline_dirty - {HEADER_RELPATH}
    if unexpected:
        # Restore before dying: never leave the tree in a mystery state.
        with open(path, "w") as f:
            f.write(_strip_arm_block(new_text)[0])
        sys.exit(f"checkout_arm({arm}): unexpected modified files {sorted(unexpected)}; aborting.")
    print(f"ARM: {arm} ({'block removed' if define is None else define})", flush=True)


def clear_kernel_cache():
    # The JIT dephash sidecar (jit_build/build.cpp:563-568) would invalidate
    # stale ELFs on a header change, but between A/B arms we clear the cache
    # outright: a guarantee beats a mechanism the measurement has to trust.
    shutil.rmtree(KERNEL_CACHE_DIR, ignore_errors=True)
    print(f"CACHE: cleared {KERNEL_CACHE_DIR}", flush=True)


# ---------------------------------------------------------------------------
# Orchestrator: one Tracy subprocess per cell per trial.
# ---------------------------------------------------------------------------
def _reports_glob():
    return os.path.join(REPO, "generated/profiler/reports/*/ops_perf_results_*.csv")


def _newest_report_after(t0):
    candidates = [p for p in glob.glob(_reports_glob()) if os.path.getmtime(p) >= t0]
    return max(candidates, key=os.path.getmtime) if candidates else None


def parse_tracy_for_cell(csv_path, cell, iters):
    """Per-config attribution: filter to the cell's device op by OP CODE, order
    by GLOBAL CALL COUNT, take the LAST `iters` rows (drops the correctness
    call, the warmup iters, and any cache-miss compile row). The subprocess ran
    exactly one cell, so this is exact, and the ATTRIBUTES/W/CORE COUNT columns
    are recorded as provenance rather than used as a fragile join key (note:
    ttnn.topk rounds k up to a tile multiple host-side, so ATTRIBUTES shows the
    ADJUSTED k, not the requested one)."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    dur_col = next((c for c in rows[0] if "DEVICE KERNEL DURATION [ns]" == c.strip()), None)
    if dur_col is None:
        dur_col = next((c for c in rows[0] if "DEVICE KERNEL DURATION" in c.upper()), None)
    if dur_col is None:
        return None
    matched = []
    for r in rows:
        name = (r.get("OP CODE") or "").strip()
        raw = (r.get(dur_col) or "").strip()
        if not name or not raw or not _op_code_matches(cell["op"], name):
            continue
        try:
            matched.append(
                {
                    "call": int(float(r.get("GLOBAL CALL COUNT") or 0)),
                    "ns": float(raw),
                    "cores": int(float(r.get("CORE COUNT") or 0)),
                    "attrs": (r.get("ATTRIBUTES") or "")[:160],
                }
            )
        except ValueError:
            continue
    if not matched:
        return None
    matched.sort(key=lambda m: m["call"])
    measured = matched[-iters:] if len(matched) >= iters else matched
    return {
        "ns_median": statistics.median(m["ns"] for m in measured),
        "ns_samples": [m["ns"] for m in measured],
        "cores": max(m["cores"] for m in measured),
        "n_rows_total": len(matched),
        "n_rows_used": len(measured),
        "attrs": measured[-1]["attrs"],
        "csv": csv_path,
    }


def result_path(outdir, cell_id, arm, trial):
    return os.path.join(outdir, "results", f"{cell_id}.{arm}.t{trial}.json")


def run_cell(cell, arm, trial, args):
    """Fresh `python -m tracy -r -v <this file>` subprocess under a watchdog.
    Writes results/<cell>.<arm>.t<trial>.json and returns its dict."""
    outdir = args.out
    workdir = os.path.join(outdir, "work")
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "results"), exist_ok=True)
    tag = f"{cell['id']}.{arm}.t{trial}"
    spec_path = os.path.join(workdir, f"{tag}.spec.json")
    manifest_path = os.path.join(workdir, f"{tag}.manifest.json")
    log_path = os.path.join(workdir, f"{tag}.log")
    with open(spec_path, "w") as f:
        json.dump({"cells": [cell], "iters": args.iters, "warmup": args.warmup, "manifest": manifest_path}, f)

    env = dict(os.environ)
    env[CHILD_SPEC_ENV] = spec_path
    # DPRINT/Watcher and the device profiler share on-device SRAM and cannot
    # coexist (docs/profiling.md); a leaked env var here poisons every number.
    for var in list(env):
        if var.startswith("TT_METAL_DPRINT") or var.startswith("TT_METAL_WATCHER"):
            env.pop(var)

    result = {
        "cell": cell,
        "arm": arm,
        "trial": trial,
        "status": "",
        "error": "",
        "ns_median": None,
        "cores": None,
        "max_abs_err": None,
        "notes": "",
    }
    t0 = time.time()
    try:
        with open(log_path, "w") as log:
            subprocess.run(
                [sys.executable, "-m", "tracy", "-r", "-v", SCRIPT_PATH],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=args.timeout,
                check=False,
            )
    except subprocess.TimeoutExpired:
        result["status"] = "FAILED"
        result["error"] = f"watchdog timeout {args.timeout}s -- device may need tt-smi -r before continuing"
        _write_result(outdir, tag, result)
        return result

    if not os.path.exists(manifest_path):
        result["status"] = "FAILED"
        result["error"] = f"child produced no manifest (crash before close_device?); see {log_path}"
        _write_result(outdir, tag, result)
        return result

    with open(manifest_path) as f:
        entry = json.load(f)["cells"][0]
    result["max_abs_err"] = entry.get("max_abs_err")
    if "index_match_frac" in entry:
        result["notes"] = f"index_match_frac={entry['index_match_frac']:.3f}"
    if entry["status"] != "RAN":
        result["status"] = entry["status"]  # UNSUPPORTED or FAILED, message verbatim
        result["error"] = entry["error"]
        _write_result(outdir, tag, result)
        return result

    csv_path = _newest_report_after(t0)
    parsed = parse_tracy_for_cell(csv_path, cell, args.iters) if csv_path else None
    if parsed is None:
        result["status"] = "FAILED"
        result["error"] = (
            "cell RAN but no Tracy rows matched -- was the run under `python -m tracy -r`? " f"csv={csv_path}"
        )
    else:
        result["status"] = "MEASURED"
        result.update(
            {
                "ns_median": parsed["ns_median"],
                "cores": parsed["cores"],
                "notes": (result["notes"] + f" attrs={parsed['attrs'][:80]}").strip(),
            }
        )
    _write_result(outdir, tag, result)
    return result


def _write_result(outdir, tag, result):
    path = os.path.join(outdir, "results", f"{tag}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=1)


# ---------------------------------------------------------------------------
# LLK layer: ingest tt_metal/tt-llk/perf_data/perf_topk_rebuild_xl (the file
# that carries baseline and ours as paired arms in one run). Cyc/vector via
# the same two-point slope the driver documents: slope over rebuild_iter_count
# cancels the ~30-cycle marker pair and every one-time cost inside the zone.
#
# CALIBRATION HONESTY: validated against the on-disk post.csv, the RATIOS here
# reproduce the driver's published speedups to the third decimal (1.069x /
# 1.120x / 1.165x rebuild; 1.978x / 2.192x / 2.331x merge; 1.136x / 1.208x /
# 1.255x step), but the ABSOLUTE cyc/vec differs from the driver's published
# cyc/call by a constant normalization inside the post-processing. Treat the
# speedup column as the measurement and the absolute columns as relative scale.
# ---------------------------------------------------------------------------
LLK_PAIRS = [  # (row name, baseline arm, ours arm)
    ("rebuild", "RbCall", "RbCallFull"),
    ("merge", "XlMerge", "MacroMerge"),
    ("step", "XlStep", "FullStep"),
]


def ingest_llk_rows():
    path = os.path.join(REPO, "tt_metal/tt-llk/perf_data/perf_topk_rebuild_xl/perf_topk_rebuild_xl.post.csv")
    if not os.path.exists(path):
        return [], f"llk: {path} not found -- run the driver first (see --run-llk)"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    # mean(MATH_ISOLATE) per (arm, k, iter_count), averaged over the 5 runs.
    acc = {}
    for r in rows:
        try:
            key = (
                r["rebuild_arm"].split(".")[-1],
                int(r["rebuild_k"]),
                int(r["rebuild_iter_count"]),
            )
            acc.setdefault(key, []).append(float(r["mean(MATH_ISOLATE)"]))
        except (KeyError, ValueError):
            continue

    def cyc_per_vec(arm, k):
        pts = sorted((ic, statistics.mean(v)) for (a, kk, ic), v in acc.items() if a == arm and kk == k)
        if len(pts) < 2:
            return None
        (lo_i, lo_m), (hi_i, hi_m) = pts[0], pts[-1]
        slope = (hi_m - lo_m) / (hi_i - lo_i)
        vecs = 2 if arm in ("CtrlLoad", "CtrlSwap") else k // 32
        return slope / vecs

    out, note = [], ""
    ks = sorted({k for (_, k, _) in acc})
    for k in ks:
        # Tripwire from the driver: CtrlSwap must be 2.00x CtrlLoad, or the
        # profiler measured the RISC-V push rate and the whole run is invalid.
        cl, cs = cyc_per_vec("CtrlLoad", k), cyc_per_vec("CtrlSwap", k)
        suspect = ""
        if cl and cs and not (1.9 <= cs / cl <= 2.1):
            suspect = f"TRIPWIRE: CtrlSwap/CtrlLoad = {cs / cl:.3f}, expected 2.00 -- run invalid"
        for name, base_arm, ours_arm in LLK_PAIRS:
            b, o = cyc_per_vec(base_arm, k), cyc_per_vec(ours_arm, k)
            if b is None or o is None:
                continue
            out.append(
                {
                    "layer": "llk",
                    "op": f"llk:topk_xl:{name}",
                    "batch": "",
                    "n": "",
                    "k": k,
                    "dtype": "bf16",
                    "dim": "",
                    "anchor": "MATH_ISOLATE cyc/vec",
                    "expected_factory": "",
                    "cores": 1,
                    "llk_cyc_per_vec_base": round(b, 3),
                    "llk_cyc_per_vec_ours": round(o, 3),
                    "speedup": f"{b / o:.3f}x" if o else "",
                    "status": "MEASURED" if not suspect else "FAILED",
                    "notes": suspect or f"{base_arm} vs {ours_arm}; slope cancels marker overhead",
                }
            )
    return out, note


def run_llk_driver():
    """Run producer+consumer inside ONE flock: a concurrent producer wipes
    /tmp/tt-llk-build between phases (HANDOFF.md). Uses the LLK venv, never
    tt-metal's python_env, and never scripts/run_safe_pytest.sh (SORTING.md D6).
    The serial producer races on artefact dirs -- -n 8 is the fix (D2)."""
    tests_dir = os.path.join(REPO, "tt_metal/tt-llk/tests")
    venv_py = os.path.join(tests_dir, ".venv/bin/python")
    if not os.path.exists(venv_py):
        sys.exit(f"llk venv not found at {venv_py}; run `source setup_external_testing_env.sh` there first")
    inner = (
        f"cd {tests_dir}/python_tests && "
        f"{venv_py} -m pytest perf_topk_rebuild_xl.py --compile-producer -n 8 && "
        f"{venv_py} -m pytest perf_topk_rebuild_xl.py --compile-consumer"
    )
    cmd = ["flock", "/tmp/tt-device.lock", "bash", "-c", inner]
    print(f"LLK: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Aggregation + report.
# ---------------------------------------------------------------------------
def _load_results(outdir):
    results = {}
    for path in glob.glob(os.path.join(outdir, "results", "*.json")):
        with open(path) as f:
            r = json.load(f)
        results.setdefault(r["cell"]["id"], {}).setdefault(r["arm"], []).append(r)
    return results


def _arm_stats(trials):
    measured = [t["ns_median"] for t in trials if t["status"] == "MEASURED" and t["ns_median"]]
    if not measured:
        return None, None, len(measured)
    mean = statistics.mean(measured)
    std = statistics.stdev(measured) if len(measured) > 1 else 0.0
    return mean, std, len(measured)


def _speedup(base_mean, base_std, ours_mean, ours_std):
    if not base_mean or not ours_mean:
        return ""
    pooled = math.sqrt(base_std**2 + ours_std**2)
    if abs(base_mean - ours_mean) <= 2 * pooled:
        return "~1.00 (noise)"
    return f"{base_mean / ours_mean:.3f}x"


def aggregate(cells, results, arms, llk_rows):
    """One row per cell. Status precedence: MEASURED > UNSUPPORTED > FAILED >
    PLANNED (never attempted). Errors ride along verbatim -- half the point."""
    rows = []
    for cell in cells:
        per_arm = results.get(cell["id"], {})
        row = {
            "layer": "ttnn",
            "op": cell["op"],
            "batch": cell["batch"],
            "n": cell["n"],
            "k": cell["k"] if cell["k"] is not None else "-",
            "dtype": cell["dtype"],
            "dim": cell["dim"],
            "anchor": cell["anchor"],
            "expected_factory": cell["expected_factory"],
            "cores": "",
            "llk_cyc_per_vec_base": "",
            "llk_cyc_per_vec_ours": "",
            "max_abs_err": "",
            "status": "PLANNED",
            "notes": "",
        }
        base_mean = base_std = None
        for arm in arms:
            trials = per_arm.get(arm, [])
            mean, std, n_ok = _arm_stats(trials)
            row[f"{arm}_ns_mean"] = round(mean, 1) if mean else ""
            row[f"{arm}_ns_std"] = round(std, 1) if mean else ""
            if arm == "baseline":
                base_mean, base_std = mean, std
            elif mean and base_mean:
                row[f"speedup_{arm}"] = _speedup(base_mean, base_std, mean, std)
            else:
                row[f"speedup_{arm}"] = ""
            for t in trials:
                if t.get("cores"):
                    row["cores"] = t["cores"]
                if t.get("max_abs_err") is not None:
                    row["max_abs_err"] = round(t["max_abs_err"], 6)
                if t["status"] == "MEASURED":
                    row["status"] = "MEASURED"
                elif t["status"] in ("UNSUPPORTED", "FAILED") and row["status"] != "MEASURED":
                    row["status"] = f"{t['status']}({t['error']})"
        if row["status"] == "PLANNED" and cell["apriori"]:
            row["status"] = f"UNSUPPORTED_APRIORI({cell['apriori']})"
        elif cell["apriori"] and row["status"].startswith("MEASURED"):
            row["notes"] = f"PREDICTION MISMATCH: a-priori said unsupported ({cell['apriori']})"
        rows.append(row)

    for lrow in llk_rows:
        full = {**{k: "" for k in rows[0]}, **lrow} if rows else lrow
        rows.append(full)
    return rows


def write_reports(rows, arms, outdir):
    if not rows:
        print("REPORT: nothing to write")
        return
    arm_cols = []
    for arm in arms:
        arm_cols += [f"{arm}_ns_mean", f"{arm}_ns_std"]
        if arm != "baseline":
            arm_cols.append(f"speedup_{arm}")
    columns = (
        ["layer", "op", "batch", "n", "k", "dtype", "dim", "anchor", "expected_factory", "cores"]
        + arm_cols
        + ["speedup", "llk_cyc_per_vec_base", "llk_cyc_per_vec_ours", "max_abs_err", "status", "notes"]
    )
    csv_path = os.path.join(outdir, "canonical_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in columns})

    md_path = os.path.join(outdir, "canonical_sweep.md")
    with open(md_path, "w") as f:
        f.write("# Canonical top-k / sort sweep\n\n")
        f.write(f"Arms: {', '.join(arms)}. ns = DEVICE KERNEL DURATION median-per-trial, ")
        f.write("mean±std across trials. Speedups shown only when |delta| > 2·pooled_std.\n\n")
        for layer, title in (("ttnn", "ttnn layer (op-level, Tracy)"), ("llk", "llk layer (MATH_ISOLATE cyc/vec)")):
            sub = [r for r in rows if r.get("layer") == layer]
            if not sub:
                continue
            f.write(f"## {title}\n\n")
            hdr = [c for c in columns if any(str(r.get(c, "")) != "" for r in sub)]
            f.write("| " + " | ".join(hdr) + " |\n")
            f.write("|" + "|".join("---" for _ in hdr) + "|\n")
            for r in sub:
                cells_out = []
                for c in hdr:
                    v = str(r.get(c, ""))
                    if c == "status" and len(v) > 60:
                        v = v[:57] + "..."
                    cells_out.append(v.replace("|", "/"))
                f.write("| " + " | ".join(cells_out) + " |\n")
            f.write("\n")
        bad = [r for r in rows if r.get("status", "").startswith(("UNSUPPORTED", "FAILED"))]
        if bad:
            f.write("## Unsupported / failed cells (verbatim -- the message is the datum)\n\n")
            for r in bad:
                f.write(f"- `{r.get('op')}` N={r.get('n')} K={r.get('k')} {r.get('dtype')}: {r.get('status')}\n")
    print(f"REPORT: {csv_path}")
    print(f"REPORT: {md_path}")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layers", default="ttnn", help="ttnn,llk")
    p.add_argument("--arms", default="baseline", help="baseline,replay_load,replay_store")
    p.add_argument("--ops", default="topk,sort,topk_large_indices,moe_gate")
    p.add_argument("--ns", default=DEFAULT_NS)
    p.add_argument("--ks", default=DEFAULT_KS)
    p.add_argument("--dtypes", default=DEFAULT_DTYPES)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--timeout", type=int, default=900, help="per-cell watchdog seconds")
    p.add_argument("--out", default=os.path.join(REPO, "generated/canonical_sweep/latest"))
    p.add_argument(
        "--resume", action="store_true", help="skip cells whose result JSON already says MEASURED/UNSUPPORTED"
    )
    p.add_argument("--report", action="store_true", help="rebuild CSV/markdown from --out without measuring")
    p.add_argument("--allow-header-edit", action="store_true", help="permit the A/B edit of " + HEADER_RELPATH)
    p.add_argument(
        "--run-llk", action="store_true", help="also RUN the tt-llk perf driver (default: ingest existing perf_data)"
    )
    p.add_argument(
        "--child", action="store_true", help="internal: run as measurement child (or set " + CHILD_SPEC_ENV + ")"
    )
    args = p.parse_args()

    # Child mode: inside `python -m tracy -r -v`. Env var wins so tracy's
    # shell re-invocation never has to carry our arguments.
    spec_path = os.environ.get(CHILD_SPEC_ENV)
    if args.child or spec_path:
        run_child(spec_path or sys.argv[sys.argv.index("--child") + 1])
        return 0

    arms = args.arms.split(",")
    for arm in arms:
        if arm not in ARM_DEFINES:
            sys.exit(f"unknown arm '{arm}'; choose from {list(ARM_DEFINES)}")
    layers = args.layers.split(",")
    os.makedirs(args.out, exist_ok=True)
    cells = build_grid(args)
    with open(os.path.join(args.out, "grid.json"), "w") as f:
        json.dump(cells, f, indent=1)

    llk_rows = []
    if "llk" in layers:
        if args.run_llk and not args.report:
            run_llk_driver()
        llk_rows, llk_note = ingest_llk_rows()
        if llk_note:
            print(llk_note)

    if args.report:
        rows = aggregate(cells, _load_results(args.out), arms, llk_rows)
        write_reports(rows, arms, args.out)
        return 0

    # Non-baseline arms need the header edit; fail BEFORE any device work.
    if any(a != "baseline" for a in arms) and not args.allow_header_edit:
        sys.exit("arms beyond 'baseline' edit " + HEADER_RELPATH + "; pass --allow-header-edit")

    git_dirty = set(
        subprocess.run(
            ["git", "-C", REPO, "diff", "--name-only"], capture_output=True, text=True, check=True
        ).stdout.split()
    )
    if HEADER_RELPATH in git_dirty and not args.allow_header_edit:
        sys.exit(f"{HEADER_RELPATH} is already modified; refusing to measure a mystery arm.")

    def _done(cell, arm, trial):
        path = result_path(args.out, cell["id"], arm, trial)
        if not (args.resume and os.path.exists(path)):
            return False
        with open(path) as f:
            status = json.load(f)["status"]
        # UNSUPPORTED is deterministic (validate() fires before dispatch);
        # FAILED gets retried -- it may have been a transient hang.
        return status in ("MEASURED", "UNSUPPORTED")

    try:
        if "ttnn" in layers:
            for arm in arms:
                checkout_arm(arm, args.allow_header_edit, git_dirty)
                clear_kernel_cache()
                for cell in cells:
                    # A-priori unsupported cells: attempted ONCE, baseline arm
                    # only -- the real error message is the datum, and running
                    # them per-arm/per-trial would just re-collect it.
                    trials = range(args.trials)
                    if cell["apriori"]:
                        if arm != "baseline":
                            continue
                        trials = range(1)
                    for trial in trials:
                        if _done(cell, arm, trial):
                            print(f"SKIP (resume) {cell['id']}.{arm}.t{trial}", flush=True)
                            continue
                        r = run_cell(cell, arm, trial, args)
                        print(
                            f"{r['status']:<11} {cell['id']}.{arm}.t{trial} "
                            f"{r['ns_median'] or ''} {r['error'][:80]}",
                            flush=True,
                        )
                        # Rewrite the reports after every cell: an early stop
                        # (or a hang two hours in) keeps everything measured.
                        rows = aggregate(cells, _load_results(args.out), arms, llk_rows)
                        write_reports(rows, arms, args.out)
                        # UNSUPPORTED is deterministic; don't burn trials on it.
                        if r["status"] == "UNSUPPORTED" and not cell["apriori"]:
                            break
    finally:
        # Never leave the tree on a non-baseline arm, whatever happened above.
        if args.allow_header_edit:
            checkout_arm("baseline", True, git_dirty)

    rows = aggregate(cells, _load_results(args.out), arms, llk_rows)
    write_reports(rows, arms, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
