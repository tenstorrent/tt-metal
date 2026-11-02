---
name: llk-perf-headroom
description: Use when deciding whether a Quasar LLK kernel can be made faster, what limits it, or which optimization to try next — before guessing at replay buffers, SFPI rewrites, NOP removal or interleaving, and when a perf change needs evidence beyond a cycle count.
---

# LLK Perf Headroom (Quasar)

## Overview

Measure, don't guess. One waveform of one perf-test variant, cut to the exact `TILE_LOOP` window,
classifies **every cycle of every thread** as issue / stall-by-reason / idle, using the hardware's
per-thread issue and stall signals. Divided per tile it must reconcile with the perf CSV; then the gap between the
measured time and the issue floor *is* the list of causes, ranked in cycles per tile.

## Steps

`TT` = tt-llk dir of the branch under test, `CG` = tt-llk dir of the codegen checkout,
`EV` = an evidence dir on shared storage, `PRIV` = the private tool,
`${LLK_CODEGEN_PRIVATE_ROOT:-/proj_sw/user_dev/llk_code_gen}/tools/llk_wave_debug`.
Steps 3–5 (capture, export, breakdown) are in `$PRIV/README.md`, section **Perf capture**; it also
lists known Quasar bubbles. Measured issue costs: `$PRIV/codegen/llk_wave_debug/rates/quasar_issue_rates.private.json`.

1. **Freeze the metric first** (write it down before measuring): the bottleneck thread's isolate
   metric (`mean(MATH_ISOLATE)` for SFPU/FPU ops) must improve > 0.5 % and `mean(L1_TO_L1)` must
   not regress. Runs are deterministic on the emulator (repeat runs are byte-identical).
2. **Baseline sweep** via the run-test skill (`run_test.sh run --test perf_<op>_quasar.py --k <op>`);
   keep `perf_data/.../*.post.csv`.
3. **Capture one variant** with waves and raw profiler zones (`llk_profiler_dump` pytest plugin in
   `$CG/codegen/scripts/pytest_plugins`), per `$PRIV/README.md` → Perf capture. With waves on, the
   5th run type (L1_CONGESTION) is killed by `run_test.sh`'s no-output watchdog; that is expected.
4. **Export** the perf signal set to `$EV/waves.vcd.gz` (same README section), then delete the dump (~1 GB).
5. **Breakdown**: `python $CG/.claude/scripts/llk_wave_debug.py perf $EV/waves.vcd.gz --profiler-csv $EV/profiler/002_MATH_ISOLATE_run0.csv --output $EV/perf_MATH_ISOLATE.json`.
   For L1_TO_L1 use `000_L1_TO_L1_run0.csv --zone-start-thread unpack --zone-end-thread pack`
   (the window the perf CSV measures). Exit 2 = `capture_clipped`: re-capture.
6. **Verdict**: `python $CG/codegen/scripts/perf_headroom.py --wave-report $EV/perf_MATH_ISOLATE.json --perf-csv <post.csv> --select <col>=<val> ... --run-type MATH_ISOLATE --rates $CG/codegen/references/quasar_issue_rates.json --rates $PRIV/codegen/llk_wave_debug/rates/quasar_issue_rates.private.json [--best-count N --best-count-source "..."]`
   (`--best-count` = fewest instructions per tile you can justify, e.g. a hand count of the minimal
   sequence; without it only stall headroom is reported.) Repeat for L1_TO_L1.
   Exit 2 / `NOT RECONCILED` / "missing core signals" = the window or capture is wrong. Fix that before
   reading any cause.
7. **Pick the fix from the top cause** (table), change the kernel, run the functional sweep, then
   `perf_eval.py` against the baseline, and re-capture to confirm the targeted cause shrank.

## Reading the verdict

| Cause (cyc/tile) | Means | Typical fix |
|---|---|---|
| `bubble.A->B` | B could not issue the cycle after A. The hardware does **not** flag it as a stall; it shows as an idle cycle between two issues. It may be a data dependency **or** a structural hazard on B's kind of instruction (known cases: `$PRIV/README.md`) | Put an independent instruction of a *different kind* after A: software-pipeline (load and prepare the next pair in a second LREG set) and re-capture. If the bubble just moves to `A-><new instr>`, it is structural on that kind, so reorder until A is followed by an ALU op |
| high issue, low stall/idle | Instruction-bound: floor ≈ measured | Fewer instructions: fold `INCRWC` into SFPLOAD/SFPSTORE immediate offsets (one INCRWC per block), drop redundant MOVs |
| `wait.<resource>` / `sync` | STALLWAIT/SEMWAIT on that condition | Synchronization placement, not the kernel math |
| `sfpu_data_hazard_stall`, `dest_stall_*`, `srca/srcb/srcs_stall_*` | Hardware-flagged hazard on that resource | Reorder or decouple accesses to that resource |
| `idle` not between issues | TRISC not feeding instructions (loop or branch overhead, per-tile setup) | Unroll or replay **only if** this is large |
| `waiting_threads` | Another thread is stalled on the bottleneck | Not a cause: optimize the bottleneck |

**Stop** when `stall_headroom_pct` is small and `instruction_headroom` (against a cited best count) is small.

## Common mistakes

- Treating "instruction FIFO empty" as front-end starvation and reaching for a replay buffer: check
  `bubbles` first. A dependency wait looks identical.
- Assuming latency from instruction descriptions (e.g. "SFPADD is 2-cycle"): a dependent
  instruction often issues with no bubble. Only measured bubbles count.
- Building throwaway kernel variants to infer per-instruction costs: one breakdown gives all of them.
- Predicting the gain and not re-capturing: after every change, re-capture and confirm the targeted
  cause shrank. A cycle count that improves less than predicted means a cause moved; the new
  breakdown names where it went.
- Comparing a window that is not `TILE_LOOP`: use `--profiler-csv`, and require `reconcile.ok`.
- Calling the most-stalled thread the bottleneck: in L1_TO_L1, unpack and pack stall *waiting on* math.
