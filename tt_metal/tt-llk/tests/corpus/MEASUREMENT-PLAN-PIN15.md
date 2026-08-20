# MEASUREMENT-PLAN-PIN15 — the pin-15 measurement pass (lane DZ prep, 2026-08-20)

The exact invocations for the moment the pin-15 union toolchain installs.
Everything below assumes the ceremony is DONE in this order (the apply
script prints the same list):

1. pin-install-fast installed the gate-proven union binaries (cc1plus +
   drivers TOGETHER — .opt changed, cc1plus alone is insufficient) and
   `sha256sum $(tests/sfpi/compiler/bin/riscv-tt-elf-g++ -print-prog-name=cc1plus)`
   equals the gated sha.
2. `tools/pin15/apply_pin15.sh --cc1plus-sha … --driver-sha …
   --sfpi-gcc-commit …` ran GREEN (conf-lint + selftests), and the
   ceremony commit landed (conf prose + values, SAME commit).
3. `REVIEW_RECORD-<cc1plus-12hex>.md` exists in corpus/review_records/
   AND the sweep evidence parent (`~/sfpi-uplift/sweep-2x2/`).
4. `witness_preflight.py --work /tmp/witness-pin15` GREEN on the
   INSTALLED binary (all 25 reviewed rows — the ON set is UNCHANGED, so
   no new witness rows; a stale witness here is a stop).
5. The live tree (`~/sfpi-uplift/tt-metal`) is on the merged nkapre/sfpi
   tip carrying agent/pin15-prep; `tests/sfpi` symlink →
   `~/sfpi-uplift/sfpi/build/sfpi`; `tests/.venv` present.

All device work runs under the dual flocks (the sweep does this itself);
never launch a sweep inside a harness background task — always
`setsid nohup … & disown`.  `SWEEP_DATE` is the sanctioned root-name
override and is REQUIRED here: pin-14 wrote `headline-20260820`-class
roots all day, and the collision guard will (correctly) refuse a plain
date-derived root that already exists under the pin-14 pin.

## Run 0 — TRUE-DEFAULT gate leg + DR delta verification (compile-only, no device)

The stock-user surface (DR three-leg contract, RECIPES.md §1) gets its
pin-15 store entries seeded FIRST, so every later lane byte-gates without
recompiling, and the TRUE-DEFAULT inventory diff vs pin 14 is banked as
evidence before any silicon:

    cd ~/sfpi-uplift/tt-metal/tt_metal/tt-llk/tests/corpus
    # seed the three legs at the installed pin (RUNNER_TEMP private, laneCU gotcha)
    python3 corpus_leg_store.py ensure --arch bh --flags ''                       # TRUE-DEFAULT
    python3 corpus_leg_store.py ensure --arch bh --flags "<OFF_FLAGS verbatim>"   # flags-off
    python3 corpus_leg_store.py ensure --arch bh --flags "<ON_FLAGS verbatim>"    # ON-25

(The union gate lane may already have seeded these from its gate legs via
tools/leg_store_seed.py — `corpus_leg_store.py --list` first; `ensure` is
a no-op on a hit.)

Pre-registered expectations (adjudicate, don't assume):
* TRUE-DEFAULT pin-15 vs pin-14 store entries are DIFFERENT KEYS (new
  cc1plus) — the informative artifact is the .text INVENTORY diff of the
  TRUE-DEFAULT leg vs pin-14's: expected changed classes on BH are the
  Init(1)-surface deltas only (DL latency-audit families; DR's WH
  lut-guard and replay-raw-capture census both verified ZERO-delta on the
  mapped BH corpus at the DW regate — corpus has no WH lut fires and no
  raw REPLAY captures).  Any OTHER BH TRUE-DEFAULT delta is a FINDING
  (file it, do not measure past it).
* DR's WH-guard rows: classify-level only — the WH CRAQ legs of rows
  whose craq_archs include wh re-verify bit-exact; the NaN-guard refusal
  is proven by the dg twin (lut-wh-negative-nan-divergent), not by a
  corpus fire.

## Run 1 — headline + knob legs (THE results-bearing run; device)

One headline invocation carries: (a) the on-plus knob legs for the three
crown jewels, (b) the pin-15 unlock rows deferred by lane DX, (c) the
flip-guard headline surface.  Knob attribution + knob silicon are
weekly-only by default — pass them through explicitly (headline forwards
trailing args to sweep_2x2.py verbatim):

    cd ~/sfpi-uplift/tt-metal/tt_metal/tt-llk/tests/corpus
    SWEEP_DATE=pin15-$(date +%Y%m%d) \
    SWEEP_CLASSIFY_WORKERS=12 \
    SWEEP_PREV_CHAIN=6 \
    setsid nohup ./headline_bh_sweep.sh \
      --ops lcm-fresh,welford,hardsigmoid-fresh,exp,ceil-fresh,xielu-fresh,absint32,castfp32tofp16a,sigmoidappx-tree,tanhderivlut-fresh,log-fresh,sqrt-fresh,rsqrt-fresh,clamp-fresh,hardtanh-fresh,tanh-fresh,silu-fresh,relu,hardtanh,clamp,silu,tanh,tanhderivative-lut,typecast,hardshrink-fresh,softsign-fresh,minmax-min,minmax-max,where,reduce-sdpa,sdpa,gcd-fresh \
      --knob-attribution \
      --knob-silicon-rows welford,lcm-fresh,xielu-fresh,absint32,hardsigmoid-fresh,castfp32tofp16a,typecast,tanhderivlut-fresh \
      --priority-ops lcm-fresh,welford,hardsigmoid-fresh,exp,ceil-fresh,xielu-fresh,absint32,castfp32tofp16a,sigmoidappx-tree,tanhderivlut-fresh \
      > ~/sfpi-uplift/sweep-2x2/sweep-logs/headline-pin15-$(date +%Y%m%d).log 2>&1 & disown

Notes on the flags:
* `--priority-ops` (results-bearing rows first, DC rolling waves — silicon
  starts minutes in).  The wrapper auto-passes `--priority-ops
  $HEADLINE_ROWS`; the explicit list above rides later on the command
  line and WINS (argparse last-occurrence) — it is a strict subset of
  `--ops`, which the loud-typo refusal requires.
* `SWEEP_PREV_CHAIN=6` reaches back past the booking runs to
  `weekly-pin14-20260820` and `headline-pin14-20260820` for cross-pin
  cell reuse: OFF/hand legs whose archived .text equals this run's
  classify hashes adopt with REUSED_FROM provenance (hand kernels are
  byte-identical across pins; expect roughly the hand half of device work
  adopted).  Sem legs recompile under the new pin — correct.
* The knob-silicon rows are the on-plus booking targets: the three NEW
  knobs (list-schedule, lreg-alloc [+per-kernel dst-layout-32b via the
  harness wiring — dest_acc:Yes nodes only, fail-closed], milp) run
  wherever attribution shows them firing on those rows; the pin-14
  holdover knobs (int-abs on absint32, replay-loop-unroll on
  castfp32tofp16a, lut-select-leaf-ext LICENSED on tanhderivlut-fresh,
  replay-hoist on typecast) re-book on the new pin in the same pass.

### Pre-registered row expectations (honest outcomes named BEFORE silicon)

(a) Crown-jewel on-plus knob legs:

| row | knob | expectation |
|---|---|---|
| welford | list-schedule | the ONLY corpus ON-delta of DQ's merge (snapshot TU, 4-insn interleave reorder, CRAQ 16/16 bit-exact, −4 modeled slots) — small win or parity; a LOSS is a stop-and-file |
| lcm-fresh | list-schedule | round-chain stalls are the named DX residual (d); modeled band parity-to-win vs 675.86 booked |
| lcm-fresh | lreg-alloc | 2-chain interleave needs ~10 live LREGs; the allocator may unlock it — honest possibility: classify IDENTICAL (refusal recorded, spill not priced-in on this shape) |
| lcm-fresh, welford, xielu-fresh | milp | fires only on pressure>8 regions — most rows honestly record REFUSAL_BYTE_IDENTICAL; any fire is news |
| xielu-fresh | lreg-alloc | the loop-held-alphas ICE-victim shape (DS arsenal); dest_acc mode decides whether the 32b declaration rides (check the leg's flags-*.txt) |

(b) Deferred pin-15 unlock rows (lane DX table, merged-but-uninstalled work):

| row | pin-14 booked | unlock | expectation |
|---|---|---|---|
| log-fresh / sqrt-fresh / rsqrt-fresh | +2.95 / +1.61 / +1.67 | DM residency widenings | HONEST CAVEAT: DM's own union gate measured an EMPTY corpus ON-delta (invariant-loadi preempts counted CC-clean shapes; runtime-trip shape unrepresented in mapped rows) — classify IDENTICAL here is the documented preemption fact, NOT a failed unlock; only a CHANGED classify earns silicon |
| hardsigmoid-fresh | +2.22 | DN ccmask LT/GE completion | DN's ON-delta includes hardsigmoid (2x SFPLE-LT fires, CRAQ-green) — expect a shave; also relu + hardtanh changed (guard rows) |
| exp | +1.90 | DL capture-rotation widening | production exp fire sim-proven bit-exact at DL's gate — expect movement toward parity |
| ceil-fresh | +9.0 | DL latency audits (SFPSTORE/TTINCRWC/SFPLUT) | replay-reissue-latency-unproved refusal lifts; LUT loops now replay-priceable |
| clamp/hardtanh/tanh/silu fresh+prod, sigmoidappx-tree, tanhderivative-lut | — | DL ON-delta TU families (9 TUs, all CRAQ-adjudicated bit-exact) | re-measure; drift guard |

(c) DR delta rows: none expected on BH (see Run 0); WH verification is
classify/CRAQ-side only.

(d) Flip guards: minmax-min/max (−5.1% WIN), where (154.5 WIN),
reduce-sdpa (832.8 WIN), sdpa (−16.2%), gcd-fresh (−19.16% WIN),
absint32/castfp32tofp16a (parity books).  A WIN→LOSS flip = STOP and
bisect, never re-patch shapes.

## Run 2 — full-surface weekly on the new flow (device; after Run 1 lands)

The complete pin-15 re-baseline (186 rows, pipelined flow, projected
2.5-4 h with adoption):

    cd ~/sfpi-uplift/tt-metal/tt_metal/tt-llk/tests/corpus
    SWEEP_DATE=pin15-$(date +%Y%m%d) \
    SWEEP_CLASSIFY_WORKERS=12 \
    SWEEP_PREV_CHAIN=6 \
    setsid nohup ./weekly_bh_sweep.sh \
      > ~/sfpi-uplift/sweep-2x2/sweep-logs/weekly-pin15-$(date +%Y%m%d).log 2>&1 & disown

The weekly's own `--prev-run` chain now ALSO includes Run 1's root
(newest first), so every Run-1 cell adopts instead of re-measuring.
Weekly adds what headline omits: full knob attribution on every CHANGED
row (the three new knobs now attribute everywhere), the knob-silicon legs
on the conf's HEADLINE_ROWS (now incl. welford/lcm-fresh/xielu-fresh),
and the DejaGnu byte-parity suites.

## After the runs

1. Tally + dashboard rebuild (booking discipline: measured claims need
   compiler-fire + measured-leg-flags + silicon-cell verification chain).
2. Baseline refresh = a REVIEWED manual step (never sweep-side).
3. lreg-alloc/list-schedule/milp ON-set promotion decisions go to the
   owner WITH their R9 union fire witnesses — the conf patch deliberately
   cannot promote them.
4. Arm the liveness watchdog on the sweep log; the guard's pgrep must
   exclude its own shell (pin-14 lesson).
