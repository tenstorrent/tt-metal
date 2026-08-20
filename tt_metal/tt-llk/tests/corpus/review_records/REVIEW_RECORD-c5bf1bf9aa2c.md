# REVIEW_RECORD — pin-15 UNION CANDIDATE (pin cut 2026-08-20)

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com) — independent of lanes DP/DQ/DR/DV (authors); union gates executed by lane DY and re-verified by the orchestrator (this session)

Candidate: sfpi-gcc `staging/pin15` @
`f5aa5bbf676b90ca4f2045edf6613658b76fd0db`
= nkapre/sfpi `a0ea2fcd9aa1620364d37537bb67e275da8626a7` (DQ list-scheduler merge)
+ M1 `agent/lreg-allocator` @ `e3e442ff63c860b1b1478c63398b5182bf6b2447` (DP DSATUR allocator)
+ M2 `agent/milp-and-hygiene` @ `1dd3bc8d4cde8a658cdccc1a09dbaf077a65e71b` (DR vendored B&B + WH lut NaN guard + replay census)
+ M3 `agent/synth-ice-fix` @ `46047176d6c1fa3ac9f7c0a0c0e30f38ecbbe3d0` (DV synth-renumber ICE fix)

At REVIEW_RECORD finalization this file becomes
`REVIEW_RECORD-<cc1plus-sha12>.md` in BOTH required locations
(tt-metal `tt_metal/tt-llk/tests/corpus/review_records/` AND
`~/sfpi-uplift/sweep-2x2/`), with literal '## Reviewed' and '## Gates'
headings and the full 64-hex cc1plus sha (lint enforces).

## Reviewed

- Merge fidelity: ALL THREE merges proven PURE UNIONS byte-exact in both
  directions (laneDY BOTH-PARENTS-AUDIT.md part 2; DW's part 1 covers the
  a0ea list-scheduler merge). Zero conflict hunks; riscv.opt exactly
  additive (913 -> 917 -> 925 lines); every new Var verified in generated
  options.cc at the union build.
- Lane-level reviews carried: DP (DU red-team 19 findings closed, DS
  acceptance arsenal), DQ (DU 7 findings closed, DT arsenal, oracle gap-0),
  DR (fire twin + fail-close verifications), DV (byte-identity both legs).
- All four new/changed mechanisms are default-off (Init(0)):
  -mtt-tensix-optimize-list-schedule, -mtt-tensix-optimize-lreg-alloc,
  -mtt-tensix-dst-layout-32b, -mtt-tensix-pressure-schedule-use-milp.
  ON-set promotion deferred until on-plus knob-leg silicon (owner order).

## Gates

(All at the union tip f5aa5bbf676b, laneDY build; fill VERDICT lines.)

- BUILD: stockcfg (laneDW recipe, --with-as/--with-ld COMDAT parity),
  ccache. Binaries:
  - cc1plus sha256: `c5bf1bf9aa2c9a36f210a3656b9ea40e6350f2387e760d138e431fe70b557393`
  - driver xg++ (== install riscv-tt-elf-g++) sha256: `0ce0f02f5033afb45e3a90c40b5e67857684767c3ed59f61eaddfd0b53f5577a`
  - xgcc sha256: `6ef8b21efd071baf9593aa12922412e4d334d876b9a59040d86d67f20c841830`
  - cc1 sha256: `a75df54a33a8d0d6a7c783ae6a133c87197d0083267c544247a5f3724a963d02`
  - lto1 sha256: `5674d1feb9863e59d1eb57144277303b06eb816c8fc58040feb4aaacc6500705`
- OPTCHECK: riscv_tt_opt_list_schedule / riscv_tt_opt_lreg_alloc /
  riscv_tt_dst_layout_32b / riscv_tt_pressure_schedule_use_milp all
  present in generated options.cc (2 refs each; bnb help text present)
  and all four flags ACCEPTED by the union driver: **GREEN**
  (optcheck/optcheck.txt)
- DEJAGNU full rvtt.exp (pin-14 blessed recipe: SFPI env + pinned
  -B/-isystem): **GREEN** — 4887 PASS, 16-row FAIL set byte-IDENTICAL to
  the pin-14 union reference (frozen-9 + 7 documented sfpi-env rows;
  diff empty, laneDY-dejagnu/fail-set.txt).  Families: lregalloc/ 53/53,
  list-sched-arsenal 31 PASS, list-schedule twins 21 PASS, lreg-alloc
  twins 53 PASS (incl. the 5 dg-error refusal twins passing WITH DP's
  inform notes — the DU note-3 dg-inform check), milp 40 PASS,
  lut-select-wh-nan 4 PASS, replay-raw-capture 6 PASS, synth-renumber 28
  PASS (the 4 constfold twins that ICE on pre-fix compilers PASS here).
- DS ACCEPTANCE GATE (tools/lreg_arsenal_gate.py --mode future, DP's
  wrapper-driver integration contract: union xg++ +
  -mtt-tensix-optimize-lreg-alloc -mtt-tensix-dst-layout-32b): **GREEN**
  — FUTURE: 0 row(s) failed, 25 PASS, row-for-row IDENTICAL to lane DP's
  accepted FINAL run (lreg-arsenal-gate-future.out).
- CORPUS byte-legs (shared farm b532e9958236, bh, legs rc=0):
  - flags-OFF (22 -mno) vs store b284dafdaacf/6b6b4facb92a: **GREEN**
    BYTE-IDENTICAL, SAME 3213 / CHANGED 0 / MISSING 0 / EXTRA 0, exit 0.
  - ON-25 vs pin-14 store 01aed0d8d58d/b5ef5f34d44f: **GREEN** —
    CHANGED exactly 13 / MISSING 0 / EXTRA 0, row-for-row IDENTICAL to
    lane DV's banked pre-existing DN/DL/DM widening set (already
    CRAQ-adjudicated at those merges); ZERO rows added by pin-15.
  - ON-25 vs a0ea2fcd9aa control (laneDW same-farm manifest): **GREEN**
    BYTE-IDENTICAL 3214/3214 — the three pin-15 merges are provably
    inert at the reviewed ON set.
  - TRUE-DEFAULT vs pin-14 truedefault entry (laneDV same-farm store
    entry, installed-pin compiler): **GREEN** BYTE-IDENTICAL, SAME 3213 / CHANGED 0 / MISSING 0 / EXTRA 0, exit 0.

## Install (staged, DO NOT RUN until validation legs report green)

    ~/sfpi-uplift/sfpi/scripts/pin-install-fast.sh \
      ~/sfpi-uplift/gcc-build-laneDY \
      ~/sfpi-uplift/sfpi/build/sfpi/compiler \
      --expect-cc1plus c5bf1bf9aa2c9a36f210a3656b9ea40e6350f2387e760d138e431fe70b557393 \
      --flags list-schedule,lreg-alloc

    # manual smoke for the two non-"optimize-" flags after install:
    ~/sfpi-uplift/sfpi/build/sfpi/compiler/bin/riscv-tt-elf-g++ \
      -mtt-tensix-dst-layout-32b -mtt-tensix-pressure-schedule-use-milp \
      -fsyntax-only -x c++ /dev/null
