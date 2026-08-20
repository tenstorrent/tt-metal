# REVIEW_RECORD-01aed0d8d58d — silicon authorization for the pin-14 build

Pin: cc1plus sha256 `01aed0d8d58dc79459d32eaaba7e1ad3fa02dede5552c1be224c663705c14bb3`
Driver (xg++): sha256 `5d3742f5847279f59f155b6cdad907f4c2e561b529d60ea2265bffb5ba21d290`
Built from: sfpi-gcc `staging/pin14` @ `3ca94518817` (base = census-fix tip e0754714a5b
on pin-13 union 8ae4a2d6b01). INSTALL METHOD (owner-ratified 2026-08-20 fast path,
sfpi `scripts/pin-install-fast.sh`, PIN_REVIEW-3ca94518817a in the sfpi repo): the
gate-proven union binaries installed directly — cc1plus + cc1 + lto1 + xg++/xgcc
together (driver embeds option tables); shas verified pre/post; all 25 ON flags +
4 new default-off flags accepted; default compile verified; pin-13 binaries backed up.
Date: `2026-08-20`
Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)
Independence: lane mechanisms were adversarially audited cross-lane (CJ audited CG;
CT refuted CO's flagship; CY2 audited CY; the union gates ran independently of every
lane in the pin-14 gcc batch merger). Not third-party independent — an independent
re-review supersedes this record.

## Reviewed commits/branches (9 lanes + census fix, zero exclusions)
- Census-rooting fix CG `8dee5e84029` + CJ production twins `e0754714a5b` — LIFTS the
  wave-9 quarantine (conditions: census roots-or-fails-closed, init-hoist ICE fixed,
  zero-trip prose true; adversarial audit verdict SUFFICIENT; wave-10 verified clean).
- CT cast-peephole refusal record + proof template (docs/proofs only, fc70df6a87b4a)
- CI commuted-SFPMUL24 derive admission (28fcc88c5d3)
- CU int-abs ccmask-fold, flag -mtt-tensix-optimize-int-abs (8bdf02155f0)
- CP replay-owner barrier witness, testsuite-only (f5ab0aaad47c)
- CK planner-launch effect records (045a57db457)
- CN representation-propagation pass, -mtt-tensix-optimize-repr-prop (baf80e128d4)
- CV replay-loop-unroll pass, -mtt-tensix-optimize-replay-loop-unroll (7f3f31464d5)
- CY lut-select leaf/arity extension, -mtt-tensix-optimize-lut-select-leaf-ext (cf2ad76ab8c)
- CZ derive-core vocab admissions + offline enumerator (080726caac9)
All new flags Init(0) — OUT of the reviewed ON set; measured via knob legs only.

## Gates checked (pin-14 gcc batch merger, evidence ~/sfpi-uplift/pin14-evidence-20260820/)
Full rvtt.exp 4540 PASS / FAIL set byte-identical to base (frozen-9 + 7 documented
sfpi-env rows); focused twins all families 0 FAIL; corpus flags-off 3211/3211 AND
reviewed-ON-22 3211/3211 byte-identical vs base; CRAQ pinned sims 17/17 witness legs.

## ON set at this pin: 25 flags (22 + lifted crosscall-hoist/crossloop-hoist/init-hoist)
## Pre-registered movers (first silicon = the pin-14 full sweep)
minmax ~16.72 WIN, where ~154.5 WIN, exp ~73.8, reduce-sdpa ~832.75 WIN-back,
welford ~322, eqz -> parity (Option A body), castfp32tofp16a -> ~15.9 parity
(golden re-spec), gcd-fresh ~770 modeled BEAT-hand, log/sqrt/rsqrt +1.6-3% band;
knob legs: absint32 (int-abs), hardshrink/hardsigmoid/softsign (replay-loop-unroll),
tanhderivlut (leaf-ext, licensed-leg-pending-owner), repr-prop rows, typecast A/B.
