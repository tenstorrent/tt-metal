# REVIEW_RECORD — pin 28

cc1plus sha256: 2a71feada1d944b4c5b8c114495a8e084c722dd1dbb4cea3ac617f0ae25d69af
driver (g++) sha256: 5811b9eb1ad2db68bb825c5059ef93e3c4ec6d9b55f3311257662be0edf437be
(driver rebuilt with cc1plus; sha read from the CURRENT
PIN-INSTALL-MANIFEST entry per the pin-27 ceremony rule)
source: sfpi-gcc nkapre/sfpi fd2bb4a481d (merge of lane GJ
agent/serial-chain-pairing 47200951071 off pin-27 tip 0045d296318;
clean merge). Companion: tt-metal nkapre/sfpi f891e976cd (on-plus knob
registration for the new flag). Built in gcc-build-laneFR
(build-pin28.log rc=0); both window-pairing flags smoke-accepted
together on the union driver; installed via pin-install-fast with loud
--expect-cc1plus verification; no live sweeps at install; no sfpi
header changes.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed commits/branches

- Lane GJ (serial-chain window-pairing class, closing report
  2026-08-24), sfpi-gcc agent/serial-chain-pairing 47200951071 +
  tt-metal agent/window-pairing-stride-knob 6f428139cd. Mechanism =
  candidate (c): stride-phase generalization of FT's audited
  pending-event model, new Init(0) flag
  -mtt-tensix-optimize-window-pairing-stride (on-plus knob). The
  absorbing Dst advance may ride ANY issued word; every Dst footprint
  rebases by its carrying word's stride phase (pending +phi*stride,
  follower row j +(j+phi)*stride). Exactness audited as rvtt-cost.md
  F5': SFPLOAD resolves its address before ApplyPartialAddrMod (= F5);
  SFPLOADMACRO-hosted events latch their Dst row AT LAUNCH
  (SFPLOADMACRO.md StoreSubUnit + craq-sim 9f324140 macro_dst_row
  latch); absorber-on-last => all phases 0 => F5 verbatim; unprovable
  carrying word / carrier_pos<0 fail closed; no new constants.
- Root cause fixed: FT's tuner refused GG's limb-2 macro schedule by
  window-pairing-stride-unproven (store+stride absorption hosted on
  the FIRST issued word vs F5's absorber-on-last invariant) — GG's
  banked "2 drain nops/row delivery ceiling" on mulint32 was a
  compiler gap, not physics.
- Per-target autopsy verdicts (AUTOPSY.md): roundingops +7.92 = the
  full FI section-3c cross-row pairing build (rename + interleave +
  CC-state-equality placement + Dst re-planning) — named refusal with
  the gap arithmetic (1.19 cy/row x 4096 = the booked 4871-cycle gap);
  lcm-fresh +6.61 = no-mechanism (RecMII recurrence floor; 2-row
  pairing needs 10 live > 8 LREGs; DP-Dst-spill of the two round-phase
  invariants = named unlock); recip +1.01 = no-mechanism (window
  already paired per FI's ilv2 certificate; residual = per-face
  re-records = record-hoist crosscall class, EC v1 scope-out).
- Twins: window-pairing-stride-{limb2-fire-bh,
  limb2-fire-renamed-varied-bh, limb2-off-identity-bh,
  inert-signbit-wh, interleaved-keep-bh} — the last is the mandated
  GG interleave-2 keep twin (mechanism never re-walks GG's refused
  shape); window-pairing-stride-unproven retained + extended
  fail-closed.

## Gates checked

- Union full rvtt.exp (dejagnu-pin28, srcdir at merged tip): 5905
  PASS (= pin-27's 5876 + GJ's 29 exactly); FAIL set 16 rows
  LINE-IDENTICAL to frozen (diff vs dejagnu-pin27/fail-set.txt empty).
- Lane GJ gates (its build 2e32e5c08211; REVIEW_RECORD-2e32e5c08211.md
  staged alongside): dg window-pairing 56/56 (5 new twins) +
  macro-planner 1200 + drain 180 + replay 646 + record-hoist 160 +
  dst-autoincr 376, 0 FAIL; full rvtt.exp 5905 frozen-16 identical on
  its binary; corpus 7-leg battery base-vs-fix OFF/TRUE-DEFAULT/ON-28
  = 3252/3252 .text-identical each; KNOB vs ON-28 = exactly ONE
  changed TU corpus-wide (mulint32-fresh corr TU, CRAQ PASS pinned
  bh sim 32489dda, lane-side + in-sweep paired legs).
- SILICON (headline-laneGJ-20260824c/d, BH p150, 3 reps,
  corr-before-perf, device healthy): mulint32-fresh knob FIRE —
  KERNEL 38669 -> 35077.7 (-9.29%), vs-hand +5.11% -> -4.65%
  LOSS->WIN (interrow drain 2->1, window 0,30->0,27, boundary
  pairs->singles, matching the model's ~28 slots/tile); residual
  bound = named window-pairing-lreg-overlap (real fixed-VD WAR
  hazard; per-row VD alternation via descriptor synthesis = named
  follow-up). roundingops / lcm-fresh / recip knob legs
  byte-identical = measured honest no-fire, as pre-registered.
  Composition controls hold: blaze-sdpareducerow-max-t8 +0.97,
  hardsigmoid-fresh +0.89, all verdicts GREEN vs baseline.
- Install: sha-verified 981d8af93ff0... -> 2a71feada1d9...; driver
  sha from the manifest (rebuilt, as expected); flag smoke OK for
  both window-pairing flags together; no include/ changes.
- Evidence: ~/sfpi-uplift/laneGJ-evidence-20260824/ (SHA256SUMS, 122
  files), ~/sfpi-uplift/dejagnu-pin28/.
