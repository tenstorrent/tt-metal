# REVIEW_RECORD-2e32e5c08211 — lane GJ window-pairing-stride measurement hybrid (NOT a pin)

Pin: cc1plus sha256 `2e32e5c082112e5e9e7fae1b4a53d74e4ba997c841648050843bd40184332042`
driver (g++/xg++) sha256 `be064551c6f1e8f450e74f5dbf6b81fa5792b326a224af0a6fd375fb8f66780f`
Built from: sfpi-gcc `agent/serial-chain-pairing` @ 47200951071 (single commit
off pin-27 canon 0045d296318), gcc-build-laneGJ (laneFR stock configure +
ccache).  The change ADDS -mtt-tensix-optimize-window-pairing-stride, so
hybrid-laneGJ-fix swaps BOTH cc1plus and the driver (laneAO/FT new-flag
rule).  This record authorizes lane GJ's TARGETED measurement sweep only
(mulint32-fresh / roundingops / lcm-fresh / recip + composition controls
blaze-sdpareducerow-max-t8 and hardsigmoid-fresh); it does NOT install or
pin a toolchain — the installed pin stays 981d8af93ff0.
Date: 2026-08-24
Reviewer: lane GJ session (Claude, operated by nkapre@tenstorrent.com)
Independence: NOT independent — self-review by the authoring lane, per the
targeted-measurement-hybrid precedent (laneFT cf5ab965d544, laneFZ
e49855142a77); the mechanical gates below are the independent half.

## Reviewed commits/branches

- sfpi-gcc 47200951071 — window-pairing stride-phase generalization: the FT
  inter-row drain tuner admits the advancing address mode on ANY issued row
  word under the new Init(0) flag; every Dst footprint rebases by its
  carrying word's stride phase (rvtt-cost.md F5': SFPLOAD.md
  resolve-before-ApplyPartialAddrMod; SFPLOADMACRO.md StoreSubUnit
  launch-latched Addr "regardless of whether SFPLOADMACRO advanced any
  RWCs"; craq-sim 9f324140 macro_dst_row latch = L1).  Fail-closed:
  flag-off keeps the compact-form refusal byte-identically; no provable
  carrying word / carrier_pos<0 refuse by the established name.  No new
  constants.
- tt-metal agent/window-pairing-stride-knob @ ad0ba68aa8 — KNOBS/KNOB_MODES
  on-plus registration only (conf-lint GREEN; knob-legs/report/enforcement
  selftests green).

## Gates checked

- [x] byte-identity of default codegen vs the pin: 7-leg corpus battery,
  base(981d8af93ff0) vs fix(2e32e5c08211) x {OFF, TRUE-DEFAULT, ON-28} =
  3252/3252 .text-identical EACH (store corpus-legs-laneGJ); KNOB
  (ON-28 + flag) vs ON-28 = EXACTLY ONE changed TU corpus-wide (the
  mulint32-fresh corr TU f9ef8980..., .text 8337da54 -> b6e43a10).
- [x] focused DejaGnu families green (window-pairing 56/56 incl. 5 new
  stride twins; macro-planner 1200, drain 180, replay 646, record-hoist
  160, dst-autoincr 376 — all 0 FAIL); full rvtt.exp 5905 PASS, FAIL-16
  LINE-IDENTICAL to the frozen pin-27 set (dejagnu-laneGJ, final binary).
- [x] paired CRAQ green: the single changed TU passes pinned bh sim
  32489dda (libttsim 1fb30514fcab staging), tuned 0,27-window ELF
  .text b6e43a10 byte-verified in the CRAQ build.
- [x] refusals byte-identical where required: flag-off recompile of the
  mulint32 TU == pin bytes (7be4ed7d4c78dad8); roundingops / lcm / recip
  knob recompiles byte-identical to their ON-28 dumps (honest no-fire).
- [x] no hardcoding introduced: the change is pure position arithmetic on
  the existing audited model (phase = carrying-word position vs absorber
  position); no op names, calendars, coefficients, or magic words; the
  renamed-varied twin pins structure-not-identity.
- [x] known risks / carry-forwards: the remaining 1-nop bound is the REAL
  fixed-VD WAR hazard (window-pairing-lreg-overlap) — full closure needs
  per-row VD alternation through descriptor synthesis (named follow-up);
  the GG interleaved-source shape stays refused (keep twin).

## Limitations

Self-review by the authoring session; the honesty half rests on the gates
above and the archived evidence (laneGJ-evidence-20260824).  WH coverage is
an inertness control (no WH macro shape derives a mid-row absorber today).
