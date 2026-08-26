# REVIEW_RECORD — pin 30

cc1plus sha256: 106e98daddc7cb9b15f65e65bc11d97fd5e9509b50dc9e6a019378c065f83f1b
driver (g++) sha256: 79bd9fb1b877c8d65ac1b002afba2ef3c2ffaead6d2eba18ebd061a487c93438
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi ca8e9e19386 = pin-29 union tip 3f5af960548
+ four lane merges (HB agent/rule-b-preservation-seed 9a785004d8a,
HF agent/tanhderivlut-residual fbfdb66f498, HC agent/lut-prefix-hoist
8e5f422b3d8, HH agent/topk-window-density 977386f087a). Companions:
tt-metal knob chain through 2c58d562c9 (HE ON-34 promotion conf
371f8162fb -> HC knob 790321c7d2 -> HB knob 7b3ed05bd3 -> HH knob
2c58d562c9; each knob-table both-append conflict union-resolved,
py-parse + conf-lint verified each time). No sfpi include/ changes
since pin 29. Built in gcc-build-laneFR (build-pin30.log rc=0); all
four new flags smoke-accepted together (OPTCHECK,
NEW-FLAGS-SMOKE-OK); installed via pin-install-fast with loud
--expect-cc1plus; no live sweeps at install.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

Four lanes, each with its own closing report, evidence dir (+SHA256SUMS)
and memory file: HB (DESIGN-V2 Rule-B preservation-seed rename for
crossrow pairing: full-lane roots rename SEED-FREE — the audited
rvtt.md all-lanes SFPMOV-mod-2 effect fact carried into the atom
interior — while predicated roots get charged one-word preservation
seeds in the same II model; II 32->28 = word floor, DT makespan 50->48
= audited lower bound; silicon -4097cy EXACTLY = cycle-exact model
transfer; roundingops vs-hand +7.80 -> +1.20), HF (the pin-16
EL-composition PHANTOM: EL's invariant-loadi widening hoists the free
creg read LReg10=1.0 and lut-select's transactional pressure budget
counted it as a 9th pinned LREG, refusing ALL six coefficient
placements since pin 16 — empirical .pin-backup ladder 14..29 in
BISECTION.md; fix = laneGU's exact-obligation counting extended to the
FP32-direct leaf-ext path + creg_resident_p soundness belt, riding the
reviewed lut-select-leaf-ext flag; tanhderivlut 8->5-word loop = hand
parity, +47.50 -> +17.75), HC (crosscall config-prefix: the PRGM
programming pair joins the crosscall contract and the placement
residency walk lifts to kernel entry, matching hand's once-per-kernel
discipline; the brief's crossloop-census hypothesis honestly REFUTED —
the true blocker was crosscall-callee-vector-outside-loop; geluappx
+6.25 -> PARITY +0.03), HH (launch-flatten: GIMPLE cunroll's size
estimate starved typed arms of the complete unroll raw-word arms
always get; annotation-only pass sets loop->unroll = proven trips
immediately before pass_complete_unroll; fail-closed admission —
innermost/single-exit/SCEV-constant trips, typed-content requirement,
per-loop XTT_REPLAY_LOOP_UNROLL_{MIN,MAX}_WORDS + per-function
1024-word budgets; topk-perf +4.64 -> WIN -0.82, sem faster than hand
in the sort body itself).

## Gates checked

- Union rvtt.exp WITH the full SFPI env (dejagnu-pin30): 6343 PASS,
  FAIL set 16 rows LINE-IDENTICAL to the pin-29 frozen baseline (diff
  empty). All four new flags smoke-accepted together on the union
  driver.
- Per-lane corpus byte-gates vs the pin-29 stores: OFF/TD/ON-34 legs
  byte-identical in every lane (Init(0) proven; HF's plain legs
  byte-identical by construction under the leaf-ext gate); knob deltas
  fully adjudicated per lane with paired CRAQ green on the pinned bh
  sim.
- Per-lane silicon (BH p150, 3-rep, corr-first; controls hold):
  roundingops +1.20 (HB, -4097cy exact model transfer); tanhderivlut
  33972 KERNEL -20.2%, +17.75, corr PASS both legs (HF); geluappx
  +0.03 PARITY (HC); topk-perf 5708 = -0.82 W vs booked hand, -1.04 vs
  same-leg hand-under-knob, 3/3 reps cycle-exact (HH).
- Evidence: laneHB-evidence-20260825, laneHC-evidence-20260825,
  laneHF-evidence-20260826, laneHH-evidence-20260826 (SHA256SUMS in
  each).
- Install: sha-verified f47f72b40b8a... -> 106e98daddc7...; driver
  read from the fresh manifest entry; no sfpi include/ staging owed.
- ON set UNCHANGED at 34. BOARD at cut: 74W/33P/28L.

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-34 on the
installed binary (run at ceremony commit time; outputs in
~/sfpi-uplift/sweep-2x2/pin30-ceremony/).
