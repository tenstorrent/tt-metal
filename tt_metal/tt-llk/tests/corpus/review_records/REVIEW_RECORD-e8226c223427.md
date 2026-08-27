# REVIEW_RECORD — pin 38

cc1plus sha256: e8226c223427f1fdc49ab54d4a4fae9b8bc0f5936931bb9d9ae2bece48f80b45
driver (g++) sha256: eaa9c0fd5122b149b9d774bef17462abd1565f5173ffc4bb0f7d9b688458c8d8
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver rebuilt with cc1plus)
source: sfpi-gcc nkapre/sfpi 9290941b633 = pin-37 union tip 39617ab327c
+ lane ID merge (agent/trig-loadi-gap: 3d2641cae41 loop-prgm-reclaim +
9290941b633 counted-row final lockstep audit, fast-forward).
Companions: tt-metal chain through a126050eb4c3 (ID knob registration
+ measured rows; the tanh-fold ratification 423ea34387 merged earlier
same day). KNOB_MODES dup grep clean (only the known benign
lut-select-fp16 pair). No sfpi include/ changes. Built in
gcc-build-laneFR (build-pin38.log rc=0); flags smoke-accepted
(OPTCHECK, mcpu=tt-bh); installed via pin-install-fast with loud
--expect-cc1plus; no live sweeps at install. Known infra note: the
tt-metal local hub carries a pre-existing missing object
b4651bda442ea654 (gc/repack fails in fresh clones; commits and
pushes unaffected) — cleanup owed.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

One lane, two deliverables (closing report, evidence dir + SHA256SUMS
1199, memory file):

(A) -mtt-tensix-optimize-loop-prgm-reclaim (Init(0)). Autopsy
reproduced HW row A7 exactly (sem 79 words/row with 13 in-loop
sfploadi vs hand 77/12; Acosh vehicle; ten in-loop candidates
refusing prgm-exhausted + lreg-file-exhausted at 8/8 pressure) and
named the mechanism: the shared production init claims PRGM L13/L14
with constants nothing in the sem TU reads — DEAD claims. The cheap
test honestly refuted the IC-composition premise (ON-36 +
hoisted-prgm-reuse and + crossrow-2datum both byte-identical: the
hoisted-only reclaim never serves in-loop candidates). Fix =
DEAD-claim reclaim offered to the walk's own in-loop classes, with a
window proof (crossloop-/cc-lifted entries refuse
loop-reclaim-call-window), reclaimed-slot reprogram + value-tier
belts, and a dead-scan slot discipline (never steal a pending
candidate's value-reuse home — found live as a digamma bring-up
regression; a words-saved ranking experiment was reverted as
pressure-blind). 4 dg twins incl the capacity near-miss.
trigonometry-fresh +5.58 -> +2.73 BOOKED (sem 395704 x3
cycle-identical vs hand 385199, anchors exact, causal -6.22, hand
byte+cycle inert); the row's word parity FLIPPED (76w/11 loadi vs
hand 77/12); the +2.73 residual is HW's chain-execution share, now
measured — successor = dependence-chain restructuring, plus a priced
reuse-vs-reclaim arbitration.

(B) UNGATED WRONG-CODE FIX — counted-row FINAL LOCKSTEP AUDIT. A
composed probe golden-FAILED tanh corr on the pinned sim; the root
cause is pre-existing: the counted-row canonicalization commits
occupancy-cascade renames AFTER lockstep verification with no final
re-check, and the pass's own BH fire twin's expected output decodes
as wrong code with no new flags involved. Fix = a fail-closed
structural re-verify under the final assignment (refusal
counted-row-final-lockstep-divergence); corpus byte-inert at every
flag state (mapped-corpus blast radius zero); both fire twins
updated to the sound outcomes. Named successor: a counted-row
rename-repair pass (recover the fires the audit now refuses).

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin38; pinned-install
  env): 6749 PASS = pin-37's 6728 + ID's 21 exactly; FAIL set 16
  rows LINE-IDENTICAL to the pin-37 frozen baseline (diff empty).
  Flags smoke-accepted (OPTCHECK).
- Corpus (ID's own store): OFF/TD/ON-36 3300/3300 x3 .text-identical;
  loop-prgm-reclaim knob corpus delta = ZERO TUs (perf-vehicle fire
  only, the HO-F2 pattern); IC's crossrow-2datum delta preserved at
  its exact 12 TUs; full-unary compile sweep 5171 pass (2
  pre-existing flag-independent fails).
- Preservation: 71-leg loss+WIN screen 0 CHANGED.
- Paired CRAQ 2/2 + 2/2 on the pinned sims (bh 1d162f0adf67); device
  corr corr-first every session.
- Silicon (BH p150, 3-rep cycle-identical): trigonometry-fresh
  +2.73 booked; anchors (sem 406712 / hand 385199 / off 421946)
  reproduced exactly.
- Board: b995f7b1368b -> 079910d60b9b (trig row only; tally
  78W/35P/21L, incl the same-day owner-ratified tanh twin fold).
- Evidence: laneID-evidence-20260827 (+SHA256SUMS 1199).
- Install: sha-verified ec1ceaa776dd... -> e8226c223427...; driver
  read from the fresh manifest entry.
- ON set UNCHANGED at 36 (loop-prgm-reclaim registers as an on-plus
  booking knob; the lockstep audit is ungated fail-closed
  soundness).

## Gates

conf_lint GREEN and witness_preflight ALL GREEN at ON-36 on the
installed binary (outputs in ~/sfpi-uplift/sweep-2x2/pin38-ceremony/).
