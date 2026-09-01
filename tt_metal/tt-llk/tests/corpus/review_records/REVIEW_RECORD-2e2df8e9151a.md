# REVIEW_RECORD — pin 52 (FABLE_GOES_BURR wave 4 + priced-placement promotion)

cc1plus sha256: 2e2df8e9151a53ad2bf6a83f08abd5a717af1e8f8a5d7d87525cfe734e1be6e6
driver (g++) sha256: 845201943ca9314acb8df56de72678fe0127f4c68452c32264fb846f42a3a12e
(read from the CURRENT PIN-INSTALL-MANIFEST entry; driver REBUILT —
wave-4 .opt options: rename-temporal, cc-region-general)
source: sfpi-gcc nkapre/sfpi 61a98b80944 = pin-51 1b531ee842f + three
wave-4 merges in seam order: KK agent/laneKK-w4c-v1-retirement
6fbb945e706 (fast-forward), KJ agent/laneKJ-r1-lreg-reattack 8382ffe7e66,
KL agent/laneKL-r2-cc-dance 52902ac80e5 — ZERO conflicts (the KJ/KK
shared rtl-rvtt-lreg-rename.cc composed exactly per both lanes'
pre-exchanged NOTE files: KK's v1 deletion + KJ's temporal tier/service
bodies; verified post-merge: v1 pass absent from passes.def, temporal
tier + undo record present).  tt-metal canon: + KJ 750a0a113d (knob) +
KL f7fd18f681 (knob) + KM agent/laneKM-promotion 7e351e4becb (THE
PROMOTION) — one sweep_2x2.py KNOB_MODES conflict resolved keeping both
KM's promoted drop-one row and KL's cc-region-general row.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed

Full lane records in laneK{J,K,L,M}-evidence dirs; headline facts:

KK W4-C: v1 single-shape rename RETIRED AS WRONG CODE — 11/13 corpus
fires adjudicated miscompiles (own-regno-read-blind whole-pattern writer
edit; the calculate_i0 two-word constant pair split into garbage;
15-line reproducer banked).  NOTHING EVER SHIPPED (flag outside every
production/reviewed set at all times).  Flag retained as a frozen-API
alias to the general engine; alias == v2 == composed fire-set identity
at 2,446 sites, refusal tallies string-identical to KE's profile.

KJ R1: temporal rename tier + cyclic-interior consumer, every rename
through KE's service (which gained an exact-undo record, byte-proven).
TRIG: modeled II 99->95; silicon -1.00% on the booked cell (395833 ->
391865 x3 reps cycle-identical, laneJU anchor reproduced exact first,
hand arm byte/cycle-inert); vs_hand +3.86 -> +2.82 LOSS STANDS,
re-booked in place on FINAL-BOARD.  Wall -25.8% corpus-wide
(no-free-lreg 22,602 -> 16,769; 2,598 temporal renames).  erfinv
pressure family REFUSED-BY-NAME with a structural argument (storage
bijections cannot reduce value-liveness pressure; relief = #13).

KL R2: five tree-licensed admissions behind
-mtt-tensix-optimize-cc-region-general Init(0): ccmask EQ/NE folds
PROVEN EQUAL over 2^32 per direction (new proof dir
tt/proofs/ccmask-eqne-zero; compare-direction-unsupported 20 -> 0);
stage-B any-layout matcher; crossloop all-lanes-entry V2 +
balanced-frame V1 facts (872 -> 851 walk-stops); narrowing-writer audit
(+5 families, tt/proofs/cc-narrowing-writers); reassoc window
transparency.  Knob legs move exactly 15 rows, all adjudicated;
lgamma/polygamma boundary refusals STAND BY NAME (mask-equality
follow-on identified).

KM PROMOTION (conf-only, measured at the pin-51 binary):
priced-placement JOINS THE REVIEWED ON SET, 36 -> 37.  All four
obligations green: (1) erfinv MEASURED — kernel 376756 -> 370739
(-1.60% x3 cycle-identical; causal -1.31 -> -2.88; device-golden 18/18;
the laneJT-era "no perf selector" fact was stale at pin 51); (2)
softplus re-certified at floor by bytes (12/12 TUs); (3) all six
pin-34 loss rows byte-identical corr+perf (the laneHY shape
structurally excluded); (4) corpus ON-36-vs-ON-37 re-diff = exactly the
erfinv corr TU.  Board consequence: erfinv-fresh vs-hand -42.03 ->
-41.08 WIN STANDS (the promotion accelerates the hand arm too — the
honest laneHE shape).  dst-ownership witness upgraded to the fold's
fire line (its 9>8 refusal is cleared by the promotion); pre-existing
selftest witness-group budget fixed (16 -> 17).

Paper lane KI (private nkapreTT/2027-cgo_craq-sfpi only, visibility
verified): laneKC silicon certification folded into both papers.

## Gates checked

- Union rvtt.exp WITH the SFPI env (dejagnu-pin52): 7578 PASS = 7488 +
  KK 9 + KJ 22 + KL 59 EXACT; FAIL set 16 rows LINE-IDENTICAL; dg
  ERROR 0.
- UNION CORPUS GATES (sweep-2x2/pin52-ceremony/corpus/): pinned-51 vs
  union ON-36 3300/3300 BYTE-IDENTICAL (diff-on.txt empty); union chkon
  rc=0, ZERO assert ICEs, .text == union-ON (diff-chkon.txt empty).
  CEREMONY LESSON BANKED: the first reference-leg attempt was refused
  by the leg store's SOURCE-TREE-CHANGED-MID-LEG tripwire — the
  tt-metal wave-4 merges landed while it compiled; re-run on the stable
  tree with the frozen ON-36 list.  tt-metal merges wait for corpus
  legs at future ceremonies.
- ON-37 + ON-37-chk legs at the union hybrid (post-promotion set),
  results in the ceremony dir.
- witness_preflight at ON-37 on the INSTALLED pin-52 binary: ALL GREEN
  (incl KM's new priced-placement row and the upgraded dst-ownership
  row).  conf_lint GREEN.
- Trig temporal cell re-measure at the ON-37 union (KJ's named pin-52
  obligation): executed at this ceremony; verdict appended below.
- Board: tally UNCHANGED 84W/35P/15L (KJ trig + KM erfinv re-books in
  place with provenances).
- ON set 36 -> 37; KNOB_MODES 49 -> 51 (+rename-temporal,
  +cc-region-general; priced-placement -> drop-one).
- Install: sha-verified 77ce6392c080... -> 2e2df8e9151a...; driver
  rebuilt, read from the fresh manifest entry.
- PIN-53 DELETION LIST (carried): KH's #15 shadows + legacy asm parser.
- Push state: pending at record time; pushed tt-metal + sfpi-gcc all
  hops incl the private craq mirror (verified before pin close).

## Trig temporal-cell re-measure at the ON-37 union (KJ's pin-52 obligation)

CONFIRMED — booking HOLDS EXACT, board untouched: sem control 395833.0
x3 / treatment 391865.0 x3 at the ON-37 union leg on the installed
pin-52 binary, both cycle-identical to laneKJ's pin-51 booking; vs_hand
+2.82 LOSS unchanged, causal -8.90 unchanged.  Compile-probe
certificates first: the promotion + pin-52 union are INSTRUCTION-INERT
on the trig row (both arms identical to KJ's ELFs modulo the 6
farm-path profiler LUIs) — consistent with laneKM's one-moved-TU diff.
Hand arms skipped by the certificate rule (instruction-identical to
KJ's measured binaries, 381105 x3 both arms).  RECORD-ONLY precision
fix for the next trig-note edit: the hand perf TU is cycle-inert but
NOT byte-inert under +rename-temporal (register-field-only renames);
KJ measured the load-bearing cycle-inertness on device, so the
comparator stands.  Evidence: pin52-trig-remeasure-20260901/.
