# REVIEW_RECORD — pin 55 (the rename/CC fix pin)

cc1plus sha256: 4556f1644d9bf174d2f272dae313fd2cf6efcaf973a8e8cb39a1174f478a98ef
driver (g++) sha256: dded27bb7d1d726cfc9f1d07a92c00c3aee2493797caf341d10bd9ba28eb1626 (UNCHANGED)
source: sfpi-gcc nkapre/sfpi 53f02910df4 = pin-54 6191f71fa91 + KZ
ad2ba97346 (chains DF-liveness fix) + LA 7d55995f361 (temporal
dest-reuse fix) + LB 2e286448eb8 (cc-region peel-entry fix).  Zero
merge conflicts (KZ/LA NOTE-agreed boundary held).

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed
The three silicon-adjudicated defects from promotion round 5, each
root-caused, fixed FAIL-CLOSED, red/green-proven on sim AND silicon,
with named refusals and belt strengthenings (full records in
laneK{Z}/laneL{A,B}-evidence-20260902/):
- KZ: DF hard-register liveness is unreliable in opaque-instruction
  functions; whole-block renames there now refuse
  regrename-liveness-untrusted (fires 532 -> 69); the belt fails
  closed on the class.
- LA: temporal chains whose writer reads its own destination lose the
  writer's kill; refuse regrename-temporal-dest-reuse (fires 102 ->
  80, all refusals named).
- LB: peel-class placements are entry-anchored (refuse
  crossloop-peel-entry-anchored); this also cured SILENT wrong code on
  2 ATAN2 corpus rows (device-proven) — any cc-region-general
  re-promotion must carry it.  Gate gap named: corpus lacks
  board-harness TUs.
TRIG RE-MEASURE (LA moved the licensed leg's bytes): WIN -1.51 STANDS
EXACT (sem 375354 x3 / hand 381105 x3, booking chip, corr-first, hand
measured fresh); causal re-based -8.66; the temporal token buys 0
cycles on the row post-fix.  Board post c376e9b57f378198; tally
85W/35P/14L unchanged.
CLEANUP WAVE 6A closed off this union (LD legal/style incl 2 latent
rots exposed by the DUMP conversions; LF the manual, +1,788 texi
lines, 97/10/6 option triage; LG hygiene, 170 artifacts evacuated, 6
unit harnesses wired, 29-row cited-inventory for the vocabulary lane).
All byte-inert, dg 7722 each; branches MERGE OWED pin 56.

## Gates checked
- Union rvtt.exp (dejagnu-pin55): 7722 = 7708 + 4 + 10 + 0 EXACT;
  FAIL-16 LINE-IDENTICAL; ERROR 0.
- Corpus: pinned-54 vs union ON-39 3300/3300 BYTE-IDENTICAL; chkon
  rc=0, ZERO ICEs, .text == ON.
- conf_lint GREEN; witness_preflight at ON-39 (below).
- Board 85W/35P/14L; ON 39; KNOB_MODES 56.
- Install: 46d3116c469d (held during fix-lane compiles, correctly) ->
  eaccb93b0e98 (pin 54) -> 4556f1644d9b.
- PIN-56 QUEUE: 6A merges + 6B (LE/LH/LJ) when closed; re-promotion
  round for the fixed rename/CC stack; lp-schedule alias (owner);
  addint/sem-corr-wh archaeology.
