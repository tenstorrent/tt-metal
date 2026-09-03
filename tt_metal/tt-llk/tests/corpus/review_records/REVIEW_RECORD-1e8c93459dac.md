# REVIEW_RECORD — pin 58 (the perf-refine pin)

cc1plus sha256: 1e8c93459dac0268c13379a15d275821d42075b7432a91e5acc6ee062ee8d43c
driver (g++) sha256: f285b235fceeb8c802d058e8cbe92353587b44968f2a4478e9c5293c31e1168d (UNCHANGED)
source: sfpi-gcc nkapre/sfpi f048362a2d5 = pin-57 11c316a9aa3 + LN
980e03321b8 (ccg priced WHEN-gate) + LO d36bdf10379 (temporal
strict-gain self-pricing) + LP f42423babf6 (chains periodic-window
structural refusal).  Zero conflicts.

Reviewer: orchestrator session (Claude, operated by nkapre@tenstorrent.com)

## Reviewed
Round 6's three perf-refine successors, each closing its regression by
PRICE or STRUCTURE (never a row name), each keeping the pin-55
wrong-code fixes intact (KZ/LA/LB reproducers pass at every leg):
- LN: one mechanism explained all six ccg regressions (the 6-word
  EQ/NE mask composition vs the 2-word CC arm, replay-amplified);
  priced through the delivery-cost engine; ON+flag corpus-byte-inert;
  14/14 device cells +0.000%.
- LO: two named cost classes (span-external coupling; stream-identity
  externality); the temporal tier self-prices at a strict-gain bar and
  is honestly fail-closed until the timing model grows a class a
  rename can win; both regressions +0.00%; trig cell reproduced exact.
- LP: repeated delivered-word windows never rename (divergence
  guaranteed; SCRATCH-widened pattern equality); rotate90 exact parity
  with the i1 win preserved.
Also committed: corpus/tools/galaxy-kit/ (the reusable Exabox campaign
kit + the project EXABOX.md: route, hold-vs-ssh doctrine, etiquette,
measurement honesty).

## Gates checked
- Union rvtt.exp (dejagnu-pin58): 7783 = 7765 + 5 + 5 + 8 EXACT;
  FAIL-16 LINE-IDENTICAL; ERROR 0.
- Corpus (parallel legs): pinned-57 vs union ON-39 3300/3300
  BYTE-IDENTICAL; chkon rc=0, ZERO ICEs, .text == ON.
- conf_lint GREEN; witness_preflight at ON-39 (below).
- Board UNCHANGED 85W/35P/14L @ c376e9b57f378198; ON 39; KNOB_MODES 56.
- Install: d7bcca93125a -> 1e8c93459dac.
- OPEN QUEUE: the galaxy replication ledger (laneLK, running on hold
  75700) -> paper/dashboard enrichment + the abs stale-cell
  adjudication; promotion round 7 (extra-corpus value case); owner
  items (optimize-combine, lp-schedule alias, FSF, HotCRP).
