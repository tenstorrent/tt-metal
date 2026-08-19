# REVIEW_RECORD-conf-quarantine-census-wave9 — wave-9 quarantine of the three census-dependent flags

Change class: conf/harness ONLY (sweep ON set + knob table + witness-table
split + conf-lint R10).  The toolchain pin is UNCHANGED: cc1plus
`8e87fba0e35f2a2b4a80981310afc3601a1ce34131518dea1b62c6afc1b030d5` (pin 13)
stays installed; the quarantined flags exist in the binary but are never
passed.  This record is required because removing reviewed ON-set entries
is a reviewed change, not a bare flip (same discipline as adding them).

Date: `2026-08-19`
Reviewer: lane CQ (Claude, quarantine-enforcement lane, operated by
nkapre@tenstorrent.com), executing the wave-9 adjudication verdict.
Independence: the VERDICT is independent (wave-9 adjudication, sfpi repo
commit `4adac11aac1`, docs/handoff-20260817/HANDOFF.md item 13); this
record implements its REQUIRED clause mechanically.

## Verdict being enforced (sfpi 4adac11aac1, HANDOFF item 13, REQUIRED clause)

"crosscall/crossloop/init-hoist flags must be treated as QUARANTINED
(never exercised, never promoted, no silicon) until the census roots
externally-visible/entry symbols or fails closed, the init-hoist ICE is
fixed, and the zero-trip prose matches behavior; the census fix is now
blocking THREE mechanisms instead of one."

Grounds (from the verdict): the wave-8 CONFIRMED wrong-code-capable
TU-census rooting hole is UNFIXED (compute_executable_closure
byte-identical; repro re-confirmed on a cold tip build); crossloop-hoist
and init-hoist are NEW consumers of the same census whose MOP-safety
audits pass vacuously on production-shaped (externally-entered) TUs;
init-hoist SEGFAULTS the compiler on any TU whose census is non-trivially
rooted (it has never executed with a functioning census); the pass
header's zero-trip soundness clause remains empirically false.

## Changes in this commit

- `sweep_2x2.py` ON_FLAGS: removed `-mtt-tensix-optimize-crosscall-hoist`,
  `-mtt-tensix-optimize-crossloop-hoist`, `-mtt-tensix-optimize-init-hoist`
  (25 -> 22 flags) with an in-place QUARANTINE block citing the verdict
  and lift conditions.  OFF_FLAGS keeps the `-mno-` spellings (asserting
  the flags off is not exercising them).
- `sweep_2x2.py` KNOBS: removed the `crosscall-hoist` and `crossloop-hoist`
  entries — a knob-attribution leg compiles OFF + the single positive
  knob, which would exercise a quarantined flag.  (init-hoist never had a
  knob entry.)
- `sweep_2x2.conf`: QUARANTINE prose in the CURRENT PIN block; PIN
  HISTORY #13 quarantine note (same pin, conf-only change); the
  crosscall-hoist and crossloop-hoist rows MOVED VERBATIM from
  `_REVIEWED_FIRE_WITNESSES` to a new `_QUARANTINED_FIRE_WITNESSES`
  table (evidence preserved, never deleted; witness_preflight.py reads
  only the reviewed table, so quarantined rows are never compiled).
  init-hoist had no witness-table row (its pin-13 witness lives in
  laneCA-evidence-20260819 and REVIEW_RECORD-8e87fba0e35f); its
  quarantine is enforced by the ON-set removal.
- `conf_lint.sh`: new rule R10 — the quarantined table must parse with
  R9's row format, and its flags must NOT be in the ON set nor carry a
  reviewed-table row (the inverse coherence of R9: a quarantined flag
  cannot silently return to the union).
- `selftest_conf_lint.sh`: checks 16-18 exercise R10 against the REAL
  linter (quarantined non-ON flag -> GREEN; quarantined ON-set flag ->
  RED [R10] "quarantine violated" + "appears in BOTH" diagnoses).

## Lift conditions (all three, from the verdict — expected vehicle: lane CG
`agent/census-rooting-fix`, in gating, rides pin 14)

1. The TU census roots externally-visible/entry symbols or fails closed
   (refuses on no-root), refusing the wave-8 repro pair's unrooted shape
   while preserving the three pin-13 fires.
2. The init-hoist ICE on non-trivially-rooted censuses is fixed.
3. The zero-trip soundness prose matches actual behavior.

Lifting = a reviewed commit that restores the ON-set lines, moves the
witness rows back to `_REVIEWED_FIRE_WITNESSES` (re-verified on the
union build per R9), restores the knob entries, and cites the pin-14
review record.

## Consequences for pre-registered pin-13 expectations

DEFERRED to pin 14 by this quarantine (their mechanisms are the
quarantined flags): minmax ~16.72 and where ~154.5 (init-hoist +
drain-backedge path), exp ~73.8 (crossloop-hoist), sigmoid-tree
crosscall parity.  STILL RIDE the 22-flag weekly sweep: reduce-sdpa
~832.75 (BV pricing split), BW's 11 un-withheld rows
(tanh/sdpafw/sigmoid/celu/elu/mish/selu/... first cells), CF's
log/sqrt/rsqrt (~+1.6-3%), welford 322, sdpa ~928, storm re-measures.

## Evidence

- Killed 25-flag run preserved at
  `~/sfpi-uplift/sweep-2x2/weekly-20260819-quarantined-25flag/` (moved,
  not resumed: its ON/knob cells were measured under the quarantine-
  violating 25-flag set; the relaunched sweep starts a fresh
  `weekly-20260819/` so no stale cell is reused or overwritten).
- Lane evidence: `~/sfpi-uplift/laneCQ-evidence-20260819/` (EVIDENCE.md
  + SHA256SUMS): lint/selftest transcripts, relaunch proof.
