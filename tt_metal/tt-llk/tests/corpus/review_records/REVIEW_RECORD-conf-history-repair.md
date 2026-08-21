# REVIEW_RECORD — conf PIN HISTORY repair (audit-trail corruption, pin-17 ceremony)

Reviewer/author: AUDITOR SIDE (Claude, operated by nkapre@tenstorrent.com) —
the repair flagged as "in flight" by the wave-14 review; independent of the
pin-16/pin-17 ceremony lanes (EN, EQ) whose commits introduced/carried the
corruption.  Prose-only change: NO pin value, leg, knob, witness table, or
any other sourced conf state is touched (proof below).

## The corruption (wave-14 ledger 21, "NEW ISSUES")

HANDOFF wave-14 ledger item 21: "**pin-17's ceremony CORRUPTED the conf PIN
HISTORY** (pin-16 sha erased; entry 16 carries pin-17's sha with pin-16's
story; garbled half-sentence — 3rd instance of the narrative-below-lint
class; auditor repair in flight from in-repo ground truth)."

Concretely, in `tt_metal/tt-llk/tests/corpus/sweep_2x2.conf`, the pin-17
ceremony commit `abacfce46b` edited PIN HISTORY entry 16 IN PLACE instead of
appending an entry 17:

1. Pin-16's installed cc1plus
   `547fe7ffa11364a3d554bff051262f1ded223fb7cda0f9b9be7307d9333c2d0b` (and
   its driver `830d903a9fed…`) was ERASED from the history entirely — the
   only history that ever carried it was overwritten.
2. History entry "16." carried pin-17's cc1plus sha `ae7342e4fda3…` and
   pin-17's driver `8996f902c104…` over pin-16's provenance story ("the
   pin-16 pass-wave-2 union … sfpi-gcc 927bce7e5263…") — wrong: the
   ae7342e4fda3 build is from sfpi-gcc `75ace9d643e` = the 927bce7e526
   union + the lane EQ W_drain=7 crossing-charge fix (EP-F1).
3. The CURRENT-PIN paragraph carried a garbled half-sentence — "Supersedes
   pin 16, which was  This prose block and history entry 16 carry
   placeholders…" — the pin-17 rewrite grafted onto the stale pin-16
   pre-ceremony placeholder paragraph (itself already false after
   `07f16ea44e` substituted the shas), plus a leftover
   `REVIEW_RECORD-<cc1plus12>` placeholder in the retained pin-16 prose.

The corruption slid UNDER conf_lint.sh: R1/R2/R3/R5 check sha prefixes and
(CURRENT) placement, all of which agreed with the new pin values — the
narrative-below-lint class (ledger 17 "two narrative-level pin-lineage
falsehoods below conf-lint's anchor checks"; this is the 3rd instance).
The baseline TSV header (`sfpu_device_baseline_p150_v1.tsv` PIN LINEAGE,
entries 16/17) carried the CORRECT lineage the whole time — the conf's own
RULE ("pin values ↔ CURRENT PIN prose ↔ PIN HISTORY (CURRENT) entry ↔
baseline TSV header pin anchor" must agree) was violated at the narrative
level.

## Ground truth (all in-repo)

Every fact in the reconstruction traces to one of:

- `review_records/REVIEW_RECORD-547fe7ffa113.md` (pin-16, lane EN): union
  927bce7e526 = bbd6feec5d2 + two pure-union merge fixes (e576b202c45,
  927bce7e526); the EB…EL/DP/EK lane payloads; cc1plus
  `547fe7ffa11364a3d554bff051262f1ded223fb7cda0f9b9be7307d9333c2d0b`;
  driver xg++
  `830d903a9fedc5d553df2e801f7b3fcce7861b57ad475dfdb7882623fd2c5e29`;
  gates (16-row frozen FAIL set identical, corpus OFF/TRUE-DEFAULT
  byte-identical 3213/3213, ON-25 delta exactly EB58 ∪ EL19 = 71 rows,
  DS/DT arsenals green, CRAQ spot-checks 5/5 on the pinned sim,
  witness_preflight ALL GREEN on the installed binary).
- `review_records/REVIEW_RECORD-ae7342e4fda3.md` (pin-17): candidate
  sfpi-gcc `75ace9d643e` = pin-16 union 927bce7e526 + lane EQ W_drain
  crossing-charge correction (EP-F1); installed cc1plus
  `ae7342e4fda3d039e3c59a0237dae947c84cfb2bf31dbed2b59f35380bbbedd8`,
  driver `8996f902c104fa792f9adfe5ab6e5d4bd6821acf778a6b41425d8a10fe43d021`;
  gates (dg 330/330, rvtt.exp FAIL set 16 lines line-identical to the
  pin-16 frozen reference at 5339 PASS, ON-25 delta = exactly 51 rows =
  strict EB-58 subset, CRAQ 49/49 ops on the pinned sim, witnesses
  preserved).
- Ceremony commits `07f16ea44e` (pin-16: the pre-corruption conf state —
  entry 16 = 547fe7ffa113/830d903a9fed with the pin-16 story) and
  `abacfce46b` (pin-17: the in-place overwrite, visible in
  `git diff 07f16ea44e abacfce46b -- tt_metal/tt-llk/tests/corpus/sweep_2x2.conf`).
- The baseline TSV header PIN LINEAGE (`sfpu_device_baseline_p150_v1.tsv`
  lines "17. ae7342e4fda3… (CURRENT …)" and "16. 547fe7ffa113…
  (superseded by 17)"), which the conf must agree with per its own RULE.

## The repair (this commit)

In `sweep_2x2.conf`, prose (comment) lines only:

- PIN HISTORY entry 16 restored: cc1plus 547fe7ffa113… (full 64-hex,
  marked "superseded by 17"), driver 830d903a9fed…, pin-16's true story
  (union 927bce7e5263… payload list, ON set unchanged at 25, EB/EL
  adjudicated deltas, pin-16 ceremony facts), citing
  REVIEW_RECORD-547fe7ffa113.md inline.
- PIN HISTORY entry 17 appended as (CURRENT): cc1plus ae7342e4fda3…,
  driver 8996f902c104…, pin-17's true story (sfpi-gcc 75ace9d643e =
  pin-16 union + EQ W_drain=7 fix, gates), citing
  REVIEW_RECORD-ae7342e4fda3.md inline, plus a HISTORY-REPAIR NOTE
  recording what was corrupted and where the reconstruction traces.
- CURRENT-PIN paragraph: the garbled half-sentence and the stale
  placeholder block replaced by a complete, truthful "Supersedes pin 16
  (cc1plus 547fe7ffa113… / driver 830d903a9fed… — built from sfpi-gcc
  927bce7e5263…, NOT pin-17's 75ace9d643e source)" sentence citing both
  review records; the retained pin-16 build story is now labeled
  "SUPERSEDED PIN 16 PROSE (audit trail):" (the file's existing pin-15
  convention) and its `REVIEW_RECORD-<cc1plus12>` placeholder substituted
  with the real record name.

## Prose-only proof + gates (executed at this commit)

- `git diff` on the conf: 40 insertions, 13 deletions, EVERY changed line
  begins with `#` (verified:
  `git diff -- sweep_2x2.conf | grep -E '^[+-]' | grep -vE '^(\+\+\+|---)' | grep -vE '^[+-]#'`
  matches nothing).  `_REVIEWED_*` assignments, witness tables, knobs,
  legs, rows: byte-untouched.  Only `sweep_2x2.conf` (this record aside)
  changes in the commit.
- `conf_lint.sh` on the real conf+baseline: **GREEN** (R1–R10; pin values ↔
  conf prose ↔ PIN HISTORY (CURRENT) ↔ baseline header agree at cc1plus
  ae7342e4fda3…, driver 8996f902c104…).
- `selftest_conf_lint.sh`: **ALL GREEN** (every red fixture refuses,
  shipping state lints GREEN).
- Repaired history greps: entry 16 carries `547fe7ffa113…` (64-hex),
  entry 17 carries `ae7342e4fda3…` (64-hex, the only "(CURRENT)" entry,
  highest-numbered).

Limitation (disclosed): conf-lint's anchor checks still cannot catch this
narrative class mechanically — extending the linter beyond sha prefixes
(e.g., cross-checking history-entry driver shas against the superseded
record set) remains the open ledger-17 recommendation; this record is the
audit-trail closure for the pin-16/17 instance only.
