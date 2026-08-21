# REVIEW_RECORD — pin-16 UNION (pin cut 2026-08-21, lane EN)

Reviewer: lane EN (Claude, operated by nkapre@tenstorrent.com) — independent of lanes EB/EC/EG/EH/EI/EJ/EK/EL/DP (payload authors); union gates executed by lane EN (this record).

Candidate: sfpi-gcc `nkapre/sfpi` @ `927bce7e526` (the pin-16 union tip)
= `bbd6feec5d2` (nine post-pin-15 payloads:
EB `agent/dstautoincr-pricing` 2d5eb60af21, EC `agent/replay-record-hoist`,
EH `agent/prera-pressure-schedule`, EG `agent/delivery-shape-solver`,
DP df-notes fix f7dcf07e8b1, EJ `agent/reassoc-license` 88077d3b929,
EI `agent/round-chain-interleave`, EL `agent/cc-restore-loadi`,
EK `agent/store-folds`)
+ `e576b202c45` riscv.opt record-separator restore (pure union)
+ `927bce7e526` round-interleave census call-site rename fix (pure union).

This file is `REVIEW_RECORD-547fe7ffa113.md` in BOTH required locations
(tt-metal `tt_metal/tt-llk/tests/corpus/review_records/` AND
`~/sfpi-uplift/sweep-2x2/`).

## Reviewed

- MERGE FIDELITY (the three textually union-resolved merges, 3-way
  `git merge-file` reconstruction, laneEN-evidence-20260821/merge-audit/):
  - EL `b0d07ce51bc` and EK `bbd6feec5d2`: riscv.opt AND rvtt-protos.h both
    equal the clean 3-way auto-merge byte-exactly — FAITHFUL UNIONS.
  - EI `2048bee8499`: genuine 1-hunk conflicts in riscv.opt and
    rvtt-protos.h.  rvtt-protos.h resolution = marker-stripped union
    BYTE-EXACT (faithful).  riscv.opt resolution DROPPED the blank record
    separator both parents carried before mtt-tensix-optimize-round-interleave
    — the .opt parser swallowed the whole record; riscv_tt_opt_round_interleave
    never reached options.h and the tree DID NOT BUILD.  Second casualty of
    the same merge: EI's new census call sites in
    gimple-rvtt-replay-unroll.cc still called the pre-rename helper names
    (row_words_for/counted_trips) that EG's mainline had renamed to the
    exported rvtt_replay_unroll_* vocabulary (helper bodies proven identical
    modulo rename).  BOTH fixed as pure unions on the tip
    (`e576b202c45`, `927bce7e526`); whole-file separator scan found no
    sibling defect.
- ON-SET BEHAVIOR CHANGES WITHIN REVIEWED FLAGS (the PIN_REVIEW deltas):
  - EB dst-autoincr pricing (rides reviewed -mtt-tensix-optimize-dst-autoincr):
    58 adjudicated ON-25 rows (23 EQUALS-OFF reverts + 35 PARTIAL), CRAQ
    59/59 at EB's gate (laneEB-evidence-20260821).
  - EL cc-restore proof family (rides reviewed
    -mtt-tensix-optimize-invariant-loadi): 19 adjudicated ON-25 rows, CRAQ
    790/0 at EL's gate (laneEL-evidence-20260821); 6 rows overlap EB's set.
  - EK store-folds: ON-25 byte-INERT at its lane gate (flags are Init(0));
    its 269-row adjudicated delta is the store-fold+int-not KNOB leg
    (laneEK-evidence-20260821/CORPUS-GATES.md), not an ON-25 delta.
  - Expected pin-16 ON-25 delta vs the pin-15 base = EB ∪ EL = 71 rows
    (58 + 19 − 6 overlap), built row-for-row from the lanes' banked lists
    (laneEN-evidence-20260821/expected/).
- All seven new flags are Init(0)/default-off:
  -mtt-tensix-optimize-delivery-shape (+ -mtt-tensix-delivery-shape-min-benefit=),
  -mtt-tensix-optimize-replay-record-hoist,
  -mtt-tensix-optimize-pressure-schedule-prera,
  -mtt-tensix-optimize-round-interleave,
  -mtt-tensix-optimize-reassoc (licensed: fires only with -fassociative-math),
  -mtt-tensix-optimize-store-fold, -mtt-tensix-optimize-int-not.
  ON set UNCHANGED at 25.  ON-set promotion deferred to on-plus knob-leg
  silicon (owner order).
- COUPLED tt-metal merge now legal: agent/reassoc-license-conf 6bbc39ac8c
  (EJ licensed knob leg) merges at this pin — the pin-16 driver accepts
  -mtt-tensix-optimize-reassoc.

## Gates

(All at the union tip 927bce7e526, gcc-build-laneEN, stockcfg laneDW recipe +
ccache; evidence ~/sfpi-uplift/laneEN-evidence-20260821/ + SHA256SUMS.)

- BUILD binaries:
  - cc1plus sha256: `547fe7ffa11364a3d554bff051262f1ded223fb7cda0f9b9be7307d9333c2d0b`
  - driver xg++ (== install riscv-tt-elf-g++) sha256: `830d903a9fedc5d553df2e801f7b3fcce7861b57ad475dfdb7882623fd2c5e29`
  - xgcc sha256: `8c7ae6700fe41a7cf9c888b818d603d061528a774cd7736a6d5f7a305d99af07`
  - cc1 sha256: `754c82137d40f55427a0861aedc40f4accf8b7ef90caf34cdcc76d27c455463c`
  - lto1 sha256: `dd020897c7c67aeeb1496911010fd398cb90eecd4c31db56a6f7f4c416359951`
  - cpp sha256: `606127f276a7c1d94ee376a69ae4a55a0fc66ec2d8f785840ee591fdd3c7fc06`
  - install/ binaries verified identical to gcc/ (cmp).
- OPTCHECK: **GREEN** — all 14 Vars of the new families present in generated
  options.cc/options.h (delivery-shape + min-benefit param, record-hoist,
  prera, round-interleave, reassoc, store-fold, int-not, lreg-alloc,
  dst-layout-32b, list-schedule, pressure-schedule + use-milp,
  replay-hoist-min-benefit), all 13 flag spellings ACCEPTED by driver+cc1plus
  incl. -mtt-tensix-delivery-shape-min-benefit=60 (optcheck/optcheck.txt).
- DEJAGNU full rvtt.exp (pin-15 blessed recipe: SFPI env + pinned
  -B/-isystem): **GREEN** — 5309 PASS, 16-row FAIL set BYTE-IDENTICAL to the
  pin-15 frozen reference (frozen-9 + 7 documented sfpi-env rows; diff
  empty).  New-lane families all green: presched 38, delivery 69, reassoc 64,
  ccrestore 40, storefold 34, intnot 29, round-interleave 43, record-hoist 61,
  lregalloc 53, list-sched 52, milp 40, synth-renumber 28 — 0 FAIL,
  0 UNRESOLVED (dejagnu/).
- DS ACCEPTANCE GATE (tools/lreg_arsenal_gate.py --mode future, DP wrapper
  contract + --base-gxx pinned driver): **GREEN** — FUTURE 0 failed, 25 PASS,
  all 6 compile-noop rows noop-gate=byte-identical vs the installed pin-15
  driver (lreg-arsenal-gate-future.out).
- DT ARSENAL: **GREEN** — 31/31 list-sched-arsenal dg rows PASS at the union
  tip; makespan/RecMII oracle self-test battery (tt/tools/run-oracle-tests.py)
  all PASS.
- CORPUS BYTE-LEGS (shared farm, same-farm-path A/B: base legs recompiled at
  the current head with the PINNED pin-15 toolchain into a private store,
  mine legs with the laneEN hybrid driver; --flags= equals-form):
  - flags-OFF (22 -mno): **BYTE-IDENTICAL 3213/3213, exit 0** (corpus/gate-off.out)
  - TRUE-DEFAULT (empty flags): **BYTE-IDENTICAL 3213/3213, exit 0** (corpus/gate-truedef.out)
  - ON-25: **CHANGED exactly 71, MISSING 0, EXTRA 0 — row-for-row EQUAL to the expected EB∪EL set** (observed-not-in-EB∪EL = 0, expected-not-observed = 0, rows outside EB∪EL∪EK-allowed = 0; corpus/on25-*.txt).  EK contributed ZERO ON-25 rows, matching its lane gate (its 269 are knob-leg-only)
- CRAQ SPOT-CHECKS (5 random rows of the expected 71-row set, pinned BH sim
  32489dda4fd6): **GREEN 5/5** — fmod/mish/round/sigmoid_appx/abs sampled seeded-random from the 71-row set; each pytest node compiled EXACTLY the sampled variant sha under ON-25 with the union compiler and PASSED on the pinned sim (craq-spotcheck/)
- INSTALL: **DONE** — pin-install-fast gcc-build-laneEN -> ~/sfpi-uplift/sfpi/build/sfpi/compiler, --expect-cc1plus verified, --flags store-fold,int-not smoke-accepted; manual post-install smoke: ALL 13 new-flag spellings accepted on the installed driver incl. -mtt-tensix-delivery-shape-min-benefit=60 and the reassoc licensed triple (install-smoke.txt)
- witness_preflight on the INSTALLED binary: **ALL GREEN** — every declared witness of the 25-flag reviewed ON set fires on the union at the INSTALLED binary (witness_preflight.out)
