# Phase Review

Review round 12: `optimization-loop`, iterations 38-40 (commits `76f2b996f76`, `50fa4eb2737`, `1956d891df7`; HEAD
`1956d891df7`), 2026-11-02. Reviewer: fresh subagent, read-only (no device, no runs; artefact-analysis scripts only).
Previous round: `review_round11.md` (iterations 28-37). Paths: `O/` = `doc/optimization/evidence/`.

Verdict: more-work-needed

No P1. The three kernels (`skinny_contract` multicast + column ranges, `skinny_matmul` transposed mode, the SwiGLU
transposed modes), the autograd contract, the harness cases and the chain artefacts re-derive clean: the MFU rows, the
chip-0 kernel sums with and without the collectives, the per-chip clocks, the harness counts, the gradient check and the
memory probe all match their raw files. The required work is contract-mandatory documentation and evidence hygiene: a
loss-match cell contradicted by its artefact, a ranked cost table that mixes iteration-37 headers with iteration-39/40
rows, a "clock mix" attribution the slowest chip's own rows contradict, an iteration-38 net figure that does not add up,
a missing evidence file, an op-test file that was stale for fifteen iterations and whose rerun has no log, and a
three-change commit.

## Required Work

- P2: Iteration 40's loss-match cell contradicts its artefact
  Evidence: `O/iter40_loss_comparison.txt` and `O/iter40.log`: `max |diff| 0.0285 (tol 0.1), mean +0.0048`; the
  host-fp32 trajectory moved vs 39's at every step from 2 on (e.g. step 3 11.3190 -> 11.3175, step 9 8.9646 -> 8.9635).
  `perf_analysis.md` Iteration 40 Verification says "the single-chip loss match (PASS 0.0301) runs the qkv / out dB
  routing change"; `work_log.md` Iteration 40 says "loss match PASS (0.0301, identical)".
  Why this matters: the cell quotes a number from the previous iteration while stating that this iteration's change
  is exercised; the trajectory did change (the dB routes qkv / out from ttnn's transposed-a matmul + transpose to the
  all-core op, a different accumulation order), which is exactly what the loss match exists to show. The Ledger
  generalisation (round 11) that the loss match "never runs the TP-shard-keyed changes" does not cover this one.
  Required next step: correct both cells to 0.0285 / mean +0.0048, and add a Ledger bullet: the first single-chip
  loss-match change since iteration 32, caused by the dB route, within tolerance.

- P2: The ranked cost table is a mixture of iteration 37 and iteration 39/40
  Evidence: `perf_analysis.md` "Ranked cost table": source line "TP=4 after iteration 37, chip 0 of
  `O/iter37_tp_ranked_ops_full.txt` ... 46.5 ms of kernels ... 8.9 collectives", column header "rows (ms per 4 layers,
  chip 0, iteration 37)"; but the matmul row carries 39's numbers (gate_up fwd 5.42 / dX^T 5.38: 37's are 5.84 / 5.52),
  the SwiGLU row 40's (bwd 0.80 + fwd 0.45: 37's 0.74 + 0.45), the LoRA row 40's (gate_up dB 0.38, w2 dA 0.20 + dB
  0.13 + h^T 0.29 + 0.04), while the layout row still lists "`TransposeSplit` 0.40 (w2 dX^T)", an op gone since
  iteration 38 (`O/iter40_tp_ranked_ops_full.txt` has only the gate_up `TransposeSplit` 0.248).
  Why this matters: the table is the loop's ranking instrument; a reader cannot tell which profile a row comes from,
  and the kernel total (46.5 / 8.9) belongs to a profile three iterations old (iteration 40: 42.90 kernels, 6.10
  collectives on chip 0; chip 2: 42.16 / 5.71).
  Required next step: regenerate every row and the source line from `O/iter40_tp_ranked_ops_full.txt` (chip 0, with
  chip 2 noted as the profile's critical chip), drop the w2 `TransposeSplit`, keep the "remaining rung" notes.

- P2: "The collectives +0.36 with this run's clock mix" (iteration 40) is contradicted by the slowest chip's rows
  Evidence: re-derived from `O/iter39_tp_ranked_ops_full.txt` and `O/iter40_tp_ranked_ops_full.txt`, chip 2 (the
  critical chip of both profiles, 1.123 / 1.131 GHz): collectives 4.883 -> 5.714 ms per 4 layers (+0.83;
  `ReduceScatterMinimalAsync [1024, 5120]` 2.032 -> 2.643, `AllBroadcast` 0.092 -> 0.241); `O/iter40_tp_analysis.txt`
  per-chip collectives 6.8 / 5.9 / 5.9 / 6.3 against 39's 5.8 / 5.8 / 4.7 / 4.7 (minimum over chips 4.7 -> 5.9).
  Why this matters: the slowest chip waits least, so a rise on it is not the other chips' wait; the cycle-time result
  (-0.46 without collectives) is quoted as if the collective rows were pure skew. For iteration 39 the explanation
  holds (38b's per-chip minimum 4.5 -> 39's 4.7, chip 0 became a fast chip); for 40 it does not. The MFU run (a
  separate, cooled run) improved 460.9 -> 458.3 and the best step 456.6 -> 452.8, so the likely cause is profile-run
  collective variance (iteration 27's Ledger entry: the reduce-scatter row swings between profiles), not the change,
  but that is a hypothesis the docs do not state.
  Required next step: in the Iteration 40 Result / Ledger cells replace "the clock mix" with the per-chip collective
  minima (39: 4.7, 40: 5.9) and classify the anomaly (Anomaly Ledger below); a repeat 4-layer profile is the control if
  one is cheap, otherwise state "controlled by the MFU result".

- P2: Iteration 38's "net -0.02 ms per 4 layers" is reproducible neither from the named rows nor from the tables
  Evidence: row-level chip-0 diff 37 -> 38b (`O/iter37_tp_ranked_ops_full.txt` vs `O/iter38b_tp_ranked_ops_full.txt`):
  the named rows (`TransposeSplit` -0.396, `SwigluPackedBw` +0.060, `SwigluPackedFw` +0.039, h^T 0.257 -> 0.338, dA
  0.193 -> 0.349, dB 0.140 -> 0.226) sum to +0.03 (+0.06 with the new `FillPad` 0.023 and `PadCodegen` 0.011); the
  full non-collective total is 37.635 -> 37.553 = -0.08 (norm forward rows -0.03, embeddings -0.02, `Concat` -0.01,
  `LowrankUpdate` -0.01 drift). Iteration 39 and 40 quote the full-table figure (-0.302, -0.456); iteration 38 does not.
  Why this matters: the loop's attribution convention (round 11 answer 1) is the full chip-0 table; "neutral" holds
  either way, but the number in three places (perf_analysis Result and Attribution, README row 38, commit message) is
  neither derivation.
  Required next step: state -0.08 (full table) with the named rows summing to +0.03 and the rest as row drift; the
  README row and the work log follow.

- P2: Stale references and cells
  Evidence: `perf_analysis.md` Iteration 40 Evidence cites `O/iter40_db_routes.log (39's)`: the file does not exist
  (the route table is `O/iter39_db_routes.log`); `work_log.md` Commits table row 40 reads "(this commit)" while
  HEAD is `1956d891df7`; README iteration rows are ordered 38, 40, 39; Iteration 39 Result says the real-weight
  losses are "within 0.001 of 38b's" while `O/iter39.log` vs `O/iter38b.log` differ by 0.002 at steps 6-8 (0.3418 ->
  0.3398, 0.2617 -> 0.2637, 0.3086 -> 0.3066; one bf16 ulp at that magnitude).
  Why this matters: cited evidence must exist and bisecting needs the SHA; the tolerance claim is wrong by a factor of
  two even if harmless.
  Required next step: fix the four cells.

- P2: The SwiGLU op-test file was stale since iteration 24 and its iteration-39 rerun has no artefact
  Evidence: commit `50fa4eb2737` changes `tests/python/test_swiglu_packed.py`
  `test_rejects_upstream_grad_that_is_not_half_the_packed_width` from `expect_error(RuntimeError, "halved")` to
  `">= I"` (the message iteration 24 introduced), i.e. that test has failed, or not been run, for fifteen iterations;
  `work_log.md` Iteration 39 records it "fixed in passing". Iteration 39's Verification cell claims "the SwiGLU op
  tests `TestTransposedIO` (9, forward bit-exact ...)" but no `O/*` log of a `test_swiglu_packed.py` run exists
  (`grep -l test_swiglu_packed O/*.log` finds none; `O/iter39_swiglu_diag.log` is a 4-case script, not the test file).
  The chain (`O/iter39.sh`, `O/iter40.sh`) runs only `test_mistral.py`.
  Why this matters: the owned SwiGLU kernels gained four modes in iterations 38-39; their op-level tests are the
  only place the validation paths and the small shapes are exercised, and the evidence for "9 passed" is prose.
  Required next step: run `tests/python/test_swiglu_packed.py`, commit its log under `O/` and cite it from the
  Iteration 39 Verification cell; add the file to the chain script (or state in the README conventions why the op
  tests run outside the chain).

- P2: Commit `1956d891df7` (iteration 40) bundles three independently measurable kernel changes
  Evidence: the commit changes `skinny_contract` (a) small-operand multicast, (b) per-core column ranges, and
  `skinny_matmul` (c) the transposed-x / transposed-output mode plus the routing. (a) and (b) were measured only
  together (`O/iter40_contract_after.log`: 123 -> 101, 68 -> 53, 49 -> 35 us); the per-change split in the docs
  ("the small read 15-18 us", "imbalance ~20") comes from the removal attribution (`O/iter40_contract_attrib.log`:
  `ATTRIB_SKIP_SMALL` -18 / -15 / -15) and an estimate, not from an A/B of each. (c) has its own A/B
  (`O/iter40_ht_routes.log`).
  Why this matters: the contract's one-change-per-commit rule; the history does not bisect between (a), (b) and (c).
  Review round 10 treated iteration 26's three-in-one the same way: recorded, not re-split.
  Required next step: record in the Commits table and the Ledger (as for iteration 26) that iteration 40 is three
  changes in one commit with (a)+(b) measured jointly; no re-split required.

## Other Concerns

- The out-projection dB routed to the op is slower in device time: 0.138 vs ttnn's 0.119 + 0.010 transpose
  (`O/iter40_cycle_diff.txt`; +0.009 ms per 4 layers), justified as "a launch saved" (Ledger) while the loop's own
  finding is that the host/dispatch gap is ~0 (device busy 99.8 %). The skill rejects fewer ops as evidence. Cost is
  0.02 % of the step, so not required work, but either show the wall benefit or route by the measured best path
  (`SKINNY_CONTRACT_MAX_N` back to 4096 keeps qkv on the op, where it does win: 0.089 -> 0.065).
- Iteration 39's `SwigluPackedFw` standalone +6 us (129 -> 135, `O/iter39_swiglu_time.log`, three placements tried,
  "cause not found") has the opposite sign in the step (0.489 -> 0.446 ms per 4 layers = -11 us per call,
  `O/iter39_cycle_diff.txt`). The row is 0.45 ms per 4 layers, so the open question is worth ~0.06 ms per 4 layers;
  acceptable as recorded, but it belongs in the Ledger as a standalone / in-step disagreement, not only in the
  Attribution cell.
- `reader_skinny_contract.cpp:37`: `small_acc` is set but unused on the receiver cores (warning in every harness and
  gradient-check log); move the accessor under `MCAST_SENDER`.
- `modules/lora.py` `skinny_contract` docstring still describes the iteration-18 routing ("wide dB ... stays on
  ttnn's transposed-a matmul ... the narrow dB goes to the all-core op") while the code routes every N < 8192 and every
  transposed operand to the op.
- `O/iter39_cycle_diff.txt` ends with a bogus "chip 2: total 0.000 -> 41.812 (+41.812)" line (the 38b table has no
  chip-2 section); harmless, but a reader may take 41.8 as a delta.
- Memory: the `W_gu_cat^T` copies take 6.33 GiB per chip for -0.30 ms per 4 layers (~-3 ms/step, 0.65 %); 27.06 of
  31.83 GiB after the forward (85 %, plan §4 updated). Within the recipe (seq 1024, batch 1) this is fine, but it
  forecloses longer sequences on this box; the trade is recorded in the ranked table but not flagged as a decision for
  the user.
- The `skinny_contract` Python wrapper's transposed branch skips the Python-side guards (`T <= 4096`, `r <= 16`, one
  image) and relies on the device op's `TT_FATAL`s; fine for the model's shapes, worth one line in the docstring.

## Hard-Check Gaps

- `skinny_matmul(x_transposed=False, transpose_out=True)` (the reduce kernel's `transpose_dest` path, reachable from
  `lora.skinny_matmul(..., transpose_out=True)` on a token-major `a`) has no harness case; the model does not use it.
  Either test it or make the validation reject `transpose_out` without `x_transposed`.
- The multicast handshake was re-derived from the code only: sender (0, 0) excluded from its own rectangle
  (`noc_async_write_multicast` without loopback, `num_receivers = grid - 1`), writes flushed before the VALID
  multicast on Blackhole, every core of the rectangle (idle ones included) reserves `cb_small` at the same L1 address
  and counts in before the data is sent, flags reset per launch. The 18 parametrised cases and the model's program
  cache hits exercise repeated launches; a device test of two back-to-back calls with different small-operand
  addresses in one program-cache entry would close the remaining gap.
- The `TestTransposedIO` run log (P2 above).

## Anomaly Ledger

- Observed anomaly: iteration 40's profile shows higher collective rows on every chip, including the slowest.
  Evidence: chip 2 collectives 4.883 -> 5.714 ms per 4 layers (39 -> 40), chip 0 5.744 -> 6.100; per-chip minimum
  4.7 -> 5.9 (`O/iter40_tp_analysis.txt`); clocks 1.169 / 1.139 / 1.131 / 1.148 GHz (tighter than 39's).
  Affected path: TP all-reduce (reduce-scatter `[1024, 5120]`, the loss broadcast).
  Control or comparison: the cooled MFU run improved 460.9 -> 458.3 (best 456.6 -> 452.8); iteration 27's profile had
  a 4x reduce-scatter swing with no code change.
  Likely subsystem: run-to-run collective variance of the 4-layer profile (chip skew at the first collective of the
  backward), not the change.
  Investigation performed: none in the docs (attributed to "the clock mix"); reviewer's per-chip re-derivation above.
  Resolution: more-work-needed (state the per-chip minima; control by the MFU result or a repeat profile).
- Observed anomaly: the single-chip loss match moved for the first time since iteration 32.
  Evidence: `O/iter40_loss_comparison.txt` max 0.0285 / mean +0.0048 vs 0.0301 / +0.0052 in 32-39.
  Affected path: qkv / out dB on the all-core op (`SKINNY_CONTRACT_MAX_N` 4096 -> 8192).
  Control or comparison: harness `test_skinny_contract_matches_torch` 18 cases PCC >= 0.9999, rel <= 1e-2 vs fp64;
  TP gradient check PASS (PCC 0.99989-0.99992, rel 1.3-1.65e-2, as in 39).
  Likely subsystem: fp32 accumulation order of the op vs ttnn's matmul (bf16-ulp effects on the host loss).
  Investigation performed: none (the docs quote the old number).
  Resolution: more-work-needed (correct the cells; no code action).
- Observed anomaly: `SwigluPackedFw` +6 us standalone, -11 us per call in the step (iteration 39).
  Evidence: `O/iter39_swiglu_time.log` (129 -> 135 / 137, three placements), `O/iter39_cycle_diff.txt` 0.489 -> 0.446.
  Affected path: SwiGLU forward on `[2I, T]`.
  Control or comparison: ABBA alternation in one process; column-major blocks 135.
  Likely subsystem: DRAM page order of the row layout vs the profile's neighbourhood (the iteration-24 Ledger entry
  shows the same kind of in-step-only change for the backward).
  Investigation performed: three standalone variants; cause not found.
  Resolution: controlled (0.06 ms per 4 layers; recorded in the Attribution cell; add to the Ledger).
- Observed anomaly: iteration 38's net figure (-0.02) matches neither derivation (+0.03 named rows, -0.08 full table).
  Evidence: row-level diff of `O/iter37_tp_ranked_ops_full.txt` vs `O/iter38b_tp_ranked_ops_full.txt`.
  Affected path: documentation only.
  Control or comparison: iterations 39 / 40 use the full-table figure.
  Likely subsystem: hand arithmetic.
  Investigation performed: reviewer's script.
  Resolution: more-work-needed (P2 above).
- Observed anomaly: `test_swiglu_packed.py` carried a failing expectation from iteration 24 to 39.
  Evidence: the diff in `50fa4eb2737`; no op-test log in `O/` for any iteration since 24.
  Affected path: the SwiGLU op tests (not the harness).
  Control or comparison: the harness's MLP cases cover the model's shapes; the op tests cover the validation paths.
  Likely subsystem: the chain runs `test_mistral.py` only.
  Investigation performed: none recorded.
  Resolution: more-work-needed (P2 above).

## Scope Inspected

- Goal/skill paths: `skills/plugins/tt-train-bringup/skills/optimization-loop/SKILL.md`,
  `skills/plugins/tt-train-bringup/skills/stage-review/SKILL.md`; the contract in the review prompt; `doc/plan.md`
  §4 (memory row updated to 27.06 / 21.17 GiB with the two copy families).
- Artifact paths: `doc/perf_analysis.md` (MFU history rows 38-40, Ranked cost table, Iterations 37-40, Rejected,
  Ledger); `doc/optimization/README.md` (rows 38-40, history and lessons lines); `doc/optimization/work_log.md`
  (Iterations 38-40 checklists, review-11 answers, Commits table); `O/iter38.{sh,log}`, `O/iter38b.log`,
  `O/iter39.{sh,log}`, `O/iter40.{sh,log}`; `O/iter{37,38b,39,40}_tp_ranked_ops_full.txt`,
  `O/iter{38,38b,39,40}_tp_analysis.txt`, `O/iter{39,40}_cycle_diff.txt`, `O/iter{38b,39,40}_tp_mfu_tp4.{log,json}`,
  `O/iter{38,38b,39,40}_harness.log`, `O/iter38_harness_subset.log`, `O/iter{39,40}_tp_grad_check_tp4.log`,
  `O/iter{39,40}_loss_comparison.txt`, `O/iter{38b,39,40}_full_tp4_real.log`, `O/iter38_mem_probe.log`;
  standalone: `O/iter38_swiglu_bw_time.log`, `O/iter38_contract_check.log`, `O/iter38_lora_products.log`,
  `O/iter38_scoping.log`, `O/iter39_scoping.log`, `O/iter39_swiglu_time.log`, `O/iter39_swiglu_diag.log`,
  `O/iter39_db_routes.log`, `O/iter40_contract_attrib.log`, `O/iter40_contract_after.log`, `O/iter40_ht_routes.log`,
  `O/iter40_lora_products.log`.
- Code paths: `metal/ops/skinny_contract/{skinny_contract.cpp, device/skinny_contract_device_operation.cpp,
  device/skinny_contract_program_factory.cpp, device/kernels/dataflow/reader_skinny_contract.cpp,
  device/kernels/dataflow/writer_skinny_contract.cpp, device/kernels/compute/skinny_contract_kernel.cpp}`;
  `metal/ops/skinny_matmul/{skinny_matmul.cpp, device/skinny_matmul_device_operation.cpp,
  device/skinny_matmul_program_factory.cpp (transposed parts), device/kernels/*}`;
  `metal/ops/swiglu_packed_bw/device/kernels/compute/swiglu_packed_bw_kernel.cpp`,
  `metal/ops/swiglu_packed_fw/device/swiglu_packed_fw_device_operation.cpp` (validation),
  `metal/ops/common/swiglu_packed_common.hpp` (block size <= 4); `ttml/modules/lora.py` (routing),
  `ttml/modules/fused_lora.py` (wide_t forward / backward, the flags); `tests/python/test_mistral.py`
  (`test_skinny_contract_matches_torch`, `test_skinny_matmul_on_the_transposed_wide_buffer`);
  `tests/python/test_swiglu_packed.py` (the commit-39 diff).
- Commands run: `git log`, `git status`, `git show --stat` of the three commits, `git show 50fa4eb2737 --
  tests/python/test_swiglu_packed.py`; python3 scripts over the ranked tables (chip sums with / without collectives,
  row-level 37 -> 38b diff), the MFU JSON means (38b 462.64, 39 460.94, 40 458.31 ms over steps 3-8; 40:
  96.371e12 / (0.45831 x 594e12) = 35.40 %), PASSED-id diffs of the harness logs (49 - 5 + 18 + 2 = 64), greps of the
  logs for clocks, memory, warnings, loss rows. No device opened, no training or profiling run, no implementation
  file modified.

## Residual Risk

- The attribution rests on two-step 4-layer profiles whose collective rows vary run to run by more than the
  iterations' gains; the cooled MFU run remains the only result, and its run-to-run spread (steps 452.8-462.6 within
  one run) is of the order of the per-iteration deltas (-2.6 ms). Iterations 38-40 together: 460.5 -> 458.3 ms wall,
  -0.84 ms per 4 layers chip-0 kernels without collectives (37.635 -> 36.795), 35.23 -> 35.40 % MFU.
- 27.06 GiB per chip (85 %) after the forward leaves 4.8 GiB; any recipe change (seq 2048, batch 2) needs the copies
  revisited.
- The `skinny_contract` multicast assumes the compute grid is one rectangle with (0, 0) inside it (true on this
  board); another grid shape would hang in the handshake rather than fail validation.
- The step's remaining structure is unchanged: ~192 ms/step of ttnn matmuls at 82-88 % of peak (a kernel limit per
  iteration 34), 40-90 ms of collectives not overlappable on one sub-device (iteration 33); the 40 % target (405 ms)
  is still ~53 ms away and the loop's own ceiling statement (~36-37 %) stands.
