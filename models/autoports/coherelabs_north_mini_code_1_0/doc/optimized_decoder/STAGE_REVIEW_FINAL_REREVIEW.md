# Stage Review

**Reviewed commit:** `8ce223940dea69cb3aa8ee0153bb44807014ee51`

**Verdict:** `more-work-needed`

The focused P2 timing-reproducibility finding from `STAGE_REVIEW_FINAL.md` is closed. The new comparison uses one decoder and device session, balances invocation order, records 50 samples for each label at both sequence lengths, and compares policies whose prefill program fields are identical. The independently recomputed means, medians, sample counts, and deltas agree with the reports. The earlier 5.30% sequence-33 mean gap does not reproduce.

No additional model-local required work was found in this commit or in the previously reviewed model-local implementation. The overall verdict remains strict because the unchanged batch-32 path still executes dense all-expert MoE and therefore does not satisfy the routed active-expert optimization contract. Its characterized remedy remains a shared-TTNN local combine or compact-output capability outside the authorized model-local write scope.

## Required Work

### P1 — Implement routed active-expert execution for batch 32

**Evidence**

- Commit `8ce223940de` changes the focused timing harness, distributions, and documentation; it does not change `tt/optimized_decoder.py` or shared TTNN.
- `tt/optimized_decoder.py:1431-1436` still restricts active prefill to batch 1 and sends the other workloads with at least 32 tokens through `_dense_expert_moe_chunk`.
- `tt/optimized_decoder.py:1285-1383` still broadcasts tokens to all experts and computes all expert gate/up/down projections before route weighting and reduction.
- `STAGE_REVIEW_FINAL.md` independently verified the shared-TTNN boundary: the fast `moe_compute(compute_only=True)` path exposes an incomplete rolling two-slot output, while the complete selective reduce-combine path requires fabric links.

**Why this matters**

The active optimization contract requires batch-32 gate-selected active-expert execution. A correct and fast dense all-128-expert fallback is not the required optimized algorithm. The user-imposed model-local write scope explains why the shared implementation is absent but does not waive the gate.

**Required next step**

Provide the shared-TTNN local-only combine or compact persistent output, integrate it into the model-local decoder, and produce authentic batch-32 decode/prefill correctness, warmed traced performance, trace-stability, profiler, and watcher evidence. Until that dependency or an explicit contract change is supplied, the stage remains `more-work-needed`.

## Closed Finding

### P2 — Final-prefill timing reproducibility

**Harness inspection**

- `_interleaved_prefill_s2` in `tests/optimized_decoder_perf.py:228-322` constructs both policies once, uses one decoder/session/input/cache set, performs five warmup pairs, then records 50 measured pairs.
- Pair order is balanced: retained-S2 runs first on even pairs and final-default runs first on odd pairs.
- The control is valid for this question. `retained_s2 = replace(final_policy, ...)` changes only decode `sparse_*` fields. All other fields remain equal, and the harness separately asserts equality of the 12 phase-specific prefill program fields.
- The measured grouped-prefill implementation consumes the phase-specific fields at `tt/optimized_decoder.py:1211-1266`; shared prefill fields such as grouped-mode, chunk size, dtype, fidelity, and memory placement are equal by construction.
- Timing surrounds each individual call and synchronizes the device before the sample is recorded. The JSON metadata records the final checked-in policy, five warmups, 50 iterations, batch 1, layer 1, and the intended sequence.

**Distribution inspection**

| Sequence | Retained-S2 mean / median | Final mean / median | Final delta |
|---:|---:|---:|---:|
| 33 | 4.147100 / 4.102691 ms | 4.127520 / 4.099534 ms | -0.47% / -0.08% |
| 128 | 13.733069 / 13.737201 ms | 13.757862 / 13.804296 ms | +0.18% / +0.49% |

- Both JSON files contain exactly 50 samples per label, and their stored mean, median, minimum, maximum, and percentage deltas recompute exactly from those samples.
- The labels each occupy the first and second position 25 times, so the result is not confounded by a fixed within-pair order.
- An independent paired calculation gives a mean final-minus-retained delta of -0.01958 ms at sequence 33 and +0.02479 ms at sequence 128. Approximate 95% paired intervals are `[-0.05360, +0.01444]` ms and `[-0.03511, +0.08469]` ms, respectively; both include zero.
- The interquartile and 10th–90th percentile ranges overlap strongly at each sequence length. There is no remaining evidence of a material final-default regression.

**Resolution:** Fixed/controlled. The evidence supports classifying the earlier difference as non-persistent process/session timing noise. It does not isolate host scheduling as the unique physical cause, so that phrase in the updated docs should be read as a practical attribution rather than a root-causal proof; this does not leave a material performance anomaly unresolved.

## Other Concerns

- No new model-local correctness or performance defect was found.
- The prior non-blocking concern remains that current review7 profiler evidence is layer-1-only. Layer-4 correctness exists, and the selected program topology is sufficiently tied to the measured path for the present contract, so this is not promoted to required work.
- `_sparse_program` still lacks the proposed early `K_tiles % in0_block_w` validation. Checked-in defaults are legal and covered, so this remains defensive follow-up rather than a current-path defect.

## Hard-Check Gaps

- The new interleaved branch has no dedicated static unit test. Source inspection, syntax parsing, artifact-schema checks, and exact distribution recomputation cover the focused claim; a new evidence format is not required for closure.
- Existing `human_report.txt` files remain command/status output rather than rendered operation tables. Their advice-enabled filtered CSVs provide the runtime-row evidence, so this remains an evidence-presentation gap.
- No new hardware correctness, profiler, or watcher run accompanies this timing-only commit. None is required to resolve P2 because the changed harness compares identical prefill programs and does not change production runtime code.

## Anomaly Ledger

### A1 — Separate-process sequence-33 final mean was 5.30% slower

- **Evidence:** Prior `current_s2_prefill33.json` and `final_default_prefill33.json`; new `interleaved_prefill33.json`.
- **Affected path:** Batch-1 phase-specific prefill performance reporting.
- **Control or comparison:** Alternating retained-S2 and final-default labels in one session with identical prefill program fields, five warmup pairs, and 50 measured pairs.
- **Likely subsystem:** Run-to-run host/session timing variance, not final-default program selection.
- **Investigation performed:** Full sample retention; recomputed mean/median/min/max; balanced-order inspection; parity/order split; percentile inspection; paired-delta analysis.
- **Resolution:** Fixed/controlled. Final mean/median are -0.47%/-0.08% relative to retained-S2, with no statistically material paired difference.

### A2 — Sequence-128 final/default comparison

- **Evidence:** `interleaved_prefill128.json`.
- **Affected path:** Batch-1 phase-specific prefill performance reporting.
- **Control or comparison:** Same harness and identical-program control as sequence 33.
- **Investigation performed:** The same distribution and paired checks.
- **Resolution:** Controlled. Final mean/median are +0.18%/+0.49%, with an approximate paired interval spanning zero.

### A3 — Batch-32 dense all-expert fallback

- **Evidence:** Unchanged runtime source, `ROUTED_MOE_HYPOTHESIS.md`, `AUTOFIX.md`, and the prior independent review.
- **Affected path:** Batch-32 decode and prefill MoE.
- **Control or comparison:** Batch-1 routed sparse execution; slower materialized sparse b32 candidates; fast but incomplete `moe_compute` output.
- **Likely subsystem:** Shared TTNN MoE output/combine capability.
- **Investigation performed:** Previously characterized model-local and shared-TTNN paths; this commit introduces no relevant implementation change.
- **Resolution:** More-work-needed; P1.

## Scope Inspected

- **Goal/skill paths:** Full `.agents/skills/stage-review/SKILL.md`; active optimized-decoder contract and prior `STAGE_REVIEW_FINAL.md`.
- **Artifact paths:** `candidates/sparse_subblocks/interleaved_prefill33.json`, `interleaved_prefill128.json`, and the prior separate-process prefill JSON controls.
- **Code paths:** The full commit diff, `tests/optimized_decoder_perf.py`, and the phase-specific grouped-prefill reads in `tt/optimized_decoder.py`.
- **Documentation:** `README.md`, `SPARSE_SUBBLOCK_HYPOTHESIS.md`, `AUTOFIX_CURRENT.md`, and `work_log.md`.
- **Commands run:** `git status/rev-parse/show/diff`, `sed`, `nl`, `rg`, `jq` schema/statistical recomputation, `git diff --check`, and a read-only Python AST parse.

No device test, profiler collection, watcher execution, reset/recovery action, server, or vLLM workflow was run during this rereview.

## Residual Risk

P2's evidence uses deterministic full-shape synthetic weights and activations, matching the earlier timing artifacts whose variance it investigates. Since both labels execute the same prefill program and are interleaved in one session, this is sufficient to rule out the claimed persistent final-default wiring regression. It does not make individual host-timed samples noise-free, so the newly reported distributions are more meaningful than any single minimum.

The dominant residual risk is unchanged: no performant compact batch-32 routed output exists in the reviewed implementation, and therefore neither correctness nor performance of the required b32 algorithm has been demonstrated. That shared dependency alone prevents a clean pass; no additional model-local required work was identified.
