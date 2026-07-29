# Stage Review

**Reviewed commit:** `327a8ffac6307d3330e7fd3a0d1b2c965e94f6b9`

**Verdict:** `more-work-needed`

The sparse-subblock remediation is now credible: the sweep isolates each role against an explicit 1x1 control, the selected decode geometry reproduces independently, phase-specific prefill defaults avoid the measured decode-winner regression, exact sparse runtime rows are present in current profiles, and the final correctness and watcher suites are clean. This closes the sparse-evidence defect from `STAGE_REVIEW_REREVIEW.md`.

The stage still cannot clean-pass the active optimized-decoder contract because batch-32 decode and prefill execute all 128 experts through the dense fallback. The evidence also shows that the performant missing route requires a local combine or compact-output capability below model-local code. The current model-local write restriction explains why the implementation is absent, but it does not satisfy or waive the routed active-expert requirement.

## Required Work

### P1 — Implement routed active-expert execution for batch 32

**Evidence**

- `tt/optimized_decoder.py:106` leaves `dense_expert_batch_threshold=32` as the default.
- `tt/optimized_decoder.py:1431-1436` enables `active_prefill` only for batch 1 and sends any other workload with at least 32 tokens to `_dense_expert_moe_chunk`.
- `_dense_expert_moe_chunk` at `tt/optimized_decoder.py:1285-1383` broadcasts the input across `num_experts`, computes gate/up/down for every expert, then applies route weights and reduces. The b32 tests therefore validate a correct dense all-expert implementation, not routed active-expert execution.
- `ROUTED_MOE_HYPOTHESIS.md` and `AUTOFIX.md` provide a useful negative result: dynamic sparse candidates measured 20.535–21.896 ms, exact static sparse measured 17.831 ms, and packed static sparse measured 19.584 ms, versus 3.330 ms for the selected dense implementation.
- The promising `moe_compute(compute_only=True)` probe measured 1.642 ms, but its output is a rolling two-slot buffer rather than a complete result. This is consistent with shared TTNN source:
  - `ttnn/cpp/ttnn/operations/experimental/ccl/moe_compute/device/moe_compute_device_operation.cpp:283-327` defines the two-slot compute output and returns it directly in compute-only mode.
  - The full path at `moe_compute_device_operation.cpp:445-491` invokes selective reduce-combine.
  - `ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_device_operation.cpp:29-31` requires `num_links > 0`, and its program factory resolves fabric neighbors.
  - Ordinary sparse matmul preserves the dense batch/expert output surface, so it does not itself provide the required compact local result.

**Why this is required**

The optimize contract explicitly requires a non-Galaxy MoE default to preserve gate-selected active-expert execution and rejects dense all-expert execution as the optimized result. Correctness, good latency, and clean watcher evidence for the dense fallback do not substitute for that semantic requirement.

**Required next step**

Provide a shared-TTNN local-only combine path or a compact persistent-output mode for `moe_compute`, then integrate it into the model-local decoder. Revalidate authentic batch-32 decode and prefill at representative layers with PCC, warmed traced timing, trace replay stability, current profiler evidence, and a watcher-clean suite. Add a source/static guard that makes a dense all-expert default fail the optimized-stage contract.

Because the current task permits writes only under the model-local decoder/tests/docs area, this work cannot be completed within the present write scope. The stage remains `more-work-needed` until either the dependency and its evidence are supplied or the goal contract is explicitly changed; the scope conflict is not a clean-pass condition.

### P2 — Resolve the unexplained final-prefill timing discrepancy and report representative final-default latency

**Evidence**

- For sequence 33, the retained S2 policy measured mean/min 4.099386/4.078837 ms, while the final phase-specific default measured 4.316599/4.077707 ms. The minimum is effectively reproduced, but the mean is 5.30% slower.
- For sequence 128, the corresponding mean/min values are 13.598431/13.472163 ms and 13.755952/13.491884 ms.
- `SPARSE_SUBBLOCK_HYPOTHESIS.md`, `README.md`, and the work log highlight the final minima. The stored sample sets are visibly noisy, and there is no explanation for the material sequence-33 mean difference despite the intended prefill policy being retained.

**Why this is required**

The optimization contract requires the final checked-in default to reproduce the reported result and material slowdowns to be investigated or explained. Selecting the minimum from a noisy distribution does not establish representative warmed latency.

**Required next step**

Rerun the retained-S2 control and final-default prefill configurations in an interleaved, same-session comparison with enough warmed traced samples to characterize the distribution. Report at least mean and median (plus the sample count), use the final-default result in the final summary, and explain or fix any persistent material regression.

## Anomaly Ledger

### A1 — Batch-32 dense all-expert fallback

- **Evidence:** Direct source branch inspection; dense implementation inspection; b32 tests and final documentation; routed-MoE experiments and shared-TTNN implementation.
- **Affected path:** Batch-32 decode and prefill MoE.
- **Control:** Batch-1 decode uses routed sparse matmuls; batch-1 prefill uses grouped active experts.
- **Likely subsystem:** Shared TTNN MoE output/combine capability, followed by model-local integration.
- **Investigation:** Extensive candidate tests distinguish a correct but slow materialized sparse surface from a fast compute-only primitive whose output is incomplete. Full combine is fabric-only.
- **Resolution:** Unresolved; P1.

### A2 — Earlier sparse sweep inherited cumulative state

- **Evidence:** `candidates/sparse_subblocks/authentic_decode.json` now contains 15 isolated candidates with an explicit all-role 1x1 control. All rows report PCC 0.9993107115873155.
- **Affected path:** Batch-1 routed sparse gate/up/down selection.
- **Control:** Explicit control measured 0.781748 ms.
- **Investigation:** Independent per-role sweep, followed by four cumulative combinations.
- **Resolution:** Fixed. The cumulative winner (gate 2x2, up 2x1, down 4x4) measured 0.704813 ms and independently reproduced at 0.705102 ms.

### A3 — Decode winner regressed prefill

- **Evidence:** Applying the decode geometry globally increased the stored prefill timings; phase-specific defaults retain the earlier 2x2 prefill geometry.
- **Affected path:** Batch-1 non-aligned prefill.
- **Control:** Retained-S2 and global-decode-winner measurements at sequence 33 and 128.
- **Investigation:** Separate final-default prefill runs and exact current profiles.
- **Resolution:** The configuration defect is fixed and correctness is covered, but the remaining final-run distribution discrepancy is unresolved; P2.

### A4 — Runtime safety after sparse-default change

- **Evidence:** `review7_phase_specific_full.xml` records 41 passed/17 skipped with no failures or errors. `review7_phase_specific_watcher.xml` records the same result. The associated watcher log has a clean completion and no detected watcher failure signature.
- **Affected path:** Final optimized-decoder suite.
- **Control:** Normal and watcher-enabled runs.
- **Resolution:** Fixed/controlled for the implemented paths.

## Other Concerns and Hard-Check Gaps

- Current review7 profiles cover layer-1 decode, prefill-33, and prefill-128 only. The exact selected sparse rows are present and include advice, and layer-4 correctness is covered, but there is no current layer-4 profile to demonstrate the same runtime topology independently.
- Each `human_report.txt` under `tracy/review7_phase_specific` is command/status output from the CSV export rather than a rendered human-readable operation table. The advice-enabled `filtered.csv` files preserve the needed machine-readable evidence, so this is an evidence-presentation gap rather than a separate implementation blocker.
- `_sparse_program` validates output block/subblock legality but does not enforce the proposed `K_tiles % in0_block_w == 0` fail-fast condition. The checked-in defaults and tested candidates are legal, but future invalid overrides may fail later and less clearly.
- `git diff --check` is not clean because generated profiler CSV/text artifacts contain trailing whitespace/CRLF-style lines. No source-code whitespace defect was identified.

## Scope Inspected

- The complete `stage-review` and `optimize` skill contracts and the optimization guidance they reference.
- Commit metadata, changed-file scope, and the full model-local diff for `327a8ffac63`.
- `tt/optimized_decoder.py`, its model-local tests and perf harnesses, final README/work log, `STAGE_REVIEW_CURRENT.md`, `STAGE_REVIEW_REREVIEW.md`, `SPARSE_SUBBLOCK_HYPOTHESIS.md`, `ROUTED_MOE_HYPOTHESIS.md`, `AUTOFIX.md`, and `AUTOFIX_CURRENT.md`.
- Independent sparse JSON/XML artifacts, phase-specific correctness XML, final full/watcher XML and watcher log, and all three review7 profiler directories including raw/filtered summaries and advice.
- Relevant shared sparse-matmul, `moe_compute`, and selective-reduce-combine source, read-only, to verify the claimed dependency boundary.
- Static-only validation: `test_optimized_decoder_prefill_geometry.py` passed 9/9. Its process emitted nanobind shutdown leak diagnostics after the successful pytest result; no test failed.

No device test, profiler collection, watcher run, reset/recovery action, server, or vLLM workflow was executed during this independent review.

## Residual Risk

The batch-1 sparse result has strong candidate isolation, independent reproduction, correctness, profile, and watcher evidence. The selected batch-32 implementation is also well tested as a dense fallback. The dominant residual risk is therefore not an unknown correctness defect: it is the known mismatch between the active stage contract and the batch-32 runtime algorithm.

The shared-TTNN dependency appears real and narrowly characterized, but no compact local combine implementation or end-to-end b32 routed evidence exists yet. Until it does, performance and correctness of the required batch-32 algorithm remain unproven. The smaller prefill timing reproducibility issue also weakens the precision of the final performance claim, but it does not change the primary verdict.
