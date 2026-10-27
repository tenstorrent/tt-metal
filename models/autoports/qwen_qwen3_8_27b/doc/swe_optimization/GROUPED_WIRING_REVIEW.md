# Stage Review

Verdict: more-work-needed

Independent follow-up, 2026-10-27 15:09 UTC, of optional grouped recurrence wiring and the full-stack state/continuation harness at Metal `4a72538f09f1bbf9fc10de7b8e93be02c39a226e`, branch `mvasiljevic/qwen38-swe-opt`. This supplements `GROUPED_SOURCE_REVIEW.md`; it does not sign off stage07 or serving. No confirmed source defect in the inspected routing or alternating continuation protocol prevents launching the planned paired full64 checks.

## Required Work

- **P1: Compare the pending full-stack state receipts across both arms before accepting equivalence.**
  Evidence: `tests/benchmark_grouped_model.py:56-66,160-204` correctly records bit-preserving per-chip hashes, but an arm's `status=pass` asserts execution/reset determinism, not baseline/grouped state or continuation equality. The existing B16 comparison contains token checks only; B32 tracking/state runs remain pending.
  Why this matters: matching sampled tokens can hide different FP32 state. Recording a hash does not test it against the control.
  Required next step: the owner plans a CPU comparator. Require identical workload/precision/chunk/context settings and exact receipt keys/values for every timed repeat, mixed reset receipt and both continuation stages, plus both continuation logits receipts. Each all64-layer recurrent-state map must contain384 entries:48 linear layers × conv/recurrent × four devices. Require passing arms, tracking1/program-cache-skip0 and unchanged traces/capture counts. Preserve the clean B16 timing separately. This is an acceptance/comparison gap, not evidence that the harness currently computes wrong state.

- **P2: Keep the remaining promotion gates explicit.**
  Evidence: the B16 clean full-model comparison is complete; final-source full64 B32 strict tracking/state evidence and reduced non-serving baseline/candidate profiles are not complete at this snapshot. Serving/cache reuse and matched scored grouped SWE are also pending.
  Why this matters: the measured generator gain does not establish serving throughput or SWE accuracy/runtime.
  Required next step: finish the current paired checks/profile, inspect all three qualitative outputs with a sufficient cap, then validate the final serving configuration and a matched scored pilot before promotion or SWE100/500 forecasts. Do not profile a live server or combine watcher with profiler. The already passing B32 layer watcher check need not be repeated solely to duplicate this review.

## Other Concerns

- `state_receipt()` hashes logical contiguous tensor bytes via a uint8 view, preserving FP32 low bits; it runs outside measured generation. It deliberately covers convolution/recurrent state, not full-attention KV, positions, page tables or sampler state. Name its scope accordingly; it is not a receipt for every cache/resource.
- The continuation sequence is position-consistent: reset, prefill33 tokens for all rows, extend even rows with seven101 tokens at positions33..39, then extend odd rows with seven102 tokens at those same positions while evens remain inactive. Both parity groups then logically end at40. `prefill_continuation()` sends these through seven teacher-forced masked decode steps, since all seven fit before the next128-token prefill boundary. This usefully exercises grouped masked decode on nonzero full-stack state. It does **not** exercise snapshot restoration, a cached hit, page remapping, or crossing from the bridge into eager chunked prefill; retain those as serving/reuse gates.
- The continuation harness records whole-state maps but no before/after per-row invariance assertion. Cross-arm comparison proves agreement with the split control. The native layer varying-mask checks independently establish inactive nonzero preservation. Do not describe this full-stack continuation alone as an independent proof that the inherited continuation implementation has correct cache semantics; an uninterrupted-prefix or per-row control would be needed for that stronger claim.
- Adapter logging counts layers carrying the policy flag, which can be64; only48 linear-attention layers execute `_delta`. Treat the log as configuration identity, not a count of grouped kernel executions.

## Hard-Check Gaps

- **Resolved earlier P1 nonzero masks and P2 three-heads/core:** inspected `decoder-native-b19-varying-tracked.json` and `decoder-native-b32-watcher10-noinline-tracked.json` under `/home/mvasiljev/build/qwen38-swe/grouped-recurrence/`. Both pass tracking1/program-cache-skip0, five masks (all-active, alternating, all-inactive, complement, all-active), exact output/conv/recurrent comparisons and24 inactive-preservation checks with nonzero previous state. Native B19 exercises automatic three-head selection through the production helper. Its smallest recorded inactive nonzero count is69118; B32's is122877. Primitive `b19-h3-tracked-final.json` also passes advancing-state equality.
- **Production routing inspected:** `tt/precision.py:79-83` validates the default-off environment flag during decoder construction. Setting it after imports in the runner therefore works. `Model` passes that policy to every layer, and `from_state_dict()` copies it into the batched decode view. `tt/multichip_decoder.py:692-700` chooses the grouped helper only for one-token, non-prefill decode above the original core limit and at mostB32; prefill and smaller batches retain the original route. In-place eligibility and inactive-state `where` remain unchanged. The test probe now selects the clean policy rather than generated source.
- **B16 claims independently checked against raw arrays:** both `full-b16-{baseline,grouped}-p256.json` files pass; batch16,4096 prompt,128 output, physical256, tested8192 cache and precision match. All three timed token arrays, all three shared prompt token arrays and mixed reset/ragged token arrays match exactly. Median decode improves from13.8087 to17.5110 tokens/s/user (1.2681×); TTFT remains about12.14s. These runs have tracking0. The code example is visibly cut off at the256-token cap in both arms, so its completion remains a controlled cap limitation, not a completed qualitative gate.
- Source identity remains scoped: the clean baseline used the prior baseline harness; the candidate uses the clean helper. Subsequent formatting changed hashes. The final strict acceptance runs/comparator must record current source identity; older token comparisons remain useful narrower controls.

## Anomaly Ledger

- Observed anomaly: inline watcher fabric startup exceeded ACTIVE_ETH program config capacity,29264 versus26624 bytes.
  Evidence: local `decoder-native-b32-watcher-tracked.log` and `decoder-native-b32-watcher10-tracked.log` fail before model construction; `decoder-native-b32-watcher10-noinline-tracked.log` records `disabled features: None`, passing model checks and normal watcher shutdown.
  Affected path: instrumented fabric startup.
  Control or comparison: watcher10 with function inlining disabled, all watcher features retained.
  Likely subsystem: instrumented kernel/config capacity, not recurrence arithmetic.
  Investigation performed: inspected the failure and passing logs plus the strict tracked JSON.
  Resolution: **controlled for layer safety validation**. Instrumented timings are not production performance evidence.

- Observed anomaly: the B16 candidate launcher returned127 after successful Python/device teardown because its script text changed while executing.
  Evidence: disclosed in `grouped-full-b16-comparison.json`; raw candidate JSON passes and includes all three completed records/qualitative/mixed results, which were checked directly.
  Affected path: shell launcher exit after the measured process.
  Control or comparison: raw Python outputs and complete result arrays.
  Likely subsystem: mutable launcher, outside model execution.
  Investigation performed: rederived token equality and compared workload fields from raw JSON; did not rerun hardware.
  Resolution: **controlled for those raw model results**. Keep launchers immutable for later acceptance runs; do not describe127 as a clean launcher exit.

## Scope Inspected

- Goal/skills: unchanged supplied stage07 grouped contract and the stage-review, optimize, tracing and target-workload requirements used by the first review.
- Code: optional helper and production precision/decoder/adapter diffs; `Model` construction; current `benchmark_grouped_model.py` and native decoder probe; generator reset, prefill, continuation, decode and token-out generation methods.
- Artifacts: local native B19/B32 tracked JSON, watcher startup/success logs, primitive B19 JSON, raw B16 baseline/candidate JSON and compact `grouped-full-b16-comparison.json`; shared prompt outputs inspected directly.
- Commands: read-only git/rg/nl/sed, UTC clock, host-only JSON/hash/array analysis. No TTNN imports, devices, models, servers, hardware tools, CI, resets or implementation edits. Only this compact Markdown report was written; no raw JSON/CSV/Tracy artifacts were copied.

## Residual Risk

The inspected optional wiring preserves the inherited numerical and ownership contracts, and the new native layer checks close the concrete earlier findings. Final-source full-stack state/trace acceptance, reduced profile interpretation, representative long-context cached continuation, serving trace reuse and scored workload effect remain open. Continue those checks; this verdict is not permission to mark the stage complete or terminal blocked.
