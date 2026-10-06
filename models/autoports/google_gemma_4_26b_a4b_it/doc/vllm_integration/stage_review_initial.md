# Stage Review

Verdict: more-work-needed

Independent first review, 2026-09-27. Stage 09 vLLM integration for
`google/gemma-4-26B-A4B-it`. This inspects a live worktree: tt-metal branch
`gemma-4-26b-a4b-it`, base HEAD `eeb18d817be12d288c3c437dd593a468e9aa02ae`;
vLLM branch `gemma4-vllm-integration`, base HEAD
`5ffebf4128f81ea5cf8413175eabde52cd8c8d75`. The first full sampling run finished
during review. Later repairs and artifacts require a subsequent review.

## Required Work

- **P1: Exclude padding and stale request tokens from prefill repetition penalties.**
  Evidence: `tt/generator_vllm.py:172` passes the complete padded token matrix to
  `configure_sampling(prompt_tokens=tokens)`. The plugin constructs that matrix
  with a common maximum width in `model_runner.py:1148`; `InputBatch.add_request`
  overwrites only valid prefixes (`input_batch.py:323-331`). A shorter row's tail
  can therefore contain zeros or token IDs from an earlier request. The common
  sampler counts every in-vocabulary nonnegative ID when building its prompt
  repetition mask (`models/common/sampling/tt_penalties.py:218-222,258-292`).
  Why this matters: the first generated token can be penalized for words that
  were never in this request, including words left by another request. This is
  a source-level correctness defect, independent of stochastic reproducibility.
  Required next step: supply a separate sampler prompt-history tensor masked to
  `-1` beyond each logical `prompt_lens`, preserving the actual model token
  input. Cover heterogeneous prompt lengths and deliberately populated stale
  tails with a regression that checks the resulting prompt mask or penalized
  token behavior.

- **P1: Resolve the full sampling gate's real failures before classifying residual reproducibility differences.**
  Evidence: `readiness_vllm/sampling_tests.log` ends with **17 failed, 55 passed,
  1 skipped** in 940.66 seconds. Thirteen failures are HTTP 400 rejections of
  temperatures 5, 10, or 50; the allowlist test rejects correctly empty special
  token text; `test_mixed_params_batch` differs across permutations. Two
  presence-penalty tests also fail: every request repeats ` a b c` for penalties
  from -1.5 through 2.0, including the mixed 0/2 case
  (`sampling_tests.log:2336-2425`).
  New evidence received during review: `presence_control.json` reproduces all
  40 original-prompt tokens identically with vLLM's CPU sampler at both 0 and 2;
  `presence_prompt_control.json` demonstrates a changed continuation at 0/2 for
  `The story begins: `, with device/CPU token equality at each penalty. This
  controls the original presence-test expectation without weakening its
  variation assertion. The repaired allowlist case also passes live
  (`allowed_token_ids_live_retry.log`, 1 passed in 5.67 seconds).
  Why this matters: these behavioral failures cannot be waived using the
  reproducibility exception. Their controls explain the causes, but final
  corrected-suite validation remains outstanding.
  Required next step: retain the original failure log, rerun corrected failing
  items, and complete the final full profile, including the controlled
  presence-test prompt repair. Keep any final
  reproducibility-only findings separate and supported by passed correctness,
  logprobs, stability, and qualitative checks.

- **P2: Complete the numerical standalone/direct-adapter counterpart to the live evidence.**
  Evidence: `LOGIT_DETERMINISM_CHECK.md` and
  `tests/check_vllm_logit_determinism.py` explicitly say their pending checker
  uses the direct adapter and excludes HTTP admission/scheduler behavior.
  New `logprobs_full_server.json` compares two repeated isolated prompts and
  both AB/BA concurrent submission orders. Independent inspection confirms
  exact equality of the token and top-20 logprob dictionaries in all eight
  responses. Each response contains one token: this proves the exercised live
  last-prefill result, not later decode-step numerical equality.
  Why this matters: the skill requires both standalone and vLLM numerical
  evidence when sampling determinism fails. Direct adapter equality cannot
  exclude scheduler-dependent changes in the live path; the new live artifact
  now supplies that complementary evidence for the measured prefix.
  Required next step: run the prepared full-model standalone/adapter check and
  link its numerical comparisons to the live first-token evidence. Document
  both scopes accurately; do not describe a one-token live response as a
  multi-step decode-logit check.

- **P1: Close the active-trace allocation warning with the planned lifetime investigation.**
  Evidence: `readiness_vllm/server.log:156` warns that buffers allocated behind
  active traces may be corrupted by replay. `AUTODEBUG_trace_allocations.md`
  gives a plausible second-capture scratch explanation but explicitly leaves
  runtime lifetime safety unproven. No tracker result existed in this snapshot.
  Why this matters: this warning touches the canonical model/sampler trace
  contract. Successful output tests alone do not classify a surviving unsafe
  allocation.
  Required next step: complete the already planned bounded functional tracker
  check for both model and sampler traces, inspect any surviving allocation,
  and record the actual outcome. No serving profiler, Tracy, or perf-report is
  requested or permitted.

- **P2: Control the malformed qualitative continuations and rerun the final implementation.**
  Evidence: `vllm_qualitative_outputs.json` contains `shared_3` greedy
  “In any own-contained system”; `shared_2` sampled contains “a pair-o-glasses”
  and “brass-bound-and-etched”, while its greedy output contains
  “tiny-brass-heart”. The current verdict treats creative wording as awkward,
  but does not investigate these exact anomalies. The saved selected-policy
  greedy controls stop at 128 tokens, before the malformed thermodynamics
  phrase and later greedy story phrase. The shared qualitative file predates
  the final trimming validation and subsequent source repairs.
  Why this matters: the stage-review anomaly rule requires a control or fix
  before dismissing visible wrongness. The thermodynamics wording is malformed
  prose outside the earlier matching prefix; it is not explained by the output
  cap or the automated degeneracy check.
  Required next step: preserve comparable 256-token selected-policy standalone
  controls for the affected prompts, and investigate any serving-only drift.
  For sampled anomalies, use equivalent fixed-prefix/token evidence or an
  appropriate prompt-correct control. Rerun/read the shared suite on the final
  server source and regenerate its degeneracy result and verdict.

- **P2: Finish the remaining required serving artifacts and closure audit.**
  Evidence: `vllm_ci_serving_benchmark.json` was absent. README/work-log status
  still lists pending work and contains older unmeasured-performance wording;
  the runtime audit leaves final cleanup pending.
  Why this matters: the original goal expressly requires the secondary
  100/100/32 burst result, larger-concurrency evidence, accurate final status,
  and no leftover serving processes holding devices.
  Required next step: save the CI raw/normalized benchmark and workload
  configuration, complete cleanup after the final serialized checks, and update
  README/work log/runtime audit with final results and exact artifact paths.
  Obtain a clean rereview, then create the required local stage-owned commits
  in both repositories and record their SHAs; never push. Commits properly
  follow clean-pass rather than being a prerequisite for this review.

## Other Concerns

- The canonical device sampler caps `top_k > 32` and unrestricted `top_k < 1`
  to 32 (`models/common/sampling/generator.py:607-613`). The server advertises
  generation defaults including top_k 64; shared tests also request 100.
  Document this inherited device-sampling limitation and its difference from
  explicit host compatibility. It does not justify adding another adapter
  sampler or changing the measured greedy path.
- The serving log reports an unpinned vLLM config/tokenizer revision while the
  model generator pins the checkpoint revision. Current rendered-prompt
  metadata matches the pinned controls. Pinning the serving revision would make
  this equality easier to preserve on a later launch; no present mismatch was
  demonstrated.

## Hard-Check Gaps

- No final source manifest yet ties all final evidence to the completed edits.
  Preserve enough source/run identity during closure to distinguish the padded
  baseline, trimmed implementation, and later fixes. Existing commands, logs,
  code inspection, and JSON are acceptable evidence; no new profiler or
  universal runtime-counter artifact is required.
- Serving advertises 262144 tokens without reduction, consistent with
  `doc/context_contract.json`; the inherited full-model selected-policy maximum
  context evidence is present. This review does not invent a new requirement
  to rerun every maximum-context case in the serving stage.

## Anomaly Ledger

- Observed anomaly: shorter prefill rows include tokens beyond their logical prompt.
  Evidence: adapter prefill, plugin common-width input assembly, and penalty
  mask construction cited above.
  Affected path: device prefill with repetition penalties.
  Control or comparison: model prefill itself correctly slices `:length`;
  sampler history does not.
  Likely subsystem: adapter prompt-history translation.
  Investigation performed: followed the input buffer and penalty-mask dataflow.
  Resolution: more-work-needed.

- Observed anomaly: presence penalties 0 and 2 produce identical repetitive continuations.
  Evidence: two final full-suite failures at `sampling_tests.log:2336-2425`.
  Affected path: raw completion penalty checks.
  Control or comparison: all 40 original-prompt tokens match the vLLM CPU
  sampler at penalties 0/2; an alternative prompt changes with the penalty and
  again matches CPU tokens at each setting.
  Likely subsystem: invalid variation expectation for the original prompt.
  Investigation performed: read failing outputs and independently compared
  `presence_control.json` / `presence_prompt_control.json` token sequences.
  Resolution: controlled; corrected final full-suite result is still required.

- Observed anomaly: full-suite API rejections and empty-text allowlist failure.
  Evidence: temperature outside [0,2] errors and special-token allowlist output.
  Affected path: shared test inputs/assertions.
  Control or comparison: allowlist CPU tokenization proof and host assertion
  tests exist; repaired allowlist case now passes live; temperature API error is explicit.
  Likely subsystem: test/runtime compatibility.
  Investigation performed: inspected test diff and original full-run failures.
  Resolution: cause controlled; corrected live final gate remains more-work-needed.

- Observed anomaly: mixed-parameter seeded outputs differ after batch permutation.
  Evidence: `sampling_tests.log:210-217`.
  Affected path: sampling state across live scheduling layouts.
  Control or comparison: several seed-only tests pass; live first-token
  logprobs are identical across repeats and AB/BA submission; full
  standalone/adapter comparisons are pending.
  Likely subsystem: seed/layout history or host/device sampling transitions.
  Investigation performed: inspected adapter sampling-signature resets and
  plugin parameter/history forwarding.
  Resolution: more-work-needed; may qualify as reproducibility-only only after
  the remaining correctness conditions are met.

- Observed anomaly: device allocation warning after model trace capture.
  Evidence: server log and dedicated source investigation.
  Affected path: canonical split model/sampler capture and replay.
  Control or comparison: tracker check pending.
  Likely subsystem: second trace scratch allocation lifetime.
  Investigation performed: read capture/replay source and the allocator diagnosis.
  Resolution: more-work-needed.

- Observed anomaly: malformed words in the longer shared qualitative outputs.
  Evidence: exact `shared_2`/`shared_3` snippets above.
  Affected path: 256-token full-model serving generation.
  Control or comparison: all six 128-token selected-policy greedy prefixes
  match, but those controls do not cover all flagged text.
  Likely subsystem: unclassified model/precision behavior versus serving drift.
  Investigation performed: read all twelve outputs and both control sets;
  independently compared prompts and greedy prefixes.
  Resolution: more-work-needed.

- Observed anomaly: short reduced-layer diagnostic outputs contain special tokens and noise.
  Evidence: `requests_reduced_*.json` and reduced tie logprobs.
  Affected path: layers 0/5 contract probe only.
  Control or comparison: synchronous, asynchronous, and trimmed request token
  streams match exactly for all nine control/concurrent/lifecycle responses.
  Likely subsystem: deliberately incomplete model.
  Investigation performed: compared every recorded token stream in all three runs.
  Resolution: controlled; these are not final quality evidence.

- Observed anomaly: shared outputs terminate at the 256-token test cap.
  Evidence: supervised/unsupervised, story, thermodynamics, and Fibonacci output tails.
  Affected path: qualitative harness output limit.
  Control or comparison: earlier controls use a shorter 128-token limit.
  Likely subsystem: explicit generation cap.
  Investigation performed: read actual tails and runner metadata.
  Resolution: controlled; no complete-answer claim is justified by these capped outputs.

- Observed anomaly: startup warns about disabling chunked prefill, custom scheduler,
  skipped worker warmup, unknown RPC env variable, and file-descriptor limit.
  Evidence: `server.log:14-21,92-102`; runtime audit.
  Affected path: serving startup/runtime configuration.
  Control or comparison: full-prompt-only adapter, context-sized budget, TT async
  controller, successful multi-request logprob/structured tests, and benchmark
  startup request.
  Likely subsystem: generic upstream/runtime configuration warnings.
  Investigation performed: checked source restrictions and active launch config.
  Resolution: controlled for the exercised workloads; allocation warning is
  explicitly excluded from this classification.

## Scope Inspected

- Goal/skill paths: exact
  `bringup/artifacts/multigoal-runs/20260925T171711Z/09-09-vllm.prompt.txt`;
  `.agents/skills/{stage-review,vllm-integration,tt-device-usage,qualitative-check}/SKILL.md`.
- Artifacts: vLLM integration README, work log, runtime and qualitative verdicts,
  server command/environment JSON, dedicated trace/logit investigations;
  selected precision and context contracts; actual full qualitative and
  HF/selected-policy controls; direct adapter/cache/page/trim artifacts;
  synchronous/asynchronous/trimmed request artifacts; original full sampling
  log; current and padded-baseline raw/normalized benchmarks and server logs.
- Code: complete new adapter; generator/model diff and relevant surrounding
  code; all plugin diffs; cache, scheduler-input, async sampling forwarding,
  common sampler/penalty paths; direct logit checker and relevant host tests.
- Commands: read-only `cat`, `sed`, `nl`, `grep`, `find`, `git status/diff`, and
  Python standard-library artifact analysis. `rg` was unavailable, so grep/find
  were used. No TT import, server, hardware command, benchmark, or test was run
  by this reviewer. Only this report was written.

Independently rederived positive evidence:

- The current raw benchmark has one completed request, exactly 4096 input and
  128 output tokens at concurrency 1, temperature 0. Its normalized values
  agree with raw JSON: TTFT median/P99 **2913.8976/2913.8976 ms**; TPOT
  mean/P99 **19.70377/19.70377 ms**; ITL median/P99 **19.60778/24.13276 ms**;
  aggregate output throughput **23.63116 tokens/s**; `1000/mean_tpot_ms` is
  **50.75171 tokens/s/user**. Single-request TTFT/TPOT percentiles are not
  population latency estimates.
- The padded same-workload baseline records **58.64425 ms** mean TPOT and
  **17.05197 tokens/s/user**. The trimming speedup is supported by same-harness
  evidence. Teacher-forcing is correctly treated only as a different-workload
  lower-bound reference.
- All six recorded rendered prompt strings and token lists match the pinned HF
  controls; all six greedy serving strings preserve the selected-policy
  control prefix after removing the expected turn terminator.
- The adapter uses the selected model policy with constructor/runtime precision
  assertions, scheduler-owned per-layer KV pools, disabled prefix caching,
  nonblocking canonical model/sampling traces, and a separate explicit host
  compatibility switch. Direct stale-input/changed-page evidence and exact
  synchronous/asynchronous request comparisons support the async capability
  for their documented reduced scope.

## Residual Risk

This is a first review of unfinished live work, not a stage closure. No measured
source/kernel profiling is needed for this serving stage. Pending runtime work
belongs to the stage owner and must remain serialized under the device skill.
The existing maximum-context policy and primary performance evidence are
credible, but the sampling defects, uncontrolled anomalies, lifetime warning,
missing final gates, and cleanup audit prevent clean-pass now.

## Interim Seed-Lifecycle Source Review

Second bounded source inspection during the next server load; not a final
stage rereview. The original findings above describe the earlier snapshot.
The stage owner has since supplied prompt-mask, numeric, allocation-tracker,
and longer qualitative controls; full final evidence will be reviewed together
at closure.

- **P1: Resumed prefill still restarts sampling state after KV preemption.**
  Evidence: `TTScheduler._preempt_request` is implemented in plugin
  `scheduler.py:233`; `apply_cached_req_state_update` replaces resumed KV block
  IDs while preserving request output tokens (`input_batch.py:86-108`). The
  model runner classifies resumed requests as prefill and supplies the original
  prompt plus already generated tokens. Its new `output_token_counts` transport
  is guarded by `not is_prompt`. Adapter `prefill_forward` still calls
  `configure_sampling` with the base seed, resets output counts, and classifies
  the entire recomputed prefix as prompt history (`generator_vllm.py:169-189`).
  Why this matters: the first token emitted after recomputation uses the
  request's initial device seed again. Presence/frequency penalties also lose
  all previously generated-token counts. Later decode-state restoration cannot
  repair this already emitted token. The serving path supports preemption, so
  a request approaching cache pressure can encounter this behavior.
  Required next step: preserve the original prompt/generated-history boundary
  and number of previously emitted tokens through resumed prefill, restore the
  appropriate seed before its sample, and add a targeted resumed-prefill
  regression. Re-prefill sampling needs offset `N` for `N` prior outputs;
  decode restores `N-1` because its model trace increments before sampling.
  Do not reduce context/capacity to conceal the path.

The ordinary decode repair is consistent under source inspection:

- Counts are derived from `num_tokens - num_prompt_tokens` in current compact
  request order, padded with zero, and transported only for an opt-in DP1 model.
- The model-input builder drains pending async output before rebuilding a
  changed layout; its trusted counts then match the scheduler token inputs.
- For `N` emitted tokens, restoring `base_seed + N - 1` followed by the traced
  increment yields `base_seed + N`, matching uninterrupted decode.
- Unchanged async replay ignores stale host counts. Identical parameter
  signatures now refresh prompt/output histories on a real reset.
- State-only restoration writes existing seed/penalty buffers without releasing
  traces; a page-table update with unchanged shape need not recapture.
- The earlier prefill prompt-tail masking defect is corrected in source using
  a separate history tensor masked with `-1`, leaving model input untouched.

Anomaly ledger addition:

- Observed anomaly: resumed-prefill state would restart RNG progress and erase
  generated-token penalties.
  Evidence: reachable scheduler preemption plus the prefill/decode metadata
  distinction above.
  Affected path: first sampled token after KV recomputation.
  Control or comparison: new CPU/lifecycle cases inspect decode reset and peer
  completion; those do not exercise resumed prefill.
  Likely subsystem: serving adapter prefill sampling-state restoration.
  Investigation performed: traced scheduler resume, cached request state,
  input construction, adapter prefill, and canonical sampler reset behavior.
  Resolution: more-work-needed.

No hardware, servers, TT imports, or tests were run by this reviewer in this
interim pass. Inspection commands read the relevant diffs, source, test cases,
and `seed_lifecycle_before.json`; only this report was updated.

## Interim Resumed-Prefill Source Review

The adapter-side repair now reconstructs the original prompt/generated-token
boundary, masks the two histories separately, restores `base + N` for a
re-prefill sample after `N` retained outputs, and leaves fresh-prefill behavior
unchanged. Counts are forwarded through the opt-in DP1 prefill path as well as
decode. Source inspection finds that arithmetic consistent. The recorded
reduced device oracle also matches all five sampler-state arrays and the next
token on identical logits: seed 777317, sampled token 333. Its uninterrupted
decode token 349 is separately labeled because prefill/decode numerical paths
differ; it is not misrepresented as an uninterrupted-token-equality result.

- **P1: Reconcile worker history with the scheduler after discarding pending async tokens.**
  Evidence: scheduler `_preempt_request` marks outstanding results stale;
  `_update_request_with_output` drops those results instead of retaining them
  (`scheduler.py:233-287`). Worker
  `_apply_sampled_tokens_to_state` still appends completed tokens to the same
  `CachedRequestState.output_token_ids` (`model_runner.py:3247-3312`). On resume,
  `apply_cached_req_state_update` updates computed count and block IDs only;
  it does not reconcile the worker output history with
  `scheduled_cached_reqs.num_output_tokens` or `all_token_ids`.
  The new scalar-count producer reads the worker's `num_tokens -
  num_prompt_tokens`. Upstream constructs authoritative `all_token_ids` for
  resumed requests and clears placeholder progress when preempting, so these
  two copies of request history can differ after an outstanding token is
  discarded by the scheduler.
  Why this matters: with three retained outputs and one extra worker-side
  discarded output, a resumed prefix of `prompt + 3` is combined with count 4.
  The adapter then infers an original prompt one token too short. The worker's
  larger token count can also mark the scheduled full prefix as an intermediate
  prefill, selecting an unintended host-sampling path. Correct adapter handling
  of a manually supplied count does not prove the count producer is correct.
  Required next step: reproduce this worker/scheduler disagreement in a
  targeted host state test including a pending-result drain. Reconcile the
  resumed worker prefix/count with scheduler-authoritative retained tokens
  after outstanding outputs are drained, or provide concrete source/runtime
  proof that the disagreement cannot occur. Preserve ordinary steady-async
  behavior; do not apply scheduler placeholder counts as real token history.

Anomaly ledger addition:

- Observed anomaly: scheduler-discarded async output can remain in worker history.
  Evidence: independent scheduler stale-result filtering, unconditional worker
  output append, and missing resume reconciliation described above.
  Affected path: metadata producer for resumed prefill under async preemption.
  Control or comparison: `resumed_prefill_reduced.json` supplies a known-correct
  count directly and therefore does not exercise this disagreement.
  Likely subsystem: plugin worker/scheduler history reconciliation.
  Investigation performed: read scheduler, upstream cached-request payload,
  worker update/apply paths, adapter repair, and recorded reduced oracle.
  Resolution: more-work-needed pending a targeted proof or repair.

This remains a bounded source review, not final stage closure. No hardware or
tests were run by the reviewer.

## Interim Worker-History Repair Review

The next scoped inspection finds the previously reported resume-chain source
defects resolved. No further concrete source defect was found in this pass.
This is not a final stage clean-pass; final sampling, CI, cleanup, and artifact
review remain outstanding.

Inspected changes establish the following chain:

1. Both the ordinary model-input entrypoint and lane entrypoint drain pending
   async results before reconciling a resumed request's history. A second drain
   after batch changes cannot reappend an already applied completion.
2. `apply_cached_req_state_update` treats only resumed requests' retained
   history as authoritative. It validates supplied full token IDs against the
   original prompt and retained count, or truncates the existing prefix when
   only a count is supplied. Normal decode leaves worker-ahead history intact.
3. The helper mutates the output list in place, preserving persistent-batch and
   logits-processor references. A resident front-packed row is reloaded using
   `add_request` at the same index, replacing logical lengths, token prefix, and
   block mapping. The lane path removes/readds the row using reconciled state.
4. The count producer therefore sees the retained prefix; resumed prefill
   recovers the original prompt/output boundary and uses seed offset `N`,
   while decode reset uses `N-1` before its traced increment. Ordinary unchanged
   async replay continues to ignore stale host metadata.

`tests/test_resumed_request_history.py` covers authoritative IDs versus count
fallback, resident versus absent rows, front-packed versus lane state, ordinary
worker-ahead decode, and a pending token arriving before resume reconciliation.
The test bodies exercise actual history update and input assembly methods. The
reviewer inspected these tests but did not run them or independently claim
their final result.

Anomaly ledger update: scheduler-discarded tokens persisting into resumed
prefill are **fixed in the inspected source**. Runtime/test outcome remains part
of final stage evidence. Commands in this pass were read-only git/source/test
inspection; only this review report was updated.
