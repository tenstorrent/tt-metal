# Stage Review

Verdict: clean-pass

Independent final review, 2026-09-27, of Stage 09 vLLM integration for
`google/gemma-4-26B-A4B-it`. This verdict covers the final live worktree and
artifacts identified by `readiness_vllm/final_source_manifest.json` and
`final_artifact_manifest.json`. Independent recomputation found no mismatch in
all 11 implementation/policy hashes and 18 final artifact hashes. Earlier
findings and interim source reviews are preserved in `stage_review_initial.md`.

Reviewed bases: tt-metal branch `gemma-4-26b-a4b-it`, HEAD
`eeb18d817be12d288c3c437dd593a468e9aa02ae`; vLLM branch
`gemma4-vllm-integration`, HEAD `5ffebf4128f81ea5cf8413175eabde52cd8c8d75`.
Stage changes are intentionally uncommitted at review time. The stage owner
must now create the prescribed local checkpoints in both repositories and
record their SHAs without pushing; that action follows this clean-pass.

## Required Work

None. The initial correctness findings, subsequent seed/resume-chain findings,
and previously pending final gates are resolved by source changes and the
recorded evidence. No source change or additional device run is requested.

## Other Concerns

- The inherited common stochastic device sampler caps effective top-k at 32,
  including requests for larger or unrestricted top-k. This is documented.
  Greedy benchmarks use the canonical split sampler; optional constrained or
  logprob compatibility uses vLLM CPU sampling. Unrestricted stochastic sampling
  equivalence is not asserted.
- The API launch does not pin its config/tokenizer revision, although the model
  and saved controls are pinned. Current rendered prompts and token IDs match.
  Pinning the serving revision would improve future reproducibility; there is
  no demonstrated present mismatch.
- Some answers are incomplete at the qualitative harness's 256-token cap.
  Malformed wording reproduces outside serving. The accepted quality result is
  a serving-regression smoke, not broad task or scientific accuracy.

## Hard-Check Gaps

- Live numerical logprob comparisons cover the first generated token/top 20.
  Full-vocabulary prefill and three decode-step comparisons use standalone and
  direct-adapter execution. Their complementary scopes are documented; the
  direct-adapter check is not presented as HTTP scheduler coverage.
- Resumed-prefill state has negative CPU controls, repaired worker-history/drain
  regressions, and a reduced canonical device sampler oracle. A final full-model
  HTTP test deliberately forcing KV preemption is not claimed. Source inspection
  and focused evidence resolve the concrete defects without another closure gate.
- Context remains 262144, matching the context contract and inherited selected-
  policy maximum-context evidence. Serving did not rerun every maximum-context
  case. Non-aligned lengths and cache growth are exercised without reducing
  advertised capability.
- The primary workload contains one measured request. Its TTFT/TPOT percentiles
  describe that request, not a population distribution. CI is a separate capacity
  workload, not the headline per-user decode metric.
- Allocation tracking covers exercised configurations, not every possible trace
  address arrangement. Serving profiling is prohibited by the stage skill and
  is neither missing required evidence nor requested here.

## Anomaly Ledger

- Observed anomaly: padded or stale prefill tails contaminated repetition history.
  Evidence: initial source finding; `prefill_prompt_mask_baseline.log`,
  `prefill_prompt_mask_host_tests.log`, and final sampling-contract regressions.
  Affected path: heterogeneous prefill with repetition penalties.
  Control or comparison: logical prefixes versus deliberately populated stale tails.
  Likely subsystem: adapter prompt-history translation.
  Investigation performed: followed plugin inputs into common penalty masks;
  inspected separate sampler-history masking and resumed-prefix handling.
  Resolution: fixed. Prompt tokens are masked beyond logical length without
  altering model input, and resumed output suffixes remain separate.

- Observed anomaly: seeded targets changed when peers finished or rows reordered.
  Evidence: `seed_lifecycle_before.json`, CPU negative controls,
  `seed_lifecycle_final.json`, and `seed_lifecycle_final_seed_only.json`.
  Affected path: sampling state during live batch/layout changes.
  Control or comparison: isolated targets before/after versus peers of length
  3 and 7, both submission orders, with and without penalties.
  Likely subsystem: sampling configuration resets and retained history.
  Investigation performed: inspected optional DP1 output-count transport, seed
  offsets, state-only restore, and unchanged async feedback; compared actual
  final token arrays rather than only success booleans.
  Resolution: fixed. All four peer/order cases in each final artifact exactly
  match isolated controls; final mixed-parameter full-suite sampling passes.

- Observed anomaly: resumed prefill restarted seeds and lost generated history;
  discarded async output could remain in worker/input-batch history.
  Evidence: interim source review, `resumed_prefill_cpu_before.log`,
  `resume_worker_history_before.log`, `resumed_prefill_reduced.json`,
  `resume_worker_history_after.log`, and `resume_worker_sampling_regressions.log`.
  Affected path: KV-preempted request recomputation and both worker entrypoints.
  Control or comparison: authoritative scheduler history versus worker-ahead
  tokens; identical cloned logits through the canonical device sampler.
  Likely subsystem: prefill history boundary and worker reconciliation order.
  Investigation performed: inspected early pending-result drain, resume-only
  reconciliation, resident-row reload, count validation, prompt/output masks,
  and seed offsets. Ordinary async worker-ahead history remains untouched.
  Resolution: fixed. Recorded tests pass (37 worker/history and 78 adapter/sampling
  tests). Device state arrays match; seed is 777317 in both paths; sampled token
  333 matches the same-logits reference. Uninterrupted token 349 is correctly
  separated as a prefill/decode precision-phase comparison, not the sampler oracle.

- Observed anomaly: original suite HTTP 400 temperatures, empty-text allowlist
  failure, and ineffective presence-penalty variation.
  Evidence: `sampling_tests_initial_full.log`, temperature source proof,
  `allowed_token_ids_tokenizer_proof.json`, `presence_control.json`, and
  `presence_prompt_control.json`.
  Affected path: shared test inputs/assertions.
  Control or comparison: installed API temperature range, actual allowed special
  token IDs, and device versus vLLM CPU tokens at presence penalties 0/2.
  Likely subsystem: test/runtime compatibility and an insensitive test prompt.
  Investigation performed: inspected repaired diffs. Temperature inputs obey the
  API; allowlists check token membership/usage; replacement presence prompts
  retain variation/determinism assertions. Original ABC outputs and replacement
  prompt effects match their CPU controls.
  Resolution: fixed/controlled. Final full profile has 72 passed, 1 skipped,
  0 failed in 1123.65 seconds. No reproducibility failure is being waived.

- Observed anomaly: all-vocabulary chat logprobs are skipped.
  Evidence: `sampling_tests.log` and `test_logprobs.py:84-110`.
  Affected path: top_logprobs=-1 against the default server cap.
  Control or comparison: ordinary top-N tests, full standalone/adapter numerical
  checks, and live top-20 comparisons pass.
  Likely subsystem: documented server max_logprobs limit, not missing logits.
  Investigation performed: verified the test skips only the matching API cap
  rejection and otherwise raises; checked the final suite result.
  Resolution: controlled. This is the shared default-cap skip, not a new xfail
  or an ignored correctness failure.

- Observed anomaly: allocations after model capture triggered a corruption warning.
  Evidence: `AUTODEBUG_trace_allocations.md`, `adapter_trace_allocations.json/.log`,
  `logit_determinism_full.json/.log`, and `qualitative_256_controls.json/.log`.
  Affected path: canonical split model/sampler capture and replay.
  Control or comparison: bounded functional runs with tracking/tracebacks enabled
  and program-cache checks retained.
  Likely subsystem: temporary allocations during second-trace capture.
  Investigation performed: inspected allocator/capture lifetime reasoning and
  tracker results for reduced split replay, full logits, and four full-model
  256-token canonical split controls.
  Resolution: controlled. No surviving unsafe allocation was reported. The
  warning remains preserved and the conclusion is limited to tested paths.

- Observed anomaly: malformed wording and over-hyphenation in long outputs.
  Evidence: final shared outputs, `qualitative_256_controls.json`, and
  `qualitative_control_comparison.json`.
  Affected path: full-model chat, especially story/thermodynamics.
  Control or comparison: all six selected-policy greedy prefixes match. Extended
  story and thermodynamics greedy strings match standalone for the entire
  256-token controls, reproducing `tiny-brass-heart` and
  `In any own-contained system`. A new seeded standalone story includes similar
  `brass-bound-and-glass` wording.
  Likely subsystem: inherited selected-policy output behavior.
  Investigation performed: read all 12 final outputs and all four longer controls;
  checked prompt format/final hash and complementary exact numerical comparisons.
  The original unseeded sampled `pair-o-glasses`/`brass-bound-and-etched` draw is
  not claimed exactly reproduced.
  Resolution: controlled as a serving-regression question. Final sampled
  `brass-bound-and-etched`, `own-contained`, and `measure of-disorder` remain
  disclosed output limitations, not certified correct prose/science.

- Observed anomaly: reduced/synthetic probes contain special-token noise or
  mechanical continuations; several chat answers end before completion.
  Evidence: reduced request artifacts, `requests_full_final.json`, shared output
  tails, and generation limits.
  Affected path: incomplete-model/raw-token contract probes and capped chat.
  Control or comparison: reduced sync/async/trimmed streams agree; full isolated
  and concurrent 31/63/95-token probe streams agree exactly; the 33/1057/33 lifecycle
  returns the same repeated 33-token-prompt result.
  Likely subsystem: artificial inputs/model depth and explicit output cap.
  Investigation performed: read outputs and compared token arrays; kept artificial
  probes separate from prompt-correct chat quality evidence.
  Resolution: controlled. All nine final probes return 96 tokens; final shared
  chat degeneracy checking exits 0 with no findings. Incomplete answers are disclosed.

- Observed anomaly: startup warnings for disabled chunked prefill, custom async
  scheduler, skipped warmup, file-descriptor limits, and environment settings.
  Evidence: final `server.log`, launch JSON, and `runtime_audit.md`.
  Affected path: installed vLLM/TT runtime configuration.
  Control or comparison: full-prompt adapter, context-sized budget, proven TT
  deferred decode, on-demand capture, benchmark startup request, completed B32
  and queued workloads.
  Likely subsystem: generic upstream configuration/compatibility warnings.
  Investigation performed: checked warnings against source restrictions and
  launch configuration; treated the trace warning separately.
  Resolution: controlled for the exercised contract. Optional CPU compatibility
  is explicit and is not selected by either greedy benchmark.

- Observed anomaly: shutdown force-stops one EngineCore and prints nanobind leaks.
  Evidence: final `server.log`, `final_cleanup.json`, and
  `server_decode_seed_fixed.log`.
  Affected path: abort/timeout=0 API shutdown and interpreter binding references.
  Control or comparison: the smaller previous run reports identical counts:
  105 instances, 973 types, 4455 functions. All three final owned PIDs are absent
  after shutdown; held runner exit code is 0.
  Likely subsystem: installed runtime shutdown policy and binding teardown.
  Investigation performed: read signals, manager/FastAPI completion, both leak
  reports, and before/after process evidence.
  Resolution: controlled. No leftover serving process is evidenced. This is not
  a claim of graceful device teardown, absence of binding-reference issues, or
  measured proof about per-request memory growth.

## Scope Inspected

- Goal/skills: exact Stage 09 prompt at
  `bringup/artifacts/multigoal-runs/20260925T171711Z/09-09-vllm.prompt.txt`;
  `.agents/skills/{stage-review,vllm-integration,tt-device-usage,qualitative-check}/SKILL.md`.
- Documentation/policy: final README, work log, runtime/qualitative verdicts,
  launch/check commands, environment, context contract, selected precision,
  HF/selected controls, and AutoFix/AutoDebug investigations. Corrected README
  precision wording agrees with source.
- Source: complete `tt/generator_vllm.py`; generator/model diffs and surrounding
  trace/sampling/cache code; plugin platform, scheduler, model input/runner,
  input batch and async controller; common sampler/penalty paths; new contract
  checks and shared test changes.
- Artifacts: final suite/runner logs, raw/normalized benchmarks, request and seed
  lifecycle artifacts, cache/page/stale-input/async controls, numerical logits
  and logprobs, qualitative strings/controls, tracker, cleanup and manifests.
- Commands: read-only `cat`, `sed`, `grep`, `find`, `git status/diff`, and Python
  standard-library JSON/hash/token-array analysis. No tests, TT imports, servers,
  device commands, profiler, or benchmarks were run by this reviewer. Only this
  review report was written.

Independently verified benchmark results:

| Final workload | Completed; input; output tokens | TTFT P50 / P99 ms | TPOT mean / P50 / P99 ms | ITL P50 / P99 ms | Aggregate output tokens/s | 1000 / mean TPOT |
| --- | --- | --- | --- | --- | --- | --- |
| Primary 4096/128, B1 C1, greedy | 1/1; 4096; 128 | 2549.301 / 2549.301 | 19.713 / 19.713 / 19.713 | 19.619 / 27.752 | 25.331 | 50.728 tokens/s/user |
| Secondary CI 100/100, 32-request burst, greedy | 32/32; 3200; 3200 | 30815.582 / 30816.835 | 680.950 / 677.478 / 754.133 | 677.253 / 692.437 | 32.690 | 1.469 tokens/s/user |

Normalized values agree with raw JSON. CI has no client concurrency cap, raw
`max_concurrent_requests` is 32, and logs show 32 running requests. The same-
harness padded-row baseline has 58.64425 ms mean TPOT versus final 19.71285 ms
at 4096/128/B1/C1. Selected teacher-forcing is only a 161/100/B1 decoder latency
reference, not a same-workload serving comparison.

The full numeric artifact records 22 exact full-vocabulary comparisons across
prefill and three teacher-driven decode steps, repeated and reordered rows.
Live first-token top-20 dictionaries agree across repeats and AB/BA submissions;
the two standalone log-softmax comparisons have maximum difference 0. Final
qualitative SHA256 is
`508928954b84fb5e82a9a6152dfccab8fc34d57f6498a1ce1100d9ab1c46b904`
(after the newline-only formatting recorded below).

The adapter retains selected per-layer precision, TP4/DP1, external hybrid KV
ownership, context 262144 and max sequences 32. Normal decode delegates to
canonical nonblocking model/sampler traces and persistent device token feedback.
Stable replay does not upload stale host token/position/history or unchanged
page tables. Prefix caching stays disabled. Host compatibility is explicitly
optional and is not selected by either greedy benchmark.

## Post-Hook Audit

Clean-pass remains valid after the commit-hook cleanup. Independently compared
the staged pre-hook bytes with the current files: all eight JSON changes append
exactly one newline and preserve parsed content. Recomputed all 11 runtime/policy
hashes and all 18 refreshed artifact hashes with no mismatch. The qualitative
comparison metadata now matches the formatted file's SHA256 above; generated
strings, benchmark values, and serving results are unchanged.

Inspected the three test-script diffs: import ordering/blank-line formatting,
plus four negative-test contexts changed from `pytest.raises` to the root
`expect_error` fixture with identical exception types and regex strings.
`conftest.py:948-967` confirms the fixture delegates to `pytest.raises(error,
match=message)` and adds expected-error logging. No assertion was weakened.
The recorded post-hook run passes all 83 affected host tests in 35.43 seconds
(`readiness_vllm/post_hook_host_tests.log`); no device rerun is needed for these
changes. `post_hook_artifact_check.json` preserves the formatting audit.

The vLLM checkpoint is now
`7f72b1c6e905f5137fe3377f2e7b42738d3f271d`; its worktree is clean and the
reviewed serving source hashes remain unchanged. The tt-metal checkpoint and
final SHA recording remain the stage owner's closure action.

## Residual Risk

This accepts the documented Stage 09 serving contract on the tested runtime and
mesh, not a universal numerical or production-soak guarantee. Coverage limits,
inherited stochastic top-k restriction, output caveats, and shutdown warnings
remain visible. No required validation gate is pending. Post-review local commits
and SHA recording remain the stage owner's prescribed closure action; source
changes after this manifest would require appropriate renewed validation/review.
