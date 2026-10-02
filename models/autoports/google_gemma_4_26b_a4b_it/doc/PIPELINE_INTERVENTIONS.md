# Gemma 4 pipeline interventions

This record distinguishes what the stock multigoal pipeline accomplished from
what required human action or a new human prompt. Durations are elapsed evidence
windows, not continuous compute time.

## What the pipeline did

The original 11-stage Gemma 4 pipeline started on 2026-09-25 at 17:17 UTC. It
brought up the decoder, fused and optimized it, added multichip execution,
assembled and optimized the full model, selected datatypes, and integrated and
optimized vLLM serving.

By 2026-09-27 at 19:50 UTC, after about 50.5 elapsed hours including pauses, it
had completed the serving stages and both required performance profiles. Stage
11 passed 42 host checks and began the frozen accuracy workload, but stopped at
8/536 responses to preserve the stage's one-hour budget. The pipeline therefore
ended with valid serving/performance evidence and incomplete accuracy evidence.

The work below happened because a human repaired the environment, redirected a
blocked stage, or explicitly asked an additional agent to continue beyond that
pipeline endpoint.

## Human interventions during the pipeline

### 1. Make the pipeline runner usable

Human action:

- Mounted both `codex` and its required `codex-code-mode-host` companion into
  the development container.
- Selected the isolated personal Codex home and refreshed personal-account
  authentication after the Stage 2 HTTP 401.
- Resumed Stage 3 when weekly usage capacity became available.

Why it was needed:

The initial wrapper exposed only the `codex` executable, so app-server code
execution could not run. Authentication and usage limits were infrastructure
interruptions, not model failures.

What it unlocked:

The recorded pipeline threads resumed instead of restarting the bringup.

### 2. Continue through the Stage 4 fabric teardown failure

Human direction, paraphrased: investigate and fix the multichip teardown rather
than treating it as a terminal model failure.

Why it was needed:

The model completed collective traffic, but Watcher failed when ERISC ownership
was handed back with stale NoC packet tags. Earlier single-device stages did not
exercise this path.

Incremental result:

- Added a full NoC barrier and `noc_clear_packet_tags(NOC_INDEX)` before ERISC
  handback.
- The model-free CCL reproducer passed.
- The expert-parallel Watcher probe passed all 16 comparisons.
- The original Stage 4 thread resumed.

### 3. Redirect Stage 9 to an existing serving image

Human direction, paraphrased: inspect the machine for a usable serving runtime
and continue with it instead of accepting the generic development environment
as the only available environment.

Why it was needed:

The generic TT-Metal interpreter lacked OpenAI, uvloop, vLLM and the TT plugin.
The initial automated diagnosis did not inventory existing local serving images.

Incremental result:

- Reused the existing vLLM/TTNN serving image.
- Corrected an initial wrapper mistake that mixed the checkout's `build/lib`
  with a different compiled TTNN runtime.
- Successfully loaded the image runtime, editable TT plugin and Gemma autoport
  together without installing dependencies into the repository environment.
- Stages 9 and 10 then completed.

### 4. Authorize a separate Stage 11 benchmark client

Human prompt, paraphrased: find and apply a recovery for the missing benchmark
dependencies, while keeping the serving environment unchanged.

Why it was needed:

Neither provisioned interpreter contained `lm_eval==0.4.13`, and no existing
image supplied it.

Incremental result:

- Created `/home/mvasiljevic/.venvs/gemma4-benchmark` outside the repository.
- Reused the serving image's PyTorch/vLLM packages and added only the benchmark
  client dependencies.
- The pipeline completed both performance profiles, 42 host checks and the
  context-contract check.
- Accuracy made real progress but remained incomplete at 8/536 because the
  human did not waive or silently reset the one-hour pipeline budget.

## Human prompts that extended work beyond the pipeline

### 5. Start a dedicated TTFT optimization agent

Human prompt, paraphrased: continue after the pipeline with a separate Astra
agent focused on minimum warmed short-input concurrency-1 TTFT, while retaining
correctness, canonical sampling, full context and higher concurrency.

Agent effort:

- Same-day run on 2026-09-28; exact start clock was not recorded.
- Final runtime implementation committed at 12:31 UTC.

Problem it addressed:

Short requests were dominated by prefill and first-token host/dispatch overhead.

Improvement beyond the pipeline:

- Added bounded generator-owned prefill and canonical first-token sampling
  traces.
- S128/O128/C1 median TTFT improved from 419.10 to 95.24 ms: a 77.3% reduction.
- Repeat medians were 95.36 and 95.03 ms.
- The 6–9% C1 TPOT tradeoff was retained and documented rather than hidden.

The full mechanism and measurement progression are in
[`ttft_optimization/README.md`](ttft_optimization/README.md): eager baseline
419.10 ms, traced async 104.29 ms, and traced latency-priority sync 95.24 ms.
It also documents exact-length B1 prefill tracing, first-token sampling capture,
compact page-table views, compatible decode-graph reuse, bounded fallbacks,
correctness evidence and the scheduler/TPOT tradeoff.

### 6. Ask for remote CI, benchmarks and evals on QB2 `main`

Human prompts, summarized:

- Use `tenstorrent/tt-agentic-bringup-qb2` only as the CI dispatcher.
- Use the TT-Metal and tt-inference-server branches for implementation.
- Run benchmarks and evals separately, matching the Qwen 3.8 benchmark matrix.
- Try the same agentic evals used for Qwen.
- Reuse the Docker image, monitor every dispatched run, and fix failures.

Agent effort:

- 2026-09-28 12:31 to 2026-09-29 09:08 UTC, about 20.6 elapsed hours.

Problems it addressed:

- The exact vLLM commit could not be published with available permissions.
- The initial benchmark selected the wrong matrix and was mistakenly cancelled
  while still making progress.
- HTTP readiness preceded background trace completion.
- Agentic runs exposed missing provisioning, Docker loopback routing, CPU limits,
  tool-choice/parser flags, parser-constructor compatibility and timeout/artifact
  gaps.

Improvement beyond the pipeline:

- Embedded the exact vLLM snapshot and built one reusable source image.
- Added `/tmp/ready` gating and the Qwen-style benchmark configuration.
- Completed the full 23-row benchmark sweep with zero request failures.
- Completed GPQA at 25/40, Terminal-Bench at 2/5, and timeout-limited SWE-Bench
  at 1/5.
- Added durable CI, networking, parser and compact agentic-evidence fixes.
- No QB2 model-bringup branch was used.
- The final serialized SWE retry is still running in
  [job 109283453627](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36530661132/job/109283453627).
  It was left untouched by human direction and is not included in the 1/5
  completed-result claim above.

The CI transport and exact-source workaround are documented in
[`ttft_optimization/remote_ci_workaround.md`](ttft_optimization/remote_ci_workaround.md).
The complete run chronology, image provenance, 23-row benchmark results,
standard eval and agentic-eval outcomes are in
[`readiness_vllm/ttft_optimization_remote/README.md`](../readiness_vllm/ttft_optimization_remote/README.md).
Workflow-routing compatibility is recorded in
[`ttft_optimization/workflow_compatibility_audit.md`](ttft_optimization/workflow_compatibility_audit.md).

### 7. Start a dedicated TSU optimization agent

Human prompt: start an Astra agent to maximize the TSU actually used in the
benchmark cases, working on both device performance and vLLM/TTI overhead, and
let it analyze, experiment and measure iteratively.

Follow-up human interventions:

- Changed the agent from ultra to high reasoning.
- Required image reuse after the one necessary exact-runtime build.
- Directed it to prioritize large bottlenecks and only C1/C8/C16.
- Asked it to close without starting the remaining full sweep.

Agent effort:

- Approximately 2026-09-29 09:08 to 15:18 UTC, about 6.2 elapsed hours.

Problems it addressed:

- Prompts above 1024 tokens discarded and recaptured the decode trace for every
  request, adding roughly 300 ms/request.
- Synchronous serving added avoidable scheduling overhead.
- Dense higher-concurrency decode repeated shared MLP, collective and tail work
  for each logical row.

Improvement beyond the pipeline and TTFT work:

- Retained compatible decode traces and enabled async serving.
- Remote 4K/C1 improved from 43.04 to 50.68 TSU: +17.74%.
- Safe shared batching improved matched local 4K/C8 by 3.03% and 4K/C16 by
  4.44%.
- The focused exact-image run completed 104/104 responses with exact text and
  token lengths.
- The full 29-row C1/C8/C16 sweep was explicitly not run at human-requested
  closure. Selected remote qualification was blocked before model execution by
  checkout EACCES on runner `p04`; it is not reported as a model failure or pass.

The bounded final result and its qualification limits are in
[`tsu_optimization/closure.md`](tsu_optimization/closure.md). The full
experiment chronology, human interventions and rejected candidates are in
[`tsu_optimization/work_log.md`](tsu_optimization/work_log.md). Device/serving
bottleneck analysis is in
[`tsu_optimization/topology_audit.md`](tsu_optimization/topology_audit.md),
normalized measurements are in
[`tsu_optimization/perf_summary.json`](tsu_optimization/perf_summary.json), and
the build-once/reuse provenance is in
[`tsu_optimization/image_reuse.json`](tsu_optimization/image_reuse.json).

### 8. Independent review agents

Human/skill direction: independently audit TTFT and TSU correctness,
measurement claims, evidence packaging and closure boundaries.

Result:

- TTFT passed bounded local review while still requiring remote qualification.
- TSU review found and caused repair of a reference-harness issue, then passed
  the bounded local and closure checkpoints.
- Review did not convert the unrun full sweep, timeout-limited evals or blocked
  remote run into passes.

### 9. Eval wall-time investigation (ongoing)

Human prompt, 2026-10-01: “start an astra agent ... analyze and understand what
are the bottlenecks ... make full prediction ... iterate on small tests ... then
run actual CI tests and monitor them.” The parent delegated a dedicated Astra
agent. The retained evidence window starts at 12:43 UTC; work duration and final
CI outcomes will be closed out after the monitored probes finish.

Problem and current measured findings:

- The baseline used the old image with async scheduling disabled. At real
  12K–49K SWE contexts, same-image warmed async decode improves approximately
  5–6%, not the earlier short-context 17.7% on every request. Prompt processing
  remains unchanged; all ten compared output hashes match.
- New growing-context requests compile thousands of kernels: ten native
  30-token Django responses take 168.17 seconds on first use versus 80.13
  seconds warm. This is a cache-state comparison, not a delivered speedup.
- The task shell selected base Python instead of the task's `testbed` conda
  environment. The exact Matplotlib task image reproduces the missing-package
  failure; the corrected environment immediately reproduces the real bug.
- Django repeated the same failed command 300 times. A generic warning changes
  the next action in a matched replay; its effect on solve/reward is not yet
  validated. It is explicitly separated as an agent-policy intervention.
- Approximately 686K generated tokens are missing from saved successful
  responses, including no-tool format errors and timed-out/retried requests.
  New payload-free request telemetry captures those attempts, usage, timings
  and tool categories. Request timeout now accommodates the unchanged 32K
  output limit; final per-trial budget remains 7200 seconds.

Fast iteration strategy: existing images only; 2K–49K real-context replays,
short next-action probes, exact task-image environment checks, 33 host tests,
then capped 15-minute CI trials. No full five-task rerun has been launched yet.
CI probes: [Matplotlib36865475336](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36865475336),
[Django36867491629](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36867491629).
Detailed forecasts, configuration, image digests, assumptions and measurements
are in [`eval_speed/README.md`](eval_speed/README.md). No eval-quality success is
claimed while these outcome-bearing tests remain unresolved.

By15:00 UTC, the measured diagnostic-startup improvement is394.508 seconds
saved (79.1% of external warmup): a scoped4K probe replaces the broad context
sweep while retaining full model context. Actual CI confirms this; model load
remains590.5 seconds. Thinking-only Matplotlib CI36870715050 still times out
at900 seconds/reward0. A local20-minute Django thinking-plus-repetition-guard
trial also fails, isolating only0.62 seconds of tool execution and a670-second
discarded generation. Widening the detector to1024-token patterns stops a matched
pathological replay in91.65 seconds versus an unfinished180-second control,
with identical output prefix and matched8.62-second TTFT. Separate-server C1
CI probes36876743431 and36880038816 are monitored for actual outcome quality.
The guard changes termination policy; no solved-task or release-speed success
is inferred from early termination alone. A hash-matched bounded HF reference
and a greedy/sampled/greedy stale-state check investigate the remaining quality
problem. All inference images are reused; no rebuild has been requested.

## Remaining eval-focused work

User-requested shutdown checkpoint,2026-10-01~15:30 UTC: the Astra evidence
window was12:43–15:30 (about2h47m). All dispatched CI probes are complete and
inspected, including guard128/run36876743431 and guard1024/run36880038816;
both time out at900 seconds with reward0. The final wider guard has21 valid
tool responses,7 repetition stops and only2.48 seconds of tool execution.
No solved-task speedup is claimed. The local BFP8 configurable-weight control
was stopped during loading at the user's request, before measurements; it is
unselected and does not change the release precision policy. HF128-token
reference completed with a coherent recap but does not diagnose long-loop
causality. No CI remains to monitor. All local probes are stopped; retained
artifacts and exact next-step limits are in`eval_speed/README.md`. The full
five-task release suite has not been rerun. Resume requires a new user request.

- Analyze why standard and agentic evals take so long.
- The user revoked the stop and resumed this investigation at~18:20 UTC.
  The four-chip health smoke passed without reset. A configurable-weight BFP8
  control (fixed prefill unchanged) turns one exact seeded91.65-second repetitive
  response into a31.70-second response with a valid parsed tool call. This is
  bounded next-action evidence only, not solved-task progress. A seeded900-second
  Django outcome trial is running from18:32:53; policy remains unselected and
  full readiness accuracy remains required. See`eval_speed/weight_control.json`.
- By 18:55 UTC, that bounded Django trial has verifier reward 1: its patch
  passes the one FAIL_TO_PASS and all 103 PASS_TO_PASS tests. The agent still
  reaches the 900-second timeout after three no-tool responses, so this is a
  verified patch-quality result, not clean completion or a measured 8x speedup.
  The weight-only candidate also passes the existing 100-token traced readiness
  reference (prefill top-1 0.96, decode top-1 0.98, top-5/top-100 1.0). This narrow
  reference does not establish long-context accuracy or select a release policy.
  A same-host, same-request-seed selected-policy control started at 18:59 UTC.
  Bounded CI [36910894168](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36910894168),
  job 110533086359 on `qb2-120-p01t03`, tests the candidate with the same 900-second
  cap and reused exact image `ad58effd178b...`; no image build or full suite is
  launched. Both runs are monitored. Complete provenance is in the eval report.
- At 19:30:04 UTC that CI run completes: artifact 11189981342 confirms reward 1
  and all 104 required/regression tests pass. The paired local selected-policy
  control scored 0 at the same 900-second cap; both use the identical initial
  request and seed locally. Candidate CI also times out, with its correct edit
  made around 13m25s and only 0.679 seconds in tools. No clean-completion speedup
  is claimed. A replay identifies the local candidate's final submission command
  printed as text instead of a tool call. A default-off, audited adapter for
  only that exact final marker passes 140 host tests; a fresh capped trial is
  running from 19:28:37. This is explicitly a harness-policy intervention,
  separate from precision. The CI image was reused, and the full five-task suite
  remains unrun. Approximately 77% of the improved CI request path is now TTFT.
- At 19:38:29 UTC the local adapter trial finishes cleanly in 585.413 agent
  seconds, reward 1, no exception and all 104 tests passing. Its first 34 work
  commands match the previous candidate trial. The adapter converts exactly one
  final marker; the 276.273-second improvement across identical first-35-response
  token totals is predominantly cache-state, not a causal adapter speedup. One
  1,200-second CI follow-up (36916089719 / 110550407919) tests clean completion,
  and one separate-server 900-second Matplotlib probe tests generalization.
  Both are bounded and monitored; images are reused, the full suite stays gated.
- Matplotlib completes at 20:00:28 UTC in 647.041 agent seconds with a native
  submission, reward 1 and all 182 verifier tests passing. The adapter does not
  fire. The same loaded server begins one 900-second Astropy check at 20:03:07;
  remote Django CI has become reachable and completes its 98.167-second warmup.
  These are positive cross-task outcomes, not an extrapolated all-five pass or
  a causal speedup percentage against the older failed trajectories.
- Measure their TTFT and TSU.
- Final bounded checkpoint at 20:26 UTC: actual QB2 CI 36916089719 /
  110550407919 finishes cleanly with native submission in 845.733 agent seconds,
  reward 1 and all 104 tests passing; the adapter is enabled but never fires.
  Artifact 11191289165 is downloaded and reviewed. Local Astropy earns reward 1
  with all 427 tests passing at its 900-second cap; the passing edit and
  reproduction precede the deadline, but clean submission remains unproven.
  The candidate therefore has positive patch outcomes on three tasks, clean
  completion on two, and an actual clean CI validation—not an all-five release
  pass. Sympy/sklearn still need bounded checks. Final CI request time is 70.5%
  TTFT; its 36m52s total workflow also includes a 386.5s HF fetch, 620.6s server
  startup health wait and 98.167s diagnostic warmup. The image was reused.
  Performance-claim review keeps cache warming, precision/quality changes,
  native completion and causal serving speedups separate. About 4h53m active
  wall time was used across 12:43–15:30 and 18:20–20:26, excluding the shutdown
  gap. All dispatched CI is complete; the owned local server is stopped and
  both TT device files are free. Exact provenance, limitations and next gates
  are in `eval_speed/README.md`; release precision remains unchanged.
- Separate model/device performance from eval-framework overhead.
- Check whether the same evals can run with more parallelism and finish sooner.
- Dispatch eval subsets in parallel on multiple CI machines during development.
- Reuse the existing image for these experiments where possible.

## Evidence

2026-10-02 05:31 UTC: the user explicitly resumes with an aggregate-suite goal:
measure the original five tasks together and reduce total time while preserving
reward. A bounded Sympy/sklearn pair is dispatched serially as CI 36969576147 /
110720576884 with reused image and TTI b2ffb198; local Astropy independently
tests clean completion under a 1,200-second cap. The final topology remains one
persistent server and serial C1 trials. Request-counter instrumentation and a
suite phase/outcome summarizer make startup, warmup, inference, tools, verifier,
late responses and total dispatch-to-finish time explicit. Current progress and
forecast gates are in `eval_speed/suite_20261002.md`. No full-suite result or
ten-task expansion is claimed yet.

By 06:13 UTC, the Astropy extension fails: a passing intermediate patch is
over-edited into a syntax error before its 1,200-second deadline (reward 0).
A 600-second thinking-disabled control also times out with reward 0; it moves
reasoning into shell comments rather than removing the semantic loop. Neither
is retained as a speedup. A fresh 900-second task-independent focused-completion
prompt tests clean termination without feeding the task solution or verifier.
The Sympy/sklearn pair remains monitored; the full suite stays gated.
This resumed window has used approximately 42 minutes of active agent wall time
so far (cumulative approximately 5h35m, excluding the prior user stop interval).

06:27 UTC checkpoint: the serial pair completes in 2,462 dispatch seconds:
sklearn 688.289 agent seconds, native reward 1; Sympy 898.432 seconds, submitted
reward 0 after restoring the source and narrating an unexecuted fix. The generic
focused-completion policy then yields clean native Astropy reward 1 in 655.569
seconds (427 tests), not merely an earlier passing timeout. Matching Sympy CI
36973577503 and a local Django policy regression probe are monitored. An earlier
Sympy dispatch 36973397458 failed checkout because the agent used an abbreviated
SHA; full-SHA redispatch fixes the orchestration error without a rebuild.
TTI 95a67381 includes the explicit prompt, C1 request counters, and persistent
JIT-kernel-cache location; only the prompt's Astropy outcome is measured so far.
Active resumed window is approximately 56 minutes; cumulative approximately
5h49m. No combined suite result or all-five quality claim yet.

07:02 UTC: the focused prompt fails a fresh Django regression (900.055 seconds,
reward 0), so it is rejected as a suite-wide policy; the agent does not pick a
different prompt per task. The completed precision audit motivates an explicit
fresh-BFP8 EP4 prefill-weight control, keeping the original prompt and other
numerical policies. TT e2c286b743 adds backward-compatible schema-2 plumbing and
runtime dtype attestation; 14 host checks pass. The all-layer 100-token gate
passes prefill top1 .99 (prior .96), decode .98, top5/top100 1.0 and traced decode
50.403 tokens/s. This is narrow accuracy evidence, not a SWE solve or speedup.
The exact image is reused with read-only Python/policy overlays. Fresh-container
username/cache and Tracy-directory permission failures are diagnosed before
device use; the corrected server opens all four chips normally as UID 6002.
Sympy policy CI 36973577503 remains monitored. Active resumed time is about 91
minutes, cumulative about 6h24m; the five-task suite still has no combined result.

07:14 UTC: focused Sympy CI 36973577503 completes (1,200.011-second timeout,
reward 0), confirming that prompt is not a suite candidate. All CI is terminal.
The same-image prefill-BFP8 server passes a coherence-only six-prompt check,
with exact-text changes and a thermodynamics wording caveat disclosed. A fresh
900-second original-prompt Sympy trial now tests the numerical control without
task-specific guidance. Full-suite measurement remains gated on correct clean
completion. Active resumed time is about 103 minutes, cumulative about 6h36m.

07:25 UTC: TTI 9bfa8ea7 adds explicit persistent pinned-HF-cache wiring after
finding that launcher local-dir weights and the autoport's pinned hub loader use
different cache layouts. It sets cache environment before Python imports and
shares one immutable snapshot. All 122 related host tests pass; startup savings
are unmeasured and no image is rebuilt. The prefill-precision Sympy trial is
still bounded and monitored. Active resumed time is about 114 minutes,
cumulative about 6h47m. No combined-suite result is claimed.

07:32 UTC: prefill-BFP8 Sympy fails at 900.115 seconds (reward 0), despite its
improved short numerical gate. Most time is model requests; tools take 3.544
seconds. Exact repeated successful commands motivate a separately audited,
default-off advisory, not another broad prompt rewrite. Ninety-nine host checks
pass; a fresh 900-second same-server Sympy control starts at 07:31:46. Sampling
and task information are preserved, but prompt-policy and warmer-cache effects
are declared separately. Active resumed time is about 121 minutes, cumulative
about 6h54m. The original five still lack a combined measurement.

07:35 UTC: before the advisory fires, seven matched Sympy requests reproduce
28,216 input / 4,192 output tokens and the same commands. Warm loaded-server
proxy time is 105.852 versus 210.263 seconds; exact TTFT falls 124.355→20.143,
post-first-token time stays 85.70→85.66. This is a measured 104.411-second
warm-state saving for that prefix, not an outcome or decode-speed improvement,
and not proof of disk-cache-only restart gains. It supports measuring the five
tasks on one persistent server as the user requested.

07:44 UTC: the first advisory cleanly submits Sympy in 639.742 seconds, but
reward remains 0: the required bug is fixed while two existing regressions break.
The agent used only custom examples. A generic existing-repository-test and
diff-review requirement is added to the default-off advisory, with no task-
specific solution or hidden-test hints. A fresh same-cap control starts at
07:44:17. All 33 telemetry tests pass; 14 helper tests also validate distinct
saved-response versus censored-window counter reporting. Active resumed time
is about 133 minutes, cumulative about 7h06m. Still no all-five combined result.

08:00 UTC: the revised advisory also fails Sympy (900.118-second timeout, reward
0), so neither advisory nor the new prefill control is promoted to the suite.
To satisfy the user's actual aggregate-measurement request, the original five
tasks are restored with 7,200-second caps, one persistent C1 server, original
prompt and the configurable-BFP8 candidate with three prior clean solves.
Pre-dispatch forecast is 65m / 3h06m40s / 5h20m best/expected/conservative wall
time, with 10h summed agent hard budget; this does not forecast five correct
solves. All deviations and unresolved task-specific quality risks are recorded
in `eval_speed/suite_20261002.md`. Reused-image source-overlay support is prepared
and host-tested but is not enabled in this baseline. A saved-HF long numerical
control is prepared to investigate quality separately without another expensive
CPU generation. Active resumed time is about 149 minutes, cumulative 7h22m.

08:04 UTC: actual combined CI **36981858976 / 110758014503** starts on
`120-qb2-p04t07` with exact TTI **83d9c26a**, reused image and no build. It is
continuously monitored through the verified HTTP alias. All 220 selected host
checks pass. The local server is stopped idle and devices are unowned before a
separate bounded long-context teacher-forcing check starts locally. Full-suite
ordering/results/time and long-control agreement are pending; no success is
claimed merely from dispatch or the zero rebuild count.

08:20 UTC: first combined CI fails before trials due to an introduced pinned-
cache launcher omission (`MODEL_WEIGHTS_DIR`); TTI **f245f6ac** fixes publication
with 93 passing launcher/overlay tests. Same-image/config retry **36983437902**
is dispatched and monitored. The broader config checks expose one expected
GPQA-presence conflict with this SWE-only diagnostic branch; not hidden.
Long-context old/new-prefill teacher forcing both score 122/128 top-1; no
numerical or outcome evidence promotes the new policy. Timing differences are
JIT-cache-confounded and are not reported as precision speedups. Local devices
are closed and unowned. Active resumed time is about 169 minutes, cumulative
about **7h42m**; no further user intervention or image build.

08:32 UTC: exact pinned-tokenizer audit validates all 172 saved request counts
from the five prior candidate trajectories. Exact-output dedup saves only
0.077–0.590% of total input tokens and is not deployed. Previous-input block-
matched prefixes cover 92.68–95.22%, but real APC is still unimplemented and
must address sliding-cache/scheduler/trace correctness; these percentages are
not speedup claims. Sixteen host helper tests pass. Combined retry is monitored
on `120-qb2-p03t02`, still starting; local devices remain idle. Resumed active
time about 181 minutes, cumulative **7h54m**.

08:39 UTC: combined retry is healthy and generating on the unchanged baseline.
The offline audit identifies retained reasoning as a larger context-cost
candidate than repeated tool output. A separate local 900-second Sympy control
keeps the latest one assistant reasoning block, preserving all visible/task/
tool information and original sampling. This is explicitly a quality-sensitive
agent-context intervention, default off; TTI **d1816c8f**, 68 host tests pass.
Same image/old BFP8 policy, no source overlay or rebuild. It is not promoted to
the combined CI. Resumed active time about 188 minutes, cumulative **8h01m**.

09:00 UTC: reasoning-history limit fails Sympy (900.121-second timeout/reward0,
two repetition stops), despite lowering completed-response TTFT to131.227s;
decode/reasoning consumes643.987s. It is not promoted. After trial completion
and idle-counter verification, a separate20-minute original-history control
starts with an audited, once-only read-only diff-review submission checkpoint,
TTI **947caf78**,70 host tests pass. This targets the prior wrong submission
with restored source, not a timeout-only extension; no task-specific solution
or hidden tests enter prompts. Combined original-five CI continues unchanged
with no inference errors so far. Resumed active time~209min, cumulative~8h22m.

09:21 UTC: original-history Sympy reaches reward1 (all18 tests), but times out
at1200.028 seconds without submission. The review gate never triggers, so it
is neither a clean solve nor evidence of policy improvement. Completed-response
TTFT463.106s/post-first-token725.855s versus tools2.879s motivates a separately
declared temperature-zero control, capped900s, with all other policy unchanged.
No review/history/advisory policy is promoted; combined five-task CI remains
fixed and monitored.18 host helper checks pass. Resumed active time~230min,
cumulative~8h43m. No image build, device reset or new human intervention.

- TTFT: `doc/ttft_optimization/`.
- Remote benchmarks/evals:
  `readiness_vllm/ttft_optimization_remote/README.md`.
- Inference-server CI fixes:
  `/home/mvasiljevic/gemma4-ttft-inference-server/AUTOFIX.md`.
- TSU: `doc/tsu_optimization/closure.md` and
  `doc/tsu_optimization/work_log.md`.

Earlier pipeline closeout heads, before the eval-speed experiments: TT-Metal
`2a052971dffc0e1747a487cdf42c513a5c0dc3ab`; tt-inference-server
`6d88032ed5f8259333233c53db671cd29aad377d`.
