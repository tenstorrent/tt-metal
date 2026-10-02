# SWE evaluation speed investigation

Started 2026-10-01 UTC at the user's request: use an Astra agent to diagnose
the current eval bottlenecks, predict the attainable speed, iterate on small
measured experiments, then run and monitor actual CI. This is ongoing work,
not a claim that the five-task evaluation has been repaired.

## Current aggregate-suite continuation (2026-10-02)

The user requests the measured **original five tasks together**, then aggregate
optimization. As of **10:42 UTC**, the actual combined baseline has completed:
[CI36983437902](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36983437902),
job110763074189, runner120-qb2-p03t02, dispatched08:20:06. One persistent loaded
server, serial C1, exact original tasks,7200s per trial; reused ad58 image,
native TTc9ec/plugin c9cf, TTI**f245f6ac**, original prompt and configurable-weight
BFP8. Seed9472, async, wide repetition guard, corrected testbed PATH and the
submission adapter are declared candidate deviations. No APC/chunked prefill.
All newer local policies below are **excluded** from this fixed baseline.

**Measured dispatch-to-finish: 2h14m54s; trial span: 2h01m06.002s; agent sum:
1h58m20.457s. All five submitted without timeout, but only 2/5 passed.**
Observed order and agent seconds: Astropy1980.799 (reward1), Matplotlib2472.355
(reward0), Sympy1340.569 (reward0), Django809.311 (reward1), sklearn497.422
(reward0). This is a faster terminal result, not an all-five correctness success.
Artifacts11222185972 are retained under
`/home/mvasiljevic/gemma4-eval-speed-evidence/combined_five_36983437902`.
All task server counters match exclusive-C1 request counts/tokens; no late
responses/actions. TTFT4456.272s, post-first-token2598.299s, tools23.459s.
The63.17% TTFT fraction includes request setup/prefill, not measured pure kernels.
Matplotlib truncates source and fails imports; Sympy fails two existing
regressions; sklearn leaves the required estimator attribute absent.

The pre-dispatch terminal-event forecast was **65m /3h06m40s /5h20m**
best/expected/conservative, not a prediction of five correct solves. The
optimistic bound is exceeded; actual is51m46s below the expected terminal-event
forecast. Astropy finishes instead of timing out, while Matplotlib is much
slower than predicted. No all-five correct-solve time has been established.
The earlier startup-only attempt36981858976 failed before trials; f245f6ac
fixed the pinned-cache environment assignment with93 passing checks.

Prior bounded evidence is not a single all-five result:

| Task | Strongest relevant evidence | Limitation |
| --- | --- | --- |
| Django |845.733s clean CI, reward1,104 tests | Candidate, not original release precision |
| Matplotlib |647.041s clean local, reward1,182 tests | Local/cache state differs from CI |
| sklearn |688.289s clean CI, reward1,27 tests | Measured in a two-task serial debug run |
| Astropy |655.569s clean local, reward1,427 tests | Different focused prompt; rejected globally after Django regression |
| Sympy |1200.028s timeout, reward1,18 tests | Correct patch **without submission**, not a solve time |

Latest bounded probe: `local_sympy_old_bfp8_repeat_guarded_seed9472`, started
**10:12:50**, ends at1200.440s with reward1 but no submission. Two repeat
advisories, zero late actions, one cancelled upstream request; not promoted.
Fresh local sklearn control submits in473.752s, reward0, same missing-attribute
failure. The existing one-time generic submission-review candidate starts
10:53:02, cap900s; no task-specific hints. Local server
`gemma4-eval-history-limit` is exclusively owned by this bounded probe.

New default-off harness repairs at TTI**4fcfdcfb** stop owned container processes
before verification and cancel abandoned upstream requests.58 host tests,
a Docker delayed-writer control,20/25-second real timeout smokes and matching
post-abort native replay validate the plumbing. They are not reward or aggregate
speedup claims. Prior timeout outcomes with late source edits are flagged.
Rejected/unpromoted controls include focused prompt, new-prefill precision,
reasoning-history truncation, greedy sampling and low-yield output dedup.
Both long-context precision controls score122/128 on one saved HF sequence;
that is not broad free-generation equivalence. Prefix caching is unimplemented.

See the [suite experiment log](suite_20261002.md) for all forecasts, failures,
commands, exact provenance, raw artifact locations and live-monitor path.
The [precision audit](AUTODEBUG.md) documents numerical hypotheses, not a proven
cache/position bug. Latest TT checkpoint before this handoff is**c358bf2240**;
all scoped changes and evidence are pushed. No ten-task expansion is authorized
by the evidence yet.

## Prior checkpoint (2026-10-01 20:26 UTC)

Experimental configurable-weight BFP8 passes the existing short readiness gate
and produces a verifier-passing Django patch both locally and in actual QB2 CI.
The same-seed local selected-policy control fails at the same 900-second cap.
With a separately declared, default-off submission-marker adapter, a fresh local
trial finishes cleanly in **585.413 seconds**, reward 1, all 104 tests passing.
Its warmer cache explains most of the lower wall time; this is **not** a causal
35% adapter speedup or an 8x full-suite claim. Release precision is unchanged.

CI [36910894168](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36910894168)
is complete (reward 1, but 900-second agent timeout). Follow-up
[36916089719](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36916089719)
finishes with **native submission in 845.733 seconds**, reward 1, no exception,
all 104 tests passing and zero adapter conversions. It reuses the same image;
both CI runs were monitored through completion and their artifacts inspected.
No full five-task run has been launched; all-five release completion is unproven.
Local Matplotlib subsequently finishes **natively** in 647.041 agent seconds,
reward 1, no exception, required pickle test plus all 181 regression tests pass.
There are zero adapter conversions. Astropy's patch also passes its required test
and all 426 regression tests, but the agent hits its 900-second cap without
submission. Sympy/sklearn remain untested with this candidate. The precision
policy remains experimental. All owned local device processes are stopped, both
device files are unowned, and no dispatched CI remains running.

## Baseline and timing model

Baseline: [run 36530661132, job 109283453627](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36530661132/job/109283453627).
Artifact `workflow_logs_evals_gemma-4-26B-A4B-it_bh-qb-ge_gemma4-autoport`
(artifact ID 11055430740) is retained locally under
`/home/mvasiljevic/gemma4-eval-speed-evidence/baseline_36530661132`.
The runtime spec and server startup agree on TT image revision `919c110d`,
262144 context, C1 trials, 32 server slots, disabled prefix cache, disabled
chunked prefill, and **disabled async scheduling**. Thus the later measured
43.04 to 50.67805 tokens/s checkpoint was not used in these evals.

Use `tools/analyze_eval_timing.py ARTIFACT_ROOT --summary-only` to reproduce
the following values. Successful API durations are bounded using integer-second
`response.created` and the mini-swe response timestamp. They exclude failed,
retried, format-rejected, and still-in-flight generations. Do not equate the
remainder with tool execution. Periodic `Running: 0` is an instantaneous sample,
not a measurement of idle time; in particular, Django's earlier 18% active
estimate is contradicted by its successful request timestamps alone.

| Task | Agent seconds | Saved successful API seconds | Saved responses | Input tokens | Output tokens | Observed outcome |
|---|---:|---:|---:|---:|---:|---|
| Astropy | 7200 | 2915–2998 | 83 | 2385944 | 27044 | timeout, reward 0 |
| Django | 7200 | 5986–6310 | 324 | 3934254 | 10060 | timeout, reward 0 |
| Matplotlib | 4437 | 843–886 | 43 | 496811 | 10325 | submitted, reward 0 |
| sklearn | 7200 | 5270–5342 | 72 | 2627353 | 80438 | timeout, reward 0 |
| Sympy | 7200 | 1201–1241 | 40 | 549754 | 25030 | timeout, reward 0 |

Task setup/verifier time is separate. Matplotlib's agent phase is 73m57s;
the previously quoted approximately 74m46s includes other trial phases.
Saved token totals are 9,994,116 input and 152,897 output; pending and failed
generations are not included. The token count must not be inferred from
rounded summaries or sampled throughput.
Astropy's saved trajectory includes a final response timestamp29.35 seconds
after the recorded agent deadline, likely collected during timeout cleanup;
the script reports this outlier explicitly. Saved-response totals therefore
describe the artifact, while sampled server integration is clipped to the
agent phase. Neither is an exact completed-work count at the deadline.

Integration of the server's throughput rates over their approximately 10-second
reporting intervals reveals much more work. These are approximate counts (sample
boundary and timing error), not exact per-request usage:

| Task | Server output tokens, approximate | Saved output tokens | Approximate seconds with >20 output tokens/s |
|---|---:|---:|---:|
| Astropy | 214488 | 27044 | 4590 |
| Django | 54098 | 10060 | 920 |
| Matplotlib | 130334 | 10325 | 2830 |
| sklearn | 149649 | 80438 | 3240 |
| Sympy | 290256 | 25030 | 6220 |

Approximately 838825 server output tokens versus 152897 saved output tokens
means approximately 685928 tokens are absent from successful saved responses.
The mini-swe model records a response only after tool-call parsing succeeds;
format errors therefore also discard usage from the trajectory. Failed requests,
retries and in-flight outputs contribute too. Sympy's late logs show continuous
46–47 tokens/s decoding, disproving an interpretation of the missing 69 minutes
as tool time. Exact attribution requires request-level streaming instrumentation.

The completion milestones do not support a "nearly finished" interpretation:
Astropy is still repeatedly rewriting a patch helper for`SkyCoord.__getattr__`
and its required subclass-property test fails; Django remains in the300-command
failed search loop; sklearn's final commands contain hundreds/thousands of
repeated comments while attempting a patch helper, and its verifier also reports
pass-to-pass failures; Sympy's last saved action precedes roughly69 minutes of
unsaved generation. Matplotlib explicitly submits, but its required pickle test
still fails. These observations concern the tested implementation/trajectory;
they do not establish an inherent limitation of the HF model.

## Pre-experiment forecast

For a fixed sequence of actions, use `T = P + D + A + F`, with prefill/request
setup `P`, decode `D`, tools and harness `A`, and failed/retried generation `F`.
The current artifacts cannot uniquely identify these four terms. Streaming
replays will measure TTFT and decode at actual recorded context lengths.

The following are conditional **saved-work** predictions, not predictions that
a timed-out agent will solve its task. They use 43.04 versus 50.67805 tokens/s
for saved output tokens, hold all other work fixed, and do not extrapolate the
4K throughput result to prefill. Their expected savings are the best-supported
prior; conservative savings are zero until measured at real contexts. Best-case
serving-only limits remove all saved API time, retaining all unclassified time.

| Task | Latest image, expected seconds saved from saved decode | Conditional agent seconds after that saving | Optimistic floor after removing all saved API time |
|---|---:|---:|---:|
| Astropy | 95 | 7105 | 4202 |
| Django | 35 | 7165 | 890 |
| Matplotlib | 36 | 4401 | 3551 |
| sklearn | 282 | 6918 | 1858 |
| Sympy | 88 | 7112 | 5959 |

The floor is not a hardware forecast: some unclassified time is also inference,
and eliminating all API time is impossible. Prefix caching has a theoretical
opportunity to remove repeated prompt work, particularly Django's growing
2K–21.5K history. Its conservative benefit is zero, expected benefit is pending
measurement, and best case removes the cacheable portion of `P`, not `D`, `A`,
or `F`. Current `supports_prefix_caching=False` and the adapter's explicit
nonzero-start rejection prevent a flag-only experiment. Low-level continuation
exists but uses eager per-token decode; enabling it blindly could be slower.

An honest completion-time interval for all four timed-out tasks still has an
unbounded upper end. Django executes the identical failed search
`grep -rn "CheckConstraint" django/db/models/sql/compiler.py` **300 times**;
there is no evidence it was close to finishing. Speed alone cannot give that
trajectory a finite solve-time prediction. Sympy spends its last approximately
69 minutes without a saved successful response, with explicit API timeout
retries. Its long preceding output already repeats the same reasoning.

Agent/environment fixes have greater potential than the saved decode saving,
but no defensible numerical expected solve-time exists before a capped trial.
Combined improvements must be measured, rather than multiplying optimistic
speedups. The practical first target is: correct task interpreter, no wasted
600-second retries of valid long responses, no repeated-command loop, and a
measured reduction in real-context TTFT. The acceptance target remains the same
five tasks, sampling, output allowance, C1 topology, and 7200-second trial budget.

After accounting for server-generated rather than only saved outputs, applying
the prior 43.04→50.678 tokens/s ratio predicts approximately 751/189/456/524/1016
seconds saved for Astropy/Django/Matplotlib/sklearn/Sympy respectively, about
49 minutes total. This remains a conditional same-token-work estimate: the
baseline logs often measure 46–47 tokens/s at these contexts, which would reduce
the gain. It is not a solve-time prediction. Eliminating discarded generation
has an approximate 4-hour decode-work opportunity at 46 tokens/s, but deleting
it without recovering useful actions would not satisfy the quality criterion.

## Findings to test

1. SWE task images contain a `testbed` conda environment, but mini-swe's shell
   inherits the base interpreter. The Matplotlib trajectory installs a current
   wheel into `/opt/miniconda3/lib/python3.11/site-packages`, then edits that
   installation. The task verifier explicitly activates `testbed` and evaluates
   `/testbed`. Astropy misses `erfa`, Sympy misses `mpmath` and then encounters
   Python-version incompatibilities. Validate the environment fix without model
   calls first; it changes harness correctness, not task content or sampling.
2. Non-streaming 32768-token generations can exceed the configured 600-second
   client timeout even at the newer 50.68 tokens/s (646.6 seconds of decode
   before prefill). Existing timeout/retry lines prove the path is exercised.
   Preserve the output budget and trial timeout while testing request handling.
3. Keep policy experiments such as repetition recovery separate from serving
   and environment changes. A fast reward-zero loop is not success.

## Experiment log

- Reused existing baseline and candidate Docker images; no image rebuild.
- Verified no local device owners with `fuser` before accessing the existing
  Gemma containers. A source-backed device count probe found four Blackhole
  devices. No device profiler is used for this serving investigation, following
  the optimize/device-usage skills.
- Local baseline launch initially failed before device work because UID6002 has
  no passwd entry in the image; explicit USER/LOGNAME and writable matplotlib/
  inductor caches correct the inherited container environment.
- Real-context microbenchmark uses `tools/replay_eval_requests.py`: C1, contexts
  selected near 2K/12K/21K/32K/49K from actual trajectories, two repeats each,
  fixed128 output for timing only. These are diagnostic probes, not accuracy
  evals; final task limits remain unchanged. Full raw prompts remain local.
- Baseline replay, old image `919c110d`, sync: first-use/warmed TTFT at exact
  rendered contexts 1989/12028/21018/32155/48600 tokens was
  12.466/1.146, 20.619/7.067, 24.236/12.196, 31.036/18.581,
  and 54.991/28.643 seconds. Prompt token counts exactly reproduce the saved
  trajectory usage. Each pair returns identical generated text hashes.
- Exact Matplotlib task image `sha256:e730789910800c84441072e81399d623440a0b03838867d203e54a5f65058cb3`
  reproduces the environment bug: base import fails; task-shell PATH selecting
  `/opt/miniconda3/envs/testbed/bin` imports `/testbed/lib/matplotlib` and
  reproduces the intended weakref pickling error in approximately one second.
- TTI commit `086092b7`: task subprocess environment fix and 1800-second request
  timeout, preserving the 7200-second trial limit and 32768-token allowance.
  25 host config tests pass. Probe-only commit `ef88b3277629613787ab9da8b977baf7ff615017`
  selects Matplotlib with a 900-second cap; it must be removed for final validation.
- CI36865255386 failed in checkout because Actions interprets a short SHA as a
  branch name. Corrected to the full40-character SHA; no image rebuild.
- [CI36865475336](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36865475336)
  is the corrected capped probe, job110379903662, runner `qb2-120-p01t03`.
  Image digest `sha256:461668dba77e062db3a1dae6743f63dcb8ba1c9b5f88b582c332a5661c680e6f`,
  TT `eb1d2af61c7630c7424e9593b646093ef7be45af`, TTI `ef88b3277629613787ab9da8b977baf7ff615017`,
  plugin `c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42`, async scheduling enabled.
- New-image real-context replay used existing image TT `c9ec3469f1b875e7e5e505660c4421e5126e8dad`,
  digest `sha256:ad58effd178b9d8c7689a392159532d30aea0c1fe24bec82c06dbc0d3ebf4bbd`.
  Its additional shared-batch changes do not alter the tested C1 model path.
  All ten output hashes match old-image baseline, new async, and new sync.
  Warmed same-image results (seconds):

  | Context | Sync TTFT | Async TTFT | Sync 127-token decode span | Async decode span | Decode throughput gain |
  |---:|---:|---:|---:|---:|---:|
  | 1989 | 1.153 | 1.435 | 2.937 | 2.500 | 17.49% |
  | 12028 | 7.076 | 7.067 | 2.658 | 2.500 | 6.29% |
  | 21018 | 12.209 | 12.189 | 2.650 | 2.522 | 5.11% |
  | 32155 | 18.584 | 18.569 | 2.681 | 2.543 | 5.41% |
  | 48600 | 28.644 | 28.618 | 2.707 | 2.568 | 5.41% |

  These are short warmed measurements, not full-trial speedups. First-use TTFT
  varied substantially, but disk kernel-cache warmth differs between launches;
  it does not support a scheduler-only cold-TTFT claim. A consecutive-context
  replay tests the gap between these warmed times and baseline API durations.
- Request telemetry captures all non-streaming responses, including no-tool
  responses mini-swe rejects. The default-disabled proxy preserves response
  bytes, HTTP status, auth forwarding, sampling and tool history; logs contain
  timings, usage and hashes, not prompts, output text or credentials.
- Explicit agent-policy experiment: append a generic warning after three
  identical failed tool actions/results. Do not block a tool, drop history,
  change sampling or alter output limits. A control replay of the recorded
  12028-token Django context repeats the failed compiler.py search. The exact
  implemented warning instead searches the correct constraints.py class
  definition in 10.92 seconds (142 output tokens versus control 30). Enabling
  thinking separately produces a broader search (191 tokens). These are only
  next-action diagnostics, with no tools executed and no solve/reward claim.
- 32 host tests pass for harness config, telemetry HTTP success/error fidelity,
  secret-free logs, feedback opt-in behavior, history/sampling preservation,
  and repetition detection. [Django probe CI36867491629](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36867491629)
  pins TTI `fd2bffeb7dbcff8f73a3d542a7dbafa1f8038195`, the same existing CI image,
  one C1 task, 900-second diagnostic cap, and repetition feedback after three
  failures. It is not the final five-task release configuration.

## Forecast updated from real contexts

For the **same amount of recorded work**, the following uses approximate server
output counts, 46.5 baseline output tokens/s, and a 5.5% expected decode-rate
gain. The optimistic serving case extrapolates the measured short-context
17.735% gain; it is not expected at the long contexts. Conservative wall time
is unchanged. Prefix columns assume a hypothetical correct cache eliminates
repeated saved-prompt work at approximately 1700 input tokens/s, retaining at
least one maximum-size prompt. They omit cache-management cost and therefore
are opportunity ceilings, not implemented speedups.

| Task | Conservative seconds | Expected async seconds | Optimistic async seconds | Ideal saved-prefix seconds removable | Expected async + ideal prefix seconds |
|---|---:|---:|---:|---:|---:|
| Astropy | 7200 | 6960 | 6505 | 1375 | 5585 |
| Django | 7200 | 7139 | 7025 | 2302 | 4838 |
| Matplotlib | 4437 | 4291 | 4015 | 276 | 4015 |
| sklearn | 7200 | 7032 | 6715 | 1478 | 5555 |
| Sympy | 7200 | 6875 | 6260 | 305 | 6570 |

The four 7200-second rows are censored workload horizons, **not successful
completion predictions**. A smaller number says how fast that same unsuccessful
work could run, not that the task would finish. Current prefix-cache expected
gain is zero because its required continuation contract is unimplemented.
Agent/environment changes have zero guaranteed benefit and no identifiable
expected solve time before outcome-bearing trials. Their optimistic opportunity
is to avoid the observed failed environments, 300 repeated Django commands and
approximately 686K discarded generated tokens; assuming those vanish and all
tasks solve would be unsupported. Combining that unknown with serving gains
does not produce a defensible numerical solve-time estimate.

Baseline tool completion timestamps follow subprocess execution in mini-swe
2.2.8. Saved action durations total only 2.57/1.38/9.72/7.34/4.52 seconds for
Astropy/Django/Matplotlib/sklearn/Sympy respectively. This does not count trial
setup or verifier builds, and does not cover an unrecorded final action. It
rules out test/build acceleration as the explanation for most of these agent
phase timeouts. The practical target is useful completed actions and reward
within the original 7200 seconds, not more tokens or more repeated commands.

## Growing-context control and reproduction

`tools/probe_context_sequence.py` uses ten consecutive recorded Django requests
(12028–12550 input tokens), the actual chat/tool-parser endpoint, native stop,
unchanged 32768-token allowance and sampling, and a diagnostic seed. All twenty
calls across first/repeated sequences emit the same failed grep command, exactly
30 tokens each. Total request time is 168.17 seconds first use and 80.13 seconds
warm; median 18.42 versus 8.00 seconds. The compiled-kernel directory gains 2999
ELF files during the first sequence and zero during the second. This supports
substantial first-use compilation/setup cost; it is **not** an optimization or
a successful agent loop. Cache prewarming would have to include its own cost,
and persisted caches would need an independently measured reuse experiment.

Compact raw evidence is retained beside this document:
`baseline_replay.json`, `async_replay.json`, `sync_replay.json`,
`django_next_action.json`, `sequence_first_use.json`, `sequence_warm.json`.
Percentages are point estimates from these bounded probes, not confidence
intervals. Local server logs, launch intents and runtime caches remain under
`readiness_vllm/eval_speed/` and are intentionally not committed.

From the tt-metal root, with an exclusively owned running server:

```bash
python3 models/autoports/google_gemma_4_26b_a4b_it/tools/replay_eval_requests.py \
  /home/mvasiljevic/gemma4-eval-speed-evidence/baseline_36530661132 \
  --output /tmp/gemma4-replay.json
python3 models/autoports/google_gemma_4_26b_a4b_it/tools/probe_context_sequence.py \
  /home/mvasiljevic/gemma4-eval-speed-evidence/baseline_36530661132 \
  --output /tmp/gemma4-sequence.json
```

Server launcher: `tools/ttft_server.py --output OUTPUT_DIR`, optionally
`--no-async-scheduling` for the matched scheduler experiment and `--tool-calls`
for the chat replay. It records the full launch command. It preserves full262144
context, 32 server slots, C1 client probes, disabled prefix/chunked prefill,
selected precision, pinned HF revision, and sample-on-device `all`. The local
existing image's Python is `/home/container_app_user/tt-metal/python_env/bin/python`;
run from `/tmp` to use that image's matching TTNN installation. No model source
overlay or image rebuild was used in these measurements.

CI dispatch uses workflow342177897 in `tenstorrent/tt-agentic-bringup-qb2`,
model `gemma-4-26B-A4B-it`, runner `bh-qb-ge`, device `p300x2`, workflow `evals`,
implementation `gemma4-autoport`, full immutable TTI/TT/plugin SHAs and the image
digest recorded above, with AI summaries and issue comments disabled. Both
build jobs are skipped when reusing the image. Probe overrides live only on
TTI branch `mvasiljevic/gemma4-eval-speed` and must be removed before a final
full-subset comparison.

## First outcome-bearing CI result

Matplotlib CI36865475336 completed at approximately13:37 UTC. Workflow success
does **not** mean task success: the trial reached the deliberate900-second cap
with reward0. The corrected task environment is active. The agent correctly
reproduces the weakref error using `/testbed/lib/matplotlib`, inspects the real
Grouper implementation, and records25 actions/2057 output tokens in about303
seconds. It then spends approximately597 seconds without another completed
action. Integrated server output is approximately31034 tokens, so most of this
probe's generation again never becomes a saved action. The timeout is not
evidence that it was about to solve. Artifact11167046842 is retained under
`/home/mvasiljevic/gemma4-eval-speed-evidence/matplotlib_probe_36865475336`.

CI timing also exposes an iteration bottleneck outside the trial: server launch
13:02:16, healthy13:12:12, background warmup done13:20:31, agent start13:20:43.
The external warmup sweep includes130944-input-token requests despite the
15-minute diagnostic trial. TTI commit `f185d2ce8365d35a696f21f36b3b5083fa5b2357`
adds a validated optional `background_trace_context_lens` override and selects
only4096-input/4-output tokens for the next diagnostic. This does not alter
model context or trial limits. Expected startup saving is most of the observed
499-second external sweep, but the next CI must measure it; model load remains.
78 relevant host tests pass.

The next Matplotlib probe separately enables the canonical Gemma thinking and
reasoning-parser configuration. This is a declared policy change, not a serving
speed claim. On the already-degenerate42K Sympy context, neither thinking alone
nor a generic interrupted-generation warning produces a completed action within
120 seconds: both continue repetitive comments inside a bash argument. That
recovery hypothesis is rejected for the entrenched context.

## Discarded-generation diagnosis and bounded guard

Django CI [36867491629](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36867491629)
also reaches the 900-second diagnostic cap with reward 0. Payload-free telemetry
records 20 request starts and 19 responses. After 18 short tool responses, one
request takes **668.9739 seconds**, emits **32768 tokens**, finishes with `length`,
and supplies **zero tool calls**. The next request is pending at timeout. The
failed-command feedback never fires in this fresh trajectory; its earlier
next-action replay is not an end-to-end success. Artifact11168053915 is retained
under `/home/mvasiljevic/gemma4-eval-speed-evidence/django_probe_36867491629`.

The existing vLLM image already implements native output-token repetition
detection. A request can explicitly select `repetition_detection` with
`min_pattern_size=16`, `max_pattern_size=128`, and `min_count=8`. This changes
termination policy, not sampling probabilities or the 32768-token allowance,
and must be reported separately from serving optimization. On the entrenched
Sympy context it stops after 803 tokens/42.83 seconds with finish reason
`repetition`; its text is an exact prefix of the unguarded 120-second capped
output. Cache state differs, so this is **not a measured 2.8x latency speedup**.
Normal Django output remains the identical 30-token tool call. After standard
no-tool feedback, Sympy again repeats and stops at 945 tokens/53.45 seconds:
containment does not by itself restore useful reasoning. Compact evidence:
`sympy_native_repetition.json`, `django_native_repetition.json`, and
`sympy_format_recovery.json`.

A fresh local Matplotlib Harbor trial tests integration, using the exact pinned
dataset and Harbor commit, existing local async image, C1, engine seed9472,
corrected testbed environment, unchanged sampling/output allowance, and a
900-second cap. It finishes with reward0/AgentTimeoutError. HTTP telemetry has
69 responses: 68 tool-call responses and one native repetition stop at
1054 tokens/38.34 seconds. The agent resumes valid tools after that stop.
The verifier applies the produced patch but the required test still fails.
There are 67 saved assistant actions, 6051 saved output tokens, and 4.28 seconds
of saved tool execution. Fourteen exact commands repeat, including successful
read-only inspection, which the failed-command feedback intentionally does not
classify as an error. One saved response occurs 28.59 seconds after the agent
deadline during cleanup; artifact totals must not be equated with the exact
900-second execution interval.

This local trial is diagnostic, not a matched speed/reward comparison: it uses
the local c9ec image, warm persisted kernel caches, and no external warmup
sweep. Its unseeded requests are not deterministic copies of CI. It uses Harbor
`1da0bfd8c71cadbff17413fac984b8e391d2afc2` in a separate Python3.12 environment;
transitive host dependencies can differ from CI. Artifacts are under
`/home/mvasiljevic/gemma4-eval-speed-evidence/local_matplotlib_guard`.

Thinking-enabled CI [36870715050](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36870715050)
is the next separate policy test. It does not yet enable native repetition
detection. A concurrent **independent local server** tests thinking plus the
guard on one Django trial capped at1200 seconds. No same-server concurrent
trials and no full five-task rerun are used. Redacted response text statistics
(character counts and repeated-line counts, never response text) were added in
TTI `72185ee1` to distinguish discarded repetitive output in future probes;
104 relevant host tests pass.

`tools/audit_repetition_guard.py` performs a CPU-only retrospective audit using
the exact installed vLLM detector and pinned tokenizer. Across562 saved baseline
responses,696 parsed response fields and153286 field tokens,17 fields trigger:
Sympy message81; sklearn messages112,114,116,118,120,122,126,128,130,134,136,138,
142,144,146; Astropy message47. All contain conspicuous degenerate repetition,
including1953 identical sklearn comment lines and69 identical malformed Astropy
assignments. No legitimate repeated code was identified among these flags.
This is not an exhaustive false-positive guarantee: parsed fields do not
preserve the original raw token stream or missing responses, and a command can
contain useful work after a long repeated segment that the guard would prevent
from executing. The policy therefore still needs outcome-bearing validation.

The configuration-matched thinking-plus-guard CI probe is
[36876743431](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36876743431),
job110418160465, TTI `35a0fd50e01b8f335761b396149d090f0ce47455`.
It reuses the preceding CI's image, seed, task,900-second cap and short warmup;
native repetition termination and payload-free response statistics are the
declared differences. No new image was built. Both actual CI runs are monitored
through their job state and read-only server metrics.

## Additional fast hypothesis checks

Exact repeated-tool-output compaction keeps the first complete output and
replaces later copies only when both the command and complete output are
identical, with a minimum256-character output. Five host invariants cover
input immutability, first-copy retention, changed commands, changed outputs,
short outputs and unmatched messages. The tokenizer-only experiment finds
**zero eligible savings in the five baseline terminal contexts**. On the fresh
local Matplotlib trajectory,13 repeated outputs reduce the final tested context
from24294 to20647 tokens (15.0%). At the measured approximate1700 input tokens/s,
that suggests about2.1 seconds less warmed prefill for this specific next turn,
not a measured serving or solve-time gain. No live agent compaction is enabled;
next-action quality remains untested. Evidence: `local_context_dedup.json` and
`tools/probe_context_dedup.py`.

Coarse whole-prompt padding is not a configuration-only cache optimization.
The scheduler allocates KV pages for the actual prompt; padding to1024-token
buckets can write beyond those allocated pages unless the model's valid-length
and padded-query contracts are extended correctly. Selecting the last padded
token's logits is also incorrect. No such unvalidated padding was enabled.
Similarly, vLLM exposes a thinking-token budget, but the installed TT plugin has
no matching thinking-budget state integration; its existence in the HTTP schema
alone is not evidence that the device-sampling path enforces it.

## Measured diagnostic-startup improvement and remaining failures

Thinking-only CI36870715050 completes with reward0 at900.10 seconds. It has
20 valid tool responses,3341 saved output tokens and2.16 seconds of saved tool
execution. Its final529.70 seconds have no completed action while server
generation continues. Thinking alone is therefore rejected as a sufficient fix.
Artifact11170236882 is retained under
`/home/mvasiljevic/gemma4-eval-speed-evidence/thinking_probe_36870715050`.

The **diagnostic-startup change is measured in actual CI**. Using the same
existing image, healthy-to-background-warmup-complete changes from498.978 seconds
(13:12:10.701–13:20:29.679) to104.470 seconds
(14:22:58.443–14:24:42.913): **394.508 seconds saved,79.1% less warmup time**.
The candidate's single4K request itself takes100.091 seconds. Both logs report
590.5 seconds of model startup, which this change does not improve. The scoped
warmup makes diagnostic iterations faster; it does not demonstrate faster task
completion and changes which shapes have been precompiled before the trial.

The independent local Django thinking-plus-guard trial reaches its1200-second
cap with reward0. It has22 HTTP responses:14 valid tool calls,7 native repetition
stops, and one32768-token length stop. Saved tool execution totals0.62 seconds.
The escaped response takes670.393 seconds; payload-free statistics identify
115164 characters,1722 long lines but only95 unique long lines, with91.6% of
characters in repeated lines. The128-token/eight-repeat detector does not cover
every repetitive pattern. Seven other guarded responses total273.73 seconds;
they recover control but do not yield a correct patch. This configuration is
not a validated solution to the task timeouts.

A focused next test widens only the native detector's maximum pattern size
from128 to1024, retaining eight repeats and the32768-token output allowance.
A CPU-only audit flags the same17 baseline fields at the same first-stop
positions, adding no observed false positives in that finite saved-response
sample. Matched bounded replays use the escaped Django context, which tokenizes
to exactly14467 tokens, matching its actual HTTP usage. No claim about the
wider guard is made before that experiment and outcome validation.

The matched wider-window replay subsequently stops at4155 tokens in91.651
seconds, while the128-window control is still generating at its180.012-second
diagnostic cap. Their TTFTs are8.622 and8.626 seconds, and the candidate text is
an exact prefix of the control text. This is at least88.36 seconds of avoided
pathological generation by the observation horizon, not a task solve-time
speedup. Evidence: `wide_repetition_comparison.json`; full texts remain in the
untracked readiness artifact directory. The unchanged sampling policy is
temperature1/top-p0.95/top-k20/seed9472/max32768.

Wider-guard CI
[36880038816](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36880038816),
job110429390062, uses TTI `0a2b40446aa7f784b1e0225fbef13459e80fb5d9` and the same
existing eb1 image. It obtains a second runner,120-qb2-p03t02; this is parallel
debugging on separate C1 servers, not same-server concurrency or a final
matched-host release comparison. The128-window trial remains on qb2-120-p01t03.

A bounded HF greedy next-action reference is also attempted in a CPU-only,
network-disabled container using the existing local image,12GiB memory limit,
four CPUs and a1200-second external timeout. It uses the exact pinned model and
prompt token hash1585459ea18f529e1f6d3babdfaf74c642665284481434a2c17e05dcab6ee13e.
The HF prompt must normalize tool argument strings to JSON objects and message
contents to OpenAI text parts, matching the server's native template path;
plain string rendering differs by one token. The hash check gates model loading.
This is a greedy128-token qualitative control, explicitly not the scored
sampling policy or a CPU-versus-TT performance comparison. Initial host import
and template-shape errors were corrected before any model execution; no failed
reference is treated as evidence against model quality.

A generic execution-policy warning was tested on the same Django history:
use the already-activated testbed, keep bash commands executable and concise,
and avoid repeated commentary/unchanged inspections. With the wider guard, it
still produces no completed action and stops for repetition at1406 tokens.
This is rejected as a quality recovery, not promoted because its useless
response happens to finish sooner. The warning is not enabled in CI.

A focused stale-mode check serves the same14467-token greedy prompt, then a
different14549-token sampled prompt, then the original greedy prompt again.
The two128-token greedy completions match exactly (SHA256
708f9608460d66ac4aeff5e4c1ccbd4ef473da9b3fd43548fef469daa4ce3b7c).
This does not reproduce a stale sampling-mode/request-state failure in that
bounded case. It does not prove full long-generation correctness or identify
the cause of the observed repetitive output. The CPU reference co-runs during
the second request, so their latency difference is not a performance claim.

## Narrow-guard CI outcome and seed contract

CI36876743431 completes at900 seconds with reward0:31 request starts,30 HTTP
responses,27 valid tool responses and3 repetition stops. It produces25382 output
tokens; its longest guarded response is12613 tokens/269.061 seconds. Artifact
11170759518 is retained under
`/home/mvasiljevic/gemma4-eval-speed-evidence/guard128_probe_36876743431`.

The initial thinking-only and narrow-guard responses differ **before any guard
fires**, despite equal1154-token prompts and the same engine seed9472. Scored
requests do not specify their own seed. In `tt/generator.py`,
`_reset_sampling_seeds` uses`secrets.randbits(63)` when a request seed is absent;
the engine seed does not make those draws reproducible. Thus these are
configuration-matched independent trajectories, not paired request-by-request
trials.27 versus20 saved actions cannot be attributed causally to the guard.
The local microreplays explicitly supply request seed9472 and retain matching
output prefixes, which is why their bounded causal claim is stronger. Future
paired diagnostics should explicitly pin request seeds and declare that change;
the current scored CI policy has not silently been changed to do so.

## Wider-guard outcome and bounded reference

CI36880038816 finishes at15:23:43 UTC with an agent timeout at900 seconds and
reward0. Its telemetry records29 starts,28 responses,21 tool responses and7
repetition stops,19789 generated tokens and a longest completed request of
90.138 seconds. Saved valid actions account for4481 tokens; discarded responses
still matter. This independent stochastic trajectory does not establish a
task-level speedup or quality recovery. Artifacts are retained under
`/home/mvasiljevic/gemma4-eval-speed-evidence/guard1024_probe_36880038816`.
Both narrower and wider guards therefore remain experimental, not a validated
release-quality fix. All dispatched CI runs are now completed and inspected.

The HF control completes successfully at15:14:20 UTC:2.421-second mmap load,
1001.174-second CPU generation,128 greedy tokens. Both HF and TT begin coherent
recaps; neither128-token sample completes an action. This does not identify the
cause of long repetitive generations or prove a precision failure. The run used
12GiB memory and four CPUs. A16GiB limit update happened after process exit and
did not affect the result. Raw completion/token evidence is in
`readiness_vllm/eval_speed/hf_django_prefix.json`. New progress-streaming support
was also added after this run, so it is not retroactively attributed to it.

A further local diagnostic raises only BFP4 weight fields to BFP8, leaving all
activation, KV, CCL and compute-fidelity policy fields unchanged. The complete
policy and93-field manifest are generated by`tools/prepare_eval_weight_control.py`;
candidate SHA256 is46389c08f1c99f068669f66e902cc34daf00e54e7c7d8014dc1539b3dd1af954.
It is an isolated read-only bind mount over the existing image's selected-policy
path, not a repository default change or image rebuild. Full262144 context and
the same serving sampling remain intact. It is explicitly unselected and must
pass full datatype-sweep accuracy/readiness/qualitative gates before adoption.
The first launch fails before model execution because the image's profiler
import unconditionally creates an unwritable trace directory; an owned tmpfs
at that exact directory resolves the launch permission issue without enabling
profiling or modifying runtime source.
The first96-field candidate is rejected by the schema because prefill precision
fields are fixed. The actual93-field control leaves those fixed prefill fields
unchanged and raises only configurable decode/head weights. This is not an
all-BFP8 end-to-end model; no validator is weakened to permit it.

## User-requested stop checkpoint — 2026-10-01 15:30 UTC

The user requested immediate shutdown. No further experiments or CI dispatches
are authorized by this continuation. All five actual CI probes and the earlier
checkout-only failed dispatch are completed and inspected; no CI job remains
running. The final wider-guard job
is110429390062/run36880038816, workflow success but task timeout/reward0.
The owned local container`gemma4-eval-weight-bfp8-v2` is stopped during model
loading. It reached layer5 in the last inspected progress log; no request,
accuracy, speed or quality measurement was obtained for that policy. Its
configuration remains unselected. HF and previous local serving experiments
had already exited; no local probe/monitor script remains active.

Measured delivered improvements are warmed async decode at real12K–49K
contexts (approximately5–6%, identical bounded output hashes) and394.508 seconds
less diagnostic warmup (79.1%). These are not solved-eval claims. Guards bound
some pathological requests but every outcome-bearing probe still has reward0.
The full five-task7200-second release suite was deliberately not rerun.

Resume only on a new user request. Start with the retained artifact directories
and this report; do not rebuild images merely to rerun. The unfinished precision
diagnostic requires runtime-policy attestation and short matched replay before
any accuracy/qualitative adoption gates. Large local artifacts remain under
`readiness_vllm/eval_speed/` and`/home/mvasiljevic/gemma4-eval-speed-evidence/`;
they are intentionally not added wholesale to git.

## Resume — user revoked the stop request

The user explicitly authorized continuation after checkpoint0fdb72b2da. Resume
starts with exclusive-device checks and a bounded mesh open/close, then the same
isolated configurable-weight control. The previous stop was user-requested, not
a model failure: Docker's20-second grace expired and the container exited137 at
15:28:53 UTC; both device files were subsequently unowned. The image's standalone
tt-smi wrapper lacks pyluwen, so that utility error is not classified as hardware
failure. No new image or full five-task CI run is needed for this control.
The bounded mesh check passes on all four chips at18:21:19 UTC without reset;
the initial probe needed its log directory pointed at the owned artifact path.
The policy-isolation host test passes: reversing exactly93 configurable weight
changes recovers the original complete policy, including fixed prefill fields.

The wider-guard CI initial prompt differs from the narrow-guard prompt only in
the harness-provided kernel/version string in`system_information`, explaining
1153 versus1154 input tokens on the two hosts. This adds another reason not to
treat those independent trajectories as paired causal evidence. Exact replay
controls retain the saved prompt and explicit request seed.

### Resumed configurable-weight control

The same image initializes all30 layers and the full262144-token serving
contract. Construction calls`precision_summary()`, which validates allocated
weight dtypes and bound compute configs against the mounted complete policy.
Here “fixed prefill unchanged” means the fixed per-layer prefill policy, not
identical full-prefill logits: the configurable output head is shared by prefill
and decode, and its BFP4-to-BFP8 change affects both phases.
Observed`/server_info?config_format=json` confirms async scheduling and on-device
sampling. No profiler is enabled on the live serving path.

The first14,467-token greedy128-token request has cold TTFT129.104 seconds and
2.453-second decode. It is compilation evidence, not warmed speed. The subsequent
fixed-seed sampled request finishes naturally after1100 tokens in31.701 seconds
(TTFT9.041, decode22.618). The selected policy on this exact context/seed with
the same1024 guard previously produces4155 tokens in91.651 seconds and stops
for repetition. The candidate writes an incomplete, comment-heavy reproduction
script: reduced pathological generation is observed, but solve-quality is not
established. Native chat parsing repeats the1100-token response in31.537 seconds
with finish_reason`tool_calls` and one valid bash call; no tool is executed by
these replays.

The official six-prompt suite is replayed C1 with thinking explicitly disabled,
matching its existing HF and selected-policy prompt format. All six outputs
are coherent: haiku, learning explanation, story, thermodynamics, French
translation and Fibonacci code. No visible degeneracy appears. Exact-text
equality against the selected policy fails, as expected for this deliberately
different precision; the harness retains that failure rather than turning it
into a quality pass. Longer responses end at the suite's256-token diagnostic
allowance. This small qualitative check is not the full readiness accuracy gate.

At18:32:53 UTC one fresh900-second Django trial starts against this local
candidate. It explicitly supplies request seed9472, guard1024 and the existing
task/sampling/max-output settings. Per-request seeding is a declared paired-
diagnostic policy change, not a scored release default. Artifacts:
`/home/mvasiljevic/gemma4-eval-speed-evidence/local_django_weight_bfp8_seed9472`.
No full release suite or new CI image is launched on this evidence alone.

Short replay tools now enforce a real wall deadline across blocked first-token
and stream reads, preserving partial evidence on expiry. Seven host invariants
pass, including timer restoration and complete policy isolation. A client abort
does not reset or forcibly terminate the serving engine.

At18:48:18 the capped Django trial finishes with reward1 and`resolved=true`:
the required`test_simplecol_query` and all103 existing regression tests pass.
The agent nevertheless reaches900.020 seconds without clean submission, so its
`AgentTimeoutError` remains. It produced and locally checked the source fix
before the deadline; this is a verified patch-quality improvement, not proof
of clean agent completion or a paired8x speedup over the old failed7200-second
run. The explicit request seed and other previously disclosed differences
prevent attributing all improvement to precision. The next gates are the existing
100-token traced readiness reference and a same-host selected-policy seeded
trial, followed by scoped CI validation. The full five-task run remains gated.

The candidate subsequently passes the existing100-token readiness reference:
prefill top1=96%, decode top1=98%, top5/top100=100% in both phases. Traced
teacher-forcing counters verify99 model/sampling replays, with50.336 warmed
teacher-forcing tokens/s. This short161-token-input reference is not long-context
SWE accuracy or a serving-throughput headline. Its hash is
9b792e83314a35e9619e58434f11f4d3c4b1e1edb9f18d14f1799495ff7c4e69.
The full262144 serving allocation had already succeeded separately. The model
default remains unchanged while a selected-policy seeded comparison is pending.

TTI530d9b17 adds an inactive dev-only`autoport_precision_config` metadata hook:
an explicit read-only policy mount, source restricted to`reference_config`,
target derived from one autoport implementation, and SHA256 logged. Symlink
escapes, non-dev use, absolute/out-of-tree paths and invalid policy headers are
rejected. The relevant113-test suite passes. This permits an honest explicit
runtime-policy override while reusing an immutable CI image; it is not a hidden
rebuild or a claim that the image's original selected policy has changed.

Scoped CI[36910894168](https://github.com/tenstorrent/tt-agentic-bringup-qb2/actions/runs/36910894168)
is dispatched at18:58:56 UTC: one Django900-second trial, request seed9472,
guard1024 and explicit unselected BFP8 configurable-weight overlay. It reuses
the exact local image digest`sha256:ad58effd178b9d8c7689a392159532d30aea0c1fe24bec82c06dbc0d3ebf4bbd`,
TT-Metal`c9ec3469f1b875e7e5e505660c4421e5126e8dad`, transport/plugin input
`c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42` and TTI
`3546715680015285dcf8e7fb136df7f8c9d581e0`. No image build is requested.
The same-host selected-policy900-second seeded control starts separately after
server readiness, under`local_django_selected_seed9472`. Independent servers
are used; there is never more than one live trial per inference server.

### Same-seed outcome control and submission diagnostic

The selected-policy control finishes at 19:15 UTC with reward 0 and
`AgentTimeoutError` after 900.007 seconds. Its required `test_simplecol_query`
still fails; all 103 PASS_TO_PASS tests pass. Both local trials have identical
initial request bytes, SHA256
`19da7e61b624431d250cb4bc54bcb0e03ae3d18f34e41389c858298ab288435e`,
same host, image, seed 9472, task and sampling settings. Only the explicit
precision policy differs intentionally; cache state differs, and one seed is
not a population-quality estimate.

| Local 900-second control | Selected weights | Configurable BFP8 weights |
| --- | ---: | ---: |
| Verifier reward | 0 | 1 |
| Clean submission | No | No |
| Recorded completed responses | 43 | 37 |
| Native repetition stops | 14 | 0 |
| Recorded output tokens | 28,787 | 12,118 |
| Recorded prompt tokens | 366,553 | 608,044 |

The selected-policy totals include one 193-token response recorded 15.295
seconds after the harness's agent-end timestamp. In-budget completed responses
are 42 / 28,594 output tokens. Another request starts after the nominal agent
deadline but has no recorded response. Do not divide the untrimmed API-duration
sum by the 900-second budget and interpret it as utilization. Candidate has
three no-tool natural-stop responses after its last valid action. The candidate
uses more prompt processing but substantially less discarded decode; reward
improves at the same cap, not by extending it. This is not a clean-completion
speedup measurement.

A post-trial replay exposed a diagnostic-helper omission: saved assistant
`reasoning_content` was discarded. On the candidate's 70-message completed-patch
context, omission shortens 26,142 tokens to 19,475. The native replay helper now
preserves both supported reasoning fields, with a host regression test (nine
host tests pass). Actual Harbor trials were unaffected. The corrected selected-
precision replay sees the original 26,142 tokens and emits the proper submission
bash call in 20.957 seconds / 138 output tokens. The earlier stripped replay
instead asks to re-read a file and is not a matched original-context experiment.
No tool is executed by either replay. A restarted candidate server will test
the same complete context; no release-policy change follows from this alone.

Reproduction commands (repository root; server runs in the retained exact-image
container with the read-only policy mount recorded above):

```bash
python3 models/autoports/google_gemma_4_26b_a4b_it/tools/prepare_eval_weight_control.py \
  --source models/autoports/google_gemma_4_26b_a4b_it/doc/datatype_sweep/selected_precision_config.json \
  --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/eval_speed/weight_control/precision.json
OMP_NUM_THREADS=8 python -m models.autoports.google_gemma_4_26b_a4b_it.tests.run_datatype_candidate \
  --config models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/eval_speed/weight_control/precision.json \
  --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/eval_speed/weight_control/readiness.json --repeats 1
PATH=/home/mvasiljevic/gemma4-eval-speed-evidence/docker_cli:/home/mvasiljevic/.local/bin:$PATH \
/home/mvasiljevic/gemma4-ttft-inference-server/.venv/bin/python \
  models/autoports/google_gemma_4_26b_a4b_it/tools/run_local_eval_probe.py \
  --tti-root /home/mvasiljevic/gemma4-ttft-inference-server \
  --harbor-python /home/mvasiljevic/gemma4-eval-speed-evidence/harbor_venv/bin/python \
  --source-config /home/mvasiljevic/gemma4-eval-speed-evidence/local_django_thinking_guard/local_django_thinking_guard_harbor_config.json \
  --output /home/mvasiljevic/gemma4-eval-speed-evidence/local_django_selected_seed9472 \
  --task django__django-11299 --seconds 900 --request-seed 9472 \
  --repetition-detection '{"min_pattern_size":16,"max_pattern_size":1024,"min_count":8}'
```

Use a fresh output directory for repetitions. The candidate trial uses the same
last command against the candidate server with output directory
`local_django_weight_bfp8_seed9472`. Server argv, environment overrides and source
hashes are retained in `weight_control/server/launch.json` and
`weight_control/selected_seeded_server/launch.json` under the untracked artifact
directory. Never run the readiness process concurrently with a device server.

The candidate's corrected 26,142-token replay reproduces the original
724-token natural-stop/no-tool signature and reveals a final plain-text line
`echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, but no native tool call. It takes
32.084 seconds on the restarted server; the prior trial took 41.849 seconds for
that response. This cache-state timing difference is not a delivered speedup.
The native parser is not discarding a valid wire-format call: the model prints
the command in ordinary final text instead of invoking it.

TTI `4eb09ae1` adds a default-off, audited `normalize_submission_marker` harness
adapter. It requires exactly one assistant response, natural `stop`, no existing
tool/function call or refusal, an available bash command tool, and the exact
standalone final command line outside an unclosed Markdown fence. Only the fixed
submission command is converted; arbitrary generated shell text is never
interpreted. Original content, reasoning and usage remain unchanged. Telemetry
records original finish/tool counts and response hash, plus a separate conversion
event and forwarded hash. This is a declared harness-policy change, not a native
model/tool-format pass. It can still accept an incorrect model-chosen completion;
the ordinary external verifier remains mandatory, as for a native submission.

The actual replay response passes this adapter offline. The broader host suite
passes 140 tests after synchronizing its fake Harbor config with the existing
telemetry fields and new default-off flag (`63bb532c`). At 19:28:37 UTC a fresh
same-seed Django 900-second local trial starts with only this additional adapter,
using `--normalize-submission-marker` and output
`/home/mvasiljevic/gemma4-eval-speed-evidence/local_django_bfp8_submit_seed9472`.
The separate CI run remains unchanged and has no submission adapter enabled.

### Monitored QB2 CI result

Run 36910894168 / job 110533086359 completes successfully at 19:30:04 UTC.
Artifact **11189981342**, downloaded under
`/home/mvasiljevic/gemma4-eval-speed-evidence/bfp8_probe_36910894168`, confirms
**reward 1 / resolved true**, with the required test and all 103 regression tests
passing. Agent duration is 900.012785 seconds with `AgentTimeoutError`: correct
patch, still no clean submission. The exact overlay hash is logged at 19:02:42;
async scheduling is active and APC disabled. This is an actual monitored CI
quality result, not an inference from the workflow's green status.

This trajectory has 46 saved native tool responses, zero repetition stops,
615,491 prompt tokens and 10,056 output tokens. Tool time totals only 0.679
seconds. The successful source edit occurs about 804.6 seconds into the trial;
the reproduction then passes before the cap. The agent continues inspecting
the correct patch rather than producing a final text marker. Thus the optional
submission adapter cannot be assumed to shorten this distinct trajectory.
One final saved response arrives 9.457 seconds after the nominal agent deadline;
the fixed patch predates that tail. The local and CI initial prompts differ in
the runner's kernel/version string (1,696 versus 1,697 tokens), so these are two
distinct same-seed trajectories, not identical replay replications.

Server counters through the last completed response give approximately 698.50
seconds TTFT and 206.87 seconds post-first-token time after subtracting the
4K/4 warmup (98.323 seconds). They include the small post-deadline response tail;
they are not exact within-budget utilization. About 77% of this response path
is now TTFT, and 23% decoding. On this **fixed action/output workload**, a further
5% decode-rate gain saves only about 10 seconds; doubling prompt-processing
speed would reduce approximately 905 seconds to 556 seconds, while an impossible
zero-TTFT ceiling is about 207 seconds (4.4x response-path speedup). These are
conditional Amdahl bounds, not forecasts of solver completion, and APC cannot
be enabled by a flag. Cold compilation is part of TTFT and must be separated
from attention compute before claiming a practical caching gain. The verified
result justifies further bounded cross-task/termination checks; it does not
establish that all five tasks now finish correctly within two hours.

### Clean local completion and bounded CI follow-up

`local_django_bfp8_submit_seed9472` finishes at 19:38:29 UTC: agent time
585.412629 seconds, no exception, reward 1, required test plus all 103 regression
tests pass. Exactly one audited normalization converts the explicit final marker
to the fixed submission call. All preceding 34 work commands match the earlier
candidate trial. The first 35 upstream responses also have identical aggregate
568,688 input / 11,437 output tokens. Their API sum changes from 855.653 to 579.381
seconds, a **276.273-second cache-state difference**, not a measured intervention
in device code or an adapter speedup. The adapter avoids the later format-error
retry tail and proves clean termination on this trajectory. Other trajectories
may still over-inspect or fail to choose completion at all.

The local trial uses TTI `63bb532c` plus the helper subsequently committed in
TT-Metal `0a91168f24`; model image and precision overlay are unchanged. Subsequent
TTI `41b847e6` also rejects constrained/non-auto tool choices for normalization;
this does not affect the tested auto-tool path. All 140 related host tests pass.
Owned local servers are stopped at 19:40 UTC; both device files are unowned.

CI follow-up 36916089719 is dispatched at 19:41:02 UTC with TTI
`215e63753b1c617b1bfede127d68a25ab128377d`, the same `ad58effd...` image,
`c9ec3469...` TT-Metal and `c9cfebcf...` plugin inputs. It runs one Django trial,
seed 9472, cap 1,200 seconds, BFP8 overlay and explicit submission normalization.
The increased diagnostic cap is disclosed; it is not a matched 900-second wall-
time comparison or the final 7,200-second release recipe. No image rebuild is
requested. Monitor command:

```bash
python3 models/autoports/google_gemma_4_26b_a4b_it/tools/monitor_eval_ci.py 36916089719 \
  --server http://qb2-120-p04t07:8000 \
  --output /home/mvasiljevic/gemma4-eval-speed-evidence/bfp8_submit_ci_live_metrics.jsonl
```

The monitor records read-only job states and serving counters every 30 seconds,
handles unassigned runner names, and exits only when the run completes. Its
completed-run path is checked against run 36910894168. Artifact/verifier review
is still required after a green workflow result.

The follow-up uses job **110550407919**, Actions runner `120-qb2-p04t07`.
Its verified DNS alias is `qb2-120-p04t07` (10.32.49.17). The monitor initially
assumed a `qb2-` runner prefix and missed this job in its filtered view; checking
the unfiltered Actions job list corrected that at 19:44–19:45 UTC. Job selection
now uses the workflow's run-tests job name, with an explicit verified server
alias when needed. The run was executing, not waiting for a runner.

At 19:46 UTC the owned local BFP8 server is restarted for one parallel, separate-
server Matplotlib generalization probe: task `matplotlib__matplotlib-25332`,
900 seconds, seed 9472, same image/precision/guard and explicit submission
normalization. Output is
`/home/mvasiljevic/gemma4-eval-speed-evidence/local_matplotlib_bfp8_submit_seed9472`.
This is a cross-task outcome check, not a matched speed comparison with the
earlier unseeded Matplotlib failures. There is still only one trial per server.

Matplotlib finishes at 20:00:28 UTC with native submission, agent time
647.041329 seconds and reward 1 / no exception. Required
`lib/matplotlib/tests/test_pickle.py::test_complete[png]` and all 181 PASS_TO_PASS
tests pass. There are 35 responses, 368,062 prompt / 13,589 output tokens,
3.529 seconds in tools, zero repetition stops and zero adapter conversions.
The policy improves observed outcome versus previous failed Matplotlib probes,
but explicit seed, cache and earlier setup differences prevent a paired causal
speedup percentage. The baseline's 4,437-second incorrect submission is not a
time-to-correct-solution baseline.

The same already-loaded local server starts one 900-second Astropy-14096 trial
at 20:03:07 UTC (`local_astropy_bfp8_submit_seed9472`), keeping seed 9472, guard
1024 and the opt-in adapter. This avoids another model load and remains a
separate-server C1 test. Meanwhile the remote follow-up server becomes reachable
at 20:00:57 and completes a 98.167-second warmup around 20:03. Thus the DNS alias
is confirmed by its model metrics; exact attribution of its longer startup
awaits the final CI logs.

### Completed CI and Astropy checkpoint

Follow-up CI **36916089719 / 110550407919** completes at 20:17:54 UTC. Artifact
**11191289165** is downloaded under
`/home/mvasiljevic/gemma4-eval-speed-evidence/bfp8_submit_36916089719`.
Trial `django__django-11299__5fh38H3` runs from 20:03:07.645288 to
20:17:13.378679: **845.733391 agent seconds, reward 1, resolved, no exception**.
The required `test_simplecol_query` and all 103 PASS_TO_PASS tests pass. All 38
responses contain native tool calls, including the final submission; the
enabled adapter converts zero responses. There are no repetition stops or late
responses. Recorded totals are 466,648 input / 12,030 output tokens,
840.273 proxy API seconds and 1.742 tool seconds. This is a clean, actual QB2
outcome within even the earlier 900-second cap, not just a green workflow.

The policy hash remains `46389c08f1c99f068669f66e902cc34daf00e54e7c7d8014dc1539b3dd1af954`,
logged at 19:44:08.524. Image, TT-Metal, plugin and TTI provenance are the exact
inputs recorded above. The job builds no image. End-to-end dispatch-to-workflow
completion is **36m52s**, much longer than the 14m06s agent phase: the uncached
HF snapshot fetch alone takes approximately **386.5 seconds** (19:44:11.708 to
19:50:38.172), then the startup health poll measures **620.6 seconds**, and the
4K/4-token diagnostic warmup takes **98.167 seconds**. Model/cache creation and
startup are outside the per-agent cap. Reusing images does not ensure that a
different runner has HF weights, converted tensors, or compiled kernels cached.
Persistent exact-revision caches and staying on the loaded server are practical
iteration priorities; no cache-transfer speedup has yet been measured.

Endpoint counters before and after this complete trial isolate **592.468298
seconds TTFT** and **247.685461 seconds post-first-token latency**, total
840.153759 seconds. Proxy/API bookkeeping accounts for approximately 0.119
seconds above that total. TTFT is **70.5%** of request time. On this fixed
trajectory, another 5% decode speedup saves only about 11.8 seconds; halving TTFT
would reduce agent time to approximately **549.5 seconds**. The impossible
zero-TTFT floor is approximately **253.3 agent seconds** (3.34x ceiling relative
to this successful trial), retaining decode and non-request time. These are
conditional arithmetic bounds, not an implemented prefix-cache gain or a
forecast that other tasks will terminate. The earlier 77% TTFT observation was
a different trajectory; 70.5% here reinforces the bottleneck without claiming
identical turn sequences across runners.

Local Astropy trial `astropy__astropy-14096__8kNeQ6D` runs from
20:03:28.719137 to 20:18:28.750481: **900.031344 seconds, reward 1, resolved,
AgentTimeoutError**. The required `test_subclass_property_exception_error` and
all 426 PASS_TO_PASS tests pass. Artifact root is
`/home/mvasiljevic/gemma4-eval-speed-evidence/local_astropy_bfp8_submit_seed9472`.
The correct source edit is applied at **865.974 seconds**, and the reproduction
shows the requested error at **891.945 seconds**, both before the deadline.
One final read-only source inspection arrives **30.874 seconds after agent end**;
it does not create the passing patch. Saved totals (including that late response)
are 28 native tool responses, 442,628 input / 26,539 output tokens and 1.621 tool
seconds. The late response contains 721 output tokens. Zero normalization or
repetition stops occur. The remaining inefficiency is semantic over-analysis
and repeated small synthetic tests, not a malformed native tool loop. The patch
also leaves the method's former docstring after executable statements: verifier
reward is not a general code-quality approval.

At 20:23 UTC the owned `gemma4-eval-weight-bfp8-v2` container is stopped cleanly;
`fuser /dev/tenstorrent/0 /dev/tenstorrent/1` reports no owners. No device reset,
unrelated-container stop, image rebuild, or full five-task run was performed.
The checkpoint covers approximately **4h53m active agent wall time** across
12:43–15:30 and 18:20–20:26, excluding the user-requested stop interval.
Final host revalidation passes all **140 TTI harness tests** and **9 model-tool
invariant tests**, plus JSON parsing and `git diff --check`. The host system
Python lacks pytest; use the TTI virtualenv. Model-tool tests need their tools
directory as the working directory and `-c /dev/null --confcutdir=.` to avoid
unrelated TT-Metal conftest dependencies. The temporary pytest-cache warning
from `/dev/null` does not affect results; disable the cache provider if desired.

### Remaining gates and recommended next bounded work

- Validate Sympy and sklearn with the same candidate in 15–20 minute C1 trials,
  preferably sequentially on one already-loaded server. Do not infer their
  outcome or completion time from Django/Matplotlib.
- Test Astropy clean termination with a bounded extension or a separately
  declared generic progress policy; never inject hidden verifier tests or
  task-specific solution guidance. Keep reward and submission separate.
- Broaden the narrow 100-token precision correctness gate before selecting
  BFP8 for release. Record precision, guard, seed, environment and adapter as
  distinct interventions. The best local paired Django comparison is still
  only one seed and differing cache states.
- Only then perform the intended serial C1, five-task 7,200-second validation.
  Current TTI `215e6375...` is deliberately a one-Django/1,200-second diagnostic
  recipe with explicit seed, not that release config. Restore the actual task
  set/cap and declare policy/sampling deviations before a final comparison.
- Optimize measured prefill/cache costs after outcome stability. Real prefix
  caching remains unsupported; do not enable its flag or claim ideal savings
  as delivered performance. Preserve request telemetry and timeout-tail checks.
