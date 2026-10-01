# SWE evaluation speed investigation

Started 2026-10-01 UTC at the user's request: use an Astra agent to diagnose
the current eval bottlenecks, predict the attainable speed, iterate on small
measured experiments, then run and monitor actual CI. This is ongoing work,
not a claim that the five-task evaluation has been repaired.

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
sweep, which changes engine RNG history versus CI. It uses Harbor
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
