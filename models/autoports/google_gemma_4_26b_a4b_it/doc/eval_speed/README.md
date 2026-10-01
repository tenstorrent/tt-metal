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
