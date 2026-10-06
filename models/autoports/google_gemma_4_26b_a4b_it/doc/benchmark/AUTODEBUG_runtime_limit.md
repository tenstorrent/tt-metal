# AutoDebug: frozen accuracy runtime limit

Inspection-only diagnosis of the current Stage 11 attempt. No inference, hardware
access, profiling, dependency changes or implementation edits were performed.

**Conclusion:** no actionable transport or scheduling correctness defect is
established by these artifacts. Execution was making slow progress through the
configured host-sampling path. There is no validated repair or measured execution
plan that supports completing the remaining frozen accuracy workload within the
original hour. This is not a proof that completion was mathematically impossible:
accuracy was deliberately interrupted before the deadline, and uncompleted
answers' lengths and remaining execution times are unknown.

## Verified evidence

- [Budget decision](run/budget-decision.json) records deliberate SIGTERM of the
  owned accuracy client at 942.539 seconds with 8/536 responses. The
  [client log](run/accuracy-shared.log) reports its eighth response at 12:13 of
  API-request time; that interval excludes earlier server launch and setup, so
  it must not replace the original stage clock. There is no client exception
  preceding that deliberate interruption.
- [Raw responses](run/mmlu_pro/responses.jsonl) contain eight normal `stop`
  completions of 216, 235, 299, 310, 328, 330, 336 and 348 tokens: 2402 total,
  mean 300.25. None reached the 4096 cap. The
  [partial scoring record](run/partial-accuracy.json) leaves 272 MMLU-Pro and
  256 IFEval answers unscored. The eight completed answers are completion-biased.
- [Server log](run/server-b32.log), lines 183–209, repeatedly reports 32 running,
  zero waiting, roughly 13.6–14.0% KV usage and 32 generated tokens/s during
  19:06:53–19:10:03. Lines 210–268 show successful chat HTTP responses, running
  counts of 31–32, replacement-prefill activity and repeated
  `LOGITS_TRACE_READY`. Throughput sometimes drops to 0–6.2 tokens/s around
  those transitions, then returns to 32. This supports real execution and batch
  turnover, not HTTP serialization to one request, a dead engine or persistent
  KV-capacity starvation. Zero server waiting requests alone does not mean the
  client has no queued work; its concurrency is capped at 32.
- The later `EngineDeadError` follows the explicit server shutdown at 19:27:20;
  it is not evidence that the earlier accuracy pass crashed. The
  [stopped server record](stopped-b32-5920.json) records the 176.195-second
  startup and `leader_alive=false`.
- [Configuration](run_config.json) freezes temperature 1, top-p .95, top-k 64,
  native nonthinking mode, 4096 output cap and `logprobs=true,top_logprobs=0`.
  The actual plugin's `check_perform_device_sampling`,
  `vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:2641–2687`,
  returns false for any requested logprobs on this four-device configuration.
  Its host path consumes logits at lines 3035 onward. The retained
  [routing probe](setup-previous/host_sampling_route_probe.json) agrees.
- [Adapter](../../tt/generator_vllm.py), lines 243–257, explicitly requests
  traced logits, disables device feedback and reads those logits to the host
  each decode. [Generator](../../tt/generator.py):194–196 concatenates the
  vocabulary shards and converts them to float. Host sampling is therefore an
  intentional, observable cost of the currently selected compatibility path.
  Tracing is enabled; this is not accidentally untraced decode.
- The TT scheduler explicitly separates prefill/decode and normally prioritizes
  pending prefill (`vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/scheduler.py`,
  lines 86–113 and 165–179). The generator serializes prompt rows
  ([generator.py](../../tt/generator.py):256–271). Host prefill releases the
  decode trace ([generator_vllm.py](../../tt/generator_vllm.py):183–188), while
  changed batch/active slots trigger another warm/capture with synchronization
  ([generator.py](../../tt/generator.py):377–439). These source paths explain
  plausible refill/trace costs consistent with the log; no duration breakdown
  for accuracy proves how much each contributes.
- [Report](run/REPORT.md) and [summary](run/summary.json) show both complete
  4096-input/128-output performance profiles: C32 TPOT 699.92 ms and C1
  19.61 ms, with output throughput 24.30 and 25.47 tokens/s respectively.
  These use greedy device sampling and different server slot counts. The
  latency ratio is not evidence of an HTTP bug, nor a valid estimate of the
  exact host-sampled accuracy workload's potential speedup.

## Estimates and unresolved hypotheses

Using the eight completed answers' mean only as a hypothetical workload,
528 remaining answers × 300.25 tokens / 32 aggregate tokens/s is approximately
4954 seconds (82.6 minutes) of generation, before prefills and trace rebuilding.
This extrapolation is not an ETA or lower bound: the completed sample is biased,
remaining answer lengths are unknown, and some interrupted requests had already
generated unretained work. At the parent's approximately 19:45 UTC observation,
about 833 seconds remained on the original clock. Even maintaining 32 tokens/s
for all that time would provide about 26,656 tokens, or 50.5 per missing answer,
before restart, prefill and reporting costs. These conditional calculations show
why another launch lacks evidentiary support; they do not prove impossibility.

The strongest concrete optimization hypothesis is trace churn during request
turnover. Existing logs show repeated trace readiness, and source identifies
the invalidation conditions. A second hypothesis is full-logit transfer/host
sampling cost. Their relative contributions, safe removable work and attainable
speedups remain unmeasured. Deleting trace release/invalidation is not a proven
fix: traces bind batch/active slots, cache and tensor buffers. Enabling chunked
prefill is not an available repair either: the adapter explicitly rejects
nonzero continuation positions at lines 184–185. No evidence supports treating
template rendering, EOS, scorers, retries or transport settings as the cause.

## Concrete next step

Keep this attempt incomplete, preserve the eight responses and both valid
performance profiles, and retain the original absolute deadline. Do not count
partial scoring or another fresh clock as completion. There is no established
bug to patch safely within this audit.

For a separately authorized optimization effort, start with the identified
trace invalidation path: establish which bound buffers/active-slot signatures
must change across one completion and refill, then test a candidate that avoids
only demonstrably unnecessary recapture while preserving exact sampling, KV
ownership and scheduler behavior. Host tests must cover batch shrink/refill and
cache-table changes; any performance claim then requires actual unchanged-policy
accuracy-path measurements. Until that evidence exists, this is a bounded
engineering investigation, not a supported route to finish the current hour.
