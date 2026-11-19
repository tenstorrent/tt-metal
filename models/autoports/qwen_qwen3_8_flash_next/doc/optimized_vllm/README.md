# Optimized vLLM — stage 10 complete

Primary warmed single-user serving measures **7,368.667 ms TTFT and 9.654300 decode tokens/s/user** for **128 input / 128 output tokens, one request, concurrency1**. Before:7,367.276ms and9.650107t/s/u under the same server/workload. The0.0435% decode difference is within noise; no meaningful primary speedup is claimed. Mean TPOT/ITL is103.580787ms after versus103.625797ms before; output throughput is6.236666 versus6.235355tok/s. Resident capacity2, context262144, P300x2 mesh1×4, greedy device sampling, budget9, alignment8 and interleave1/2 apply to both arms.

These measurements run through the real vLLM TT plugin and `tt/generator_vllm.py`, using `readiness_check.run_vllm_server` and one request warmup. Single-user t/s/u is1000/mean TPOT in milliseconds. Matched-source/runtime and cleanup checks pass in [perf_summary.json](perf_summary.json). Long-context timing, critical eval-load checks, qualitative output and trace/async checks pass. Sampling correctness is accepted with four separately recorded reproducibility-only failures; independent [round2 review](stage_review_round2.md) is **clean-pass**.

Secondary CI capacity evidence uses **100 input / 100 output tokens,32 requests, uncapped offered concurrency, resident2/C262144**: all32 complete in both arms. Aggregate output is6.220888tok/s after versus6.217375 before; mean TTFT251345.289ms versus251479.417ms; mean TPOT235.875647ms versus236.006829ms. These aggregate rates are not headline single-user decode rates.

## Matched measured metrics

Both arms: all48/C262144/resident2/P300x2/mesh1×4, budget9, alignment8,
interleave1prefill:2decode, async, greedy device sampling, ignore EOS, one warmup.
TTFT and TPOT p99 equal p50 for the single measured primary request; this is
not a multi-run statistical confidence estimate.

| Workload | Metric | Before | After |
| --- | --- | ---: | ---: |
|128in/128out/1request/c1|TTFT median ms|7367.275664|7368.667097|
|128in/128out/1request/c1|TTFT p99 ms|7367.275664|7368.667097|
|128in/128out/1request/c1|TPOT mean ms|103.625797|103.580787|
|128in/128out/1request/c1|TPOT p99 ms|103.625797|103.580787|
|128in/128out/1request/c1|ITL median ms|103.607503|103.553426|
|128in/128out/1request/c1|ITL p99 ms|104.444487|104.513866|
|128in/128out/1request/c1|Aggregate output tok/s|6.235355|6.236666|
|128in/128out/1request/c1|Decode t/s/u (1000/TPOT)|9.650107|9.654300|
|100in/100out/32requests/uncapped offered|TTFT median ms|251474.499896|251339.334627|
|100in/100out/32requests/uncapped offered|TTFT p99 ms|494495.170169|494207.325512|
|100in/100out/32requests/uncapped offered|TPOT mean ms|236.006829|235.875647|
|100in/100out/32requests/uncapped offered|TPOT p99 ms|238.200538|238.098797|
|100in/100out/32requests/uncapped offered|ITL median ms|201.998741|201.902779|
|100in/100out/32requests/uncapped offered|ITL p99 ms|563.421202|565.363560|
|100in/100out/32requests/uncapped offered|Aggregate output tok/s|6.217375|6.220888|

## Measurement status

| Evidence | Workload and configuration | Result |
| --- | --- | --- |
| Historical primary |128in/128out,1request/c1, resident2/C262144, budget256/alignment128 | TTFT4648.148ms; TPOT103.516583ms; median ITL103.493736ms; output throughput7.192993tok/s; decode9.660288t/s/u. |
| Historical secondary CI burst |100in/100out,32requests, uncapped offered concurrency, resident2/C262144, budget256/alignment128 |32complete; median TTFT214182.769ms; TPOT201.637464ms; median ITL201.829421ms; aggregate output7.262191tok/s. |
| Current full-model comparison | Full48,128in/128out,127decode steps,B1/C262144, selected policy, mesh1×4 | Resident model+sampling102.270285ms/step =9.778011t/s/u; public serving-boundary resident102.240057ms/step =9.780902t/s/u. These are generator measurements, not HTTP headline metrics. |
| Completed chunk sweep |4096/32768 input and4 output, one128-input background decoder, resident2/C262144, alignment8/interleave1/2 | Selected9: worst client ITL592.1/615.7ms; foreground TTFT355182.9/2918032.7ms. Budget17 fails4K at1184.1ms;33 fails both at1339.8/1364.7ms. |
| Matched primary and CI | Same run_vllm_server workload and server configuration on both arms, selected budget9 | Both runs and comparability/cleanup checks pass; see headline and perf_summary.json. |

Artifacts: [historical baseline](../../readiness_vllm/stage10_before/vllm_benchmark.json),
[historical CI](../../readiness_vllm/stage10_before/vllm_ci_serving_benchmark.json),
[fresh full model](full_model_comparison_v3.json),
[chunk execution manifest](chunk_sweep_execution.json), [verified selection](chunk_selection.json),
[all six coverage windows](chunk_v4_background_coverage.json), and
[owned cleanup and interrupted teardown](chunk_v4_cleanup.json).

## Fixed serving contract

- All48 layers, pinned checkpoint revision
  `812d96e10a0dd82b69d749e68693698cf6c36daa`, selected
  [combined_shared4_mixed policy](../datatype_sweep/selected_precision_config.json).
- [Context contract](../context_contract.json):262144 for serving, benchmark
  and eval. Small reduced-layer diagnostics are labeled separately.
- Resident capacity2, page size32, mesh1×4/P300x2, `FABRIC_1D_RING`, trace region0,
  `trace_mode=all`, `sample_on_device_mode=all`, async scheduling. Capacity2 is
  the tested serving configuration, not a claim of the largest feasible batch.
  CI/eval32 users queue through these two resident slots.
- Non-aligned prompts and arbitrary supported prefill continuations remain
  supported through the existing `prefill_chunk_alignment=8` option. Partial chunks end on an8-token grid; final chunks retain arbitrary lengths. Selected decode interleave is one prefill step and two decode steps at budget9; the previous1/1 sweep failed the1000ms bound at all three budgets.
- Prefix caching has the documented `recurrent_state` exclusion: KV-only reuse
  cannot restore GDN/convolution/PLE/indexer history. Dense batched prefill is
  not adopted. The matched serial-prefill burst (128in/32out,32requests/c32) has after/before mean TTFT ratio0.999543 and p50 ratio0.999538; no batched path is claimed.
- No serving profiler, Tracy, tt-perf-report or ReadDeviceProfiler. Measurements
  use benchmark JSON, ordinary route/capture events and correctness probes.

## Changes and current correctness evidence

Persistent prefill state-copy traces preserve operands and cache ownership.
Prompt-history allocation/reset is skipped when penalties do not consume it.
A multicast completion race was repaired with alternate-route completion
credits and a persistent-output epoch barrier. This correctness repair is
shared by both final performance arms; see [AutoFix](AUTOFIX.md).

Required eval top-k40 now remains on device. A single common sampler owns
persistent32/64 candidate families, offsets, RNG and token feedback; the wide
path selects globally paired candidates before native sorted64 sampling.
Public support is top-k40. Raw unsupported requests are rejected before state
mutation. Inactive rows are neutralized, unchanged effective parameters are
not reuploaded, and both families are warmed before trace capture. The
greedy path remains canonical split sampling. No plugin or vLLM source changed
in this stage; the plugin remains at
`629f07cb385494b14d0ee3d138de039d71d09546`.

Focused evidence passes: [native sampling](wide_native_v2.json),
[exact old narrow-token regression](narrow_sampler_regression.json),
[wide seeded lifecycle](wide_sampler_lifecycle_v2.json),
[greedy queued async watcher](wide_repair_greedy_async_watcher_v1.json),
[eval queued async watcher](wide_repair_eval_async_watcher_v1.json), and
[trace-tracker positive control](trace_alloc_tracking/wide_positive_control.json).
These scoped probes do not replace final full48 real-vLLM acceptance.

All12 fresh full48 TT qualitative outputs exactly reproduce the prior control's
256 raw tokens per request (six prompts, greedy and sampled k20, B1/C262144).
[Regression evidence](quality_wide_regression.json) and
[manual HF/TT review](quality_controls_review.md) retain the sampled-haiku form
defect and answers truncated at the shared cap. Fresh HTTP qualitative output now matches all12 TT control sequences exactly
through EOS or the256-token cap; [manual review](final_quality_review.md) passes
within that scope. The eval-profile load completed all32 requests and passed all seven critical checks; three advisory findings are recorded below.

## Acceptance and limitations

The [owner decision](owner_decision_shutdown_2026_11_17.json) declines the proposed plugin change. The plugin remains unchanged at `629f07cb`. Slow/incomplete orderly teardown is a plugin-owned upstream limitation; existing default-timeout cleanup must leave no owned processes or device handles. Trace release counts and forced termination remain visible in audits. The sweep recorded 0/201 releases at budget9 before worker termination; it does not establish orderly closure. With profiling disabled the native profiler helper returns without collection, and logs do not isolate that hook as the cause of interrupted teardown.

Matched primary, CI and serial-prefill burst measurements pass. Both tracked
synchronous and asynchronous full48 replay suites pass, including exact
repeated/reordered greedy histories, pending-prefill slot relocation, mixed
non-aligned traffic and trace retention. The asynchronous histories match the
synchronous controls exactly and exercise actual overlap. See
[trace/async summary](final_trace_async_summary.json). The mixed windows permit
one deferred first capture and retain every existing trace; this is not a
zero-capture claim.

Final full sampling produced63 passed,75 skipped and four output-reproducibility
failures. Raw pytest is not a pass. Current native full-logit repeat/row-swap checks pass; all400 served/native token-step top20 logprob maps match exactly. The four failures are separated as reproducibility-only under the serving skill, with scope limits in [sampling disposition](final_sampling_summary.json) and [logit evidence](final_logit_summary.json). The three supported chunked-prefill recall tests now pass on the
same budget9 server using the existing chat/reasoning options (low effort,
1528-token reasoning allowance); the default24-token legacy invocation ended
inside thinking and its failed artifacts are retained. No assertions changed.
The remaining72 skips are speculative serving (71) and all-vocabulary logprobs
beyond the API's top20 limit (1).

HTTP qualitative passes12/12 exact control comparisons. Clean mixed traffic
passes18 requests (129/130/131/257/513/515 input,96 output,c2), device-only routing
and trace retention with one deferred first sampler capture. Its text-stream
gap p50/p99/max is201.722/584.013/1944.979ms; these are nonempty text-chunk gaps,
not exact token ITL. See [mixed evidence](final_mixed_summary.json).

P5 timing passes all four checks, C262144/resident2/budget9,32 output tokens:

| Input tokens | TTFT (ms) | Mean TPOT (ms) |
| --- | ---: | ---: |
|1024|59783.0|104.60|
|8192|482918.2|110.63|
|32768|1954437.8|118.00|
|131072|8004681.3|129.54|

The127/128-input,4-output alignment probe also passes (median TTFT7569.3/7392.8ms).
See [long-context timing](../eval_readiness/long_context_timing.json).

P4 offered32 users through resident2/C262144 with the unchanged
`lm_eval_reasoning` recipe (T1,p0.95,k40,seed42),6000-token cap and natural EOS.
All32 completed:29 natural stops and3 at the cap; median428.5 output tokens.
All seven critical checks pass: no engine error, host fallback, trace capture,
preemption, leaked slot or retained KV. Plain/eval256-output comparison bursts
(32 requests each) give median TPOT213.02/213.23ms, ratio1.001.
Actual37315 output tokens over4006.0s yield9.314778 aggregate tok/s, a capacity
measurement. See [load evidence](final_eval_summary.json).

Three load advisories remain in [limitations](../eval_readiness/limitations.json):
no established resident2 functional floor, insufficient median answer length,
and insufficient bucket coverage for the packaged forecast. Its raw zero-hour
placeholder is not an estimate. The separate [wall-time forecast](eval_forecast_wall.json)
projects120.104851h for the planning-default4027500 output tokens, exceeding
the shared14.4h allowance. It extrapolates this short-prompt mix; only three
answers reached6000, so it does not establish12K-answer performance or actual
eval accuracy. Illustrative reduced question counts are hand-off planning,
not changes to required eval scope or context. Only four saved answer tails
were available for manual review; their displayed conclusions are coherent
and correct, with no all32-answer accuracy claim.

[Final server cleanup](final_serving_cleanup.json) verifies unchanged model/shared
runtime, idle requests, no owned PIDs and no device handles. The runner's default
timeout interrupted trace release (0/205 observed); this remains the owner-accepted
plugin limitation. No TT reset or plugin change was needed.

[run_server.sh](run_server.sh), [benchmark.sh](benchmark.sh),
[eval load plan](eval_load_plan.md) and [work log](work_log.md) record commands.
[The comparison script](../../tests/compare_optimized_vllm.py) validates matching
server contracts and workloads before deriving primary rates and capacity
ratios. The built-in P4 forecast assumes32 simultaneous decoders; preserve it
as raw output and publish a corrected forecast using actual total output tokens
divided by load wall time for this resident2 server.

The inherited [owner decision](../optimized_full_model/owner_decision_2026_11_16.json)
accepts the run's B1 decode-floor shortfall and prohibits further tuning for
that floor; it does not make the numerical floor pass or waive correctness.
Other limitations must be backed by final probe evidence in
[limitations.json](../eval_readiness/limitations.json).

The [optimization checklist](optimization_checklist.md) records tried and
untried items. The eight-hour Stop Rule fired at 11:06:35 UTC. The owner subsequently loosened the wall cap for remaining required gates and slow context/load runs; optional tuning remains stopped. Local implementation checkpoints include `4e5a1ddd`, `572c9415`,
`be261ee9`, `755bf420` and control checkpoint `acc20fc1`; full SHAs are in the
work log. Nothing has been pushed.

Final acceptance is indexed by [the manifest](final_acceptance_manifest.json).
The runner check returns advisory1 with no critical errors; raw sampling retains
four separately classified reproducibility failures. Local checkpoint receipts
are recorded in the work log. No plugin/vLLM change or push was made.
