# B32 lifecycle host timeline and capacity limits

The revised full-model B32 scheduler configuration passes lifecycle checks and reduces the cold multi-turn wave from 524.390 to 33.763 seconds. This is an offline host-event analysis of existing serving artifacts, not device profiling. The tested multi-turn profile is **32 conversations × 4130 initial tokens, then +33 tokens**, not 32 × 32768. The original configuration and its diagnosis below are retained for comparison.

Artifacts: `readiness_vllm/full_b32_host_timeline.png`, `full_b32_host_timeline.json`, reproducible offline script `analyze_b32_timeline.py`. Sources: `full_b32_lifecycle.json`, `.requests.json`, `full_b32_events.jsonl` lines 3901–5084, and `full_b32_server.log`. All filenames are under `readiness_vllm/` unless specified otherwise.

The offline analyzer accepts `--prefix full_b32_budget131072` to read that prefix's lifecycle, requests, events and optional server log and write separate timeline PNG/JSON outputs. Default `--prefix full_b32` retains the original input/output names. It leaves missing queue samples unknown and labels them unavailable in the plot; it does not infer a zero queue. Both runs have now been analyzed; both resulting PNGs were visually inspected.

## Measured configuration comparison

The revised run uses `--max-num-batched-tokens 131072 --long-prefill-token-threshold 4096`, full64 selected precision, fixed B32 and native262144 context. Evidence is `full_b32_budget131072_lifecycle.json`, its requests file, archived server log and `full_b32_budget131072_host_timeline.{json,png}`. Same-run event slice `[1,904)` is worker PID1654081; captures stay4, retirements0 and host-logits read delta0. All112 requests complete, greedy repeats remain stable and mixed-shape/chunk/concurrent-decode gates pass.

| Host observation and workload | Original budget8192 | Revised budget131072 / threshold4096 |
|---|---:|---:|
| Cold32 ×4130 wave wall seconds | 524.390 | 33.763 |
| Cold32 ×4130 prefill host seconds | 515.926 | 32.468 |
| Cold32 ×4130 adapter prefill calls | 64 | 4 |
| Warm32 ×4163 wave wall seconds | 3.444 | 3.446 |
| Whole112-request mixed lifecycle seconds | 746.342 | 255.429 |
| Whole112-request TTFT P99 seconds | 521.623 | 32.687 |
| Whole112-request SSE pause P99 seconds | 32.765 | 0.426 |
| Whole112-request maximum SSE pause seconds | 32.766 | 31.959 |

The cold wave now groups its aligned work into one1-row and one31-row4096-token call, followed by one1-row and one31-row34-token tail call. This replaces32 individual calls at each extent and validates the predicted removal of avoidable repeated fixed-B32 prefill work. Cold-wave wall time improves15.53×. Warm-turn reuse remains131072/132160 =99.1768%, with all32 admissions at4096 and32 active slots observed. The warm wave does not improve materially because the original already batched its short cached tails.

Sampled API log queue counts peak at waiting0 versus30 before, but polling does not prove that no request ever waited; the revised log samples also miss much of the short multi-turn phase. Adapter events, not these queue samples, establish the32-slot observation. Maximum SSE pause remains31.959 seconds during mixed long-prefill traffic: coalescing removes the repeated cold-wave work, but does not remove the latency of synchronous fixed-B32 prefill or guarantee a hard decode service-time bound. These correctness/diagnostic runs use trace-allocation tracking and synthetic tokens and are not headline performance benchmarks. A separately controlled tracking-disabled benchmark remains necessary. The32768-token planning pool limit below is unchanged.

## What the timestamps establish

The 112 synthetic requests finish in 746.342 seconds. Mixed-workload TTFT P99 is 521.623 seconds; largest SSE inter-chunk pause is 32.766 seconds. SSE gaps are host delivery intervals, not device token latency. Adapter prefill host phases occupy 647.220 seconds (86.72% of this interval). The single longest prefill call is 30.069 seconds. Captures remain exactly four and retirements remain zero, in the same worker PID 1634546. Decode and sampling signatures remain the two decode masks plus ordinary/buffered top-k sampling.

The cold 32 × 4130-token wave lasts 524.390 seconds, with 515.926 seconds inside 64 prefill calls, 160 physical chunk events and only 23 decode submissions. Every cold request contributes a one-row 4096-token call and a one-row 34-token tail. The initial timeline is explicit: slot 0 prefill takes 14.455 seconds; slot 1 takes another 15.041 seconds; slot 0's tail takes 1.609 seconds; only then is decode submitted. The recurring pair of long and tail prefill work explains pauses of approximately 32 seconds and the staircase of first-token latencies. API server logs independently report up to 30 waiting requests. This is observable admission/serialized-prefill pressure, not a hidden recapture interval.

The subsequent reversed-order warm wave (32 × 4163-token requests) takes 3.444 seconds: 2.317 seconds in two prefill calls and seven decode submissions. Its calls process one 67-token row and then 31 × 67-token rows. All 32 request admissions resume at actual scheduler position 4096; they reuse 131072 / 132160 previously sent prompt tokens = **99.1768%**. The request denominator for a cache-query hit rate is different: 131072 / (32 × 4163) = 98.3906%. Do not label the previous-sent-token reuse fraction as a metrics query-hit fraction. Event evidence records 32 simultaneously owned/active slots in the warm wave. Cold-wave queue counts and this warm-wave peak are different observations. All completion checks passed, and the metric `num_preemptions_total` remains zero.

## Mechanism and scope of the conclusion

`tt/generator.py::prefill_forward` uses the generator's fixed batch size for token, position and activation buffers, groups physical lengths, and forwards that fixed batch through `tt/model.py::prefill_forward` and all layers. `tt/generator_vllm.py::prefill_forward` builds those fixed-B32 rows even when only one scheduler slot has work. Masking protects inactive recurrent state; it does not remove the fixed-width dense work. Physical chunks are 1024 tokens. Scheduler blocks are 4096 and the server budget is 8192; splitting a scheduler chunk into internal physical chunks does not let the scheduler admit decode between chunks of the same synchronous adapter call. The plugin's `submit_prefill_chunks` additionally separates intermediate and final rows when their output obligations differ. Source inspection explains the observed widths, but the event stream cannot attribute every individual admission decision to a specific scheduler branch.

The measured dominant overhead is shared fixed-batch prefill work plus its integration with serialized scheduling/admission. It is potentially optimizable (for example, coalescing compatible prefills or a separately validated narrow prefill path), not a demonstrated hardware lower bound. The result must not be described as an unavoidable decode limitation or as proving that all vLLM-specific overhead is absent. No device-time split between kernels, dispatch and synchronization can be derived from these host spans.

The independently reported B1 4096-input/128-output vLLM decode result of 36.229 tokens/s/user versus prior autoregressive 36.171 is approximately +0.16%, consistent with no material added steady-state B1 decode overhead in those measurements. This comparison does not remove B32 cold-prefill admission stalls or establish exact equivalence across separate runs. Teacher forcing is not used as a measured autoregressive result here.

## Scheduler configuration hypothesis and host validation history

The proposed and subsequently validated flags are `--long-prefill-token-threshold 4096 --max-num-batched-tokens 131072`. The threshold bounds each individual request chunk at 4096; the total budget can cover 32 such chunks. Pinned vLLM 0.30 has no `max_num_partial_prefills` or `max_long_partial_prefills` fields/flags, so those proposed older-version knobs must not be used. TT's `_schedule_prefill_only` calls the upstream scheduler with partial and waiting requests, while temporarily excluding pure decodes.

Host-only `readiness_vllm/check_b32_scheduler_budget.py` exercised actual `TTScheduler.schedule()` with 32 waiting 4130-token requests, threshold/alignment 4096. Budget 8192 admitted two 4096-token chunks; budget 131072 admitted all 32. The probe uses the plugin's full-attention host cache fixture and therefore validates scheduler budget admission, not the production hybrid pool, grouping at the adapter, device memory fit or latency. The host run passed; its environment also emitted a nonfatal model-info cache-write warning for `/.cache`.

The measured comparison above confirms the reduction predicted from avoiding the8192-budget pattern where a34-token continuation and one4096-token new chunk leave only4062 tokens, insufficient for another aligned chunk. It does not guarantee a single adapter call when request offsets or final/intermediate status differ. Nor does the4096 threshold guarantee small wall latency: one fixed-B32 4096-token chunk still takes substantial time. Root executed the new serving validation; this analysis only read its completed artifacts.

## Pool ledger: configured limit versus physical limit

The prepared event records 289 shared scheduler block IDs, each 4096 tokens, and four cache groups (one full-attention group and three linear-state groups). Per device: attention pools 10,303,307,776 bytes; recurrent snapshots 11,122,458,624 bytes; live B32 recurrent state 1,231,552,512 bytes. Allocation checkpoint: 29,208,974,848 bytes allocated, 4,835,538,432 free. After preparation: 29,313,353,216 allocated and 4,731,160,064 free. These are allocator checkpoints, not a runtime peak or a largest-feasible-capacity search.

At 32 × 32768, cached full-attention prefixes alone consume 32 × 8 = 256 IDs; one retained linear checkpoint per conversation per three linear groups adds 96 IDs. Thus retaining all prefixes requires at least 352 IDs plus the null block (353), before extra live/staging reservations. The existing **289-ID pool cannot hold that working set**. This is a proven bound of the configured pool; this lifecycle never exercised that larger target and provides no measured eviction/latency result for it.

With the current dense-per-layer pool layout, one additional global block costs 74,137,600 bytes/device; increasing 289 to 353 adds 4,744,806,400 bytes/device. That already slightly exceeds the 4,731,160,064 bytes free at the post-preparation checkpoint, before transient runtime workspace. It makes the current layout's capacity pressure concrete, but is not a proof of a hard device/model limit: different pool packing, scratch lifetimes, precision or cache allocation policy could change it. Do not reduce native advertised context or call 32 × 32768 physically impossible on this evidence. Record a configured-pool/layout shortfall and the unmeasured larger-workload limitation explicitly.
