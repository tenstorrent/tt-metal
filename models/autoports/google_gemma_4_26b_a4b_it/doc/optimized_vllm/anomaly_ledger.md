# Anomaly ledger

## First-use TPOT and ITL disagree

Observed: candidate first-use4096/128/B1/C1 TPOT17.7136ms while meanITL19.5620ms.
Evidence: candidate_first_use/vllm_result.json; independent initial review.
Affected path: HTTP streaming benchmark accounting.
Control: warmed baseline and final shared benchmark, identical configured workload.
Investigation: client ITL is per streamed chunk whereas TPOT uses reported token
count. Values imply115 rather than127 streamed intervals; exact chunk history
was not saved, so coalescing is a supported hypothesis, not a proven root cause.
Resolution: controlled. Final warmed meanTPOT19.747422ms and meanITL19.747424ms
agree; no first-use decode speedup claim is accepted.

## Watcher Ethernet instrumentation capacity

Observed: ACTIVE_ETH program28464bytes exceeds26624-byte config buffer;
subsequent exception cleanup segfaulted.
Evidence: batch_cache_watcher.log, exit139 before model construction.
Affected path: instrumented fabric startup, not model token replay.
Control/investigation: skill-prescribed retry with TT_METAL_WATCHER10 and
TT_METAL_WATCHER_DISABLE_ETH1, same multirow probe, no compute assertions disabled.
Resolution: scoped control passes, batch_cache_watcher_retry.log/.json. All four
devices opened/closed normally. Ethernet Watcher coverage is not claimed.

## Output wording and truncation

Observed: greedy “own-contained” and long answers ending at256-token cap.
Evidence: after/vllm_qualitative_outputs.json and qualitative_comparison.json.
Affected path: generated prose.
Control: identical prior validated full-model/serving greedy output and pinned
HF/selected-policy comparison documented by stage09.
Investigation: read all12 candidate outputs; all6 greedy strings match exactly.
Resolution: controlled inherited wording/length limitation; no new mechanical
repetition, wrong-language drift or cross-request contamination.

## Device allocation warning behind live trace

Observed: generic warning on new request shapes while trace exists.
Evidence: server logs; reduced adapter allocation-tracker run.
Affected path: request/capture lifecycle.
Investigation: track allocations with program-cache checking retained, inspect
persistent source/output ownership, exercise repeated and changed shapes.
Resolution: exercised reduced path passes tracking and full requests match
controls. No universal all-shape allocator stability or peak-memory claim.
