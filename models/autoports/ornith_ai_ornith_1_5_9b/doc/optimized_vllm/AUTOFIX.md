# Serving prefill repair investigation

Started 2026-11-14 approximately 01:30 UTC, after the three unchanged-source chunk trials. AutoDebug inspected source and lowered contracts without hardware. Its report is retained beside this file. The failure is a worst streamed ITL above1000ms during32K arrivals with31 background decoders: budgets512/1024/2048 produce5337.4/5259.3/5096.9ms. No request errors, six captures, no releases. This is an eval-load performance failure, not demonstrated output corruption.

The observed causes must explain original-horizon dependence, successful decode interleaving and cost despite small scheduled fragments. Canonical32768/16384 execution padding fits these observations. Event entry gaps are not kernel timings.

Prioritized hypotheses and discriminators:

1. Restrict large canonical executions with consumed rows≤8192 to the already prepared8192 profile. Keep original planner tags, caps, horizon, selected large-prefill BFP8 norms and LoFi projection policy in the experiment. Compare canonical and candidate real-weight reduced-layer logits, FP32 recurrent/conv state, KV and following traced decode. Exact equality is a question, not an assumption; an approximate result requires honest profile identity and full-model quality gates before adoption.
2. Skip terminal norm/head when authoritative intermediate discard is true. Compare exact continuation state/final outputs and unchanged sampler counters; preserve head warmup and all final/diagnostic paths.
3. If required, use existing scheduler interleave settings to emit decode after one prefill. This changes server configuration and requires a fresh identical-config baseline. It is not a substitute for fixing excessive padded work.

Separate serving-boundary candidate1 removes duplicate penalty history reset and moves nonblocking output transfer into the deferred-read callback.11 focused CPU tests pass; tracked device validation is in progress. This candidate does not claim to fix the long-prefill padding failure.

No plugin/vLLM source edits, reduced context, alignment-only restrictions, profiler calls or resets. Hardware experiments are parent-owned and serialized. Results and rejected hypotheses will be appended here before promotion.

## First padding discriminator

The first fixture attempted decode at2048 while declaring an unfinished32768-token original prompt; generator correctly rejected the negative inferred output count. `compact_prefill_probe.json/.log` retain the failed fixture. No runtime guard was changed. The corrected diagnostic uses context65536, completes the full32768-token prompt before decode, and retains intermediate comparisons.

`compact_prefill_probe_completed_prompt.json`: exact-pass for real layers0/3 under selected policy, tracked TP4 B1. Both1024→2048→32768 and1033→2066→32768 have bit-equal all-shard recurrent/conv/valid-KV state, intermediate/final logits and next-decode logits. Existing program counts and all six traces remain stable. Diagnostic completed fragment times: aligned125.99/127.66→48.83/50.62ms; nonaligned244.64/253.87→92.33/95.14ms. These are reduced-layer diagnostics with explicit synchronization, not serving benchmarks. Full-stack and actual mixed-serving validation remain required; no runtime compact policy adopted yet.

## Full-stack padding discriminator

`compact_prefill_full_model.json`: both aligned and nonaligned sequences pass exact all-shard state/logit comparisons with all32layers, context262144, selected policy, retained traces and stable programs. Each completes32768-token prefill then a traced decode; this is full-model numerical evidence, not HTTP serving evidence.

- aligned_1024_then_1024_then_completion: canonical fragment times[1859.2270920053124, 1887.429469730705, 2839.700068347156]; compact times[588.4432243183255, 604.3833531439304, 2843.0080078542233]ms; all comparisons exact.
- nonaligned_1033_then_1033_then_completion: canonical fragment times[3774.5885839685798, 3837.3483731411397, 2844.335330184549]; compact times[1173.6117573454976, 1201.4844417572021, 2844.79279210791]ms; all comparisons exact.

## Adopted repair validation and first serving trial

Production compact policy plus skipped discarded heads pass132host tests, five tracked device probes, corrected production-selector exact comparison, and all-layer tracked mixed traffic (15requests, modes/nonaligned8193/32769, no recapture/retirement). Full-model standalone decode remains8.703087ms resident at context262144. Fresh tracker positive control passes.

First post-repair P5 budget512 with responsive1prefill/2decode: worst32K ITL1031.9ms, TTFT4K2999.4ms/32K53433.6ms, errors[]. This is a large reduction from original5337.4ms but still exceeds1000ms; do not call it passed.1024/2048 trials continue. This comparison changes scheduler configuration as well as source; matched-original-source baseline at the final selected config is separately required and prepared.

The completed compact8192/responsive sweep has worst ITL512=1031.9,1024=1029.0,2048=1010.0ms, all errors[]. No budget satisfies1000ms. `compact8192_sweep.json` preserves all rows. Next bounded hypothesis: an added6144-row prepared execution extent remains in the same >4096 numerical branch while reducing padded work further. It must not change canonical planner bucket choices. The diagnostic patches only preparation/validation catalogues, compares against original canonical execution and requires exact state/logits/nextdecode with retained traces before any runtime adoption. Reduced tracked experiment running; no6144 runtime edit yet.

## Rejected 6144 extent and next discriminator

The reduced 6144 probe completed both schedules but failed exact equality: 28 state/logit comparisons per schedule differ, beginning in full-attention KV. Logit PCC remains >0.99997 and top1/top5 agree, which is insufficient to adopt this change under the exactness guard. No production 6144 change was made. The next discriminator retains exact8192 computation and changes the existing scheduler interleave configuration to one prefill/four decode steps. This is a supported config experiment, without plugin edits.

The selected-LoFi terminal norm/head geometry sweep completed: production three readers/K2 median0.570413ms (repeat0.570273ms), best alternative four readers/K2 0.573664ms; one-reader controls exceed L1. All legal alternatives are slower. Retain production geometry. This standalone reduced-layer diagnostic is not HTTP throughput.

## Rejected scheduler-only discriminator

One prefill/four decode at2048 completed without request errors but worstITL1010.4ms remains above1000ms (one/two was1010.0ms), with32K TTFT13892.6ms vs13574.2ms. Reject1:4 and retain1:2. This rules out insufficient decode insertion as the residual cause under this workload. Next source-grounded discriminator removes redundant uploads of identical linear masks and shared row-index inputs, with changed/unchanged-input tests and exact full-stack comparisons.

## Cached prefill metadata discriminator

Applied minimal prefill_ops/prefill_runtime cache after devices released. Masks/history indices are per persistent layer/profile and keyed by valid_count; shared row indices are keyed by height/absolute and refreshed by their sole writer. Both invalidate metadata before a possibly partial copy. Eight host tests pass, including changed/unchanged values, switching profiles, changed source with same indices, and injected partial-copy failures. Tracked all32-layer/context262144 force-refresh versus cached-refresh comparison now runs with identical production8192 geometry; both arms compare state/logits/following decode and count actual target copies. No scheduler or arithmetic change.

Tracked all32-layer refresh control passes exact state/logits/nextdecode on both schedules, with retained traces/programs. Aligned targeted copies linear216→0 and row107→21; nonaligned linear360→288 and row299→212. Aligned fragment completed wall times584.01/595.12/2844.24→578.22/590.83/2817.98ms; nonaligned1178.90/1209.69/2844.86→1171.63/1202.28/2823.58ms. These are diagnostic completed times, not HTTP benchmarks. Actual cached2048/responsive1:2 mixed serving now runs.

Cached2048 mixed-serving discriminator completed without request errors: worstITL1003.7ms, TTFT4K886.5/32K13468.3ms. The same1:2 prior candidate had1010.0ms and13574.2ms32KTTFT, improvements0.624% and0.780%, respectively. Still fails1000ms; no passing budget declared. Next candidate investigates repeated immutable attention-plan construction across layers and unchanged per-field uploads, preserving all physical operations.

Candidate5 adds an8-entry immutable attention-plan cache keyed by every scalar plus a fresh immutable mapping snapshot, and skips only per-adapter fields whose last transferred values are unchanged. Binding checks precede refresh; current plan clears during updates and each field invalidates before a possibly failed copy.148host tests pass, including8new cache/map/partial-failure controls. Tracked all32-layer force-all-refresh versus candidate comparison is running with unchanged8192 arithmetic.

Candidate5 full-stack refresh control passes exact all-state/logits/nextdecode for aligned and nonaligned schedules, retaining programs/traces. Aligned attention-input copies128→96 alongside linear216→0 and row107→21. Completed aligned fragment times583.58/594.74/2840.25→545.83/558.92/2757.48ms; these are diagnostic timings, not serving. Same-config2048/responsive1:2 HTTP discriminator now begins.

## P5 repair validation after attention-input caching

Candidate5 closes the investigated1000ms mixed-load gap at2048: discriminator971.7ms, then frozen-source final sweep973.8ms; both31background users,4K/32K arrivals,B32/context262144,responsive1:2,errors[]. Final512 also passes994.9ms. This is measured HTTP evidence, not an inference from reduced probes. Exact full-stack1024/1033 and4096/4105 schedules validate changed/unchanged mask, row and attention-input refresh caching; canonical horizons and full32768-token completion plus next decode remain equal.6144 remains rejected. The final4096 trial, selected-budget full acceptance and independent review remain pending. No waiver or plugin change is used to close the ITL gap.

An adjacent CPU audit found the standalone public replay method lacked the new deferred-output overwrite guard. Added closed/unqueued checks before execution; queued reads remain usable.151 host tests and repository hooks pass. This fixes a concrete alternate-entrypoint lifetime risk; measured vLLM already used guarded decode_forward. Final serving/source snapshots follow this guard.
