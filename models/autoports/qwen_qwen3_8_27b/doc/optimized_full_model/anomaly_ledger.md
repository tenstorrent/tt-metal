# Stage07 anomaly ledger

|Observed issue|Evidence / affected path|Investigation and resolution|
|---|---|---|
|LMHead1D optional input memory configuration absent|head_c8192_r1_k5.log; candidate only|Already-device input does not receive LazyWeight sharding. Added explicit reshard and reran adapted candidates successfully.|
|DRAM-sharded K5 L1 capacity failure|head_c8192_r1_k5_s8.log; candidate only|Reduced chunk/K and swept legal wider shards/readers; eight successful exact-policy controls. All slower; rejection is measured.|
|Generic K16 L1 capacity failure|head_generic_k16.log; candidate only|Split2/K16 and split4/K20 interleaved controls fit, pass real-logit comparison and are slower.|
|Validation wrapper hides required enable_trace argument|full_validation_signature_failure.log; test harness|Removed redundant wrapper; final full_validation.log completes prefill, teacher forcing, AR and long cases. No generator contract weakened.|
|Original128-step collector overschedules short EOS responses|interim_stage_review.md; default host orchestration|Added16-step EOS-aware windows using same persistent storage/trace. Factual47-token result schedules49. Paired clean medians: streamed1739.87ms,buffered1284064.64ms,buffered161797.60ms. Keep16 (+3.32% versus streaming) to bound EOS overrun while preserving device feedback; reject128 for EOS-enabled defaults.|
|B32 trace-transition tracker coverage missing|interim_stage_review.md; all-active/masked model traces sharing collector|buffered_final_b32_transition.json passes all-active/mixed/all-active for greedy and sampled modes with tracking and program-cache allocations included.|
|Apparent EOS metadata mismatch between reduced and full probes|interim review; checkpoint metadata|Concern withdrawn: full generation_config contains248046 and248044, matching prior metadata. Reduced probe uses a different stop configuration. Final full metadata records both; HF controls remain unchanged.|
|ND-shard view warning on packed decode shape|buffered_final_b32_transition.log; inherited packed view|TTNN states equivalent2D WIDTH_SHARDED memory layout is preserved. Decoder source/layout policy is unchanged; exact mixed/full trace controls pass. No inter-layer gather or replicated stream is introduced.|
|Long4096 throughput fixture does not reach EOS at1024 cap|default_long_quality_cases.json; inherited quality behavior|Final tokens match prior stage exactly; coherent reasoning then prose. Nearby4095 ends at892; factual4096 returns exact JSON and EOS47. Existing HF control and stage06 investigation retained. No new quality regression or claim of HF parity.|

The timing-only allocator warning and legacy gather deprecation notice are classified
in runtime_audit.md, with tracker-on lifetime evidence and measured route controls.

Watcher launch initially exceeded ACTIVE_ETH's26,624-byte config with29,264 bytes
of inline-instrumented program/arguments. This was before model execution, not a
kernel correctness failure. The accepted prior watcher campaign used NOINLINE;
restoring TT_METAL_WATCHER_NOINLINE=1 fits without disabling Ethernet, assertions,
NOC or CB checks. Bounded reset/list/ring smoke passed. Both final watcher runs
pass14 exact cases total, with tracker/program-cache allocations enabled and at
least1,216bytes stack headroom in recorded dumps (`watcher_summary.json`). Raw
watcher logs contain no error/exception/overflow/invalid/corrupt markers.
