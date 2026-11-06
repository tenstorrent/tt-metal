# Matching default-pool C4 completion cadence

Full60/B8/C4, default10248 pool IDs, override unset; all15 source files match4b7d9a3b. Four independent synthetic histories over five context rounds each reached full4-active residency and>=128 emitted tokens per request, with no preemptions/errors. Release cap126976 and timeout14400 unchanged; own streams disconnected, then fresh idle and graceful closure. No scored or full-completion claim.

| Context | C4 interval end steps | Intervals | C4 mean ms | C4 tokens/s/user | C8 tokens/s/user | C8/C4 aggregate ratio |
|---:|:---|---:|---:|---:|---:|---:|
|4096|13–113|101|29.135547|34.322335|32.107408|1.870934|
|32768|225–331|107|34.742705|28.783021|25.931766|1.801879|
|65536|447–572|126|40.873468|24.465749|21.449875|1.753462|
|98304|668–784|117|47.168805|21.200452|18.132195|1.710548|
|131072|889–1013|125|53.473974|18.700686|15.662486|1.675071|

Windows begin after each final prefill and end at preserved residency proof, before cancellation. Exclude prefill→decode transition. Same extraction reproduces previous C8 interval counts115/105/109/118/119. Completion cadence includes host/scheduler gaps; overlapping submit→complete latency is recorded separately in JSON, never used as tokens/s. Median and nearest-rank p95 are descriptive, not confidence intervals.

293 complete JSONL frames, 1017 unique matched submit/completion pairs, final pending/buffer0, errors empty/no loss. Late prefix reuse 0.997502. Device hold 0.142023051111h from queue start to verified close2026-11-06T21:04:49.910472+00:00, owners empty, versus0.20h forecast/0.30h box; wrapper 0.142837376022h includes later exit.

These short synthetic C4/default-pool versus C8/expanded-pool windows differ in active rows, independent histories and warm/cache state. Aggregate ratios are descriptive shape timing, not a causal GPQA speed claim, scored configuration acceptance or a full-tail bound. No profiling, device-time metrics, central edits or score produced.
