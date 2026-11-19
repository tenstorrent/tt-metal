# Final serving measurements

Primary128/128/1, concurrency1: TTFT291.277070ms; decode37.625576t/s/u. Metric validation only; no overall stage-pass claim.

All profiles: all64 layers, P300x2 TP4, context262144, max_num_seqs32, greedy temperature0, ignore_eos, async/device sampling, same server config before/after. CI100/100/32 has uncapped client admission and is secondary capacity evidence.

| Workload / source | TTFT P50/P99 ms | TPOT mean/P99 ms | ITL P50/P99 ms | Output tok/s |1000/meanTPOT t/s/u|
|---|---:|---:|---:|---:|---:|
|Primary128/128/1 c1 before|276.979049/276.979049|26.573071/26.573071|26.566793/27.194867|35.047923|37.632082|
|Primary128/128/1 c1 after|291.277070/291.277070|26.577666/26.577666|26.570258/27.321591|34.905595|37.625576|
|CI100/100/32 uncapped before|3764.804708/6118.650668|100.565408/123.711529|56.212523/1133.620407|229.325791|9.943777|
|CI100/100/32 uncapped after|3814.207492/6197.994571|101.056727/124.577216|56.225626/1151.054108|227.731638|9.895432|
|Burst128/32/32 c32 before|3924.777816/6588.571701|167.918460/245.881359|56.286939/1562.256804|108.626447|5.955271|
|Burst128/32/32 c32 after|4002.460270/6727.737757|170.045541/249.496937|56.226111/1595.222883|107.038529|5.880778|

Primary128/128/1 analytical functional floor: 9.185971t/s/u; measured pass=True.

Standalone128/128/B1: 37.702224t/s/u, 26.523634ms/step and 75.333871ms TTFT. Serving-minus-standalone decode gap: 0.054032ms/token; rate ratio 0.997967.

Standalone resident nonblocking model/sample traces synchronize once per127-step window with no per-token readback. Serving streams tokens through plugin/scheduler/HTTP; same logical work, different measurement boundary. This is not a serving optimization before/after comparison.

Serial prefill off/off128/32/32 TTFT ratios(after/before): P50=1.019793, P99=1.021122, mean=1.021251. Batched prefill was not adopted; no batched-equals-solo claim.

Prohibited in vLLM serving stages: no Tracy, tt-perf-report, ReadDeviceProfiler or live-server/adapter profiling. Device fields are null; analytical floor and resident standalone timing are not device profiler results.

Exact workloads, configuration, comparisons and artifact paths: [perf_summary.json](perf_summary.json).
