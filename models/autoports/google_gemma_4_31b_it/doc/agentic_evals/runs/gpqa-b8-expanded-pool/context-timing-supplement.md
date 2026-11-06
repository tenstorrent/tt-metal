# C8 context-dependent cadence from the existing capacity run

Same full60/B8/C8 expanded-pool source and private phase file as timing-note.md; no rerun or profiler. For each round, select consecutive eight-active decode completions after its final prefill, ending at that round's preserved residency-proof step before cancellation. Exclude prefill→decode transition. Cadence is host completion-to-completion wall time; submission latency overlaps asynchronously and is not device time or throughput.

| Input context | Completion interval step IDs | Intervals | Mean ms/token/user | Median ms | p95 ms | Tokens/s/user | Mean submit→complete ms |
|---:|:---|---:|---:|---:|---:|---:|---:|
|4096|418–532|115|31.1455|30.0894|36.9841|32.1074|59.6265|
|32768|712–816|105|38.5627|37.5722|46.4153|25.9318|74.1815|
|65536|1027–1135|109|46.6203|46.1117|54.3289|21.4499|90.1026|
|98304|1342–1459|118|55.1505|54.6471|62.5446|18.1322|106.9594|
|131072|1657–1775|119|63.8468|63.1901|71.4131|15.6625|124.0756|

p95 uses nearest rank. All selected steps have eight active rows; positions increase slightly beyond each nominal input context, and rows enter final decode with staggered progress. These short windows support a context-cost trend, not full-tail stationarity or statistical confidence bounds. Token retention/streaming/host costs remain inside cadence. Differences between rows' histories and sequential warm state prevent an isolated causal context experiment.

A clearly conditional interpolation example: keep all eight rows active, hold the4K mean below4K, linearly interpolate mean cadence between measured contexts, and integrate126976 decode steps starting at position481 (ending127457). That gives1.637656h, versus2.251949h if every step had the measured131K cadence. Neither includes initial prefill/setup/queue, neither is a bound, and neither is a measured scored request. Real GPQA active-row counts and trajectories vary; long histories do not all share equal lengths. Therefore this table may inform later cohort projection only alongside retained per-request work/occupancy. It does not justify a2x concurrency benefit or admission of larger cohorts by itself.

Coordinator budget update: device box3.2h; actual local0.3301978594h + high-context tail sensitivity2.251949h + historical setup0.137184h =2.7193308594h, leaving approximately0.480669h. Zero speed saving credited. Scored20 remains conditional on valid repaired20 and prior normal40 dispatch. implementation-plan.md reflects this revision; prior completed timing and confirmation evidence is preserved.
