# Accepted batch performance

Device timestamp envelopes of complete layers, including gaps. All rows use4096 prefill positions followed by128 supplied-hidden-input traced decode steps; final decode context4224. Batches1,8,16,32 are separate device shapes. Planning workload:32 concurrent conversations,32768 live tokens each, typical4096 context; this is not a serving concurrency measurement.

|Layer kind|Device batch|Prefill device us|Traced decode device us|Estimated DRAM/peak|
|---|---:|---:|---:|---:|
|linear_attention|1|284743.541|2660.428|56.827%|
|linear_attention|8|1891512.151|18181.605|66.522%|
|linear_attention|16|3780949.337|35725.386|67.710%|
|linear_attention|32|unmeasured|70860.339|68.274%|
|full_attention|1|108787.730|2456.269|60.576%|
|full_attention|8|771712.277|16160.754|73.656%|
|full_attention|16|1531264.299|31819.109|74.819%|
|full_attention|32|3047594.455|63195.406|75.343%|

`performance.json` contains exact timestamps-derived measurements, roofline numerators/peaks, assumptions and artifact paths. DRAM is an estimate including separate batch-plane weight reads, not a hardware utilization counter. Layer kinds are never averaged.

The linear B32 supplemental row deliberately profiles decode only: a second full prefill crosses Tracy's32K source-location ceiling. Its4096 prefill still runs and passes correctness. Warmed prefill for both kinds is fully measured atB1.

Accepted directories under `tracy/`: linear_b1_complete, full_b1, linear_b8, full_b8, linear_b16, full_b16, linear_b32_decode, full_b32. Each retains ops.csv, filtered decode_perf_report.csv, human decode_perf_report.txt and whole_layer_timing.json; all except linear_b32_decode also retain corresponding prefill reports. Linear B16's late optional GUI failure is adjudicated in capture_validation.json with matching measured_cpp_device_rows.csv. Rejected directories linear_b1 and linear_b32 are never used for timings.
