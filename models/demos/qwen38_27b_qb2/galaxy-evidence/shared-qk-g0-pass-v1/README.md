# Eight physical TP4 replicas: shared-Q/K source qualification

G0 passed Oct 8 at 16:29:48 UTC, including clean process exit and one passing
JUnit test. Total test time 2565.76 seconds, dominated by sequential loading.
All 32 distinct physical chips participated in eight TP4 groups.

Five isolated and five concurrent decode measurements per group; output
agreement passed. Concurrent/isolated TPOT ratios range 0.999951-1.000226,
within the unchanged 1.03 maximum for every group. Concurrent TPOT ranges
28.548-30.629 ms at **B1, short prompt**. JIT telemetry reported 803/803 hits.

B1 retains the fused-normalization fallback. This check qualifies the frozen
model source and chip partition; it does not measure B16/B32 shared-Q/K
throughput or long-context Galaxy scaling. The 32K/B32 Galaxy projection of
2989.6 output tok/s remains a projection from one TP4.

`full-model.json` contains inputs, source hashes, group bindings and repeated
measurements. `hardware.xml.gz` is the passing process receipt. The startup
snapshot shows the dependent GPQA controller accepted this qualification and
began serving startup; it is not an evaluation result.
