# Epilogue padding result and qualification batching

Collected October 11, 2026, shortly after 02:00 UTC from 10.228.203.98. The
persistent epilogue controller completed successfully at 01:56:22 UTC, with
clean teardown. No hardware experiment remained running in the checked user
systemd queue. Earlier v2 epilogue/prefill followers failed their predecessor
gate and were not live queued work.

[Controller receipt](receipts/gdn-epilogue-padding-v3/queue.json) records 18
shape/layout/memory comparisons, exact correctness and stable zero/skip/zero
controls. The hardware test also compares 4096 changing-input real-weight
updates at B16/B32. Source and command provenance are in
[the launch record](receipts/gdn-epilogue-padding-control-v3/launch.json) and
its adjacent frozen manifest. The controller took about 70 seconds; pytest's
test call took 60 seconds, with a warm JIT cache. The initial 5-15 minute
estimate was conservative.

For the B16 packed-offset compact L1 boundary, zero-padding initialization
took **66.2537 us**, versus **39.3360 us** when unused input rows were left
uninitialized: **1.6843x kernel speedup**. Drift was 0.0567%. This projects to
**1.2920 ms across 48 layers**, about **+0.53 TSU** from a 49.913-ms baseline.
No full-model saving or new GPQA score has been measured for this change.
Precision, live-word math and destination-padding behavior are unchanged.

The earlier resident-state/compact-gate combination measured **+0.3240 TSU**.
The updated qualification controller recomputes absolute TSU gain and defers
separate G0/API/GPQA for gains below 1 TSU. Its focused 22-test CPU gate and
recalculation of the actual combined receipt are retained under
`receipts/batched-qualification-cpu-v1`. Small candidates will be combined;
isolated estimates must not be added and described as a measured model gain.

No serving promotion, AgentX launch or new full qualification was performed.
The separate full-64-layer prefix/SSD correctness test also completed, but its
serving adapter and transfer performance remain open work.

Next design: [GDN fusion](../../experiments/GDN-FUSION-PLAN.md).
