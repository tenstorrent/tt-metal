# Exact custom-kernel attribution of the completed compact profile

The four `GenericOpDeviceOperation` compute hashes were matched to cached
`chlkc_math.cpp` files. Removing only their three-line generated wrapper gives
byte-identical expanded source from the GPQA-qualified frozen compact snapshot.
The generated sources, SHA256 values and mapping are retained. This identifies
actual compiled kernels, rather than inferring roles from call order.

`python3 analyze.py` checks the retained CSV parts and generated-source hashes,
then reproduces the table. Every stage has 48 calls on each of four ranks in each
of three model-trace replays. Their per-rank totals reconcile exactly with the
previous 9.1845405-ms generic-operation inventory.

| Stage | Median kernel sum | Reader, including waits | Compute, including waits |
|---|---:|---:|---:|
| Recurrence | 4.4051 ms | 4.4035 ms | 4.3685 ms |
| Epilogue | 2.8531 ms | 2.8521 ms | 2.7378 ms |
| Convolution | 0.9959 ms | 0.9958 ms | 0.9775 ms |
| Direct preparation | 0.9309 ms | 0.9307 ms | 0.7819 ms |

These are instrumented kernel sums, not additive wall time, active-compute
percentages or physical DRAM/NoC counters. The matched whole-step profiler
overhead was 4.43%; do not subtract it uniformly from individual stages.

The next epilogue hypothesis is to avoid initializing unused tile rows for
compact output: the current reader clears 8 FP32 tiles plus 8 BF16 tiles per
worker invocation, while compact writers emit only the live row and explicitly
zero output padding. Source inspection suggests 1-2 ms of full-step opportunity,
about 2-4% throughput near 20 TSU, but this is **unmeasured**. It needs poisoning
and wrap/replay tests to prove unused rows cannot affect live results. Public
output retains its padding contract. No reader change is included in the running
combined-policy experiment.

For recurrence, repeated V-tile clearing and waiting for state DMA before L1
broadcast preparation remain separate hypotheses. Its full measured stage is
4.41 ms, so reader improvements cannot plausibly recover the entire 16.58-ms gap
to 30 TSU. Attention and weight-delivery redesign still need attention as well.
