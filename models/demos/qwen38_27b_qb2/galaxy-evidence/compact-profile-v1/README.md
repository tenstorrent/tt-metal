# Completed compact full-model profile

The October 10 capture passed three restored-state replays on all four ranks,
with identical input, source, precision and output hashes between the unprofiled
and profiled arms. The unprofiled median was **50.0873 ms**, and the profiled
median **52.3057 ms**, or **4.43% whole-step profiler overhead**. This uses real
weights and populated synthetic caches at B16/32K; natural-prompt performance
remains 49.9128 ms / 20.035 TSU from the separate qualified run.

The ordinary timing export exceeded its 8-GiB limit after the hardware test
passed. The already queued CPU-only recovery exported the retained capture
successfully at 21:38:58 UTC. The inventory finished at 21:39:14 UTC. No hardware
rerun or reset was required for that export. Original failure receipts remain.

[Complete inventory](receipts/operator-profile-v1/report/INVENTORY.md) covers
33 operation types and 2553 device-operation records per rank/replay. These are
not program counts. Median per-rank kernel sums identify the largest targets:

| Operation family | Kernel sum per step |
|---|---:|
| Matmul | 18.9871 ms |
| SDPA decode | 12.6267 ms |
| Custom GDN/conv/epilogue kernels | 9.1845 ms |
| All-reduce | 2.6468 ms |

These sums are not an additive critical-path budget. In particular, the generated
firmware interval partition's overlap bucket represents overlapping lifetimes,
including waits; it does not demonstrate useful compute overlap. A small
exclusive attention interval does not mean attention costs only that interval.
Reader/compute/writer durations include waits and are not utilization counters.
The four custom-kernel hashes are retained in `generic-patterns.json`; individual
stage attribution is not yet established by this report.

Weight-byte estimates put output projection at 56.7%, MLP down at 67.6%, and
MLP gate/up at 82.3% of an assumed 512-GB/s/chip peak. Those estimates assume one
read of each padded BFP8 weight tile, exclude activation/extra transactions,
and are not physical bandwidth measurements.

`capture.json` records hashes for all collected artifacts. Reassemble its CSV
parts in order and gzip-decompress to reproduce the raw CSV. The receipts here
are a snapshot from before the first recovered projection experiment failed;
see [the subsequent L1 failure and recovery](../projection-l1-recovery-v1/README.md)
for the newer queue state. Neither profile collection nor the new experiments
change the qualified precision or serving configuration.
