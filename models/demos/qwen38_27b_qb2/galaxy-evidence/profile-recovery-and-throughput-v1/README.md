# Full-model throughput, physical partial-query test and profile recovery

Collected October 8, 2026 UTC. `collection.json` records original byte lengths,
SHA256 hashes, remote paths and persistence state. Gzip files preserve original
bytes; source captures and failed receipts remain unchanged on the host.

## 32K/batch-32 full-model pair

Both arms completed three measured repetitions with clean device close. Prompt
and runtime source hashes match except the explicit recurrence configuration.
Precision remains BFP8 KV, BF16 activations, FP32 recurrent state and the existing
BFP4 matrix weights. Each arm repeats its own output hash; this is not an eval.

| Per TP4 replica | Native recurrence | Single-step recurrence |
|---|---:|---:|
| Output tokens/s, excluding prefill | 250.133 | 350.417 |
| Tokens/s/user | 7.817 | 10.951 |
| TPOT, ms | 127.932 | 91.320 |
| Input tokens/s | 5184.788 | 5185.078 |
| TTFT p50, seconds | 202.505 | 202.499 |

Decode throughput improves **40.092%**. The eight-replica projection is 2803.34
output tokens/s/Galaxy; physical Galaxy scaling and model-eval requalification
remain open. The user's 2680 reference is a checkpoint, not a stopping target.

## Accurate partial-query attention on physical TP4

All 30 full/partial/full cases completed, with ten qualified comparisons and
clean close. Every candidate passed the original numerical checks and was
bit-identical to its full-query control. Production instructions were used;
the simulator compatibility fallback was disabled. Timing drift stayed below
0.2%. These are synthetic attention-call measurements, not model speedups.

| Context | Batch | Full query, us | Partial query, us | Throughput change |
|---|---:|---:|---:|---:|
| 16K | 8 | 249.406 | 233.025 | +7.03% |
| 16K | 16 | 433.543 | 418.944 | +3.48% |
| 16K | 32 | 756.705 | 746.397 | +1.38% |
| 32K | 8 | 432.174 | 414.433 | +4.28% |
| 32K | 16 | 809.898 | 792.769 | +2.16% |
| 32K | 32 | 1480.025 | 1467.550 | +0.85% |
| 128K | 8 | 1559.418 | 1531.773 | +1.80% |
| 128K | 16 | 3044.449 | 3054.634 | -0.33% |
| 262016 | 4 | 1510.786 | 1528.777 | -1.18% |
| 262016 | 8 | 3055.848 | 3020.870 | +1.16% |

Partial-query is not a global default: gains are small at high batch, with two
long-context regressions. Full-model boundary/quality validation remains open.
Placement gains from the separate sweep are not assumed additive.

## Profiling collector correction

At 06:53 UTC the v3 hardware test passed and generated its compact device report,
but the collector rejected absent compute timings on data-movement-only ops.
The source/hash lists were empty and all three compute binary sizes were zero.
Those ops have no compute kernel; the original collector incorrectly required
an interval anyway. Reader/writer and total kernel timings were present.

The fix marks compute intervals not applicable only with all five provenance
fields proving absence. Unknown metadata and missing timings from actual compute
kernels still fail. Regression checks include each missing/contradictory field.
The frozen remote suite passed **292 tests plus 40 subtests**. A local attempt
could not run pytest because that tooling environment did not contain it.

Reanalysis of a copy of the original capture passes all gates for all four ranks:
161 device-op rows per rank, including 96 proven data-movement-only rows.
No original raw capture was edited and no hardware rerun was needed for recovery.
At 07:25:25 UTC the remaining profiling work was relaunched as
`qwen38-bounded-layer-profile-v4-20261008.service`: 12 captures, shared device
lock, 12-hour service cap, 64-GiB host-memory cap and bounded output files.
It was verified active with `Linger=yes`; SSH/session disconnects do not stop it.
Reboot recovery is not configured. The capacity pair also remained active.

## Roofline interpretation

The [current roofline discussion](../../experiments/PLAN-STATUS.md) uses an
optimistic memory-traffic lower bound at an assumed 512 GB/s/chip. It is not a
complete compute/communication roofline or measured DRAM-counter utilization.

The recovered profile is a warm eager **native-recurrence**, real-weight,
synthetic-cache two-layer diagnostic at 32K/B16. It must not be labeled a
single-step profile or extrapolated by summing firmware intervals: for example,
a slice's firmware interval includes waiting for preceding attention, while
its own kernel interval is about 2 us. Per-RISC intervals overlap and include
waits. Full-model traced timing reconciliation remains outstanding.
