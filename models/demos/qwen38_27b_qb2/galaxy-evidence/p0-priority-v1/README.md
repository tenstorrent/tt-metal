# B16/32K full-model trace reconciliation

Completed October10,2026 at05:39:31UTC on one TP4 replica. BFP8 weights/KV,
BF16 activations and FP32 recurrent state. All64 real-weight layers, device
sampling and token-history recording are included. Caches contain deterministic
synthetic values; this is performance attribution, not natural-prompt accuracy.

The unprofiled restored-state trace measured67.240/67.253/67.240ms, matching
the separate natural-prompt sweep's67.245ms. Profiling measured
72.240/72.290/72.259ms: median instrumentation overhead7.46%. All profiled and
unprofiled logits/token hashes match exactly.

All64 layers, four ranks and three replays have complete coverage. Each rank
has5001 device-operation records per replay; this is not a program count.
The longest device spans are71.948/71.957/71.945ms. Differences from the same
fenced host steps are0.403/0.461/0.433%, passing the5% reconciliation gate.
Firmware intervals are partitioned into disjoint layer time, overlaps and gaps.
No cross-chip clock subtraction or sum of rank durations is used.

| Operation family | Median profiled kernel sum per rank/replay |
|---|---:|
| Layout, padding, slicing and conversion | 22.58ms |
| Matmuls | 18.89ms |
| SDPA | 12.63ms |
| Other kernels, including recurrence, norms and collectives | 14.82ms |

The first family includes tilize/untilize, slice, reshape, fill-pad, pad,
reshard, typecast, concat, transpose, interleaved/sharded conversion and QKV
head creation. Its entire time is not necessarily removable. Profiler overhead
is not necessarily uniform across families; do not rescale each row by7.46%.
Median uncovered inter-operation gap is0.336ms. Device sampler disjoint time
is0.481ms. RISC intervals include waiting and do not prove NoC congestion.
The profile supports prioritizing graph fusion and compact intermediates.

## Capture failure and recovery

Both hardware tests passed and closed their devices. Enlarging profiler program
capacity to8192 enabled the retained full capture, but ordinary CSV export
exceeded the8-GiB file cap. The default exporter emits every CPU zone and only
later filters to TT_DNN/TT_METAL operations in pandas.

A persistent CPU-only recovery exported TT_ zones directly, retaining all
messages/signposts and the original compact device timing report. The host
timing export became11,827,326bytes; messages are10,659,892bytes. Optional
host child-function timing is omitted. The original304-MiB capture and partial
8.3-GiB export were not modified. No repeated hardware run was needed.

The original priority service resumed the exact experiment parent automatically.
Offline recovery service `qwen38-p0-export-recovery-v1-20261010` completed.
Complete raw capture/results are also copied to host disk at
`/home/ttuser/qwen38-artifacts-20261007/p0-profile-result-v1`.

The broader P0 gate remains explicitly false: these results do not qualify
all contexts, active compute utilization, hardware DRAM counters or TP8 costs.
They do complete the requested B16/32K full-model timing reconciliation.

[Analysis](completed/analysis.json), [terminal recovery receipt](completed/queue.json),
[hardware receipt](completed/profile.json), [raw-file hashes and reconstruction](completed/capture.json).

The published CPU-test `control/unit.xml` has a final newline added by the
repository formatter. `control/unit.raw.xml.gz` preserves the exact original
bytes; both are included in the evidence inventory.
