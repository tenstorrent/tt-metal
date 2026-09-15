# Why streaming and FP32 have similar full-prefill runtime

September 9, 2026. Same candidate, P100A, B=1/H=4/S=262144/D=128,
causal, Q/K chunks 128/512, HiFi2, shifted Q preprocessing, seed 1236.

Hardware counter capture, not an estimate from FLOP count:

| Metric | BF16 destination / streaming | Improved FP32 destination |
|---|---:|---:|
| Profiled device kernel duration, ms | 1583.773 | 1637.694 |
| Full-chip average FPU active cycles, % | 14.375 | 14.573 |
| Full-chip average SFPU active cycles, % | 4.405 | 26.887 |
| Full-chip average combined math active cycles, % | 15.446 | 41.460 |
| Per-core FPU utilization median, % | 16.337 | 16.788 |

The chip reports 120 maximum compute cores; the program uses 110. The full-grid
metrics divide summed activity by 120 cores and the entire operation duration,
including inactive cores and completion tails. Normalizing instead by the 110
used cores over the same duration gives FPU utilization 15.68% and 15.90%.
The per-core median uses each core's own counter reference duration and is a
different normalization. These are instruction-active-cycle metrics, not a
percentage of useful peak FLOP/s. Combined math is the OR of FPU and SFPU
activity, not their sum. The profiler header reports 1350 MHz.

Streaming really does save substantial compute work, especially SFPU activity.
It simply does not appreciably reduce the dominant wall-time cost here.

## Bandwidth evidence

In `reader_interleaved.cpp`, each causal Q chunk independently reads its needed
K/V chunks from DRAM. KV chain forwarding is explicitly noncausal-only. For
2048 Q chunks/head and 512 K chunks/head, each successive group of four Q chunks
reads 1, 2, ..., 512 K/V chunks. Therefore K/V payload read volume is:

```
4 heads * [4 * sum(1..512)] * [2 tensors * 512 rows * 128 elements * 2 bytes]
= 550,829,555,712 bytes
```

This excludes the much smaller Q reads/output writes and protocol overhead.
Dividing this source-derived traffic by measured kernel duration implies
347.8 GB/s for streaming and 336.3 GB/s for FP32. The P100A is specified at
[448 GB/s GDDR6 bandwidth](https://tenstorrent.com/hardware/cards), so streaming
is already delivering approximately 78% of the advertised peak as K/V payload.

Together with low FPU occupancy and the much larger SFPU activity in FP32,
this strongly suggests a DRAM/NoC delivery bottleneck hiding most compute
savings. This capture measures FPU/SFPU counters, not DRAM-controller traffic;
the bandwidth numbers are calculated from the reader, not hardware BW counters.
Contention, synchronization and unequal core completion times can also matter.
This is not proof that DRAM alone accounts for every idle cycle.

Thus the earlier +3.4% accuracy cost is shape/bandwidth dependent, not evidence
that the refined compute path is intrinsically only 3.4% more expensive.
Changing Q chunk size or enabling KV reuse would be useful next diagnostics;
neither was changed for this comparison.

## Reproduction and raw evidence

Using the same remote environment as REPORT.md:

```bash
python_env/bin/python -m tracy -r -p \
  --profiler-capture-perf-counters=fpu --check-exit-code \
  -o experiments/sdpa-l2/hifi2-rounding/utilization-profile \
  tests/ttnn/unit_tests/operations/sdpa/repro_sdpa_l2.py \
  --kv-lens 262144 --causal --heads 4 --variants hifi2 fp32_hifi2 \
  --q-round-bits 6 --q-prescale 1.0027 --q-bitceil --seed 1236 \
  --label utilization-profile \
  --output experiments/sdpa-l2/hifi2-rounding/utilization-profile.jsonl
```

One full operation per path was profiled; their hardware durations agree with
the earlier nine-replay unprofiled medians (1583.821/1637.917 ms). Compilation
and the host reference/preprocessing are outside the device kernel duration.
Report rows follow the specified order: BF16 first, FP32 second.

Raw report: `utilization-profile/reports/2026_09_09_18_42_18/ops_perf_results_2026_09_09_18_42_18.csv`.
Raw device counters and timings: `profile_log_device.csv` in the same directory.
Counter definitions: `tt_metal/tt-llk/docs/performance_counters/performance_counters.md`,
Compute Utilisation; aggregation: `tools/tracy/perf_counter_analysis.py` and
`tools/tracy/process_ops_logs.py`.
