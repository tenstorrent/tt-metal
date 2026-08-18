# `tt-perf-report` Gemma analysis

This directory contains `tt-perf-report` 1.2.8 output generated from:

```text
../captures/gemma4-31b-prefill-trace-1x4.csv
```

The source capture is a four-device Blackhole Gemma 4 31B traced-prefill run. The measured region is explicitly bounded by the `start` and `stop` signposts.

## Install and run

The tool is maintained in Tenstorrent's separate [`tt-perf-report`](https://github.com/tenstorrent/tt-perf-report) repository and published on PyPI. The recommended isolated installation is:

```bash
pipx install tt-perf-report
```

Generate the detailed operation table and operation summary:

```bash
tt-perf-report \
  tracy_guide_docs/captures/gemma4-31b-prefill-trace-1x4.csv \
  --start-signpost start \
  --end-signpost stop \
  --no-color \
  --csv tracy_guide_docs/tt_perf_report/gemma4-prefill-detailed.csv \
  --summary-file tracy_guide_docs/tt_perf_report/gemma4-prefill-by-op
```

Useful variations:

```bash
# Aggregate compute, data movement, tensor manipulation, and other work.
tt-perf-report <ops.csv> \
  --start-signpost start --end-signpost stop \
  --group-by category \
  --summary-file category-summary

# Preserve each device instead of merging corresponding mesh operations.
tt-perf-report <ops.csv> \
  --start-signpost start --end-signpost stop \
  --no-merge-devices \
  --summary-file per-device-summary

# Focus on operation IDs 100 through 200.
tt-perf-report <ops.csv> --id-range 100-200
```

When the capture uses paired signposts, pass both names explicitly. The default selects the last signpost and is intended for tests that place one marker immediately before their final performance pass.

## Generated files

- `gemma4-prefill-detailed.csv`: one row per logical operation after four-device merging, with device time, op-to-op gap, utilization, bottleneck classification, and advice.
- `gemma4-prefill-by-op.csv` / `.png`: operation-name summary and stacked visualization.
- `gemma4-prefill-by-category.csv` / `.png`: compute/data-movement/tensor-manipulation summary.
- `gemma4-prefill-by-memory.csv` / `.png`: operation and input-0 memory-layout summary. All operations in this capture use DRAM-interleaved input 0, so this split is identical to the operation summary.
- `gemma4-prefill-per-device.csv` / `.png`: summaries without cross-device merging.
- `*.log` and `gemma4-prefill-report.txt`: format detection, signpost selection, architecture detection, warnings, and output paths from each invocation.

## What the tool adds

`tt-perf-report`:

- Detects this report as CSV v2.1, Blackhole, with 120 available worker cores.
- Selects named signpost ranges and filters operation IDs.
- Merges corresponding operations from multiple devices.
- Derives device time, op-to-op gaps, DRAM bandwidth, FLOP utilization, fidelity, bottleneck labels, and generic optimization advice.
- Exports detailed tables, grouped summaries, and stacked PNG plots.
- Accepts multiple CSVs from the same multi-host workload and offsets their device IDs before combining them.

For multi-device merging, it uses the maximum duration among devices for each non-collective logical operation. It uses the average duration for AllGather, ReduceScatter, and AllReduce operations. This differs from simply summing one device or taking the slowest device's total.

## Gemma findings

The signpost window contains 7,248 raw device rows: 7,220 rows from measured replay session 3 plus 28 device rows without a replay-session ID. Four-device merging produces 1,812 logical operation rows.

Timing decomposition:

- Signposted wall time: 96.984 ms.
- Merged logical-operation device time: 82.986 ms.
- Reported op-to-op gaps: 12.564 ms.
- Device time plus gaps: 95.550 ms.
- Remaining difference from signpost wall time: approximately 1.434 ms.

The largest gap is 11.403 ms immediately before the first `EmbeddingsDeviceOperation`; the embedding kernel itself takes only 8.307 µs. This indicates launch/orchestration time at the start of the measured replay, not a slow embedding kernel. The report therefore explains most of the previously observed difference between signposted latency and summed kernel time.

Operation summary:

- Matmul: 40.307 ms, 48.57%.
- LayerNorm: 19.767 ms, 23.82%.
- ReduceScatter: 5.087 ms, 6.13%.
- AllGather: 4.161 ms, 5.01%.
- NlpCreateHeads: 3.326 ms, 4.01%.
- Binary operations: 2.658 ms, 3.20%.
- Rotary embedding: 2.500 ms, 3.01%.
- SDPA: 2.250 ms, 2.71%.

Category summary:

- Compute: 67.952 ms, 81.88%.
- Data movement: 9.248 ms, 11.14%.
- Tensor manipulation: 5.520 ms, 6.65%.
- Other: 0.267 ms, 0.32%.

Additional observations:

- All 301 merged matmul rows are classified as DRAM-bound by the tool.
- Matmul's device-time-weighted FLOP utilization is 21.17%, with individual rows ranging from 2.07% to 23.98%.
- The modeled overall DRAM roofline is 38.1%, or 195 GB/s. This aggregate includes modeled operations across the window; it does not contradict individual matmuls exceeding the tool's 65% DRAM-bound threshold.
- The largest matmul shape is `128 x 5376 x 5376`: 180 calls and 26.210 ms, about 65% of all matmul time.
- LayerNorm accounts for most low-core work: 261 four-core rows totaling 18.658 ms. A low core count is a lead to investigate, not proof that a wider program configuration will be faster.
- Full-window per-device totals range from 82.211 to 83.056 ms, a 0.845 ms or approximately 1.03% spread. Overall device work is well balanced even though individual collective totals vary by device.
- `PagedFillCacheDeviceOperation` is not classified by version 1.2.8 and appears under `Other`. Its timing remains present; only category labeling is affected.

The tool recommends DRAM-sharded matmul program configurations for the DRAM-bound rows. Treat this as a hypothesis: benchmark the exact shapes and topology, then confirm end-to-end signposted latency and PCC before keeping a change.

## Why these totals differ from `analyze_tracy_csv.py`

The local analyzer defaults to replay session 3 only and reports per-device sums, using the slowest device total as the critical-path estimate. That produces 80.814 ms for the slowest device and 38.538 ms of matmul work.

This `tt-perf-report` run analyzes the entire `start`/`stop` window, which includes 28 additional raw device rows, and merges each logical operation across devices using operation-specific rules. It therefore reports 82.986 ms of merged device time and 40.307 ms of matmul work.

Neither view is inherently wrong:

- Use replay-session filtering and per-device totals to inspect the traced replay's device critical path.
- Use the explicit signpost window in `tt-perf-report` to decompose end-to-end measured work, gaps, roofline metrics, and optimization leads.
