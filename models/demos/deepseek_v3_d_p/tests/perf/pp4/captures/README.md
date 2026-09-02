# Single-layer Tracy captures — Mistral Small 4, PP=4 x (8,1) vs single-rank

The profiling data behind `models/demos/deepseek_v3_d_p/docs/MISTRAL4_PP4_VS_SINGLE_RANK.md` §2,
committed so the results can be read and re-analysed **without a galaxy**.

Captured 2026-08-31 on `bh-glx-110-a04u02`: 1 layer per stage, driven through the real chunked runner
so the KV cache actually deepens — 8 chunks x 5,120 = 40,960 context, eager, LM head enabled.

| file | what it is |
|---|---|
| `report_<n>.txt` | rendered `tt-perf-report` op table from the **full** capture, plus this repo's two analyzer summaries. Read this first — no tooling needed. |
| `ops_<n>.csv.gz` | the `ops_perf_results` rows, **all devices**, reduced to the columns the analyzers here read. |

`<n>` is `pp4_stage0..3` (rank 0 = first layer + embedding, rank 3 = last layer + norm + LM head) and
`1rank` (SP8 x TP4).

## Re-analyse

```bash
S=models/demos/deepseek_v3_d_p/tests/perf/pp4
gunzip -c $S/captures/ops_pp4_stage0.csv.gz > /tmp/s0.csv
python3 $S/analyze_layer_budget.py /tmp/s0.csv "stage 0"   # per-layer budget, socket waits excluded
python3 $S/analyze_kv_ramp.py      /tmp/s0.csv "stage 0"   # per-chunk ramp -> the cost of KV depth
```

These reproduce the document's numbers exactly (11.89 / 14.63 / 11.97 / 16.17 ms per layer for the
four PP stages, 5.59 ms for single-rank).

## What was dropped, and why it matters

The raw captures are **13 GB** (a 461 MB `profile_log_device.csv` and a 544 MB `tracy_ops_times.csv`
per rank) and the `ops_perf_results` CSVs alone are 7.5–33 MB, because they carry **457 columns**.
Committed here: all rows and all devices, six columns.

* **`tt-perf-report` will NOT run on `ops_<n>.csv.gz`** — it needs many of the dropped columns. Use
  `report_<n>.txt`, which was rendered from the full capture, or re-run the profile.
* **Cross-device skew is preserved** (every device's rows are kept), which is why the analyzer numbers
  match. Subsetting to a single device does *not* work: on the 32-device single-rank capture, device 0
  alone reads 6.51 ms/layer against the 5.59 ms all-device mean, a 16% error.
* Full captures for this run live outside git under `<repo>/mistral4_perf_profile/` on the machine
  that produced them.

## Reading these correctly

* **`InboundSocketServiceSyncOperation` is ~99% of device time in a PP stage and is not transport** —
  it is the receiver blocking until upstream data arrives. `analyze_layer_budget.py` excludes it and
  reports it separately; `tt-perf-report` does not, so its "Total %" column is dominated by idle.
* **One row per device.** A stage spans 8 chips running concurrently, so an op's wall cost is the max
  across devices, not the sum.
* **Never sum the four stage reports.** The sum is roughly one chunk's latency through the pipeline;
  throughput is `1 / max(stage)`.
* Eager and instrumented, whereas the end-to-end numbers are traced: read durations and ratios, not
  op-to-op gaps or absolute wall time.
