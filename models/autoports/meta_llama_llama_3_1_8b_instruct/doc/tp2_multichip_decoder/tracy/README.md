# Tracy and tt-perf-report provenance

Source capture command (production-default BFP4/LoFi policy, batch 1, logical prefill 18):

```bash
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_BATCH=1 \
MULTICHIP_DECODER_SEQ_LEN=18 MULTICHIP_DECODER_PREFILL_REPEATS=3 \
MULTICHIP_DECODER_TRACE_REPLAYS=1 python -m tracy -r -p -v \
  -o models/autoports/meta_llama_llama_3_1_8b_instruct/doc/multichip_decoder/tracy \
  -m pytest -q -s --tb=short \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k warmed_perf
```

Canonical source CSV: `reports/2026_07_17_01_24_39/ops_perf_results_2026_07_17_01_24_39.csv`.

Each `analysis/{single,multi}_{prefill,decode}.txt` records the corresponding `tt-perf-report` command result, and each adjacent `.csv` plus `_summary.csv.csv` is the human-auditable operation table. Raw host traces, duplicate device logs, and generated PNGs were pruned after the reports were generated; they are reproducible from the command above.

Key findings over three profiled iterations:

| Path | DRAM roofline | Matmul device time | CCL device time | Main data movement |
| --- | ---: | ---: | ---: | --- |
| single prefill | 18.0%, 92 GB/s | 2,835.65 us / 15 ops, 76.39% | none | create/concat heads 10.51% |
| TP2 prefill | 20.3%, 104 GB/s | 883.35 us / 15 ops, 53.14% | AG 67.15 us + RS 66.30 us / 6+6 ops, 8.03% | create/concat heads 11.84%, norms 17.92% |
| single decode | 38.4%, 295 GB/s | 906.90 us / 15 ops, 81.71% | none | cache/tilize/untilize/reshard conversions are individually below 3% |
| TP2 decode | 27.3%, 209 GB/s | 460.35 us / 15 ops, 58.92% | AG 60.25 us + RS 46.70 us / 6+6 ops, 13.69% | binary/norm 6.96%; named DM/TM conversions each below 3% |

The all-reduce implementation appears as paired reduce-scatter and all-gather records. TP2 halves projection work enough to dominate the new CCL and data movement: final end-to-end speedups are 1.903x prefill and 1.449x traced decode. The report marks several transformer and CCL ops unclassified; their device time is still present and is included explicitly above rather than being silently assigned to compute.
