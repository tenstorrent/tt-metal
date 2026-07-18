# Tracy and tt-perf-report evidence

The canonical capture profiles the final default policy at batch 1 and logical
prefill length 18.  It uses three warmed prefill iterations and one decode
trace replay inside Tracy to keep the raw artifact bounded; absolute speedup
comes from the uninstrumented long run in `../logs/perf_final.log`.

## Capture

```bash
TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=0 \
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_VARIANT=default \
MULTICHIP_DECODER_BATCH=1 MULTICHIP_DECODER_SEQ_LEN=18 \
MULTICHIP_DECODER_PREFILL_REPEATS=3 MULTICHIP_DECODER_TRACE_REPLAYS=1 \
python -m tracy -r -p -v -o \
  models/autoports/meta_llama_llama_3_1_8b_instruct/doc/multichip_decoder/tracy \
  -m pytest -q -s --tb=short \
  models/autoports/meta_llama_llama_3_1_8b_instruct/tests/test_multichip_decoder.py \
  -k warmed_perf
```

Capture log: `../logs/tracy_capture.log`.

Canonical retained op CSV archive:

`reports/2026_07_18_09_34_42/ops_perf_results_2026_07_18_09_34_42.csv.gz`

The archive is lossless; its uncompressed SHA-256 is
`4cc2908fa199b76a2db8e914154b143d18d1e920b561a9d0370b22f0e9b3569a`.
The raw CSV is compressed only to satisfy the repository's 500 KB artifact
limit. The filtered CSVs below remain directly readable.

Reproducible Tracy host traces and profiler-device logs were pruned after the
op CSV and reports were verified.  Generated PNG summaries were also pruned;
their CSV equivalents are retained.

## Report generation

Reconstruct the raw CSV, then run the two commands below for these four
signpost pairs:

| Prefix | Start | End |
| --- | --- | --- |
| `single_prefill` | `PERF_SINGLE_PREFILL` | `PERF_SINGLE_PREFILL_END` |
| `single_decode` | `PERF_SINGLE_DECODE` | `PERF_SINGLE_DECODE_END` |
| `multi_prefill` | `PERF_MULTI_PREFILL` | `PERF_MULTI_PREFILL_END` |
| `multi_decode` | `PERF_MULTI_DECODE` | `PERF_MULTI_DECODE_END` |

```bash
PROFILE_RAW_CSV=$(mktemp --suffix=.csv)
gzip -dc \
  reports/2026_07_18_09_34_42/ops_perf_results_2026_07_18_09_34_42.csv.gz \
  > "$PROFILE_RAW_CSV"

tt-perf-report "$PROFILE_RAW_CSV" \
  --start-signpost "$START" --end-signpost "$END" \
  --no-color --csv "analysis/${PREFIX}.csv" \
  --summary-file "analysis/${PREFIX}_summary" \
  > "analysis/${PREFIX}.txt" 2>&1

tt-perf-report "$PROFILE_RAW_CSV" \
  --start-signpost "$START" --end-signpost "$END" \
  --no-color --no-summary \
  > "analysis/${PREFIX}_table.txt" 2>&1

rm -f "$PROFILE_RAW_CSV"
```

Each prefix therefore has:

- `${PREFIX}.csv`: filtered per-op CSV;
- `${PREFIX}_summary.csv`: category/op summary CSV;
- `${PREFIX}_table.txt`: full human-readable per-op report table plus format,
  signpost, architecture, roofline, and advice;
- `${PREFIX}.txt`: nonempty CSV/summary generation provenance.

The current report tool also emits summary PNGs. They were verified against
the retained summary CSVs and pruned; the CSVs are the reviewable authority.

## Acceptance summary

| Region | Matmul | Norm | Reduce-scatter | All-gather | Modeled DRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| single prefill | 76.33% | 8.02% | n/a | n/a | 17.9%, 92 GB/s |
| single decode | 81.59% | 2.33% | n/a | n/a | 38.8%, 298 GB/s |
| TP4 prefill | 40.19% | 23.51% | 11.15% | 6.63% | 13.6%, 70 GB/s |
| TP4 decode | 36.20% | 4.08% | 20.76% | 12.67% | 15.5%, 119 GB/s |

TP4 decode also spends about 7.20% in explicit data movement and 7.26% in
tensor manipulation.  Communication, not DRAM bandwidth, is the largest new
TP4 cost.  The policy sweep consequently retained two-link BF16 collectives;
one link and BFP8 both made warmed decode slower.  The exact selected and
rejected timings are in `../candidate_results.csv`.  The filtered projection
rows identify `Input 1 Datatype=BFLOAT4_B` and `Math Fidelity=LoFi`, confirming
that these are measurements of the selected weight and math policy.

Every final TP4 projection is labeled `SLOW`, and the DRAM-sharded rows have a
blank output-subblock field.  This was treated as an optimization finding, not
dismissed.  [../profiler_geometry_audit.md](../profiler_geometry_audit.md)
derives the factory's hidden output subblocks and 80-core bounding launch, then
records full-decoder eight-way DRAM and explicit-subblock interleaved 1-D
comparisons.  Both same-policy alternatives are slower than the final default.
