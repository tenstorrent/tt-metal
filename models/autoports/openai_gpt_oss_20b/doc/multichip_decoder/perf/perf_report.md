# GPT-OSS 20B multichip decoder performance

All accepted results use the frozen source hashes in
`../evidence_manifest.json`, a fixed four-device Blackhole P300c 1x4 ring,
and the current `tt/optimized_decoder.py` as the isolated one-device
baseline. Watcher and profiler ran in separate processes.

## Warmed wall-clock latency

Every row uses 20 warmed prefill iterations and 500 warmed trace replays.
The baseline and multichip measurements run in separate processes.

| Layer | S | 1x1 prefill ms | 1x4 prefill ms | Prefill speedup / efficiency | 1x1 decode ms | 1x4 decode ms | Decode speedup / efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 sliding | 17 | 3.6884 | 5.0955 | 0.724x / 18.10% | 0.8736 | 0.6317 | 1.383x / 34.58% |
| 13 full | 17 | 3.7451 | 5.1102 | 0.733x / 18.32% | 6.7482 | 0.6347 | 10.632x / 265.80% raw |
| 12 sliding | 128 | 12.8805 | 21.7227 | 0.593x / 14.82% | 0.7968 | 1.0581 | 0.753x / 18.83% |
| 13 full | 128 | 10.2699 | 25.2078 | 0.407x / 10.19% | 6.7544 | 1.0417 | 6.484x / 162.10% raw |

The full-layer one-device baseline reproducibly takes about 6.75 ms because
of its distinct full-attention decode path. Its raw superlinear ratios are
reported for reproducibility but are not hardware scaling or efficiency
claims. The meaningful limitations are explicit: all measured prefills and
S=128 sliding decode regress on this fixed mesh; S=17 sliding decode improves
1.383x.

Exact accepted JSONs:

- `../logs/current_optimized_perf_layer{12,13}_seq{17,128}.json`
- `../logs/final_selected_ar_ep4_layer{12,13}_seq{17,128}.json`

## Current topology matrix

All S=128 candidates use the same 20/500 whole-layer harness and pass their
focused real-weight correctness gates.

| Candidate | L12 prefill / decode ms | L13 prefill / decode ms | Why rejected |
| --- | ---: | ---: | --- |
| ring all-reduce + EP4 | 21.7227 / 1.0581 | 25.2078 / 1.0417 | selected |
| H=2880→2944 RS+AG + EP4 | 21.7068 / 1.0857 | 25.2011 / 1.0705 | decode +2.60% / +2.76% |
| attended AG + local O + EP4 | 21.7161 / 1.1911 | 25.1914 / 1.1789 | decode +12.57% / +13.17% |
| ring all-reduce + TP4 experts | 36.4951 / 1.1111 | 39.4058 / 1.0937 | prefill +68.00% / +56.33%; decode about +5% |

The carried-width-sharded residual control pads H=2880→2944, reduce-scatters
to 736 values/rank, applies distributed RMSNorm, a row-sharded router plus
logit all-reduce, and a persistent fused gather into the next real packed
QKV. It passes with minimum PCC 0.999888 and exact router top-4, but runs at
0.7824x/0.7820x the replicated boundary. This rejects delayed gather at the
next consuming boundary, not merely at a collective microbenchmark.

The Blackhole fused matmul+reduce-scatter option is source-rejected:
the production API guards Blackhole because issue #46181 documents an
M-tiles=32 producer/consumer race. The alternative attended-AG/local-O API
was still implemented and measured as shown above.

Exact candidate JSONs:

- `../logs/final_candidate_rs_ag_ep4_layer{12,13}_seq128.json`
- `../logs/final_candidate_fused_o_ag_ep4_layer{12,13}_seq128.json`
- `../logs/final_candidate_ar_tp4_layer{12,13}_seq128.json`
- `../logs/candidate_{rs_ag_pad64,fused_o_ag_natural_bf16,tp4_experts,carried_ep_residual}_layer{12,13}.json`

## Frozen-source device profile

The representative Tracy profile covers layer 12 at S=17 with one prefill
iteration and three warmed trace replays. `tt-perf-report` merges all four
devices. This profile writes `../logs/profile_final_layer12_seq17.json` and
cannot overwrite accepted timing.

### Decode

| Operation | Device-time share | Sum (us) | Calls |
| --- | ---: | ---: | ---: |
| ring all-reduce (`AllBroadcastDeviceOperation`) | 35.16% | 895.48 | 6 |
| active-expert sparse matmul | 31.63% | 805.62 | 9 |
| ordinary QKV/O/router matmul | 7.08% | 180.22 | 9 |
| reshape/view | 3.53% | 89.91 | 15 |
| paged SDPA | 0.90% | 22.84 | 3 |
| paged K/V update | 0.79% | 20.17 | 6 |
| sharded/interleaved data movement | 1.25% | 31.69 | 27 |

Category totals are 47.69% compute, 43.61% other (principally CCL), 6.66%
tensor manipulation, and 2.04% data movement. The merged modeled DRAM
roofline is 26.4%, or 135 GB/s.

### Prefill

| Operation | Device-time share | Sum (us) | Calls |
| --- | ---: | ---: | ---: |
| active-expert sparse matmul | 41.65% | 2528.56 | 3 |
| ring all-reduce | 18.19% | 1104.36 | 2 |
| reshape/view | 14.36% | 871.68 | 7 |
| unary compute | 13.53% | 821.11 | 7 |
| binary elementwise | 5.21% | 316.13 | 10 |
| ordinary matmul | 1.50% | 90.94 | 3 |
| SDPA | 0.15% | 9.12 | 1 |

Category totals are 64.14% compute, 33.15% other, and 2.72% tensor
manipulation. The merged modeled DRAM roofline is 107.5%, or 550 GB/s. This
is an aggregate four-device model, not one device exceeding its physical
peak.

The profile confirms the decision: communication is the largest decode
component, but every currently supported attempt to replace or defer the
ring reduction increased whole-layer latency. Sparse active-expert matmuls,
not dense all-expert work, dominate the remaining compute.

## Reproduction

```bash
# Capture current 1x1 baseline, then matching selected 1x4 timing.
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PERF_SEQ=128 \
MULTICHIP_DECODER_PREFILL_REPEATS=20 MULTICHIP_DECODER_TRACE_REPLAYS=500 \
pytest -q \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_capture_current_single_chip_optimized_perf'

RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PERF_SEQ=128 \
MULTICHIP_DECODER_PREFILL_REPEATS=20 MULTICHIP_DECODER_TRACE_REPLAYS=500 \
MULTICHIP_PERF_RESULT_PATH='models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/final_selected_ar_ep4_layer{layer_idx}_seq{seq_len}.json' \
pytest -q \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf'

# Frozen selected-path profile.
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PERF_SEQ=17 \
MULTICHIP_DECODER_PREFILL_REPEATS=1 MULTICHIP_DECODER_TRACE_REPLAYS=3 \
MULTICHIP_PERF_RESULT_PATH='models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/profile_final_layer{layer_idx}_seq{seq_len}.json' \
python -m tracy -r -p --check-exit-code \
  -o models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/perf/tracy_final_frozen \
  -n gpt_oss_20b_tp4_ep4_seq17_final -m pytest -q \
  'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf[blackhole-1x4-12]'
```

## Profiler artifacts

- Raw op CSV:
  `tracy_final_frozen/reports/gpt_oss_20b_tp4_ep4_seq17_final/2026_07_23_07_18_45/ops_perf_results_gpt_oss_20b_tp4_ep4_seq17_final_2026_07_23_07_18_45.csv`
- Human-readable operation tables:
  `decode_final_frozen.csv`, `prefill_final_frozen.csv`
- Operation summaries:
  `decode_final_frozen_ops_summary.csv.csv`,
  `prefill_final_frozen_ops_summary.csv.csv`
- Category summaries:
  `decode_final_frozen_summary.csv.csv`,
  `prefill_final_frozen_summary.csv.csv`

Raw Tracy traces and device logs remain local and are intentionally excluded
from the checkpoint because they are hundreds of MiB. The raw op CSV and
compact derived tables are the stage provenance.
