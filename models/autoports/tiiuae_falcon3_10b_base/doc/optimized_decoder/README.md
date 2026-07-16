# Falcon3-10B-Base optimized decoder

This directory contains the single-device optimization evidence for the repo-local
`tiiuae/Falcon3-10B-Base` TTNN decoder. The selected implementation is
`tt/optimized_decoder.py`; its measured prefill and decode methods are owned by
`OptimizedDecoder` and do not call the functional implementation as a fallback.
The work is intentionally decoder-layer-only: no multichip decoder, block stack,
full model, generator, or vLLM path is included.

## Selected path

- Blackhole P300c, one device through a `1x1` TTNN mesh.
- Packed QKV; separate SwiGLU gate and up projections; composite TTNN SDPA.
- BFP4_B projection weights with LoFi compute and BFP8_B KV caches.
- Prefill activations and weights are DRAM interleaved, with tuned 2D matmul
  configs at normal prompt sizes. Attention matmuls use TTNN-selected configs
  above 1,024 flattened rows and the MLP uses on-device chunks of at most 1,024
  rows, preserving arbitrary logical sequence lengths and the 6,528-token contract.
- Decode projection weights are DRAM width-sharded. QKV/output use shape-derived
  grids; gate/up/down use 24-core-target activation shards for BFP4 and 48-core-
  target shards for wider precisions. These labels describe the L1 shard geometry;
  the DRAM-sharded matmul factory itself reports its fixed 12 workers in Tracy.
- Decode keeps working activations in width-sharded L1 through each material
  sub-block and exposes the established DRAM-interleaved layer boundary.

## Correctness

All selected real-weight checks use HF revision
`34bb99a889fe0426412da3dd2b46e6f64c8fd003` and the representative dense layer
20. Falcon3-10B-Base has 40 decoder layers with the same dense topology, so this
is the only meaningful layer kind.

| Check | Functional bar / baseline | Selected optimized |
|---|---:|---:|
| Recorded prefill, seq 17 | PCC >= 0.99 | 0.99998229 |
| Recorded prefill, seq 31 | PCC >= 0.99 | 0.99994827 |
| Recorded prefill, seq 128 | PCC >= 0.99 | 0.99980095 |
| Decode output, position 31 | PCC >= 0.99 | 0.99931311 |
| Decode output, position 32 | PCC >= 0.99 | 0.99922617 |
| Real layer-20 prefill vs HF | 0.99879820 functional | 0.99998229 |
| Real layer-20 decode vs HF | 0.99851787 functional | 0.99999667 |
| Key/value cache after seq 31 | PCC >= 0.99 | 0.99655404 / 0.99496777 |

The selected semantics test covers non-aligned lengths 17 and 31, length 128,
paged cache fill/update, adjacent decode positions 31 and 32, and eight repeated
decode calls with bitwise-identical results. A full optimized layer and persistent
cache pass at batch 32, sequence 6,528. The public context capability is unchanged;
the updated `../context_contract.json` records the BFP8 cache and bounded large-M
strategy.

## Performance

These are final-default reruns with genuine layer-20 weights and recorded
activations. Decode is warmed, captured, and replayed through a TTNN trace.

| Batch 32 path | Warmed prefill (ms) | Traced warmed decode (ms) | Decode PCC |
|---|---:|---:|---:|
| Functional BF16 | 4.192523 | 4.201479 | 0.99999297 |
| Advisor BFP8 candidate | 4.606521 | 1.008138 | 0.99999457 |
| Advisor all-BFP4 candidate | 3.272251 | 0.804016 | 0.99999672 |
| DRAM all-BFP4, 16-core-target control | 3.299052 | 0.809589 | 0.99999659 |
| DRAM all-BFP4, 48-core-target control | 3.331253 | 0.799102 | 0.99999670 |
| DRAM all-BFP4, attention HiFi2 | 3.289344 | 0.830651 | 0.99999666 |
| DRAM all-BFP4, MLP HiFi2 | 3.995518 | 1.116212 | 0.99999667 |
| **Selected DRAM all-BFP4 LoFi, automatic 24-core target** | **3.278021** | **0.793496** | **0.99999667** |

The selected batch-32 decode is 81.1% faster than functional, 1.31% faster than
the exact advisor BFP4 candidate, and 0.70% faster than the strongest 48-core-
target DRAM control. The final prefill result uses 11 timed samples rather than
the original three; it beats functional by 21.8% and is within 0.18% of the
independently fastest advisor-prefill result. The decode target determines the
default, and all final numbers above come from that post-review rerun.

| Batch 1 path | Traced warmed decode (ms) |
|---|---:|
| Optimized BF16 | 1.433387 |
| DRAM BFP8, 48-core target | 1.042073 |
| Advisor all-BFP4 | 0.683947 |
| DRAM all-BFP4, 16-core target | 0.684719 |
| DRAM all-BFP4, 48-core target | 0.675394 |
| DRAM all-BFP4, attention HiFi2 | 0.706141 |
| DRAM all-BFP4, MLP HiFi2 | 0.991401 |
| **Selected DRAM all-BFP4 LoFi, automatic 24-core target** | **0.668364** |

The selected batch-1 path is 53.4% faster than the BF16 optimized control, 2.28%
faster than the advisor BFP4 candidate, and 1.04% faster than the 48-core-target
DRAM control.

## Profiler conclusions

The separate Tracy run captured two warmed prefill iterations and three traced
decode replays. Prefill accounts for 2.6861 ms device kernels plus 1.1974 ms
device gaps per iteration (3.8835 ms modeled total, 4.0240 ms wall); modeled DRAM
roofline utilization is 16.0% or 82 GB/s. Decode accounts for 0.7818 ms kernels
plus 0.0327 ms gaps (0.8146 ms modeled, 0.8262 ms wall); modeled DRAM utilization
is 29.7% or 152 GB/s. Decode's remaining dominant rows are gate/up/down at roughly
132/133/122.5 us and QKV/output at roughly 38/27 us. Profiler rows verify BFP4_B
weights and LoFi fidelity on all five dominant projection roles.

The bytes-derived theoretical decode floor is 0.26528 ms: 237,502,464 BFP4
projection elements occupy 133,595,136 bytes at 576 bytes/tile, and tile-padded
K/V reads through position 17 add 2,228,224 BFP8 bytes; 135,823,360 mandatory
bytes divided by 512 GB/s is 0.26528 ms. This is 32.1% of measured wall latency,
consistent with the report's 29.7% modeled figure after non-matmul operations.

The decode report contains four necessary reshards: head-concat to output
projection, output to residual, residual to the 24-core MLP working grid, and MLP
output back to residual. The exact RoPE-row extraction contains the remaining
untilize/slice/tilize work. There are no runtime `torch`, `from_torch`, `to_torch`,
or host fallback operations in the measured implementation. See `work_log.md` for
the topology table, advisor decisions, complete candidate evidence, and checklist.

## Evidence map

- `pytest_results.xml`: ordinary optimized suite, 5 passed and 10 manual gates skipped.
- `results/final/final_batch32.json` and `final_batch1.json`: final default reruns.
- `results/precision/recorded_seq31_precision_frontier.json`: real-weight dtype/fidelity frontier.
- `results/topology_viable/candidate_sweep.json` and `results/cores_ab_1/candidate_sweep.json`: topology and 24-vs-48-core searches.
- `results/geometry_extended/`: runnable 16-core control and selected-DRAM attention/MLP fidelity controls; `logs/geometry_extended_6c.xml` and `_12c.xml` preserve their exact L1 blockers.
- `shard_advise/report.json` and `shard_advise/final_ir.mlir`: mandatory advisor output.
- `tracy/dense_layer/ops.csv`, phase CSVs, and `results/final/profile_summary.json`: profiler evidence.
- `watcher.log` and `logs/watcher_selected_semantics.xml`: separate watcher-clean run.
- `logs/context_capacity_6528_retry.xml`: passing optimized context-capacity rerun.
- `activations/layer20_inputs.safetensors`: genuine HF layer-20 inputs for lengths 17, 31, and 128.

## Core reproduction commands

Preserve the node's existing device reservation when running these commands. This
pass used `TT_VISIBLE_DEVICES=2,3`; each test requests a `1x1` mesh.

```bash
python -m pytest -q models/autoports/tiiuae_falcon3_10b_base/tests/test_optimized_decoder.py -s

FALCON3_RUN_FINAL_PERF=1 \
FALCON3_RESULTS_DIR=models/autoports/tiiuae_falcon3_10b_base/doc/optimized_decoder/results/final \
python -m pytest -q models/autoports/tiiuae_falcon3_10b_base/tests/test_optimized_decoder.py::test_warmed_prefill_and_traced_decode_candidates \
  models/autoports/tiiuae_falcon3_10b_base/tests/test_optimized_decoder.py::test_batch1_traced_decode_candidates -s

FALCON3_RUN_CAPACITY=1 \
python -m pytest -q models/autoports/tiiuae_falcon3_10b_base/tests/test_optimized_decoder.py::test_selected_decoder_context_capacity -s

TT_METAL_WATCHER=10 \
python -m pytest -q models/autoports/tiiuae_falcon3_10b_base/tests/test_optimized_decoder.py::test_selected_decoder_semantics_cache_and_repeated_decode -s
```

The advisor and Tracy commands, including required environment bootstrapping, are
recorded verbatim in `work_log.md`.
