# Gemma-4 26B A4B multichip decoder

## Result

`MultichipDecoder` is a real TP=4 implementation for the local four-P300C
`1x4` ring. It inherits `OptimizedDecoder` as its single-chip numerical and
orchestration baseline, fractures weights at load time, keeps the stack
residual replicated, and ring-reduces only row-parallel O/down partials. It is
limited intentionally to this mesh and is ready to serve as the decoder-layer
baseline for the later full-model stage.

The final implementation keeps gate-selected top-8 expert execution through
the inherited `ttnn.sparse_matmul` path. No dense all-expert candidate is used.

## Mesh and tensor plan

The pre-code plan, exact shapes, padding calculations, cache ownership,
collective choice, memory envelope, and rejected alternatives are in
[`mesh_plan.md`](mesh_plan.md). The selected local shapes are:

| Role | Global logical | Per P300C |
|---|---:|---:|
| Q heads | 16 x 256/512 | 4 x 256/512 |
| Sliding KV | 8 x 256 | 2 x 256 |
| Full KV | 2 x 512 | one head; duplicated within Q-rank pairs |
| Dense intermediate | 2112 (padded 2176) | 544 |
| Expert intermediate | 704 (padded 768) | 192 |
| Residual | 2816 | replicated 2816 |

QKV, dense gate/up, and expert gate/up are column-sharded. O, dense down, and
expert down are row-sharded and followed by a ring all-reduce. Norm and router
tensors are small and replicated. The per-device cache uses one full KV head or
two sliding KV heads; page tables and current-position tensors are replicated.

## Correctness and contracts

All commands below used `GEMMA4_RANGE_DOWNLOAD=1` and
`TTNN_CONFIG_OVERRIDES='{"throw_exception_on_fallback": true}'`.

| Gate | Sliding layer 0 | Full layer 5 |
|---|---:|---:|
| HF vs TP prefill PCC | 0.9986126386 | 0.9970876597 |
| HF vs TP decode PCC | 0.9996529166 | 0.9997855512 |
| Optimized TTNN vs TP prefill/decode | PCC >= 0.995, pass/pass | PCC >= 0.995, pass/pass |
| Non-aligned logical prefill | S=33, pass | S=33, pass |
| Advertised-context decode | position 262143, pass | position 262143, pass |
| Non-aligned capacity prefill | S=262143, 30.588 s | S=262143, 53.285 s |
| Replayed decode | bit exact | bit exact |
| Traced batch-32 vs optimized TTNN | PCC >= 0.995, bit exact | PCC >= 0.995, bit exact |

The S=33 trace test validates replicated stack input/output, replicated page
tables and current positions, exact local cache shapes, and all four output
ranks. Full attention additionally verifies ranks 0/1 share KV head 0, ranks
2/3 share KV head 1, and the two heads remain distinct. The advertised-context
test uses a rolled page table, nonzero cache baseline, three device-read
sentinels, and verifies preservation after replay. `doc/context_contract.json`
keeps the advertised 262144 context with no capability reduction and records
the TP memory consequences. A dedicated sliding-cache modulo test compares
logical lengths 1024 and 1025: slot 0 is replaced while slots 1..1023 remain at
PCC 0.9999, proving the intended bounded full-stack allocator.

Important artifacts:

- `artifacts/pcc_layer{0,5}_*.json`: real-weight HF oracle results.
- `artifacts/optimized_reference_layer{0,5}.pt`: separately captured optimized
  TTNN prefill/decode reference tensors used for direct TP comparison.
- `artifacts/{optimized,multichip}_batch32_layer*.{pt,json}`: same-harness
  batch-32 correctness, trace determinism, cache shape, and latency evidence.
- `artifacts/evidence_manifest.json`: wrapper commands plus source, test, and
  optimized-reference SHA-256 provenance.
- `artifacts/advertised_context_decode_*.json` and
  `artifacts/prefill_capacity_*_262143.json`: context and cache capacity.
- `artifacts/trace_*_batch1.json`: trace, layout, cache, and latency evidence.

## Warmed decode performance

Thirty trace replays followed five warmups at batch 1. Single-chip numbers are
the current optimized-stage baselines recorded in `doc/optimized_decoder`.

| Layer kind | Optimized 1-chip | TP4 | Speedup | TP efficiency |
|---|---:|---:|---:|---:|
| Sliding attention | 1.272 ms | 1.122 ms | 1.133x | 28.33% |
| Full attention | 1.270 ms | 1.220 ms | 1.041x | 26.02% |

The final like-for-like batch-32 harness measures both implementations with the
same inputs, cache population, five warmups, 30 replays, and current source:

| Layer kind | Optimized 1-chip | TP4 | Speedup | TP efficiency |
|---|---:|---:|---:|---:|
| Sliding attention | 14.842 ms | 12.691 ms | 1.169x | 29.24% |
| Full attention | 14.908 ms | 12.813 ms | 1.163x | 29.09% |

The modest scaling is expected for a single decode layer at batch 1: sparse
expert work is reduced fourfold, but three hidden-width ring reductions remain
on the critical path. This is still faster for both layer kinds and reduces
per-device weight/KV ownership, so it is preferable to replicated single-chip
execution for the full stack.

## Profiler audit

`artifacts/perf/{sliding_attention,full_attention}` contains the analyzed
four-device CSV, human-readable table, and grouped summary/PNG.
`artifacts/perf/provenance.json` records exact capture/report commands.
The `decode_batch32_{sliding,full}` directories are signpost-delimited to one
warmed trace replay and are the acceptance reports for CCL/data movement.

- `tt-perf-report` modeled aggregate DRAM roofline is 6.4% (33 GB/s) for
  sliding and 5.4% (28 GB/s) for full attention: this path is not near the
  modeled DRAM roof.
- Ring reductions are visible as `AllGatherDeviceOperation` followed by
  `FastReduceNCDeviceOperation`; the captured workload contains 480 per-device
  all-gather rows. No host fallback is present.
- Sparse experts remain the dominant decode compute: the raw captures contain
  `SparseMatmulDeviceOperation`, while attention, dense matmuls, CCL, and data
  movement are separately visible for audit.
- Signposted batch-32 decode-only reports show 7.4% modeled DRAM roofline
  (38 GB/s) sliding and 7.3% (37 GB/s) full, with CCL all-gather, sparse expert,
  matmul, cache, attention, and data-movement rows isolated to replay.
- Those batch-32 reports are sparse/layout dominated: approximately 53% is
  sparse matmul, 26% transpose/unary layout work, and 1.35% CCL. The selected
  six-tile expert-down block is the largest legal divisor of TP-local K; the
  smaller 1/2/3-tile alternatives add K iterations. The communication-limited
  observation below applies to batch 1, not to the batch-32 profile.

## Commands and safety evidence

Core gates:

```bash
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py::test_multichip_real_weights_prefill_decode
pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py::test_multichip_matches_optimized_single_chip
GEMMA4_FUNCTIONAL_DECODER_CONTEXT=1 pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py::test_multichip_advertised_context_traced_decode
GEMMA4_PREFILL_CAPACITY_LENGTH=262143 pytest -q models/autoports/google_gemma_4_26b_a4b_it/tests/test_multichip_decoder.py::test_multichip_prefill_capacity
```

Watcher was run with `TT_METAL_WATCHER=10` and
`TT_METAL_WATCHER_DISABLE_ETH=1`: Blackhole fabric active-Ethernet firmware is
too large when Ethernet watcher instrumentation is included, while worker-core
watcher coverage remains enabled. Both real-weight layer kinds and both
non-aligned traced layer kinds passed (4/4), with no watcher model error; see
`artifacts/watcher/results.xml` and `pytest_disable_eth.log`.

## Limitations

- Only a `1x4` P300C mesh is supported by design.
- The full-context and serving-batch axes are independent maxima. A batch of 32
  users all simultaneously holding 262144 full-attention tokens exceeds board
  DRAM; this does not reduce the public single-user context contract.
- Decode TP efficiency is communication-limited at batch 1. A reduce-scatter
  residual does not lower bytes here: `$autofix` calculated 270336 bytes/device
  for either ring all-reduce or reduce-scatter plus the mandatory next gather,
  before extra distributed-norm statistics. Every next projection retains
  global K=2816, and Blackhole fused matmul+RS is disabled for nondeterministic
  race #46181; details and source templates are in `mesh_plan.md`.
- The conservative full-stack envelope is 19.411328125 GiB/device (6.8 GiB
  weights, 2.548828125 GiB KV, 64 MiB trace, 6 GiB activation/workspace, and
  4 GiB program/allocator reserve), leaving 12.588671875 GiB on each 32 GiB
  P300C. See `context_contract.json` and `mesh_plan.md` for shape arithmetic.
