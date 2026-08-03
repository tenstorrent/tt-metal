# Optimized decoder work log

## Scope and hardware

Only `tt/optimized_decoder.py`, its tests, and its docs are stage-owned. This
stage does not start multichip, full-model, or vLLM work. Batch 32 is
explicitly out of scope and was never measured or gated.

- Blackhole, one device from a four-device host.
- `/dev/tenstorrent/{0,1,2,3}` exist; bounded 1x1 open/close passed.
- `tt-smi` is unavailable.
- Hardware commands were serialized; watcher and profiler were separate.
- MPI warned that `/dev/shm` had about 17.6 MB free. All bounded runs closed
  cleanly; unrelated shared-memory objects were not removed.
- Revision: `d11e61a842617a22dc328552fa5bb86231ee4f37`.

## Operation-topology audit

Decode is logical batch 1 with one valid row padded to one tile.

| Fused topology | Candidates | Final action | Evidence |
|---|---|---|---|
| DRAM RMSNorm/residual and interleaved dense matmuls | L1 width-sharded norm/residual; DRAM-sharded weights; K-block and fidelity cross-product | coherent 8-bank sharded chain, BFP8/LoFi | 0.320013 to 0.173888 ms; real PCC 0.999259 |
| packed QKV, head creation, RoPE, fused cache update, paged SDPA, concat, O | BFP8/BFP4, HiFi2/LoFi, explicit SDPA, DRAM-sharded MoE attention | dense sharded; MoE interleaved; default SDPA | actual MoE sharding 0.567961 vs 0.543472 ms; BFP4 attention PCC 0.956701 |
| packed dense gate/up, fused SwiGLU, down | final-topology BFP8/BFP4, LoFi/HiFi2, per-role K blocks | BFP8/LoFi, K=8/12 | BFP4 0.158435 ms but PCC 0.960960; selected 0.173888 |
| router, top-k, active-8 gate/up, all-128 down | exact active-8 down; separate exact-count mask; router layout | exact active-8 both projections | final layer 1/4 0.551025/0.530140 ms; zero-score trace PCC 1.0 |
| MoE reshape/transpose/unary movement | matched L1 chain; in-place score multiply; output tile h=1/16 | matched 32x32 L1 gate/up-to-down chain | 0.541365 to 0.530140 ms on layer 4; smaller output-only tiles fail because in0 remains 32-row |
| repeated same-input q/k/v and gate/up | packed projections | retain fused packing | inherited fused evidence/commits `cd728b63774`, `2e7e734ba3d` |
| dominant prefill all-expert gate/up/down | explicit 48/64; legal 24/32 with K blocks 2, 4, 8, 16/12 | 24/32, K block 8 | layer 4 8.003327 ms candidate; seq-1025 layer-1 PCC 0.999516 |
| BF16 paged cache and composite attention | BFP8 cache; explicit SDPA | retain BF16/default | BFP8 0.287444 vs 0.285974; explicit prefill SDPA regressed |

No decoder LM head, embedding, CCL, or multi-device operation exists.

## Candidate evidence

### Sparse expert precision and geometry

All rows are warmed traced layer-1 batch-1 decode unless stated.

| Candidate | Mean | Decision |
|---|---:|---|
| fused all-128 down | 2.129856 ms; fresh 2.131925 | baseline |
| BF16 active-8, 24/32 | 0.651831 | correct, superseded |
| BFP8 LoFi 24/32 | 0.568965 | selected family |
| BFP8 HiFi2 24/32 | 0.575001 | slower |
| BFP8 LoFi 48/64 | 0.606247 | slower |
| BFP8 LoFi 16/16 | 0.591436 | slower |
| BFP4 LoFi 24/32 | 0.548248 | real PCC 0.987855, reject |
| BFP4 LoFi 48/64 | 0.544294 | same PCC failure |
| BFP4 LoFi 16/16 | 0.581779 | slower and same PCC failure |
| BFP8 router paired control | 0.545637 vs BF16-router 0.548070 | select BFP8 |
| BFP8 output activation | 0.622046 | slower than BF16 activation |

Layer-1 selected experts remain
`[107, 119, 126, 14, 61, 79, 18, 20]`, PCC 0.999721. Layer 4 requires DRAM
router output: L1 changed a near-boundary selection and gave real PCC 0.9798;
DRAM gives prefill/decode PCC 0.999650/0.999604. Layer 1 retains faster L1
router output because it preserves the target top-8.

The exact sparse mask scatters ones at top-k indices independently of sigmoid
scores. Twenty adversarial trace replays with all routing scores underflowed
to zero passed PCC 1.0 and watcher.

### Attention, cache, and SDPA

| Candidate | Mean | Correctness / decision |
|---|---:|---|
| attention BFP8 HiFi2 | 0.299020 | PCC 0.999728, selected for MoE |
| attention BFP8 LoFi | 0.299955 | slower in interleaved topology |
| attention BFP4 LoFi | 0.299699 | PCC 0.956701, reject |
| BF16 cache | 0.285974 | selected |
| BFP8 cache | 0.287444 | correct but slower |
| explicit decode SDPA 8x8 | 0.285896 | no material gain |
| explicit prefill SDPA 8x8 | 0.583761 vs 0.573566 | regression |
| actual DRAM-sharded MoE QKV/O | 0.567961 vs selected 0.543472 | 4.5% slower |

The MoE sharding row is reproducible: policy
`dram_sharded_moe_attention` creates BFP8 DRAM-width-sharded QKV/O weights
and matching L1 activation/output shards for sparse layers.

### Final-topology dense cross-product

The one-tile in0 height is legal for the DRAM-sharded family. Initial errors
were remediated rather than used as rejection: first match the input L1 shard,
then keep output L1-sharded because a DRAM output circular buffer is illegal.

| Step/candidate | Mean | Decision |
|---|---:|---|
| legal sharded QKV/O/gate-up/down | 0.271725 | pass |
| largest legal K blocks, HiFi2 | 0.266238 | pass |
| sharded norm/residual | 0.244340 | pass |
| retain sharded QKV output | 0.220759 | pass |
| K-block cap 4 | 0.233759 | reject |
| final topology BFP8/LoFi | 0.173888 | real PCC 0.999259, select |
| final topology BFP4/LoFi | 0.158435 | real layer-0 prefill/decode PCC 0.966624/0.960101, reject |

Per-role final K blocks are QKV=8, O=16, gate/up=8, and down=12. Core count is
fixed to the eight banks in Blackhole's one-row DRAM grid. The selected
`MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig` exposes K block,
per-core M/N, and fused activation; it does not expose output-subblock knobs.
That API constraint explains `tt-perf-report`'s missing-subblock advice.

### Dominant prefill programs

The exact-final profiler attributes about 56% to all-expert gate/up and 25%
to down. This class was attacked:

- Explicit 48/64 cores needs gate/up grid `(12,4)`, which exceeds Blackhole's
  worker-grid x=11. Since sequence 128 has four M tiles, legal grid-y divisors
  are 1, 2, and 4, so no alternate 48-core 2D rectangle fits.
- Legal explicit 24/32 cores with K block 2 ran at 9.741679 ms but failed its
  sequence-1025 PCC at 0.968541.
- The adapted larger-K sweep measured block 4 at 8.546047 ms, block 8 at
  8.003327 ms, and gate/up 16 plus down 12 at 8.105063 ms on layer 4.
  Block 8 passed non-aligned sequence-1025 layer-1 PCC at 0.999516 and was
  selected. Final layer-1/layer-4 walls are 8.124654/8.090731 ms.

See `candidate_prefill_program_summary.json` and its linked logs.

### Retained MoE movement

The final graph has no host operations. Top-k returns tiled values/indices;
scatter requires row-major index/value inputs; sparse matmul requires a
32-row tiled in0 and outputs expert-batched tiles; A-sparse down requires
`[1, experts, M, K]`. Those contracts require the observed
untilize/scatter/tilize and transpose/reshape boundaries.

Two output-only lower-movement trials, heights 1 and 16, fail the kernel's
exact requirement that output tile height equal the 32-row in0 tile. The
public device `to_layout`, `tilize`, and `tilize_with_val_padding` bindings
have no output-tile argument; their C++ tilize path uses the input tile or
the default 32x32 tile. Thus a device-native smaller matched retile is not
available without adding a new kernel outside this stage's file scope.

The adapted matched trial instead keeps the existing 32x32 input/output tile
and makes SwiGLU, reshape, transpose, and down input/output L1-resident. It
passes real layer-4 prefill/decode PCC 0.999650/0.999604 and improves layer-4
decode from 0.541365 to 0.530140 ms. Logs are
`candidate_sparse_tile_h{1,16}_decode_layer4.log`,
`candidate_sparse_l1_chain_decode_layer4.json`, and
`sparse_l1_chain_real_layer4.log`.

A completion audit rechecked the apparent layer-1 0.543472-ms historical
point with a paired 100-sample run. The exact final default matched L1 chain
measured 0.550780 ms versus 0.561696 ms for the same policy without the L1
chain, a 1.94% win. The historical point was cross-run variance rather than
a reproducibly stronger configuration. Evidence:
`layer1_l1_chain_completion_audit.json`.

## Final performance and profiling

| Layer | Fused decode | Final decode | Fused prefill | Final prefill |
|---|---:|---:|---:|---:|
| 0 dense | 0.320013 | 0.174017 | 0.580181 | 0.575725 |
| 1 sliding MoE | 2.129856 | 0.551025 | 10.079501 | 8.124654 |
| 4 full MoE | 2.131000 | 0.530140 | 10.067067 | 8.090731 |

| Path | Device us | Gap us | Ops | Roofline | Same-run profiler wall |
|---|---:|---:|---:|---:|---:|
| dense decode | 153.390 | 23.257 | 25 | 48.1%, 246 GB/s | 191.916 us |
| dense prefill | 537.843 | 14.789 | 17 | 16.2%, 83 GB/s | 603.945 us |
| sliding decode | 522.982 | 45.990 | 50 | 21.9%, 112 GB/s | 588.410 us |
| sliding prefill | 7975.713 | 35.829 | 33 | 20.5%, 105 GB/s | 8059.170 us |
| full decode | 497.373 | 47.557 | 46 | 23.0%, 118 GB/s | 563.952 us |
| full prefill | 7864.111 | 30.990 | 31 | 20.8%, 106 GB/s | 7955.774 us |

Headline walls come from uninstrumented 20/50-sample runs; profiler walls
come from separate two-iteration instrumented runs. Device plus reported op
gaps reconciles most of each instrumented wall; the remainder and small
negative wall deltas are host launch/signpost/profiler overhead plus
run-to-run variance. No host op appears inside signposts.

Active BFP8 weight traffic is 37,748,736 bytes dense and about 56,885,248
bytes MoE, ideal 73.7/111.1 us at 512 GB/s. Raw Tracy scratch was removed.
The final4 MoE bundle retains gzip-compressed raw ops CSV, filtered report
CSV, human table, summary/plot, advice, and capture log. The unchanged dense
path retains its filtered report, human table, summary/plot, and capture log.

## Correctness, stress, watcher, capacity

- PCC bar: 0.995.
- Real selected layer 0: prefill 0.999772, traced decode 0.999259.
- Real selected layer 1 decode: 0.999721.
- Real selected layer 4 versus functional: prefill 0.999650, cache-consuming
  traced decode 0.999604.
- Synthetic sparse layer 1/4 prefill: 0.999516/0.999577.
- Synthetic sparse layer 1/4 trace: 0.999821/0.999800.
- Non-aligned prefill: 1, 31, 32, 33, 65; sparse 1025 and 33.
- Deterministic physical cache/replay: batch 2 and 4.
- Sliding history: 4097 and populated cache.
- Exact-final context-500000 decode, position 499999: layer 0 44.054 ms,
  layer 1 0.935 ms, layer 4 44.475 ms; all finite.
- BF16 KV bytes at context 500000: 1,024,000,000.
- Exact-final normal: 21/21 in 67.51 seconds.
- Exact-final watcher: 21/21 in 77.44 seconds; the gzip-compressed
  22,905-line raw watcher log is clean of fatal/assert, illegal NoC, timeout,
  hang, and stuck markers.
- Batch 32 not run; inherited capability unchanged.

## Commands

Candidate:

```bash
python models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_perf.py \
  --candidate dram_sharded_dense_lofi --mode decode --batch 1 --layer 0 \
  --iterations 20 --json-out <artifact.json>
```

Profiler, separately from watcher:

```bash
python -m tracy -r -o <artifact-dir> -n decode_batch1_final \
  models/autoports/coherelabs_north_mini_code_1_0/tests/optimized_decoder_perf.py \
  --mode decode --batch 1 --layer 0 --warmups 3 --iterations 2

tt-perf-report <ops.csv> --start-signpost PERF_DECODE \
  --end-signpost PERF_DECODE_END --no-color --no-host-ops \
  --active-experts 8 --csv decode_batch1_perf_report.csv \
  --summary-file decode_batch1_perf_summary.csv
```

Correctness/watcher:

```bash
pytest -q models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py \
  --junitxml=models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/pytest_results.xml

TT_METAL_WATCHER=10 TT_METAL_WATCHER_APPEND=1 \
TT_METAL_LOGS_PATH=models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/watcher_logs_final \
pytest -q models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py \
  --junitxml=models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/watcher_pytest_results.xml
```

## `$optimize` checklist

- [x] Started from the fused decoder.
- [x] Audited same-input projections, composite ops, movement, residuals, and
  expert topology before local tuning.
- [x] Crossed BFP8/BFP4 and HiFi2/LoFi with final dense sharding and PCC.
- [x] Swept legal per-role K blocks and documented the fixed 8-bank geometry
  and no-subblock API constraint.
- [x] Swept sparse 16/16, 24/32, and 48/64 geometries in BFP8 and BFP4.
- [x] Measured an actual reproducible MoE DRAM-sharded attention candidate.
- [x] Preserved packed QKV/gate-up and selected exact active-expert down.
- [x] Measured SDPA/cache candidates and dominant prefill programs.
- [x] Attacked retained MoE movement with two output-tile candidates, a
  matched L1-resident chain, and source-level evidence for the smaller-retile
  blocker.
- [x] Collected exact-final `tt-perf-report` separately for prefill/decode and
  reconciled device, gap, profiler wall, and headline wall time.
- [x] Proved selected real-weight prefill/decode across layers 0, 1, and 4.
- [x] Proved exact-count sparse `nnz` under adversarial zero routing scores.
- [x] Preserved non-aligned sequences, cache behavior, determinism, layer
  kinds, and context 500000.
- [x] Ran exact-final stress and watcher-clean suites (21/21 each).
- [x] Obtained independent `$stage-review` clean-pass after final4 artifact
  reconciliation.
- [x] Committed stage-owned implementation and evidence locally; never
  pushed.

N/A: LM head, CCL/multidevice, full-model generation, vLLM, and batch 32.

## Review remediation and commits

The first independent review returned more-work-needed. Every finding was
addressed:

- wired and measured actual MoE sharding;
- separated exact sparse mask from routing values and added adversarial trace;
- crossed final dense topology with K blocks, fidelity, and BFP4;
- attacked dominant prefill programs and MoE movement with hard evidence;
- added real selected-path layer-0/layer-4 prefill and cache-consuming decode;
- corrected stage-state metadata and same-run profiler accounting.

The final fresh xhigh rereview returned `clean-pass` with no required work.
It independently checked the real BFP4 rejection, matched L1 chain,
larger-K prefill sweep, context/correctness/watcher evidence, and final4
profiler bundle.

Local stage implementation/evidence commit:
`36e38ca877824a280e8aa4fe09762f4b2e454ce5`. The documentation-only SHA
record commit is `8445e440f6863d3a0d74ac03704256d41032e34a`.
The completion-audit evidence commit follows these commits and is reported
in the final handoff. No push was made.
