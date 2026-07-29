# North-Mini optimized decoder

This directory records the single-device optimization of
`CohereLabs/North-Mini-Code-1.0` revision
`d11e61a842617a22dc328552fa5bb86231ee4f37`. The implementation is
`tt/optimized_decoder.py`. It preserves the functional decoder's prefill,
decode, paged KV-cache, trace, determinism, batch, and arbitrary logical
sequence-length contracts. This stage contains no multichip, full-model, or
vLLM work.

## Selected runtime

- Decode keeps the residual stream width-sharded in L1 through RMS norm,
  packed QKV, attention, residual, and dense-MLP boundaries. QKV and output
  projections use DRAM-width-sharded weights and explicit DRAM-sharded matmul
  programs. Batch 1 and batch 32 have independently swept shard/program
  contracts.
- Attention weights are BFP8/LoFi. Dense non-expert MLP decode weights are
  BFP4/LoFi; phase-specific prefill weights remain BFP8/HiFi2 because BFP4
  failed authentic non-aligned prefill.
- MoE routing is BF16/HiFi2. Sparse active-expert weights are BFP8/LoFi.
  Dense serving-batch expert weights are independently selected BFP4/LoFi:
  the branch-proven selected matrix passes two active-sparse prefill and six
  dense-BFP4 rows, while a separate forced-dense BFP4 matrix passes all eight
  rows. BFP4 is materially faster than BFP8.
- Batch-1 decode and prefill execute active experts. Prefill groups routed
  tiles in chunks of 24, keeps intermediate values in L1, and performs
  device-side down projection, routing, and reduction. Batch 32 retains the
  fastest correct device-resident dense-expert path after AutoFix and a
  compatible full-chain DRAM-sharded sweep.
- Prefill uses DRAM-interleaved activations and explicit large-prefill
  programs where legal. Logical lengths are padded/chunked internally; the
  public API has no `seq_len % chunk == 0` restriction. For very large token
  matrices the runtime uses TTNN's automatic program with interleaved QKV/O
  weight copies because an explicit 500k program requested 77 MB of circular
  buffers per core. The copies are present for every batch so large
  multi-user prefill preserves the same public contract.
- Paged cache fill/update and paged SDPA stay device-resident TTNN composites.
  The cache remains BF16 with the functional page size and layout.

`OptimizedDecoder` subclasses `FunctionalDecoder` only for public validation
and shared contract helpers. Its measured forward, attention, norm, dense
MLP, and MoE methods are overridden. Path-audit tests forbid a functional
math fallback.

## Correctness and capacity

Final normal run:

```bash
pytest -q -s --timeout=900 \
  --junitxml=models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/artifacts/review3_full.xml \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py
```

Result: `30 passed, 16 skipped in 308.39s`. The skips are opt-in
DRAM-sharded candidate cases, not selected-path coverage.

| Check | Final evidence |
|---|---:|
| authentic layer-1 prefill b1 / b32 | PCC 0.999428 / 0.999428 |
| authentic layer-1 decode b1 / b32 | PCC 0.997995 / 0.997995 |
| authentic layer-4 prefill b1 / b32 | PCC 0.999990 / 0.999941 |
| authentic layer-4 decode b1 / b32 | PCC 0.997234 / 0.997234 |
| active-expert non-aligned prefill, layers 1 / 4 | sampled full-output PCC 0.99867 / 0.99871 |
| dense non-aligned lengths 1/31/33/65 | PCC 0.99913–0.99914 |
| traced layer kinds, batches 1/32 | PCC 0.99837–0.99927 |
| repeated trace replay, 10 runs | deterministic; PCC 0.999398 |
| permuted paged cache, positions 5/17/31/63 | K/V PCC 0.99957–0.99970 |

The acceptance bar is PCC `>= 0.995`. The authentic dense-expert BFP4 matrix
is the selection gate; synthetic random-weight BFP4 results remain diagnostic
only and do not override equivalent target-weight evidence.

Optimized prefill was also run at the advertised limit:

| Layer kind | Logical context | Result | Single pass |
|---|---:|---|---:|
| dense/full/forced-RoPE | 500,000 | finite output | 159,869.562 ms |
| sliding/RoPE/MoE | 499,999 | finite output | 193,125.821 ms |
| full/no-RoPE/MoE | 499,999 | finite output | 347,770.355 ms |

The batch-1 BF16 KV allocation is 1.024 GB at context 500,000. Final-policy
optimized batch-32 decode allocates and replays the 32.768-GB advertised
cache at position 499,999 in 131.105 ms. `doc/context_contract.json` records
both prefill and decode optimized evidence; supported context remains
500,000.

## Warmed performance

Means below use sequence 128, three warmups, and 20 samples. Decode measures
complete-forward trace replay. Functional values are the stage-entry
baselines.

| Layer kind | Phase | Functional b1 | Optimized b1 | Functional b32 | Optimized b32 |
|---|---|---:|---:|---:|---:|
| dense/full/forced-RoPE | prefill | 0.636 ms | 0.516 ms | 13.758 ms | 4.708 ms |
| dense/full/forced-RoPE | decode | 0.356 ms | 0.187 ms | 6.652 ms | 0.252 ms |
| sliding/RoPE/MoE | prefill | 14.908 ms | 14.191 ms | 147.182 ms | 139.959 ms |
| sliding/RoPE/MoE | decode | 9.528 ms | 0.792 ms | 11.122 ms | 2.220 ms |
| full/no-RoPE/MoE | prefill | 14.655 ms | 14.264 ms | 146.699 ms | 139.855 ms |
| full/no-RoPE/MoE | decode | 9.524 ms | 0.795 ms | 11.129 ms | 2.215 ms |

Batch-1 decode improves 1.90x for dense and about 12x for MoE. Batch 32
improves 26.4x for dense and about 5.0x for MoE; no serving-batch row
regresses. Exact distributions and selected policies are in
`candidates/review3_final_runtime/`.

## Operation-topology audit

| Area | Current topology / candidates | Action and evidence |
|---|---|---|
| Q/K/V | three same-input linears; packed QKV | Packed QKV selected, removing two weight-stream dispatches. |
| dense gate/up | packed versus separate same-input projections | Separate wins 0.2535/0.3188 ms versus 0.2628/0.3401 ms at b1/b32. |
| output/layout chain | 8/12/16/32-core coherent shards, direct inputs, reshards | 16 cores selected. Four required boundary conversions total about 6.4 us; eliminating them regressed the full layer. |
| cache update | fused versus separate K/V updates | Fused b1; separate b32. Initial overlapping-grid error was adapted with disjoint grids before comparison. |
| attention | explicit versus framework-selected paged SDPA | Framework-selected program wins 0.1865/0.2521 ms versus 0.1919/0.2545 ms. |
| dense precision | BF16/BFP8/BFP4 and LoFi/HiFi2 | Decode MLP BFP4/LoFi selected; BFP4 attention and prefill fail authentic PCC, so BFP8 retained there. |
| prefill programs | default, 8x8, 10x10, chunked, automatic large-M | 10x10 aligned large-prefill and 512-row compatibility chunks selected; automatic large-M plus interleaved QKV/O selected beyond 8192 after exact CB and sharding failures. |
| RoPE layout | rectangular versus exact row-wise sub-core order | Exact order selected; rectangular layout corrupted decode lanes 8–31. |
| MoE router | lower fidelity; BF16/HiFi2; FP32 accumulation | BF16/HiFi2 with FP32 destination accumulation preserves authentic top-k. |
| sparse MoE | token-at-a-time, grouped active tiles, DRAM/L1 intermediates, chunk 4–128 | Grouped chunk 24/L1 selected for b1 prefill: 14.191/14.264 ms versus functional 14.908/14.655 ms. Chunk 128 screens at 10.331 ms on synthetic routes but collapses the route union across the entire sequence toward dense execution and lacks equivalent authentic correctness, so it is not an eligible active-expert selection. |
| expert precision | sparse/dense BFP8 and BFP4 | Sparse BFP8 retained. A branch-proven selected mixed matrix passes 8/8; an independently forced-dense BFP4 matrix also passes 8/8 at PCC 0.997234–0.999990. BFP4 reduces b32 decode from about 3.39 to 2.22 ms. |
| final dense-expert geometry | BFP4 auto; explicit 64/80/100 cores; gate/down widths 4/3, 8/6, 16/4, 16/12; split/packed | Auto split remains selected at 2.218/2.215 ms for layers 1/4. Best explicit is 2.243/2.244 ms; auto packed is 2.294/2.291 ms. This directly closes the final profile's `in0_block_w>=2` advice under selected precision. |
| DRAM-sharded dense experts | real full chain; split/packed gate-up; groups 8/16/32/64; BFP4/BFP8; legal blocks | Split G8 BFP4 traces at 2.818 ms; packed G8 BFP4 passes PCC 0.99981–0.99985 and traces at 2.827 ms, versus selected 1.887–1.888 ms. G16 is numerically invalid (PCC 0.675–0.687); G32/G64 hit exact L1/CB capacity limits. Rejected after compatible full-chain evidence, not a first API error. |
| composites | top-k, sigmoid, scatter, paged cache, SDPA | Kept device-resident. Scatter's internal untilize/scatter/tilize is intrinsic to its row-major mask contract. |
| collectives | none | Not applicable to this single-device stage. |

## Fresh profiler accounting

Tracy and watcher were collected separately. Each final profile is signposted;
`tt-perf-report` 1.2.8 ran with advice enabled and `--active-experts 8` on
batch-1 MoE rows.

| Profile | Ops | Device | Profile wall | Gap | DRAM roofline | Dominant operation |
|---|---:|---:|---:|---:|---:|---|
| dense prefill b1 / b32 | 17 / 143 | 466.6 / 4102.9 us | 715.0 / 4750.1 us | 248.4 / 647.2 us | 18.9% / 13.0% | matmul 50.6% / 41.3% |
| dense decode b1 / b32 | 27 / 28 | 166.7 / 236.3 us | 208.0 / 271.7 us | 41.4 / 35.3 us | 33.2% / 23.4% | matmul 60.3% / 42.5% |
| layer-1 MoE prefill b1 / b32 | 202 / 227 | 13711.5 / 138508.9 us | 14598.3 / 139828.9 us | 886.8 / 1320.0 us | 3.8% / 14.4% | sparse/dense matmul 60.7% / 65.3% |
| layer-1 MoE decode b1 / b32 | 47 / 50 | 765.2 / 2185.9 us | 827.7 / 2255.4 us | 62.5 / 69.5 us | 14.6% / 31.9% | sparse/dense matmul 57.3% / 45.7% |
| layer-4 MoE prefill b1 / b32 | 200 / 225 | 13714.2 / 138261.5 us | 14481.3 / 139807.2 us | 767.1 / 1545.7 us | 3.8% / 14.4% | sparse/dense matmul 60.8% / 65.5% |
| layer-4 MoE decode b1 / b32 | 43 / 46 | 764.7 / 2190.3 us | 826.7 / 2259.8 us | 62.0 / 69.6 us | 14.6% / 31.8% | sparse/dense matmul 58.0% / 46.0% |

The final profiles contain no Torch, `from_torch`, `to_torch`, or host
fallback operation in the measured windows. Gzip-compressed raw ops CSVs,
filtered CSVs, full human-readable tables/advice in `human_report.txt`,
runtime JSON, and summary CSV/PNG are under
`tracy/review3_selected/`. The profile confirms batch-1 MoE is limited by
active sparse matmuls (85% of prefill device time across DRAM/L1 sparse
rows), while batch-32 MoE is limited by dense expert matmuls and routing.
The report's suggestion to increase dense-expert `in0_block_w` was swept
under final BFP4/LoFi across legal explicit widths and core counts; automatic
split remains 1.2% faster than the best explicit candidate.

## Watcher and artifacts

Final watcher command:

```bash
TT_METAL_WATCHER=10 \
TT_METAL_LOGS_PATH=models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/watcher/review3_final \
pytest -q -s --timeout=900 \
  --junitxml=models/autoports/coherelabs_north_mini_code_1_0/doc/optimized_decoder/artifacts/review3_watcher.xml \
  models/autoports/coherelabs_north_mini_code_1_0/tests/test_optimized_decoder.py
```

Result: `30 passed, 16 skipped in 347.109s`. The 2,170-line watcher log has no
fatal, invalid-NoC, CB-bounds, overflow, sanitizer, timeout, hang, tripped, or
kernel-error signature. Post-run `tt-smi 6.0.0` reports four p300c boards,
DRAM healthy, live heartbeats, zero GDDR errors, and zero thermal trips.

- `artifacts/review3_{full,watcher}.xml`: final suite evidence.
- `artifacts/review3_selected_authentic_matrix.xml`: eight authentic mixed
  precision rows.
- `artifacts/review4_selected_mixed_matrix.xml` and
  `review4_forced_dense_bfp4_matrix.xml`: branch-proven selected and
  all-dense authentic matrices.
- `artifacts/review4_full.xml`: revised full suite, 30 passed and 16 opt-in
  candidate skips.
- `artifacts/review4_dense_bfp4_watcher.xml` and
  `watcher/review4_dense_bfp4/`: layer-1/layer-4 forced-dense BFP4 serving
  decode passed under watcher with no fault signature.
- `artifacts/review4_dram_packed.xml` and
  `candidates/review4_dram_full_chain_packed/`: packed DRAM full-chain PCC
  and trace evidence.
- `candidates/review4_dense_bfp4/`: final-precision geometry and packing
  sweep.
- `candidates/review3_final_runtime/`: final 3-warmup/20-sample wall matrix.
- `candidates/review3_dram_full_chain_*`: compatible DRAM-sharded sweep.
- `prefill_layer{0,1,4}_context*_review3.json`: final optimized capacity.
- `context500000_decode_b32_review3.json`: final-policy serving-capacity
  allocation and trace replay.
- `context500000_decode_b32_review3_watcher.json` and
  `watcher/review3_capacity_final/`: watcher-clean replay with the final
  resident-weight set.
- `tracy/review3_selected/`: final 12-profile raw and analyzed evidence.
- `watcher/review3_final/generated/watcher/watcher.log`: watcher evidence.
- `AUTODEBUG_REVIEW3.md` and `STAGE_REVIEW_3.md`: diagnosis and review record.
- `work_log.md`: full experiment, checklist, commands, and commit ledger.
