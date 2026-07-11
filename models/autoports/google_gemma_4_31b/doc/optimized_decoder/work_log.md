# Optimized decoder work log

Hardware: one P150 Blackhole, 110 worker cores, eight DRAM banks, firmware
19.9.0. Initial `tt-smi -ls --local` and 1x1 mesh smoke were clean. All hardware
runs were serialized; no reset or recovery was required.

## Baseline and audit

The fused decoder checkpoint was `bea8302a6ac`. Its measured topology had five
BF16 matmuls accounting for roughly 96–97% of decode device time. Repeated
matmul audit: QKV was already packed (keep and compare), O was standalone,
gate/up were same-input separate projections, and down was standalone. The
fused SDPA/cache/head-layout rewrites were retained. Layout audit found small
residual norm conversions and material interleaved projection traffic. MoE and
CCL are N/A for this dense single-device layer.

## Search chronology

1. Loaded phase-specific interleaved prefill weights and eight-bank
   DRAM-width-sharded decode weights. Explicit phase state avoids mistaking a
   logical seq-32 prefill for padded decode.
2. Adapted the MLP after two simultaneous 21,504-wide BF16 results collided
   with static CB storage: spill only up, fuse GELU into gate, restore up into
   the same L1 width shards, multiply, and run sharded down.
3. All-BFP4 real-weight attention failed PCC at 0.992205. BFP8 attention with
   QKV block 3 passed at 0.998668 sliding / 0.998983 full. QKV block 7 required
   1,733,376 static bytes and exceeded 1,572,864-byte L1.
4. Swept exact selected precision: gate/up block 3 = 1.211 ms, block 7 = best,
   block 21 exceeded 1,572,864-byte L1. Down block 12 = 1.189 ms, 14 = 1.189 ms,
   21 = 1.186 ms (keep), 28 collided at static end 1,285,120 vs live buffer
   1,013,760. QKV block 1 = 1.293 ms; block 3 kept. O block 16 = 1.187 ms vs
   block 8 = 1.186 ms. Larger legal divisors do not divide the relevant K
   tiles/core.
5. Four-core adaptations progressed from QKV block 7 OOM to QKV3/gate3/down12,
   then failed the down static/live boundary (638,976 vs 454,656). More than
   eight logical width shards is incompatible with the eight-bank DRAM-sharded
   weight/program contract; widths also reject ten and a one-row x=12 grid is
   outside the P150 x dimension.
6. HiFi2 families lost: attention 1.351 ms, gate/up 1.539 ms, down 1.363 ms.
7. Packed gate/up was adapted from BF16 to BFP8 output. At block 7 the paths
   required 2,516,224/2,699,008 static bytes. Legal block 3 reduced static
   storage but still collided with live allocations (BF16: 1,338,624 vs
   1,185,792; BFP8: 1,521,408 vs 1,347,072). Block 1 finally ran: BF16 output
   passed PCC 0.998721 at 1.518 ms and BFP8 output passed PCC 0.998637 at
   1.440 ms, both slower than the 1.186 ms separate path. Tuned split Q/K/V
   was correct but measured 1.211 ms; packed QKV remained faster.
8. BFP8 KV first exposed the required paged-fill cast, then the distinct update
   contract. Final bulk fill uses BFP8 inputs while tail/decode update uses BF16
   and repacks inside the op. Non-aligned seq-33 PCC is 0.998574.
9. `tt-perf-report` advised DRAM-sharded prefill QKV. The first M=128 program
   failed the M=1 contract; the adapted four-M=32 candidate ran and measured
   PCC 0.998493 and measured 3.356 ms versus the selected auto-2D path's
   2.672 ms.

## Final commands and artifacts

Correctness and stress XML lives under `evidence/`. Final warmed timing:

```bash
GEMMA4_OPT_BENCH=1 GEMMA4_OPT_BENCH_IMPLEMENTATION=fused pytest -q -s models/autoports/google_gemma_4_31b/tests/test_optimized_decoder.py -k warmed_latency
GEMMA4_OPT_BENCH=1 pytest -q -s models/autoports/google_gemma_4_31b/tests/test_optimized_decoder.py -k warmed_latency
```

Final profiler pattern, once for each layer-kind/mode pair:

```bash
GEMMA4_OPT_PROFILE=1 python -m tracy -r -p -v --output-folder <tracy-dir> -m pytest <single-profile-node> -s
tt-perf-report <mode>_ops.csv --start-signpost OPT_PERF_<MODE> --end-signpost OPT_PERF_<MODE>_END --csv <mode>_perf_report.csv
tt-perf-report <mode>_ops.csv --start-signpost OPT_PERF_<MODE> --end-signpost OPT_PERF_<MODE>_END --no-summary
```

Final report totals: sliding prefill 2,587 us / 29 ops; full prefill 3,300 us /
26 ops; sliding decode 1,149 us / 48 ops; full decode 1,297 us / 47 ops. The
raw and filtered CSVs plus advice tables are in each `tracy/.../final_bound/`
folder; every profile console log binds the frozen source hashes and policy.

## Operation-topology audit

| Measured sequence / issue | Candidate or lower-movement replacement | Constraint | Action and evidence |
|---|---|---|---|
| residual I2S -> learned RMSNorm -> S2I | persistent width-sharded residual | fused layer boundary is DRAM interleaved | sharded norm retained; each crossing is 1-2 us and avoids slower interleaved norm |
| one packed same-input QKV matmul | split Q/K/V | identical BFP8/LoFi and traced workload | packed 1.188 ms beat tuned split 1.212 ms |
| head split -> Q/K/V norm -> RoPE -> cache update -> SDPA | further QK/cache fusion or sharded GQA output | per-head learned norms, circular modulo cache, and GQA SDPA output contract | dedicated ops retained; fused full K/V update inherited, sliding needs two modulo-aware updates |
| head concat -> O projection | keep concat output sharded through O | decode concat/head geometry differs from width-sharded projection input | narrow conversion retained; DRAM-sharded O row is 88% modeled bandwidth |
| separate same-input gate/up | packed gate/up | 43,008-wide output plus static CBs pressures P150 L1 | BF16/BFP8 block 7 OOM and block 3 static/live collisions; block 1 passes PCC 0.998721/0.998637 but is slower at 1.518/1.440 ms; tuned separate path kept |
| up intermediate | keep both gate/up in L1 | two 21,504-wide outputs plus next matmul CB overlap | only `up` spills to DRAM; this is the minimum successful movement contract |
| gate GELU -> multiply -> down | fuse activation and tune geometry | GeGLU multiply cannot fold into down matmul | GELU fused into gate program; BFP4/LoFi block 7/21 selected |
| prefill interleaved 2D projections | DRAM-sharded M=128, then four M=32 chunks | decode DRAM-sharded op requires M=1 | adapted legal chunks passed PCC 0.998493 but measured 3.356 ms vs 2.672 ms final; interleaved 2D kept |
| CCL / MoE active expert | fused CCL+matmul, sparse expert path, persistent CCL buffers | dense single-device decoder contains neither | not model-applicable, no runtime rows |

## Reproduced candidate evidence

All commands below use the real layer shape and target weights. Logs are under
`candidates/`; the failed BFP4 attention XML is an intentional rejected-policy
artifact, not a stage gate.

| Candidate | Traced sliding decode / PCC | Decision |
|---|---:|---|
| final BFP8-attention/BFP4-MLP LoFi, blocks 3/8/7/21 | 1.185875 ms; PCC 0.998668 | keep |
| attention BFP4/LoFi | PCC 0.992205 | reject: below 0.995 real-weight bar |
| attention BFP8/HiFi2 | 1.351409 ms | reject |
| gate/up BFP4/HiFi2 | 1.539094 ms | reject |
| down BFP4/HiFi2 | 1.363184 ms | reject |
| QKV `in0_block_w=1` | 1.292700 ms | reject; block 3 wins |
| gate/up `in0_block_w=3` | 1.211484 ms | reject; block 7 wins |
| down `in0_block_w=14` | 1.188860 ms | reject; block 21 wins |
| O `in0_block_w=16` | 1.187105 ms | reject; block 8 final rerun is 1.185875 ms |
| split Q/K/V | 1.211050 ms; PCC 0.998668 | reject; packed wins |
| packed gate/up BF16, block 1 | 1.518328 ms; PCC 0.998721 | reject; separate wins |
| packed gate/up BFP8, block 1 | 1.440399 ms; PCC 0.998637 | reject; separate wins |
| packed gate/up BF16/BFP8, block 3 | exact static/live L1 collisions | reduce again to block 1 |
| four cores, QKV3/gate3/O8/down12 | down static end 638,976 vs live 454,656 | reject |

## Final gates and anomaly ledger

- Standard suite: `evidence/standard_suite.{xml,log}`: 21 passed, 12 explicitly
  gated benchmark/profile/long-oracle tests skipped. Inherited batch, mutable-trace,
  wrap, and long helpers bind both `OptimizedDecoder` and the BFP8 cache helper.
- Same-harness fused baseline: `evidence/fused_warmed_baseline.{xml,log}`: four
  passed; 3.519/4.326 ms prefill and 2.606/2.949 ms traced decode.
- Final default timing: `evidence/final_warmed_latency.{xml,log}`: four passed;
  2.671/3.395 ms prefill and 1.186/1.332 ms traced decode.
- Exact context BFP8 cache: `context_262144_{sliding,full}_bfp8.{xml,log}`:
  both pass. Non-aligned context: `context_262113_bfp8.{xml,log}`: two pass.
- Distinct-token exact-context HF oracle:
  `evidence/context_262144_distinct_hf_oracle.{xml,log}`: sliding/full pass at
  PCC 0.997758/0.998387 at absolute position 262143; wrong-position controls
  have higher RMSE. The capacity rows above are intentionally bounded-oracle
  checks and are not represented as full-history HF PCC.
- Watcher: `evidence/watcher_mutable_trace.{xml,log}`: four pass with
  `TT_METAL_WATCHER=10`; `watcher_final/generated/watcher/watcher.log` has no
  fatal/assert/NOC/overflow/sanitizer finding.
- Observed anomaly: the first combined exact-context run timed out only the
  full layer at pytest's 300-second bound, at chunk 144384. The board listed
  healthy immediately afterward. The unchanged node passed in 269 seconds
  with `--timeout=900`; resolution is controlled/fixed harness budget. Initial
  evidence is under `evidence/rejected_harness/`.
- Earlier stale XML captured pre-fix failures while implementation work was
  still in progress. It is preserved only under `evidence/rejected_harness/`;
  the final standard suite supersedes it.
- Final source hashes before independent review: optimized decoder
  `9da6bf3ef64d5e4d2ea5fd8071e518e2b666d39802701adab42ab4021efa928b`;
  optimized tests
  `a096a0bc643d72e2fb29ada72d71992e407fcf54133ecc53f7ad37adeac9eb6b`;
  inherited fused decoder
  `941dd1d16b64246111e8402d875cbb7fc1cc6bbf6b33465ff0ba97103df90af6`.

## First review remediation

The initial independent review returned `more-work-needed`; its report is
preserved as `stage_review_initial.md`. Every finding was closed before the
fresh rereview:

- Final suite, baseline/final latency, all four Tracy captures, filtered
  `tt-perf-report` tables/CSVs, and watcher coverage were rerun after the
  decoder/test sources were frozen. `RUN_BINDING` in every console artifact
  records the complete effective policy, relevant environment, and SHA-256
  source hashes; `evidence/run_manifest.json` indexes them.
- Candidate runners now record their effective policy. Earned artifacts cover
  split-QKV correctness/timing, BFP4 real-weight failure, HiFi families,
  block-width alternatives, packed gate/up BF16 and BFP8 adaptations, exact
  QKV/gate/down/four-core L1 blockers, and the correct-but-slower M32-sharded
  prefill adaptation. Pre-binding and one misspelled-env probe were moved to
  `evidence/rejected_harness/stale_candidates_before_final_binding/`.
- The second review identified the remaining reduced-block gap. Packed
  gate/up block 3 and block 1 were then tested for BF16 and BFP8 outputs;
  block 1 was correct but slower for both. The four-core QKV3/gate3/O8/down12
  adaptation was reproduced with its exact down-path L1 collision. Bound
  artifacts are `candidates/packed_gate_up_{bf16,bfp8}_block{3,1}*` and
  `candidates/cores4_adapted_qkv3_gate3_down12.*`.
- The context documentation now distinguishes bounded capacity/self-
  consistency probes from full-history HF comparisons. A separate exact-limit
  periodic-history HF oracle passes distinct late-token traced decode at PCC
  0.997758/0.998387 for sliding/full, with wrong-position negative controls.
- Roofline byte counts are explicitly labeled nominal payload accounting;
  BFP tile/container metadata is excluded.

## Optimize checklist

- [x] Decode is fully traced; source and profiler audits show no host fallback.
- [x] Decode projection activations use width-sharded L1; residual/norm and GQA
  boundary crossings are measured and justified.
- [x] Prefill uses DRAM-interleaved activations and the faster large-M 2D path.
- [x] Topology audit, repeated same-input projections, layout movement, SDPA,
  cache operations, and lower-movement candidates are recorded above.
- [x] Best-candidate and same-model-reference searches are complete; final
  default beats fused and every correct material candidate.
- [x] Profiler rows prove BFP8/LoFi QKV/O and BFP4/LoFi gate/up/down reached the
  measured decode operations.
- [x] Packed QKV wins tuned split; packed gate/up was adapted through BF16 and
  BFP8 output contracts and reduced from block 7 through block 3 to block 1.
  The legal block-1 variants pass real-weight PCC but lose whole-layer latency.
- [x] Important memory, program, SDPA, and compute-kernel configs are explicit.
- [x] Dominant role geometry and LoFi/HiFi2 searches are reproduced above;
  larger legal blocks and four-core adaptations are recorded with exact limits.
- [x] BFP4 attention and both BFP4 MLP projection groups were tested on real
  weights; synthetic evidence did not veto a real-weight win.
- [x] Decode weights are DRAM width-sharded over all eight banks.
- [x] CCL, persistent collective buffers, fused CCL+matmul, MoE sparse experts,
  LM head, and sampling are not applicable to this dense single-device layer.
- [x] BFP8 cache fill/update contracts, non-aligned lengths, batch 2/32,
  deterministic mutable trace replay, stress, and watcher are covered.
- [x] Roofline, device time, end-to-end time, and residual gap are reconciled in
  `README.md`.
- [x] No decoder optimization is deferred to multichip, full-model, or vLLM.

## Review and checkpoint

The final fresh independent review returned `clean-pass`; see
`stage_review.md`. The stage-owned implementation, tests, documentation, and
evidence checkpoint is local commit `5e21925512d` (`Add optimized Gemma 4 31B
decoder`). It was not pushed. The earlier review reports are preserved as
`stage_review_initial.md` and `stage_review_second.md` with their remediations
recorded above.
