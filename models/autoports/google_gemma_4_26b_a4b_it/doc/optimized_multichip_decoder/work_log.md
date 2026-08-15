# Optimized multichip decoder work log

Stage implementation/evidence commit: `c78b0d5959a` (local only; not pushed).

## 2026-08-15 inventory and operation-topology audit

- Stage baseline: commit `e45e0994778`; completed TP=4 decoder implementation
  commit `aef721434f4`; target hardware is the four-P300C `1x4` ring.
- Unrelated pre-existing worktree state excluded from this stage:
  `models/autoports/qwen_qwen3_6_27b/` and `third_party/tt-metal/`.
- `timeout 60 tt-smi -ls --local` listed all four Blackhole P300C devices.
- Read `$optimize`, `$tt-device-usage`, and section 4 of
  `tech_reports/LLMs/llms.md` before changing code.
- Baseline profile source:
  `doc/multichip_decoder/artifacts/perf/decode_batch32_{sliding,full}/analyzed.csv`.
  These are signpost-delimited, four-device warmed trace replays from the
  completed multichip stage. Fresh same-harness batch-1/prefill baselines are
  still required before accepting a candidate.

### Initial topology

| Boundary / repeated input | Baseline operation sequence and layout | Material movement / constraint | Candidate family and required evidence | Initial action |
| --- | --- | --- | --- | --- |
| Decoder residual | Replicated BF16 DRAM residual -> local norms/projections; row-parallel O, dense-down, and expert-down each return replicated BF16 through ring all-reduce | Three hidden-width collectives per layer; layer input/output are replicated | Compare replicated AR against stack-compatible fractured residual, delayed gather, fused matmul+RS/AG+matmul, and collective placement. Do not count RS followed by immediate AG as a rejection. | Audit APIs/source and retain exact blockers or whole-compatible-chain measurements. |
| Attention Q/K/V | One load-time packed QKV weight and one local matmul -> on-device head creation | Already removes three repeated same-input matmuls; full-attention KV heads are pair-duplicated by the TP ownership contract | Compare kept packed path to legal separate projections only under equal dtype/layout/config; verify final profile contains one QKV matmul. | Packed incumbent; benchmark separate/adapted candidate or preserve precise blocker. |
| Attention output | Local SDPA/head concat -> DRAM-interleaved O matmul -> DRAM all-reduce | Profile row `32x1024x2816` or `32x2048x2816` is not DRAM-sharded; input is DRAM; 1x1 subblock | Restore mesh-compatible DRAM-sharded weights and L1-sharded input/output; sweep geometry/fidelity; test fused matmul+collective and persistent CCL buffers. | First implementation family. |
| Dense gate/up | One load-time packed gate/up weight -> local packed matmul -> split/activation/multiply | Packed repeated-input projection already selected, but profile `32x2816x1088` is DRAM-interleaved with DRAM input and 1x1 subblock | Compare packed versus separate under best legal BFP8/BFP4 and geometry; carry L1-sharded activation across the full dense chain. | First implementation family. |
| Dense down | Local intermediate -> `32x544x2816` row matmul -> ring all-reduce | Profile has DRAM input, non-DRAM-sharded weight, `in0_block_w=1`, 1x1 subblock | Mesh-compatible DRAM sharding; sweep legal K divisors 1/2/4/17 as layout permits, output subblocks, fidelity, and fused/placed collective. | First implementation family. |
| Router / active experts | Replicated router/top-k -> gate-selected top-8 `sparse_matmul` gate/up -> activation/score -> sparse down -> ring all-reduce | Sparse gate/up is ~48% and transpose/unary layout work ~26% of batch-32 device time; local K widths are 2816 and 192 | Preserve active-expert execution; sweep role-specific sparse geometry, `nnz` exactness, L1 intermediate placement, BFP4/LoFi, and persistent/fused CCL feasibility. Dense all-expert is not an acceptable final path. | Highest-cost family after dense-layout repair. |
| Cache / SDPA | Local-head paged BF16 cache, explicit SDPA config, multiple head-memory conversions | Cache is already TP-local; conversions remain visible. Public non-aligned lengths are internally padded/sliced. | Cross BFP8 cache and attention/CCL dtype with best topology; test head-layout conversion removal without changing page/cache ownership. | Measure after baseline refresh. |
| Inter-layer contract | Replicated BF16 DRAM `[1,1,M,2816]` | Stack-compatible but forces every row-parallel contraction to complete an AR before the next layer | Final contract must avoid an inter-layer collective or prove a remaining collective wins whole-stack-compatible latency. | Open; no old-contract immediate restore may be used as sole evidence. |

### Baseline profiler observations that drive the first candidates

- Sliding batch-32 device rows total `11579.18525 us` across the merged
  four-device report. Sparse gate/up is `5577.891 us`, sparse down is
  `588.461 us`, transpose plus unary is `3041.070 us`, and the three
  reduce-scatter/all-gather implementation pairs total `155.61625 us`.
- Dense/attention matmul advice identifies DRAM-interleaved inputs,
  non-DRAM-sharded weights, small `in0_block_w` (dense-down is 1), and 1x1
  output subblocks. These recommendations are actionable and must be tried.
- Runtime rows prove packed dense/QKV weights use BF16 or BFP8 as configured,
  while sparse expert gate/up/down use BFP8+LoFi with L1-interleaved inputs.
  Final precision claims will be rechecked from the final profiler rows.

## Candidate sweep and decisions

All entries are warmed traced batch-1 decode milliseconds on the four-device
mesh with fallback raising.

| Family | Sliding | Full | Decision |
| --- | ---: | ---: | --- |
| Fresh baseline | 1.146865 | 1.252950 | Reference. |
| DRAM O + packed gate/up + down, blocks 4/11/17 | 1.11167 | 1.19150 | Kept. Gate/up block 22 was 1.11286/1.19184; O block 8 was retried but illegal for sliding's four-tile local width. |
| Expert gate block 22, N=3 / N=2 | 1.15885 / 1.09203 | 1.23837 / 1.17056 | N=3 rejected; N=2 retained for further sweep. |
| Expert gate block 44 / 88, N=2 | 1.08924 / 1.09676 | 1.16816 / 1.17429 | Block 44 kept. Down N=4 was 1.09179/1.17151; block 6/N=2 retained. |
| Packed / separate dense gate-up | 1.07880 / 1.08924 | 1.15732 / 1.17368 | Packed kept under equal DRAM-sharded BFP8 policy. |
| Ring links 1 / 2 | 1.08924 / 1.07880 | 1.16816 / 1.15732 | Two links kept. |
| Persistent async all-reduce | 1.08369 | 1.11153 | Layer-specific: rejected sliding, kept full. DRAM buffer, L1-interleaved buffer, and undersized width-sharded buffer failures were adapted; a 4x width-sharded-L1 buffer is legal. |
| Sliding attention BFP8 LoFi / HiFi2 | 1.07562 / 1.08036 isolated | n/a | LoFi decode PCC 0.99258 failed. HiFi2 passed 0.998164/0.999497 and wins in final composition. |
| Attention BFP4 | n/a | n/a | Rejected: sliding PCC 0.96932/0.97925, full 0.98832/0.97880. |
| Expert BFP4 | 1.08328 | 1.18927 | Full prefill PCC 0.994712 failed; combined sliding candidate was not faster than final BFP8. |
| Decode-only packed gate/up BFP4 | 1.06782 | 1.10939 | Kept. `$autofix` proved the earlier corruption came from selecting decode weights for 32-row prefill. Explicit phase selection plus independent per-rank host upload passes 0.998493/0.997778 sliding and 0.997408/0.998347 full. |
| BF16 / BFP8 CCL payload | 1.07880 / 1.09268 | 1.15732 / 1.17184 | BF16 kept; cast overhead exceeds byte saving. |
| Decode QKV DRAM auto / block 11 | 1.07168 / 1.06833 | 1.13256 / 1.11327 | Initial timing bypassed the role-aware helper. After wiring the material path, sliding decode PCC was 0.992642; rejected and removed from defaults. |
| Whole activation BF16 / adapted BFP8 | 1.06918 / n/a | 1.10904 / n/a | BFP8 first hit the BF16-only head splitter; a BF16 boundary cast retried the whole path, then prefill PCC failed at 0.97662 for both layer kinds. |
| BF16 / adapted BFP8 KV cache | 1.06918 / 1.07216 | 1.10904 / 1.11181 | BFP8 first hit packed-input rejection in `paged_update_cache`; retry leaves update tokens BF16 so the kernel repacks. PCC passes but decode is slower, so BF16 remains default. |

## Collective and residual findings

Two-link placement and persistent buffers were measured as whole layer
families, including conversions. The final replicated residual is the output
of the winning chain, not an immediate restoration used only for measurement.

`matmul_reduce_scatter_async` requires a 2D-multicast program and produces a
704-wide residual shard. The first assumption that every next consumer needed
an explicit restore was refuted by `$autofix`: an exact 1x4 hardware chain now
passes distributed RMSNorm followed by fused all-gather+matmul for sliding
QKV (N=5120), full QKV (N=8192), packed dense gate/up (N=4352), and replicated
router logits (N=128). A fixed selected expert gate projection (N=768) also
passes. All five use 704-wide rank-local residual input, PCC >= 0.99, and
fallback exceptions. See `artifacts/fused_agmm_residual_chain.json` and its
JUnit file.

That feasibility does not complete a coherent active-expert decoder path.
The fused AG+matmul API is rank-4, whereas `sparse_matmul` consumes dynamic
device routing and rank-5 `[1,128,K,N]` weights. Adapting one selected expert
to rank 4 passes, but doing so for dynamic top-8 IDs requires a new fused
AG+sparse-matmul interface; flattening all 128 experts is the prohibited dense
all-expert path. In the producer direction, fused dense-down and expert-down
also remain illegal after output-spec, buffer, block, and grid adaptations.

The exact-shape fused-RS repro was also adapted through output-spec, buffer,
block, and grid fixes. Sliding/full O producers pass after fixing the fused
op's nonsquare output-spec bug; dense/expert down remain illegal because 13
output blocks exceed the 11-core grid. The passing fused consumers therefore
do not remove both exact producer and dynamic sparse-consumer blockers. See
`artifacts/fused_rs_repro.json`.

Profiler advice was tried: DRAM-sharded decode weights and inputs, L1 through
the persistent family, larger sparse geometry/subblocks, BFP8 HiFi2, BFP4,
and packed repeated-input projections. HiFi4 was the slower incumbent; LoFi
was rejected on PCC where applicable. QKV stays load-time packed (one runtime
matmul); separate dense projections were directly slower.

## Final gates

- Default PCC: sliding 0.998493 prefill / 0.997778 decode; full 0.997408 /
  0.998347 (threshold 0.995).
- Default sequence 1024, batch 1, five extra warmups and 30 replays: sliding
  77.817 ms prefill / 1.068942 ms decode; full 86.199 / 1.110293 ms.
- Batch-32 trace, logical sequence 33 prefill/repeated decode, and advertised
  current position 262143 pass for both layer kinds.
- Fallback audit passes. Separate worker watcher run with
  `TT_METAL_WATCHER=120 TT_METAL_WATCHER_DISABLE_ETH=1` passes 2/2 with no
  error markers. Full Ethernet watcher cannot initialize instrumented fabric
  firmware because 27920 bytes exceeds its 25600-byte config buffer.
- Final four-device profiles and tt-perf-report CSVs are in
  `artifacts/final_profile_{sliding,full}`; exact provenance is
  `artifacts/provenance.json`.
