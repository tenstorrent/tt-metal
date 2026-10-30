# Graph fusion assessment

Every graph-fusing skill pattern is classified below. Applicable candidates have
been tried and adapted where possible; selection uses PCC and measured latency.
Final affected-path acceptance and independent review are recorded in README.md.

## Functional graph

| Region | Ops | Movement / ownership |
|---|---|---|
| Input | RMSNorm | DRAM tiled BF16, weighted |
| QKV | Q/K/V linear (Q/K only for full), reshape, prefill permute | Separate projection weights; full K/V share raw projection |
| Head processing | Q weighted RMSNorm, K weighted RMSNorm, V unweighted RMSNorm | Per-head last-dimension normalization |
| RoPE | two slices, negate, concat, two multiplies, add per Q/K | Rotate-half; per-row absolute-position cos/sin |
| Cache | prefill batch slice, two paged fill; decode two reshard + paged update | Caller-owned BF16 paged DRAM cache |
| Attention | chunked paged SDPA / paged decode SDPA | Causal, sliding1024 or full; full decode long-context config retained |
| Attention output | prefill permute, reshape, O linear, weighted RMSNorm, residual add | Residual also consumed by final add |
| MLP | weighted RMSNorm, gate/up linear, GELU-tanh, multiply, down linear | Gate/up share normalized LHS |
| Output | weighted RMSNorm, residual add, scalar multiply | Output preserves input logical shape |

## Dedicated operations (first priority)

- Activation recognition: GELU already dedicated; test fusion into multiply or matmul.
- Softmax and SDPA: already dedicated chunked/paged SDPA; replacing with nonpaged
  SDPA would lose cache/continuation semantics. No explicit exp/sum graph remains.
- RMSNorm: already dedicated seven norms. Residual-input RMSNorm does not return
  the summed residual needed by final add; simply fusing input would duplicate sum.
- Distributed RMSNorm: no collective or distributed norm in this single-device stage.
- QKV split: `experimental.nlp_create_qkv_heads` supports tied KV (full attention)
  and generic GQA. `nlp_create_qkv_heads_decode` has no tied-KV parameter. Test packed
  projection and specialized split; adapt full decode independently.
- Concat heads: test `transformer.concatenate_heads` for prefill; decode specialized
  concat requires height-sharded input and will be compared including that transfer.
- RoPE: primitive baseline replaced by rotate-half dedicated kernel in prefill;
  HF decode kernel supports per-row positions with sharded cos/sin. Test preserving
  sharded result through SDPA/cache update. Ordinary token_index kernel broadcasts
  one position, so cannot alone support distinct batch positions.
- TopK: no sorting or top-k in this decoder.
- Reusable generic modules and kernel sources inspected: transformer QKV/head ops,
  rotary_embedding/HF kernels, normalization/rmsnorm, eltwise unary/binary and matmul.
  No removed Gemma implementation or Gemma-specific framework branch consulted.

## Structural rewrites (second priority)

- RepVGG conv-sum: no convolution.
- Shared-LHS projections: test QKV/QK packing and gate/up packing. Full K/V raw
  projection is already shared; do not silently duplicate it in final weights.
- Spatial mean: no spatial reduction.
- Permute/reshape composition: head sequence-to-channel permutation is not identity;
  dedicated head transforms are the applicable alternative. Metadata-only sequence
  views already avoid FillPad and preserve logical tails.

## Adjacent merging (third priority)

- Conv bias/scale/activation, BatchNorm folding and pad/pooling: no such nodes.
- Matmul activation: test GELU_TANH fused linear; compare with binary GELU fusion.
- Binary input activation: test GELU_TANH on gate input to multiply.
- Matmul bias: pinned architecture has no projection bias.
- Transpose matmul: weights are transposed once during setup; SDPA owns QK transpose.
- Slice into matmul: all projected Q/K/V/gate/up channels are consumed, so narrowing
  would remove required outputs. Packing slices partition outputs, not discard them.
- Stable softmax: handled inside dedicated SDPA; no external max/subtract exists.
- Reduction/reshape and sum/mean: no exposed reduction graph outside RMSNorm/SDPA.
- Decode RoPE layout: dedicated HF decode consumes [1,B,H,D] directly; no runtime
  token-index conversion or host position read is required.
- Scalar output merge: test parameterized scalar activation on final residual add.

## Run timeline and prediction

Setup converts weights and builds persistent inputs/cache. Warmup compiles physical
signatures. Prefill runs four1024-token chunks for4096; each fills pages then attends.
Decode captures once, replay uses caller-updated persistent inputs. Profile harness
refresh/drain gaps between replays are outside each whole-layer device denominator.
B1 initial profile has37 device ops/decode vs42 functional, but only0.46% speedup:
weight reads dominate, and sharded RoPE adds transfers. Retaining sharded outputs
predicts fewer transfers; packing predicts fewer weight-read/dispatch boundaries.
Neither prediction is acceptance evidence.

Additional repo-discovered dedicated pattern: `paged_fused_update_cache` replaces
the two decode cache writes. Its inputs must occupy disjoint equal-sized core
sets (`paged_fused_update_cache_device_operation.cpp:237-244`). Generic attention_1d
and the dedicated unit test demonstrate the signature. Candidate keeps K on its
existing first-B cores and places V on the next-B cores (all B<=32 fit110 cores),
then releases both before SDPA. This preserves original page/cache dtype/layout.

Additional algebraic candidate for full attention: raw K=V projection is already
shared; compute unweighted RMSNorm once for V and multiply by K gamma before RoPE,
instead of reducing the same vector twice. BF16 rounding changes, so this needs
real-weight PCC and long-context validation if retained.

Batch-dependent decode transforms: dedicated decode head splitting is slower at B1
but faster at B32. The batch matrix tests1/8/16/32 separately and includes shared
cos/sin transfer. Direct height-sharded RMSNorm is rejected by the current op's
explicit validation (layernorm_device_operation.cpp:226-227); the adapted interleaved
norm path passes and is timed. This is why splitter-to-norm transfers remain.
No whole fused path is abandoned because of the first layout error.

Generic module inspection confirms MLP binary activation fusion in
models/common/modules/mlp/mlp_1d.py and cache fusion in attention_1d.py. Generic
rmsnorm_1d.py provides interleaved and sharded variants, but no fused operation
returns both residual sum and normalized output. `indexed_fused_update_cache`
requires precomputed physical indices, whereas the public interface supplies page
tables plus device positions; paged_fused_update_cache performs that addressing
without extra indexing ops. No pairwise batched-prefill fill kernel exists for
this separate K/V BF16 paged-cache interface. Deepseek padded/FP8 cache kernels
have a different cache contract and are not replacements for paged_fill_cache.

Independent review identified a separate applicable batching rewrite: paged_fill_cache
supports batch_idx_tensor, so each K/V fill can consume all batch rows without
per-row slices. This differs from a pairwise K/V fill. The review-requested
batched_fill_candidate uses prepared device row indices and tests both meaningful
kinds at B32, including non-aligned logical sequence1025 and traced decode.
The candidate passed real-weight B32/S1025 correctness for both kinds. Controlled
ABBA comparisons (four samples/path) favor batched fill at B8/B16/B32 for both
kinds; all six medians are in batched_fill/*controlled.json. Selected production
uses setup-owned row identities and one per-prompt device index slice. B1 keeps
two scalar fills with no index work. The wider-batch fallback preserves support
outside the prepared grid capacity. Final watched B32 checks include tails,
continuation and cache ownership; see batched_acceptance_manifest.json.

## Final disposition

| Applicable transformation | Decision | Evidence |
|---|---|---|
| Dedicated prefill/decode RoPE | Keep, flatten prefill batch and share decode cos/sin transfers | candidates, candidates_adapted, candidates_batches |
| Q/K/V projection packing | Full QK replaces original weights; sliding packed copy only for B32+ prefill | candidates_combined, candidates_batches, README memory ledger |
| Gate/up projection packing | Reject: correct but slower | candidates_combined/manifest.json |
| Dedicated prefill head split/concat | Keep; split full concat into two16-head groups to fit L1 | candidates_adapted, full_lifetime_retry4.json |
| Dedicated decode head split | Keep B16/B32; smaller batches use faster reshapes | candidates_batches/manifest.json |
| Dedicated decode concat | Reject after padding adaptation: slower | candidates_adapted/manifest.json |
| Preserve sharded Q/K | Keep sliding Q and cache K; full Q requires DRAM for long SDPA L1 | full_lifetime_retry2.log, final context checks |
| Direct sharded split-to-norm | Exact op-contract rejection; adapted DRAM norm path is measured and retained | candidates_batches, layernorm_device_operation.cpp |
| Matmul GELU fusion | Reject: correct but slower than binary-input GELU | candidates_adapted/manifest.json |
| Binary GELU and residual scalar fusion | Keep | candidate_summary.json, direct equivalence checks |
| Shared full K/V normalization | Keep after long-context PCC validation | full_context_final.json, equivalence_5_1.json |
| Fused paged K/V decode update | Keep where disjoint B-core sets fit; preserve individual cache ops beyond that | candidates_combined, boundary checks |
| Batched prefill cache fill | Keep after six controlled comparisons and watched correctness | batched_fill/, batched_final_0_32.json, batched_final_5_32.json |

No applicable untried pattern remains in this inventory. Already-dedicated SDPA
and RMSNorm remain dedicated; unrelated convolution, pooling, TopK, distributed
collectives and exposed softmax/reduction patterns have no matching subgraph.
Low-level tiling/fidelity/SDPA tuning belongs to the subsequent optimized stage
and is not started here. The retained transfers satisfy demonstrated op contracts.
