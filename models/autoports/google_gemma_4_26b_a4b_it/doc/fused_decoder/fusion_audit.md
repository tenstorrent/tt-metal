# Stage 02 graph-fusion audit

Model: `google/gemma-4-26B-A4B-it`

Hardware: one P300 Blackhole, device 3, 11x10 compute-with-storage grid

Baseline checkout: `3fb5e87495c272665ea83cd95f1dc4a1aadc002d`

This is the complete operation-topology audit for the single-device decoder.
The audit covers both representative layer kinds: layer 0 sliding attention and
layer 5 full attention.

## Runtime operation sequence

| Region | Functional sequence | Stage-02 result |
|---|---|---|
| Attention input | RMSNorm | Already a dedicated device op; retained |
| QKV projection | one packed `linear` -> `nlp_create_qkv_heads_*` | Already packed/fused; retained |
| Q/K normalization | dedicated RMSNorm on Q and K | Retained; weights and accumulation differ, so no legal merge |
| RoPE | Q RoPE + K RoPE | HF RoPE retained; Llama fused-QK op has a different transformation-matrix contract |
| Prefill cache | paged K fill + paged V fill | No compatible two-output fill op exists |
| Sliding decode cache | paged K update + paged V update | Replaced with `paged_fused_update_cache` |
| Full decode cache | paged K update + paged V update | Batch 1 uses fused K/V update; batch 32 retains two updates because it beat forced fusion in the 200-sample control |
| Flexible decode cache | paged K update + paged V update | Retained for shared physical HMA views and modulo addressing; fused op has no `block_size`, `num_kv_heads`, or modulo contract |
| Attention | chunked/paged SDPA or paged decode SDPA | Already dedicated ops; retained |
| Head output | concat heads -> layout conversion -> output projection | Layout changes are required by concat/projection contracts; retained |
| Dense FFN | gate linear + GELU + up linear + multiply + down linear | GELU folded into multiply consumer |
| Router input | RMSNorm -> per-hidden multiply -> hidden scalar multiply -> FP32 cast | Both immutable scales folded into the FP32 router projection at construction |
| Router | projection -> BF16 cast -> top-k -> softmax -> scatter -> expert scale | Existing dedicated ops retained; DeepSeek gate ops require preallocated height-sharded tensors and different grouped/sigmoid semantics |
| Sparse experts | sparse gate/up -> GELU -> multiply -> sparse down | GELU folded into multiply for serving decode; batch-1 uses measured-faster accepted sparse GeGLU |
| Parallel FFN join | add branches -> RMSNorm -> residual add -> layer scalar | Residual-input RMSNorm candidate rejected because it regressed latency |

## Pattern-by-pattern assessment

| Graph-fusing pattern | Candidate | Outcome and evidence |
|---|---|---|
| Dedicated-op replacement | Two paged decode cache updates | Sliding attention uses one fused dispatch and beats the isolated two-update control: `2.994960` vs `3.002534` ms at batch 1 and `68.766853` vs `68.790083` ms at batch 32. Full batch 1 also uses fused update (`3.186425` vs `3.190858` ms); full batch 32 selects two updates (`68.587883` vs forced-fusion `68.592882` ms). |
| Dedicated-op replacement | Q/K fused Llama RoPE | Rejected as semantically incompatible. This decoder uses `rotary_embedding_hf`; the available fused op requires Llama transformation matrices, repeated Q/K cos/sin, disjoint sharding, and only decode. The repository itself selects separate `_hf_rope_decode` and `_mllama_rope_fused_qk_decode` paths. |
| Dedicated-op replacement | DeepSeek/generalized MoE gate | Rejected as contract-incompatible: the available ops implement DeepSeek grouped/sigmoid gate conventions and preallocated height-sharded outputs, not Gemma softmax-top8 plus per-expert scale. |
| Dedicated-op replacement | Attention/softmax/top-k/sparse matmul | Already represented by dedicated TTNN operations. |
| Structural rewrite | GELU producer into multiply consumer | Kept for dense FFN at the winning layer-kind/batch shapes and for serving-batch sparse FFN. Batch-1 sparse fusion was rejected, so shape specialization preserves the faster graph. |
| Structural rewrite | Residual add into RMSNorm | A real `residual_input_tensor` candidate was tried and rejected: `3.027858`/`68.781340` ms sliding and `3.220825`/`68.626988` ms full. The final graph uses explicit add followed by dedicated RMSNorm. |
| Structural rewrite | One batch-wide row-major routing conversion | Tried and rejected at batch 32: `69.729676` ms sliding and `69.498792` ms full. |
| Structural rewrite | Three batch-wide sparse MoE calls | Correct traced candidate tried in DRAM and rejected: `71.969304` ms sliding and `71.792087` ms full. Chunk sizes 2/4/8 were also tried; the best chunk-8 result was still `73.533700`/`73.286981` ms. |
| Structural rewrite | Pack gate/up projections | Not applicable under the skill's repeated-shared-input threshold: there are two, not three, consumers. Packing would also require a wider weight/output followed by slicing and has no sparse-matmul equivalent for expert weights. QKV, the qualifying three-projection case, is already packed. |
| Fold adjacent ops | Router hidden scale and `H^-0.5` | Kept. Both constants are multiplied into the FP32 projection once outside the runtime graph. `no_router_fold` regressed traced decode. |
| Fold adjacent ops | Bias/activation/scale into matmul | No biases exist in these Gemma projections. GELU is legally accepted by the multiply consumer, not by the relevant sparse matmul. Remaining router scale follows top-k/softmax and cannot move through either operation. |
| Merge normalization | Q/K or branch RMSNorm | Not legal: distinct inputs and learned weights. Residual-input RMSNorm was benchmarked separately and rejected. |
| Collective/communication fusion | Any | Not applicable in this single-device stage. |
| Full-stack transformer fusion | Any | Out of Stage-02 scope and no compatible TTNN whole-layer op exists. |

## Data-movement audit

The measured hot methods contain no Torch call, `from_torch`, `to_torch`, or
host fallback. The source-level gate in `test_fused_decoder_hot_path_fallback_audit`
enforces this.

The remaining layout operations in the per-op CSVs are required by device-op
contracts:

- head tensors move between DRAM interleaved and height-sharded layouts for
  RMSNorm, HF RoPE, cache update, and head concatenation;
- sparse MoE requires row-major routing masks and tiled sparse-matmul inputs;
- batch-32 production MoE uses independent top-8 routing rows. A legal
  batch-wide sparse-MoE candidate and 2/4/8-row chunks removed repeated
  conversions but padded intermediates in DRAM and were materially slower;
- non-aligned prefill is internally padded and sliced, preserving the public
  logical-length contract.

No standalone `reshard` op or host conversion appears in the measured reports.
The compact tables are in `tracy/`; decode reports contain positive device time
for every row (72/74 rows at batch 1 and 756/757 at batch 32).

## Rejected candidate artifacts

The `*_no_*.json` files in this directory are the isolated candidate results.
The decisive controls were:

- removing sliding cache fusion: `3.002534` ms batch 1 and `68.790083` ms batch
  32;
- removing router folding: `3.004453`/`69.029683` ms sliding and
  `3.195401`/`68.667063` ms full;
- removing all GeGLU consumer fusion: serving batch regressed to `68.902507`
  ms sliding and `68.664624` ms full;
- disabling only sparse or dense GeGLU identified the final layer-kind/batch
  specialization. Cells in which a control names the same selected graph can
  differ by a few microseconds of run noise; no distinct graph beats the final
  selection;
- residual-input RMSNorm, batch-wide routing, batch-wide sparse MoE, and
  chunked sparse MoE all regressed and remain reproducible test variants.

Every viable pattern found in the repository and every adjacent runtime
sequence above was either kept, measured and rejected, already dedicated, or
ruled out by an exact operation contract.
