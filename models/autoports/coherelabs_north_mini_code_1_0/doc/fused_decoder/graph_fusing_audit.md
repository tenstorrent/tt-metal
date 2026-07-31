# Graph-fusing audit

This audit is scoped to the single-device decoder layer introduced by Stage 02.
Collective fusions, cross-layer residual/norm fusions, LM head, sampling,
full-model, and serving integration belong to later stages.

## Measured op sequences

The detailed row-by-row sequences are the checked-in `tt-perf-report` tables.
The following is the complete logical sequence; optional RoPE rows distinguish
layers 0/1 from layer 4.

| Region | Prefill | Decode | Movement / reason |
|---|---|---|---|
| norm/QKV | RMSNorm -> packed QKV matmul -> create heads | RMSNorm -> packed QKV matmul -> create QKV heads | Q/K create-head outputs are already required height-sharded layouts |
| position | Q RoPE -> K RoPE, when enabled | Q RoPE -> K RoPE, when enabled | dedicated TTNN RoPE kernels |
| cache | paged fill K -> paged fill V | required V reshard -> paged fused update K+V | V must use a core set disjoint from K for the fused NoC contract |
| attention | SDPA -> concatenate heads -> output matmul | paged SDPA decode -> concat heads decode -> output matmul | concat decode requires the existing height-sharded input |
| dense MLP | packed gate/up matmul -> slice gate -> slice up -> multiply(SiLU(gate), up) -> down matmul | same | packing is setup-only; runtime stays device-resident |
| MoE router | router matmul -> topk -> sigmoid -> scatter | same | exact North sigmoid top-8 routing |
| MoE, >32 tokens | repeat token input across experts -> packed gate/up batched matmul -> two slices -> fused-SiLU multiply -> down batched matmul -> route multiply -> reduce | n/a for required decode batches | TTNN layout transitions visible in reports are imposed inside topk/scatter/reduction |
| MoE, exactly 32 tokens | same through down batched matmul -> L1 expert view -> fused score multiply/reduce | same for batch 32 | the fused kernel requires a one-tile token geometry and setup-only identity mapping tables |
| MoE, <32 tokens | n/a for measured prefill-128 | top-8 sparse gate/up -> two slices -> fused-SiLU multiply -> all-expert sparse down -> route multiply -> reduce | active gate/up rows make all-expert down exact; current A-sparse mask is per expert batch, not per token row |
| residual | attention + MLP -> hidden residual add | same | North uses parallel attention/MLP branches from one norm |

The final filtered totals are 17/33/31 device ops for dense/sliding/full
prefill; 22/43/39 for batch-1 decode; and 22/41/37 for batch-32 decode. All
nine windows contain zero host ops. Equal op counts versus the functional graph
do not mean equal topology: packed gate/up replaces a projection but needs two
views, fused cache update replaces one update but needs its required V reshard,
and the specialized kernels reduce dominant compute/device time by 7.5-77.9%.
The serving fused reduction adds required layout/mapping dispatches but replaces
the separately materialized routing-score multiply and reduction, reducing
device and wall latency.

## Dedicated fused ops

| Pattern | Assessment / experiment | Decision |
|---|---|---|
| activation recognition | Functional code already used dedicated TTNN activations. | retained |
| softmax | Attention already uses fused SDPA softmax; North router requires sigmoid top-8, not softmax. | retained / router pattern inapplicable |
| RMSNorm | Functional decoder already uses `ttnn.rms_norm`. | retained |
| distributed RMSNorm | Single-device Stage 02 has no collective. | later multichip stage |
| SDPA | Prefill and paged decode already use dedicated TTNN SDPA. | retained |
| split QKV / split heads | Functional path already has packed QKV and dedicated create-head ops. | retained |
| decode create heads | Already `nlp_create_qkv_heads_decode`. | retained |
| decode concat heads | Already `nlp_concat_heads_decode`. | retained |
| prefill concatenate heads | Already `transformer.concatenate_heads`. | retained |
| RoPE | Existing Q/K calls are dedicated RoPE. The combined QK RoPE candidate requires non-overlapping Q/K grids, a duplicated `[1,2*batch,32,D]` cos/sin tensor, a transformation matrix, and extra Q/K/cos/sin reshards. The public stable buffers are `[1,batch,1,D]`; adapting them adds more runtime movement/dispatch than it removes. | rejected on exact op contract/topology |
| TopK | Already `ttnn.topk`. | retained |
| residual-add + RMSNorm | Invalid for parallel `hidden + attention(norm(hidden)) + mlp(norm(hidden))`; folding the final residual into the next norm changes branch semantics and crosses this layer's boundary. | inapplicable |
| fused matmul + collective | No collective in single-device scope. | later multichip stage |
| sparse MoE experts | Exact top-8 packed `sparse_matmul` tested and retained for token counts below 32. At batch 32 it measured 13.395 ms, slower than the 11.122 ms functional baseline; packed all-expert measured 8.299 ms and was retained. `moe_compute(compute_only=True)` does support a 1x1 Blackhole path, so it was run at the exact North geometry: 128 experts, 32 tokens, top-8, hidden 2048, intermediate 768, and SILU. Its internally fixed BFP4 expert weights produced PCC 0.992762 and 0.990510, below the stage's 0.995 bar, before the still-required external routing-score combine. | hybrid retained; exact sparse and `moe_compute` candidates rejected |
| fused MoE gate/router | `topk_router_gpt` implements linear+topk+softmax for exactly 32 throughput tokens. North requires sigmoid weights and also needs a dense routing mask for the selected sparse path. DeepSeek/grouped/hash gate variants have different routing algebra. | rejected as numerically non-equivalent |
| fused weighted expert reduction | `deepseek_moe_fast_reduce_nc_fused` was tested both at serving batch 32 and prefill. At batch 32 it preserved PCC 0.998193 and improved sliding/full wall latency from 8.299/8.292 ms to 8.273/8.279 ms (device 8257/8239 us to 8248/8234 us), so it is retained. At measured prefill length 128 one whole-shape call was faster (9.974 versus 10.080 ms) but fell to PCC 0.408818 versus identical 0.987648 functional-decoder and selected-fallback controls on the same new synthetic stress. Adapting it as four 32-token calls plus concat restored functional/fallback-equivalent numerics (PCC 0.987649) but regressed wall latency to 10.410 ms. Logical length 33 likewise produced PCC 0.758886; at chunk 1024 its 536,870,912-byte L1 request was 4,882,432 bytes/bank versus 1,461,504 available. It is therefore selected only for its valid and faster one-tile geometry. | retained at exactly 32 tokens; rejected elsewhere |
| paged KV update | Initial fused update failed because K and V grids overlapped. Retried with a disjoint V core set and one required reshard; PCC passed and dense b1 decode improved from 0.325 to 0.320 ms. | retained; watcher clean |

## Graph rewrites

| Pattern | Assessment / experiment | Decision |
|---|---|---|
| RepVGG conv sum | No convolution. | inapplicable |
| shared-LHS matmul | Q/K/V were already packed. Dense gate/up and all 128 expert gate/up peers were newly packed at setup. Original weights are deallocated after packing. A further dense cross-branch QKV+gate/up pack (11264-wide) passed prefill/decode PCC but regressed every mandatory dense regime: prefill 0.580 to 0.610 ms, decode b1 0.320 to 0.329 ms, and decode b32 5.698 to 5.886 ms. The MoE normalized input has only QKV and router as immediate matmul peers, below the skill's three-peer consolidation trigger; router output also feeds chunk-dependent routing before expert execution. | gate/up retained; measured cross-branch candidate rejected |
| spatial mean | No spatial reduction. | inapplicable |
| permute-reshape-permute identity | All remaining permutations change semantic axes for heads, experts, tokens, or residual output. Reshape-view rows dispatch no data movement. | no removable identity found |

## Op merging

| Pattern | Assessment / experiment | Decision |
|---|---|---|
| conv bias/scale/activation, BN+conv, pad+pool | No convolution, batch norm, or pooling. | inapplicable |
| matmul + activation | SwiGLU cannot put SiLU on the packed matmul because only the gate half is activated. | not equivalent |
| input activation + binary | `multiply(gate, up, input_tensor_a_activations=[SILU])` removes the standalone SiLU dispatch. | retained |
| matmul + bias | North decoder projections have no bias. | inapplicable |
| transpose + matmul | Weight orientations are packed once at setup; remaining transposes change token/expert/head axes and are not RHS transposes. | no candidate |
| slice after matmul | The two packed gate/up slices are both consumed; narrowing either operand would restore two matmuls. Final output slices remove tile padding and cannot narrow the mathematical projection. | retained |
| numeric-stable softmax | Router uses sigmoid; attention softmax is inside SDPA. | inapplicable |
| reduction + reshape | The expert reduction's output rank is already the required rank. Explicit `fast_reduce_nc` was tried for batch 32: 8.293376 ms versus 8.293310 ms for `sum`, with no gain. The score-weighted fused reduction was then tested and retained at exactly 32 tokens as described above. Other token counts keep `sum`, which lowers to `FastReduceNCDeviceOperation` when profitable. | explicit plain reduction rejected; score-weighted fusion retained at 32 |
| scaled sum -> mean | Routing is a nonuniform weighted sum, not arithmetic mean. | inapplicable |
| decode RoPE/layout fold | Existing dedicated decode RoPE consumes the create-heads layout; the combined-QK candidate has the incompatible contract described above. | rejected |

## Other explored implementation choices

- `ttnn.split(..., 2, dim=-1)` for packed gate/up compiled into a Blackhole
  kernel with an undeclared `single_tile_size_bytes` symbol at sequence 128.
  Two width slices are the working equivalent and passed PCC/performance.
- Exact per-token sparse down originally attempted the same top-8 mask as
  gate/up. The current A-sparse API accepts one mask per expert batch, not per M
  row; at batch 32 its padded sparsity volume was 4096 where 128 was required.
  Since inactive gate/up rows are exactly zero, all-expert sparse down is
  numerically exact for the sub-tile path.
- A fully exact sparse batch-32 graph was correct but slower than the functional
  baseline. The packed all-expert graph wins at batch 32 and prefill while the
  exact sparse graph wins at batch 1, hence the tile-boundary dispatch.
- The dense QKV+gate/up cross-branch pack passed PCC but lost 5.2% in prefill,
  2.9% in batch-1 decode, and 3.3% in batch-32 decode. Raw 20-sample timings
  and the comparison are retained in `candidate_cross_pack_*.json`.
- The fused score/reduce candidate is selected only when `token_count == 32`.
  Its complete PCC, L1-contract, wall, and device comparison is retained in
  `candidate_fused_reduce_summary.json`, including the measured sequence-128
  prefill rejection at PCC 0.408818 and the four-tile adaptation rejected at
  10.410 ms. `candidate_seq128_remediation_matrix.json` additionally records
  the unchanged functional control and five further precision/sparse
  adaptations tried during review remediation.
- Custom sparse program configurations use the largest legal rectangular
  divisor of output tiles and the largest legal K block from 8/4/2/1. No
  program configuration is applied outside the specialized sparse operator.

No remaining single-device decoder subgraph matches an available TTNN fused-op,
graph-rewrite, or op-merging contract while preserving North semantics and
improving the measured workload.
