# Fusion pattern assessment

This inventory covers the graph-fusing skill and additional existing native ops.
Each applicable graph rewrite has been tried or excluded by the exact native op contract. Timings and adapted failures are recorded in work_log.md and candidate_metrics.csv.

| Pattern | Applicability and evidence/experiment |
|---|---|
| Dedicated activations (ReLU/ReLU6/hardsigmoid/SiLU/mish/GELU) | Only SiLU/sigmoid/softplus occur. SiLU convolution dedicated; remaining gates folded into binary consumers. Other activations absent. |
| Softmax recognition | Full attention already uses native chunked/paged SDPA, including softmax. No standalone exp/sum/div graph. |
| RMSNorm recognition | Input/post-attention and full Q/K already dedicated. Linear Q/K L2 absorbed by flat scan; output norm/gate absorbed by KDA op. |
| Distributed RMSNorm | Single-device contract has no collective or distributed norm. |
| SDPA recognition | Already native chunked/paged SDPA; primitive linear recurrence is not softmax attention. Native delta scan tested for both modes. |
| Prefill split-QKV/heads | Packed/reordered full Q,K,V,gate; dedicated split op selected for prefill. |
| Decode create-QKV-heads | Dedicated decode op tested and selected. Height-sharded norm restriction adapted via block sharding. |
| Decode concat heads | Redundant transpose pair removed; direct logical flatten compared with generic concatenate_heads. Specialized sharded concat required SDPA output conversion and logical batch trimming; adapted valid candidate took 2.3122 ms versus direct flatten 2.3056 ms. Rejected. |
| Prefill concatenate heads | Dedicated op tested and selected; generic decode use rejected as slower. |
| RoPE | Dedicated rotate-half op over actual 64 rotary channels, preserving other 192. Decode token_index=0 consumes refreshed caller cos/sin. |
| TopK | No sorting or sampling in a decoder layer. |
| RepVGG convolution sum | No parallel 3x3/1x1 spatial convolutions. The causal depthwise temporal convolution is handled by its dedicated native op. |
| Shared-LHS matmul | Full Q/K/V/gate and linear QKV/Z/B/A packing tested. MLP gate/up packing tested with/without binary activation fusion; separate is faster. |
| Spatial mean | No spatial reduction. Q/K squared sums are incorporated in native scan; RMS means are native norms. |
| Permute-reshape-permute identity | Decode attention redundant pair removed. Scan head-major output removes output-layout round trip. Non-identity head splits use dedicated ops. |
| Conv+bias | Target conv has no bias. |
| Conv+scale | No following constant per-channel scale. Learned convolution taps are already setup tensors. |
| Conv+activation | Dedicated qkv_causal_conv1d_silu includes SiLU and output split. |
| Matmul+activation | Explicit-grid matmul SiLU epilogue tested and selected; default activation without core_grid appends a unary op. The packed MLP alternative was slower. |
| Input activation+binary | Full output sigmoid, MLP SiLU, decay softplus folded. Linear output gate handled by fused RMSNorm. |
| Matmul+bias | Model projections have no biases. A's dt_bias is added after BF16 projection -> FP32, so moving into the projection changes a rounding boundary; decay remains FP32. |
| Transpose+matmul | Full attention transpose is internal to SDPA. Linear recurrence outer-product transpose eliminated by native scan. Weights transposed only during setup. |
| Slice after matmul into RHS | All actual Q/K/V/gate/up/A/B/Z outputs are consumed; only hardware tile padding is discarded. No removable learned output channels. |
| BatchNorm+conv | No BatchNorm in the target. |
| Pad+pool/conv | No pooling. Native causal conv has no padding argument; its exact contract requires tile-aligned T. Internal zero padding is required for tails and never exposed as a public restriction. |
| Stable softmax max-subtract | Internal to native SDPA; no external max/subtract to remove. |
| Reduction+reshape | No remaining primitive reduction with removable keepdim reshapes. |
| Scaled sum->mean | No such standalone subgraph; L2 normalization is native scan, RMSNorm is already native. |
| Decode RoPE/layout fold | Decode rotary token-index mode handles head rows; direct SDPA flatten eliminates output transpose. |
| Additional: dual cache update | paged_fused_update_cache tested with disjoint update-core sets; no speculative overlapping-core call. |
| Additional: convolution history | Row-major persistent history plus direct latest-row slice removes unnecessary conversions and full-history concat. |
| Additional: scan normalization/head mapping | Flat token-major q/k/v allows in-kernel L2, scale, GQA expansion; head-major output feeds fused norm directly. |

Source contracts inspected: transformer/chunk_gated_delta_rule and its prep/scan;
experimental/kda/{qkv_causal_conv1d_silu,sigmoid_gated_rms_norm};
experimental/transformer/{rotary_embedding,nlp_create_qkv_heads_decode,
nlp_concat_heads_decode}; experimental/paged_cache; normalization/layernorm;
matmul/matmul.cpp; eltwise/unary/common/unary_op_utils.cpp and binary bindings.
Idioms inspected in models/experimental/gated_attention_gated_deltanet/tt;
Python golden contracts in ttnn/ttnn/operations/transformer.py and exact native
unit tests for the selected operations. Native sparse/MoE/DiT convolution/CCL
ops have no corresponding subgraph in this dense single-device decoder.

Residual-add plus RMSNorm was assessed: the unnormalized residual is also used by the final residual add. The native RMSNorm pre-add does not return that residual, so folding it would repeat the addition rather than eliminate a graph node.

Matmul + untilize was tried after splitting the packed linear projection to
satisfy the 1D output-block constraint. Both automatic and explicit row-major
output variants were adapted and checked; the executable variants fail PCC.
The otherwise identical tiled-output control passes but takes 2.45809 ms
traced decode and retains its conversion. See the final-layout section of
work_log.md. The selected single conversion into the native row-major causal
convolution is required by the correct available path.
