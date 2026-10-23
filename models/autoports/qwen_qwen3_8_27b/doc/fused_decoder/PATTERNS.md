# Fusion inventory

Inspection roots: ttnn/cpp/ttnn/operations, ttnn/ttnn/operations/transformer.py and binary.py golden functions, tests/ttnn/nightly/unit_tests/operations/experimental, models/common/modules/mlp/mlp_1d.py, models/tt_dit/utils/tensor.py, models/demos/deepseek_v3_b1/micro_ops.

## Baseline graph and movement

All weights BF16/HiFi4, FP32 accumulators; recurrent arithmetic/state FP32. Residuals DRAM interleaved.

|Section|Sequence|Movement|
|Common input|RMSNorm|DRAM output|
|Linear projections|qkv,a,b,z independent matmuls|same LHS read four times|
|Linear convolution|concat history; four slices*multiply; three adds; SiLU; copy history|sliced tile tails|
|Linear q/k|split, reshape; cast FP32; square,sum,epsilon,rsqrt,multiply|explicit untilize/tilize for reshape|
|Linear recurrence|head permute, GQA repeat, scale, scalar exp/reshape; FP32 generic recurrence; output reshape|head-major padded per-token tensors; tilize/untilize|
|Linear output|RMSNorm, SiLU(z), multiply, reshape, out projection|reshape roundtrip|
|Full projections|q,k,v,q_gate independent matmuls; q/k RMSNorm|reshape roundtrips, head permutation|
|Full RoPE|slice rotary64/rest192; split halves,neg,concat; two mul,add; concat|several intermediates|
|Full cache/attention|paged fill/update; chunked/paged SDPA|decode K/V height sharding required by update op|
|Full output|permute/reshape; sigmoid gate,multiply; out projection|head concatenation|
|Common finish|residual add,RMSNorm; gate/up matmuls; SiLU,mul; down matmul; residual add|DRAM intermediates|

Baseline B1 decode: linear101 device ops, matmuls1977.859us/whole2660.428us; full61 ops, matmuls1923.443us/whole2456.269us. Complete envelopes include all gaps. Native reshape and binary activations reduced linear to89 ops but did not improve prefill.

## Skill patterns

|Pattern|Assessment / experiment|
|Dedicated activations|Already SiLU/sigmoid/softplus; binary fusion tested candidate1|
|Softmax / SDPA / stable softmax|Already dedicated paged/chunked SDPA; no explicit scores or max subtraction|
|RMSNorm|Already dedicated. Decode L2 replaced with FP32 RMSNorm, epsilon1e-6/128 and weight1/sqrt128; l2_rms.json passes direct equivalence|
|Distributed RMSNorm|No collective/single device; not applicable|
|Split QKV / decode create heads|Packed projection + dedicated prefill split selected; decode-specific create heads adapted and measured without a gain (decode_heads_crop/native_concat.json)|
|Prefill concatenate heads|candidate2 dedicated concatenate_heads passed real-weight prefill|
|Decode concat heads|Native reshape selected; dedicated sharded concat adapted for padded batch output, passes but slower (candidate9/10)|
|RoPE|candidate2 dedicated rotate-half; per-request cos/sin slicing preserves ragged positions. Initial decode padded output fixed by logical crop|
|TopK|No sort/topk in decoder|
|RepVGG conv sum|No parallel spatial convolutions|
|Shared-LHS projections|Mixer QKV/z/a/b or q/k/v/gate packed. MLP packed gate/up tested, then separate gate matmul with fused SiLU wins both phase profiles (candidate15)|
|Spatial mean|No spatial reduction|
|Permute-reshape-permute|Native reshape candidate1; recurrence permutations need real ordering, not identity|
|Conv bias/scale/activation|No bias/scale; dedicated causal conv+SiLU selected after supported-config retry; no bias/scale to fold|
|Matmul activation|Dedicated SwiGLU rejected after supported geometry sweep; separate gate matmul with SiLU fused by explicit core grid selected (candidate15)|
|Binary input activation|candidate1 SiLU/sigmoid gates|
|Matmul bias|All target projections bias-free|
|Transpose into matmul|Weights transposed once in setup; no runtime transposed matrix operand|
|Slice after matmul|All projected columns consumed; packing slices are partitioning, cannot narrow total work|
|BatchNorm fold|No BatchNorm|
|Pad into pool/conv|No pooling; causal convolution accepts physical32-row padding with logical history updates; all public tails supported|
|Reduction keepdim / scaled sum|L2 uses keepdim=True already; no redundant reshape of a reduction or explicit mean|
|Decode RoPE layout folding|Decode-mode token_index0 tested with dedicated decode heads, including logical cropping; correct but no measured win over selected interleaved layout|
|Additional chunk delta rule|selected flat rank3 prefill path fuses L2/GQA/recurrence; long context262142+decode and262144 pass; batch kernel split preserves32 users|
|Additional fused SwiGLU|minimal_matmul tile-pair packing correct but slower, including geometry adaptations; whole-MLP one-expert kernel exceeds L1 after supported-config retries|

## Additional dedicated-op assessment

- `sigmoid_gated_rms_norm`: prefill selected, consuming head-major recurrence and packed gate columns, followed by multiplication byz (SiLU semantics). Decode adaptation with padded32 time rows is correct but slower; norm_decode.json.
- `rotary_embedding_llama_fused_qk`: this kernel multiplies each32-column tile by a32x32 transformation (`device/kernels/compute/rotary_embedding_llama_sharded.cpp`). Qwen's partial64 rotate-half exchanges the two32-column tiles; the selected rotate-half op implements that math directly. The tile-local Llama transform cannot express that permutation without additional channel interleaving/uninterleaving and different physical K-cache convention. It is not a drop-in equivalent to this graph's partial rotate-half. The correct dedicated rotate-half and decode-mode alternatives were measured.
- Distributed RMS, fused CCL, DeepSeek MLA/routing, spatial convolution/DiT modules, top-k and sampling do not appear in this single-device dense-decoder graph. No MoE active-work assumption is used.
- Conv `with_carry` is a sequence-parallel history-exchange wrapper; no neighboring sequence rank exists here. Current logical tail history must exclude physical pad rows. Ordinary fused causal convolution plus explicit logical history update preserves this contract.
- `minimal_matmul_split` was directly tested for equal-width gate/up with K16/N10, matched M-block, BF16/HiFi4 and fused binary SiLU. minimal_split.json passes HF/functional equivalence but takes2789.998us in the whole traced host screen versus2357.664us selected gate-activation control. It is rejected. Unequal Q/K/V/gate widths do not satisfy its equal-chunk contract.
- Fused residual RMSNorm cannot eliminate the residual addition here: the same summed residual is required by the final residual addition. Folding it into the norm would retain or duplicate that addition.

Row-major convolution history is selected: row_major_state.json passes both update branches, padded tails, continuation and direct equivalence; final_linear profile2328.099us traced decode beats the best tiled-state candidate2352.030us. It removes five device ops and reduces state storage by593920 bytes/request. The affected final validation is tracked in linear_state_validation.log.

Final review exposed 32 prefill history tilize/untilize pairs. Converting before the unaligned logical tail slice removes all 64 operations; `tracy/history_slice_final` records 931 prefill ops, 83008.778 us prefill and 2325.657 us traced decode. `history_slice_validation.log` passes all affected watcher/context/batch/boundary checks.

All listed patterns and relevant dedicated-op candidates have now been applied, measured slower after adaptation, or excluded by exact math/op contracts. No unresolved graph-fusion candidate remains in this inventory. This statement covers graph fusing on this hardware and precision; it does not claim that later layout/sharding/precision optimization stages have no work.
