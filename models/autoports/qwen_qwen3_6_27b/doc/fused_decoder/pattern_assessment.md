# Graph-fusing pattern assessment

All named patterns have a candidate outcome or a source-backed applicability decision. Additional native opportunities from AutoDebug were tested through AutoFix; final gates/review are recorded in README and acceptance.json. Every named pattern in graph-fusing 0.1.23 is listed.

| Pattern | Applicability / experiment |
|---|---|
| Dedicated activations | SiLU/sigmoid/softplus already dedicated. Try input activation fusion below. |
| Dedicated softmax | Full attention already native paged/chunked SDPA; no spelled-out softmax remains. |
| Dedicated RMSNorm | Native RMSNorm cast absorption and fused cast/sharded transfers are kept after matched measurements (norm_absorb, norm_cast_only, combined_final). L2 remains native chunk-GDN flat normalization. |
| Distributed RMSNorm | Inapplicable: one device, no all-gather or slice. |
| SDPA | Already native paged decode and chunked prefill. GDN is not softmax attention; dedicated GDN candidates instead. |
| Split QKV and heads | Kept packed_qkv_3 / integrated_3: packed QKV and native split heads pass direct functional equivalence and are faster. |
| Decode create QKV heads | Tried decode_heads_3 with L1 output and compatible norm layout; 2.264 ms vs packed generic 2.218 ms, rejected. |
| Decode concatenate heads | Tried decode_concat_3; adapted padded batch output in decode_concat_adapted_3; 2.353 ms vs generic 2.334 ms, rejected. |
| Prefill concatenate heads | Tried: concat_3. Dedicated op supports logical decode length as well. |
| RoPE | Tried: rope_3 failed padded logical concat; rope_sliced_3 repairs slicing and passes. Final rope_hf_3 uses native shape-preserving HF rotate-half and removes repair slicing; 64 dims rotate,192 stay unchanged. |
| TopK | Inapplicable: no token sampling/router in decoder. |
| RepVGG conv-sum | Inapplicable: no parallel 3x3 and 1x1 spatial convolutions. |
| Shared-LHS matmul | Kept packed QKV and packed GDN QKV/Z/A/B; rejected packed MLP gate/up and packed QKV+output gate (packed_qkvg_3: 2.232 vs 2.218 ms). |
| Spatial mean | Inapplicable: no spatial H/W reductions. |
| Permute-reshape-permute | Kept flat_direct_0 (2.564 ms vs chunk_decode_fixed_0 2.622 ms) and head_major_0 (prefill 6.097 vs conv_carry_0 6.306 ms). |
| Conv+bias | No conv bias in checkpoint. |
| Conv+scale | No post-conv constant per-channel scale; SiLU is nonlinear. |
| Conv+activation | Kept KDA causal conv+SiLU with returned carry after adapting packer_l1_acc=False. Internal padding and actual_end preserve logical history; conv_carry_0 / contract_0. |
| Matmul+activation | Tried linear_activation_0/3; API without core grid uses trailing activation. Retried linear_activation_grid_0 with real fusion and matched-grid control; slower than binary input activation (2.637 vs 2.564 ms), rejected. |
| Input activation+binary | Kept binary_0/3 and gate_fused_adapted_0: SiLU/sigmoid/softplus folded into binary ops, with explicit FP32 GDN bias/output. |
| Matmul+bias | No dense biases except FP32 GDN dt_bias after BF16 projection. linear_bias_0 passes PCC but has no material gain (2.563 vs 2.564 ms); rejected. |
| Transpose+matmul | GDN chunk primitive replaces transposed matrix sequence; wrapper output layout selected to stay tiled. No such edge in native SDPA. |
| Slice after matmul | Already splits HF interleaved query/gate weights at load, and removes obsolete q_proj uploads. Other slices retain every projection output. |
| BatchNorm+conv | Inapplicable: no BatchNorm. |
| Pad+pool/conv | No pooling. KDA conv requires tile-aligned sequence, so keep logical-length padding and slice inside candidate. |
| Stable softmax max subtraction | Native SDPA owns stable softmax; no separate max subtraction. |
| Reduction+reshape | L2 already keepdim=True; native GDN removes remainder. |
| Scaled-sum to mean | L2 sum is not a mean and cannot change scale/epsilon. No scaled-sum mean subgraph. |
| RoPE decode layout folding | decode_rope_3 tried token_index=0 on per-row decode layout; 2.348 vs 2.334 ms, rejected. |

Additional dedicated kernels found: chunk_gated_delta_rule, fused_recurrent_gated_delta_rule, KDA qkv_causal_conv1d_silu. KDA sigmoid_gated_rms_norm uses sigmoid, whereas Qwen GDN uses SiLU, so not equivalent. Residual RMSNorm fusion cannot remove the residual add because its sum is also the residual stream output; a fused norm would duplicate addition and alter the BF16 rounding boundary if using FP32 operands.

Source exploration includes ttnn/cpp/ttnn/operations/transformer, normalization, experimental/kda, experimental/transformer; attached goldens in ttnn/ttnn/operations/transformer_golden.py; matching transformer and KDA unit tests; real model wiring in models/demos/qwen3_vl/tt/attention.py and models/experimental/qwen_image_2_1/tt/text_encoder.py.

Native factory variants also tested: gdn_phased_0 and gdn_fused_0. Automatic selection retained because phased is slightly faster for one-token decode, fused is faster for 512-row prefill. The native mono factory cannot consume flat QKV or fold L2/head-map; it is a benchmark reference, not an untried topology after testing both compatible implementations.

Additional structural shared-weight rewrite: fold batch/sequence into matmul M, then restore the public shape. `batch_fold_0/3` passes PCC and materially improves batch32; kept. No batch dimension is hard-coded to one, and logical batch remains distinct from tile padding.

## Additional native fusion evidence from AutoFix

- Native minimal_matmul SwiGLU: tile-pair-interleaved BF16 gate/up; default and Kblock8/16/20,Nblock10 adaptations pass PCC but are slower. `native_swiglu_0/3`, `swiglu_k*_0`; rejected.
- GDN conv-output slice/pad cancellation, flat Z gate ordering, and logical/padded output metadata: `padded_inputs_adapted`, `flat_z`, `flat_z_view` kept. A broader padded-chain version lost and was rejected.
- Native phased prep/scan gate masking at B1/s1: `logical_gates_0` kept; eliminates two explicit gate pads. Larger shapes use public native GDN waves.
- Native sigmoid-gated RMSNorm plus an external Z multiply expresses SiLU gating algebraically. Initial packer flag and rank1 weight requirements were adapted; `gated_norm_weight_0` passes PCC but2.167ms is slower than flat_z_view2.162ms before logical-gate fusion. Rejected.
- Native fused K/V update: `cache_fused_3` uses disjoint shard core sets, passes and improves2.067->2.061ms. Kept.
- Native HF rotary shape retention: `rope_hf_3` passes and improves2.061->2.058ms. Kept.
- General native BF16-input RMSNorm keeps FP32 intermediates/gamma; improves prefill. Decode uses cast-fused eight-core normalization, which beats native unsharded control. `combined_final_0/3` measures both together.

The explicit stopping condition was reached after these experiments (see roofline.json and work_log). Further native alias-state/ring and residual-epilogue ideas are hand-off notes, not claims of measured rejection. No change lowers the0.995 acceptance bar.
