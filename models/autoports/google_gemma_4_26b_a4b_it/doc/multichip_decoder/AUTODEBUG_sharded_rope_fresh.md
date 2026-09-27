# Source-only AutoDebug: sharded decode RoPE

## Verdict

There are two separate native sharded HF RoPE defects.

1. Capacity bug, still proven: the sharded multi-tile kernel processes all `Wt` tiles inside single DEST acquire regions. D256 is `Wt=8`; D512 is `Wt=16`. Current FP32 half-DST capacity is 4 tiles, and the sharded factory reads but drops `dst_full_sync_en`, so Python cannot enable full-sync for this op.
2. Format-state bug, now proven by controls plus source: FP32 sharded decode fails at D64/D128, where capacity is not exceeded. The sharded compute kernel starts hardware for `(input, sin) -> sin_interm`, then immediately runs `(input, scalar) -> rotated_interm` without source or packer data-format reconfiguration. The interleaved kernel does perform those transitions.

I did not run tests, import TTNN, spawn agents, touch devices, read credentials, or modify implementation. This report uses only source plus the provided JSON/log artifacts.

## Component Evidence

- `models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/sharded_rope_contract.json:5-27` shows FP32/FP32 D64 failing with PCC `0.642`/`0.887` for 1/2 heads. `:41-75` shows FP32/FP32 D128 failing with PCC `0.573`/`0.934`/`0.895` for 1/2/4 heads. Those widths are at or below the FP32 half-DST capacity of 4 tiles.
- The same contract shows D256 with `requested_full_sync=true` still failing, PCC `0.525`/`0.545`/`0.577` (`sharded_rope_contract.json:113-147`), consistent with the factory dropping full-sync.
- `models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/sharded_rope_formats.json:5-17` shows FP32 input + FP32 tables at D64 failing with PCC `0.228`. `:103-115` shows mixed FP32 input + BF16 tables at D128 failing with PCC `0.089`.
- Homogeneous BF16 controls isolate the good lower-precision path: BF16 input + BF16 tables + BF16 destination is bit-exact to interleaved at D64 and D256 (`sharded_rope_formats.json:33-45`, `:75-87`). BF16 input/tables with FP32 destination is good at D64/D128 but fails at D256 (`:19-31`, `:47-73`), and D512 BF16 destination fails capacity (`:89-101`).
- The probe varies width, `fp32_dest_acc_en`, full-sync, input dtype, and table dtype, then compares interleaved `FusedAttention.rotary` against sharded `ttnn.experimental.rotary_embedding_hf` (`models/autoports/google_gemma_4_26b_a4b_it/tests/probe_sharded_rope_contract.py:35-61`, `:73-118`).

## Capacity Proof

- Decode selects the sharded factory unconditionally for HF RoPE (`ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/rotary_embedding_hf_device_operation.cpp:15-20`) and validates height-sharded input/cos/sin in decode mode (`:55-68`). HF RoPE does not contain the nearby Llama guard that rejects `head_dim > 128` with FP32 DEST (`ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/rotary_embedding_llama_device_operation.cpp:67-70`; fused QK has the same guard at `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/rotary_embedding_llama_fused_qk_device_operation.cpp:59-62`).
- The sharded factory computes `head_dim_t = shard_spec->shape[1] / TILE_WIDTH`; D256 => 8 tiles and D512 => 16 tiles (`ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/rotary_embedding_hf_sharded_program_factory.cpp:240-244`). It passes that as compute arg `Wt` (`:363-375`).
- The factory extracts `dst_full_sync_en` (`rotary_embedding_hf_sharded_program_factory.cpp:247-248`) but constructs `ComputeConfigDescriptor` with only `.math_fidelity` and `.fp32_dest_acc_en` (`:385-388`). `ComputeConfigDescriptor::dst_full_sync_en` defaults false (`tt_metal/api/tt-metalium/program_descriptors.hpp:99-105`) and is otherwise propagated into kernel build options (`tt_metal/impl/program/program.cpp:452-458`, `tt_metal/impl/kernels/kernel.cpp:921-925`).
- `get_dest_reg_count` computes 16 BF16 tiles for full DEST, halves that when `dst_full_sync_en` is false, and halves again when `fp32_dest_acc_en` is true (`ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.cpp:13-17`, `:137-160`). Therefore current FP32 half-DST capacity is 4 tiles; FP32 full-sync would be 8; BF16 half-DST is 8; BF16 full-sync is 16.
- The sharded compute kernel uses `Wt` tiles in one acquire/commit region for `rotated*sin`, `input*cos`, and final add (`ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/compute/rotary_embedding_hf_sharded.cpp:94-118`, `:122-135`). D512 also uses `half_Wt=8` in the rotate-half stages (`:65-89`). That proves D256 FP32 and D512 BF16/FP32 overflow current capacity independent of format sequencing.

## Format-Sequence Proof

- `compute_kernel_hw_startup` must be called before operation init with CB IDs matching the next operation; if later operands need different data formats, callers must call `reconfig_data_format` first (`tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:21-29`). Startup programs unpack/math/pack formats from those CBs (`:53-73`).
- The sharded factory assigns input/cos/sin/output formats from tensor dtypes, but the scalar CB is hard-coded `Float16_b` (`rotary_embedding_hf_sharded_program_factory.cpp:222-235`, `:269-315`). Source does not prove BF16 cos/sin intermediates for FP32 tables: `rotated_input_interm` uses input format and `cos_interm`/`sin_interm` use cos/sin formats (`:317-349`).
- The sharded compute kernel starts with `compute_kernel_hw_startup(in_cb_id, sin_cb_id, sin_interm_cb_id)` (`rotary_embedding_hf_sharded.cpp:40`). Its first real op is `mul_bcast_scalar_init(in_cb_id, scalar_cb_id)` and it packs to `rotated_in_interm_cb_id` (`:65-76`). There is no `reconfig_data_format(in_cb_id, scalar_cb_id)` and no `pack_reconfig_data_format(... rotated_in_interm_cb_id ...)` between those two format contexts.
- The bcast init APIs do not silently perform the missing full format setup: `mul_bcast_scalar_init` and `mul_bcast_rows_init` call `state_configure`, math init, and unpack AB init (`tt_metal/hw/inc/api/compute/bcast.h:498-503`, `:533-537`), not `llk_unpack_hw_configure` / `llk_pack_hw_configure`.
- For FP32 input + FP32 tables, startup configures SrcB for FP32 `sin`, then the first op consumes BF16 scalar. That source mismatch explains D64/D128 FP32 failures where capacity fits. For mixed FP32 input + BF16 tables, scalar SrcB format matches but startup pack format is BF16 `sin_interm` while the first pack writes FP32 `rotated_input_interm`; the D128 mixed failure matches this.
- The interleaved compute kernel is the contrast case: it starts with `(rotated_in, scalar) -> rotated_in_interm` (`ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/compute/rotary_embedding_hf.cpp:62`), reconfigs source and packer before scalar/output changes (`:68-69`, `:81-82`, `:86-87`, `:100-101`), and processes one tile per acquire (`:64-113`).

## Adaptations

- Full-sync alone is insufficient from Python because the sharded factory drops `dst_full_sync_en`. Even after a native pass-through, FP32 full-sync capacity would be 8 tiles: enough for D256 but not D512.
- Paired-half chunks are capacity-only. They preserve rotate-half pairing for width chunks, but they do not fix the FP32 operand path because D64/D128 FP32 already fail. Paired-half FP32 chunks are not an FP32-preserving workaround until native format sequencing is fixed.
- There is no source-supported, FP32-preserving model-only adaptation for current native sharded HF RoPE. A native fix should add the missing `reconfig_data_format` / `pack_reconfig_data_format` transitions, thread `dst_full_sync_en`, and validate or chunk against `get_dest_reg_count(compute_config)`.
- The lower-precision model-local candidate is separate: homogeneous BF16 input/tables with BF16 destination is component-equivalent through D256, and D512 could use paired-half D256 chunks. Hardware owner will test that for model accuracy/latency. It is not FP32-preserving and should not be presented as the native FP32 fix.
- Current local worktree note: `models/autoports/google_gemma_4_26b_a4b_it/tt/multichip_decoder.py:173-225` already appears edited toward that lower-precision BF16 candidate: it sets `fp32_dest_acc_en=False`, uses `(32, 256)` sharded memory, typecasts value/cos/sin to BF16, and pairs D512 halves. I did not author or change that implementation in this AutoDebug pass.

## Table/Layout Notes

Table reuse is unlikely to be the primary bug. The non-sharded model path repeats decode tables over heads (`models/autoports/google_gemma_4_26b_a4b_it/tt/precision_ops.py:23-32`; `models/autoports/google_gemma_4_26b_a4b_it/tt/fused_decoder.py:528-536`). The sharded compute kernel pushes one cos/sin row per batch and reuses it across `heads_per_batch_t` (`rotary_embedding_hf_sharded.cpp:46-58`, `:141-142`). For local Q4/K2/K1 padded into one 32-row tile, row broadcast should be equivalent to repeated tables. The new controls point much more directly at format-state and DEST capacity.

The saved full run remains consistent with K-only RoPE damage: `models/autoports/google_gemma_4_26b_a4b_it/doc/multichip_decoder/sliding_router109_sharded_rope.log:39-41` fails cache PCC with first four entries around `.958-.965` and last four around `.999996`; `run_multichip_decoder.py:584-599` builds `cache_pcc` by iterating cache tensors then ranks, so the first group is K and the second is V.
