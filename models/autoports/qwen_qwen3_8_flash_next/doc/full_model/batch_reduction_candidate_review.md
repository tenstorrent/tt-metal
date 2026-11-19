# Batched decode reduction candidate: independent source review

Reviewed 2026-11-14, CPU/source only. No device execution or runtime edits by the reviewer. **No policy acceptance or repaired accuracy/performance claim.**

The diagnostic [probe](../../tests/probe_batch_reduction_counterfactual.py) has no blocking source defect identified. Its BF16 mode is a candidate scheduling optimization within the existing multichip strategy, conditional on measured performance and unchanged accuracy gates. FP32 mode changes the accepted batched accumulation precision and is diagnostic only.

| Contract | Source finding |
|---|---|
| Batched decode fidelity and destination | BF16 mode explicitly selects HiFi4, `math_approx_mode=False`, `fp32_dest_acc_en=False`, `dst_full_sync_en=False`. These match the native `post_combine_reduce` factory and `ComputeConfigDescriptor` defaults. |
| Accumulation schedule | Changed explicitly: native per-expert scalar products and BF16 L1 pack accumulation become one vectorized `[1,n,1,10] @ [1,n,10,2560]` matmul with `packer_l1_acc=False`. This changes rounding order; it does not promise bitwise equality to native arithmetic. |
| Activation and output precision | BF16 values, scores, matmul output and public result. FP32 conversion is confined to existing expert-ID ownership-mask metadata, as in the accepted single-row method. |
| Optimized multichip strategy | EP4 grouped expert dispatch/execution and its BFP4/LoFi policy are untouched. No per-request Python expert loop, replicated model, single-chip fallback or host arithmetic is introduced. Shared-expert computation and BF16 CCL remain outside the replacement. |
| Single-row decode and prefill | The scoped hook calls the original method for `n == 1` or outside `decode_forward`; accepted FP32 single-row reduction and all prefill reduction stay unchanged. |
| Residual and cache contract | Hook changes only local routed-expert reduction. Inter-layer residual layout, cache dtype/ownership, positions, page tables and sampling remain unchanged. |
| Layout | Internal untilize/reshape/tilize removes physical M=32 padding before moving logical batch rows into `[1,1,n,2560]`; result is BF16 tiled DRAM. Hardware support and cost of these intermediate layouts are still unproved by source review. |
| Trace safety | Patch covers generator preparation as well as capture. `prepare()` warms every configured logical decode width. Configuration caching retains host descriptors, not newly allocated persistent TT tensors. Selection uses static Python call context and shape during warmup/capture; traced replay does not execute this Python selection. Hardware tracking must still validate the concrete op sequence and all supported widths. |
| Diagnostic isolation | Context manager restores patched methods and call depth after exceptions. Output paths must contain `counterfactual`, cannot overwrite existing output, and both modes force `passed=False` and `acceptance_eligible=False`; original gate outcome is separately `diagnostic_checks_passed`. |

Source references: `tt/multichip_decoder.py` (`MeshMoe.__call__`, `_reduce`, `_reduce_single`, `MultichipDecoder.decode_forward`); `tt/decoder_components.py::WeightLoader`; `tt/generator.py::prepare`; `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/post_combine_reduce_program_factory.cpp`; `tt_metal/api/tt-metalium/program_descriptors.hpp`.

The user's preservation requirement permits an evidence-backed faster BF16 device schedule that retains the established multichip, dtype, fidelity, cache, CCL and residual contracts. It does not justify accepting a slower fallback, widening batched DEST to FP32, weakening the rejection ledger, or relaxing batch accuracy thresholds. Any production selection needs explicit decode dispatch, fresh full-model accuracy, frozen batch/lifecycle gates through B32, trace evidence and measured performance. The reduced diagnostic does not supply those acceptance proofs.

## Evidence motivating this candidate

The independent [AutoDebug report](batch_reduction_AUTODEBUG.md) completed after paired launcher session `46361` exited 0 (process result supplied by the parent device owner). The [launcher log](batch_reduction_autodebug_paired.log) records completion. Its CPU replay of the documented exact ISA arithmetic matches **61,440/61,440 saved native BF16 output bit patterns** across three synthetic fixtures, including the one-owned-expert case. Saved input: `/home/ttuser/qwen38-flash/reference/moe_reduction_rows_v2.pt`, SHA256 `fc993d7611099f85665d8b6772d8efcdd8b12be470afbb2c0af1d986bf938614`. This rejects the earlier approximate CPU RNE goldens as sufficient evidence of a native kernel defect. It is not an all-domain arithmetic proof or full-model acceptance.

The subsequent [real capture](moe_decode_capture_v1.json) completes in 149.946 s using real layers 0/1/3, matched prompts of lengths 3 and 33, and forced decode inputs 42/43. It localizes the first recorded divergence to **layer 0 routed reduction** for both inputs:

| Forced input | Layer-0 input / IDs / weights / owned combined / shared output | Reduced different elements /20,480 | Reduced maximum absolute delta | Reduced PCC |
|---|---|---:|---:|---:|
| 42 | All exact | 8,901 | 0.000244140625 | 0.999992786970207 |
| 43 | All exact | 9,196 | 0.00048828125 | 0.9999951384689607 |

“Owned combined” deliberately excludes nonowned stale slots; the artifact does not claim all combined storage is equal. Later layers already receive different inputs. The capture adds the internal real-model localization absent from the earlier AutoDebug report; it does not retroactively change that report's scope or establish that any proposed repair passes the full-model gate. Matched prefill is exact. All 12 exported fixtures and their SHA256 hashes are listed in the capture JSON; complete CPU capture is `/home/ttuser/qwen38-flash/reference/moe_decode_geometry/moe_decode_capture_v1_capture.pt`.

The existing shared-pool v5 gate remains failed for cross-geometry token43 and the survivor after B2 history, despite exact same-geometry pool/private and migration controls. No threshold or requirement was removed by this review.

Reviewed probe SHA256: `c6514bcd5d1a8849994a622bb76d1ba87d6be370050a0a9bb6e843576a97e874`.
