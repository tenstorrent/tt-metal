# Batched decode convolution and GDN investigation

Source-only, 2026-11-12. No hardware runs; no changes to optimized_decoder.py. The helper `candidates/batched_conv.py` passes Python syntax compilation only and is an unselected candidate.

## Convolution candidate

For one token the exact operation is `silu(((history0*tap0 + history1*tap1) + history2*tap2) + token*tap3)`; history becomes `[history1, history2, token]`. Native compute (`experimental/kda/qkv_causal_conv1d_silu/device/kernels/compute/qkv_causal_conv1d_silu.cpp`) constructs BF16 diagonal tap matrices and accumulates four HiFi4 matmuls in FP32 DST without intermediate BF16 packs, then executes SILU before BF16 packing. Therefore simply calling the functional BF16 multiply/add path is not the right numerical control.

The candidate keeps four products and the first two sums FP32, fuses SILU into the final FP32-input add with BF16 output, and retains the existing state object via `copy`. `binary_ng_program_factory.cpp:1134-1145` enables FP32 destination accumulation when any input **or output** is FP32, including the final BF16-output add. `binary_ng_device_operation.cpp:388-415,682-685` permits ROW_MAJOR operands and preserves ROW_MAJOR output when both operands are ROW_MAJOR. This is a supported device path, not a host fallback.

All arithmetic uses compact `[1,B,10240]` ROW_MAJOR tensors. The `[B,1,10240]` token is untilized once; each history row is sliced and metadata-reshaped. Four tap tensors are converted to ROW_MAJOR once at construction. There is no per-user loop or per-user padded intermediate. The final compact result is reshaped back to `[B,1,10240]`, split into q/k/v and tilized for the existing GDN consumer. This removes 32 native conv calls, per-user slices/concats, exchange operations and padded-token arithmetic. Necessary consumer layout conversions and one history copy remain explicit and must be timed.

Numerical caveat: these are the same FP32 algebra and accumulation order, but native matmul and elementwise multiply/SFPU SILU are different implementations. Exact output equality is not established by inspection. BF16 products are representable in FP32, so intermediate storage need not introduce BF16 rounding; compare real-weight output at the conv boundary, full decoder PCC, final histories and repeated traces. Do not claim native bit equality without measuring. If binary-ng activation differs materially, compare separate FP32 SILU followed by BF16 conversion before rejecting. If ROW_MAJOR broadcasting encounters a runtime issue, adapt the compact `[1,B,C]` operands to TILE as a second candidate; that still pads once across users, not once per user.

## Stronger GDN opportunity: remove full recurrent-state copy

The public wave path allocates a fresh full final state and then decoder copies it. The existing **private Python-bound** `ttnn._ttnn.operations.transformer.chunk_gdn_fused` accepts `o_out` and `final_state_out` together, as well as `num_heads` and `head_offset` (`chunk_gated_delta_rule_nanobind.cpp:797-840`). It can reuse the current wave48/pool geometry while writing final state directly into the original recurrent allocation:

1. Prepare a persistent FP32 output `[B*48,1,32,128]` before trace capture.
2. Metadata-reshape recurrent state to `[B*48,128,128]` (last two dimensions unchanged).
3. Launch the same legal 48-head waves via the private binding, preserving current q/k/v, padded gates, constant tensors, kernel config, and `v_flat=True, HV=48, qk_norm=True, scale=128**-0.5, qk_flat=True, Hk=16`.
4. For every wave pass the same state view as both `initial_state` and `final_state_out`, the persistent output as `o_out`, and disjoint `head_offset=0,48,...`. Return the same logical/padded output reshape as the existing code; eliminate the trailing full-state copy.

**Source-supported alias invariant:** the prep producer never reads recurrent state (`reader_chunk_gdn_prep.cpp` header). Each fused receiver reads exactly its own `(head,V stripe)` once before compute (`reader_chunk_gdn_scan.cpp:240-253,518-525`): all K rows are read and a NoC read barrier finishes before the CB is published. The scan writer waits for that receiver's final-state CB, then writes the same disjoint stripe (`writer_chunk_gdn_scan.cpp:63-75`). Different receiver V stripes and wave head ranges do not overlap. Consequently a stripe cannot be overwritten before its complete initial read, and no other receiver needs its old value. The supplied-output validation checks shape/dtype/memory and does not prohibit aliasing (`chunk_gdn_device_operation.cpp:140-150,283`). This supports an in-place candidate but still requires repeated replay/allocation tracking and watcher validation. Keep backing allocation ownership alive for every trace; do not replace `state.recurrent` with an independently allocated return tensor.

This preserves native arithmetic and all 32 wave launches, but can remove the reported ~0.5 ms copy and the extra ~96 MiB final-state allocation for B32. This is more direct than replacing the recurrent math.

## Can 32 wave calls become one phased call?

No: phased scan asserts `BH <= grid.x*grid.y` (`chunk_gdn_phased_program_factory.cpp:149-151`). Here BH=1536, well above 110 cores. Mono and fused recurrent primitives have the same one-head-per-core limit (`chunk_gdn_mono_program_factory.cpp:88`, `fused_recurrent_gated_delta_rule_program_factory.cpp:67`). Switching to those entire-batch primitives cannot legally eliminate waves. That is a source-proven contract blocker, not a first API error.

A legal comparison is two users/96 heads per **phased** call, with `scan_serial=True` (NV=1), giving 16 calls of two kernels each. It does not reduce launch count versus 32 single-kernel fused waves and introduces seven FP32 DRAM intermediates, though larger prep parallelism could be measured. Phased output state has no supplied-output binding, so its copy remains unless separately adapted.

A batched S=1 elementary recurrence (the functional `_delta_step` equations) avoids the head-core limit entirely, but reads/writes the 96 MiB recurrent state repeatedly through elementwise products and reductions. It is a legal performance candidate rather than an obvious win. The direct private fused in-place path is the first recommendation because it preserves current arithmetic/geometry and removes a measured full-state transfer.

## Required validation

Conv: compare q/k/v and histories against native at B1/B8/B32 on real projection output, nonzero history, sequential distinct tokens, and changed replay inputs; then full layer PCC, determinism and traced latency. In-place GDN: compare output/state to out-of-place native wave48 for multiple consecutive steps, mixed users and state resets; verify state buffer identity and no cross-user corruption; run representative repeated replay with allocation tracking and watcher separately. Neither candidate is a delivered or measured improvement yet.
