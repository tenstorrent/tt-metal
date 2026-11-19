# BF16 state rounding AutoFix

Original evidence is unchanged: `bf16_state_staged_direct_watcher.B1.wave104.step0.rank0.state_vs_quantized_control.pt` and its log/JSON. The decoder's original BF16-staged long test also failed its unchanged .995 output gate at decode70/request8 (PCC .993284, reported by root). Neither is waived.

## Saved-value diagnosis

CPU inspection finds 108 differing values out of 131072, all in head1. Every mismatch is one BF16 representable step: the candidate has a larger absolute magnitude and odd retained significand LSB; the control has even retained LSB. The maximum absolute difference is 1.52587890625e-5. Reconstructing the initial BF16 state from seed7906, every mismatch midpoint corresponds to the same decay factor .107421875. Step0 has zero Q/K, so its recurrence reduces to state decay and output is identically zero. This pattern strongly supports a rounding-boundary discrepancy rather than an index/CB ownership error. `failure_analysis.json` records tensor hash, counts, and samples.

The original failure artifact lacks the pre-round FP32 control values, so it does not prove the exact pre-round bit patterns or the hardware packer's tie rule. We do not relabel a bitwise failure as a pass.

## Source distinction and isolated repair

The control's `ttnn.typecast(FP32, BF16)` preserves FP32 input (`typecast.cpp:33–40`, `typecast_program_factory.cpp:42–45`) and explicitly executes SFPU RNE conversion: shift to obtain retained LSB, add0x7fff plus LSB, mask low16 bits, store FP32 (`tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h:273–287`). The candidate previously selected BF16 pack format and directly packed the accumulated FP32 DST values without that explicit SFPU step. These are distinct rounding paths.

Fresh `tests/multichip_bf16_state_rne_kernel.py`, `_candidate.py`, `_probe.py`, and `tests/kernels/bf16_state_rne` preserve original files. The only recurrence change applies the public `typecast_tile_init<Float32,Float16_b>()` and `typecast_tile<Float32,Float16_b>(j)` to each final-state DST tile after both matmul contributions and before committing/packing it. The resulting BF16-representable FP32 value then packs exactly. This directly matches the control's explicit quantization primitive without inserting another state-memory round trip.

Typecast initialization resets MATH address modifiers/counters (`ckernel_sfpu_typecast.h:608–613`), so final-state `matmul_init` is moved inside each DST block, before acquire and accumulation, restoring the matmul state after the prior block's SFPU conversion. Output/diff/intermediate state arithmetic and order are unchanged. Reader, writer, all CB formats/counts, and persistent BF16 storage are unchanged. The shared FP32 arithmetic and exact HiFi4 policy remain in force. `scan.diff` captures the complete recurrence modification.

## Pending discriminating tests

`doc/optimized_multichip_decoder/bf16_state_rne_jobs.json` contains direct Watcher plus B6/B32 short and B32 1024-answer decoder jobs. The direct wrapper runs the existing unchanged .995 per-head/output gate and bitwise quantized-state gate with the new kernel. No tolerance option or waiver exists. Its unquantized FP32 control still reports precision loss separately. The long HF-reference decoder gate is mandatory: fixing quantization equivalence cannot establish that storing BF16 state retains model quality. A repeated long failure would support rejecting the precision policy, while a direct failure still requires implementation diagnosis.

Python compilation passed and parent file hashes remain unchanged. Public typecast API signatures and initialization side effects were inspected locally. No C++ device build or hardware test was run by this agent; all hardware conclusions remain pending serialized root execution.
