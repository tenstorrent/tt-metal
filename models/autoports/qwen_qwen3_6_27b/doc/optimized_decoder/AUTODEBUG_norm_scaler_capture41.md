# Sharded RMSNorm scaler audit and remaining step209 failure

CPU/source only, no device or main decoder edits. New artifacts: `norm_scaler_cpu.json`, `capture41_cpu.json`; the latter is reproducible with `tests/analyze_long_capture.py --capture-dir .../capture41 --output .../capture41_cpu.json`.

## Scalers: a concrete representation mismatch, but insufficient alone

The current 8-core/640-column sharded RMSNorm does **not** use `1/sqrt(640)` and `1/sqrt(8)`. Its reductions are `REDUCE_ROW`, so the factory selects `winv=8/5120=1/640` and `cinv=1/8` (`layernorm_op_multi_core_sharded.cpp:265-278`). Square root is applied after the global sum. Float32 input and affine do not change these constants.

Source chain:

1. Factory runtime context stores `pack_two_bfloat16_into_uint32({bfloat_cinv,bfloat_cinv})` and similarly winv (`:392-394`). The host BF16 conversion is RNE (`tt_metal/impl/data_format/bfloat16.cpp:20-36`).
2. `sharded_layernorm_factory_helpers.cpp:1872-1880` passes these packed words as `scalar_c` and `scalar_w`.
3. `writer_unary_sharded_ln.cpp:73-76,91-96` bitcasts those words to **float**, then calls `prepare_reduce_scaler`.
4. `sharded_layernorm_factory_helpers.cpp:649-660` declares the local scaler BF16 but the global scaler **Float32 whenever FP32 destination accumulation is enabled**.
5. `reduce_helpers_dataflow.inl:27-40` preserves the complete float bits for a Float32 scaler. For BF16 it takes the high16 bits, restoring the intended packed local value.

Therefore the local scaler becomes BF16 `0x3acd` = `.00156402587890625`, a +.09765625% error versus1/640. The global scaler word `0x3e003e00` is interpreted as FP32 `.12523651123046875`, a +.18920898% error versus1/8. Its lower BF16 copy is spuriously part of the FP32 significand. With no other operand rounding, the computed mean is multiplied by `1.002870500087738`, predicting a normalization gain of **.9985678324973484** (epsilon negligible for these activations).

This source-derived bias is real but **does not fully explain** observed gains .996323/.995854, nor their sample variation. TF32 operand intake, intermediate packing, and later affine/output boundaries remain additional contributors. The source itself notes Float32 FPU operands pass through TF32 (`api/compute/reduce.h:109-112`; sharded helper unpack-mode comments). The calculation above is explicitly not a complete hardware simulation and must not be used as a fitted output correction.

## Decoder-local options

- **Use the measured unsharded FP32 norm candidate now.** It bypasses this sharded global-scaler construction, and independently improves the sequence. It still has a residual numerical discrepancy (below).
- **A primitive FP32 norm control avoids the nonstandard reduce scaler entirely:** square FP32 x; `ttnn.sum(...,dim=-1,keepdim=True,compute_kernel_config=HiFi4/FP32)`; multiply the result by FP32 `1/5120`; add epsilon; FP32 rsqrt; multiply FP32 x and FP32 folded `(1+weight)`; finally cast BF16. `generic_reductions.hpp` exposes the compute-kernel config. Keep sum and division separate so the reduction scaler is exactly1. This is a legal device-only decoder-local control, not a performance claim; validate output and sequence before fusing it into a local kernel. TF32 reduction intake can still introduce error and should be measured. A local SFPU reduction that explicitly uses `UnpackToDestFp32` is the higher-precision fallback if this primitive control remains insufficient.
- **Changing core count alone is not a complete repair.** Width1024 on5cores or512 on10cores makes the local power-of-two reciprocal exact, but packed global Float32 contamination remains; idealized gains are .998919/.998924. Other core counts and exact values are in `norm_scaler_cpu.json`. Such configurations are candidate controls only, not claimed fixes.
- `use_welford=True` is **not legal for RMSNorm**: `layernorm_device_operation.cpp:331-335` rejects it explicitly. `legacy_reduction` is read/stored but not used by this factory to repair the scalar representation. Disabling FP32 destination would change the global scaler to BF16 and remove the mixed-format issue, but would also lower statistics precision; no evidence supports keeping that as the accuracy repair.
- Analytically rescaling gamma/epsilon could cancel a pure scalar bias in exact arithmetic, but this measured path contains additional errors and no supported scaler override. Do not choose fitted output scaling. Shared-kernel changes are outside this stage's authorized file scope.

## Capture41: state rounding is active; head4 still amplifies state error

Capture41 is the exact round+unsharded policy at step209/user25. Eager and traced output are bit-identical. The TT old state is **exactly representable in BF16**, confirming the boundary round was executed. This is no longer an omitted cache-cast failure.

| Counterfactual, complete exact HF tail with TT z | Final PCC |
|---|---:|
| Actual TT output | .992493177 |
| Captured TT core + HF tail | .992520065 |
| TT inputs / TT old state / HF qk norm | .993817975 |
| HF inputs / TT old state / HF qk norm | **.996107753** |
| TT inputs / HF old state / HF qk norm | **.999467195** |
| HF inputs / HF old state / HF qk norm | .999939654 |

As before, the kernel matches its own-input FP32 CPU recurrence: core max error1.49e-8, state1.19e-7. Current q/k normalization alone still fails. Improving all current inputs clears this specific cell, and replacing historical state clears it more strongly. Both accumulated and local discrepancies contribute; do not infer a sufficient whole-sequence fix from one counterfactual.

Head4 contributes54.74% of gated squared error. Its HF core RMS is .000434827, core PCC .983939, old-state relative RMS error1.4951%, and state fitted gain .995962. Across the selected user's state, relative RMS error is1.3958% and fitted gain .997885. Other major gated-error heads are25(10.28%) and40(6.88%). Prefix whole-batch state relative RMS error was already .9341% / PCC .99996768. The saved artifact contains no selected-user prefix state, so **this capture cannot partition prefix error from subsequent historical drift**. A controlled exact-HF-prefix-state diagnostic, or saved selected-user prefix tensors, is still needed for that question.

Even unsharded input norm has relative error .2638% and fitted gain .998768. Exact BF16 QKV on TT input norm differs from HF by .3417% RMS; actual TT QKV differs from exact same-input projection by only .0987%. Thus norm/input precision remains a more promising repair than further weight decomposition. Exact HF MLP on TT post-norm/residual still gives .994358 (below .995), versus actual TT .992493; it agrees with TT at .999840. MLP arithmetic contributes modestly here but still cannot alone clear the unchanged gate.

Priority: correct hidden norm accumulation/scale without fitted compensation, then test HF q/k boundary controls and isolate prefix state. Preserve BFP8 weights and the .995 gate.
