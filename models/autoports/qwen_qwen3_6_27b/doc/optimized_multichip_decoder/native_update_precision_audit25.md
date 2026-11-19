# Native recurrent update precision: independent follow-up

CPU/source-only audit, 2026-11-13. No TTNN/Torch import, device execution, build, production edit, or shared-probe edit. Original failure artifacts are unchanged. The extraction in `native_update_precision_evidence25.json` pins the exact input JSON by SHA256. This is a diagnosis and bounded AutoFix plan, not a verified repair.

## What the new evidence establishes

`native_readout_perf25.json` completes only B1/normal, then fails required quality after 702 identical-input updates (2 warmups + 7 samples × 100 replays). B3/B10/B32 were requested but not reached. Timing resets each path to its original state; the reference advances exactly the same 702 times. The timing loop does not contain an uncounted per-sample warmup (`probe_prepared_native_precise_readout.py:599–628`).

| Path | Complete boundary median µs | Output minimum head PCC | State minimum head PCC |
|---|---:|---:|---:|
| Custom BF16 state | 52.66576 | 0.9999972567 | 0.9999997759 |
| Custom FP32 state | 52.50886 | 0.9999999840 | 1.0000000000 |
| Prepared native FP32 | 89.08246 | 0.1938923913 | 0.9998690953 |
| Prepared native BF16 | 88.43071 | 0.9921885688 | 0.9999849616 |
| Precise readout FP32 | 97.87186 | 0.1774554954 | 0.9998690953 |
| Precise readout BF16 | 97.31198 | 0.9925352433 | 0.9999849616 |

Each state dtype is compared against its own persistence-matched FP64 trajectory. Original/prepared-precise state is bit-identical for both dtypes. Thus the earlier readout-only fix preserves native update arithmetic, but cannot repair this later failure. State PCC passing does **not** imply the Q-projected output passes: a contraction can expose a small state error in a direction with a small true signal. The FP32 result does not establish a large global state error or nonfinite corruption.

The earlier zero-state first-step repair remains real: `native_readout_zero_smoke_v2.json` improved native output PCC ~0.98878 → ~0.99999971 with unchanged state. `native_readout_stress24.json` passes eight changed-input cases. These are complementary observations; neither covers this 702-step fixed-input limit.

## Concrete hypothesis and source boundaries

Leading hypothesis: **the native recurrence converges to a perturbed fixed-point matrix because it repeatedly narrows state/decay/prediction/update operands; near cancellation in Q·K amplifies the resulting matrix error in output.** This is stronger than a remaining readout-only explanation, but the particular update phase has not yet been isolated.

For fixed normalized K, V, beta and decay d, exact arithmetic has

```
S_next = d (I - beta k kᵀ) S + beta k vᵀ
S_infinity = beta / (1 - d + d beta ||k||²) * k vᵀ
```

Orthogonal initial-state components decay as d^t. The fixture's least-negative g is -0.04, so the ideal initial orthogonal component at 702 is at most exp(-28.08), about 6.4e-13. Rank1/head6 has g=-0.055. Output at convergence is proportional to (q·k)v. A small q·k therefore exposes elementwise matrix quantization error that high state PCC can hide. This is an analytical explanation, not a measured q·k for the failing heads: the JSON contains metrics, not the required raw checkpoint tensors. BF16 persistence also changes the target fixed point, so better BF16 output PCC does not prove that BF16 is inherently more accurate than FP32.

The inspected native compute source is `ttnn/cpp/ttnn/operations/transformer/fused_recurrent_gated_delta_rule/device/kernels/compute/fused_recurrent_gated_delta_rule.cpp`:

- `bcast_scalar_mul`, line 141: FPU state × decay narrows both FP32 inputs.
- `mm`, line 39, called for prediction: FPU K × decayed-state contraction narrows matrix entries.
- `delta_rule_residual`, line 104: FPU subtraction narrows prediction before beta's SFPU multiply.
- `transpose_block`, line 159: K's path to the outer product also uses the ordinary unpack/transpose path.
- `rank1_update`, line 63: ordinary `copy_tile(sd,...)` seeds DST from narrowed decayed state; FPU K-column × U then accumulates into that seed. Merely preserving the final FP32 buffer does not recover lost input bits.

`prepared_native_precise_readout.py` configures only recurrence CB0/CB8 direct-to-Dest; CB5 state, CB3 decay, CB10 decayed state, CB11 prediction, CB12 delta and CB13 K-column remain on their inherited paths. `prepared_native_recurrent_candidate.py:270–300` invokes exactly those helpers before the precise readout. The unchanged-state evidence makes a readout-induced state corruption hypothesis unlikely for this case. The existing AutoDebug lowering analysis and `gdn_decode_test_utils.py:47` explicitly identify TF32-class FPU operand precision.

Preparation differences remain a weaker alternative. A fixed scalar decay or K scale perturbation alone mainly changes the rank-one scale at ideal convergence, whereas elementwise state narrowing can destroy separability. Check the actual prepared Q/K/decay against the custom/host preparation before attributing all error to a particular phase. Do not assume a software TF32 emulator's rounding mode exactly reproduces the hardware.

## Minimal discriminating experiment before more rewriting

Use a NEW isolated helper/probe; keep existing controls and failures frozen. The first hardware case is the exact B1/normal seed 52813+97*B+rank, bank0, both state dtypes, 702 fixed updates. Collect checkpoints 1/2/4/16/64/128/256/512/702 and save all paths' tensors before any failing metric assertion. Include changed-input 50-step and zero-state first-step controls afterward. No weakened per-head/user .995 gate, no cross-dtype reference substitution.

At each checkpoint save prepared Q/K/V/d/beta, native/custom state, output, initial state, and host oracle. Report per-head centered output norm, q·k, state error, `||qᵀ E||`, and output error. For FP32 persistence, the stored final native state equals the live packed state: contract that state with the **actual prepared Q** on CPU in FP64. If this agrees with precise-readout hardware while both disagree with the trajectory oracle, it directly localizes the failure upstream of readout. BF16 persistence cannot use this shortcut because its live output precedes persistence rounding.

Also restore the same saved incoming state into each family for one update at steps 1 and the first failing checkpoint. This distinguishes accumulated state error from one-step arithmetic error without requiring 702 repeated intermediate downloads.

### Bounded candidate ladder

Keep the two-launch prepared layout, compact final output, state dtype, ownership and precise readout unchanged. No public/native source patch is needed for this diagnostic.

1. **Preserve only the rank-one DST seed.** During native decay, pack each unchanged FP32 DST result to both existing CB10 and a new direct-FP32 CB17. Native FPU prediction keeps CB10/Default. Change only the rank-one seed copy to CB17/direct; explicitly select CB17 before copy and restore `(cb_kcol,cb_u)` before native matmul. CB17 is 16 FP32 tiles (65,536 bytes/core), reserved/pushed/popped once per head. This isolates the second narrowing of decayed state during the seed copy while preserving the original native decay, prediction, delta, outer-product and packing. Never mark CB10 direct globally while FPU prediction still consumes it. Confirm physical L1 headroom; the existing FP32 recurrence CB estimate becomes 606,208 bytes/core, excluding kernels/other allocations.

2. **Preserve decay, then prediction/residual as a coherent group if seed-only is insufficient.** Load CB5 state and CB3 decay directly and apply the existing exact SFPU scalar-product helper, pack the same CB10/CB17 values. Next, mirror prepared K as a column directly into new CB18 (four FP32 tiles, 16,384 bytes/core); keep original K row/transpose for the unchanged native outer-product control. Reuse the already verified `precise_readout` ordered contraction on CB17 × CB18 for prediction. Replace residual subtraction/beta multiplication with direct FP32 SFPU operations; after removal of its FPU subtraction consumer, CB11 can be direct. V/beta are BF16-representable fixture values but must still use explicit formats. Keep each adaptation separately selectable so phase attribution remains possible.

3. **If necessary replace only the outer-product update last.** With direct decayed state, K-column and U, form a separately rounded FP32 K×U product and FP32 add to decay, then pack the unchanged live/persistent outputs. CB12 can become direct only after its FPU matmul consumer is removed. Use at most the existing four half-sync DST slots; do not silently switch to FMA or change BF16 RNE. This completes a full-FP32 coherent family. It should converge toward the accepted custom arithmetic, but hardware proof is still required.

After the seed-only discriminator, the shortest decisive follow-up can jump directly to the complete coherent FP32-update family. Intermediate per-phase variants are useful only if locating a smaller retained native FPU path could plausibly save latency. All variants need explicit unpack reconfiguration and pack-mask clearing; a direct-only CB passed to remaining FPU math is an invalid experiment, not evidence against precision adaptation. Reuse the existing prepared source/descriptor proof approach; do not add a full-tensor FP32 restore, extra global state copy, collective, or public input constraint.

## Selection versus rejection

The **measured prepared-native and readout-only families** already have defensible rejections: complete-boundary B1 latency is 1.70×–1.85× the dtype-matched custom control and their 702-step output gate fails. These are adapted kernels with actual precision repair, layout preparation, matched traces, and numerical evidence, not first-API-error dismissals. Record them as slower and quality-failing; do not report their timings as accepted optimized numbers.

This does not prove every possible fused FPU/SFPU native topology is slower. If the parent requires one more numerical AutoFix discriminator, run seed-only plus the same-state CPU contraction first; use the bounded coherent FP32 update only if evidence warrants it. There is no justification to integrate the current native family, promote FP32 persistence, relax the gate, or hide the failing heads. The already passing custom fused FP32-arithmetic recurrence remains the production choice; a fully repaired two-launch native family must beat that complete boundary before further integration work.

Hardware acceptance still requires changed/rebound inputs, all relevant batches/worker reuse, exact eager-versus-trace replay, watcher, allocation positive control, and matched uninstrumented timings. This audit does not claim that any proposed update repair passes or that the overall optimization stage is complete.
