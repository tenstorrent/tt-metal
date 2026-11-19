# V6 full-mantissa FP32 product correction: source feasibility

**A model-local inline SFPU implementation using split products and error-free sum expansions is technically feasible. It is a new arithmetic helper to implement and validate, not an existing exact-matmul switch.** No hardware was accessed and the running `v6_chunk64_state_probe.py` was not changed. This memo does not claim that an exact implementation, acceptable latency, or a passing decoder has been achieved.

## What the source establishes

| Question | Repository evidence | Consequence |
| --- | --- | --- |
| Can inline Python-owned code run SFPU math? | `tests/ttnn/unit_tests/operations/debug/test_dst_math_clear.py:20–69,129–151` supplies an inline compute kernel to `ttnn.generic_op`, uses `TRISC_MATH`, SFPU instructions and FP32 destination registers. | C++ source can remain embedded in the allowed model Python file. No native/shared source edit or host calculation is required. |
| Can inputs retain all FP32 mantissa bits? | `tt_metal/api/tt-metalium/base_types.hpp:32–40` explicitly defines `UnpackToDestFp32` as full-precision FP32 unpack, incompatible with SRCA/B. `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:640–684` exposes FP32 destination and mutable `unpack_to_dest_mode`. | Configure each FP32 input/scratch CB for direct destination unpack. Avoid the FPU/SRCA/B path for the arithmetic requiring full mantissas. Merely setting `fp32_dest_acc_en` is insufficient. |
| Is native SFPMAD an IEEE FP32 FMA? | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_power.h:266–270` says its product keeps **four bits beyond FP32**, rather than the full exact product. `tests/ttnn/unit_tests/operations/eltwise/test_ternary_category1_float32.py:54–69` independently documents the lack of single rounding. | Sequential `addcmul` is an approximation control. It cannot be promoted as an exact implementation of the proven CPU sequential `std::fma` order from source inspection alone. |
| Are compensated SFPU operations practical? | The same unary-power header at `:190–207` implements FastTwoSum, and `:287–302` implements a Veltkamp split with `4097.0f` to retain product tails. The comments explicitly account for vector-register pressure. | There is an existing on-device precedent for full-mantissa arithmetic represented as two FP32 values. |
| Can the final rounding step inspect and adjust float bits? | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h:152–207` implements FP32 `nextafter` using `sfpi::as<vInt>`, integer increments, predication, and signed-zero/NaN classification. | Midpoint/tie correction can be vector code; it does not require scalar CPU emulation. |
| Can the compiler be kept from changing error-free operation boundaries? | `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h:704–724` exposes separate SFPMAD and SFPMUL instructions; SFPU cumsum uses explicit SFPADD instructions. | Emit the required separate rounded operations explicitly, and inspect generated kernel code. Do not assume C++ temporary assignments prevent multiply-add contraction. |

No exact CPU-FMA-equivalent matmul mode was identified in the inspected matmul, ternary, or compensated-SFPU implementation paths. Existing accurate-attention helpers such as `models/demos/k2_horizon_7b_qb2/tt/accurate_attention/accurate_decode.hpp:7–30` preserve FP32 state through A2D copies/SFPU arithmetic, but do not supply the requested exact single-rounded dot-product recurrence.

## Concrete candidate to implement

Implement a **vector FP32 multiply-add correction helper**, then execute it once per K element in the already-proven CPU order. This is SFPU arithmetic across output lanes, with device dataflow feeding operands; it is not a software emulation of the decoder on a RISC-V CPU.

1. Split each operand with the existing Veltkamp pattern:

   ```text
   c = RN(4097 * x)
   x_hi = RN(c - RN(c - x))
   x_lo = RN(x - x_hi)
   ```

   For finite, suitably scaled normal FP32 operands, the halves have short significands. The four pairwise partial products can preserve the complete product, unlike one SFPMAD. The existing power implementation only splits one operand because its other operand is a small integer; a general full-mantissa GEMM needs both operands split.

2. Use the standard split-product reconstruction, with distinct rounded operations:

   ```text
   p  = RN(a * b)
   e1 = RN(p  - RN(a_hi * b_hi))
   e2 = RN(e1 - RN(a_lo * b_hi))
   e3 = RN(e2 - RN(a_hi * b_lo))
   e  = RN(RN(a_lo * b_lo) - e3)
   ```

   Under the error-free transform's range/rounding assumptions, `(p,e)` represents the exact product. This property must be established for the actual SFPU arithmetic, including full-mantissa inputs; the repository's basic multiply grid uses BF16-width mantissas and is insufficient by itself.

3. Add the current FP32 accumulator to the product expansion using **general TwoSum**, not FastTwoSum unless its magnitude precondition is explicitly established:

   ```text
   s = RN(x + y)
   z = RN(s - x)
   error = RN(RN(x - RN(s - z)) + RN(y - z))
   ```

   Grow the increasing-magnitude expansion `[e,p]` by the scalar accumulator using TwoSum, eliminate zero components, then normalize/compress the resulting at-most-three-term expansion. Preserve the exact residual through the final rounding decision.

4. Round the exact expansion **once to FP32**, including midpoint ties. Simply computing `RN(high + RN(low1 + low2))` is not a proven replacement: the low sum can round onto a midpoint and lose the sign of a smaller term. Compare the normalized exact tail against the half-spacing to the adjacent FP32 number; use a residual/sticky sign for exact midpoint decisions and the output mantissa parity for ties-to-even. The existing `nextafter` bit manipulation supplies the adjacent-value operation. Handle exact cancellation and signed zero explicitly.

5. Start the K accumulator at the same zero as the CPU routine. Apply this corrected, single-rounded update **at every K iteration**. A compensated accumulator carried over the entire dot product computes a different numerical algorithm and is not a substitute for the frozen CPU recurrence, even if it is more accurate in real arithmetic.

6. Keep the accumulator and short expansion in SFPU registers where possible; spill temporary components only to FP32 destination scratch tiles/CBs when the vector-register budget requires it. Split setup/reader/compute/writer using inline `SOURCE_CODE` descriptors, and ensure every scratch reload bypasses SRCA/B. A one-tile multiply-add probe should precede a complete custom GEMM.

These are an implementable sequence and explicit correctness obligations, not a verified algorithm implementation. The midpoint normalization/rounding helper is required work; leaving it out would silently weaken the proposed exactness claim.

## Actual-prefix range issue: subnormals are present

The existing CPU matrix snapshot `v6_chunk64_state_cpu_setup_final.json.cpu_matrices.pt` was inspected read-only. In the length128 case, chunk0 has179 subnormal entries in `A_final`,611 in `transformed_k`, and627 in `weighted_k`; chunk1 has177,597, and513 respectively. The smallest nonzero entries are `1.401298464324817e-45`. Final states are normal-valued: minimum nonzero magnitudes are approximately `7.83e-10` and `8.49e-11` for chunks0 and1.

The source tests explicitly state that ordinary device FP32 arithmetic flushes subnormal results. Therefore a normal-only Veltkamp/TwoSum proof does **not** establish exact parity for all actual intermediates. Per-lane power-of-two normalization using bit/exponent operations is a feasible extension, but it must preserve the accumulator alignment and final underflow/tie behavior. Alternatively, prove by an explicit CPU counterfactual that flushing these particular terms changes neither the relevant FP32 states nor their BF16 cache codes; that would be finite-fixture evidence only, not a universal implementation contract.

## Recommended next discriminating experiment

Before implementing a full matmul, build one inline SFPU tile probe containing actual `(a,b,acc)` triples from the six captured GEMM stages, plus cancellation, midpoint, signed-zero, exponent-gap and subnormal cases. Compare raw FP32 codes to the CPU sequential-FMA oracle and to native `addcmul`. First validate split-product reconstruction and TwoSum separately, then the final tie-aware rounding. Preserve the source/compiled-kernel identity and characterize any unsupported range explicitly. Only after that passes should the corrected primitive replace GEMM updates in the chunk64 component.

A failure of this candidate would reject the tested implementation or range assumptions. It would not prove that exact device arithmetic is impossible, and it does not authorize changing the HF oracle, cache contract, PCC bar, or latency gate.
