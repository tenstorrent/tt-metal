## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

**Note**: This analysis was performed on a "deeply nuked" codebase where the upper dispatch layers (compute API header, Metal LLK dispatch, TTNN dispatch case) for hardtanh have been deleted. The core SFPU implementation (`ckernel_sfpu_hardtanh.h`) is intact in tt_llk for both Wormhole and Blackhole. The analysis below reconstructs the full expected call chain based on the surviving infrastructure and analogous operations (e.g., swish, frac, clamp).

### Unary Dispatch Summary
- **UnaryOpType**: `HARDTANH`
- **Compute kernel**: `eltwise_sfpu.cpp` (inferred from `get_compute_kernel_path()` default case, since HARDTANH has no explicit case)
- **SFPU_OP_CHAIN_0 expansion**: `hardtanh_tile(0, param0, param1)` (nuked from dispatch; reconstructed from Doxygen reference `hardtanh_tile.rst` showing signature `hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1)`)

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(HARDTANH)` in `unary_op_utils.cpp` -- falls through to `default: return false` (no explicit case for HARDTANH) |
| Template parameter (SFPU_OP_CHAIN) | Parameterized with 2 runtime `uint32_t` args (not a template bool) | `is_parametrized_type(HARDTANH)` returns `true`; the `get_op_init_and_func_parameterized` case for HARDTANH was nuked, but Doxygen shows `hardtanh_tile(idst, param0, param1)` |
| Effective SFPU path | `APPROXIMATION_MODE` is `false`; however, the `_calculate_hardtanh_` function does not branch on `APPROXIMATION_MODE` at all -- the template parameter is unused | The entire function body executes identically regardless of `APPROXIMATION_MODE` value |

### SFPU Abstraction Layers
The upper layers have been deleted in this deeply-nuked codebase. Paths are reconstructed based on the codebase conventions and analogous surviving operations.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/hardtanh.h` [DELETED -- nuked; reconstructed from pattern of `swish.h`, `frac.h`] |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_hardtanh.h` [DELETED -- nuked; reconstructed from pattern of `llk_math_eltwise_unary_sfpu_swish.h`] |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h` (identical file at `tt_llk_blackhole/...`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole) |

### Call Chain
The expected call chain (reconstructed for nuked layers, verified for surviving layers):

1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `hardtanh_tile(0, param0, param1)`, invoking the compute API function.

2. **API Header** (`hardtanh.h` -- nuked): `hardtanh_tile(idst, param0, param1)` calls `MATH((llk_math_eltwise_unary_sfpu_hardtanh<APPROX>(idst, (int)VectorMode::RC, param0, param1, param2)))` where `param2` is computed from `param0` and `param1` (the negated `pos_threshold`). The init function `hardtanh_tile_init()` calls `llk_math_eltwise_unary_sfpu_hardtanh_init<APPROX>()`.

3. **LLK Dispatch** (`llk_math_eltwise_unary_sfpu_hardtanh.h` -- nuked): Based on the pattern of similar 3-param ops, this would use `SFPU_UNARY_THREE_PARAM_KERNEL_FN(_calculate_hardtanh_, RC, APPROXIMATE, idst, param0, param1, param2)` which expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::_calculate_hardtanh_<APPROXIMATE>, idst, (int)VectorMode::RC, param0, param1, param2)`.

4. **Parameters Dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): The `_llk_math_eltwise_unary_sfpu_params_` function sets DEST write address, configures address mode, stalls until SFPU is ready, then iterates over 4 faces (in RC mode), calling the SFPU function once per face with the variadic args forwarded. Between faces, `TTI_SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole) advances the DEST pointer.

5. **Core SFPU Implementation** (`ckernel_sfpu_hardtanh.h`): The `_calculate_hardtanh_<APPROXIMATION_MODE, ITERATIONS>` function is called with `iterations` (default 8 per face), and the three `uint32_t` parameters. It processes 8 sfpi rows per face, applying the hardtanh clamping logic to each 32-element vector.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (Face 0, Face 1, Face 2, Face 3), covering all 1024 elements.
- **Operation invocation**: The core SFPU function `_calculate_hardtanh_` is called once per face (4 times total for RC mode). Each invocation runs 8 iterations (the default `ITERATIONS` template parameter), processing one sfpi row (32 elements) per iteration.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `TTI_SETRWC` between faces on Wormhole / `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` on Blackhole). The configured address mode is `ADDR_MOD_7` with `dest.incr = 0` (no auto-increment from the address mod -- the SFPI `dst_reg++` handles per-iteration advancement internally).

### Annotated SFPU Kernel Source

The kernel uses SFPI abstractions (`sfpi::vFloat`, `sfpi::dst_reg`, `v_if`/`v_endif`), so Style A (inline-commented source code) is used.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h
// (Identical in tt_llk_blackhole)

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_hardtanh_(const int iterations, std::uint32_t param0, std::uint32_t param1, std::uint32_t param2)
{ // APPROXIMATION_MODE=false (unused -- no branch on it), ITERATIONS=8
    // All params are in FP16_B format
    // param0 = -(neg_threshold)
    // param1 = -(pos_threshold - neg_threshold)
    // param2 = -(pos_threshold)

    sfpi::vFloat p0 = sfpi::s2vFloat16b(param0); // SFPLOADI: load -(neg_threshold) as FP16_B immediate into LREG
    sfpi::vFloat p1 = sfpi::s2vFloat16b(param1); // SFPLOADI: load -(pos_threshold - neg_threshold) as FP16_B immediate into LREG
    sfpi::vFloat p2 = sfpi::s2vFloat16b(param2); // SFPLOADI: load -(pos_threshold) as FP16_B immediate into LREG
// SFPU microcode
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++) // 8 iterations per face
    {
        sfpi::vFloat val = sfpi::dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG

        val += p0; // SFPMAD(val, 1.0, p0): val = val + (-(neg_threshold)) = val - neg_threshold
        v_if (val < 0.0f) // SFPSETCC with LT0: sets CC for lanes where (val - neg_threshold) < 0, i.e., val < neg_threshold
        {
            val = 0.0f; // SFPLOADI 0.0 (CC-guarded): for clamped lanes, set val to 0
        }
        v_endif; // SFPCOMPC: restore CC state

        val += p1; // SFPMAD(val, 1.0, p1): val = val + (-(pos_threshold - neg_threshold))
        // For unclamped lanes: val was (x - neg_threshold), now becomes (x - pos_threshold)
        // For clamped-low lanes: val was 0, now becomes -(pos_threshold - neg_threshold)
        v_if (val >= 0.0f) // SFPSETCC with GTE0: sets CC for lanes where val >= 0, i.e., x >= pos_threshold
        {
            val = 0.0f; // SFPLOADI 0.0 (CC-guarded): for clamped-high lanes, set val to 0
        }
        v_endif; // SFPCOMPC: restore CC state

        val += p2; // SFPMAD(val, 1.0, p2): val = val + (-(pos_threshold))
        // For unclamped: (x - pos_threshold) + (-pos_threshold) -- wait, let's trace carefully:
        // Unclamped: val = (x - neg_threshold) + (-(pos - neg)) = x - pos, then + (-pos) = x - 2*pos?
        // Actually: the clamping zeroes val THEN adds next param.
        // Unclamped path: val = x + p0 = x - neg; val += p1 = x - neg - (pos - neg) = x - pos; val += p2 = x - pos - pos
        // This doesn't look right for identity. Let me re-read the comments...
        // Re-analysis: p0 = -(neg_threshold), p1 = -(pos_threshold - neg_threshold), p2 = -(pos_threshold)
        // But the KEY is: when val < 0 after adding p0, val is SET TO 0 (not the original value).
        // Path analysis for the 3 regions:
        //
        // Case 1: x < neg_threshold
        //   val = x + p0 = x - neg_thresh < 0 => val = 0
        //   val = 0 + p1 = -(pos - neg) < 0 (assuming pos > neg)
        //   val += p2 = -(pos - neg) + (-pos) = -(2*pos - neg) ... hmm
        //
        // Actually the parameters encode the thresholds differently for the arithmetic to work.
        // The encoding is designed so that:
        //   - x < neg: val=0 after first clamp, then val=p1 (negative), stays negative in second clamp, then val=p1+p2
        //     With p1=-(pos-neg), p2=-pos: p1+p2 = -(pos-neg)-pos = -(2pos-neg) = neg - 2pos ... that gives neg_threshold if 2pos = 0?
        //     For default neg=-1, pos=1: p0=1, p1=-(1-(-1))=-2, p2=-1
        //     x < -1: val = x+1 < 0 => val=0; val = 0+(-2) = -2 < 0 (stays); val = -2+(-1) = -3? Should be -1.
        //
        // Let me re-read: param0 = -(neg_threshold). For neg_threshold = -1: param0 = -(-1) = 1
        // param1 = -(pos_threshold - neg_threshold). For pos=1, neg=-1: param1 = -(1-(-1)) = -2
        // param2 = -(pos_threshold). For pos=1: param2 = -1
        //
        // Case 1 x=-5 (should clamp to -1):
        //   val = -5 + 1 = -4 < 0 => val = 0
        //   val = 0 + (-2) = -2 < 0, so val >= 0 is FALSE => stays -2
        //   val = -2 + (-1) = -3 ... not -1
        //
        // Something is off. Let me re-read the source more carefully.
        // The function comment says "param0 = -(neg_threshold)". That is the ENCODING.
        // So param0 is passed from the host as the bit pattern for -(neg_threshold) in FP16_B.
        //
        // Wait - I need to reconsider. These are uint32_t params that get interpreted as FP16_B.
        // The CALLER must pre-compute these values. Let me check how hardtanh was called in the
        // original (non-nuked) dispatch path. The Doxygen says hardtanh_tile(idst, param0, param1).
        // So the API takes 2 user params, and the third is computed internally.
        //
        // The actual parameter encoding must be:
        //   User provides: low (neg_threshold), high (pos_threshold)
        //   Kernel receives:
        //     param0 = bit_cast<uint32_t>(-low) = -(neg_threshold)
        //     param1 = bit_cast<uint32_t>(-(high - low)) = -(pos - neg)
        //     param2 = bit_cast<uint32_t>(-high) = -(pos_threshold)
        //
        // With default low=-1, high=1:
        //   param0 = 1.0, param1 = -2.0, param2 = -1.0
        //
        // For x = -5 (clamp to low = -1):
        //   val = -5 + 1 = -4 < 0 => val = 0
        //   val = 0 + (-2) = -2, which is < 0, so second v_if (>= 0) is false
        //   val = -2 + (-1) = -3
        //   Result: -3. Expected: -1. WRONG.
        //
        // Hmm, let me reconsider the parameter encoding. Perhaps the comments mean something
        // different. Looking at clamp for comparison:
        //   clamp uses: val < min => val = min; val >= max => val = max; then dst = val + offset
        //   That's a straightforward clamping.
        //
        // hardtanh uses a DIFFERENT approach -- it encodes the clamping as additive offsets.
        // Let me think about this differently.
        //
        // Actually, re-reading the kernel VERY carefully:
        // The encoding is: p0 = -low, p1 = -(high-low), p2 = -high
        // But note that the intermediate "val" after zeroing is NOT the final output.
        // The trick is that it does TWO clamps with zeroing, then adds back.
        //
        // Let me trace all 3 cases correctly with default low=-1, high=1:
        // p0=1, p1=-2, p2=-1
        //
        // Case A: x in [-1, 1] (unclamped, should output x):
        //   val = x + 1, which is >= 0 (since x >= -1), first v_if false
        //   val = (x+1) + (-2) = x - 1
        //   Since x <= 1, x - 1 <= 0, so val < 0, second v_if (>= 0) false
        //   val = (x-1) + (-1) = x - 2. Expected x. WRONG AGAIN.
        //
        // I must be misunderstanding. Let me re-read the kernel ONE MORE TIME.
        // OH WAIT -- I see: the default ITERATIONS for _calculate_hardtanh_ is a template
        // param, but the actual iteration count is passed as the RUNTIME `iterations` arg.
        // That's not the issue though.
        //
        // Let me look at this from an information-theoretic standpoint. The kernel takes
        // 3 FP16_B params. The output should be clamp(x, low, high).
        //
        // Re-examination: Maybe p2 is NOT -pos_threshold. Let me consider:
        //   p0 = -low, p1 = -(high-low), p2 = low (not -high)
        //
        // x = 0 (in range):
        //   val = 0 + (-low) = -(-1) = 1 >= 0, first v_if false
        //   val = 1 + (-(1-(-1))) = 1 + (-2) = -1 < 0, second v_if false
        //   val = -1 + low = -1 + (-1) = -2. Nope.
        //
        // Let me try: p0=-low, p1=low-high, p2=high
        //   With low=-1, high=1: p0=1, p1=-2, p2=1
        //   x=0: val=0+1=1 >=0 => skip. val=1+(-2)=-1 <0 => skip. val=-1+1=0. Correct!
        //   x=-5: val=-5+1=-4 <0 => val=0. val=0+(-2)=-2 <0 => skip. val=-2+1=-1. Correct!
        //   x=5: val=5+1=6 >=0 => skip. val=6+(-2)=4 >=0 => val=0. val=0+1=1. Correct!
        //
        // SO THE ACTUAL ENCODING IS:
        //   param0 = -low (= -(neg_threshold))
        //   param1 = low - high (= neg_threshold - pos_threshold = -(pos-neg))
        //   param2 = high (= pos_threshold)... but the comment says -(pos_threshold)
        //
        // The comment in the source says: param2 = -(pos_threshold)
        // With default pos=1, that means param2 = -1.
        //   x=-5: val=0+(-2)=-2 <0 => skip. val=-2+(-1)=-3. Wrong.
        //
        // So EITHER the comment is wrong OR my understanding is wrong.
        // Let me try the interpretation where the comments are correct AND see what math works:
        //   param0 = -neg_thresh, param1 = -(pos - neg), param2 = -pos
        //   With neg = -1, pos = 1: param0 = 1, param1 = -2, param2 = -1
        //
        // For the kernel to produce correct results, there must be something I'm missing.
        // Wait -- perhaps "neg_threshold" in the comment doesn't mean -1.
        // In PyTorch, hardtanh(x, min_val=-1, max_val=1).
        // Perhaps the comments use "neg_threshold" to mean |min_val| = 1, and
        // "pos_threshold" = max_val = 1.
        // So: param0 = -(1) = -1, param1 = -(1-1) = 0, param2 = -(1) = -1
        //   x=0: val=0+(-1)=-1<0 => val=0. val=0+0=0 >=0 => val=0. val=0+(-1)=-1. Expected 0. Wrong.
        //
        // That doesn't work either. Let me just accept the code as-is and document
        // the ALGORITHM the kernel implements, noting the parameter encoding comment
        // may use non-standard naming.
        //
        // THE CORRECT interpretation that makes the math work is:
        //   param0 = -low, param1 = -(high - low), param2 = high
        //   (where low and high are the clamp bounds)
        // This can be verified: for default hardtanh (low=-1, high=1):
        //   param0 = 1, param1 = -2, param2 = 1
        //   x=0: 0+1=1>=0,skip; 1-2=-1<0,skip; -1+1=0 CORRECT
        //   x=-5: -5+1=-4<0,val=0; 0-2=-2<0,skip; -2+1=-1 CORRECT
        //   x=5: 5+1=6>=0,skip; 6-2=4>=0,val=0; 0+1=1 CORRECT

        sfpi::dst_reg[0] = val; // SFPSTORE: write result back to DEST

        sfpi::dst_reg++; // Advance DEST pointer by 1 sfpi row (2 physical rows, 32 elements)
    }
}
```

**Algorithm explanation**: The hardtanh kernel implements `clamp(x, low, high)` using an additive offset technique that avoids explicit comparisons against the threshold values. Instead, it offsets the value to bring the comparison point to zero, leveraging the SFPU's efficient zero-comparison condition codes (`LT0`, `GTE0`). The technique works in three stages:

1. **Lower clamp check**: Add `-low` to shift the lower bound to zero. If `val + (-low) < 0`, then `x < low`, so zero out val (the "excess below low" is discarded).

2. **Upper clamp check**: Add `-(high - low)` to shift the upper bound to zero (relative to the already-shifted value). If `val >= 0` after this addition, then `x > high`, so zero out val.

3. **Restore**: Add `high` to translate the result back to the correct output range:
   - **Unclamped** (`low <= x <= high`): val ends up as `(x - low) - (high - low) + high = x - high + high = x` -- wait, let me re-derive:
     After step 1: `val = x - low` (not zeroed since x >= low)
     After step 2: `val = (x - low) - (high - low) = x - high` (not zeroed since x <= high means x - high <= 0)
     After step 3: `val = (x - high) + high = x`. Correct.
   - **Clamped low** (`x < low`): val zeroed in step 1, becomes `0 - (high - low) = low - high` after step 2 (negative, not zeroed), then `(low - high) + high = low`. Correct.
   - **Clamped high** (`x > high`): val passes step 1, becomes `x - high > 0` after step 2 so zeroed, then `0 + high = high`. Correct.

**Parameter encoding (corrected from source comments)**:
- `param0` = `-low` (negated lower bound, in FP16_B)
- `param1` = `-(high - low)` (negated range width, in FP16_B)
- `param2` = `high` (upper bound, in FP16_B)

**Note on source comments**: The source file comments state `param2 = -(pos_threshold)`, but the algorithm only produces correct results if `param2 = pos_threshold` (i.e., the positive upper bound without negation). Either the comment uses a non-standard sign convention (where "pos_threshold" is already negative of the actual bound), or the comment is simply inaccurate. The algorithmic analysis above is authoritative.

### SFPU Instructions Used

| Instruction | SFPI Abstraction | Description |
|-------------|-----------------|-------------|
| `SFPLOADI` | `sfpi::s2vFloat16b(param)` | Load a scalar immediate (FP16_B format) into an LREG, broadcasting to all 32 SFPU lanes. Used to load the three threshold parameters `p0`, `p1`, `p2`. Also used for the `0.0f` constant in conditional assignments. |
| `SFPLOAD` | `sfpi::dst_reg[0]` (read) | Load 32 elements (2 physical DEST rows) from the current DEST address into an LREG. One load per iteration. |
| `SFPSTORE` | `sfpi::dst_reg[0] = val` (write) | Store 32 elements from an LREG back to the current DEST address. One store per iteration. |
| `SFPMAD` | `val += pN` (vFloat addition) | Multiply-add: computes `a * 1.0 + b` to implement floating-point addition. There is no dedicated float-add instruction on the SFPU; all vFloat additions are emitted as SFPMAD with a multiply-by-one. Three SFPMAD instructions per iteration (one for each parameter addition). |
| `SFPSETCC` | `v_if (val < 0.0f)` / `v_if (val >= 0.0f)` | Set per-lane condition codes based on comparison against zero. The `< 0.0f` test sets CC for lanes where the value is negative (LT0 mode). The `>= 0.0f` test sets CC for lanes where the value is non-negative (GTE0 mode). Two SFPSETCC instructions per iteration. |
| `SFPCOMPC` | `v_endif` | Complement/restore the condition code state, ending a predicated execution block. Two SFPCOMPC instructions per iteration (one per `v_endif`). |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST rows** | Input/output tile data. Each iteration reads 2 physical DEST rows (32 elements) via `dst_reg[0]`, processes them, and writes back. 8 iterations per face, 4 faces per tile = 32 iterations total, covering all 64 physical DEST rows (1024 elements). |
| **LREG 0-3** (approximate) | Used by the SFPI compiler for `val`, `p0`, `p1`, `p2`. The three parameter values (`p0`, `p1`, `p2`) are loaded once before the loop and held in LREGs across all iterations. The working value `val` uses an additional LREG. The SFPI compiler manages LREG allocation; the exact register numbers depend on compiler decisions, but 4 LREGs are needed simultaneously (3 constants + 1 working value). |
| **Condition Code (CC)** | Per-lane predication bits. Used by `v_if`/`v_endif` blocks to conditionally zero out lanes. Two independent CC regions per iteration: one for the lower clamp (LT0) and one for the upper clamp (GTE0). CC state is fully restored after each `v_endif`. |

### Address Mode Configuration

The address mode is configured by `eltwise_unary_sfpu_configure_addrmod<SfpuType::hardtanh>()` during init. Since `SfpuType::hardtanh` does not match any of the special-cased `if constexpr` branches (topk_local_sort, typecast, unary_max/min, signbit, or reciprocal on Blackhole), only the default `ADDR_MOD_7` is set:

**Wormhole B0 and Blackhole** (identical for hardtanh):
```
ADDR_MOD_7:
  srca.incr = 0
  srcb.incr = 0
  dest.incr = 0
```

The `dest.incr = 0` means the address mode does not auto-increment the DEST pointer. Instead, per-iteration DEST advancement is handled by the SFPI `dst_reg++` construct (which the compiler emits as an internal RWC increment), and per-face advancement is handled by:
- **Wormhole**: Two `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` instructions between faces (in `_llk_math_eltwise_unary_sfpu_params_`), each advancing DEST by 8 rows, totaling 16 physical rows = 1 face.
- **Blackhole**: `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice, achieving the same 16-row advance.

## Local Knowledge Sources
### Local References
1. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Core SFPU implementation of hardtanh -- the primary analysis target.
   **Key Findings**: Uses SFPI abstractions (vFloat, dst_reg, v_if/v_endif). Takes 3 FP16_B parameters encoding negated threshold values. Implements clamping via additive offset technique with zero-comparison. APPROXIMATION_MODE template parameter is unused.

2. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_hardtanh.h`
   **Reason**: Checked for hardware-specific differences between Wormhole and Blackhole.
   **Key Findings**: Identical to the Wormhole version.

3. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Parameters dispatch layer -- routes SFPU function calls across tile faces.
   **Key Findings**: Implements VectorMode::RC (4-face), R (2-face), C (2-face) dispatch. Uses TTI_SETRWC for inter-face DEST advancement on Wormhole.

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h`
   **Reason**: Blackhole variant of parameters dispatch.
   **Key Findings**: Uses `_llk_math_eltwise_unary_sfpu_start_`/`_done_` and `_inc_dst_face_addr_` instead of direct TTI_SETRWC/TTI_STALLWAIT.

5. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Init and address mode configuration for SFPU unary ops.
   **Key Findings**: `eltwise_unary_sfpu_configure_addrmod` sets ADDR_MOD_7 with dest.incr=0 for hardtanh (no special case). Init calls `_init_sfpu_config_reg()` and `reset_counters`.

6. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h`
   **Reason**: Blackhole variant of init.
   **Key Findings**: Same ADDR_MOD_7 configuration as Wormhole. Minor difference: Blackhole additionally special-cases `SfpuType::reciprocal` for ADDR_MOD_6.

7. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.cpp`
   **Reason**: Dispatch configuration -- approx mode, compute kernel path, init/func definitions.
   **Key Findings**: HARDTANH falls through to defaults: approx_mode=false, compute_kernel=eltwise_sfpu.cpp. Parameterized dispatch case was nuked.

8. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/common/unary_op_utils.hpp`
   **Reason**: Type classification and parametrization check.
   **Key Findings**: `is_parametrized_type(HARDTANH)` returns true, confirming hardtanh requires runtime parameters.

9. **File**: `DEEP_NUKE_MANIFEST.md`
   **Reason**: Understand what was deleted in the deep nuke.
   **Key Findings**: hardtanh dispatch removed, compute API deleted, Metal ckernel deleted, Metal LLK deleted. tt_llk primitives (ckernel_sfpu_hardtanh.h) survive.

10. **File**: `docs/source/tt-metalium/tt_metal/apis/kernel_apis/compute/hardtanh_tile.rst`
    **Reason**: Doxygen documentation reference for the compute API signature.
    **Key Findings**: `hardtanh_tile(uint32_t idst, uint32_t param0, uint32_t param1)` -- takes 2 user-facing parameters (the third is computed internally by the LLK dispatch layer).

11. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_clamp.h`
    **Reason**: Closely related operation (clamp) for comparison and understanding parameter patterns.
    **Key Findings**: Clamp uses a simpler approach (direct comparison with min/max via v_if/v_elseif), while hardtanh uses the additive offset technique. Both take 3 uint32_t parameters.

12. **File**: `runtime/sfpi/include/sfpi_fp16.h`
    **Reason**: Understanding the s2vFloat16b scalar-to-vector conversion used in the kernel.
    **Key Findings**: s2vFloat16b constructs an FP16_B immediate from a uint32_t (raw pass-through) or float (conversion). Used with SFPLOADI to broadcast scalar constants to all SFPU lanes.

13. **File**: `.claude/references/sfpu-hardware-model.md`
    **Reason**: Authoritative reference for SFPU addressing model, tile geometry, and instruction semantics.
    **Key Findings**: Confirmed stride-2 addressing (SFP_DESTREG_STRIDE=2), 8 iterations per face, 32 elements per sfpi row, SFPMAD for all float additions.

14. **File**: `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h`
    **Reason**: Macro library for SFPU kernel dispatch patterns.
    **Key Findings**: `SFPU_UNARY_THREE_PARAM_KERNEL_FN` macro matches the expected hardtanh dispatch pattern (function, mode, approximate, dst_idx, param0, param1, param2).

15. **File**: `docs/sfpu_operations/key_notes/hardtanh_key_notes.md`
    **Reason**: Operation semantics reference.
    **Key Findings**: Formula is clamp(x, min_val, max_val) with defaults min=-1, max=1. Deterministic, mode-independent.
