## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: N/A -- `rand` is not a standard `UnaryOpType`. It is a standalone compute API (`rand_tile` / `rand_tile_init`) exposed via `tt_metal/hw/inc/api/compute/eltwise_unary/rand.h`. It is used by custom compute kernels such as `compute_uniform.cpp` (ttnn uniform op) and `sampling.cpp` (ttnn sampling op), rather than the standard `eltwise_sfpu.cpp` dispatch.
- **Compute kernel**: Custom (e.g., `ttnn/cpp/ttnn/operations/uniform/device/kernels/compute_uniform.cpp`)
- **SFPU_OP_CHAIN_0 expansion**: N/A -- `rand_tile(idst, from, scale)` is called directly from the compute kernel, not via `SFPU_OP_CHAIN_0`.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | N/A (not routed through `get_op_approx_mode`) | `rand` is not a `UnaryOpType`, so `get_op_approx_mode()` is never consulted. The `APPROX` template parameter is set by the macro `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(rand, RC, APPROX, ...)` which uses the compile-time `APPROX` define. |
| Template parameter (SFPU_OP_CHAIN) | `APPROX` (compile-time define, typically `true`) | The `APPROX` symbol is a compile-time define injected by the program factory. In `rand.h`: `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(rand, RC, APPROX, idst, from, scale)` passes `APPROX` directly. |
| Effective SFPU path | The `APPROXIMATION_MODE` template parameter is accepted but **unused** by `ckernel::sfpu::rand<APPROXIMATION_MODE>` -- the kernel has no `if constexpr` branches on this parameter. Both approximate and exact modes execute identical code. | `ckernel_sfpu_rand.h` -- the function body contains no conditionals on `APPROXIMATION_MODE`. |

### SFPU Abstraction Layers
List the file path for each abstraction layer.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/rand.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist -- `rand_tile()` invokes `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` macro directly, which expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::rand<APPROXIMATE>, idst, (int)VectorMode::RC, from, scale)` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h` (Wormhole) / `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h` (Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole) |
| **Init Dispatch** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_init.h` (shared init entry point) -> `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h` (Wormhole `_llk_math_eltwise_unary_sfpu_init_`) |

### Call Chain
1. **`rand_tile(idst, from, scale)`** (in `rand.h`) -- the API entry point. Calls `MATH(SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(rand, RC, APPROX, idst, from, scale))`, which is active only on the MATH RISC-V thread.
2. **`SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` macro** (in `llk_math_eltwise_unary_sfpu_macros.h`) -- expands to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::rand<APPROXIMATE>, idst, (int)VectorMode::RC, from, scale)`.
3. **`_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>`** (in `llk_math_eltwise_unary_sfpu_params.h`) -- the params dispatch function. Sets DEST write address, stalls for SFPU availability, then loops over 4 faces (VectorMode::RC), calling `ckernel::sfpu::rand<APPROXIMATE>(from, scale)` for each face with SETRWC/inc_dst_addr between faces.
4. **`ckernel::sfpu::rand<APPROXIMATE>(from, scale)`** (in `ckernel_sfpu_rand.h`) -- the core SFPU kernel. Loads parameters, generates random floats via PRNG, normalizes, scales, and stores to DEST.

For initialization:
1. **`rand_tile_init(seed)`** (in `rand.h`) -- calls `MATH(SFPU_ONE_PARAM_KERNEL_INIT(unused, sfpu::rand_init, APPROX, seed))`.
2. **`SFPU_ONE_PARAM_KERNEL_INIT` macro** -- expands to `llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::rand_init<APPROXIMATE>, seed)`.
3. **`llk_math_eltwise_unary_sfpu_init`** (in `llk_math_eltwise_unary_sfpu_init.h`) -- calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>()` to configure SFPU config reg + addr_mod + reset counters, then calls `rand_init<APPROXIMATE>(seed)`.
4. **`rand_init<APPROXIMATE>(seed)`** (in `ckernel_sfpu_rand.h`) -- calls `init_prng_seed(seed)`.
5. **`init_prng_seed(seed)`** (in `ckernel.h`) -- writes `seed` to `PRNG_SEED_Seed_Val_ADDR32` config register, then executes 600 `TTI_SFPNOP` instructions to allow the PRNG state to settle.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed. Each face is 16x16 = 256 elements.
- **Operation invocation**: The params dispatch function (`_llk_math_eltwise_unary_sfpu_params_`) loops `for (int face = 0; face < 4; face++)`, calling `ckernel::sfpu::rand<APPROXIMATE>(from, scale)` once per face. Inside `rand()`, there is a loop `for (int d = 0; d < 8; d++)` processing 8 iterations per face. Each iteration processes 32 elements (2 physical DEST rows x 16 elements/row), so 8 iterations x 32 elements = 256 elements = 1 full face.
- **DEST address progression**: On Wormhole, `ADDR_MOD_7` is configured with `dest.incr = 0` (no auto-increment from the address mode itself). The SFPU kernel uses `dst_reg++` within the inner loop (8 iterations) to advance the DEST write pointer by 1 sfpi row (= 2 physical DEST rows = 32 elements) per iteration. Between faces, on Wormhole: `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` x2 advances by 16 physical rows (1 face); on Blackhole: `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice to achieve the same.

### Annotated SFPU Kernel Source

This kernel uses **raw TTI instructions** (Style B candidate), but has **no condition code manipulation** -- there are no `SFPSETCC`, `SFPENCC`, or CC-modifying instructions. Therefore **Style A** (inline-commented source code) is appropriate.

#### Wormhole B0

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h

template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE is accepted but unused
inline void rand_init(uint32_t seed) {
    init_prng_seed(seed); // Writes seed to PRNG config register, then 600x SFPNOP to settle PRNG state
}

template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE is accepted but unused
inline void rand(uint32_t from, uint32_t scale) {
    // Load scale param (float32, bit-cast as uint32) into LREG1 via two half-word loads
    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);  // instr_mod0=10: load lower 16 bits of LREG1
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);       // instr_mod0=8: load upper 16 bits of LREG1

    // Load from param (float32, bit-cast as uint32) into LREG2 via two half-word loads
    TT_SFPLOADI(p_sfpu::LREG2, 10, from & 0xFFFF);    // instr_mod0=10: load lower 16 bits of LREG2
    TT_SFPLOADI(p_sfpu::LREG2, 8, from >> 16);         // instr_mod0=8: load upper 16 bits of LREG2

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {                       // 8 iterations per face (ITERATIONS=8)
        // Generate pseudorandom uint32 via hardware PRNG
        TTI_SFPMOV(0, 9, p_sfpu::LREG0, 8);            // lreg_c=9(LCONST_0), instr_mod1=8: PRNG mode -> LREG0

        // Normalize random bits to float in [1.0, 2.0):
        // Clear sign bit (force positive)
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG0, 1); // instr_mod1=1: set sign from imm12=0 -> sign=0
        // Set exponent to 127 (IEEE754 bias), mantissa unchanged -> value in [1.0, 2.0)
        TTI_SFPSETEXP(127, p_sfpu::LREG0, p_sfpu::LREG0, 1); // instr_mod1=1: set exp from imm12=127

        // Subtract 1.0 to shift range from [1.0, 2.0) to [0.0, 1.0)
        TTI_SFPADDI(0xbf80 /*-1.0f in bfloat16*/, p_sfpu::LREG0, 0); // LREG0 = LREG0 + (-1.0)
        TTI_SFPNOP;                                      // Pipeline stall required on Wormhole after SFPADDI

        // Scale from [0,1) to [from, from+scale): result = LREG0 * LREG1 + LREG2
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0); // LREG0 = LREG0*scale + from
        TTI_SFPNOP;                                      // Pipeline stall required on Wormhole after SFPMAD

        TTI_SFPSTORE(0, 3, 3, 0);                        // Store LREG0 to DEST (lreg=0, instr_mod0=3/FP32, addr_mode=3, dest_addr=0)
        dst_reg++;                                        // Advance DEST pointer by 1 sfpi row (32 elements)
    }
}
```

#### Blackhole

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h

template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE is accepted but unused
inline void rand_init(uint32_t seed) {
    init_prng_seed(seed); // Writes seed to PRNG config register, then 600x SFPNOP to settle PRNG state
}

template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE is accepted but unused
inline void rand(uint32_t from, uint32_t scale) {
    // Load scale param (float32, bit-cast as uint32) into LREG1 via two half-word loads
    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);  // instr_mod0=10: load lower 16 bits of LREG1
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);       // instr_mod0=8: load upper 16 bits of LREG1

    // Load from param (float32, bit-cast as uint32) into LREG2 via two half-word loads
    TT_SFPLOADI(p_sfpu::LREG2, 10, from & 0xFFFF);    // instr_mod0=10: load lower 16 bits of LREG2
    TT_SFPLOADI(p_sfpu::LREG2, 8, from >> 16);         // instr_mod0=8: load upper 16 bits of LREG2

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {                       // 8 iterations per face (ITERATIONS=8)
        // Generate pseudorandom uint32 via hardware PRNG
        TTI_SFPMOV(0, 9, p_sfpu::LREG0, 8);            // lreg_c=9(LCONST_0), instr_mod1=8: PRNG mode -> LREG0

        // Normalize random bits to float in [1.0, 2.0):
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG0, 1); // Clear sign bit (force positive)
        TTI_SFPSETEXP(127, p_sfpu::LREG0, p_sfpu::LREG0, 1); // Set exponent=127, mantissa unchanged -> [1.0, 2.0)

        // Subtract 1.0 to shift range from [1.0, 2.0) to [0.0, 1.0)
        TTI_SFPADDI(0xbf80 /*-1.0f in bfloat16*/, p_sfpu::LREG0, 0); // LREG0 = LREG0 + (-1.0)
        // No SFPNOP needed on Blackhole (deeper pipeline / hazard resolution)

        // Scale from [0,1) to [from, from+scale): result = LREG0 * LREG1 + LREG2
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0); // LREG0 = LREG0*scale + from
        // No SFPNOP needed on Blackhole

        TTI_SFPSTORE(p_sfpu::LREG0, FP32, ADDR_MOD_7, 0); // Store LREG0 to DEST (FP32 format, ADDR_MOD_7, dest_addr=0)
        dst_reg++;                                        // Advance DEST pointer by 1 sfpi row (32 elements)
    }
}
```

### SFPU Instructions Used

| Instruction | Opcode | Description |
|-------------|--------|-------------|
| **SFPLOADI** | 0x71 | Load a 16-bit immediate value into an LREG. With `instr_mod0=10`, writes the lower 16 bits; with `instr_mod0=8`, writes the upper 16 bits. Used in pairs to load full 32-bit float parameters (`scale` and `from`) into LREG1 and LREG2. |
| **SFPMOV** (PRNG mode) | 0x7C | Normally moves data between LREGs. When `instr_mod1=8` and `lreg_c=9`, it activates the hardware PRNG and writes a pseudorandom 32-bit integer into the destination LREG. The PRNG is a hardware unit seeded via `PRNG_SEED_Seed_Val_ADDR32` config register. |
| **SFPSETSGN** | 0x84 variant | Sets the sign bit of a floating-point value. With `instr_mod1=1` and `imm12=0`, clears the sign bit (forces the value positive). Used to ensure the random value has a positive sign. |
| **SFPSETEXP** | 0x82 | Sets the exponent field of a floating-point value. With `instr_mod1=1` and `imm12=127`, overwrites the exponent with 127 (IEEE754 bias for 2^0), keeping the random mantissa. This normalizes the value to the range [1.0, 2.0). |
| **SFPADDI** | 0x75 | Adds a 16-bit immediate (bfloat16 format) to an LREG. `0xbf80` is bfloat16 for -1.0f. This shifts the range from [1.0, 2.0) to [0.0, 1.0). Two-cycle instruction on Wormhole (requires subsequent SFPNOP). |
| **SFPMAD** | 0x84 | Fused multiply-add: `LREG_dest = LREG_a * LREG_b + LREG_c`. Computes `LREG0 = LREG0 * LREG1(scale) + LREG2(from)`, scaling the [0,1) value to [from, from+scale). Two-cycle instruction on Wormhole (requires subsequent SFPNOP). |
| **SFPNOP** | 0x8F | No-operation. Used on Wormhole as a pipeline stall after multi-cycle instructions (SFPADDI, SFPMAD). Not needed on Blackhole. |
| **SFPSTORE** | 0x72 | Stores an LREG value back to a DEST register row. On Wormhole: `TTI_SFPSTORE(0, 3, 3, 0)` -- LREG0, FP32 format (instr_mod0=3), addr_mode=3, dest_addr=0. On Blackhole: `TTI_SFPSTORE(LREG0, FP32, ADDR_MOD_7, 0)` -- uses named constants. |

### SFPU Register Usage

| Register | Role | Lifetime |
|----------|------|----------|
| **LREG0** | Working register: receives PRNG output, undergoes sign/exp manipulation, subtraction, and MAD. Final result stored to DEST from this register. | Per-iteration: written by SFPMOV, modified in-place, stored by SFPSTORE. |
| **LREG1** | Holds the `scale` parameter (float32, bit-cast). Loaded once per face invocation via two SFPLOADI instructions. | Entire face (8 iterations). Read-only during the inner loop. |
| **LREG2** | Holds the `from` parameter (float32, bit-cast). Loaded once per face invocation via two SFPLOADI instructions. | Entire face (8 iterations). Read-only during the inner loop. |
| **PRNG config register** (`PRNG_SEED_Seed_Val_ADDR32`) | Hardware PRNG state. Written once during `rand_init()` with the seed value. The PRNG auto-advances on each `SFPMOV(PRNG)` invocation. | Persistent across all tiles until re-seeded. |
| **DEST rows** | Output destination. Each `SFPSTORE` writes to the current DEST row, and `dst_reg++` advances the write pointer. | Written sequentially; 8 rows per face, 32 rows per tile. |

### Address Mode Configuration

The address mode `ADDR_MOD_7` is configured during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>()` (called from `rand_tile_init`). The configuration is the **same on both Wormhole and Blackhole**:

```cpp
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
    .set(ADDR_MOD_7);
```

This means `ADDR_MOD_7` has **zero auto-increment** for all register domains. The DEST address progression relies entirely on the explicit `dst_reg++` in the kernel loop and the SETRWC/inc_dst_addr calls between faces in the params dispatch function.

Note: The `SfpuType::unused` passed to the init function means the default ADDR_MOD_7 configuration is used (no special cases like `topk_local_sort` or `typecast` that configure `ADDR_MOD_6`).

**Wormhole-specific**: The Wormhole params dispatch (`_llk_math_eltwise_unary_sfpu_params_`) also calls `math::set_addr_mod_base()` before the face loop and `math::clear_addr_mod_base()` after. Between faces, it uses `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` twice (advancing DEST by 8+8=16 physical rows = 1 face).

**Blackhole-specific**: The Blackhole params dispatch uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` between faces, which calls `math::inc_dst_addr<8>()` twice to achieve the same 16-row face stride.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the SFPMOV instruction with instr_mod1=8 generate pseudorandom numbers? What is the SFPU PRNG mechanism?"
   **Reason**: Needed to understand the PRNG generation mechanism in SFPMOV
   **Key Findings**: DeepWiki was unavailable (HTTP 429 Too Many Requests). Understanding was derived from source code analysis: the dropout kernel's comment in `ckernel_sfpu_dropout.h` explicitly documents that `SFPMOV` with `instr_mod1=8` and `lreg_c=9` generates a pseudorandom uint32, and `init_prng_seed()` configures the hardware PRNG state via the `PRNG_SEED_Seed_Val_ADDR32` config register.

### Confluence References
No Confluence pages were consulted (DeepWiki and source code analysis provided sufficient understanding).

### Glean References
No Glean searches were performed.
