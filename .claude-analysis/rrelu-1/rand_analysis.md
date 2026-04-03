## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: N/A -- `rand` is NOT a standard `UnaryOpType`. It is a standalone compute API (`rand_tile` / `rand_tile_init`) invoked directly by operation-specific compute kernels (e.g., `compute_uniform.cpp` for `ttnn::rand`, `compute_bernoulli.cpp` for `ttnn::bernoulli`).
- **Compute kernel**: `ttnn/cpp/ttnn/operations/rand/device/kernels/compute_uniform.cpp` (for `ttnn::rand`)
- **SFPU_OP_CHAIN_0 expansion**: N/A -- not dispatched via `SFPU_OP_CHAIN_0`. The compute kernel directly calls `rand_tile(0, from, scale)`.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. You MUST report both.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `true` | `rand_program_factory.cpp:86` sets `.math_approx_mode = true` |
| Template parameter (SFPU_OP_CHAIN) | N/A -- not parameterized | `rand_tile` uses `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(rand, RC, APPROX, ...)` where `APPROX` is the JIT-generated `constexpr bool` from `math_approx_mode` |
| Effective SFPU path | `APPROXIMATION_MODE=true`, but the `rand` kernel does NOT branch on `APPROXIMATION_MODE` -- the template parameter is accepted but unused. Both `true` and `false` execute identical code paths. | `ckernel_sfpu_rand.h` -- no `if constexpr (APPROXIMATION_MODE)` branch exists |

### SFPU Abstraction Layers
List the file path for each abstraction layer.

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/rand.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist -- the API header uses the `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` macro which directly expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::rand<APPROX>, ...)` |
| **Core SFPU Implementation (WH)** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h` |
| **Core SFPU Implementation (BH)** | `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h` |
| **Parameters Dispatch (WH)** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |
| **Parameters Dispatch (BH)** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |
| **Init / ADDR_MOD Config (WH)** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h` |
| **Init / ADDR_MOD Config (BH)** | `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu.h` |

### Call Chain
1. **Compute kernel** (`compute_uniform.cpp`) calls `rand_tile_init(seed)` then `rand_tile(0, from, scale)`.
2. **`rand_tile_init(seed)`** (in `rand.h`) expands via `SFPU_ONE_PARAM_KERNEL_INIT(unused, sfpu::rand_init, APPROX, seed)` to `llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROX>(rand_init<APPROX>, seed)`. This calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>()` (configures ADDR_MOD_7, resets counters) then invokes `rand_init<APPROX>(seed)` which calls `init_prng_seed(seed)` to seed the hardware PRNG.
3. **`rand_tile(idst, from, scale)`** (in `rand.h`) expands via `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(rand, RC, APPROX, idst, from, scale)` to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::rand<APPROX>, idst, (int)VectorMode::RC, from, scale)`.
4. **`_llk_math_eltwise_unary_sfpu_params_`** (in `llk_math_eltwise_unary_sfpu_params.h`) sets the DEST write address, stalls for SFPU availability, then iterates over 4 faces (VectorMode::RC), calling `ckernel::sfpu::rand<APPROX>(from, scale)` once per face, with `SETRWC` (WH) or `inc_dst_face_addr` (BH) between faces.
5. **`ckernel::sfpu::rand<APPROX>(from, scale)`** (in `ckernel_sfpu_rand.h`) is the core SFPU implementation that loads parameters, generates random numbers, shapes them to [from, from+scale), and stores results to DEST.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (4 faces x 8 iterations per face = 32 iterations = full 1024-element tile).
- **Operation invocation**: The params dispatch calls `ckernel::sfpu::rand<APPROX>(from, scale)` once per face. Each invocation runs an internal loop of 8 iterations (hardcoded `d < 8`), processing one face (256 elements).
- **DEST address progression**: Standard DEST progression. On **Wormhole**, init configures `ADDR_MOD_7` with `.dest = {.incr = 0}` (no auto-increment from addr_mod -- the kernel uses `dst_reg++` for per-iteration advancement); between faces, `TTI_SETRWC(CR_D, 8) x2` advances by 16 physical rows (= 1 face). On **Blackhole**, the same `ADDR_MOD_7` is configured; between faces, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice (= 16 physical rows). Within each face, `dst_reg++` advances 1 sfpi row (= 2 physical DEST rows) per iteration for 8 iterations.

### Annotated SFPU Kernel Source

**Kernel style determination**: The kernel uses raw `TTI_`/`TT_` instructions exclusively (`TTI_SFPMOV`, `TTI_SFPSETSGN`, `TTI_SFPSETEXP`, `TTI_SFPADDI`, `TTI_SFPMAD`, `TTI_SFPSTORE`, `TT_SFPLOADI`) combined with `dst_reg++` from the SFPI abstraction. However, there is NO condition code manipulation -- no `SFPSETCC`, `SFPENCC`, `SFPCOMPC`, or `v_if`/`v_else` usage. Therefore, **Style A** (inline-commented source) is appropriate since CC flow is trivially "always enabled".

#### Wormhole B0 Implementation

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void rand_init(uint32_t seed) {
    init_prng_seed(seed); // Seeds the hardware PRNG LFSR in all 32 SFPU lanes
}

template <bool APPROXIMATION_MODE>
inline void rand(uint32_t from, uint32_t scale) { // APPROXIMATION_MODE is unused
    // Load scale param to lreg1 as a raw 32-bit FP32 value (two 16-bit halves)
    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);  // InstrMod=10 (LO16_ONLY): write lower 16 bits, preserve upper
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);       // InstrMod=8 (HI16_ONLY): write upper 16 bits, preserve lower

    // Load from param to lreg2 as a raw 32-bit FP32 value
    TT_SFPLOADI(p_sfpu::LREG2, 10, from & 0xFFFF);    // LO16_ONLY
    TT_SFPLOADI(p_sfpu::LREG2, 8, from >> 16);        // HI16_ONLY

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        // Generate random float by reading the PRNG counter via RS[9]
        TTI_SFPMOV(0, 9, p_sfpu::LREG0, 8); // InstrMod=8: read from RS view; VC=9 (PRNG Counter); advances PRNG then copies 32-bit random value to LREG0

        // Unset sign bit and Set exponent to 127 to ensure the float is within the range [1, 2).
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG0, 1); // InstrMod=1: sign bit = Imm12[0] = 0; result: positive float with random mantissa
        TTI_SFPSETEXP(127, p_sfpu::LREG0, p_sfpu::LREG0, 1); // InstrMod=1: exponent = Imm12[7:0] = 127 (FP32 bias); result: 1.mantissa = [1.0, 2.0)

        // -1 to ensure the float is within the range [0, 1).
        TTI_SFPADDI(0xbf80 /*-1.0 in FP16_B*/, p_sfpu::LREG0, 0); // LREG0 = LREG0 + (-1.0); latency=2
        TTI_SFPNOP; // Required: SFPADDI has 2-cycle latency; dependent SFPMAD must wait 1 cycle

        // Scale the float from [0, 1) to [from, from + scale)
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0); // LREG0 = LREG0 * LREG1 + LREG2 = rand * scale + from; latency=2
        TTI_SFPNOP; // Required: SFPMAD has 2-cycle latency; dependent SFPSTORE must wait 1 cycle

        TTI_SFPSTORE(0, 3, 3, 0); // lreg_ind=0 (LREG0), instr_mod0=3 (FP32), addr_mode=3 (ADDR_MOD_3), dest=0
        dst_reg++; // Advance SFPI address by 1 (= 2 physical DEST rows = 32 elements)
    }
}
} // namespace ckernel::sfpu
```

#### Blackhole Implementation

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void rand_init(uint32_t seed) {
    init_prng_seed(seed); // Seeds the hardware PRNG LFSR in all 32 SFPU lanes
}

template <bool APPROXIMATION_MODE>
inline void rand(uint32_t from, uint32_t scale) { // APPROXIMATION_MODE is unused
    // Load scale param to lreg1 as a raw 32-bit FP32 value (two 16-bit halves)
    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);  // LO16_ONLY
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);       // HI16_ONLY

    // Load from param to lreg2 as a raw 32-bit FP32 value
    TT_SFPLOADI(p_sfpu::LREG2, 10, from & 0xFFFF);    // LO16_ONLY
    TT_SFPLOADI(p_sfpu::LREG2, 8, from >> 16);        // HI16_ONLY

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        // Generate random float by reading the PRNG counter via RS[9]
        TTI_SFPMOV(0, 9, p_sfpu::LREG0, 8); // InstrMod=8: read from RS view; VC=9 (PRNG Counter); advances PRNG then copies 32-bit random value to LREG0

        // Unset sign bit and Set exponent to 127 to ensure the float is within the range [1, 2).
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG0, 1); // InstrMod=1: sign = Imm12[0] = 0
        TTI_SFPSETEXP(127, p_sfpu::LREG0, p_sfpu::LREG0, 1); // InstrMod=1: exponent = 127

        // -1 to ensure the float is within the range [0, 1).
        TTI_SFPADDI(0xbf80 /*-1.0 in FP16_B*/, p_sfpu::LREG0, 0); // LREG0 = LREG0 + (-1.0); latency=2
        // NOTE: No SFPNOP here on Blackhole (unlike Wormhole) -- BH pipeline may handle this differently

        // Scale the float from [0, 1) to [from, from + scale)
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0); // LREG0 = rand * scale + from
        // NOTE: No SFPNOP here on Blackhole (unlike Wormhole)

        TTI_SFPSTORE(p_sfpu::LREG0, FP32, ADDR_MOD_7, 0); // Uses ADDR_MOD_7 (configured by init with dest.incr=0)
        dst_reg++; // Advance SFPI address
    }
}
} // namespace ckernel::sfpu
```

#### Key Differences Between Wormhole and Blackhole

| Aspect | Wormhole B0 | Blackhole |
|--------|-------------|-----------|
| NOP after SFPADDI | Yes (`TTI_SFPNOP`) | No |
| NOP after SFPMAD | Yes (`TTI_SFPNOP`) | No |
| SFPSTORE addr_mode | `ADDR_MOD_3` (hardcoded `3`) | `ADDR_MOD_7` (symbolic) |
| SFPSTORE format | Hardcoded `3` (FP32) | Symbolic `FP32` |
| SFPSTORE lreg | Hardcoded `0` (LREG0) | Symbolic `p_sfpu::LREG0` |

The Blackhole version omits the SFPNOP instructions between the 2-cycle latency instructions (SFPADDI, SFPMAD) and their dependent consumers. This suggests the Blackhole pipeline either has shorter latencies, better forwarding, or the compiler/hardware handles the hazards automatically.

### SFPU Instructions Used

| Instruction | Opcode | Description | Latency | Used in rand |
|-------------|--------|-------------|---------|--------------|
| **SFPLOADI** | 0x71 | Load 16-bit immediate to LREG. Used with `LO16_ONLY` (InstrMod=10) and `HI16_ONLY` (InstrMod=8) modes to construct 32-bit FP32 values in LREG1 (`scale`) and LREG2 (`from`). | 1 | Pre-loop parameter setup |
| **SFPMOV** | 0x7C | Register move. With `InstrMod=8`, reads from the RS (SFPU Status) view instead of the RG (GPR) view. When `VC=9`, RS[9] maps to the **PRNG Counter** -- the hardware advances the 32-bit LFSR PRNG before returning the value. This is the random number generation mechanism. | 1 | Once per iteration (generates random bits) |
| **SFPSETSGN** | 0x89 | Set sign bit of FP32 value. With `InstrMod=1`, sets sign from `Imm12[0]` (here `Imm12=0`, so sign=0=positive). Preserves exponent and mantissa from VC (LREG0). | 1 | Once per iteration (force positive) |
| **SFPSETEXP** | 0x82 | Set exponent of FP32 value. With `InstrMod=1`, sets exponent from `Imm12[7:0]` (here 127, the FP32 bias for 2^0=1). Preserves sign and mantissa from VC (LREG0). Combined with SFPSETSGN, this produces `+1.mantissa` = a value in [1.0, 2.0). | 1 | Once per iteration (set exponent to 127) |
| **SFPADDI** | 0x75 | Floating-point add with 16-bit immediate. Adds FP16_B value `0xBF80` (-1.0) to LREG0, converting the [1.0, 2.0) range to [0.0, 1.0). | 2 | Once per iteration (subtract 1.0) |
| **SFPMAD** | 0x84 | Fused multiply-add: `VD = VA * VB + VC`. Computes `LREG0 = LREG0 * LREG1 + LREG2` = `rand_val * scale + from`, mapping [0,1) to [from, from+scale). | 2 | Once per iteration (scale and offset) |
| **SFPNOP** | 0x8E | No operation. Required on Wormhole to satisfy 2-cycle latency dependency between SFPADDI/SFPMAD and their dependent consumers. Not used on Blackhole. | 1 | WH only: twice per iteration |
| **SFPSTORE** | 0x72 | Store LREG to DEST register file. Uses FP32 format mode (InstrMod=3). On WH uses ADDR_MOD_3, on BH uses ADDR_MOD_7. | 2-3 | Once per iteration (write result to DEST) |

### SFPU Register Usage

| Register | Role | Lifetime |
|----------|------|----------|
| **LREG0** | Working register for the random value. Receives PRNG output via SFPMOV, gets sign/exponent manipulated, undergoes subtract and FMA, then is stored to DEST. | Per-iteration (overwritten each iteration) |
| **LREG1** | Holds the `scale` parameter as FP32. Loaded once before the loop via two SFPLOADI calls (LO16 + HI16). | Entire function call (read-only during loop) |
| **LREG2** | Holds the `from` parameter as FP32. Loaded once before the loop via two SFPLOADI calls (LO16 + HI16). | Entire function call (read-only during loop) |
| **RS[9]** | PRNG Counter (architectural, read-only via RS view). Each read via SFPMOV(InstrMod=8, VC=9) advances the 32-bit LFSR by one step. | Hardware state, persists across calls |
| **DEST** | Target register file. 8 sfpi rows are written per face (32 elements each = 256 elements/face). `dst_reg++` advances the write pointer. | Per-tile accumulation |

### Address Mode Configuration

**Wormhole B0:**
- **Init configures `ADDR_MOD_7`** with `{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}` -- no auto-increment on any field.
- **The SFPSTORE instruction uses `ADDR_MOD_3`** (hardcoded as literal `3`). ADDR_MOD_3 is NOT explicitly configured by the rand init or the standard SFPU init. Its behavior depends on whatever default values exist in the addr_mod registers. Since the kernel relies on `dst_reg++` (SFPI abstraction) rather than addr_mod auto-increment for DEST address progression, the actual ADDR_MOD_3 configuration is inconsequential for correctness -- the SFPSTORE writes at the current RWC position and `dst_reg++` advances it afterward.
- Between faces, `TTI_SETRWC(CR_D, 8) x2` advances by 16 physical rows (one face stride).

**Blackhole:**
- **Init configures `ADDR_MOD_7`** with `{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}` -- identical to Wormhole.
- **The SFPSTORE instruction uses `ADDR_MOD_7`** (symbolic constant), consistent with the init configuration.
- Between faces, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice (= 16 physical rows).
- Both architectures use `dst_reg++` within the loop body for per-iteration DEST address advancement (1 sfpi row = 2 physical rows per step).

### Algorithm Summary

The `rand` SFPU kernel implements a uniform random number generator in the range `[from, from + scale)` using the following algorithm:

1. **PRNG Seeding** (`rand_init`): Seeds the per-lane 32-bit LFSR via the `PRNG_SEED` configuration register. A 600-cycle SFPNOP delay follows to allow the LFSR to stabilize.

2. **Parameter Loading**: The `from` and `scale` FP32 values are loaded into LREG1 and LREG2 respectively, using pairs of `SFPLOADI` calls (LO16_ONLY + HI16_ONLY) to construct full 32-bit values from 16-bit immediates.

3. **Per-iteration random generation** (8 iterations per face, 4 faces per tile = 32 iterations total):
   - Read the PRNG counter via `SFPMOV(RS[9] -> LREG0)`, which simultaneously advances the LFSR
   - Force the sign bit to 0 (positive) via `SFPSETSGN`
   - Set the exponent to 127 (FP32 bias) via `SFPSETEXP`, producing a value in [1.0, 2.0) with random mantissa bits
   - Subtract 1.0 via `SFPADDI(-1.0)`, producing a value in [0.0, 1.0)
   - Apply affine transformation via `SFPMAD(rand * scale + from)`, producing the final value in [from, from + scale)
   - Store to DEST via `SFPSTORE(FP32)`
   - Advance DEST pointer via `dst_reg++`

The PRNG implementation is a 32-bit LFSR shifting towards the LSB, with XNOR taps at positions 31, 30, 10, and 0 (equivalent to primitive feedback polynomial `x^32 + x^31 + x^11 + x^1 + 1`), providing a period of `2^32 - 1` cycles.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does SFPMOV with instr_mod1=8 generate random numbers in SFPU?"
   **Reason**: Needed to understand the PRNG access mechanism via the RS view
   **Key Findings**: DeepWiki was unavailable (repository not indexed). Fell back to Confluence ISA page.

### Confluence References
1. **Section**: "SFPMOV" instruction definition (Tensix SFPU ISA, page 1170505767)
   **Key Findings**: SFPMOV with InstrMod=8 reads from the RS (SFPU Status) view. When VC=9, it accesses the PRNG Counter and calls `AdvancePrng()` before returning the value. This is the sole mechanism for generating random bits in the SFPU.

2. **Section**: "PRNG" (Tensix SFPU ISA, page 1170505767)
   **Key Findings**: Each SFPU lane has a 32-bit LFSR PRNG with XNOR taps at positions 31, 30, 10, 0. Period >= 2^32-1. Can be re-seeded dynamically. SFPU instructions cannot alter PRNG state directly -- they only consume values and request advancement.

3. **Section**: "SFPU Status (RS) View" (Tensix SFPU ISA, page 1170505767)
   **Key Findings**: RS[9] maps to the PRNG Counter. Reading from this view has the side effect of advancing the PRNG by 1 step.

4. **Section**: "SFPLOADI" instruction definition (Tensix SFPU ISA, page 1170505767)
   **Key Findings**: InstrMod=8 (HI16_ONLY) writes upper 16 bits preserving lower; InstrMod=10 (LO16_ONLY) writes lower 16 bits preserving upper. Used to construct 32-bit values from two 16-bit immediates.

5. **Section**: "SFPSETSGN" instruction definition (Tensix SFPU ISA, page 1170505767)
   **Key Findings**: InstrMod=1 sets sign bit from Imm12[0]. Preserves exponent and mantissa from the VC source register.

6. **Section**: "SFPSETEXP" instruction definition (Tensix SFPU ISA, page 1170505767)
   **Key Findings**: InstrMod=1 sets exponent from Imm12[7:0]. Preserves sign and mantissa from the VC source register.

### Glean References
No Glean queries were needed for this analysis.
