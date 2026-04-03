## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: N/A -- `rand` is NOT a standard `UnaryOpType`. It has its own dedicated program factory (`RandDeviceOperation::ProgramFactory`) at `ttnn/cpp/ttnn/operations/rand/device/rand_program_factory.cpp`.
- **Compute kernel**: `ttnn/cpp/ttnn/operations/rand/device/kernels/compute_uniform.cpp`
- **SFPU_OP_CHAIN_0 expansion**: N/A -- the compute kernel directly calls `rand_tile_init(seed)` and `rand_tile(0, f2u_from.u, f2u_scale.u)` without going through the standard `SFPU_OP_CHAIN_0` macro mechanism.

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `true` | Set explicitly in `rand_program_factory.cpp` line 87: `ComputeConfig{..., .math_approx_mode = true, ...}` |
| Template parameter (SFPU_OP_CHAIN) | `APPROX` (passed through macro) | `rand_tile()` uses `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(rand, RC, APPROX, ...)` where `APPROX` is the `math_approx_mode` value from `ComputeConfig` |
| Effective SFPU path | Approximation mode has no effect on `rand` -- the `APPROXIMATION_MODE` template parameter is accepted but never referenced in the kernel body | The `rand<APPROXIMATION_MODE>` function ignores this parameter entirely; there are no `if constexpr (APPROXIMATION_MODE)` branches |

**Additional ComputeConfig notes**: The rand factory also sets `fp32_dest_acc_en = true` (to avoid precision errors that could push generated numbers out of the `[from, to)` range) and `math_fidelity = MathFidelity::HiFi4`.

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/rand.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist -- the API header directly invokes the macro `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` which expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::rand<APPROX>, idst, (int)VectorMode::RC, from, scale)` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h` (Wormhole B0) / `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h` (Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole B0) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole) |

### Call Chain
1. **Compute kernel** (`compute_uniform.cpp`): calls `rand_tile_init(seed)` once before the tile loop, then `rand_tile(0, from, scale)` per tile.
2. **API header** (`rand.h`): `rand_tile_init(seed)` expands via `SFPU_ONE_PARAM_KERNEL_INIT(unused, sfpu::rand_init, APPROX, seed)` to `llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROX>(sfpu::rand_init<APPROX>, seed)`. This calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::unused>()` (configures addr_mod, resets counters, inits SFPU config reg), then calls `rand_init<APPROX>(seed)` which writes the seed to the hardware PRNG register.
3. **API header** (`rand.h`): `rand_tile(0, from, scale)` expands via `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(rand, RC, APPROX, 0, from, scale)` to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::rand<APPROX>, 0, (int)VectorMode::RC, from, scale)`.
4. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): Sets up DEST addressing for tile index 0, stalls until SFPU is free, then loops 4 times (once per face in `VectorMode::RC` mode), calling `ckernel::sfpu::rand<APPROX>(from, scale)` each time, advancing the DEST face address between iterations with `SETRWC`.
5. **Core SFPU function** (`ckernel_sfpu_rand.h`): `rand<APPROX>(from, scale)` -- loads the `scale` and `from` parameters into LREGs, then loops 8 times per face, generating random floats in `[from, from + scale)` and storing them to DEST.

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (32x32 = 1024 elements total).
- **Operation invocation**: The core `rand<APPROX>(from, scale)` function is called 4 times (once per face). Each invocation processes 8 iterations of 32 elements = 256 elements = one full face.
- **DEST address progression**: Standard DEST progression (ITERATIONS=8 per face, `dst_reg++` per iteration, `SETRWC` between faces).
  - **Wormhole B0**: `ADDR_MOD_7` is configured during init with `{srca.incr=0, srcb.incr=0, dest.incr=0}`. However, the `SFPSTORE` in the kernel loop uses `ADDR_MOD_3` (value 3 in the third argument), so the store auto-increment behavior is determined by `ADDR_MOD_3`. The `dst_reg++` in the loop body handles the per-iteration advance explicitly.
  - **Blackhole**: `ADDR_MOD_7` is configured during init with `{srca.incr=0, srcb.incr=0, dest.incr=0}`. The `SFPSTORE` also uses `ADDR_MOD_7`. The `dst_reg++` handles per-iteration advance.
  - Between faces, `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` advances by 16 physical DEST rows (2 calls to `inc_dst_addr<8>`).

### Annotated SFPU Kernel Source

The kernel uses raw `TT_`/`TTI_` instructions. However, the CC (Condition Code) is never manipulated -- all operations are unconditional. Therefore Style A (inline-commented source) is appropriate.

#### Wormhole B0

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void rand_init(uint32_t seed) {
    init_prng_seed(seed); // Writes seed to PRNG_SEED_Seed_Val_ADDR32 config register, then waits 600 NOPs for PRNG to stabilize
}

template <bool APPROXIMATION_MODE>
inline void rand(uint32_t from, uint32_t scale) { // APPROXIMATION_MODE is unused
    // Load scale param to lreg1 (32-bit via two 16-bit halves)
    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);      // LREG1.lo16 = scale[15:0]; mod0=10 means "unsigned short, low 16 bits"
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);           // LREG1.hi16 = scale[31:16]; mod0=8 means "float upper 16 bits"

    // Load from param to lreg2 (32-bit via two 16-bit halves)
    TT_SFPLOADI(p_sfpu::LREG2, 10, from & 0xFFFF);        // LREG2.lo16 = from[15:0]
    TT_SFPLOADI(p_sfpu::LREG2, 8, from >> 16);            // LREG2.hi16 = from[31:16]

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        // Generate random float: read PRNG counter (RS[9]), advance PRNG by 1 step
        TTI_SFPMOV(0, 9, p_sfpu::LREG0, 8);              // LREG0 = RS[9] (PRNG value); mod1=8 reads from RS[] special register view; src=9 is PRNG Counter

        // Construct float in [1.0, 2.0): set sign=0, exponent=127, keep random mantissa
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG0, 1);  // LREG0.sign = Imm12[0] = 0; mod1=1 selects immediate source for sign bit
        TTI_SFPSETEXP(127, p_sfpu::LREG0, p_sfpu::LREG0, 1); // LREG0.exp = 127 (bias for 2^0); mod1=1 selects immediate source for exponent

        // Subtract 1.0 to get [0.0, 1.0)
        TTI_SFPADDI(0xbf80 /*-1.0 in FP16_B*/, p_sfpu::LREG0, 0); // LREG0 = LREG0 + (-1.0); 2-cycle latency
        TTI_SFPNOP;                                        // NOP required: dependent read of LREG0 follows 2-cycle SFPADDI

        // Scale: LREG0 = LREG0 * scale + from, mapping [0,1) to [from, from+scale)
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0); // LREG0 = LREG0 * LREG1 + LREG2; mod1=0 (no indirect, no sign inversion)
        TTI_SFPNOP;                                        // NOP required: dependent store of LREG0 follows 2-cycle SFPMAD

        TTI_SFPSTORE(0, 3, 3, 0);                         // Store LREG0 to DEST in FP32 format; lreg=0, instr_mod0=3(FP32), addr_mod=3(ADDR_MOD_3), dest=0
        dst_reg++;                                         // Advance 1 sfpi row = 2 physical DEST rows = 32 elements
    }
}
}  // namespace ckernel::sfpu
```

#### Blackhole

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_rand.h

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE>
inline void rand_init(uint32_t seed) {
    init_prng_seed(seed); // Writes seed to PRNG_SEED_Seed_Val_ADDR32 config register, then waits 600 NOPs for PRNG to stabilize
}

template <bool APPROXIMATION_MODE>
inline void rand(uint32_t from, uint32_t scale) { // APPROXIMATION_MODE is unused
    // Load scale param to lreg1 (32-bit via two 16-bit halves)
    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);      // LREG1.lo16 = scale[15:0]
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);           // LREG1.hi16 = scale[31:16]

    // Load from param to lreg2 (32-bit via two 16-bit halves)
    TT_SFPLOADI(p_sfpu::LREG2, 10, from & 0xFFFF);        // LREG2.lo16 = from[15:0]
    TT_SFPLOADI(p_sfpu::LREG2, 8, from >> 16);            // LREG2.hi16 = from[31:16]

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        // Generate random float: read PRNG counter (RS[9]), advance PRNG by 1 step
        TTI_SFPMOV(0, 9, p_sfpu::LREG0, 8);              // LREG0 = RS[9] (PRNG value); mod1=8 reads from RS[] view; src=9 is PRNG Counter

        // Construct float in [1.0, 2.0): set sign=0, exponent=127, keep random mantissa
        TTI_SFPSETSGN(0, p_sfpu::LREG0, p_sfpu::LREG0, 1);  // LREG0.sign = 0
        TTI_SFPSETEXP(127, p_sfpu::LREG0, p_sfpu::LREG0, 1); // LREG0.exp = 127

        // Subtract 1.0 to get [0.0, 1.0)
        TTI_SFPADDI(0xbf80 /*-1.0 in FP16_B*/, p_sfpu::LREG0, 0); // LREG0 = LREG0 - 1.0; 2-cycle latency
        // No SFPNOP on Blackhole: pipeline handles the data dependency differently

        // Scale: LREG0 = LREG0 * scale + from
        TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG0, 0); // LREG0 = LREG0 * LREG1 + LREG2
        // No SFPNOP on Blackhole

        TTI_SFPSTORE(p_sfpu::LREG0, FP32, ADDR_MOD_7, 0); // Store LREG0 to DEST in FP32 format using ADDR_MOD_7
        dst_reg++;                                          // Advance 1 sfpi row = 2 physical DEST rows = 32 elements
    }
}
}  // namespace ckernel::sfpu
```

#### Key Differences Between Wormhole B0 and Blackhole

| Aspect | Wormhole B0 | Blackhole |
|--------|-------------|-----------|
| NOP after SFPADDI | Yes (`TTI_SFPNOP`) | No |
| NOP after SFPMAD | Yes (`TTI_SFPNOP`) | No |
| SFPSTORE addr_mod | `ADDR_MOD_3` (hardcoded as literal `3`) | `ADDR_MOD_7` (named constant) |
| SFPSTORE format arg | `3` (literal, = FP32) | `FP32` (named enum) |

The Wormhole version requires explicit NOPs after 2-cycle instructions (SFPADDI, SFPMAD) when the next instruction depends on the result. The Blackhole version omits these NOPs, suggesting either improved pipeline forwarding or different hazard handling in the Blackhole microarchitecture.

### SFPU Instructions Used

| Instruction | Opcode | Description | Usage in `rand` |
|-------------|--------|-------------|-----------------|
| `SFPLOADI` | 0x71 | Load 16-bit immediate to LREG | Loads `scale` and `from` parameters as FP32 bit patterns into LREG1 and LREG2 (two loads each: low 16 bits then high 16 bits) |
| `SFPMOV` | 0x7C | Register copy (with special register view access when `mod1=8`) | Reads PRNG Counter (`RS[9]`) into LREG0; the side effect of reading RS[9] advances the PRNG by 1 step |
| `SFPSETSGN` | 0x89 | Set the sign bit of a floating-point value | Clears sign bit of LREG0 (sets to 0 via `Imm12[0]=0, mod1=1`) to ensure positive value |
| `SFPSETEXP` | 0x82 | Set the exponent field of a floating-point value | Sets exponent to 127 (bias for 2^0, via `Imm12=127, mod1=1`) to place the random mantissa in range [1.0, 2.0) |
| `SFPADDI` | 0x75 | Floating-point add with 16-bit FP16_B immediate | Subtracts 1.0 (`0xbf80` = -1.0 in FP16_B) from LREG0 to shift range from [1.0, 2.0) to [0.0, 1.0) |
| `SFPMAD` | 0x84 | Fused multiply-add: VD = VA * VB + VC | Computes `LREG0 = LREG0 * LREG1 + LREG2` to scale the uniform random from [0, 1) to [from, from + scale) |
| `SFPNOP` | 0x8E | No-operation (pipeline timing) | Wormhole only: inserted after SFPADDI and SFPMAD to satisfy 2-cycle latency data dependency |
| `SFPSTORE` | 0x72 | Store LREG to DEST register with format conversion | Writes the final random value from LREG0 to DEST in FP32 format |

### SFPU Register Usage

| Register | Purpose | Lifetime |
|----------|---------|----------|
| **LREG0** | Working register: holds the random value through all transformation stages (raw PRNG -> [1,2) -> [0,1) -> [from, from+scale)) | Per-iteration (written by SFPMOV, modified by SFPSETSGN/SFPSETEXP/SFPADDI/SFPMAD, stored to DEST by SFPSTORE) |
| **LREG1** | Holds `scale` parameter as FP32 bit pattern | Loaded once before the loop, constant across all 8 iterations within a face invocation; reloaded on each face call since the function is called 4 times |
| **LREG2** | Holds `from` parameter as FP32 bit pattern | Same as LREG1 |
| **RS[9]** | Hardware PRNG Counter (read-only special register view) | Accessed via `SFPMOV` with `mod1=8, VC=9`; reading advances the PRNG state by one step |
| **DEST** | Output destination register file | Written by SFPSTORE; each iteration writes 32 elements (2 physical rows x 16 columns); `dst_reg++` advances the write address |
| **PRNG_SEED_Seed_Val_ADDR32** | Hardware configuration register for PRNG seed | Written once during `rand_init()` via `init_prng_seed(seed)` |

### Address Mode Configuration

**Wormhole B0:**
- **Init-time configuration**: `ADDR_MOD_7` is set to `{srca.incr=0, srcb.incr=0, dest.incr=0}` by `eltwise_unary_sfpu_configure_addrmod<SfpuType::unused>()`. Since `SfpuType::unused` does not match any special-case `if constexpr` branches, only the default `ADDR_MOD_7` is configured.
- **Runtime store**: The `SFPSTORE` instruction uses `ADDR_MOD_3` (hardcoded literal `3` in the third argument). This means the store's auto-increment is determined by whatever `ADDR_MOD_3` is configured to. Since the rand kernel does not explicitly configure `ADDR_MOD_3`, its behavior depends on whatever default or prior configuration exists. The per-iteration advance is handled explicitly by `dst_reg++` in the loop body.
- **Between faces**: `TTI_SETRWC` (via `_llk_math_eltwise_unary_sfpu_params_`) advances the DEST write counter by 8 twice (16 physical rows = 1 face stride).

**Blackhole:**
- **Init-time configuration**: `ADDR_MOD_7` is set to `{srca.incr=0, srcb.incr=0, dest.incr=0}` by `eltwise_unary_sfpu_configure_addrmod<SfpuType::unused>()`. Same as Wormhole.
- **Runtime store**: The `SFPSTORE` instruction uses `ADDR_MOD_7` (named constant). Since `ADDR_MOD_7` is configured with `dest.incr=0`, the store does not auto-increment the DEST address. The per-iteration advance is handled by `dst_reg++`.
- **Between faces**: `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` calls `math::inc_dst_addr<8>()` twice (16 physical rows = 1 face stride).

### Algorithm Summary

The `rand` kernel generates uniformly distributed random floating-point numbers in the range `[from, from + scale)` using the following technique:

1. **PRNG sampling**: Each SFPU lane has a hardware PRNG (32-bit LFSR with XNOR taps at positions 31, 30, 10, and 0; polynomial `x^32 + x^31 + x^11 + x^1 + 1`; period >= 2^32 - 1). The `SFPMOV` instruction with `mod1=8, VC=9` reads the current PRNG value into an LREG and advances the PRNG state by one step.

2. **Float construction**: The raw 32-bit PRNG value is transformed into a float in [1.0, 2.0) by clearing the sign bit (SFPSETSGN) and forcing the exponent to 127 (SFPSETEXP). This preserves the random mantissa bits, producing a uniformly distributed float in [1.0, 2.0).

3. **Range adjustment**: Subtracting 1.0 (SFPADDI with -1.0 in FP16_B) shifts the range to [0.0, 1.0).

4. **Scaling**: The fused multiply-add `LREG0 = LREG0 * scale + from` (SFPMAD) maps from [0, 1) to the target range [from, from + scale).

This is a standard technique for generating uniform floats from integer PRNGs, exploiting the IEEE 754 float representation where setting the exponent to 127 and using random mantissa bits produces a uniform distribution in [1, 2).

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the SFPU PRNG work in tt-metal? How does SFPMOV with register source 9 access the PRNG value?"
   **Reason**: Needed to understand the PRNG access mechanism via the RS[9] special register view
   **Key Findings**: DeepWiki was unavailable (repository not indexed). Analysis relied on Confluence ISA page and source code.

### Confluence References
1. **Page**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
   **Sections consulted**:
   - **PRNG section**: Confirmed each SFPU lane has a PRNG (32-bit LFSR, XNOR taps at 31, 30, 10, 0), can be reseeded, produces 32 bits/clock, period >= 2^32 - 1
   - **Register Views table**: Confirmed RS[9] = PRNG Counter (RO), reading advances PRNG by 1 step
   - **SFPMOV Algorithmic Implementation**: Confirmed `InstrMod==0x8` copies `RS[VC]` to `RG[VD]`, and when `VC==9` the PRNG is stepped first via `AdvancePrng()`
   - **SFPSETSGN**: Confirmed `InstrMod[0]=1` takes sign from `Imm12[0]`; sets sign/exp/man of result
   - **SFPSETEXP**: Confirmed `InstrMod[1:0]=1` takes exponent from `Imm12[7:0]`
   - **SFPADDI**: Confirmed floating-point add with FP16_B immediate, 2-cycle latency, flushes subnormals

### Glean References
No Glean queries were necessary for this analysis.
