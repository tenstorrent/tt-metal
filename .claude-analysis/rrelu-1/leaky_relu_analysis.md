## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `LEAKY_RELU`
- **Compute kernel**: `eltwise_sfpu.cpp` (default path in `get_compute_kernel_path()`)
- **SFPU_OP_CHAIN_0 expansion**: `leaky_relu_tile(0, <slope_as_uint32>u)`

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(LEAKY_RELU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (uses default `APPROX` from macro) | `get_op_init_and_func()` returns `leaky_relu_tile_init()` / `leaky_relu_tile(idst, slope_u32)` -- no parameterized template argument for approximation |
| Effective SFPU path | `APPROXIMATION_MODE=false` passed to `_calculate_lrelu_<false>`. However, the function body does not branch on `APPROXIMATION_MODE` -- the same raw TTI instruction sequence executes regardless. | `_calculate_lrelu_` in `ckernel_sfpu_relu.h` has no `if constexpr (APPROXIMATION_MODE)` branches |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist -- the API header calls the macro `SFPU_UNARY_ONE_PARAM_KERNEL_FN` which directly invokes `_llk_math_eltwise_unary_sfpu_params_` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (BH) |
| **Metal Wrapper** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h` (WH) / `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h` (BH) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h` (WH) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h` (BH) |

### Call Chain
1. The compute kernel's `SFPU_OP_CHAIN_0` expands to `leaky_relu_tile(0, slope_u32)` (defined in `relu.h`).
2. `leaky_relu_tile()` invokes the macro `SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_lrelu, RC, APPROX, idst, slope)`, which expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_lrelu<APPROX>, idst, (int)VectorMode::RC, slope)`.
3. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets up the DEST write address, stalls for SFPU availability, then iterates over 4 faces in `VectorMode::RC`, calling `calculate_lrelu<false>(slope)` once per face.
4. `calculate_lrelu()` (in metal wrapper `ckernel_sfpu_relu.h`) simply delegates to `_calculate_lrelu_<APPROXIMATION_MODE>(ITERATIONS=8, slope)`.
5. `_calculate_lrelu_()` (in `tt_llk_*/common/inc/sfpu/ckernel_sfpu_relu.h`) executes the raw TTI instruction sequence that performs the leaky relu computation on 8 sfpi rows (one face).

### Parameters Dispatch Summary
- **Vector mode**: `VectorMode::RC` -- all 4 faces of the tile are processed (faces 0-3).
- **Operation invocation**: The params dispatch function calls `calculate_lrelu<false>(slope)` once per face (4 times total), with each invocation processing `ITERATIONS=8` sfpi rows. Between faces, `TTI_SETRWC` advances the DEST address by 16 physical rows (2 increments of 8 rows each on WH via the params dispatch, or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` on BH).
- **DEST address progression**: Standard DEST progression. Within a face, `sfpi::dst_reg++` advances 1 sfpi row (= 2 physical DEST rows) per iteration; SFPLOAD/SFPSTORE use `ADDR_MOD_3` on Wormhole (remapped to physical `ADDR_MOD_7` via `set_addr_mod_base()`) and `ADDR_MOD_7` directly on Blackhole. Both are configured with `.dest = {.incr = 0}` -- meaning the load/store instructions themselves do NOT auto-increment DEST RWC; the increment is performed exclusively by `sfpi::dst_reg++`. Between faces, `SETRWC` advances by the face stride (16 physical rows).

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with condition code manipulation. The CC pattern is simple (a single SFPSETCC/SFPENCC pair per iteration), so **Style A** (inline-annotated source) is used.

**Wormhole B0 implementation** (identical logic to Blackhole, only ADDR_MOD index differs):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h

template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE=false (unused in function body)
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope)
{
    // Load the 32-bit float slope value into LREG2 using two 16-bit immediate loads
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);   // InstrMod=10 (LO16): write lower 16 bits of LREG2
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);        // InstrMod=8 (HI16_ONLY): write upper 16 bits of LREG2, preserving lower
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)                // iterations=8 (one face)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);        // Load current DEST element into LREG0 (implied format)
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                         // CC.Res = (LREG0 < 0), i.e. sign bit; InstrMod=0 means "set CC if negative"
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // LREG0 = LREG0 * LREG2 + 0.0 (x * slope); CC-guarded: only executes on lanes where CC.Res=1 (negative elements)
        TTI_SFPENCC(0, 0, 0, 0);                                                      // InstrMod=0,0: keep CC.En unchanged, set CC.Res=1 -- effectively disables conditional masking since all lanes now have Res=1
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);       // Store LREG0 back to DEST (implied format)
        sfpi::dst_reg++;                                                                // Advance to next sfpi row (2 physical DEST rows = 32 elements)
    }
}
```

**Blackhole implementation** (functionally identical, ADDR_MOD_7 used directly):

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h

template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE=false (unused in function body)
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope)
{
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);        // load from dest into lreg[0]
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                         // condition - if value in LREG0 is negative
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // Multiply LREG0 * LREG2 (x * slope)
        TTI_SFPENCC(0, 0, 0, 0);                                                      // clear cc result reg
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);       // store from lreg0 into dest register
        sfpi::dst_reg++;
    }
}
```

**Metal wrapper** (both WH and BH, identical structure):

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h
// (Blackhole version is at the equivalent path under blackhole/)

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_lrelu(const uint slope) {
    _calculate_lrelu_<APPROXIMATION_MODE>(ITERATIONS, slope);
}
```

### SFPU Instructions Used

| Instruction | Opcode | Count per iteration | Description |
|-------------|--------|---------------------|-------------|
| `SFPLOADI` | 0x71 | 2 (once before loop) | Load 16-bit immediate into LREG. Used twice to assemble the 32-bit float `slope` value into LREG2: first `InstrMod=10` (LO16, writes lower 16 bits), then `InstrMod=8` (HI16_ONLY, writes upper 16 bits). Uses `TT_SFPLOADI` (runtime instruction buffer) because `slope` is a runtime parameter. |
| `SFPLOAD` | 0x70 | 1 | Load data from DEST into LREG0 with implied format conversion. Reads 32 elements (2 physical rows) from the current DEST address. |
| `SFPSETCC` | 0x7B | 1 | Set condition code result based on LREG value. `InstrMod=0` means `CC.Res = (LREG0 < 0)` -- sets the sign bit. This is a predicated instruction: only executes on lanes where `LaneEnabled` is true. Since CC.En is initially 0, all lanes are enabled (unconditional), so all lanes' CC.Res is set based on the sign of their value. |
| `SFPMUL` | 0x86 | 1 | Floating-point multiply (alias of SFPMAD with VC=0.0). Computes `LREG0 = LREG0 * LREG2 + LCONST_0` = `x * slope + 0.0`. This is CC-guarded: only executes on lanes where CC.Res=1 (negative values). Positive values remain unchanged in LREG0. |
| `SFPENCC` | 0x8A | 1 | Modify CC enable/result state. `InstrMod[3:0]=0, VC=0, Imm12=0`: keeps CC.En unchanged, sets CC.Res=1. Since CC.En was never set to 1 (the kernel relies on the default CC.En=0 state where all lanes are unconditionally active), this effectively resets CC.Res for the next iteration. Executes on all lanes regardless of CC state. |
| `SFPSTORE` | 0x72 | 1 | Store LREG0 back to DEST with implied format conversion. Writes the (possibly modified) value back to the same DEST address. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Working register -- holds the current element value loaded from DEST. After the CC-guarded SFPMUL, contains the original value (if non-negative) or `x * slope` (if negative). Stored back to DEST. |
| **LREG2** | Holds the `slope` parameter as a 32-bit FP32 value. Loaded once before the loop via two `SFPLOADI` instructions and reused across all iterations and all 4 face invocations. |
| **LCONST_0** | Hardware fixed constant = 0.0 (index 9 in the constant register file). Used as the addend in `SFPMUL` (which is SFPMAD with VC=0.0), making it a pure multiply. |
| **DEST** | Source and destination for tile data. Each SFPLOAD reads 32 elements from the current DEST address; each SFPSTORE writes 32 elements back. The DEST RWC is advanced by `sfpi::dst_reg++` (not by the ADDR_MOD auto-increment, which is configured to 0). |

### Address Mode Configuration

The SFPU init function `_llk_math_eltwise_unary_sfpu_init_<SfpuType::lrelu>()` calls `eltwise_unary_sfpu_configure_addrmod()` which configures `ADDR_MOD_7` with:
- `.srca = {.incr = 0}` -- no SrcA auto-increment
- `.srcb = {.incr = 0}` -- no SrcB auto-increment
- `.dest = {.incr = 0}` -- no DEST RWC auto-increment

This is the same configuration on both Wormhole B0 and Blackhole.

**Wormhole B0 specifics**: The params dispatch calls `math::set_addr_mod_base()` which sets the addr_mod base register to 1. This causes instruction-level ADDR_MOD fields 0-3 to be remapped to physical registers 4-7. So `ADDR_MOD_3` in `TTI_SFPLOAD`/`TTI_SFPSTORE` references physical `ADDR_MOD_7`. After the SFPU operation completes, `math::clear_addr_mod_base()` resets the base to 0.

**Blackhole specifics**: The params dispatch does NOT call `set_addr_mod_base()`. The kernel directly uses `ADDR_MOD_7` in the instruction field, referencing physical `ADDR_MOD_7` without remapping.

Both platforms ultimately use the same physical ADDR_MOD_7 with `.dest = {.incr = 0}`. DEST address progression is handled entirely by `sfpi::dst_reg++` after each iteration.

The `lrelu` SfpuType does not match any special-case in `eltwise_unary_sfpu_configure_addrmod()` (which only has special cases for `topk_local_sort`, `typecast`, `unary_max/min`, and `signbit`), so only the default `ADDR_MOD_7` configuration applies.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the SFPU leaky relu (_calculate_lrelu_) kernel work in tt-metal?"
   **Reason**: Needed to understand the SFPU instruction sequence and CC mechanism for leaky relu.
   **Key Findings**: DeepWiki was unavailable for tenstorrent/tt-metal (repository not indexed). Analysis was conducted entirely from source code and the Confluence ISA page.

### Confluence References
1. **Page**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
   **Sections consulted**:
   - **SFPSETCC**: Confirmed `InstrMod=0` sets `CC.Res = RG[VC].Sgn` (sign bit test = negative check). Confirmed this instruction is predicated (only executes on lanes where `LaneEnabled` is true).
   - **SFPMUL**: Confirmed this is an alias of SFPMAD with VC declared as 0.0. Full FMA operation `RG[VD] = RG[VA] * RG[VB] + RG[VC]`. Opcode 0x86, latency 2 cycles, IPC 1.
   - **SFPENCC**: Confirmed `InstrMod[3:0]=0` means: keep CC.En unchanged, set CC.Res=1. Executes on all lanes unconditionally.
   - **SFPLOADI**: InstrMod=10 writes LO16, InstrMod=8 writes HI16_ONLY, enabling 32-bit immediate construction from two 16-bit loads.

### Glean References
No Glean queries were needed for this analysis. The source code and Confluence ISA page provided sufficient detail.
