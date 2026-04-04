## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to.

### Unary Dispatch Summary
- **UnaryOpType**: `WHERE_TSS`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/where_tss_kernel.cpp` (custom kernel, NOT `eltwise_sfpu.cpp`)
- **SFPU_OP_CHAIN_0 expansion**: `where_tile<DataFormat::Float16_b>(0, 1, 2, 0)` (or `DataFormat::Float32`/`DataFormat::Int32` depending on input dtype)

**Note on dispatch path**: WHERE_TSS uses a **custom compute kernel** (`where_tss_kernel.cpp`) rather than the standard `eltwise_sfpu.cpp`. This kernel performs three setup steps before invoking the SFPU chain:
1. Copies the input (condition) tile from CB `c_0` into DEST register 0 via `copy_tile`
2. Fills DEST register 1 with the `true_value` scalar via `fill_tile`
3. Fills DEST register 2 with the `false_value` scalar via `fill_tile`
4. Then invokes `SFPU_OP_CHAIN_0` which calls `where_tile<DataFormat>(0, 1, 2, 0)` — condition at DEST[0], true at DEST[1], false at DEST[2], output overwrites DEST[0]

The two scalar parameters (`true_value`, `false_value`) are passed as packed runtime args from the program factory, which extracts them from the `EltwiseUnaryWithParam` params vector.

#### Approximation Mode Resolution

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(WHERE_TSS)` in `unary_op_utils.cpp` — returns `false` (switch has only a `default: return false` case) |
| Template parameter (SFPU_OP_CHAIN) | none — `where_tile<DataFormat>` takes `DataFormat` as its only template argument, not an approximation parameter | `where_tile` delegates to `llk_math_eltwise_ternary_sfpu_where<APPROX, data_format>` where `APPROX` comes from the global `APPROX` define |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `_calculate_where_` — however, the template parameter is **unused** by the implementation (no `if constexpr (APPROXIMATION_MODE)` branch exists). Both paths are identical. | `ckernel_sfpu_where.h` — `APPROXIMATION_MODE` appears only in the template signature |

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/where.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_where.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_where.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_ternary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** (`where_tss_kernel.cpp`): `SFPU_OP_CHAIN_0` expands to `where_tile<DataFormat::...>(0, 1, 2, 0)`.
2. **API Header** (`where.h`): `where_tile<data_format>(idst0, idst1, idst2, odst)` calls `llk_math_eltwise_ternary_sfpu_where<APPROX, data_format>(idst0, idst1, idst2, odst)`.
3. **LLK Dispatch** (`llk_math_eltwise_ternary_sfpu_where.h`): `llk_math_eltwise_ternary_sfpu_where` calls `_llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>()` with `_calculate_where_<APPROXIMATE, data_format, 8>` as the callable function pointer.
4. **Parameters Dispatch** (`llk_math_eltwise_ternary_sfpu_params.h`): `_llk_math_eltwise_ternary_sfpu_params_` handles vector mode iteration (4 faces for `VectorMode::RC`), calling the SFPU function once per face and issuing `SETRWC` to advance between faces.
5. **Core SFPU Implementation** (`ckernel_sfpu_where.h`): `_calculate_where_<APPROXIMATION_MODE, data_format, 8>()` executes the actual SFPU instructions — loading data from 3 DEST tile regions, conditionally selecting true/false values based on EQ0 test, and storing the result.

Similarly, `where_tile_init()` calls `llk_math_eltwise_ternary_sfpu_where_init<APPROX>()` which calls `_llk_math_eltwise_ternary_sfpu_init_<SfpuType::where>()` (address mode configuration) and `_init_where_<APPROXIMATE>()` (SFPLOADMACRO instruction template programming).

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` (default) — all 4 tile faces are processed. Each face is processed in one call to the core SFPU function.
- **Operation invocation**: The params dispatch loops over 4 faces (`for (int face = 0; face < 4; face++)`), calling `_calculate_where_<APPROXIMATE, data_format, 8>(dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out)` once per face. Within each face, the core function's replay loop iterates 8 times (`ITERATIONS=8`), processing 8 sfpi rows × 32 elements = 256 elements per face.
- **DEST address progression**: Unlike standard unary SFPU operations that use `dst_reg++` auto-increment, this ternary operation manages DEST addressing through the `lltt::replay` mechanism and ADDR_MOD configurations. ADDR_MOD_7 (BH) / ADDR_MOD_3 (WH) is configured with `dest.incr = 0` (no auto-increment), while ADDR_MOD_6 (BH) / ADDR_MOD_2 (WH) is configured with `dest.incr = 2` (advance by 2 physical DEST rows per access). The replay buffer replays load/store instructions that auto-increment the DEST address via the `incr=2` mode, covering one face per 8 replay iterations. Between faces, `TTI_SETRWC(CLR_NONE, CR_D, 8, ...)` advances the DEST base address by 8 sfpi rows (= 16 physical rows = one face boundary).

### Annotated SFPU Kernel Source

This kernel uses raw `TT_`/`TTI_` instructions with complex SFPLOADMACRO superscalar dispatch. Style B is used for the `_calculate_where_` function, and Style A for the `_init_where_` function since it only programs instruction templates (no conditional execution logic).

#### `_calculate_where_` (Blackhole variant)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h

template <bool APPROXIMATION_MODE, DataFormat data_format, int ITERATIONS>
inline void _calculate_where_(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_in2, const std::uint32_t dst_index_out)
{
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b || data_format == DataFormat::Int32 || data_format == DataFormat::UInt32,
        "Unsupported data format for _calculate_where_(). Only Float32, Int32, UInt32, and Float16_b are allowed.");

    int offset0 = (dst_index_in0 * 32) << 1;
    int offset1 = (dst_index_in1 * 32) << 1;
    int offset2 = (dst_index_in2 * 32) << 1;

    constexpr std::uint32_t mod0 = data_format == DataFormat::Float16_b ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;

#ifdef DISABLE_SFPLOADMACRO
    int offset3 = (dst_index_out * 32) << 1;

    lltt::record(0, 6);
    TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_7, offset0);
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, offset1);
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0);
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, offset2);
    TTI_SFPENCC(0, 0, 0, sfpi::SFPENCC_MOD1_EU_R1);
    TT_SFPSTORE(p_sfpu::LREG1, mod0, ADDR_MOD_6, offset3);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        lltt::replay(0, 6);
    }
#else
    if (dst_index_out == dst_index_in0)
    {
        // Implementation notes, see the original file for more details

        load_replay_buf(
            0,
            3,
            [offset0, offset1, offset2]
            {
                TT_SFPLOADMACRO((0 << 2), mod0, ADDR_MOD_7, offset0);
                TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_7, offset1);
                TT_SFPLOAD(0, mod0, ADDR_MOD_6, offset2);
            });

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            lltt::replay(0, 3);
        }
    }
    else
    {
        // Implementation notes, see the original file for more details

        int offset3 = (dst_index_out * 32) << 1;

        load_replay_buf(
            0,
            4,
            [offset0, offset1, offset2, offset3]
            {
                TT_SFPLOADMACRO((1 << 2), mod0, ADDR_MOD_7, offset0);
                TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_7, offset1);
                TT_SFPLOAD(0, mod0, ADDR_MOD_7, offset2);
                TT_SFPSTORE(0, mod0, ADDR_MOD_6, offset3);
            });

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            lltt::replay(0, 4);
        }
    }
#endif
}
```

#### CC State Machine — `_calculate_where_` (DISABLE_SFPLOADMACRO path)

This diagram shows the fallback non-LOADMACRO path, which is clearer about instruction-level CC behavior. The LOADMACRO path achieves the same CC transitions but they happen inside the macro replay at superscalar speed.

```
_calculate_where_ — CC State Transitions (DISABLE_SFPLOADMACRO path)
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED (CC.En = 0)              <-- initial state
       |
       |  SFPLOAD L0 = DEST[offset0]             (no CC effect) -- load condition value
       |  SFPLOAD L1 = DEST[offset1]             (no CC effect) -- load true_value
       |
       v
  +---------------------------------------------+
  | SFPSETCC  mod1=6 (LREG_EQ0)                |
  |   src: LREG0 (condition value)              |
  |                                             |
  | CC.En <- 1, CC.Res <- (L0 == 0)            |
  |    = (condition is zero/false)              |
  +-------------------+-------------------------+
                      |
                      v
  CC State: ENABLED where condition == 0 (false lanes)
       |
       |  SFPLOAD L1 = DEST[offset2]   (CC-guarded: overwrite L1 with false_value
       |                                  ONLY on lanes where condition == 0)
       |
       v
  +---------------------------------------------+
  | SFPENCC  mod1=0 (EU_R1)                    |
  |                                             |
  | CC.En <- unchanged (stays 1)               |
  | CC.Res <- 1 (force all lanes active)       |
  |    = (re-enable all lanes for STORE)        |
  +-------------------+-------------------------+
                      |
                      v
  CC State: ALL_ENABLED (CC.En = 1, CC.Res = 1 on all lanes)
       |
       |  SFPSTORE DEST[offset3] = L1   (unconditional: stores the selected value)
       |
       v  (iteration complete, L1 holds true_value where cond!=0, false_value where cond==0)
```

**Key CC observations:**
- `SFPSETCC` with `LREG_EQ0` mode tests if the condition register (LREG0) is exactly zero, setting CC.Res=1 for zero (false) lanes.
- The subsequent `SFPLOAD L1 = DEST[offset2]` is CC-guarded: it only overwrites LREG1 (which already holds `true_value`) with `false_value` on lanes where the condition was zero. Non-zero (true) lanes retain the `true_value` in L1.
- `SFPENCC` with `EU_R1` mode sets CC.Res=1 on all lanes without changing CC.En, effectively re-enabling all lanes for the unconditional SFPSTORE.
- This is a classic "load both, conditionally overwrite one" pattern — highly efficient at only 6 instructions per 32 elements.

#### `_calculate_where_` — SFPLOADMACRO Superscalar Path (default)

When `DISABLE_SFPLOADMACRO` is **not** defined (the normal case), the kernel uses the SFPLOADMACRO superscalar execution mechanism for maximum throughput. There are two sub-paths:

**Path A: In-place output (`dst_index_out == dst_index_in0`)** — 3 cycles per 32 elements:
The SFPLOADMACRO replay buffer contains 3 instructions. Each replay cycle executes:
- **Load Unit**: SFPLOAD from condition/true/false DEST regions (one per cycle)
- **Simple Unit**: SFPSETCC (EQ0 test) and SFPENCC (re-enable) scheduled via Macro 0 and Macro 2 instruction templates
- **Store Unit**: SFPSTORE result back to condition DEST region (scheduled via Macro 0's store template)

**Path B: Separate output** — 4 cycles per 32 elements:
Same logic but with an additional SFPSTORE cycle since the output goes to a different DEST region, requiring Macro 1 (no store template) and a separate SFPSTORE instruction.

#### `_init_where_` (identical for Wormhole and Blackhole)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/common/inc/sfpu/ckernel_sfpu_where.h

template <bool APPROXIMATION_MODE> // APPROXIMATION_MODE=false, unused
inline void _init_where_()
{
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]: SFPSETCC with LREG_EQ0 mode — tests condition for zero
    TTI_SFPSETCC(0, 0, 12, 6); // slot=12 means instruction template position 0; mod1=6 = SFPSETCC_MOD1_LREG_EQ0

    // InstructionTemplate[1]: SFPENCC with EU_R1 — re-enables all lanes after conditional load
    TTI_SFPENCC(0, 0, 13, 0); // slot=13 means instruction template position 1; mod1=0 = SFPENCC_MOD1_EU_R1

    // Macro 0: in-place case where(a, b, c, a) — output overwrites first input
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 4; // template_idx=0 (SFPSETCC), delay=4
        constexpr std::uint32_t mad_bits    = 0; // no MAD instruction
        constexpr std::uint32_t round_bits  = 0; // no rounding instruction
        constexpr std::uint32_t store_bits  = 0x00 | 0x00 | (2 << 3) | 3; // SFPSTORE at template position, delay=3

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits); // program lower 16 bits
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits); // program upper 16 bits
        TTI_SFPCONFIG(0, 4 + 0, 0); // commit Macro 0 configuration
    }

    // Macro 1: separate-output case where(a, b, c, d)
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 4; // template_idx=0 (SFPSETCC), delay=4
        constexpr std::uint32_t mad_bits    = 0; // no MAD instruction

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1); // commit Macro 1 configuration
    }

    // Macro 2: shared between both paths — schedules SFPENCC
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 5; // template_idx=0 maps to SFPENCC (idx 1), delay=5
        constexpr std::uint32_t mad_bits    = 0; // no MAD instruction

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 2, 1); // commit Macro 2 configuration
    }

    // Misc configuration: {UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1} for all macros
    TTI_SFPCONFIG(0x770, 8, 1);
#endif
}
```

#### `_calculate_where_` (Wormhole B0 variant)

The Wormhole B0 variant is functionally identical to the Blackhole variant. The only differences are the ADDR_MOD indices used:

| Purpose | Blackhole | Wormhole B0 |
|---------|-----------|-------------|
| No auto-increment (loads) | `ADDR_MOD_7` | `ADDR_MOD_3` |
| Auto-increment dest by 2 (stores/last load) | `ADDR_MOD_6` | `ADDR_MOD_2` |

The Wormhole variant also uses `lltt::record`/`lltt::replay` directly (without `load_replay_buf` wrapper) for the LOADMACRO path, but the instruction sequence and CC logic are identical.

### SFPU Instructions Used

| Instruction | Description | Usage in this kernel |
|-------------|-------------|---------------------|
| `SFPLOAD` | Load data from DEST register row into LREG with format conversion | Loads condition (DEST[offset0]→L0), true_value (DEST[offset1]→L1), and conditionally loads false_value (DEST[offset2]→L1) |
| `SFPLOADMACRO` | Superscalar load that combines SFPLOAD with up to 4 pre-programmed instructions | Used for peak throughput — combines SFPLOAD with scheduled SFPSETCC/SFPENCC/SFPSTORE in parallel |
| `SFPSTORE` | Store LREG data back to DEST register row with format conversion | Writes the selected value (L1) back to DEST output region |
| `SFPSETCC` | Set condition code based on register comparison | Tests if condition value (L0) equals zero — `SFPSETCC_MOD1_LREG_EQ0` mode. Sets CC.Res=1 for zero lanes |
| `SFPENCC` | Enable/disable condition code, set/clear CC.Res | Re-enables all lanes after conditional load — `SFPENCC_MOD1_EU_R1` mode (Enable Unchanged, Result=1) |
| `SFPLOADI` | Load 16-bit immediate to LREG | Used in `_init_where_` to program SFPLOADMACRO instruction templates |
| `SFPCONFIG` | Configure SFPU control register / programmable constants | Used in `_init_where_` to commit macro configurations (instruction templates and misc settings) |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Holds the condition value loaded from DEST. Used by SFPSETCC to test for zero. In the LOADMACRO path, L0 is reused for all loads since the macros pipeline the test-before-overwrite. |
| **LREG1** | Holds the true_value (loaded from DEST[offset1]). Conditionally overwritten with false_value (from DEST[offset2]) on lanes where the condition is zero. Final value stored to output. In the DISABLE_SFPLOADMACRO path, this register is used explicitly. |
| **DEST registers** | Three input tile regions indexed by `dst_index_in0`, `dst_index_in1`, `dst_index_in2` and one output region `dst_index_out`. Each is a full 32×32 tile occupying 64 physical DEST rows. Offsets computed as `(dst_index * 32) << 1` = `dst_index * 64` physical rows. |

### Address Mode Configuration

Two address modes are configured in `eltwise_ternary_sfpu_configure_addrmod<SfpuType::where>()`:

**Wormhole B0:**
| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_3` | 0 | 0 | 0 | Used for SFPLOAD instructions — no auto-increment of DEST address, so the same base + replay-advanced offset is used |
| `ADDR_MOD_2` | 0 | 0 | 2 | Used for SFPSTORE (and last SFPLOAD in the in-place path) — auto-increments DEST address by 2 physical rows (= 1 sfpi row = 32 elements) after each access |

**Blackhole:**
| ADDR_MOD | srca.incr | srcb.incr | dest.incr | Purpose |
|----------|-----------|-----------|-----------|---------|
| `ADDR_MOD_7` | 0 | 0 | 0 | Same as WH ADDR_MOD_3 — no auto-increment |
| `ADDR_MOD_6` | 0 | 0 | 2 | Same as WH ADDR_MOD_2 — auto-increment by 2 physical rows |

**Note on Blackhole vs Wormhole ADDR_MOD indices**: On Wormhole, `_llk_math_eltwise_ternary_sfpu_start_` calls `math::set_addr_mod_base()` which shifts the address mode base, so ADDR_MOD_2 and ADDR_MOD_3 in WH code map to higher physical registers. On Blackhole, this call is replaced by `TTI_SETC16(2, 0)` in `_done_()`, and ADDR_MOD_6/7 are used directly. The functional behavior is identical.

## Local Knowledge Sources
### Local References
1. **File**: `.claude/references/sfpu-hardware-model.md`
   **Reason**: Authoritative reference for SFPU architecture, instruction semantics, CC mechanism, and addressing model
   **Key Findings**: Confirmed SFPSETCC_MOD1_LREG_EQ0 = mode 6, SFPENCC_MOD1_EU_R1 = mode 0, SFP_DESTREG_STRIDE=2, SFPLOADMACRO superscalar semantics

2. **File**: `.claude/references/diagram-templates.md`
   **Reason**: Template for CC State Machine diagrams
   **Key Findings**: Used the generalized CC state machine template format for documenting the DISABLE_SFPLOADMACRO fallback path

3. **File**: `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h`
   **Reason**: Core SFPU implementation for Blackhole architecture
   **Key Findings**: Uses SFPLOADMACRO superscalar dispatch for 3-4 cycles per 32 elements; has two sub-paths for in-place vs separate output; identical CC logic to WH

4. **File**: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_where.h`
   **Reason**: Core SFPU implementation for Wormhole B0 architecture
   **Key Findings**: Functionally identical to BH with different ADDR_MOD indices (2/3 vs 6/7); uses lltt::record/replay directly instead of load_replay_buf wrapper

5. **File**: `tt_metal/hw/inc/api/compute/eltwise_unary/where.h` (from main repo)
   **Reason**: API header exposing where_tile and where_tile_init to compute kernels
   **Key Findings**: where_tile delegates to llk_math_eltwise_ternary_sfpu_where; where_tile_init delegates to llk_math_eltwise_ternary_sfpu_where_init

6. **File**: `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_where.h`
   **Reason**: LLK dispatch layer bridging API to ckernel implementation
   **Key Findings**: Passes _calculate_where_<APPROXIMATE, data_format, 8> to _llk_math_eltwise_ternary_sfpu_params_; ITERATIONS fixed at 8

7. **File**: `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_ternary_sfpu_params.h`
   **Reason**: Parameters dispatch controlling face iteration and DEST address progression
   **Key Findings**: VectorMode::RC processes all 4 faces; SETRWC advances by 8+8 sfpi rows between faces; identical across WH and BH

8. **File**: `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_ternary_sfpu.h`
   **Reason**: Ternary SFPU init, start, done, and address mode configuration
   **Key Findings**: SfpuType::where configures ADDR_MOD_6/7 (BH) or ADDR_MOD_2/3 (WH) with dest.incr=0 and dest.incr=2 respectively

9. **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/where_tss_kernel.cpp`
   **Reason**: Custom compute kernel for WHERE_TSS that sets up DEST registers before SFPU dispatch
   **Key Findings**: Loads condition tile to DEST[0], fills DEST[1] with true_value, fills DEST[2] with false_value, then calls SFPU_OP_CHAIN_0

10. **File**: `runtime/sfpi/include/lltt.h`
    **Reason**: Low-level replay buffer interface (lltt::record, lltt::replay)
    **Key Findings**: lltt::record starts recording instructions into replay buffer; lltt::replay replays recorded instructions without re-decoding
