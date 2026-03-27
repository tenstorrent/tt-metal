## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the PRELU_SFPU unary operation.

### Unary Dispatch Summary
- **UnaryOpType**: `PRELU_SFPU`
- **Compute kernel**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`
- **SFPU_OP_CHAIN_0 expansion**: `prelu_tile({idst}, {param0_hex})` where `param0` is the PReLU slope encoded as a `uint32_t` bit-cast of the float value

#### Approximation Mode Resolution
There are TWO independent controls for approximation. Both are reported below.

| Control | Value | Source |
|---------|-------|--------|
| `math_approx_mode` (ComputeConfig) | `false` | `get_op_approx_mode(PRELU_SFPU)` in `unary_op_utils.cpp` -- falls through to `default: return false` |
| Template parameter (SFPU_OP_CHAIN) | none (uses `APPROX` which resolves to the `math_approx_mode` value from ComputeConfig) | `get_op_init_and_func()` returns `prelu_tile_init()` / `prelu_tile(idst, param0_hex)` -- no explicit template parameter override |
| Effective SFPU path | `APPROXIMATION_MODE=false` in `calculate_prelu<false>`. However, the kernel body does not branch on `APPROXIMATION_MODE` at all -- it has no `if constexpr(APPROXIMATION_MODE)` blocks, so the template parameter has no effect on execution. | `ckernel_sfpu_prelu.h` -- the function body is identical regardless of `APPROXIMATION_MODE` |

### SFPU Abstraction Layers
| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/prelu.h` |
| **LLK Dispatch** | This level of abstraction doesn't exist (the API header uses the `SFPU_UNARY_ONE_PARAM_KERNEL_FN` macro which directly calls `_llk_math_eltwise_unary_sfpu_params_`) |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h` (Wormhole) / `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h` (Blackhole) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Wormhole) / `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (Blackhole) |

### Call Chain

1. **Compute kernel** (`eltwise_sfpu.cpp`): The `SFPU_OP_CHAIN_0` macro expands to `prelu_tile(0, param0_hex)`.

2. **API Header** (`prelu.h`): `prelu_tile(uint32_t idst, uint32_t param0)` wraps the call in a `MATH(...)` guard and invokes the macro `SFPU_UNARY_ONE_PARAM_KERNEL_FN(calculate_prelu, RC, APPROX, idst, param0)`.

3. **Macro expansion** (`llk_math_eltwise_unary_sfpu_macros.h`): The macro expands to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_prelu<APPROX>, idst, (int)VectorMode::RC, param0)`.

4. **Parameters dispatch** (`llk_math_eltwise_unary_sfpu_params.h`): This function sets the DEST write address, stalls until SFPU is ready, then iterates over 4 faces (VectorMode::RC), calling `calculate_prelu<false>(param0)` once per face, advancing the DEST address between faces via `TTI_SETRWC` (Wormhole) or `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (Blackhole).

5. **Core SFPU kernel** (`ckernel_sfpu_prelu.h`): `calculate_prelu<false, 8>(param0)` executes 8 iterations per face, each processing 32 elements (2 physical DEST rows). For each iteration, it loads an element from `dst_reg[0]`, conditionally multiplies by the slope if negative, and writes back.

### Parameters Dispatch Summary

- **Vector mode**: `VectorMode::RC` -- processes all 4 faces of the tile (the full 32x32 = 1024 elements).
- **Operation invocation**: The params dispatch function calls `calculate_prelu<false>(param0)` once per face in a loop of 4 iterations. Each invocation of `calculate_prelu` internally loops 8 times (`ITERATIONS=8`), covering one full 16x16 face (8 sfpi rows x 32 elements/row = 256 elements).
- **DEST address progression**: Standard DEST progression. On **Wormhole**, the params dispatch uses `math::set_addr_mod_base()` and `TTI_SETRWC(CLR_NONE, CR_D, 8, 0, 0, SET_D)` x 2 between faces (each SETRWC advances by 8 sfpi rows). Within a face, `dst_reg++` in the SFPI kernel advances 1 sfpi row (= 2 physical DEST rows, due to `SFP_DESTREG_STRIDE=2`) per iteration, covering 32 elements per iteration. On **Blackhole**, the params dispatch uses `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` which calls `math::inc_dst_addr<8>()` twice between faces (equivalent to 16 physical DEST rows = 1 face).

### Annotated SFPU Kernel Source

The kernel uses pure SFPI abstractions (`vFloat`, `dst_reg`, `v_if`/`v_endif`, `Converter`) -- Style A.

The Wormhole and Blackhole implementations are functionally identical. The only difference is the GCC unroll pragma: Wormhole uses `#pragma GCC unroll 8` (fully unrolled) while Blackhole uses `#pragma GCC unroll 0` (no unrolling). The annotated source below shows both variants.

#### Wormhole B0

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_prelu(uint value) { // APPROXIMATION_MODE=false, ITERATIONS=8
    // SFPU microcode
    vFloat init = Converter::as_float(value); // Reinterpret uint32_t param as float (the PReLU slope)

#pragma GCC unroll 8 // Fully unroll the 8-iteration loop on Wormhole
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG
        v_if(a < 0.0f) { a = a * init; } // SFPSETCC (sign check) + SFPPUSHC + SFPMAD (a * init + 0.0) + SFPPOPC
        v_endif; // SFPENCC: restore CC state
        dst_reg[0] = a; // SFPSTORE: write 32 elements back to DEST
        dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}
}  // namespace sfpu
}  // namespace ckernel
```

#### Blackhole

```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_prelu.h

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_prelu(const uint value) { // APPROXIMATION_MODE=false, ITERATIONS=8
    // SFPU microcode
    vFloat init = Converter::as_float(value); // Reinterpret uint32_t param as float (the PReLU slope)

#pragma GCC unroll 0 // No unrolling on Blackhole (loop kept as-is)
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0]; // SFPLOAD: load 32 elements from current DEST row pair into LREG
        v_if(a < 0.0f) { a = a * init; } // SFPSETCC (sign check) + SFPPUSHC + SFPMAD (a * init + 0.0) + SFPPOPC
        v_endif; // SFPENCC: restore CC state
        dst_reg[0] = a; // SFPSTORE: write 32 elements back to DEST
        dst_reg++; // Advance DEST pointer by 1 sfpi row (= 2 physical rows = 32 elements)
    }
}
}  // namespace sfpu
}  // namespace ckernel
```

### SFPU Instructions Used

The following instructions are emitted by the SFPI compiler from the `calculate_prelu` kernel. Since this kernel uses SFPI abstractions, these are compiler-generated rather than explicitly written in source.

| Instruction | Source Abstraction | Description |
|-------------|-------------------|-------------|
| **SFPLOAD** | `vFloat a = dst_reg[0]` | Loads 32 elements (2 physical DEST rows) from the current DEST address into an LREG. Uses IMPLIED format mode which auto-detects FP16_A/FP16_B/FP32 based on the DEST register configuration. |
| **SFPLOADI** | `Converter::as_float(value)` | Loads the immediate slope parameter into an LREG as a float constant. The `uint32_t` parameter is bit-reinterpreted as `float` on the host side; the SFPI compiler loads it via SFPLOADI with FP16_B or similar format encoding. |
| **SFPSETCC** | `v_if(a < 0.0f)` | Sets the per-lane condition code (CC.Res) based on the sign of the value in the LREG. Uses `InstrMod=0` (set CC Result if value is negative), enabling predicated execution for subsequent instructions on lanes where the input is negative. |
| **SFPPUSHC** | `v_if(...)` | Pushes the current CC state (CC.En, CC.Res) onto the CC stack before entering the conditional block. This preserves the outer CC context so it can be restored by `v_endif`. |
| **SFPMAD** | `a = a * init` | Fused multiply-add computing `(a * init) + 0.0`. This is the core PReLU computation: multiplying negative elements by the slope. The addition of 0.0 is the identity addend since there is no separate multiply-only instruction (SFPMUL is an alias of SFPMAD with zero addend). Only executes on lanes where `LaneEnabled` is true (i.e., where `a < 0.0f`). |
| **SFPPOPC** | `v_endif` (partial) | Pops the CC stack to restore the CC state from before the `v_if` block, re-enabling all lanes for subsequent operations. |
| **SFPENCC** | `v_endif` (partial) | Adjusts CC enable/result state to complete the conditional block exit. Restores full lane enablement. |
| **SFPSTORE** | `dst_reg[0] = a` | Stores the 32-element result from the LREG back to the current DEST row pair. Uses IMPLIED format mode matching the load. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST register file** | Source and destination for tile data. Each `dst_reg[0]` access reads/writes 32 elements (2 physical rows x 16 elements/row) at the current DEST address. The address auto-advances via `dst_reg++` within a face, and via `SETRWC`/`inc_dst_addr` between faces. |
| **LREG (local registers)** | The SFPI compiler allocates LREGs for intermediate values. `a` occupies one LREG (holding the loaded DEST value), and `init` occupies another LREG (holding the PReLU slope constant loaded from the `value` parameter). The multiply result is written back to the LREG holding `a`. |
| **CC (Condition Code)** | The per-lane CC register is used for predicated execution. `SFPSETCC` sets `CC.Res` based on the sign bit of `a` (negative = true). `SFPPUSHC` saves CC state to the stack. Inside the `v_if` block, only lanes with `CC.Res=1` (negative values) execute the multiply. `SFPPOPC`/`SFPENCC` restore the CC state afterward. |
| **CC Stack** | One level of CC stack is used: `v_if` pushes the current CC state, and `v_endif` pops it. This is a single-level nesting depth. |

### Address Mode Configuration

The address mode is configured in `eltwise_unary_sfpu_configure_addrmod<SfpuType::prelu>()` during `_llk_math_eltwise_unary_sfpu_init_()`.

**Both Wormhole and Blackhole** set `ADDR_MOD_7` with all-zero increments:

```
ADDR_MOD_7: { .srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0} }
```

Since `SfpuType::prelu` does not match any of the special-case `if constexpr` conditions (which are for `topk_local_sort`, `typecast`, `unary_max`, `unary_min`, etc.), only `ADDR_MOD_7` is configured.

The zero-increment address mode means the SFPU does not auto-increment DEST addresses between SFPLOAD/SFPSTORE pairs via the address mode mechanism. Instead, address progression is handled explicitly:
- **Within a face**: The SFPI `dst_reg++` operator advances the DEST pointer by 1 sfpi row per loop iteration (the stride-2 is implicit in SFPI addressing).
- **Between faces**: The params dispatch function advances the DEST address by 16 sfpi rows (2 x `inc_dst_addr<8>()`) per face transition, equivalent to 32 physical DEST rows = one full 16x16 face.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How do SFPI abstractions like v_if, vFloat, and dst_reg map to actual SFPU instructions?"
   **Reason**: Needed to understand the instruction-level behavior of the SFPI abstractions used in the PReLU kernel (`dst_reg[0]` load/store, `v_if` conditional, `vFloat` multiplication).
   **Key Findings**: SFPI abstractions are compiled by the GCC SFPI backend into SFPU instructions. `dst_reg` load/store maps to SFPLOAD/SFPSTORE, `vFloat * vFloat` generates SFPMAD, and `v_if`/`v_endif` generate CC manipulation instructions (SFPSETCC, SFPPUSHC, SFPPOPC, SFPENCC). The compiler performs instruction combining for optimization.

### Confluence References
1. **Page**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
   **Sections consulted**:
   - **SFPLOAD** (position ~74499): Verified load semantics, format conversion (IMPLIED mode auto-detects FP16/FP32), and addressing via AddrMod.
   - **SFPSTORE** (position ~84000): Verified store semantics and format conversion back to register file format.
   - **SFPMAD** (position ~82000): Confirmed fused multiply-add `(A * B) + C` operation. SFPMUL is an alias with zero addend. IPC=1, latency=2.
   - **SFPLOADI** (position ~79003): Verified immediate load for constant values. Supports FP16_B, FP16_A, UINT16, INT16 formats.
   - **SFPSETCC** (position ~90000): Confirmed CC Result set based on sign (InstrMod=0: set if negative), which is the mechanism behind `v_if(a < 0.0f)`.
   - **SFPPUSHC** (position ~87000): Verified push of CC state to CC stack for nested conditional support.
   - **SFPCOMPC** (position ~92000): Verified conditional complement of CC.Res for else-block support.
   - **SFPENCC** (position ~93000): Verified direct setting of CC.En and CC.Res for block exit.
   - **SFPPOPC** (position ~88000): Verified pop of CC stack to restore previous CC state.

### Glean References
No Glean queries were needed for this analysis. The SFPI abstractions and instruction semantics were fully documented through the Confluence SFPU ISA page and DeepWiki.
