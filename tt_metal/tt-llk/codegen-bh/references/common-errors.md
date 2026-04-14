# Common Compilation Errors and Fixes

This document catalogs common errors encountered when writing Blackhole SFPU kernels and how to fix them.

## Running Compilation Check

```bash
cd codegen-bh
source ../tests/.venv/bin/activate
PYTHONPATH=.. python scripts/check_compile.py ../tt_llk_blackhole/common/inc/sfpu/your_kernel.h -v
```

---

## Error Categories

### 1. Undefined SFPI Symbol

**Error Message**:
```
error: 'dst_reg' is not a member of 'sfpi'
```

**Cause**: Missing SFPI include or wrong namespace.

**Fix**: Add the required include:
```cpp
#include "sfpi.h"
```

---

### 2. Missing Include File

**Error Message**:
```
error: 'sfpi_fp16.h' file not found
```
or
```
error: 's2vFloat16b' was not declared in this scope
```

**Cause**: Missing required include files.

**Fix**: Ensure these includes are at the top of your file:
```cpp
#include "sfpi.h"
#include "sfpi_fp16.h"
#include "ckernel_sfpu_load_config.h"
```

---

### 3. Wrong LUT Function

**Error Message**:
```
error: 'lut2' was not declared in this scope
```

**Cause**: LUT functions require specific includes and LUT register setup.

**Fix**:
1. Include the correct header:
```cpp
#include "sfpi.h"
```

2. Ensure LUT registers are saved at function start:
```cpp
sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
// ... etc
```

---

### 4. Wrong Constant Loading

**Error Message**:
```
error: '_sfpu_load_imm32_' was not declared in this scope
```

**Cause**: Missing include for load config.

**Fix**: Add the include:
```cpp
#include "ckernel_sfpu_load_config.h"
```

---

### 5. Wrong v_if Syntax

**Error Message**:
```
error: expected ')' before 'v_endif'
```
or
```
error: 'v_endif' was not declared in this scope
```

**Cause**: Missing semicolon or wrong conditional syntax.

**Fix**: Ensure proper v_if/v_endif syntax:
```cpp
// Wrong
v_if (val < 0.0f)
    val = 0.0f;
v_endif;

// Correct
v_if (val < 0.0f)
{
    val = 0.0f;
}
v_endif;
```

---

### 6. Type Mismatch

**Error Message**:
```
error: cannot convert 'sfpi::vFloat' to 'sfpi::vInt'
```

**Cause**: Trying to assign float to int without explicit conversion.

**Fix**: Use reinterpret or proper type conversion:
```cpp
// Wrong
sfpi::vInt exp = val;

// Correct
sfpi::vInt exp = exexp(val);  // Extract exponent
// or
sfpi::vInt bits = sfpi::reinterpret<sfpi::vInt>(val);
```

---

### 7. Missing Template Parameter

**Error Message**:
```
error: no matching function for call to '_calculate_sigmoid_<>'
```

**Cause**: Template function called without required template arguments.

**Fix**: Provide all template parameters:
```cpp
// Wrong
_calculate_sigmoid_(iterations);

// Correct
_calculate_sigmoid_<APPROXIMATION_MODE, ITERATIONS>(iterations);
```

---

### 8. Undefined Builtin Function

**Error Message**:
```
error: '_sfpu_exp_' was not declared in this scope
```

**Cause**: Using a helper function that's defined in another file.

**Fix**: Include the file containing the helper:
```cpp
#include "ckernel_sfpu_exp.h"  // For _sfpu_exp_
#include "ckernel_sfpu_recip.h"  // For _sfpu_reciprocal_
```

---

### 9. Wrong Namespace

**Error Message**:
```
error: 'LReg0' is not a member of 'sfpi::LRegs'
```

**Cause**: Wrong case or namespace for LUT register constants.

**Fix**: Use correct case:
```cpp
// Wrong
sfpi::LRegs::LReg0
sfpi::LRegs::LREG0

// Correct
sfpi::LRegs::LReg0
```

---

### 10. TTI Instruction Errors

**Error Message**:
```
error: 'TTI_SFPLOADMACRO' was not declared in this scope
```

**Cause**: TTI instructions require specific includes.

**Fix**: Add the required include:
```cpp
#include "ckernel_ops.h"
```

---

### 11. Wrong Programmable Constant

**Error Message**:
```
error: 'vConstFloatPrgm3' is not a member of 'sfpi'
```

**Cause**: Using a programmable constant that doesn't exist.

**Fix**: Only these programmable constants are available:
```cpp
sfpi::vConstFloatPrgm0
sfpi::vConstFloatPrgm1
sfpi::vConstFloatPrgm2
// vConstFloatPrgm3+ may not be available
```

---

### 12. Missing v_and for Nested Conditions

**Error Message**:
```
error: expected ';' before 'v_and'
```

**Cause**: v_and must be inside a v_if block.

**Fix**: Use v_and inside an existing v_if:
```cpp
v_if (exp >= 0)
{
    for (int s_iter = 0; s_iter < 7; s_iter++)
    {
        exp = exp - 1;
        v_and(exp >= 0);  // Narrow predication
        val = val * val;
    }
}
v_endif;
```

---

### 13. Wrong Unroll Pragma

**Error Message**:
```
warning: ignoring #pragma GCC unroll [-Wunknown-pragmas]
```

**Cause**: Unroll pragma should be immediately before for loop.

**Fix**: Place pragma directly before the loop:
```cpp
// Wrong
#pragma GCC unroll 8

// Some other code

for (int d = 0; d < ITERATIONS; d++) { ... }

// Correct
#pragma GCC unroll 8
for (int d = 0; d < ITERATIONS; d++) { ... }
```

---

### 14. Linker Errors with Helper Functions

**Error Message**:
```
undefined reference to '_sfpu_reciprocal_<2>(sfpi::vFloat)'
```

**Cause**: Helper function template not instantiated.

**Fix**:
1. Ensure helper function is `inline` or `sfpi_inline`
2. Check that the include is present
3. Verify template arguments match

---

## Quick Reference: Valid SFPI Syntax

```cpp
// Load from dest
sfpi::vFloat val = sfpi::dst_reg[0];

// Store to dest
sfpi::dst_reg[0] = result;

// Increment
sfpi::dst_reg++;

// Arithmetic
sfpi::vFloat sum = a + b;
sfpi::vFloat prod = a * b;
sfpi::vFloat neg = -x;

// Constants
sfpi::vFloat c = sfpi::s2vFloat16b(0.5f);
sfpi::vFloat one = sfpi::vConst1;

// Conditionals
v_if (val < 0.0f) { val = 0.0f; } v_endif;

// LUT
sfpi::vFloat result = lut2(val, l0, l1, l2, l4, l5, l6, mode);

// Exponent/mantissa operations
sfpi::vInt exp = exexp(val);
val = setexp(val, 126);
val = setsgn(val, 0);

// Programmable constants
sfpi::vConstFloatPrgm0 = 0.5f;  // In init function
sfpi::vFloat half = sfpi::vConstFloatPrgm0;  // In calculation

// Load immediates
_sfpu_load_imm32_(0, 0x32F433D9);
_sfpu_load_imm16_(0, 0x28FF);
```

---

## Debugging Tips

1. **Start simple**: Get a minimal operation working first (just load + store)

2. **Check existing examples**: Compare with working kernels in `tt_llk_blackhole/common/inc/sfpu/`

3. **Use verbose mode**: `check_compile.py -v` shows full compiler output

4. **One change at a time**: When fixing errors, change one thing and recompile

5. **Read the suggestions**: Compiler often suggests correct name (e.g., "did you mean...?")

6. **Check namespace**: Most SFPI symbols are in `sfpi::` namespace

7. **Verify includes order**: Some headers depend on others being included first

8. **Test with simple data**: Use known values to verify computation logic

---

## Unpack/Pack Kernel Errors (Blackhole-Specific)

### 15. TENSIX TIMEOUT on Unpacker

**Error Message**:
```
TENSIX TIMED OUT - waited 2 seconds for Unpacker
```

**Cause**: Wrong MOP configuration, incorrect tile dimensions, or wrong instruction sequence.

**Common Fixes**:
1. Check MOP loop counts (`outerloop` and `innerloop` values)
2. Verify tile dimensions (`Tile_x_dim`, `Tile_z_dim`) are configured
3. Use `TTI_STALLWAIT` + `TTI_WRCFG` instead of `TTI_REG2FLOP`
4. Check context-based addressing (use `unp_cfg_context`)
5. Verify unpacker x-end configuration

---

### 16. Wrong Config Write Sequence

**Symptom**: Kernel compiles but hangs or produces wrong results.

**Cause**: Using Wormhole `TTI_REG2FLOP` pattern instead of Blackhole `TTI_WRCFG` pattern.

**Fix**:
```cpp
// WRONG (Wormhole pattern)
TTI_REG2FLOP(1, 0, 0, 0, THCON_SEC0_REG2_Out_data_format_ADDR32 - THCON_CFGREG_BASE_ADDR32, p_gpr_unpack::TMP0);

// CORRECT (Blackhole pattern)
TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
TTI_WRCFG(p_gpr_unpack::TMP0, p_cfg::WRCFG_32b, THCON_SEC0_REG2_Out_data_format_ADDR32);
```

---

### 17. Missing Replay Buffer

**Error Message**:
```
error: 'load_replay_buf' was not declared in this scope
```

**Cause**: Missing include or using wrong MOP approach.

**Fix**: Add include and use replay buffer pattern:
```cpp
#include "ckernel.h"  // Contains load_replay_buf

load_replay_buf(0, replay_len, [] {
    // TTI instructions here
});
```

---

### 18. Wrong Template Type

**Symptom**: Kernel compiles but produces wrong results or hangs.

**Cause**: Using `ckernel_template` where `ckernel_unpack_template` is needed.

**Fix**:
```cpp
// For context-aware unpacking with zmask
ckernel_unpack_template tmp = ckernel_unpack_template(
    false,  // src B
    false,  // halo
    lltt::replay_insn(0, len),  // context 0
    0, 0, 0,
    lltt::replay_insn(len, len),  // context 1
    0, 0);
```

---

### 19. Missing Tile Dimension Configuration

**Symptom**: Kernel hangs or reads/writes wrong memory locations.

**Cause**: Tilize/untilize modes require explicit tile dimension setup.

**Fix**:
```cpp
// Set tile X dimension
const std::uint32_t Tile_x_dim = face_r_dim * num_faces * FACE_C_DIM;
cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(Tile_x_dim | (Tile_x_dim << 16));

// Set tile Z dimension (1 for tilize, num_faces for standard)
cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 0, 0xffff0000>(0 | (Tile_z_dim << 16));

// Set unpacker x-end
TT_SETADCXX(p_setadc::UNP0, Tile_x_dim - 1, 0x0);
```

---

### 20. Wrong Context Addressing

**Symptom**: Every other tile is wrong or kernel produces inconsistent results.

**Cause**: Not using context-based register selection.

**Fix**:
```cpp
// Context-based address configuration
if (0 == unp_cfg_context)
{
    cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address;
}
else
{
    cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address;
}
```

---

### 21. CFGSHIFTMASK Not Working

**Symptom**: Address auto-increment in replay buffer not working.

**Cause**: SCRATCH register not initialized with increment value.

**Fix**:
```cpp
// In init: Set scratch register with increment value
const std::uint32_t c_dim_size = SCALE_DATUM_SIZE(src_format, dim) >> 4;
TT_SETDMAREG(0, LOWER_HALFWORD(c_dim_size), 0, LO_16(p_gpr_unpack::TMP0));
TT_SETDMAREG(0, UPPER_HALFWORD(c_dim_size), 0, HI_16(p_gpr_unpack::TMP0));
TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::THCON);
TTI_WRCFG(p_gpr_unpack::TMP0, 0, SCRATCH_SEC0_val_ADDR32);
TTI_NOP;
```

---

## Unpack Kernel Debugging Checklist

Before submitting for functional testing, verify:

- [ ] Uses `TTI_WRCFG` (not `TTI_REG2FLOP`) for config writes
- [ ] Has `TTI_STALLWAIT` before each `TTI_WRCFG`
- [ ] Uses `unp_cfg_context` for context-based addressing
- [ ] Tile dimensions configured (for tilize/untilize modes)
- [ ] Unpacker x-end configured
- [ ] Uses correct template type (`ckernel_template` vs `ckernel_unpack_template`)
- [ ] Replay buffer initialized if using `CFGSHIFTMASK`
- [ ] Unused parameters marked with `[[maybe_unused]]`
- [ ] `LLK_ASSERT` validations for parameter constraints
- [ ] Checked similar existing Blackhole kernels for patterns
