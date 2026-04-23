# Common Compilation Errors and Fixes

This document catalogs error patterns observed during kernel development. It is a **starting point** for debugging — always verify fixes against the actual codebase and architecture docs.

## How to Use This Document

1. Match your error against the patterns below
2. Try the suggested fix
3. If the fix doesn't work, **investigate dynamically**:
   - **Confluence** (primary): Use `mcp__atlassian__getConfluencePage` with page ID `1613201604` (Tensix ISA) for per-instruction details
   - **Existing code**: `grep -r "{symbol}" tt_llk_{target_arch}/ --include="*.h" | head -20`
   - **assembly.yaml** (cross-check): `grep -A 20 "^{INSTRUCTION}:" tt_llk_{target_arch}/instructions/assembly.yaml`

---

## SFPU Kernel Errors

### 1. Undeclared Symbol

**Pattern**: `'X' was not declared in this scope`

**Common causes**:
- Wrong instruction/function name
- Missing include file
- Wrong namespace

**Investigation**:
```bash
# Find where the symbol IS used correctly
grep -r "X" tt_llk_{target_arch}/ --include="*.h" -l
# Check what includes that file uses
head -20 {file_that_uses_X}
```

---

### 2. Not a Member of Namespace

**Pattern**: `'X' is not a member of 'Y'`

**Common causes**:
- Wrong namespace prefix for the symbol
- Symbol exists in a different namespace

**Investigation**:
```bash
# Find the correct namespace for the symbol
grep -r "X" tt_llk_{target_arch}/ --include="*.h" | grep -v "^Binary"
```

---

### 3. Wrong Argument Count

**Pattern**: `too many arguments to function 'X'` or `too few arguments`

**Investigation**:
```bash
# Check the function/macro signature
grep -A 5 "X" tt_llk_{target_arch}/instructions/assembly.yaml
# Or find usage in existing code
grep -r "X(" tt_llk_{target_arch}/ --include="*.h" | head -10
```

---

### 4. Missing Include

**Pattern**: Various symbols undeclared that should be available

**Investigation**:
```bash
# Check what includes similar kernels use
head -10 tt_llk_{target_arch}/common/inc/sfpu/ckernel_sfpu_*.h
# Or for non-SFPU
head -10 tt_llk_{target_arch}/llk_lib/llk_math_*.h
```

---

### 5. Deprecated/Renamed Instruction

**Pattern**: Instruction exists on reference arch but not on target

**Investigation**:
```bash
# Verify the instruction doesn't exist
grep -c "^{INSTRUCTION}:" tt_llk_{target_arch}/instructions/assembly.yaml

# Search for similar instruction names
grep -i "{partial_name}" tt_llk_{target_arch}/instructions/assembly.yaml

# Check how existing code handles this operation
grep -r "{operation_concept}" tt_llk_{target_arch}/ --include="*.h"
```

---

### 6. Missing Include File

**Error Message**:
```
error: 'sfpi_fp16.h' file not found
```
or
```
error: 's2vFloat16b' was not declared in this scope
```

**Fix**: Ensure these includes are at the top of your SFPU kernel:
```cpp
#include "sfpi.h"
#include "sfpi_fp16.h"
#include "ckernel_sfpu_load_config.h"
```

---

### 7. Wrong LUT Function

**Error Message**:
```
error: 'lut2' was not declared in this scope
```

**Fix**:
1. Include `"sfpi.h"`
2. Ensure LUT registers are saved at function start:
```cpp
sfpi::vUInt l0 = sfpi::l_reg[sfpi::LRegs::LReg0];
sfpi::vUInt l1 = sfpi::l_reg[sfpi::LRegs::LReg1];
// ... etc
```

---

### 8. Wrong Constant Loading

**Error Message**:
```
error: '_sfpu_load_imm32_' was not declared in this scope
```

**Fix**: Add the include:
```cpp
#include "ckernel_sfpu_load_config.h"
```

---

### 9. Wrong v_if Syntax

**Error Message**:
```
error: expected ')' before 'v_endif'
```

**Fix**: Ensure proper v_if/v_endif syntax with braces:
```cpp
// Correct
v_if (val < 0.0f)
{
    val = 0.0f;
}
v_endif;
```

---

### 10. Type Mismatch

**Error Message**:
```
error: cannot convert 'sfpi::vFloat' to 'sfpi::vInt'
```

**Fix**: Use explicit conversion:
```cpp
sfpi::vInt exp = exexp(val);  // Extract exponent
// or
sfpi::vInt bits = sfpi::reinterpret<sfpi::vInt>(val);
```

---

### 11. Missing Template Parameter

**Error Message**:
```
error: no matching function for call to '_calculate_sigmoid_<>'
```

**Fix**: Provide all template parameters:
```cpp
_calculate_sigmoid_<APPROXIMATION_MODE, ITERATIONS>(iterations);
```

---

### 12. Undefined Builtin Function

**Error Message**:
```
error: '_sfpu_exp_' was not declared in this scope
```

**Fix**: Include the file containing the helper:
```cpp
#include "ckernel_sfpu_exp.h"  // For _sfpu_exp_
#include "ckernel_sfpu_recip.h"  // For _sfpu_reciprocal_
```

---

### 13. TTI Instruction Errors

**Error Message**:
```
error: 'TTI_SFPLOADMACRO' was not declared in this scope
```

**Fix**: Add the required include:
```cpp
#include "ckernel_ops.h"
```

---

### 14. Linker Errors with Helper Functions

**Error Message**:
```
undefined reference to '_sfpu_reciprocal_<2>(sfpi::vFloat)'
```

**Fix**:
1. Ensure helper function is `inline` or `sfpi_inline`
2. Check that the include is present
3. Verify template arguments match

---

## Unpack/Pack Kernel Errors

### 15. TENSIX TIMEOUT on Unpacker

**Error Message**:
```
TENSIX TIMED OUT - waited 2 seconds for Unpacker
```

**Cause**: Wrong MOP configuration, incorrect tile dimensions, or wrong instruction sequence.

**Common Fixes**:
1. Check MOP loop counts (`outerloop` and `innerloop` values)
2. Verify tile dimensions (`Tile_x_dim`, `Tile_z_dim`) are configured
3. Use `TTI_STALLWAIT` + `TTI_WRCFG` instead of `TTI_REG2FLOP` — verify which pattern your target uses:
   ```bash
   grep -c "TTI_WRCFG" tt_llk_{target_arch}/llk_lib/*.h
   grep -c "TTI_REG2FLOP" tt_llk_{target_arch}/llk_lib/*.h
   ```
4. Check context-based addressing (use `unp_cfg_context`)
5. Verify unpacker x-end configuration

---

### 16. Wrong Config Write Sequence

**Symptom**: Kernel compiles but hangs or produces wrong results.

**Cause**: Using the wrong config write pattern for the target architecture. Different architectures may use different patterns (e.g., `TTI_REG2FLOP` vs `TTI_STALLWAIT` + `TTI_WRCFG`).

**Investigation**:
```bash
# Check which pattern existing target kernels use
grep -r "TTI_WRCFG\|TTI_REG2FLOP" tt_llk_{target_arch}/llk_lib/ --include="*.h" | head -20
```

---

### 17. Missing Replay Buffer

**Error Message**:
```
error: 'load_replay_buf' was not declared in this scope
```

**Fix**: Add include and check existing code for the correct replay pattern:
```bash
grep -r "replay" tt_llk_{target_arch}/ --include="*.h" -l
```

---

### 18. Wrong Template Type

**Symptom**: Kernel compiles but produces wrong results or hangs.

**Cause**: Using `ckernel_template` where `ckernel_unpack_template` (or vice versa) is needed.

**Investigation**:
```bash
# Check what template type similar kernels use
grep -r "ckernel_template\|ckernel_unpack_template" tt_llk_{target_arch}/llk_lib/llk_unpack_*.h
```

---

### 19. Missing Tile Dimension Configuration

**Symptom**: Kernel hangs or reads/writes wrong memory locations.

**Cause**: Tilize/untilize modes require explicit tile dimension setup.

**Investigation**:
```bash
# Check how existing kernels configure tile dimensions
grep -r "Tile_x_dim\|Tile_z_dim" tt_llk_{target_arch}/llk_lib/ --include="*.h" | head -10
```

---

### 20. Wrong Context Addressing

**Symptom**: Every other tile is wrong or kernel produces inconsistent results.

**Cause**: Not using context-based register selection.

**Investigation**:
```bash
# Check context addressing patterns in existing code
grep -r "unp_cfg_context" tt_llk_{target_arch}/llk_lib/ --include="*.h" | head -10
```

---

### 21. CFGSHIFTMASK Not Working

**Symptom**: Address auto-increment in replay buffer not working.

**Cause**: SCRATCH register not initialized with increment value.

**Investigation**:
```bash
# Check how existing code initializes CFGSHIFTMASK
grep -B5 -A10 "CFGSHIFTMASK" tt_llk_{target_arch}/llk_lib/ --include="*.h" -r
```

---

## Unpack Kernel Debugging Checklist

Before submitting for functional testing, verify:

- [ ] Uses the correct config write pattern for target arch (check existing kernels)
- [ ] Has `TTI_STALLWAIT` before config writes if the target requires it
- [ ] Uses context-based addressing where needed (`unp_cfg_context`)
- [ ] Tile dimensions configured (for tilize/untilize modes)
- [ ] Unpacker x-end configured
- [ ] Uses correct template type (`ckernel_template` vs `ckernel_unpack_template`)
- [ ] Replay buffer initialized if using `CFGSHIFTMASK`
- [ ] Unused parameters marked with `[[maybe_unused]]`
- [ ] `LLK_ASSERT` validations for parameter constraints
- [ ] Checked similar existing target kernels for patterns

---

## Runtime / Test Debugging

### All zeros in Dest with unpack_to_dest

**Symptom**: Test produces all-zero output when golden expects non-zero values. Typically fails when there is a bit-width mismatch between input format and Dest accumulation mode.

**Cause**: `unpack_to_dest` writes directly from the unpacker to the Dest register, bypassing the FPU. It only works when the format bit-width matches the Dest mode. When there is a mismatch (e.g., 16-bit data → 32-bit Dest, or 32-bit data → 16-bit Dest), the unpacker cannot do the format conversion — the FPU/datacopy path (Mov2D/ELWADD) is needed. Without it, Dest gets zeros.

**Fix**: Use `unpack_to_dest` only when format bit-width matches Dest mode:
- Non-32-bit + `dest_acc=No` → `unpack_to_dest=True` (16-bit → 16-bit Dest)
- 32-bit + `dest_acc=Yes` → `unpack_to_dest=True` (32-bit → 32-bit Dest)
- Non-32-bit + `dest_acc=Yes` → `unpack_to_dest=False` (16-bit → 32-bit mismatch, needs FPU)
- 32-bit + `dest_acc=No` → `unpack_to_dest=False` (32-bit → 16-bit mismatch, needs FPU)

In Python test: `unpack_to_dest = (formats.input_format.is_32_bit() == (dest_acc == DestAccumulation.Yes))`

---

## General Debugging Tips

1. **Start with the compiler suggestion** — "did you mean X?" is usually correct
2. **Compare with working code** — the most reliable fix source is existing implementations
3. **One fix at a time** — change one thing, recompile, check
4. **Structural vs. superficial** — if individual fixes keep failing, the problem may be structural (wrong includes, wrong namespace setup, wrong function pattern). Compare the full file against a similar working kernel.
5. **Check the spec** — the error may trace back to a wrong instruction mapping in the spec
6. **Use verbose mode**: `check_compile.py -v` shows full compiler output
7. **Check namespace**: Most SFPI symbols are in `sfpi::` namespace
8. **Verify includes order**: Some headers depend on others being included first
