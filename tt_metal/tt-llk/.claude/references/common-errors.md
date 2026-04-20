# Common Errors and Investigation Patterns

Error patterns observed during LLK kernel development. This is a starting point — always verify fixes against the actual codebase.

## How to Use

1. Match your error against the patterns below
2. Run the investigation commands (replace `{target_arch}` with your architecture)
3. If the pattern doesn't match, investigate dynamically:
   - Existing code: `Grep for "{symbol}" in tt_llk_{target_arch}/ (*.h files)`
   - assembly.yaml: `Grep for "^{INSTRUCTION}:" in tt_llk_{target_arch}/instructions/assembly.yaml`

---

## SFPU Kernel Errors

### Undeclared Symbol
**Pattern**: `'X' was not declared in this scope`

```
# Find where the symbol IS used correctly
Grep for "X" in tt_llk_{target_arch}/ (*.h files) to find matching files
# Check what includes that file uses
Read the file that uses X
```

### Not a Member of Namespace
**Pattern**: `'X' is not a member of 'Y'`

```
Grep for "X" in tt_llk_{target_arch}/ (*.h files)
```

### Missing Include
**Pattern**: Various symbols undeclared that should be available

Common SFPU includes:
```cpp
#include "sfpi.h"
#include "sfpi_fp16.h"
#include "ckernel_sfpu_load_config.h"
```

```
# Check what includes similar kernels use
Glob for tt_llk_{target_arch}/common/inc/sfpu/ckernel_sfpu_*.h, then Read the top of each file
```

### Wrong v_if Syntax
**Pattern**: `expected ')' before 'v_endif'`

Must use braces:
```cpp
v_if (val < 0.0f) { val = 0.0f; } v_endif;
```

### Type Mismatch
**Pattern**: `cannot convert 'sfpi::vFloat' to 'sfpi::vInt'`

Use explicit conversion:
```cpp
sfpi::vInt exp = exexp(val);
sfpi::vInt bits = sfpi::reinterpret<sfpi::vInt>(val);
```

### Deprecated/Renamed Instruction
**Pattern**: Instruction exists on reference arch but not on target

```
# Verify the instruction doesn't exist
Grep for "^{INSTRUCTION}:" in tt_llk_{target_arch}/instructions/assembly.yaml
# Search for similar names
Grep (case-insensitive) for "{partial_name}" in tt_llk_{target_arch}/instructions/assembly.yaml
# Check how existing code handles this
Grep for "{operation_concept}" in tt_llk_{target_arch}/ (*.h files)
```

---

## Instruction Encoding Errors

### TTI_ vs TT_OP_ Confusion

**This is one of the most common error sources.**

| Macro | Behavior | When to Use |
|-------|----------|-------------|
| `TTI_*` | Executes immediately (inline asm) | Inside `load_replay_buf` lambdas, direct execution |
| `TT_OP_*` | Returns encoding (constexpr) | For `ckernel_template` constructors |

**WRONG**: `ckernel_template tmp(1, 4, TTI_UNPACR(...), ...);`
**CORRECT**: `static constexpr auto op = TT_OP_UNPACR(...); ckernel_template tmp(1, 4, op, ...);`

**Error**: `impossible constraint in 'asm'` — usually means a TTI_ macro is used where TT_OP_ is needed.

### Boolean Instead of Constant
**WRONG**: `TT_OP_UNPACR_NOP(..., pool_type == PoolType::MAX, ...);`
**CORRECT**: Use `p_unpacr_nop::CLR_SRC_NEGINF` or `p_unpacr_nop::CLR_SRC_0`

```
# Find the correct constants
Grep for "p_unpacr_nop|p_pacr" in tt_llk_{target_arch}/common/inc/ (*.h files)
```

### Wrong Namespace
Common mistakes:
```
# Find the correct namespace for a symbol
Grep for "{symbol}" in tt_llk_{target_arch}/common/inc/ (*.h files)
```

| Wrong | Correct (verify on your target) |
|-------|---------|
| `Srcs::SrcA` | `SrcA` (unqualified) — verify: Grep for "SrcA" in `tt_llk_{target_arch}/common/inc/` (*.h files) |
| `Srcs::SrcA` in UNPACR_NOP | `p_unpacr_nop::UNP0` — verify in existing code |

---

## Unpack/Pack Kernel Errors

### TIMEOUT on Unpacker/Packer
**Pattern**: `TENSIX TIMED OUT - waited 2 seconds for Unpacker`

```
# Check MOP loop counts in similar working kernels
Grep for "outerloop|innerloop" in tt_llk_{target_arch}/llk_lib/llk_{type}_*.h

# Check config write pattern
Grep for "TTI_WRCFG|TTI_REG2FLOP" in tt_llk_{target_arch}/llk_lib/ (*.h files)

# Verify tile dimension configuration
Grep for "Tile_x_dim|Tile_z_dim" in tt_llk_{target_arch}/llk_lib/ (*.h files)
```

### Wrong Config Write Pattern
**Symptom**: Kernel compiles but hangs or produces wrong results.

Different architectures use different config write patterns:
```
Grep (count mode) for "TTI_WRCFG" in tt_llk_{target_arch}/llk_lib/ (*.h files)
Grep (count mode) for "TTI_REG2FLOP" in tt_llk_{target_arch}/llk_lib/ (*.h files)
```

### Wrong Template Type
**Symptom**: Compiles but hangs or wrong results.

```
Grep for "ckernel_template|ckernel_unpack_template" in tt_llk_{target_arch}/llk_lib/llk_unpack_*.h
```

### Missing Replay Buffer
**Pattern**: `'load_replay_buf' was not declared in this scope`

```
Grep for "replay" in tt_llk_{target_arch}/ (*.h files)
```

### CFGSHIFTMASK Not Working
**Symptom**: Address auto-increment in replay buffer not working.

```
Grep for "CFGSHIFTMASK" in tt_llk_{target_arch}/llk_lib/ (*.h files) with surrounding context
```

---

## Runtime Errors

### All Zeros in Dest (unpack_to_dest)
**Symptom**: All-zero output when golden expects non-zero. Happens with format/Dest bit-width mismatch.

`unpack_to_dest` only works when format bit-width matches Dest mode:
- Non-32-bit + `dest_acc=No` → `unpack_to_dest=True` (16b → 16b Dest)
- 32-bit + `dest_acc=Yes` → `unpack_to_dest=True` (32b → 32b Dest)
- Mismatch → `unpack_to_dest=False` (needs FPU/datacopy path)

### Reconfig Escape
**Symptom**: Test passes alone, fails when run after another test.

```
# Check init/uninit symmetry
Grep for "_init_" in tt_llk_{target_arch}/llk_lib/llk_{type}_{op}.h with context
Grep for "_uninit_" in tt_llk_{target_arch}/llk_lib/llk_{type}_{op}.h with context
```

Every register modified in `_init_` must be restored in `_uninit_`. Do NOT reset the device — it masks the bug.

---

## Debugging Checklist

### Before Submitting for Testing
- [ ] Function signatures match test harness (`tests/sources/*{op}*.cpp`)
- [ ] Uses correct instruction conventions for target arch (check existing kernels)
- [ ] Init/uninit symmetry maintained
- [ ] Unused parameters marked `[[maybe_unused]]`
- [ ] Checked similar existing kernels for patterns

### General Tips
1. **Start with the compiler suggestion** — "did you mean X?" is usually correct
2. **Compare with working code** — existing implementations are the most reliable reference
3. **One fix at a time** — change one thing, recompile, verify
4. **Structural vs superficial** — if individual fixes keep failing, compare the full file against a similar working kernel
5. **Use verbose mode** — `QUIET=0` to see full output
