# Common Errors and Investigation Patterns

Error patterns observed during LLK kernel development. This is a starting point — always verify fixes against the actual codebase.

## How to Use

1. Match your error against the patterns below
2. Run the investigation commands (replace `{target_arch}` with your architecture)
3. If the pattern doesn't match, investigate dynamically:
   - Existing code: `grep -r "{symbol}" tt_llk_{target_arch}/ --include="*.h" | head -20`
   - assembly.yaml: `grep -A 20 "^{INSTRUCTION}:" tt_llk_{target_arch}/instructions/assembly.yaml`

---

## SFPU Kernel Errors

### Undeclared Symbol
**Pattern**: `'X' was not declared in this scope`

```bash
# Find where the symbol IS used correctly
grep -r "X" tt_llk_{target_arch}/ --include="*.h" -l
# Check what includes that file uses
head -20 {file_that_uses_X}
```

### Not a Member of Namespace
**Pattern**: `'X' is not a member of 'Y'`

```bash
grep -r "X" tt_llk_{target_arch}/ --include="*.h" | grep -v "^Binary"
```

### Missing Include
**Pattern**: Various symbols undeclared that should be available

Common SFPU includes:
```cpp
#include "sfpi.h"
#include "sfpi_fp16.h"
#include "ckernel_sfpu_load_config.h"
```

```bash
# Check what includes similar kernels use
head -10 tt_llk_{target_arch}/common/inc/sfpu/ckernel_sfpu_*.h
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

```bash
# Verify the instruction doesn't exist
grep -c "^{INSTRUCTION}:" tt_llk_{target_arch}/instructions/assembly.yaml
# Search for similar names
grep -i "{partial_name}" tt_llk_{target_arch}/instructions/assembly.yaml
# Check how existing code handles this
grep -r "{operation_concept}" tt_llk_{target_arch}/ --include="*.h"
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

```bash
# Find the correct constants
grep -r "p_unpacr_nop\|p_pacr" tt_llk_{target_arch}/common/inc/ --include="*.h" | head -20
```

### Wrong Namespace
Common mistakes:
```bash
# Find the correct namespace for a symbol
grep -rn "{symbol}" tt_llk_{target_arch}/common/inc/ --include="*.h" | head -10
```

| Wrong | Correct (verify on your target) |
|-------|---------|
| `Srcs::SrcA` | `SrcA` (unqualified) — verify: `grep "SrcA" tt_llk_{target_arch}/common/inc/*.h` |
| `Srcs::SrcA` in UNPACR_NOP | `p_unpacr_nop::UNP0` — verify in existing code |

---

## Unpack/Pack Kernel Errors

### TIMEOUT on Unpacker/Packer
**Pattern**: `TENSIX TIMED OUT - waited 2 seconds for Unpacker`

```bash
# Check MOP loop counts in similar working kernels
grep -A10 "outerloop\|innerloop" tt_llk_{target_arch}/llk_lib/llk_{type}_*.h | head -30

# Check config write pattern
grep -r "TTI_WRCFG\|TTI_REG2FLOP" tt_llk_{target_arch}/llk_lib/ --include="*.h" | head -20

# Verify tile dimension configuration
grep -r "Tile_x_dim\|Tile_z_dim" tt_llk_{target_arch}/llk_lib/ --include="*.h" | head -10
```

### Wrong Config Write Pattern
**Symptom**: Kernel compiles but hangs or produces wrong results.

Different architectures use different config write patterns:
```bash
grep -c "TTI_WRCFG" tt_llk_{target_arch}/llk_lib/*.h
grep -c "TTI_REG2FLOP" tt_llk_{target_arch}/llk_lib/*.h
```

### Wrong Template Type
**Symptom**: Compiles but hangs or wrong results.

```bash
grep -r "ckernel_template\|ckernel_unpack_template" tt_llk_{target_arch}/llk_lib/llk_unpack_*.h
```

### Missing Replay Buffer
**Pattern**: `'load_replay_buf' was not declared in this scope`

```bash
grep -r "replay" tt_llk_{target_arch}/ --include="*.h" -l
```

### CFGSHIFTMASK Not Working
**Symptom**: Address auto-increment in replay buffer not working.

```bash
grep -B5 -A10 "CFGSHIFTMASK" tt_llk_{target_arch}/llk_lib/ --include="*.h" -r
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

```bash
# Check init/uninit symmetry
grep -A20 "_init_" tt_llk_{target_arch}/llk_lib/llk_{type}_{op}.h
grep -A20 "_uninit_" tt_llk_{target_arch}/llk_lib/llk_{type}_{op}.h
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
