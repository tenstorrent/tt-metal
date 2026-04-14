# Porting Guide — Cross-Architecture Kernel Porting

This guide provides the methodology for porting LLK kernels between Wormhole B0, Blackhole, and Quasar. It teaches HOW to investigate and port — it does not hardcode architectural knowledge.

## General Principle

**Discover patterns from the target architecture. Never assume the source architecture's patterns apply.**

Use the source implementation only for SEMANTICS (what the kernel does). Use the target architecture's existing kernels for IMPLEMENTATION (how it does it).

---

## Step 1: Understand the Source Algorithm

Read the source implementation and extract:
- **Purpose**: What does this kernel compute?
- **Algorithm**: Step-by-step logic (pseudocode level)
- **Template parameters**: What configurations are supported?
- **Function signatures**: init, main, uninit entry points
- **Data format handling**: Which formats are supported, any special paths?
- **Dependencies**: Other files/functions used

```bash
# Find the source kernel
ls tt_llk_{source_arch}/llk_lib/llk_{type}_{op}*.h
ls tt_llk_{source_arch}/common/inc/sfpu/ckernel_sfpu_{op}.h

# Read it
cat tt_llk_{source_arch}/llk_lib/llk_{type}_{op}.h
```

---

## Step 2: Research the Target Architecture

### 2a: Find the Closest Existing Target Kernel

```bash
# List all kernels of the same type on the target
ls tt_llk_{target_arch}/llk_lib/llk_{type}_*.h

# For SFPU
ls tt_llk_{target_arch}/common/inc/sfpu/ckernel_sfpu_*.h
```

Read the closest existing kernel LINE BY LINE. Document:
- MOP structure (outer/inner loops)
- Instruction patterns (macros, parameter counts)
- Config write patterns
- Address modifiers
- Replay buffer usage
- Counter resets
- Init/uninit symmetry

### 2b: Check the Test Harness (API Contract)

The test file defines what function signatures the target expects:

```bash
# Find the test source
ls tests/sources/*{op}*.cpp

# Extract target-arch-specific signatures
grep -A20 "ARCH_{TARGET}" tests/sources/*{op}*.cpp
```

**If the test harness and source implementation disagree on signatures, the test harness wins.**

### 2c: Check the Parent/Caller File

```bash
grep -r "#include.*{op}" tt_llk_{target_arch}/llk_lib/ --include="*.h"
```

Read wrapper functions, verify template params, find helper functions.

### 2d: Check assembly.yaml for Instruction Availability

```bash
# Verify the instruction exists on the target
grep -c "^{INSTRUCTION}:" tt_llk_{target_arch}/instructions/assembly.yaml

# Get full instruction definition
grep -A 50 "^{INSTRUCTION}:" tt_llk_{target_arch}/instructions/assembly.yaml

# If instruction doesn't exist, find alternatives
grep -i "{partial_name}" tt_llk_{target_arch}/instructions/assembly.yaml
```

---

## Step 3: Map Constructs

### Instruction Macros

Check which macro style the target uses:
```bash
grep -c "TTI_WRCFG" tt_llk_{target_arch}/llk_lib/*.h
grep -c "TTI_REG2FLOP" tt_llk_{target_arch}/llk_lib/*.h
```

### Template Types

```bash
grep -r "ckernel_template\|ckernel_unpack_template" tt_llk_{target_arch}/llk_lib/llk_{type}_*.h
```

### Replay Buffer

```bash
grep -r "load_replay_buf\|replay" tt_llk_{target_arch}/llk_lib/ --include="*.h" -l
```

### Config Writes

```bash
grep -r "TTI_STALLWAIT.*TTI_WRCFG\|TTI_REG2FLOP" tt_llk_{target_arch}/llk_lib/ --include="*.h" | head -10
```

---

## Step 4: Handle Architecture-Specific Differences

### WH vs BH/QSR Common Differences

Discover these dynamically — do NOT assume:
```bash
# Check PACR parameter count
grep -r "TT_OP_PACR\|TTI_PACR" tt_llk_{target_arch}/llk_lib/ --include="*.h" | head -5

# Check SFPU instruction availability
grep "^SFPLOADMACRO:" tt_llk_{target_arch}/instructions/assembly.yaml
grep "^SFPSHFT2:" tt_llk_{target_arch}/instructions/assembly.yaml
```

### Quasar-Specific

Quasar uses different file naming. The equivalent of a WH/BH file may have a completely different name:
```bash
# Don't search for llk_unpack_AB.h on Quasar — it doesn't exist
# Search by concept instead
grep -r "binary.*operand\|dual.*operand" tt_llk_quasar/llk_lib/ --include="*.h" -l
```

Check for TensorShape usage:
```bash
grep -r "TensorShape\|tensor_shape" tt_llk_quasar/llk_lib/llk_{type}_{op}*.h
```

---

## Step 5: Plan the Implementation

### Template Parameters
- Source param not in target test/parent → **DROP it**
- Target test/parent has param source doesn't → **ADD it**
- Exists in both but different meaning → **Follow target**

### Init/Uninit Symmetry
For every HW register `_init_` modifies, document how `_uninit_` restores the default. This prevents reconfig escapes.

### Format Paths
Check which data format paths exist on the target:
```bash
grep -r "DataFormat\|data_format\|is_32_bit\|is_fp32" tt_llk_{target_arch}/llk_lib/llk_{type}_{op}*.h
```

---

## Step 6: Implement

1. Start with the closest existing target kernel as your template
2. Port the algorithm from the source, using target conventions
3. Verify function signatures match the test harness
4. Check init/uninit symmetry
5. Compile and test with a simple format first (Float16_b)
6. Expand to other formats

---

## Common Pitfalls

1. **Assuming source patterns work on target** — Always verify against existing target code
2. **Wrong instruction macro type** — TTI_ (immediate) vs TT_OP_ (encoding) confusion
3. **Missing init/uninit symmetry** — Causes reconfig escapes in test suites
4. **Wrong file on Quasar** — Quasar uses semantic naming, not letter-based
5. **Boolean instead of constant** — Use explicit constants (e.g., `p_unpacr_nop::CLR_SRC_NEGINF`), not boolean expressions
