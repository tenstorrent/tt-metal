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

```
# Find the source kernel
Glob for tt_llk_{source_arch}/llk_lib/llk_{type}_{op}*.h
Glob for tt_llk_{source_arch}/common/inc/sfpu/ckernel_sfpu_{op}.h

# Read it
Read tt_llk_{source_arch}/llk_lib/llk_{type}_{op}.h
```

---

## Step 2: Research the Target Architecture

### 2a: Find the Closest Existing Target Kernel

```
# List all kernels of the same type on the target
Glob for tt_llk_{target_arch}/llk_lib/llk_{type}_*.h

# For SFPU
Glob for tt_llk_{target_arch}/common/inc/sfpu/ckernel_sfpu_*.h
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

```
# Find the test source
Glob for tests/sources/*{op}*.cpp

# Extract target-arch-specific signatures
Grep for "ARCH_{TARGET}" in tests/sources/*{op}*.cpp (20 lines context)
```

**If the test harness and source implementation disagree on signatures, the test harness wins.**

### 2c: Check the Parent/Caller File

```
Grep for "#include.*{op}" in tt_llk_{target_arch}/llk_lib/ (*.h files)
```

Read wrapper functions, verify template params, find helper functions.

### 2d: Check assembly.yaml for Instruction Availability

```
# Verify the instruction exists on the target
Grep (count mode) for "^{INSTRUCTION}:" in tt_llk_{target_arch}/instructions/assembly.yaml

# Get full instruction definition
Grep for "^{INSTRUCTION}:" in tt_llk_{target_arch}/instructions/assembly.yaml (50 lines context)

# If instruction doesn't exist, find alternatives
Grep (case-insensitive) for "{partial_name}" in tt_llk_{target_arch}/instructions/assembly.yaml
```

---

## Step 3: Map Constructs

### Instruction Macros

Check which macro style the target uses:
```
Grep (count mode) for "TTI_WRCFG" in tt_llk_{target_arch}/llk_lib/ (*.h files)
Grep (count mode) for "TTI_REG2FLOP" in tt_llk_{target_arch}/llk_lib/ (*.h files)
```

### Template Types

```
Grep for "ckernel_template|ckernel_unpack_template" in tt_llk_{target_arch}/llk_lib/llk_{type}_*.h
```

### Replay Buffer

```
Grep (files only) for "load_replay_buf|replay" in tt_llk_{target_arch}/llk_lib/ (*.h files)
```

### Config Writes

```
Grep for "TTI_STALLWAIT.*TTI_WRCFG|TTI_REG2FLOP" in tt_llk_{target_arch}/llk_lib/ (*.h files)
```

---

## Step 4: Handle Architecture-Specific Differences

### WH vs BH/QSR Common Differences

Discover these dynamically — do NOT assume:
```
# Check PACR parameter count
Grep for "TT_OP_PACR|TTI_PACR" in tt_llk_{target_arch}/llk_lib/ (*.h files)

# Check SFPU instruction availability
Grep for "^SFPLOADMACRO:" in tt_llk_{target_arch}/instructions/assembly.yaml
Grep for "^SFPSHFT2:" in tt_llk_{target_arch}/instructions/assembly.yaml
```

### Quasar-Specific

Quasar uses different file naming. The equivalent of a WH/BH file may have a completely different name:
```
# Don't search for llk_unpack_AB.h on Quasar — it doesn't exist
# Search by concept instead
Grep (files only) for "binary.*operand|dual.*operand" in tt_llk_quasar/llk_lib/ (*.h files)
```

Check for TensorShape usage:
```
Grep for "TensorShape|tensor_shape" in tt_llk_quasar/llk_lib/llk_{type}_{op}*.h
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
```
Grep for "DataFormat|data_format|is_32_bit|is_fp32" in tt_llk_{target_arch}/llk_lib/llk_{type}_{op}*.h
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
