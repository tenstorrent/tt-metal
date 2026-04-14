# LLK Porting Guide

This guide teaches the **methodology** for porting kernels between architectures. It does NOT provide specific instruction mappings — those must be discovered from authoritative sources for each architecture pair.

---

## Porting Philosophy

Porting a kernel is NOT line-by-line translation. It's understanding what the reference does algorithmically, then implementing that algorithm using the target architecture's capabilities. The target may be simpler (hardware support for what the reference does in software) or more complex (missing features that need workarounds).

---

## Key Translation Rules (All Kernel Types)

These rules apply to every porting effort, regardless of kernel type:

1. **Target-first design**: Start from existing target architecture patterns, use the reference only for understanding **semantics** (what the kernel does), NEVER for implementation (how it does it).

2. **Template params come from target**: Derive template parameters from the target's test harness and parent file, NOT from the reference architecture. The test file defines what signatures the target expects.

3. **Init/uninit symmetry**: `_uninit_` must reverse every hardware state change made in `_init_`. Every register write in init needs a corresponding restore in uninit.

4. **Don't over-port**: Drop reference features not referenced in the target's test harness or parent file (e.g., extra addressing modes, diagnostic parameters, deprecated variants).

5. **Verify against existing target kernels**: Read the closest target kernel of the same type line-by-line before writing any code.

6. **Test harness is the API contract**: The test file defines what signatures the target architecture expects. If the test expects different parameters than the reference, follow the test.

---

## Step 1: Understand What the Reference Does

Before looking at the target architecture at all:
1. Read the reference implementation
2. Identify the **algorithm** (mathematical operations, data flow)
3. Identify **architecture-specific constructs** (these need translation)
4. Separate the algorithm from the implementation

Example thought process:
- "This kernel computes sigmoid(x) = 1/(1+exp(-x))"
- "The reference uses a LUT-based approximation with `lut2()` calls"
- "The LUT approach is architecture-specific — the algorithm (sigmoid) is what matters"

---

## Step 2: Research the Target Architecture

Use these authoritative sources (in priority order):

### 1. Existing target implementations (highest priority)
```
tt_llk_{target_arch}/common/inc/sfpu/*.h
tt_llk_{target_arch}/llk_lib/*.h
```
Read 2-3 existing kernels of the same type. They show you:
- What includes are needed
- What namespaces are used
- What instruction patterns work
- What register conventions exist
- What loop/iteration patterns are standard

### 2. Confluence — Architecture documentation (PRIMARY for instruction details)
- **Page ID 84508873** — Tensix NEO High Level Specification (general architecture)
- **Page ID 1613201604** — Tensix Instruction Set Architecture (per-instruction details: parameters, encoding, behavior)
- Use `mcp__atlassian__getConfluencePage` to fetch these pages
- **DeepWiki**: Query `tenstorrent/tt-isa-documentation` for reference architecture ISA

### 3. assembly.yaml (cross-check)
```bash
# Verify an instruction exists
grep -c "^{INSTRUCTION}:" tt_llk_{target_arch}/instructions/assembly.yaml

# Get local ISA definition details
grep -A 30 "^{INSTRUCTION}:" tt_llk_{target_arch}/instructions/assembly.yaml

# Verify an instruction exists
grep -c "^{INSTRUCTION}:" tt_llk_{target_arch}/instructions/assembly.yaml
```

---

## Step 3: Map Constructs

For each architecture-specific construct in the reference:

1. **Does the target have a direct equivalent?**
   - Search existing target code for similar patterns
   - Check assembly.yaml for matching instructions

2. **Can it be composed from available primitives?**
   - Look at how existing target kernels handle similar operations
   - Check if multiple target instructions can achieve the same result

3. **Does it need an entirely different approach?**
   - If the reference relies on a feature the target doesn't have (e.g., LUTs, specific vector ops), redesign the algorithm using what's available
   - Check architecture docs for recommended alternatives

### Discovery method for each construct:

```bash
# Find how the target arch handles a specific concept
grep -r "{concept}" tt_llk_{target_arch}/ --include="*.h" | head -20

# Find what instructions are available for a category
grep -i "SFP" tt_llk_{target_arch}/instructions/assembly.yaml | head -30

```

---

## Step 4: Plan Register/Resource Allocation

Study existing target implementations to discover:
- How many registers are typically used
- What naming conventions exist for register purposes
- What resources (packers, unpackers, buffer descriptors) are used and how
- Any allocation constraints

Follow the same conventions as existing code.

---

## Step 5: Write the Implementation

Match existing target code style exactly:
- Same includes
- Same namespace structure
- Same function naming pattern
- Same loop/iteration pattern
- Same comment style

---

## SFPU Kernel Porting

SFPU kernels are often the most portable since the SFPI C++ library is shared across architectures. Key things to check:

- **LUT availability**: Different architectures may have different LUT modes (`lut` vs `lut2` vs `lut2_sign`)
- **Macro instructions**: Some architectures support `SFPLOADMACRO` for complex sequences
- **Instruction set differences**: Check what SFP* instructions exist on the target
- **Programmable constants**: Verify which `vConstFloatPrgm*` constants are available

---

## Unpack/Pack Kernel Porting

**CRITICAL**: Unpack and pack kernels have significant architectural differences between architectures. Do NOT assume patterns transfer directly.

### Key Areas of Divergence

| Aspect | What to check on target |
|--------|------------------------|
| MOP Template | `ckernel_template` vs `ckernel_unpack_template` — check existing kernels |
| Replay Buffers | API differs between architectures (`load_replay_buf` vs `lltt::replay()` vs other) |
| Config Writes | Pattern differs (`TTI_REG2FLOP` vs `TTI_STALLWAIT` + `TTI_WRCFG`) |
| Address Increment | Loop-based, `TTI_CFGSHIFTMASK`, or other mechanism |
| Context Handling | Explicit `unp_cfg_context` register selection may be required |

### Investigation Commands

```bash
# Find replay buffer usage on target
grep -r "replay" tt_llk_{target_arch}/llk_lib/ --include="*.h" -l

# Find which template types are used
grep -r "ckernel_template\|ckernel_unpack_template" tt_llk_{target_arch}/llk_lib/ --include="*.h"

# Compare config write patterns
grep -c "TTI_WRCFG" tt_llk_{target_arch}/llk_lib/*.h
grep -c "TTI_REG2FLOP" tt_llk_{target_arch}/llk_lib/*.h

# Find CFGSHIFTMASK usage
grep -r "CFGSHIFTMASK" tt_llk_{target_arch}/ --include="*.h"

# Find context addressing patterns
grep -r "unp_cfg_context" tt_llk_{target_arch}/ --include="*.h"
```

### Replay Buffer Patterns

Different architectures have different replay buffer APIs. **Always check how the target does it**:
```bash
grep -r "replay" tt_llk_{target_arch}/ --include="*.h" | head -20
```

### Config Write Sequences

Some architectures require stalls before config writes. **Check the target pattern**:
```bash
grep -B2 -A2 "TTI_WRCFG\|TTI_REG2FLOP" tt_llk_{target_arch}/llk_lib/ --include="*.h" -r | head -30
```

### Tile Dimension Configuration

Tilize/untilize modes often require explicit tile dimension setup. **Check existing target kernels**:
```bash
grep -r "Tile_x_dim\|Tile_z_dim\|config_unpacker_x_end" tt_llk_{target_arch}/llk_lib/ --include="*.h" | head -10
```

### Context-Based Addressing

Some architectures require explicit context-based register selection. **Check the target pattern**:
```bash
grep -B3 -A5 "unp_cfg_context" tt_llk_{target_arch}/llk_lib/ --include="*.h" -r | head -20
```

### Unpack/Pack Reference Files

**ALWAYS check these files on the target architecture for patterns before implementing:**

| File Pattern | Pattern Demonstrated |
|------|---------------------|
| `llk_unpack_untilize.h` | Replay buffer, unpack template, state save/restore |
| `llk_unpack_AB_matmul.h` | Replay buffer with address updates |
| `llk_unpack_A.h` | Context-based addressing, x-end configuration |
| `llk_unpack_AB_reduce.h` | Multiple MOP configurations, pool-type handling |
| `llk_unpack_common.h` | Hardware configure, tile dimension setup |
| `llk_pack_common.h` | Pack initialization, MOP config |

---

## Common Porting Patterns

These patterns apply across many architecture pairs, but **always verify against actual code**:

### Pattern: Hardware acceleration
The target may have single-instruction support for what the reference implements in software (e.g., hardware sqrt vs. Newton-Raphson). Always check for built-in operations before porting complex algorithms.

### Pattern: API level differences
Different architectures may use different abstraction levels:
- High-level C++ operator overloading vs. explicit instruction calls
- Implicit register management vs. explicit register allocation
- Library functions vs. direct hardware instructions

### Pattern: Conditional execution
Different architectures handle per-element conditionals differently:
- SIMD-style predication
- Condition code registers
- Compile-time constexpr branching
- Hardware-supported operations (e.g., ReLU as max(0,x) in one instruction)

### Pattern: Iteration and data movement
Different architectures process data in different chunk sizes and have different mechanisms for advancing through memory.

---

## Verification

After implementing, verify:

1. **Compilation**: `python scripts/check_compile.py {kernel_file} -v`
2. **Instruction validity**: Every instruction used exists in assembly.yaml
3. **Pattern conformance**: The code matches existing target implementation patterns
4. **Functional correctness**: Run available tests

---

## Key Paths

| Path | Purpose |
|------|---------|
| `tt_llk_{arch}/common/inc/sfpu/*.h` | SFPU kernel implementations |
| `tt_llk_{arch}/llk_lib/*.h` | Math/pack/unpack kernel implementations |
| `tt_llk_{arch}/instructions/assembly.yaml` | ISA definition (use grep) |
| `codegen/references/common-errors.md` | Known error patterns and fixes |
