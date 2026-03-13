# Comparison: Unary vs Binary vs Ternary SFPU Operations

This document compares the architecture, data flow, and implementation requirements for adding SFPU operations of different arities (unary, binary, ternary) to the tt-metal/ttnn codebase.

---

## 1. Overview Comparison

| Aspect | Unary | Binary | Ternary |
|--------|-------|--------|---------|
| **Inputs** | 1 tensor | 2 tensors | 3 tensors or tensor+scalar combinations |
| **Variants** | None | (implied by dtype) | TTT, TTS, TST, TSS |
| **Stack Chain** | Python → Nanobind → C++ Executor → Device Op → Program Factory → Compute Kernel → API → SFPU | Python → Nanobind → C++ Op Templates → Device Op → **Program Factory Selection (is_binary_sfpu_op)** → SFPU or FPU Path | Python → Nanobind → C++ Operation → Device Op → Program Factory → Compute Kernel → API → SFPU |
| **Primary Kernel Pattern** | Pure SFPU (copy_tile required) | **Dual patterns**: FPU+SFPU or Pure SFPU | Pure SFPU (copy_tile for all inputs) |

**Key Differences:**
- **Unary** has the simplest flow - single input, single SFPU operation
- **Binary** uniquely has **two distinct patterns** depending on whether FPU can handle part of the operation
- **Ternary** supports **multiple input variants** (TTT, TTS, TST, TSS) for flexibility and always uses pure SFPU pattern

---

## 2. Architecture Flow Diagrams Comparison

### Unary Flow
```
┌─────────────────────────────────────────────────────────────────┐
│                         Python API                               │
│                    ttnn.<op_name>(tensor)                        │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Nanobind Bindings                           │
│                   unary_nanobind.cpp                             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       C++ Executor                               │
│          ExecuteUnary<UnaryOpType::<OP_NAME>>::invoke()         │
│                       unary.hpp                                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Program Factory                              │
│         Maps UnaryOpType to SFPU_OP_* defines                    │
│                  unary_program_factory.cpp                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Compute Kernel                              │
│                     eltwise_sfpu.cpp                             │
│    Expands SFPU_OP_CHAIN_0 → <op_name>_tile_init/tile()         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       SFPU Kernel                                │
│              ckernel_sfpu_<op_name>.h                            │
│     Architecture-specific (Wormhole B0 / Blackhole)              │
└─────────────────────────────────────────────────────────────────┘
```

### Binary Flow (Dual Pattern Architecture)
```
Python API (ttnn.your_op)
    ↓
Nanobind Bindings (binary_nanobind.cpp)
    ↓
C++ Operation Templates (BinaryOperation<OpType>)
    ↓
Device Operation (BinaryDeviceOperation)
    ↓
Program Factory Selection (is_binary_sfpu_op)
    ↓
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│  ┌─────────────────────────────┐   ┌─────────────────────────────┐│
│  │     Pattern 1: FPU Kernel   │   │    Pattern 2: SFPU Kernel   ││
│  │     with SFPU pre/post      │   │       (standalone)          ││
│  │                             │   │                             ││
│  │  - BFLOAT16 arithmetic      │   │  - FLOAT32/INT32 types      ││
│  │  - SFPU for transcendentals │   │  - SFPU-only operations     ││
│  │  - Intermediate CBs used    │   │  - Interleaved DEST layout  ││
│  └─────────────────────────────┘   └─────────────────────────────┘│
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
    ↓
Hardware SFPU Execution
```

### Ternary Flow
```
Python → Nanobind → C++ invoke() → ttnn::prim::ternary →
Program Factory → Compute Kernel (macros) → LLK wrapper → SFPU calculation

Note: Ternary operations ALWAYS use Pattern 2 (standalone SFPU) because
FPU operations are inherently binary. A ternary operation like a*b+c would
require multiple FPU calls with intermediate storage, losing performance benefit.
```

### Key Architectural Differences

| Component | Unary | Binary | Ternary |
|-----------|-------|--------|---------|
| **Decision Point** | Always SFPU | `is_binary_sfpu_op()` check | By variant (TTT, TTS, etc.) |
| **Kernel Patterns** | 1 pattern | **2 patterns** (FPU+SFPU or pure SFPU) | 1 pattern (pure SFPU) |
| **Intermediate CBs** | None | Yes (for Pattern 1 prescaling) | None |
| **FPU Participation** | Never | Often (Pattern 1) | Never |
| **Init Function** | `init_sfpu()` | `binary_op_init_common()` or `unary_op_init_common()` | `unary_op_init_common()` |

---

## 3. Hardware Data Flow: The Critical Constraint

**The SFPU can ONLY access DEST registers** - it has no direct path to DRAM or Circular Buffers.

| Unit | Can Read From | Writes To |
|------|---------------|-----------|
| **FPU** | SRC registers | DEST registers |
| **SFPU** | DEST registers only | DEST registers |

### Complete Data Path
```
DRAM → [NoC] → L1 (Circular Buffers) → [Unpacker] → SRC registers → [FPU] → DEST → [SFPU] → DEST → [Packer] → L1 → [NoC] → DRAM
```

### Memory Hierarchy

| Memory Level | Location | Size | Access Speed |
|--------------|----------|------|--------------|
| DRAM | Off-chip (global memory) | GB | Slow, via NOC |
| L1 Memory / Circular Buffers | On-chip (per core) | 256KB-512KB | Fast, direct L1 access |
| DEST Registers | On-chip (register file) | 16 tiles | Very fast, direct element/tile access |
| SFPU Registers (LREG0-LREG5) | On-chip (SFPU unit) | 8 elements | Fastest, for computations |

---

## 4. Data Flow Diagrams Comparison

### Unary: Simple Linear Flow
```
DRAM → L1 (CB[c_0]) → copy_tile() → DEST[0] → SFPU op → DEST[0] → pack_tile() → CB[c_2] → DRAM
```

**Detailed Unary Data Flow:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DRAM: Input tensor data                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    Reader Kernel: noc_async_read_page()
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ L1 Memory: Circular Buffer CB[c_0]                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    cb_wait_front() + copy_tile()
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DEST Registers: DEST[0] (SFPU reads and writes here in-place)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    SFPU operation (e.g., relu_tile(0))
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DEST Registers: Result in DEST[0]                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    pack_tile(0, CB[c_2])
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ L1 Memory: Circular Buffer CB[c_2] → Writer Kernel → DRAM                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Binary Pattern 1: SFPU Pre/Post + FPU (e.g., DIV, HYPOT)

**Example: DIV** = `a * SFPU_RECIP(b)`
```
                        SFPU Pre-processing (RECIP on input_b)
                        ┌─────────────────────────────────────────────────────┐
                        │  copy_tile() - Unpacker → SRC → datacopy → DEST     │
                        │                                                     │
  Input CB (cb_in1)     │    ┌──────────┐   ┌───────┐   ┌────────┐           │
  ──────────────────────┼──▶ │ Unpacker │──▶│  SRC  │──▶│datacopy│           │
         (input_b)      │    └──────────┘   └───────┘   └────┬───┘           │
                        │                                    │               │
                        │                   ┌────────────────┘               │
                        │                   ▼                                │
                        │              ┌─────────┐    ┌───────┐              │   Intermediate CB
                        │              │  DEST   │───▶│ SFPU  │──────────────┼──▶ (cb_inp1 = 1/b)
                        │              │         │◀───│ RECIP │              │
                        │              └─────────┘    └───────┘              │
                        └─────────────────────────────────────────────────────┘

                        FPU Operation (MUL: a * (1/b))
                        ┌─────────────────────────────────────────────────────┐
  cb_inp0 (input_a)     │    ┌───────┐      ┌──────┐      ┌───────┐          │
  ──────────────────────┼──▶ │Unpack │      │ SRC  │      │  FPU  │          │
                        │    │  AB   │ ──▶  │ A+B  │ ──▶  │  MUL  │          │
  cb_inp1 (1/b)         │    │       │      │      │      │       │          │
  ──────────────────────┼──▶ └───────┘      └──────┘      └───┬───┘          │
                        │                                     │              │
                        │                                     ▼              │
                        │                               ┌───────────┐        │
                        │                               │   DEST    │────────┼──▶ cb_out0
                        │                               │ (a * 1/b) │        │
                        │                               └───────────┘        │
                        └─────────────────────────────────────────────────────┘
```

### Binary Pattern 2: Pure SFPU (e.g., POWER, ADD with FLOAT32)

**Uses interleaved DEST layout and direct unpack to DEST for 32-bit formats:**
```
                        Stage 1-2: Copy inputs to DEST (interleaved)
                        ┌─────────────────────────────────────────────────────┐
                        │  copy_tile() with UnpackToDestEn=true               │
                        │  (Direct CB → DEST for FLOAT32/INT32)               │
                        │                                                     │
  Input CB A (cb_inp0)  │    ┌─────────────────────────────────┐             │
  ──────────────────────┼──▶ │ Unpacker (unpack_to_dest=true)  │             │
                        │    └────────────────┬────────────────┘             │
                        │                     ▼                              │
                        │                 ┌──────────────────────────────┐   │
                        │                 │           DEST               │   │
                        │                 │  ┌─────┬─────┬─────┬─────┐   │   │
                        │                 │  │ A0  │ B0  │ A1  │ B1  │...│   │
  Input CB B (cb_inp1)  │                 │  │idx 0│idx 1│idx 2│idx 3│   │   │
  ──────────────────────┼──▶ (same path)  │  └─────┴─────┴─────┴─────┘   │   │
                        │                 │   A: even    B: odd          │   │
                        │                 └──────────────────────────────┘   │
                        └─────────────────────────────────────────────────────┘

                        Stage 3: SFPU Binary Operation
                        ┌─────────────────────────────────────────────────────┐
                        │                 ┌──────────────────────────────┐    │
                        │                 │           DEST               │    │
                        │                 │  ┌─────┬─────┬─────┬─────┐   │    │
                        │                 │  │ A0  │ B0  │ A1  │ B1  │...│    │
                        │                 │  └──┬──┴──┬──┴─────┴─────┘   │    │
                        │                 │     │     │                  │    │
                        │                 │     ▼     ▼                  │    │
                        │                 │   ┌─────────┐                │    │
                        │                 │   │  SFPU   │                │    │
                        │                 │   │ binary  │ add_binary_tile(0, 1, 0)
                        │                 │   │   op    │                │    │
                        │                 │   └────┬────┘                │    │
                        │                 │        ▼  result → idx 0    │    │
                        │                 │  ┌─────┬─────┬─────┬─────┐   │    │
                        │                 │  │A0+B0│ B0  │A1+B1│ B1  │...│    │
                        │                 │  └─────┴─────┴─────┴─────┘   │    │
                        │                 └──────────────────────────────┘    │
                        │                       │                             │
                        │                       ▼ pack even indices           │
                        └───────────────────────┼─────────────────────────────┘
                                                ▼
                                            cb_out0
```

### Ternary: Three-Input SFPU Flow

**Example: ADDCMUL** = `a + (value * b * c)`
```
                        Standalone SFPU Operation (No FPU)
                        ┌─────────────────────────────────────────────────────┐
                        │  All THREE inputs must be copied to DEST            │
                        │  SFPU performs the entire computation               │
                        │                                                     │
  Input CB (cb_in0)     │    ┌─────────────────────────────┐                  │
  ──────────────────────┼──▶ │ copy_tile(cb_in0, 0, 0)     │                  │
         (input_a)      │    │ Unpacker → DEST[0]          │                  │
                        │    └──────────────┬──────────────┘                  │
                        │                   │                                 │
  Input CB (cb_in1)     │    ┌──────────────┴──────────────┐                  │
  ──────────────────────┼──▶ │ copy_tile(cb_in1, 0, 1)     │                  │
         (input_b)      │    │ Unpacker → DEST[1]          │                  │
                        │    └──────────────┬──────────────┘                  │
                        │                   │                                 │
  Input CB (cb_in2)     │    ┌──────────────┴──────────────┐    ┌───────┐    │
  ──────────────────────┼──▶ │ copy_tile(cb_in2, 0, 2)     │    │       │    │
         (input_c)      │    │ Unpacker → DEST[2]          │    │ SFPU  │    │
                        │    └──────────────┬──────────────┘    │addcmul│    │
                        │                   │                   │       │    │
                        │                   ▼                   │ a +   │    │
                        │         ┌─────────────────┐           │(v*b*c)│    │
                        │         │     DEST        │──────────▶│       │    │
                        │         │  [0] = a        │           │       │    │
                        │         │  [1] = b        │◀──────────│       │    │
                        │         │  [2] = c        │           └───────┘    │
                        │         └────────┬────────┘                        │
                        │                  │ DEST[0] = result                │
                        │                  ▼                                 │
                        │         ┌─────────────────┐                        │   Output CB
                        │         │   pack_tile     │────────────────────────┼──▶ (cb_out)
                        │         │  DEST[0] → CB   │                        │
                        │         └─────────────────┘                        │
                        └─────────────────────────────────────────────────────┘
```

### Data Flow Summary Table

| Aspect | Unary | Binary (Pattern 1) | Binary (Pattern 2) | Ternary |
|--------|-------|-------------------|-------------------|---------|
| **Input CBs** | 1 (c_0) | 2 (c_0, c_1) + 2 intermediate (c_3, c_4) | 2 (c_0, c_1) | 3 (c_0, c_1, c_2) |
| **DEST Usage** | DEST[0] | Varies by phase | Interleaved (even/odd) | DEST[0,1,2] → DEST[0] |
| **FPU Involvement** | None | Core operation | None | None |
| **Intermediate CBs** | None | Yes (for prescaled values) | None | None |
| **copy_tile Calls** | 1 | 1 per SFPU phase | 2 (interleaved) | 3 |
| **Direct Unpack to DEST** | Via SRC for BF16 | Via SRC for BF16 | Yes for FP32/INT32 | Via SRC for BF16 |

---

## 5. Files to Modify/Create Summary

| Step | Unary | Binary | Ternary |
|------|-------|--------|---------|
| **Enum Definition** | `unary_op_types.hpp` | `binary_op_types.hpp` + optional `ckernel_defs.h` | `ternary_op_types.hpp` |
| **SFPU Kernel (per arch)** | `ckernel_sfpu_<op>.h` | `ckernel_sfpu_<op>.h` | `ckernel_sfpu_<op>.h` |
| **Compute API Wrapper** | `eltwise_unary/<op>.h` | `eltwise_binary_sfpu.h` | `llk_math_eltwise_ternary_sfpu_<op>.h` |
| **Split Includes** | `sfpu_split_includes.h` | N/A (direct includes) | N/A (direct includes) |
| **Op Utils** | `unary_op_utils.cpp` | `binary_op_utils.cpp` | `ternary_op_utils.cpp` |
| **Device Operation** | `unary_device_operation.hpp` | `binary_device_operation.cpp` (`is_binary_sfpu_op`) | Inherited from ternary primitive |
| **C++ Registration** | `unary.hpp` (macros) | `binary.hpp` | `ternary.hpp` (struct + invoke) |
| **Python Bindings** | `unary_nanobind.cpp` | `binary_nanobind.cpp` | `ternary_nanobind.cpp` |
| **Golden Function** | `unary.py` | `binary.py` | `ternary.py` |
| **Compute Kernel** | `eltwise_sfpu.cpp` (shared) | `eltwise_binary_sfpu_kernel.cpp` | `ternary_<op>_sfpu.cpp` (per broadcast type) |

### Unique Files per Operation Type

| Operation Type | Unique Files/Patterns |
|----------------|----------------------|
| **Unary** | `sfpu_split_includes.h` for conditional compilation, `SFPU_OP_CHAIN_0` macro |
| **Binary** | `ckernel_defs.h` (hardware enum if fundamental op), `is_binary_sfpu_op()` function, dual kernel paths |
| **Ternary** | Multiple kernel files for broadcast variants (no_bcast, bcast, row_bcast), `TERNARY_SFPU_OP_*` macros |

---

## 6. Boilerplate vs Non-Boilerplate Summary

| Component | Unary | Binary | Ternary |
|-----------|-------|--------|---------|
| **Enum** | ✅ Boilerplate | ✅ Boilerplate (high-level) | ✅ Boilerplate |
| **Hardware Enum** | N/A | ❌ NOT boilerplate (if needed) | N/A |
| **SFPU Kernel** | ❌ Core math logic | ❌ Core math logic | ❌ Core math logic |
| **LLK Wrapper** | ✅ Boilerplate | ✅ Boilerplate | ✅ Boilerplate |
| **Compute API** | ✅ Boilerplate | ✅ Boilerplate | ✅ Boilerplate |
| **Op Utils** | ⚠️ Mostly (params not) | ✅ Boilerplate | ✅ Boilerplate |
| **Data Type Check** | N/A | ⚠️ Partial (dtype logic) | N/A |
| **Registration** | ✅ Boilerplate (macro) | ✅ Boilerplate | ✅ Boilerplate |
| **Bindings** | ✅ Structure, ❌ Docs | ✅ Structure, ❌ Docs | ✅ Structure, ❌ Docs |
| **Golden Function** | ❌ Requires PyTorch equivalent | ❌ Requires PyTorch equivalent | ❌ Requires PyTorch equivalent |
| **Tests** | ❌ Input ranges, tolerances | ❌ Input ranges, tolerances | ❌ Input ranges, tolerances |

---

## 7. Key Distinguishing Features

| Feature | Unary | Binary | Ternary |
|---------|-------|--------|---------|
| **SFPI Intrinsics** | `dst_reg[0]`, single input | `dst_reg[in0]`, `dst_reg[in1]` | `dst_reg[0,1,2]` |
| **Conditional Compilation** | `SFPU_OP_*_INCLUDE` system | `#ifdef SFPU_OP_*` blocks | `TERNARY_SFPU_OP_*` macros |
| **FPU Integration** | None | Core capability (Pattern 1) | None |
| **Broadcast Support** | Via program factory | FPU kernel handles | Multiple kernel files |
| **Parameter Passing** | `std::bit_cast` for floats | `std::bit_cast` for floats | Via compile-time defines + runtime args |
| **Macro Pattern** | `SFPU_OP_CHAIN_0` | `SFPU_OP_INIT_PRE_IN*_0`, `BINARY_SFPU_OP` | `TERNARY_SFPU_OP_INIT`, `TERNARY_SFPU_OP_FUNC` |

---

## 8. Pattern Comparison: When to Use Each

### Binary Operation Patterns

| Scenario | Pattern | Kernel | Key Characteristic |
|----------|---------|--------|-------------------|
| HYPOT with BFLOAT16 | Pattern 1 | FPU kernel | SFPU `square` pre → FPU `add_tiles` → SFPU `sqrt` post |
| DIV with BFLOAT16 | Pattern 1 | FPU kernel | SFPU `recip` pre → FPU `mul_tiles` |
| GT with BFLOAT16 | Pattern 1 | FPU kernel | FPU `sub_tiles` → SFPU `gtz` post |
| ADD with FLOAT32 | Pattern 2 | SFPU kernel | `copy_tile` → SFPU `add_binary_tile` |
| POWER (any dtype) | Pattern 2 | SFPU kernel | `copy_tile` → SFPU `power_binary_tile` |
| HYPOT with FLOAT32 | Pattern 2 | SFPU kernel | SFPU `square` pre → SFPU `add_binary_tile` → SFPU `sqrt` post |

### Why Ternary Operations Don't Use Pattern 1

FPU operations are inherently binary (2 inputs). A ternary operation like `a * b + c` would require:
1. FPU mul: `a * b → temp` (write to intermediate CB)
2. FPU add: `temp + c → result`

This loses the performance benefit of keeping data in DEST. Instead, ternary operations use Pattern 2 (standalone SFPU) where all three inputs are copied to DEST and processed in a single SFPU operation.

---

## 9. Summary

The three operation types share a common architectural foundation but differ in complexity:

### Unary Operations
- **Simplest** implementation path
- Single input, direct SFPU processing
- Uses conditional compilation via `sfpu_split_includes.h`
- No FPU involvement
- Data flow: CB → copy_tile → DEST → SFPU → pack → CB

### Binary Operations
- **Most complex** due to dual kernel patterns
- **Pattern 1 (FPU+SFPU):** Used when FPU can handle the core operation (BFLOAT16)
  - Uses intermediate circular buffers for prescaled values
  - SFPU handles transcendentals (recip, exp, log, sqrt), FPU handles arithmetic
  - Macros: `SFPU_OP_INIT_PRE_IN*_0`, `SFPU_OP_FUNC_PRE_IN*_0`
- **Pattern 2 (Pure SFPU):** Used for FLOAT32/INT32 or operations without FPU equivalent
  - Interleaved DEST register layout (A at even, B at odd indices)
  - Direct unpack to DEST for 32-bit formats (`UnpackToDestEn=true`)
  - Macros: `BINOP_INIT`, `BINARY_SFPU_OP`
- Requires `is_binary_sfpu_op()` decision logic

### Ternary Operations
- Handles multiple input variants (TTT, TTS, TST, TSS)
- Always uses Pattern 2 (standalone SFPU) - FPU is inherently binary
- Requires separate kernel files for different broadcast types
- All three inputs copied to consecutive DEST registers [0,1,2]
- Uses `unary_op_init_common()` (not `binary_op_init_common()`)
- Macros: `TERNARY_SFPU_OP_INIT`, `TERNARY_SFPU_OP_FUNC`

---

## 10. Quick Reference: Where to Focus Effort

| Category | Unary | Binary | Ternary |
|----------|-------|--------|---------|
| **Core Algorithm (SFPU)** | 🔴 High | 🔴 High | 🔴 High |
| **Correctness Definition (Tests/Golden)** | 🔴 High | 🔴 High | 🔴 High |
| **Architectural Decisions** | 🟢 Low | 🟡 Medium (dtype, pattern selection) | 🟡 Medium (variants) |
| **Plumbing/Wiring** | 🟢 Low | 🟢 Low | 🟢 Low |

**Bottom Line:** For all operation types, expect to spend:
- **80% of time** on SFPU implementation and testing
- **15% of time** on architectural decisions and debugging
- **5% of time** on boilerplate modifications

**Key insight for ternary:** ~90% of adding a new ternary operation is boilerplate. The only truly custom code is:
1. The SFPU calculation formula (1-10 lines)
2. The PyTorch golden function (1-5 lines)
3. Test-specific considerations (input ranges, precision thresholds)
