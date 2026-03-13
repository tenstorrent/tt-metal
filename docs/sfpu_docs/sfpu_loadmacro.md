# SFPU LOADMACRO — Comprehensive Reference

> Fused from: Confluence SFPU ISA page (page 1170505767), Glean docs (Wormhole_B0 / Blackhole SFPU Specifications), and tt-metal source code.

## 1. Overview

SFPLOADMACRO is a **meta-instruction** that enables Instruction-Level Parallelism (ILP) in the otherwise single-issue SFPU. It combines a LOAD from the Destination (or SrcS) Register file with a programmable sequence of up to 4 additional operations — Simple, MAD, Round, and Store — that execute on separate pipelines with programmer-defined delays. This allows the SFPU to achieve an effective IPC of up to ~5.0, versus the normal single-issue throughput of 1.0.

**Key properties:**
- Always executes a LOAD first (from Dest/SrcS register file, with inline format conversion).
- Triggers a macro sequence (indexed by `VD[3:2]`) that can issue up to 4 further instructions across independent execution units.
- Bypasses normal hardware instruction ordering checks — the **programmer** is responsible for all hazard management.
- Available on **Wormhole B0**, **Blackhole**, and **Quasar** architectures.

## 2. Instruction Encoding

### 2.1 Opcode

| Field | Value |
|-------|-------|
| Opcode | `0x93` |
| Execution Resource | SFPU |
| Instruction Type | SFPU |

### 2.2 ISA-Level Description (from assembly.yaml)

> "sFPU load from dest regs and then run the macro specified in lreg_dest index[3:2]"

The instruction uses the **SFPU_MEM** argument format, shared with SFPLOAD/SFPSTORE.

### 2.3 Bit Fields

#### Wormhole B0 (24-bit payload, `sfpu_addr_mode` is 2 bits starting at bit 14)

```
TT_OP_SFPLOADMACRO(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr)
  = (opcode << 24)
  | (lreg_ind       << 20)   // [23:20] 4 bits — LREG index (VD)
  | (instr_mod0      << 16)   // [19:16] 4 bits — Format select (InstrMod)
  | (sfpu_addr_mode  << 14)   // [15:14] 2 bits — Address mode register index
  | (dest_reg_addr   <<  0)   // [13:0] 14 bits — Destination register address
```

#### Blackhole (24-bit payload, `sfpu_addr_mode` is 3 bits starting at bit 13)

```
TT_OP_SFPLOADMACRO(lreg_ind, instr_mod0, sfpu_addr_mode, dest_reg_addr)
  = (opcode << 24)
  | (lreg_ind       << 20)   // [23:20] 4 bits — LREG index (VD)
  | (instr_mod0      << 16)   // [19:16] 4 bits — Format select (InstrMod)
  | (sfpu_addr_mode  << 13)   // [15:13] 3 bits — Address mode register index
  | (dest_reg_addr   <<  0)   // [12:0] 13 bits — Destination register address
```

#### Quasar (24-bit payload, additional `seq_id` and `done` fields)

```
TT_OP_SFPLOADMACRO(seq_id, lreg_ind_lo, instr_mod0, sfpu_addr_mode, done, dest_reg_addr, lreg_ind_hi)
  = (opcode << 24)
  | (seq_id          << 22)   // [23:22] 2 bits — Sequence ID (replaces VD[3:2])
  | (lreg_ind_lo     << 20)   // [21:20] 2 bits — VD low bits
  | (instr_mod0      << 16)   // [19:16] 4 bits — Format select
  | (sfpu_addr_mode  << 13)   // [15:13] 3 bits — Address mode register index
  | (done            << 11)   // [12:11] 2 bits — Done flag
  | (dest_reg_addr   <<  1)   // [10:1] 10 bits — Dest register address
  | (lreg_ind_hi     <<  0)   // [0] 1 bit — VD high bit
```

### 2.4 VD Field Semantics

The 4-bit `lreg_ind` (VD) field is split as follows:

| Sub-field | Bits | Purpose |
|-----------|------|---------|
| `VD[3:2]` | Upper 2 bits | Selects which of the 4 LOADMACRO Sequence Registers to use |
| `VD[1:0]` | Lower 2 bits | Combined with `Addr[0]` to form the LREG destination: `VD_FINAL = {1'b0, Addr[0], VD[1:0]}` |

In C macros this is typically written as:
```c
TT_SFPLOADMACRO((seq << 2) | (lreg & 3), instr_mod, addr_mod, offset | (lreg >> 2));
```

### 2.5 InstrMod (Format Select)

Same as SFPLOAD — controls inline data format conversion on the loaded value:

| Value | Name | Description |
|-------|------|-------------|
| `0x0` | DEFAULT | Implied format (FP32, FP16A, or FP16B) |
| `0x1` | FP16_A | IEEE Binary16 |
| `0x2` | FP16_B | bfloat16 |
| `0x3` | FP32 | IEEE Binary32 |
| `0x4` | INT32 | 32-bit signed magnitude integer |
| `0x5` | INT8 / SMAG8 | Sign-magnitude 8-bit |
| `0x6` | LO16 | Unsigned 32-bit, low 16 bits |
| `0x7` | HI16 | Unsigned 32-bit, high 16 bits |
| `0xC` | INT32_COMP | 32-bit two's complement integer |
| `0xE` | LO16_ONLY | Unsigned 16-bit |
| `0xF` | HI16_ONLY | Unsigned 16-bit |

## 3. LOADMACRO Registers

There are **12 shared registers** per SFPU Slice that control LOADMACRO execution: **8 Instruction Registers** and **4 Sequence Registers**, plus the **LOADMACRO Control Register**.

All of these are programmed via SFPCONFIG (config_dest indices 0x0–0x8).

### 3.1 LOADMACRO Control Register (config_dest = 0x8)

Global settings that alter SFPLOADMACRO behavior.

| Field | Bits | Description |
|-------|------|-------------|
| `DEFAULT_STORE_INSMOD` | 3:0 | Default `InstrMod` value for store instructions in a LOADMACRO sequence |
| `STORE_INHERITS_INSMOD` | 7:4 | Per-sequence bit: if set, the store inherits `InstrMod` from the LOADMACRO instead of using `DEFAULT_STORE_INSMOD`. bit4→Seq0, bit5→Seq1, bit6→Seq2, bit7→Seq3 |
| `UNIT_DEPENDENCY_ENABLE` | 11:8 | Per-unit dependent-instruction stalls. bit8→Simple, bit9→MAD, bit10→Round, bit11→Store. When set, the sequence stalls that unit if it has a pending instruction and no new instruction is being issued to the SFPU |
| `INFINITY_HANDLING_ENABLE` | (arch-specific) | Enables proper infinity handling for FP16 values when loading from Dest |

### 3.2 LOADMACRO Instruction Registers (8 total)

Each is a 32-bit word used as a template for the instruction issued by the sequence.

| Index | config_dest | Default Value | Description |
|-------|-------------|---------------|-------------|
| 0 | N/A | N/A | Pass-through: uses the newly issued instruction from Tensix this cycle |
| 1 | N/A | N/A | Reserved |
| 2 | N/A | `0x8F000000` | Hardcoded SFPNOP |
| 3 | N/A | `0x720X0000` | Hardcoded SFPSTORE (X = DEFAULT_STORE_INSMOD) |
| 4 | 0x0 | `0x00000000` | Programmable Instruction 0 |
| 5 | 0x1 | `0x00000000` | Programmable Instruction 1 |
| 6 | 0x2 | `0x00000000` | Programmable Instruction 2 |
| 7 | 0x3 | `0x00000000` | Programmable Instruction 3 |

#### Programming Methods

**Method 1: SFPCONFIG** — Write the 32-bit instruction word into LREG[0], then issue `SFPCONFIG(0, config_dest, 0)` targeting indices 0x0–0x3.

**Method 2: Backdoor Loading** — When `SFPU_CONTROL.BACKDOOR_LOADING_DISABLE == 0` (default), any SFPU instruction with VD = 12–15 is **not executed** but instead captured into Instruction Registers 4–7 respectively:

| Instruction VD | Captured Into |
|----------------|---------------|
| 12 | Instruction Register 4 |
| 13 | Instruction Register 5 |
| 14 | Instruction Register 6 |
| 15 | Instruction Register 7 |

### 3.3 LOADMACRO Sequence Registers (4 total)

Each 32-bit Sequence Register is divided into 4 slots (8 bits each) controlling which instruction executes on each unit and when.

| config_dest | Sequence |
|-------------|----------|
| 0x4 | Sequence Register 0 |
| 0x5 | Sequence Register 1 |
| 0x6 | Sequence Register 2 |
| 0x7 | Sequence Register 3 |

#### Slot Layout (per unit, 8 bits each)

The 32-bit sequence register is laid out as:
```
bits [7:0]   = Simple slot
bits [15:8]  = MAD slot
bits [23:16] = Round slot
bits [31:24] = Store slot
```

Per-unit 8-bit encoding:

| Field | Bits (within slot) | Description |
|-------|-----|-------------|
| `INSTRN_SEL` | 2:0 | Which of the 8 Instruction Registers to use (index 0–7) |
| `INSTRN_DLY` | 5:3 | Delay in cycles relative to the LOADMACRO (0 = next cycle after LM) |
| `USE_STAGING` | 6 | 1: write result to Staging Register; 0: write to LREG specified by VD |
| `SRCB_OVERRIDE` | 7 | 1: use loaded value as Source B; 0: use loaded value as Source C |

For the **Store slot**, bit 6 is `USE_STAGING` (1: read source from Staging Register) and bit 7 is `STORE_ADDR_OFFSET` (1: add store address to LOADMACRO address).

#### Example Encoding

From `ckernel_sfpu_mul_int32.h`:
```
//   (store << 24) | (round << 16) | (mad << 8) | simple
//
//  Per-unit 8 bits (high to low):
//   - bit 7: VB=VD override (instead of VC=VD)
//   - bit 6: VD=16 (staging register)
//   - bits 5:3: delay (0-7)
//   - bits 2:0: template index (4+i for programmable, 3 for SFPSTORE)
```

## 4. Execution Model

### 4.1 Basic Flow

When `SFPLOADMACRO` is issued:

1. **Cycle 0**: The LOAD executes — reads from Dest/SrcS register file, applies format conversion, stores result into `RG[VD_FINAL]`.
2. **Cycle 1+**: The selected Sequence Register defines which additional instructions fire and when:
   - **Simple unit** — executes one "simple" class instruction (e.g., SFPSWAP, SFPSETSGN, SFPNOT, SFPAND, SFPCAST, SFPABS)
   - **MAD unit** — executes one MAD-class instruction (e.g., SFPMAD, SFPMUL24)
   - **Round unit** — executes one "round" class instruction (e.g., SFP_STOCH_RND, SFPSHFT2)
   - **Store unit** — executes an SFPSTORE

### 4.2 Instruction Classes

Every SFPU instruction belongs to exactly one class. Routing an instruction to the wrong unit (e.g., a SFPSTORE on the Simple slot) results in **undefined behavior**. SFPNOP can map to any class.

### 4.3 Pipeline Depth and Throughput

| Configuration | Throughput |
|---------------|------------|
| LOAD only (no additional ops) | 1 cycle/element |
| LOAD + Simple | 2 cycles/element |
| LOAD + Simple + Store | 2 cycles/element |
| LOAD + MAD + Round + Simple + Store | ~3–4 cycles/element |
| Full 5-wide sequence | Theoretical max ~5 IPC |

In practice, throughput is limited by data dependencies and pipeline hazards.

### 4.4 Staging Register

A special per-lane register (index 16) accessible only within LOADMACRO sequences:
- Simple and MAD units can write their results to staging (bit 6 = 1 in their slot).
- Store unit can read from staging (bit 6 = 1 in Store slot).
- **No architectural hazard protection** — the programmer must ensure no conflicts.

## 5. Hazard Management

LOADMACRO bypasses normal hardware interlocking. The programmer must manually manage:

1. **Data dependencies** — RAW hazards between the LOAD result and subsequent operations require correct delay programming.
2. **LREG write port conflicts** — Multiple units writing to the same LREG on the same cycle.
3. **Pipeline conflicts** — Multiple active LOADMACRO sequences overlapping in the pipeline.
4. **Register file bank conflicts** — SrcS and Dest bank availability.
5. **Backdoor loading hazard** — Loading an Instruction Register and using it in a LOADMACRO on the very next cycle is unsafe.
6. **Control register updates** — Updating any control register while a LOADMACRO is in flight is unsafe.

### Dependent Instruction Stalls

When `UNIT_DEPENDENCY_ENABLE` is set for a unit, the LOADMACRO sequence will stall that unit when:
- It has a pending instruction, AND
- No new instruction is being issued to the SFPU.

This is used for sequences that depend on instructions issued *after* the LOADMACRO.

## 6. Programming via SFPCONFIG

SFPCONFIG (`config_dest` field) controls which register is being written:

| config_dest | Target Register |
|-------------|-----------------|
| 0x0 | Load Macro Instruction 4 (programmable) |
| 0x1 | Load Macro Instruction 5 (programmable) |
| 0x2 | Load Macro Instruction 6 (programmable) |
| 0x3 | Load Macro Instruction 7 (programmable) |
| 0x4 | Load Macro Sequence 0 |
| 0x5 | Load Macro Sequence 1 |
| 0x6 | Load Macro Sequence 2 |
| 0x7 | Load Macro Sequence 3 |
| 0x8 | Load Macro Control Register |

SFPCONFIG also supports bitwise operations on the LOADMACRO Control Register:

| `config_mode[2:1]` | Operation |
|---------------------|-----------|
| 0 | Normal write |
| 1 | OR mask (set specific bits) |
| 2 | AND mask (clear specific bits) |
| 3 | XOR mask (invert specific bits) |

## 7. Runtime Control (TT_METAL_DISABLE_SFPLOADMACRO)

The runtime option `TT_METAL_DISABLE_SFPLOADMACRO` provides a mechanism to globally disable LOADMACRO usage:

- **Environment variable**: `export TT_METAL_DISABLE_SFPLOADMACRO=1`
- **Effect**: Passes `-DDISABLE_SFPLOADMACRO` as a compiler define to all JIT-compiled kernels.
- **Code path**: `tt_metal/llrt/rtoptions.cpp` → `tt_metal/jit_build/build.cpp`
- **Kernel behavior**: When `DISABLE_SFPLOADMACRO` is defined, kernels fall back to manual SFPLOAD/SFPSTORE sequences (visible in `ckernel_sfpu_where.h`).

```cpp
// rtoptions.hpp
bool disable_sfploadmacro = false;
bool get_disable_sfploadmacro() const { return disable_sfploadmacro; }

// build.cpp
if (rtoptions.get_disable_sfploadmacro()) {
    this->defines_ += "-DDISABLE_SFPLOADMACRO ";
}
```

## 8. Usage Examples from tt-metal Kernels

### 8.1 Exponential (exp) — `ckernel_sfpu_exp.h` (Blackhole LLK)

The Schraudolph fast exponential uses two LOADMACRO sequences:

**Sequence 1 (Sanitation)**: LOAD → SWAP → STORE
- Clamps input to [-88.5, +∞) by swapping against constant LREG[14] = -88.5

**Sequence 0 (Computation)**: LOAD → MAD → STOCHRND → SHFT → STORE
- Computes `exp(x) ≈ 2^(A*x + B-C)` via integer bit manipulation

```c
// Setup: Backdoor load of Macro Instructions via VD=12-15
TTI_SFPMAD(12, 0, 13, 13, 0);  // Captured as Instruction Register 5
// ...

// Execution: 8 sanitation + 8 computation LOADMACRO calls per tile
TTI_SFPLOADMACRO(4, 0, ADDR_MOD_7, 0);   // Seq 1: sanitize, LREG[0], offset 0
TTI_SFPNOP;                                 // NOP for SWAP's 2-cycle latency
TTI_SFPLOADMACRO(5, 0, ADDR_MOD_7, 2);   // Seq 1: sanitize, LREG[1], offset 2
// ...
TTI_SFPLOADMACRO(0, 0, ADDR_MOD_7, 0);   // Seq 0: compute, LREG[0], offset 0
TTI_SFPLOADMACRO(1, 0, ADDR_MOD_7, 2);   // Seq 0: compute, LREG[1], offset 2
```

### 8.2 Binary Max/Min — `ckernel_sfpu_binary_max_min.h`

Achieves **3 cycles per input row** for FP32, using a single LOADMACRO sequence with SWAP and STORE:

```
t | Load | Simple              | MAD | Round     | Store   |
0 | [a]  |                     |     |           |         |
1 |  b   |                     |     |           |         |
2 | [c]  | swap_minmax([a], b) |     |           |         |
0 | ...  |                     |     |           |         |
1 | ...  |                     |     | L16 = [a] |         |
2 | ...  |                     |     |           | [c] L16 |
```

```c
for (int i = 0; i < ITERATIONS; ++i) {
    int a = i & 1;
    TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset0 | (a >> 2));
    TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_7, offset1);
    TT_SFPLOADMACRO((1 << 2) | (c & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset2 | (c >> 2));
}
```

### 8.3 Typecast Operations — `ckernel_sfpu_typecast.h`

Used extensively for type conversion with throughputs from **1 cycle/row** to **4 cycles/row**:

- **uint16→fp16b**: 1 cycle/row (LOAD + Simple cast + Store all pipelined)
- **fp32→uint16**: 2 cycles/row (LOAD + Simple clamp + Round + Store)
- **int32→fp32**: 4 cycles/row (LOAD + ABS + SHFT2 + CAST + MAD + Store)
- **fp32→fp16b**: 3 cycles/row (LOAD[a] + LOAD[b] + AND + SHFT2 + Store)

### 8.4 Integer Multiply — `ckernel_sfpu_mul_int32.h`

Complex 8 cycles/row using 4 different LOADMACRO sequences and discrete instructions:

```c
TT_SFPLOAD(b0, INT32, ADDR_MOD_7, offset_in1);
TT_SFPLOADMACRO((0 << 2) | (a1 & 3), INT32, ADDR_MOD_7, offset_in0 | (a1 >> 2));  // Macro 0
TT_SFPLOADMACRO((1 << 2) | (b1 & 3), INT32, ADDR_MOD_7, offset_in1 | (b1 >> 2));  // Macro 1
TT_SFPLOAD(a0, INT32, ADDR_MOD_7, offset_in0);
TT_SFPLOADMACRO((2 << 2) | (b2 & 3), INT32, ADDR_MOD_7, offset_in1 | (b2 >> 2));  // Macro 2
TTI_SFPMUL24(...);
TTI_SFPIADD(...);
TT_SFPLOADMACRO((3 << 2) | (c & 3), INT32, ADDR_MOD_6, offset_out | (c >> 2));    // Macro 3
```

### 8.5 Unary Max/Min — `ckernel_sfpu_unary_max_min.h`

**2 cycles/row** for FP32 and non-negated INT32:
```c
for (int d = 0; d < ITERATIONS; d++) {
    int a = d & 1;
    TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_6, offset | (a >> 2));
    TTI_SFPNOP;
}
```

**4 cycles/row** for negated INT32 (adds NOT + SWAP before store).

### 8.6 Reduce Max/Min — `ckernel_sfpu_reduce.h`

Uses LOADMACRO for column-wise reduction with compare-and-swap:
```c
// Setup sequences
TTI_SFPSWAP(0, p_sfpu::LREG4, (0xC | p_sfpu::LREG0), 1);  // Backdoor to Instr Reg 4
TTI_SFPLOADI(0, 0xA, 0x0084);
TTI_SFPLOADI(0, 0x8, 0x0000);
TTI_SFPCONFIG(0, 4, 0);  // Store into Sequence Register 0

// In replay buffer
TTI_SFPLOADMACRO(5, INSTRUCTION_MODE, ADDR_MOD_7, 2);
// ... compare and swap pattern
TTI_SFPLOADMACRO(0, INSTRUCTION_MODE, ADDR_MOD_7, 0);
```

### 8.7 Dest Transpose — `llk_math_transpose_dest.h`

Uses LOADMACRO to pipeline LOAD + MOV + STORE for face transposition:
```c
macro0 = TT_OP_SFPLOADMACRO((0 << 2) | 1, 4, ADDR_MOD_1, 0x3ff & -48);
macro1 = TT_OP_SFPLOADMACRO((1 << 2) | 0, 4, ADDR_MOD_2, 0x3ff & -32);
```

### 8.8 Where (conditional select) — `ckernel_sfpu_where.h`

Provides both LOADMACRO-enabled and fallback code paths:

```c
#ifdef DISABLE_SFPLOADMACRO
    // Manual SFPLOAD/SFPSTORE sequences
#else
    // LOADMACRO-based sequences with SETCC/ENCC as macro instructions
    TT_SFPLOADMACRO((0 << 2), mod0, ADDR_MOD_7, offset0);
    TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_7, offset1);
    TT_SFPLOAD(0, mod0, ADDR_MOD_6, offset2);
#endif
```

## 9. Architecture Differences

| Feature | Wormhole B0 | Blackhole | Quasar |
|---------|-------------|-----------|--------|
| `sfpu_addr_mode` width | 2 bits (bit 14) | 3 bits (bit 13) | 3 bits (bit 13) |
| `dest_reg_addr` width | 14 bits | 13 bits | 10 bits |
| Explicit `seq_id` field | No (encoded in VD[3:2]) | No (encoded in VD[3:2]) | Yes (bits 23:22) |
| Explicit `done` field | No | No | Yes (bits 12:11) |
| `lreg_ind` split | Single 4-bit field | Single 4-bit field | Split: `lreg_ind_lo` (2b) + `lreg_ind_hi` (1b) |
| `ttsync_resource` | Not present | OTHERS | OTHERS |
| Instruction swizzle | No | No | Yes (`TRISC_OP_SWIZZLE`) |

## 10. Register File Views

LOADMACRO-related registers appear in the SFPU Configuration (RC) and Status (RS) views:

### Configuration View (RC) — Write access via SFPCONFIG

| View Index | Register |
|------------|----------|
| 0 | LOADMACRO Instruction 4 (WO) |
| 1 | LOADMACRO Instruction 5 (WO) |
| 2 | LOADMACRO Instruction 6 (WO) |
| 3 | LOADMACRO Instruction 7 (WO) |
| 4 | LOADMACRO Sequence 0 (WO) |
| 5 | LOADMACRO Sequence 1 (WO) |
| 6 | LOADMACRO Sequence 2 (WO) |
| 7 | LOADMACRO Sequence 3 (WO) |
| 8 | LOADMACRO Control Register (RW) |

### Status View (RS) — Read access

| View Index | Register |
|------------|----------|
| 0–7 | Same as above (RO) |
| 8 | LOADMACRO Control Register (RO) |

## 11. Hardware Status Outputs

The SFPU exposes status signals for external synchronization:

| Signal | Description |
|--------|-------------|
| `o_loadmacro_pending` | SFPU is currently executing a SFPLOADMACRO sequence |
| `o_loadmacro_dest_bank_wr_pending` | SFPU has a pending store from LOADMACRO to the specified Dest Regs bank |
| `o_sfpu_loadmacro_store_info` | LOADMACRO sequence configuration reported to instruction issue |

## 12. Best Practices and Pitfalls

1. **Always insert enough NOPs** between LOADMACRO sequences that share LREGs or when 2-cycle instructions (SFPSWAP, SFPMAD) are involved.

2. **Use replay buffers** (`lltt::record` / `lltt::replay`) when repeating LOADMACRO patterns across tile elements for reduced code size and instruction cache pressure.

3. **Use `#pragma GCC unroll`** on LOADMACRO loops to avoid loop overhead in tight kernels.

4. **Reset the LOADMACRO Control Register** (`TTI_SFPCONFIG(0, 8, 1)` or `TTI_SFPCONFIG(0xF00, 0x8, 0x1)`) before using macros if previous operations may have set conflicting state.

5. **Alternate LREGs** (e.g., LREG0/LREG1 on even/odd iterations) to avoid write port conflicts and enable pipelining of consecutive LOADMACRO calls.

6. **Avoid updating control registers** while a LOADMACRO is in flight.

7. **Backdoor loading** is faster than SFPCONFIG for setting up Instruction Registers, but offers **no hazard protection** — always insert at least one NOP between the backdoor load and the first SFPLOADMACRO that uses it.

8. **Use `TT_METAL_DISABLE_SFPLOADMACRO=1`** for debugging to fall back to explicit LOAD/STORE sequences (only supported in kernels that implement both paths, such as `ckernel_sfpu_where.h`).

## 13. Sources

- **Confluence**: [Tensix SFPU Instruction Set Architecture](https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/1170505767) — Authoritative ISA specification
- **Glean/SharePoint**: Wormhole_B0_SFPU_Specification.docx, Blackhole_SFPU_Specification.docx
- **Confluence**: [Using LOADMACRO Safely](https://tenstorrent.atlassian.net/wiki/spaces/TA/pages/2022408406)
- **Source code** (tt-metal repo):
  - `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole,quasar}/instructions/assembly.yaml` — Instruction encoding
  - `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole,quasar}/common/inc/ckernel_ops.h` — C macro definitions
  - `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_exp.h` — exp() LOADMACRO usage
  - `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h` — Binary max/min
  - `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_unary_max_min.h` — Unary max/min
  - `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_mul_int32.h` — INT32 multiply
  - `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_typecast.h` — Typecasting
  - `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h` — Conditional select
  - `tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_reduce.h` — Reduce max/min
  - `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_transpose_dest.h` — Dest transpose
  - `tt_metal/llrt/rtoptions.{hpp,cpp}` — Runtime disable option
  - `tt_metal/jit_build/build.cpp` — JIT compile define
