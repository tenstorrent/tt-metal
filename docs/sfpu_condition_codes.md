# SFPU Condition Codes

The SFPU uses **Condition Codes (CC)** to implement **Predicated Execution** — the SFPU's mechanism for per-lane conditional branching. Since the SFPU is a SIMD vector unit, it cannot branch in the traditional sense. Instead, all instructions in both "branches" are issued to all lanes, but only the lanes whose CC state is active actually commit results.

Predicated instructions still consume an issue slot and take processor time even if no lanes are enabled.

## Two Core Structures

### 1. CC Registers (per lane)

Each SFPU lane has two CC registers:

| **CC.Enable** | **CC.Result** | **Lane Active?** |
|---|---|---|
| 0 | X | Yes (predication disabled) |
| 1 | 0 | No (masked off) |
| 1 | 1 | Yes |

- **CC Enable** (`CC.En`): On/off switch for predicated execution on that lane. When 0, instructions always execute regardless of CC Result.
- **CC Result** (`CC.Res`): When predication is enabled, determines whether the lane is active (1) or masked (0).

CC Registers alone support simple 2-way if/else branching. Example:

```
SFPENCC  // Enable Predicated Execution
SFPLE    // Test if X < 0 -> sets CC.Res per lane
SFPMOV   // Set X = 0 (only executes where test was true)
SFPCOMPC // Flip CC.Res -> now true where X >= 0
SFPMULI  // Scale by constant (only executes where X >= 0)
SFPENCC  // Disable Predicated Execution
```

This is equivalent to the following C pseudocode:

```c
if (X < 0) {
  X = 0;
} else {
  X = X * 1.1;
}
```

### 2. CC Stack (per lane, 8 entries deep)

Each lane has an **8-entry stack** that stores `{CC.En, CC.Res}` pairs. This enables:

- **Nested conditionals**: Save/restore CC state across inner branches
- **Compound boolean tests**: Accumulate results from multiple tests (AND, OR, etc.) before executing conditional code
- **If/else-if/else chains**: Combine push/pop/complement operations

#### Nested conditional example

C pseudocode:

```c
if (X > 0) {
  if (Y > 10) {
    X = X * 10;
  } else {
    X = X * Y;
  }
} else {
  X = 0;
}
```

SFPU assembly:

```
SFPENCC  // Enable predication
SFPLE    // Test X <= 0 (the else condition of the outer loop)
SFPPUSHC // Save the result of that test before moving on to the inner test
SFPGT    // Test if Y > 10
SFPMULI  // Do the fixed multiplication by 10 if the inner test was true
SFPCOMPC // Flip the CC Result to perform the else clause of the inner branch
SFPMUL   // Do the variable multiplication by Y if the inner test was false
SFPPOPC  // Restore the original CC Result
SFPMOV   // If the first test failed, overwrite the inner loop's result with 0
SFPENCC  // Disable predication
```

#### Compound boolean test example

C pseudocode:

```c
if ((X > 0) && (Y < 0)) {
  Z = X + 5;
} else {
  Z = Y - 3;
}
```

SFPU assembly:

```
SFPENCC  // Enable predication
SFPGT    // Test X > 0
SFPPUSHC // Save the intermediate result of the test
SFPLE    // Test Y < 0
SFPPOPC  // Pop the saved result for X > 0 off the stack & merge it with the result from Y < 0
SFPADDI  // Add 5 to X if both tests were true
SFPCOMPC // Flip the CC Result to perform the else clause
SFPADDI  // Subtract 3 from Y if either test was false
SFPENCC  // Disable predication
```

These two techniques leveraging the CC Stack can be further combined to achieve even more complex control flows so long as the nesting depth doesn't exceed the stack size of 8.

---

## CC Manipulation Instructions

### SFPENCC (opcode 0x8A)

| **Opcode** | **Encoding** | **Input Formats** | **Output Formats** | **IPC** | **Latency** | **Sets CC Result?** | **Sets CC Enable?** | **Sets Exception Flags?** | **Flushes Subnormals?** |
|---|---|---|---|---|---|---|---|---|---|
| 0x8A | O2 | N/A | N/A | 1 | 1 | Y | Y | N | N |

Directly sets `CC.En` and `CC.Res` based on instruction inputs. **Executes on all lanes unconditionally** regardless of the current `LaneEnabled` state.

#### Field Descriptions

- `Imm12`: Immediate value to specify new CC state.
- `InstrMod[3]`: CC Result Source Select
  - `0`: Set CC Result to 1
  - `1`: Set CC Result based on `Imm12[1]`
- `InstrMod[2]`: Reserved
- `InstrMod[1:0]`: CC Enable Source Select
  - `0`: Keep previous CC Enable value
  - `1`: Invert the current CC Enable value
  - `2`: Set CC Enable based on `Imm12[0]`
  - `3`: Reserved

#### Algorithmic Implementation

```
if (InstrMod[3]) {
  CC.Res = Imm12[1];
} else {
  CC.Res = 1;
}

if (InstrMod[1:0] == 1) {
  CC.En = !CC.En
} else if (InstrMod[1:0] == 2) {
  CC.En = Imm12[0];
}
```

---

### SFPSETCC (opcode 0x7B)

| **Opcode** | **Encoding** | **Input Formats** | **Output Formats** | **IPC** | **Latency** | **Sets CC Result?** | **Sets CC Enable?** | **Sets Exception Flags?** | **Flushes Subnormals?** |
|---|---|---|---|---|---|---|---|---|---|
| 0x7B | O2 | FP32 INT32 SMAG32 | N/A | 1 | 1 | Y | N | N | N |

Sets `CC.Res` based on the value in either `Imm12` or `RG[VC]`. **Execution of this instruction itself is predicated** by the current `LaneEnabled` state (unlike the other CC instructions).

#### Field Descriptions

- `Imm12[11]`: Source Format Select
  - `0`: Treat `RG[VC]` as an INT32 value
  - `1`: Treat `RG[VC]` as a FP32 or SMAG32 value
- `Imm12[10:1]`: Reserved
- `Imm12[0]`: Immediate value to specify new CC state.
- `InstrMod`: CC Result Action
  - `0`: Set CC Result if `RG[VC]` is negative
  - `1`: Set CC Result based on `Imm12[0]`
  - `2`: Set CC Result if `RG[VC]` is not 0
  - `3`: Reserved
  - `4`: Set CC Result if `RG[VC]` is positive
  - `5`: Reserved
  - `6`: Set CC Result if `RG[VC]` is 0
  - `7`: Reserved
  - `8`: Invert the current CC Result value
  - `9-15`: Reserved

#### Algorithmic Implementation

```
if (LaneEnabled) {
  if (CC.En) {
    bool CmpRes;
    if (Imm12[11]) {
      CmpRes = RG[VC].SMAG32 == 0;
    } else {
      CmpRes = RG[VC].INT32 == 0;
    }

    switch(InstrMod) {
      0: CC.Res = RG[VC].Sgn; break;
      1: CC.Res = Imm12[0]; break;
      2: CC.Res = !CmpRes; break;
      4: CC.Res = !RG[VC].Sgn; break;
      6: CC.Res = CmpRes; break;
      8: CC.Res = !CC.Res; break;
    }
  } else {
    CC.Res = 0;
  }
}
```

> **Note**: Since FP32 tests overload onto the SMAG32 format for this instruction, tests for zero check for true zero. Subnormal values will **not** be treated as zero.

---

### SFPCOMPC (opcode 0x8B)

| **Opcode** | **Encoding** | **Input Formats** | **Output Formats** | **IPC** | **Latency** | **Sets CC Result?** | **Sets CC Enable?** | **Sets Exception Flags?** | **Flushes Subnormals?** |
|---|---|---|---|---|---|---|---|---|---|
| 0x8B | O2 | N/A | N/A | 1 | 1 | Y | N | N | N |

Conditionally complements or clears `CC.Res`, depending on the current CC state. **Executes on all lanes unconditionally** regardless of the current `LaneEnabled` state. Used for implementing `else` clauses.

#### Field Descriptions

- `Imm12`: Reserved
- `InstrMod`: Reserved

#### Algorithmic Implementation

```
if (CC.En) {
  if (CCStack.size() == 0) {
    CC.Res = !CC.Res;
  } else if (CCStack[0].En && CCStack[0].Res) {
    CC.Res = !CC.Res;
  } else {
    CC.Res = 0;
  }
} else {
  CC.Res = 0;
}
```

The stack-aware logic ensures that complement respects the nesting context: lanes masked by an outer conditional stay masked even during an inner `else`.

---

### SFPPUSHC (opcode 0x87)

| **Opcode** | **Encoding** | **Input Formats** | **Output Formats** | **IPC** | **Latency** | **Sets CC Result?** | **Sets CC Enable?** | **Sets Exception Flags?** | **Flushes Subnormals?** |
|---|---|---|---|---|---|---|---|---|---|
| 0x87 | O2 | N/A | N/A | 1 | 1 | Y | N | N | N |

Pushes the current CC state onto the CC Stack, either creating a new stack entry or modifying the existing entry at the top of the stack. May additionally alter the current CC state. **Executes on all lanes unconditionally** regardless of the current `LaneEnabled` state.

#### Field Descriptions

- `Imm12`: Reserved
- `InstrMod`: Operation Select

| **InstrMod** | **Stack Action** | **Stack Enable** | **Stack Result** | **CC Result** |
|---|---|---|---|---|
| `0x0` | Push | `CC.En` | `CC.Res` | No Change |
| `0x1` | Replace | `CC.En` | `CC.Res` | No Change |
| `0x2` | Replace | `CC.En` | `!CC.Res` | No Change |
| `0x3` | Replace | `CC.En` | `CCStack[0].Res && CC.Res` | No Change |
| `0x4` | Replace | `CC.En` | `CCStack[0].Res \|\| CC.Res` | No Change |
| `0x5` | Replace | `CC.En` | `CCStack[0].Res && !CC.Res` | No Change |
| `0x6` | Replace | `CC.En` | `CCStack[0].Res \|\| !CC.Res` | No Change |
| `0x7` | Replace | `CC.En` | `!CCStack[0].Res && CC.Res` | No Change |
| `0x8` | Replace | `CC.En` | `!CCStack[0].Res \|\| CC.Res` | No Change |
| `0x9` | Replace | `CC.En` | `!CCStack[0].Res && !CC.Res` | No Change |
| `0xA` | Replace | `CC.En` | `!CCStack[0].Res \|\| !CC.Res` | No Change |
| `0xB` | Replace | `CC.En` | `CCStack[0].Res != CC.Res` | No Change |
| `0xC` | Replace | `CC.En` | `CCStack[0].Res == CC.Res` | No Change |
| `0xD` | Replace | `CC.En` | `!CC.Res` | `!CC.Res` |
| `0xE` | Replace | `1` | `1` | No Change |
| `0xF` | Replace | `1` | `0` | No Change |

#### Algorithmic Implementation

```
switch (InstrMod) {
  4'h0:
    for (int entry = CCStack.size(); entry > 0; entry--) {
      CCStack[entry] = CCStack[entry-1];
    }
    CCStack[0].En  = CC.En;
    CCStack[0].Res = CC.Res;
    break;
  4'h1:
    CCStack[0].En  = CC.En;
    CCStack[0].Res = CC.Res;
    break;
  4'h2:
    CCStack[0].En  = CC.En;
    CCStack[0].Res = !CC.Res;
    break;
  4'h3:
    CCStack[0].En  = CC.En;
    CCStack[0].Res = CCStack[0].Res && CC.Res;
    break;
  4'h4:
    CCStack[0].En  = CC.En;
    CCStack[0].Res = CCStack[0].Res || CC.Res;
    break;
  4'h5:
    CCStack[0].En  = CC.En;
    CCStack[0].Res = CCStack[0].Res && !CC.Res;
    break;
  4'h6:
    CCStack[0].En  = CC.En;
    CCStack[0].Res = CCStack[0].Res || !CC.Res;
    break;
  4'h7:
    CCStack[0].En  = CC.En;
    CCStack[0].Res = !CCStack[0].Res && CC.Res;
    break;
  4'h8:
    CCStack[0].En  = CC.En;
    CCStack[0].Res = !CCStack[0].Res || CC.Res;
    break;
  4'h9:
    CCStack[0].En  = CC.En;
    CCStack[0].Res = !CCStack[0].Res && !CC.Res;
    break;
  4'hA:
    CCStack[0].En  = CC.En;
    CCStack[0].Res = !CCStack[0].Res || !CC.Res;
    break;
  4'hB:
    CCStack[0].En  = CC.En;
    CCStack[0].Res = CCStack[0].Res != CC.Res;
    break;
  4'hC:
    CCStack[0].En  = CC.En;
    CCStack[0].Res = CCStack[0].Res == CC.Res;
    break;
  4'hD:
    CCStack[0].En  = CC.En;
    CCStack[0].Res = !CC.Res;
    CC.Res         = !CC.Res;
    break;
  4'hE:
    CCStack[0].En  = 1;
    CCStack[0].Res = 1;
    break;
  4'hF:
    CCStack[0].En  = 1;
    CCStack[0].Res = 0;
    break;
}
```

---

### SFPPOPC (opcode 0x88)

| **Opcode** | **Encoding** | **Input Formats** | **Output Formats** | **IPC** | **Latency** | **Sets CC Result?** | **Sets CC Enable?** | **Sets Exception Flags?** | **Flushes Subnormals?** |
|---|---|---|---|---|---|---|---|---|---|
| 0x88 | O2 | N/A | N/A | 1 | 1 | Y | Y | N | N |

Pops the top of the CC Stack to modify the CC state. Optionally retains the entry at the top of the CC Stack (peek). **Executes on all lanes unconditionally** regardless of the current `LaneEnabled` state.

#### Field Descriptions

- `Imm12`: Reserved
- `InstrMod`: Operation Select

| **InstrMod** | **Stack Action** | **CC Enable** | **CC Result** |
|---|---|---|---|
| `0x0` | Pop | `CCStack[0].En` | `CCStack[0].Res` |
| `0x1` | Peek | `CCStack[0].En` | `CCStack[0].Res` |
| `0x2` | Peek | `CCStack[0].En` | `!CCStack[0].Res` |
| `0x3` | Peek | `CC.En` | `CCStack[0].Res && CC.Res` |
| `0x4` | Peek | `CC.En` | `CCStack[0].Res \|\| CC.Res` |
| `0x5` | Peek | `CC.En` | `!CCStack[0].Res && CC.Res` |
| `0x6` | Peek | `CC.En` | `!CCStack[0].Res \|\| CC.Res` |
| `0x7` | Peek | `CC.En` | `CCStack[0].Res && !CC.Res` |
| `0x8` | Peek | `CC.En` | `CCStack[0].Res \|\| !CC.Res` |
| `0x9` | Peek | `CC.En` | `!CCStack[0].Res && !CC.Res` |
| `0xA` | Peek | `CC.En` | `!CCStack[0].Res \|\| !CC.Res` |
| `0xB` | Peek | `CC.En` | `CCStack[0].Res != CC.Res` |
| `0xC` | Peek | `CC.En` | `CCStack[0].Res == CC.Res` |
| `0xD` | Peek | `0` | `0` |
| `0xE` | Peek | `1` | `1` |
| `0xF` | Peek | `1` | `0` |

#### Algorithmic Implementation

```
switch (InstrMod) {
  4'h0:
    CC.En  = CCStack[0].En;
    CC.Res = CCStack[0].Res;

    for (int entry = 0; entry < CCStack.size(); entry++) {
      CCStack[entry] = CCStack[entry+1];
    }
    break;
  4'h1:
    CC.En  = CCStack[0].En;
    CC.Res = CCStack[0].Res;
    break;
  4'h2:
    CC.En  = CCStack[0].En;
    CC.Res = !CCStack[0].Res;
    break;
  4'h3:
    CC.En  = CCStack[0].En;
    CC.Res = CCStack[0].Res && CC.Res;
    break;
  4'h4:
    CC.En  = CCStack[0].En;
    CC.Res = CCStack[0].Res || CC.Res;
    break;
  4'h5:
    CC.En  = CCStack[0].En;
    CC.Res = !CCStack[0].Res && CC.Res;
    break;
  4'h6:
    CC.En  = CCStack[0].En;
    CC.Res = !CCStack[0].Res || CC.Res;
    break;
  4'h7:
    CC.En  = CCStack[0].En;
    CC.Res = CCStack[0].Res && !CC.Res;
    break;
  4'h8:
    CC.En  = CCStack[0].En;
    CC.Res = CCStack[0].Res || !CC.Res;
    break;
  4'h9:
    CC.En  = CCStack[0].En;
    CC.Res = !CCStack[0].Res && !CC.Res;
    break;
  4'hA:
    CC.En  = CCStack[0].En;
    CC.Res = !CCStack[0].Res || !CC.Res;
    break;
  4'hB:
    CC.En  = CCStack[0].En;
    CC.Res = CCStack[0].Res != CC.Res;
    break;
  4'hC:
    CC.En  = CCStack[0].En;
    CC.Res = CCStack[0].Res == CC.Res;
    break;
  4'hD:
    CC.En  = 0;
    CC.Res = 0;
    break;
  4'hE:
    CC.En  = 1;
    CC.Res = 1;
    break;
  4'hF:
    CC.En  = 1;
    CC.Res = 0;
    break;
}
```

---

## Key Design Properties

- **All CC instructions use O2 encoding**, have **IPC = 1** and **latency = 1**.
- **SFPENCC, SFPCOMPC, SFPPUSHC, SFPPOPC** execute unconditionally on all lanes (they bypass `LaneEnabled`).
- **SFPSETCC** is the exception — it respects the current `LaneEnabled` state.
- Several other instructions (e.g., `SFPLE`, `SFPGT`) also set `CC.Res` as a **side effect**, enabling direct use in conditional flows without a separate `SFPSETCC`.
- None of the CC instructions set exception flags or flush subnormals.
- The stack depth of **8 entries** limits nesting to 8 levels of saved conditional state.
