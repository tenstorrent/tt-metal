# Diagram Templates

## CC State Machine — Generalized Template

Use this format when documenting CC state transitions in SFPU kernel analyses. The diagram traces how the Condition Code register evolves through the kernel, showing which lanes are enabled/disabled and why.

### Format Template

```
<function_name> — CC State Transitions
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED                   <-- initial state
       |
       |  <INSTRUCTION> <operands>        (no CC effect) -- <comment>
       |  <INSTRUCTION> <operands>        (no CC effect) -- <comment>
       |
       v
  +-------------------------------------+
  | <CC-MODIFYING INSTRUCTION>          |
  |   src: <register>                   |
  |                                     |
  | CC <- (<condition>)                 |
  |    = (<human-readable meaning>)     |
  +----------------+--------------------+
                   |
                   v
  CC State: ENABLED where <condition is true>
       |
       |  <INSTRUCTION> <operands>    (CC-guarded: <what happens conditionally>)
       |
       v
  +-------------------------------------+
  | SFPENCC                             |
  |                                     |
  | CC <- ALL_ENABLED                   |
  +----------------+--------------------+
                   |
                   v
  CC State: ALL_ENABLED
       |
       v  (function returns, result is in <register>)
```

### Key Elements

#### CC State Labels
Show the current CC state as a standalone line:
```
  CC State: ALL_ENABLED                   <-- initial state
  CC State: ENABLED where <condition>
  CC State: ENABLED where a != 0
  CC State: ENABLED where b_shifted == 0 (b is even)
```

#### Non-CC-Modifying Instructions
Listed with a vertical pipe, parenthetical note "(no CC effect)", and a `--` comment:
```
       |  SFPMOV L2 = L0                  (no CC effect) -- c = a
       |  SFPOR  L2 |= L1                 (no CC effect) -- c |= b
```

#### CC-Modifying Instructions (Boxed)
Drawn inside a box with the instruction, source register, the CC assignment, and a human-readable interpretation:
```
  +-------------------------------------+
  | SFPSETCC  mod1=6 (LREG_EQ0)        |
  |   src: LREG2                        |
  |                                     |
  | CC <- (LREG2 == 0)                  |
  |    = (b << d == 0)                  |
  |    = (b is even, all bits shifted   |
  |       out)                          |
  +----------------+--------------------+
```

For SFPLZ with CC_NE0:
```
  +-------------------------------------+
  | SFPLZ  L0 = clz(L0), CC_NE0        |
  |   mod1=2 (SFPLZ_MOD1_CC_NE0)       |
  |                                     |
  | CC <- (L0 != 0)                     |
  |    = (a != 0, GCD not yet found)    |
  +----------------+--------------------+
```

For SFPENCC (reset):
```
  +-------------------------------------+
  | SFPENCC                             |
  |                                     |
  | CC <- ALL_ENABLED                   |
  +----------------+--------------------+
```

#### CC-Guarded Instructions
Show with parenthetical describing the guard condition:
```
       |  SFPSWAP L0, L1, mod1=0    (CC-guarded: swap a,b only where b is even)
       |  SFPABS L2 = abs(L0)       (CC-guarded) -- L2 = +a
       |  SFPIADD L0 += L3, CC_NONE (CC-guarded: only a!=0 lanes) -- accumulated shift
```

#### Loop Sections
Delimited with `==` markers:
```
  == Replayed inner loop (30 iterations via TTI_REPLAY) ==
  Each iteration executes the 7 instructions recorded by <init_function>:

       |
       |  <instructions...>
       |
       v

  == End of replayed loop ==
```

### Accompanying Text Section

After the ASCII diagram, include a "Key CC observations" section as a bullet list:

```
**Key CC observations:**
- <Which instruction sets CC and what it means semantically>
- <Which instructions are CC-guarded and what effect that has>
- <Whether CC persists across iterations/replays>
- <How SFPENCC is used to reset CC state>
- <Any SFPIADD with CC_NONE that deliberately avoids overwriting CC>
```

### Section Header

Use this header format:
```
#### CC State Machine -- `<function_name>`
```

Followed by a one-line summary of the CC blocks in the function before the diagram.
