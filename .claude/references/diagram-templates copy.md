# TTNN Operation Diagram Templates

## SFPU Condition Code Diagram

Example is related to the _trunc_body_ function.

                                +---------------+
                                | Input: LREG0  |
                                | LREG3 = 23    |
                                +---------------+
                                        |
                                        |
                                +-------------------+
                                |    SFPEXEXP       |
                                | Extracts unbiased |
                                | exponent to LREG2 |
                                +-------------------+
                                        |
                                        |
                    +--------------------------------------------+
                    |                                            |
                LREG2 < 0                                    LREG2 >=0
                    |                                            |
        +-------------------------------+           +---------------------------+
        | Lane disabled                 |           | Lane enabled              |
        | Keep mask LREG1 = 0X8000_0000 |           | LREG2 = LREG3 - LREG2     |
        +-------------------------------+           |       = 23 - LREG2        |
                    |                               | LREG1 = LREG1 << LREG2    |
                    |                               +---------------------------+
                    |                                            |
                    +--------------------------------------------+
                                        |
                                        |
                            +-----------------------+
                            | LREG1 = LREG0 & LREG1 |
                            | Output: LREG1         |
                            +-----------------------+

---

## Suggested Alternative Diagrams

All alternatives below depict the same `_trunc_body_` / `_calculate_trunc_` logic
using different visual styles, each suited for different documentation contexts.

## CC State Machine

Focuses on how the condition code state evolves across instructions.
Best for complex kernels with nested or chained CC updates.

```
_trunc_body_ — CC State Transitions
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED                   ◁── initial state
       │
       │  SFPLOADI L3=23                  (no CC effect)
       │  SFPLOADI L1=0x8000_0000         (no CC effect)
       │
       ▼
  ┌─────────────────────────────────┐
  │ SFPEXEXP  SET_CC_SGN_EXP |      │
  │           SET_CC_COMP_EXP       │
  │                                 │
  │ CC ← (unbiased_exp ≥ 0)        │
  └────────────┬────────────────────┘
               │
               ▼
  CC State: ENABLED where exp ≥ 0
       │
       │  SFPLOADI L1=0xFFFF_FFFF    (CC-guarded: only exp≥0 lanes)
       │
       ▼
  ┌─────────────────────────────────┐
  │ SFPIADD  CC_GTE0                │
  │                                 │
  │ CC ← CC_prev AND (result ≥ 0)  │
  │    = (exp ≥ 0) AND (23-exp ≥ 0)│
  │    = (0 ≤ exp ≤ 23)            │
  └────────────┬────────────────────┘
               │
               ▼
  CC State: ENABLED where 0 ≤ exp ≤ 23
       │
       │  SFPSHFT2 L1 <<= L2        (CC-guarded: only 0≤exp≤23 lanes)
       │
       ▼
  ┌─────────────────────────────────┐
  │ SFPENCC                         │
  │                                 │
  │ CC ← ALL_ENABLED               │
  └────────────┬────────────────────┘
               │
               ▼
  CC State: ALL_ENABLED
       │
       │  SFPAND L1 = L0 & L1       (all lanes)
       ▼
```

---
