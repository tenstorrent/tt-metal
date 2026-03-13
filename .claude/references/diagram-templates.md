# TTNN Operation Diagram Templates

---

## CC State Machine — Generalized Template


### Construction Rules

0. **Account for implicit CC manipulation by TT_ and TTI_ instructions.**
   Several SFPU instructions set `CC.Res` as a **side effect** of their primary
   operation, not just the dedicated CC instructions (`SFPSETCC`, `SFPENCC`, etc.).
   Both the `TT_` (buffered/replay) and `TTI_` (immediate/inline) forms of the
   same instruction produce the **same opcode** and therefore the **same CC behavior**.

   **Default-ON** — these instructions update CC **unless explicitly opted out**
   via a modifier bit:

   | Instruction | Default CC behavior | Opt-out modifier |
   |-------------|---------------------|------------------|
   | `SFPIADD`   | `CC.Res ← (result < 0)` | `InstrMod[2]=1` (`CC_NONE`) skips CC update |
   | `SFPEXEXP`  | `CC.Res ← (result < 0)` | `InstrMod[1]=1` skips CC update |

   **Opt-IN** — these instructions update CC only when explicitly enabled:

   | Instruction | CC behavior when enabled | Enable modifier |
   |-------------|-------------------------|-----------------|
   | `SFPLE`     | `CC.Res ← (VD ≤ VC)` | `InstrMod[0]=1` enables CC update |
   | `SFPGT`     | `CC.Res ← (VD > VC)` | `InstrMod[0]=1` enables CC update |

   Both `SFPIADD` and `SFPEXEXP` also support `InstrMod[3]=1` to **invert** the
   CC result after the update (`CC.Res ← !CC.Res`).

   **When building a CC State Machine diagram:**
   - For every `TT_`/`TTI_` instruction in the sequence, check whether it has
     implicit CC side effects (consult the table above or the ISA "Sets CC Result?"
     column).
   - If an instruction sets CC by default (SFPIADD, SFPEXEXP), treat it as a
     CC-modifying instruction and **box it** — unless the modifier explicitly
     disables CC update (e.g., `SFPIADD_MOD1_CC_NONE`).
   - If an instruction has opt-in CC (SFPLE, SFPGT), only box it when the
     enable modifier is present.
   - Instructions that do **not** set CC (`SFPMAD`, `SFPADD`, `SFPMUL`, `SFPMULI`,
     `SFPADDI`, `SFPAND`, `SFPOR`, `SFPSHFT2`, `SFPLOADI`, `SFPLOAD`, `SFPSTORE`,
     `SFPMOV`, `SFPNOP`) are always safe to annotate as `(no CC effect)`.


2. **Group consecutive non-CC instructions** between state transitions. List them
   as indented lines with `│` continuation and `(no CC effect)` annotation:
   ```
        │
        │  INSTRUCTION arg1, arg2       (no CC effect)
        │  INSTRUCTION arg1, arg2       (no CC effect)
        │
   ```

3. **Box every CC-modifying instruction** — any instruction with a `SET_CC` or
   `CC_*` modifier. Inside the box:
   - Line 1: instruction mnemonic + modifier flags
   - Line 2 (optional): expanded modifier meaning
   - Line 3: CC transition formula using `CC ←`
   - For chained CC updates, show the AND composition:
     `CC ← CC_prev AND (new_condition)`
   ```
   ┌─────────────────────────────────┐
   │ INSTRUCTION  MODIFIER_FLAGS     │
   │                                 │
   │ CC ← (condition on result)     │
   └────────────┬────────────────────┘
   ```

4. **Write the new CC state** immediately after each box as a plain-text label:
   ```
   CC State: ENABLED where <human-readable predicate>
   ```

5. **Annotate CC-guarded instructions** (instructions that execute under current
   CC without modifying it) as indented lines between state labels, with a
   parenthetical noting which lanes are affected:
   ```
        │
        │  INSTRUCTION arg1, arg2    (CC-guarded: only <predicate> lanes)
        │
   ```

6. **SFPENCC always resets to ALL_ENABLED** — box it and write the reset:
   ```
   ┌─────────────────────────────────┐
   │ SFPENCC                         │
   │                                 │
   │ CC ← ALL_ENABLED               │
   └────────────┬────────────────────┘
                │
                ▼
   CC State: ALL_ENABLED
   ```

7. **End with unconditional instructions** after the final `ALL_ENABLED` state.

8. **For nested CC blocks** (SFPENCC inside a conditional region, or multiple
   independent CC regions in sequence), repeat steps 2–6 for each block,
   separated by the reset.

### Skeleton Template

Copy and fill in the placeholders (`<...>`):

```
<function_name> — CC State Transitions
════════════════════════════════════════════════════════════════

  CC State: ALL_ENABLED                   ◁── initial state
       │
       │  <unconditional instructions>    (no CC effect)
       │
       ▼
  ┌─────────────────────────────────┐
  │ <INSTRUCTION>  <CC_MODIFIERS>   │
  │                                 │
  │ CC ← (<condition₁>)            │
  └────────────┬────────────────────┘
               │
               ▼
  CC State: ENABLED where <condition₁>
       │
       │  <guarded instructions>     (CC-guarded: only <condition₁> lanes)
       │
       ▼
  ┌─────────────────────────────────┐
  │ <INSTRUCTION>  <CC_MODIFIERS>   │
  │                                 │
  │ CC ← CC_prev AND (<condition₂>)│
  │    = (<condition₁ ∧ condition₂>)│
  └────────────┬────────────────────┘
               │
               ▼
  CC State: ENABLED where <condition₁ ∧ condition₂>
       │
       │  <guarded instructions>     (CC-guarded: only <cond₁ ∧ cond₂> lanes)
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
       │  <unconditional instructions>    (all lanes)
       ▼
```

### Extending for Multiple Independent CC Blocks

Some kernels have several SET_CC → SFPENCC regions in sequence (e.g., `_floor_body_`
calls `_trunc_body_` then has its own CC block). Chain them vertically:

```
  ── Block 1 ──────────────────────────
  CC State: ALL_ENABLED
       │  ...
  CC State: ENABLED where <predicate_A>
       │  ...
  CC State: ALL_ENABLED               ◁── SFPENCC reset
       │
  ── Block 2 ──────────────────────────
       │  (unconditional instructions between blocks)
       │
  CC State: ENABLED where <predicate_B>   ◁── new SET_CC
       │  ...
  CC State: ALL_ENABLED               ◁── SFPENCC reset
```
