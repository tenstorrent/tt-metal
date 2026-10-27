# perf-optimization-audit — check catalogue A / C / D

Every finding must pass the equivalence gate + false-positive guards in `equivalence-guards-verdict.md`. Sections B and E are in `checks-selection-and-fusion.md`.

## A. Redundant Dst / LReg traffic
1. **Multiple `dst_reg[0]` stores per element under predication.** Writing a default to Dst then overwriting active lanes in each `v_if`/`v_elseif` branch issues several SFPSTOREs. Fix: accumulate into a local `vFloat` and store **once**. (Canonical: `ckernel_sfpu_softshrink.h` — 3 stores → 1.)
2. **Dead-initialized vector locals** — `vFloat a = 0.0f;` (or any init) unconditionally overwritten before its first read still emits a real SFPLOADI/SFPMOV: **`sfpi-gcc` does NOT dead-store-eliminate vector locals**, so the wasted load stays. Drop the initializer (declare, then assign) or init directly with the first real value. Common in `v_if` scaffolding where a default is set then every lane is reassigned. Confirm the store is dead on **all** paths; verify removal in the disassembly.
3. **Store-then-immediately-reload** — write to Dst/LReg and trash it by reloading the same value before any consumer reads it.
4. **Re-reading `dst_reg[0]` / re-indexing `dst_reg[k]`** where a single load + `dst_reg++` walk suffices.
5. **Register pressure / spills** — too many live `vFloat`s force spills to Dst and reloads; keep the live set within the register file.
6. **Constant reloads** — re-materializing a non-trivial immediate (SFPLOADI) every iteration instead of hoisting it once. **NOT** the common hardware constants: current `sfpi-gcc` lowers `0.0f`/`1.0f` to the const registers (`vConst0`/`vConst1`) itself, so do **not** file "use `vConstX`" against a plain literal. Scope to reloaded non-const immediates.

## C. Loop & template structure
7. **Loop-invariant work inside the loop** — hoist above the `for (d …)`: negations (`-lambda`), converter calls (`Converter::as_float`), constant/immediate loads, any value not dependent on `d` or `dst_reg`.
8. **`#pragma GCC unroll` misuse — both directions.** Missing unroll leaves per-iteration counter/branch overhead and blocks latency overlap; forced full-unroll blows the SFPU instruction-RAM budget. Check `ITERATIONS` is compile-time and the unroll factor is deliberate.
9. **`APPROXIMATION_MODE` ignored** — kernel takes the template flag but runs the identical expensive path on both settings; the "approx" branch must actually be cheaper (fewer iterations / cheaper instructions).

## D. Latency shadows / bubbles — **raw `TTI_*` sequences only**
10. **Unfilled latency shadow** — a multi-cycle-latency producer followed by NOP padding (or a stall) where an independent instruction could be scheduled into the shadow instead. Recommend the reorder, not more NOPs.
11. **Redundant NOP** — a NOP whose latency window is already covered after a legal reorder, or by a naturally-independent following instruction. (Ground the window in `sfpi-gcc` + `VectorUnit.md`; never guess.)
12. **Low-throughput instruction serialization** — back-to-back uses of a low reciprocal-throughput instruction with no independent work between them, where interleaving independent work would overlap issue.

## F. Instruction encoding — `TT_` → `TTI_` when all operands are compile-time
13. **Runtime-encoded instruction with all-`constexpr` operands.** `TTI_<INSTR>` assembles the instruction word from **compile-time immediates**; `TT_<INSTR>` takes runtime operands and builds the word on the **RISC core at runtime** (GPR loads/shifts/ORs). If every argument to a `TT_` call is compile-time-known (`constexpr`/template params), upgrade to `TTI_` to delete that per-call RISC-side assembly. Same Tensix instruction executes → **bit-identical** PERF-WIN; verify by disassembly. Guards: only when operands are *genuinely* compile-time and a `TTI_` variant exists with a wide-enough immediate — never a `TT_` deliberately fed a runtime operand.
