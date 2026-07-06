# perf-optimization-audit — check catalogue A / C / D

Every finding must pass the equivalence gate + false-positive guards in `equivalence-guards-verdict.md`. Sections B and E are in `checks-selection-and-fusion.md`.

## A. Redundant Dst / LReg traffic
1. **Multiple `dst_reg[0]` stores per element under predication.** Writing a default to Dst then overwriting active lanes in each `v_if`/`v_elseif` branch issues several SFPSTOREs. Fix: accumulate into a local `vFloat` and store **once**. (Canonical example: `ckernel_sfpu_softshrink.h` writes `dst_reg[0] = 0.0f` then overwrites in two branches — 3 stores → 1.)
2. **Store-then-immediately-reload** of the same datum into an LReg (write to Dst/LReg and trash it with a reload of the same value before any consumer reads the memory).
3. **Re-reading `dst_reg[0]` / re-indexing `dst_reg[k]`** where a single load + `dst_reg++` walk suffices.
4. **Register pressure / spills** — too many simultaneously-live `vFloat`s force spills to Dst and reloads; keep the live set inside the vector register file.
5. **Constant reloads** — re-materializing a non-trivial immediate (SFPLOADI) every iteration instead of hoisting it once. **NOT** the common hardware constants: on current `sfpi-gcc` the compiler recognizes literals like `0.0f`/`1.0f` and emits the const registers (`vConst0`/`vConst1`) itself, so open-coding them buys nothing — do **not** file "use `vConstX`" against a plain `0.0f`/`1.0f`. Confirm the pinned compiler's behavior against the freshness contract before flagging any constant-materialization finding, and scope it to genuinely reloaded non-const immediates.

## C. Loop & template structure
6. **Loop-invariant work inside the loop** — hoist above the `for (d …)`: negations (`-lambda`), converter calls (`Converter::as_float`), constant/immediate loads, any value not dependent on `d` or `dst_reg`.
7. **`#pragma GCC unroll` misuse — both directions.** Missing unroll leaves per-iteration counter/branch overhead and blocks latency overlap; forced full-unroll blows the SFPU instruction-RAM budget. Check `ITERATIONS` is a compile-time constant and the unroll factor is deliberate, not accidental.
8. **`APPROXIMATION_MODE` ignored** — kernel takes the template flag but runs the identical expensive path on both settings; the "approx" branch must actually be cheaper (fewer iterations / cheaper instructions).

## D. Latency shadows / bubbles — **raw `TTI_*` sequences only**
9. **Unfilled latency shadow** — a multi-cycle-latency producer followed by NOP padding (or a stall) where an independent instruction could be scheduled into the shadow instead. Recommend the reorder, not more NOPs.
10. **Redundant NOP** — a NOP whose latency window is already covered after a legal reorder, or by a naturally-independent following instruction. (Ground the latency/throughput window in `sfpi-gcc` + `VectorUnit.md`; never guess.)
11. **Low-throughput instruction serialization** — back-to-back uses of a low reciprocal-throughput instruction with no independent work between them, where interleaving independent lanes/iterations would overlap issue.

## F. Instruction encoding — `TT_` → `TTI_` when all operands are compile-time
12. **Runtime-encoded instruction with all-`constexpr` operands.** `TTI_<INSTR>` assembles the instruction word from **compile-time immediates**; `TT_<INSTR>` takes runtime operands and builds the word on the **RISC core at runtime** (GPR loads/shifts/ORs) before pushing. If every argument to a `TT_` call is compile-time-known (template params, `constexpr`), upgrade to `TTI_` to delete that per-call RISC-side assembly. Same Tensix instruction executes → **bit-identical** PERF-WIN; verify by disassembly (the RISC preamble vanishes). Guards: only when operands are *genuinely* compile-time (not a runtime value that looks constant at the site), a `TTI_` variant exists, and the immediate field is wide enough — never a `TT_` deliberately fed a runtime operand.
