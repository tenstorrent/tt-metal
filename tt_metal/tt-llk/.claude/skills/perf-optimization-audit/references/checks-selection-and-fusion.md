# perf-optimization-audit — check catalogue B / E

Every candidate finding must pass the semantic-equivalence gate and clear the false-positive guards in `equivalence-guards-verdict.md`. Sections A/C/D are in `checks-traffic-loops-shadows.md`.

## B. Instruction selection / strength reduction (usually the biggest wins)
1. **Predication that should be branchless.** SIMD predication executes *all* branches on *all* lanes and pushes/pops a predicate stack — reordering `v_if`/`v_elseif` "common case first" saves **nothing** (see the guards). Replace with a single op where the math allows: `v_if(x<0){x=0}` → `sfpi::max(x,0)` (SFPMAX); clamp → min+max; sign copy/flip → SFPSETSGN; magnitude → SFPABS; two-way select → conditional-move/select. Flag deep or nested `v_if` trees that have an arithmetic equivalent.
2. **Un-fused multiply-add** — separate SFPMUL + SFPADD that collapse to one SFPMAD.
3. **Hardware approximation units unused — Quasar only, and only at low precision.** The LUT-backed reciprocal / rsqrt / exp / log units exist on **Quasar** (do NOT flag this on WH/BH — they lack them, so the Newton-Raphson refinement in `ckernel_sfpu_recip.h` / `ckernel_sfpu_sqrt.h` is the correct, non-wasteful path there). Even on Quasar the LUT accuracy targets **bf16-class** data — it is **not** sufficient for fp32, so the refinement must stay for fp32. Only flag when: arch is Quasar **and** the data type is bf16-class (or the op's tolerance permits the LUT result) **and** the approx path still runs full refinement.
4. **Strength reduction** — `x*x` not `pow(x,2)`; reciprocal-multiply not divide; drop format casts (fp32↔fp16, int↔float via SFPCAST) that aren't semantically required.
5. **Common-subexpression recompute** — recomputing `-lambda`, `v-lambda`, shared partial products, etc. within the loop.
6. **Hand-written polynomial eval beating a generic `PolynomialEvaluator`.** A generic Horner helper emits a *serial* FMA dependency chain (each SFPMAD waits on the previous), so its FMA-latency shadows sit empty — and the opaque helper boundary hides the surrounding independent instructions the scheduler would otherwise interleave. Two wins, different equivalence status:
    - **(a) Same Horner order, inlined/hand-interleaved** — inline so genuinely-independent surrounding work (or another datum/lane) fills the FMA shadows. **Bit-identical** (same ops, same order) → clean PERF-WIN when independent work exists. Valid on **sfpi** code because the compiler won't hoist work *across* an opaque helper call and can't invent it.
    - **(b) Restructured chain (Estrin / split-and-recombine)** — parallel sub-polynomials to expose ILP. **Reassociation changes float rounding** → NOT bit-identical; gate under the op's `APPROXIMATION_MODE` tolerance + the equivalence gate, else downgrade to SUGGESTION.
    Only flag when the polynomial is a hot path, the chain length makes its latency dominate, and there is real independent work / ILP headroom — otherwise the helper's readability wins. Confirm FMA latency/throughput against the freshness contract first.

## E. Above the inner loop (report even though the trigger is one kernel)
7. **Op fusion missed** — separate ops round-tripping Dst↔L1 that could fuse into one SFPU pass.
8. **Init / reconfig overhead** — redundant re-init or reconfig on a hot path; over-broad drains. (For the *correctness* of a drain see `reconfig-stall-audit`; here flag only provably redundant/over-broad drains as perf, never remove one needed for ordering.)
