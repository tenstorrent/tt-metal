# perf-optimization-audit — overview & method

(Read the parent `SKILL.md` grounding block first — the freshness contract and source preflight are mandatory.)

## What this audit is (and is not)
A **performance** audit of Tensix compute kernels: it finds cycles wasted on bubbles, redundant memory/register traffic, work that shouldn't be in the loop, and instruction sequences that a cheaper equivalent computes identically. It is the perf counterpart to the nine correctness audits under `race-audit-all`, which have **no** performance coverage today.

It is **NOT** a correctness audit. Every finding must be **numerically equivalent** to the original (see the equivalence gate in `equivalence-guards-verdict.md`). Where a latency fact overlaps with `instruction-latency-audit`, that audit owns the *correctness* verdict (missing NOP → silent corruption); this audit owns the *perf* verdict (a bubble that could hold useful work, or a NOP that is provably redundant). Cross-link, never contradict.

## Provenance lens — run this FIRST (it decides which findings are even valid)
- **sfpi-compiled code** (`vFloat` / `dst_reg[...]` / `sfpi::` ops → RTL → `rtl-rvtt-schedule.cc`): the compiler **already** schedules instructions and inserts/omits NOPs, does register allocation, and does basic scheduling. So on sfpi code:
  - **Do NOT file "add a NOP" / "manually interleave to hide this stall" findings** — that is the compiler's job and hand-doing it is usually noise or a pessimization. Manual scheduling findings apply to **raw `TTI_*` / inline-asm** sequences only.
  - **DO file algorithmic findings** — fewer instructions, fewer Dst stores, hoisting, branchless rewrites, builtins, FMA, approx-mode gating. These are what the compiler cannot do for you and where the real wins are.
- **Hand-written `TTI_SFP*` / `TTI_*` / inline-asm / raw opcode pushes**: bypass the scheduler entirely → manual instruction interleaving to fill latency shadows (and removing NOPs made redundant by reordering) **is** valid here. This is the only place the "hide the bubble" class lives.

## Method
1. **Establish grounding** (freshness contract): resolve the pinned `sfpi-gcc` (all three archs); load HW latency + throughput from tt-isa-docs `VectorUnit.md` for **WH and BH separately** — **Quasar is not in tt-isa-docs**, so ground QSR HW timing in its SFPU uArch source via `race-audit-all`'s Quasar ladder; note what the compiler already schedules. **Provision the toolchain for disassembly** — run `tests/setup_testing_env.sh` to fetch the pinned sfpi (compiler + `objdump`) into `tests/sfpi/`, and confirm `tests/sfpi/sfpi.version` matches the pin (flag any mismatch — an assembly diff from the wrong compiler is not evidence). State versions consulted.
2. **Enumerate candidate kernels / sequences:**
   ```bash
   cd tt_metal/tt-llk
   # sfpi compute kernels (algorithmic findings) + hand-written sequences (scheduling findings)
   grep -rInE "dst_reg\[|v_if|v_elseif|#pragma GCC unroll|APPROXIMATION_MODE|Converter::as_float|PolynomialEvaluator|\bTT[I]?_[A-Z0-9_]+|sfpnop" \
     tt_llk_* --include=*.h | grep -v /tests/
   ```
   Also sweep the metal-side copies: `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_*.h`.
3. **Classify each block by provenance** (sfpi vs raw). This gates which checks (A–C/E vs D) apply.
4. **Run the check catalogue** (`checks-traffic-loops-shadows.md` + `checks-selection-and-fusion.md`) against each block; for every candidate finding, **pass the semantic-equivalence gate** and clear the **false-positive guards** (`equivalence-guards-verdict.md`) before recording it.
5. **Estimate the win** — instructions/stores removed per element × ITERATIONS × tiles, or bubble cycles reclaimed. Prefer a magnitude class (large / moderate / marginal) over false precision.
6. **Diff WH/BH/QSR variants** — a fix to a byte-identical copy applies to all; a divergence may itself be the finding.
