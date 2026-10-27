---
name: perf-optimization-audit
description: Audit Tensix/SFPU LLK compute kernels for PERFORMANCE — unfilled latency shadows/bubbles and redundant NOPs, redundant Dst/LReg store-load traffic, loop-invariant work, predication that should be branchless arithmetic (min/max/abs/setsgn), un-fused mul+add, ignored APPROXIMATION_MODE, and unroll/register-pressure mistakes. Use after touching any ckernel_sfpu_*.h, hand-written TTI_SFP*/TTI_* sequence, or the compute inner loop. This is a PERF audit (wasted cycles), NOT a correctness/race audit — pair it with instruction-latency-audit.
user_invocable: true
---

# /perf-optimization-audit — Tensix/SFPU compute-kernel performance review

> **Ground-truth precedence:** the live sources — the pinned `sfpi-gcc` (latencies / throughput / `xtt_delay` / what the compiler already schedules) and the tt-isa-docs MCP `VectorUnit.md` (latency + reciprocal-throughput — **WH/BH only**; Quasar is NOT in tt-isa-docs, ground QSR HW timing via `race-audit-all`'s Quasar ladder) — outrank every rule, table, and example baked into this skill (treat those as dated illustrations). If a live source **contradicts** a baked rule here, do NOT silently proceed: surface the conflict to the user and ask whether the baked rule should be overwritten, discarded, or kept. Default to the live source.
>
> **MANDATORY — before any verdict, read the shared grounding policy.** The per-architecture **source ladder** (which docs to consult), the **ground-or-abstain** rule, and the **Source preflight** (list the sources you'll consult with their reachability + hierarchy, then PAUSE for the user) are defined once in `race-audit-all` → `.claude/skills/race-audit-all/SKILL.md`. **Your FIRST action is to `Read` that file and follow its "Ground-truth source ladder", "Ground-or-abstain", and "Source preflight" sections** — a perf verdict on a stale/guessed latency or throughput number is worthless. Timing authority = pinned **`sfpi-gcc`** (all archs) + `VectorUnit` (WH/BH) — reuse the **Fetch recipe** in `instruction-latency-audit`'s freshness contract. If `race-audit-all` can't be read, say so and **abstain**.
>
> **Coverage — floor, not ceiling.** The grep patterns and site lists here are a **seed, not an exhaustive enumeration** — widen with full reasoning (semantic search by effect, resolving macros/wrappers, call-graph, and diffing WH/BH/QSR variants — often byte-identical copies, so a win in one applies to all three; a divergence is itself a signal). State residual gaps explicitly.
>
> **Execution — parallel by default.** For more than a few kernels, **fan out concurrent `Agent` calls** (one per file/kernel-family, ~10–16 concurrency); synthesis stays sequential. Agents only **return** findings; the orchestrator is the sole writer and appends each wave incrementally. The heavyweight **Workflow** tool remains explicit-opt-in.

## Reference map — read the file for the phase you're in
Split across `references/` (each file self-contained, <4000 chars). Read as you reach each phase; never skip the provenance lens or the equivalence gate.

- **`references/overview-and-method.md`** — what this audit is / is not, the **provenance lens** (run FIRST — decides which findings are valid), and the 6-step method. **Read before starting.**
- **`references/checks-traffic-loops-shadows.md`** — catalogue **A** (Dst/LReg traffic), **C** (loop & template), **D** (latency shadows — *raw `TTI_*` only*), **F** (`TT_`→`TTI_` encoding).
- **`references/checks-selection-and-fusion.md`** — catalogue **B** (instruction selection / strength reduction — the biggest wins) and **E** (fusion / reconfig above the loop).
- **`references/equivalence-guards-verdict.md`** — the **semantic-equivalence gate**, the **false-positive guards**, and the **verdict** vocabulary.
- **`references/validation-and-output.md`** — **disassembly** proof (sfpi + `objdump`), **perf-counter** validation, and the **output** format.
