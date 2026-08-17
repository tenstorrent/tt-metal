---
name: instruction-latency-audit
description: Audit hand-written Tensix/SFPU instruction sequences for missing pipeline-latency padding — where a dependent instruction consumes a multi-cycle-latency result before it is ready and a NOP (or independent-instruction spacing) is required. Use after touching any raw TTI_SFP*/TTI_* sequence, ckernel_sfpu_* kernels, or hand-assembled instruction streams. NOT a cross-thread race — an intra-thread micro-architectural hazard.
user_invocable: true
---

# /instruction-latency-audit — pipeline result-latency / NOP-padding hazard

> **Ground-truth precedence:** the live sources — the pinned `sfpi-gcc` (latencies / `xtt_delay` / `xtt_dynamic_bug`) and the tt-isa-docs MCP — outrank every rule, table, and example baked into this skill (treat those as dated illustrations). If a live source **contradicts** a baked rule here, do NOT silently proceed: surface the conflict to the user and ask whether the baked rule should be overwritten, discarded, or kept. Default to the live source.
>
> **MANDATORY — before any verdict, read the shared grounding policy.** The per-architecture **source ladder** (which docs to consult), the **ground-or-abstain** rule, and the **Source preflight** (list the sources you'll consult with their reachability + hierarchy, then PAUSE for the user) are defined once in `race-audit-all` → `.claude/skills/race-audit-all/SKILL.md`. **Your FIRST action is to `Read` that file and follow its "Ground-truth source ladder", "Ground-or-abstain", and "Source preflight" sections** — they are load-bearing: a verdict produced without them is ungrounded and MUST NOT be reported. (This audit's own authority is the pinned **`sfpi-gcc` source** — see the **Fetch recipe** in the freshness contract below.) If `race-audit-all` genuinely cannot be read, say so and **abstain** rather than proceed ungrounded. (If you were spawned by a `race-audit-all` sweep — your prompt already lists the confirmed sources — skip the Source preflight and do not pause; the orchestrator ran it once.)
>
> **Coverage — floor, not ceiling.** The grep patterns and site lists in this skill are a **seed, not an exhaustive enumeration**. After running them, widen the search with full reasoning. The techniques here are **illustrative examples, not the allowed set** — use any approach your reasoning suggests, including ones not listed: e.g. semantic search (by behavior/effect, not just token), resolving macros / wrappers / typedefs / indirection the literal pattern can't match, following the call graph to callers and callees, and diffing the WH/BH/QSR variants to catch a site present in one arch and missing in another. If you can find a hazard, primitive, or site the encoded patterns don't cover — by any means — pursue and report it; do **not** clamp a stronger analysis to this list or to these techniques. State any residual coverage gaps explicitly (no silent caps).
>
> **Execution — parallel by default.** When enumeration yields more than a few sites/files, **fan out concurrent `Agent` calls by default** (one per file/subsystem, a fresh context each), saturating the available concurrency (~10–16 at once); go inline only for a trivial set. The per-file fan-out described under *Thoroughness* is the **default**, not an exhaustive-only option. The cross-referencing/synthesis of results stays sequential (it must follow the per-unit findings). The heavyweight **Workflow** tool still requires explicit multi-agent opt-in — it is the opt-in exhaustive tier, not the default. Don't over-spawn a tiny diff.
>
> **Persisting results — single writer, incremental.** Agents only **return** their findings; they never write a shared file (no concurrent-write clobbering). If findings are persisted to a file, the orchestrator/caller is the **sole writer** and **appends each wave's returns as they arrive** — incremental, never only-at-the-end — so an interrupt preserves every completed wave's findings.

## The bug class (precise)
Within a **single** Tensix thread's instruction stream, some instructions take more than one cycle for their result to become available. If a later instruction in the same stream consumes that result before it has landed, the consumer latches a **stale value → silent numerical corruption** (no deadlock, no crash). This is NOT a race between agents — it is an intra-thread, intra-unit pipeline hazard. The fix is a **NOP** (`SFPNOP` / `TTI_NOP`) — or scheduling an independent instruction into the latency shadow — between producer and dependent consumer.

## ⚠️ Ground-truth freshness contract (read before deriving anything)
The decisive facts (per-instruction latencies, which arch's hardware auto-handles the hazard, and which instructions are exempt) live in the **`sfpi-gcc` compiler**, NOT the ISA docs — and they change across compiler versions. Therefore:
- **Consult the live source every run; never trust a list reproduced in this document.** Any instruction names / counts below are *dated illustrations for calibration*, subordinate to fresh derivation.
- Ground latency facts in the **`sfpi` version this build is pinned to** (resolve the pin from tt-metal's build/toolchain config) — that is what compiled the shipped kernels. Optionally also read the tip and **flag divergence**; the pinned version is the verdict authority.
- Source files to read & parse fresh: `gcc/config/riscv/tt/rtl-rvtt-schedule.cc` (the NOP pass), `gcc/config/riscv/tt/rvtt.md` (the `xtt_delay` and `xtt_dynamic_bug` attributes), `gcc/config/riscv/tt/sfpu-ops-{wh,bh,qsr}.h` (per-arch op tables). HW latency columns: tt-isa-docs MCP `VectorUnit.md` (WH and BH separately).
- **Instruction existence/validity** is resolved by that arch's `tt_metal/tt-llk/tt_llk_<arch>/common/inc/ckernel_ops.h` (a `TT_OP_*`/`TTI_*` with an opcode encoding = valid), **never** by ISA-doc page coverage — so never flag an instruction "invalid/absent on BH" from a missing page. The header settles *existence only*; latency / sampled registers / errata still come from the ISA + `sfpi-gcc`. (This is the general "never infer a negative from a missing doc" rule — grounded per `race-audit-all`'s ladder + Ground-or-abstain, which this skill's MANDATORY banner already requires; not re-derived here.)
- **Fetch recipe — the source is a *submodule*, NOT in the local install** (`runtime/sfpi/` ships only the *compiled* toolchain; do not stop there and fall back to a compile experiment). Resolve it each run:
  1. Read the pin in `tt_metal/sfpi-version` (e.g. `sfpi_version='7.62.0'`, `sfpi_repo='https://github.com/tenstorrent/sfpi'`).
  2. `tenstorrent/sfpi` at that tag has a **`gcc` submodule** → `tenstorrent/sfpi-gcc`; get its pinned commit: `gh api repos/tenstorrent/sfpi/contents/gcc?ref=<tag> -q .sha` (e.g. sfpi `7.62.0` → sfpi-gcc `40d9f44`).
  3. Read the files at that commit without a full clone: `gh api repos/tenstorrent/sfpi-gcc/contents/gcc/config/riscv/tt/<file>?ref=<sha>` (or clone `tenstorrent/sfpi-gcc` and `git checkout <sha>`). The example commit here is a *dated illustration* — always re-resolve from the current pin.
- If the pinned `sfpi-gcc` source genuinely can't be fetched (no network / repo unreachable), **emit no latency verdict — abstain and mark coverage bounded**; do not fall back to a stale baked-in list or a compile experiment, and never overturn a prior source-grounded verdict.

## How the compiler handles it (the model to re-derive, not memorize)
The pass `rtl-rvtt-schedule.cc` conditionally inserts `sfpnop` after Tensix insns, keyed on each insn's `xtt_delay` (`none` / `static` / `dynamic`):
- **`static` delay** → fixed pipeline bubble; a NOP is always inserted (unless one already follows). **No arch exemption — padded on WH, BH, and QSR.**
- **`dynamic` delay** → true data-hazard; a NOP is inserted only if a following insn actually reads the result register within the window.
- **Blackhole / Quasar have hardware scoreboarding** (`schedule.cc`: `TARGET_XTT_TENSIX_BH/QSR` + `XTT_DYNAMIC_BUG_*` mask) that auto-stalls on dynamic data hazards, so the compiler **omits** the dynamic NOP there — **except** for instructions flagged `xtt_dynamic_bug` for that arch (scoreboard errata), where it still inserts the NOP. **Wormhole has no scoreboard** → the compiler always pads dynamic hazards. Re-derive the exact errata-flagged and static-delay instruction sets from `rvtt.md` each run.

## Provenance lens (decides the verdict — run this FIRST)
- **Code compiled through sfpi** (sfpi intrinsics → RTL → the scheduling pass: `vFloat`, `dst_reg[...]`, `sfpi::` ops) → the compiler schedules and pads it. **Trust it** (on both WH and BH, modulo the pinned version). Do not flag.
- **Hand-written `TTI_SFP*` / `TTI_*` / inline-asm / direct opcode pushes** → bypass the pass entirely → **manual NOPs required**. This is where hazards live.

## Method
1. **Establish the freshness contract** (resolve pinned sfpi; load the live rule files; re-derive the static-delay set, the per-arch dynamic-bug errata set, and the latency table). State what you derived and from which version.
2. **Enumerate hand-written instruction sequences** (the at-risk surface):
   ```bash
   cd tt_metal/tt-llk
   grep -rInE "\bTTI_SFP[A-Z0-9_]+|\bTTI_[A-Z0-9_]+\(|sfpnop|TTI_NOP" tt_llk_* --include=*.h | grep -v /tests/
   ```
   Classify each block by provenance (sfpi-generated vs raw). Skip sfpi-generated blocks.
3. **For each raw sequence**, walk producer→consumer in program order. For every result a later instruction reads, check the producer's latency (and `xtt_delay` class) against the spacing present: enough NOPs / independent instructions to cover the latency?
4. **Apply the arch matrix.** WH: every static + dynamic hazard needs manual spacing. BH/QSR: static delays + the `xtt_dynamic_bug` errata instructions still need manual NOPs; other dynamic hazards are HW-scoreboarded. Do NOT exempt Blackhole wholesale.
5. **Watch the explicit footguns** the compiler flags: variable-LReg read/write (`rvtt.md` notes "for the user to get this right"), `SFPLOADMACRO` ("Complex" latency), and `SFPCONFIG`. Also the non-SFPU result-latencies the compiler never touches (these are pure LLK-C++, manual on every arch): `MVMUL`/FPU → `SFPLOAD`-from-`Dst` settle, config-write (`SETC16`/`WRCFG`) → consumer settle.

## Verdict
- **sfpi-compiled, or raw sequence with sufficient spacing for the arch** → SAFE.
- **Raw sequence consuming a multi-cycle result without enough spacing, on an arch that needs it** (WH dynamic; any-arch static; BH/QSR errata insn) → LATENCY-HAZARD (real, silent corruption) — fix = insert the required `SFPNOP`/`TTI_NOP` (or reorder independent work in).
- **Arch-divergent** (e.g. needs a NOP on WH but HW-scoreboarded on BH) → report per-arch; never collapse to one verdict.
- **Can't resolve the pinned compiler** → UNCERTAIN + bounded-coverage note.

## Thoroughness (optional, full sweep)
Deterministic derivation of the latency/errata tables from the pinned `sfpi-gcc` (script the parse), then one agent per file to classify provenance and check spacing, then adversarially re-check each SAFE-by-scoreboard verdict against the freshly-derived errata set. Only run a Workflow if the user opts into multi-agent orchestration. Never exempt BH without confirming the consuming insn is not in the dynamic-bug set, and never trust a latency number not re-derived this run.

## Output
For each flagged site: `file:line`, the producer instruction + its latency/`xtt_delay` class, the dependent consumer, provenance (sfpi vs raw), spacing present vs required, per-arch verdict (SAFE / LATENCY-HAZARD / UNCERTAIN), and the one-line fix. Head the report with the compiler version consulted and the derived static-delay + per-arch errata sets (with a note that they were derived fresh). End with totals per verdict per arch.
