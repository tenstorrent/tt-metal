---
name: reconfig-stall-audit
description: Audit LLK reconfig/uninit/config-write functions for a MISSING stall that drains the execution unit before its config registers are rewritten (packer→PACK, unpacker→UNPACK, math→MATH|WAIT_SFPU). Use after touching cpack/cunpack/cmath, *_reconfig_*, *_uninit_, set_packer_strides, or any function that writes ALU/THCON/ADDR_MOD/stride config.
user_invocable: true
---

# /reconfig-stall-audit — Config-register reconfig without execution-unit drain

> **Ground-truth precedence:** the live ISA doc (tt-isa-docs MCP, fetched each run) outranks every rule, table, and example baked into this skill — treat those as dated illustrations. If the live ISA doc **contradicts** a baked rule here, do NOT silently proceed: surface the conflict to the user and ask whether the baked rule should be overwritten, discarded, or kept. Default to the ISA doc.
>
> **MANDATORY — before any verdict, read the shared grounding policy.** The per-architecture **source ladder** (which docs to consult), the **ground-or-abstain** rule, and the **Source preflight** (list the sources you'll consult with their reachability + hierarchy, then PAUSE for the user) are defined once in `race-audit-all` → `.claude/skills/race-audit-all/SKILL.md`. **Your FIRST action is to `Read` that file and follow its "Ground-truth source ladder", "Ground-or-abstain", and "Source preflight" sections** — they are load-bearing: a verdict produced without them is ungrounded and MUST NOT be reported. If that file genuinely cannot be read, say so and **abstain** rather than proceed ungrounded. (If you were spawned by a `race-audit-all` sweep — your prompt already lists the confirmed sources — skip the Source preflight and do not pause; the orchestrator ran it once.)
>
> **Coverage — floor, not ceiling.** The grep patterns and site lists in this skill are a **seed, not an exhaustive enumeration**. After running them, widen the search with full reasoning. The techniques here are **illustrative examples, not the allowed set** — use any approach your reasoning suggests, including ones not listed: e.g. semantic search (by behavior/effect, not just token), resolving macros / wrappers / typedefs / indirection the literal pattern can't match, following the call graph to callers and callees, and diffing the WH/BH/QSR variants to catch a site present in one arch and missing in another. If you can find a hazard, primitive, or site the encoded patterns don't cover — by any means — pursue and report it; do **not** clamp a stronger analysis to this list or to these techniques. State any residual coverage gaps explicitly (no silent caps).
>
> **Execution — parallel by default.** When enumeration yields more than a few sites/files, **fan out concurrent `Agent` calls by default** (one per file/subsystem, a fresh context each), saturating the available concurrency (~10–16 at once); go inline only for a trivial set. The per-file fan-out described under *Thoroughness* is the **default**, not an exhaustive-only option. The cross-referencing/synthesis of results stays sequential (it must follow the per-unit findings). The heavyweight **Workflow** tool still requires explicit multi-agent opt-in — it is the opt-in exhaustive tier, not the default. Don't over-spawn a tiny diff.
>
> **Persisting results — single writer, incremental.** Agents only **return** their findings; they never write a shared file (no concurrent-write clobbering). If findings are persisted to a file, the orchestrator/caller is the **sole writer** and **appends each wave's returns as they arrive** — incremental, never only-at-the-end — so an interrupt preserves every completed wave's findings.

## The rule (what a correct function does)
When a flattened LLK function REWRITES config registers that a hardware execution unit reads *while running*, that unit must be **idle first** — otherwise you reprogram state out from under an in-flight op (a "reconfig escape"). The guard is a `TTI_STALLWAIT` (usually at the top of the function) whose **condition (2nd) operand** drains the matching unit:

| Function reconfigures… | condition (`wait_res`, 2nd arg) must include | because |
|---|---|---|
| **Packer** config (out/in fmt, strides, l1 offset, exp threshold, l1_acc, dest-rd-ctrl) | `p_stall::PACK` | packer reads these during PACR |
| **Unpacker** config (tile descriptor, out fmt, strides, base addr) | `p_stall::UNPACK` (or `UNPACK0` for SrcA, `UNPACK1` for SrcB) | unpacker reads these during UNPACR |
| **Math** config (ALU SrcA/SrcB fmt, INT8 enable, dest acc) | `p_stall::MATH \| p_stall::WAIT_SFPU` | the FPU **and** the SFPU share the math path — BOTH must drain |

`TTI_STALLWAIT(stall_res, wait_res)`: `stall_res` = **block mask** (which instruction classes can't issue: `STALL_CFG`=B7 blocks WRCFG/RMWCIB, `STALL_PACK`/`STALL_UNPACK`/`STALL_MATH`), `wait_res` = **condition mask** (what to wait on). The block mask just needs to block the instruction that does the config write; the **condition mask is what proves the unit is drained** — that's the bit to check.

## What to flag
A reconfig/uninit/config-writer that writes config registers with **no preceding STALLWAIT whose condition drains the matching unit**. Sub-cases:
1. **No stall at all** before the config write.
2. **Wrong condition** — e.g. `STALLWAIT(STALL_CFG, THCON)` only orders the GPR→cfg write (THCON = scalar-unit memory requests); it does **NOT** drain PACK/UNPACK/MATH. A `THCON`-only guard on packer-stride writes is the classic miss.
3. **Math reconfig missing SFPU** — has `MATH` but not `WAIT_SFPU` (or vice-versa).
4. **Self-containment gap / arch asymmetry** — the function relies on the *caller* having drained the unit, while its sibling on another arch self-guards. Wormhole functions tend to self-guard; Blackhole siblings have been caught relying on caller drains. Flag the divergence even if current callers happen to drain.

## Method
1. **Enumerate** candidates across `tt_llk_wormhole_b0`, `tt_llk_blackhole`, `tt_llk_quasar`:
   ```bash
   cd tt_metal/tt-llk
   grep -rInE "reconfig|reconfigure|_uninit_|set_packer_strides|set_packer_l1_offset|configure_(pack|unpack)|reconfigure_exp_threshold|reconfigure_packer_l1_acc" \
     tt_llk_* --include=*.h | grep -v /tests/
   ```
2. **For each**, read the function body. Identify config-register writes: `cfg_reg_rmw_tensix<>`, `TTI_WRCFG`/`TT_WRCFG`, `TTI_REG2FLOP`, `TTI_SETC16`, `TTI_RMWCIB*`, `TTI_SETADC*` to packer/unpacker/ADDR_MOD regs, `regfile[]=`+`REG2FLOP`, and helper calls (`set_packer_strides`, `set_packer_l1_offset`, `addr_mod_*::set`).
3. **Determine which unit** reads the written register (packer / unpacker / math). Use register-name semantics; confirm against the ISA docs when non-obvious (see `arch-lookup` skill / tt-isa-docs MCP for WH/BH; register names map to `tt_metal/hw/inc/internal/.../cfg_defines.h` by similarity).
3b. **Determine latched-at-issue vs sampled-during-execution** for that register — this is the deciding factor for whether a unit drain is needed: **sampled during execution → needs the unit drain** (`THCON | PACK` for packer, etc.); **latched at instruction-issue → no unit drain, `THCON`-only ordering suffices**.
   - **Default = sampled.** Treat every register as sampled (assume it needs the drain) **unless** the ISA doc explicitly classifies it as latched. This default is deliberately conservative because the failure modes are asymmetric: a wrong "sampled" is at worst OVER-SYNC (perf), but a wrong "latched" misses a real reconfig escape (correctness). **As of now the only register known to be latched is the packer L1 destination address (`THCON_SEC0_REG1_L1_Dest_addr`); treat all others as sampled** until the live doc says otherwise.
   - The **authority for this classification is the ISA doc — consult the tt-isa-docs MCP live each run**; when it publishes a per-register latched/sampled table, classify each register straight from that table. The latched/sampled register names in this skill (the latched `L1_Dest_addr`; sampled strides / `PCK0_ADDR_BASE_REG_0_Base`) are **dated illustrations, not authority** — re-derive from the live doc and prefer it on any mismatch.
4. **Check the guard**: is there a `TTI_STALLWAIT(..., <condition incl. that unit>)` before the first such write — *inside this function* (preferred) or unavoidably in the call tree? Trace helpers (`set_packer_strides` etc.) to see whether THEY stall on the unit or only on `THCON`.
5. **Compare arches**: diff the WH vs BH vs Quasar version of the same function. A unit-drain present in one and absent in another is a strong signal.

## Verdict & scope
- **Self-contained + correct condition** → SAFE.
- **Relies on caller drain (no self-guard), sibling arch self-guards** → report as a *self-containment gap* (hardening), and trace callers to say whether it's a *live* escape or only latent. To convert latent→safe, add the WH-style leading `TTI_STALLWAIT(STALL_CFG, <unit>)` or change a `THCON`-only helper stall to `<unit> | THCON`.
- **No drain and a caller can reach it with the unit busy** → live reconfig escape (real bug).
- **Known-latched register, `THCON`-only stall** → SAFE (do not false-positive). This applies **only** to registers the ISA doc lists as latched (default everything else to sampled — see 3b). Currently the sole known-latched register is the packer L1 destination address `THCON_SEC0_REG1_L1_Dest_addr`: the packer latches it at issue, so only the GPR→cfg write needs ordering (`STALLWAIT(STALL_CFG, THCON)`) and a unit drain is unnecessary. Flagging it as "missing PACK drain" is wrong, and *adding* `| PACK` is **OVER-SYNC** (this exact change was made once and reverted). Do NOT extend this to other packer registers without an explicit latched classification.
- **Drained by a RISC-blocking drain** — `tensix_sync()` (whole-thread) or `mop_sync()` (MOP-only) → correctness-SAFE, but **both are slow** (they stall the RISC core; `tensix_sync` is the heavier of the two, but `mop_sync` is not "cheap"). Emit a perf finding alongside the SAFE verdict: **OVER-SYNC** if a targeted **in-stream** `TTI_STALLWAIT(STALL_CFG, <unit>)` provably covers the same reconfig → recommend that (the cheap, in-stream guard), not another RISC-blocking drain; **REDUNDANT** if the unit is already drained by a preceding sync / no reachable op reads the rewritten config → recommend removal. Prove sufficiency at the site first (a per-unit `STALLWAIT` can't replace a drain that's there for cross-thread/RISC-side ordering). Prefer an in-stream Tensix sync; both RISC-blocking drains are a last resort.

## Quasar — a "shadow register" comment is NOT a drain
Quasar reconfig helpers (e.g. `_llk_pack_reconfig_data_format_`, `_llk_unpack_reconfig_data_format_src_`) sometimes omit the drain their **WH/BH siblings have**, justified by an in-code comment that the target is a "shadow register on Quasar" (TTSync register shadowing). **Do not rate that SAFE on the comment alone.** Establish two things at audit time:
1. **What primitive the write actually is** (from the code): does `cfg_rmw` (`ckernel.h`) still resolve to a Tensix `TT_RMWCIB*` instruction, or to a genuine RISC MMIO `cfg[…]=` store? The two are governed differently, so check it at the site.
2. **Whether the Quasar HW ordering mechanism (register shadowing / TTSync) actually orders THIS write against the consuming `PACR`/`UNPACR` — including an *already-in-flight* one.** This is a HW-semantics question the code does **not** answer; **read Confluence `1340276980` at audit** and determine from it: (a) whether shadowing/TTSync covers this primitive at all, and (b) whether it orders only *issue* or also guarantees the write has landed / drains an in-flight consumer.

Since the WH/BH siblings drain and the in-code comment alone does not establish (2), default to **CONFIRMED-OPEN (resolve against Confluence `1340276980` or Quasar RTL/HW)**, not SAFE.

## Thoroughness (optional, for a full sweep)
For an exhaustive pass, fan out one agent per file (Explore/general-purpose), or — only if the user opts into multi-agent orchestration — a Workflow that analyzes each candidate then adversarially re-checks each SAFE verdict (try to find the un-drained path). Always report whether each finding is a *live bug* vs *self-containment hardening*, and never silently upgrade "probably safe" to "safe".

## Reference fixes (ground truth)
- **Packer config the packer samples during PACR** must be preceded by a packer drain: reprogramming such config while the packer is still running is a reconfig escape; the fix is a leading `STALLWAIT(..., PACK)` so the packer finishes first. (Default-assume sampled; the one known latched exception — the L1 destination address — needs only `THCON` ordering, see step 3b.)
- **Packer cfg drain depends on whether the register is latched — not all packer writes need a PACK drain.** A register the packer **reads during PACR** (strides via `set_packer_strides`; the dest tile-offset base `PCK0_ADDR_BASE_REG_0_Base` in `program_packer_dest_offset_registers`) needs `STALLWAIT(STALL_CFG, THCON | PACK)`: a `THCON`-only stall orders the GPR→cfg write but does **not** drain the packer, so the WRCFG could land mid-PACR. But a register the packer **latches at PACR-issue** does NOT need a PACK drain — `program_packer_destination` writes the L1 destination address (`THCON_SEC0_REG1_L1_Dest_addr`), which is latched, so `THCON`-only is correct; a `| PACK` was added there once and **reverted** as unnecessary over-sync. Plus `_llk_pack_fast_tilize_uninit_` leading `STALLWAIT(STALL_CFG, PACK)` on WH vs BH.
- `_llk_math_reconfig_data_format_*` (`llk_math_common.h`) — canonical `STALLWAIT(STALL_CFG, MATH | WAIT_SFPU)`.

## Output
For each flagged site: `file:line`, function, register(s) written, the unit that consumes them, the guard present (or none) and its condition, correctness verdict (SAFE / HARDENING-GAP / LIVE-BUG), and whether a sibling arch self-guards. Where the guard is a `tensix_sync()` (or other RISC-blocking drain), ALSO emit a perf verdict (OK / OVER-SYNC / REDUNDANT) + the named cheaper-or-no alternative. End with a count and the one-line fix per flag (correctness and perf separately).
