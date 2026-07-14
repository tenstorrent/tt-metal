---
name: mmio-race-audit
description: Audit LLK code for races between a RISC MMIO write to a config/GPR register and a Tensix instruction/MOP/replay that consumes it. Use after touching any raw cfg[...]=/reg_write/cfg_rmw/regfile[]= write near an UNPACR/PACR/MOP/CFGSHIFTMASK, or when adding addressing/stride/format register writes.
user_invocable: true
---

# /mmio-race-audit — MMIO write vs Tensix instruction-stream race

> **Ground-truth precedence:** the live ISA doc (tt-isa-docs MCP, fetched each run) outranks every rule, table, and example baked into this skill — treat those as dated illustrations. If the live ISA doc **contradicts** a baked rule here, do NOT silently proceed: surface the conflict to the user and ask whether the baked rule should be overwritten, discarded, or kept. Default to the ISA doc.
>
> **MANDATORY — before any verdict, read the shared grounding policy.** The per-architecture **source ladder** (which docs to consult), the **ground-or-abstain** rule, and the **Source preflight** (list the sources you'll consult with their reachability + hierarchy, then PAUSE for the user) are defined once in `race-audit-all` → `.claude/skills/race-audit-all/SKILL.md`. **Your FIRST action is to `Read` that file and follow its "Ground-truth source ladder", "Ground-or-abstain", and "Source preflight" sections** — they are load-bearing: a verdict produced without them is ungrounded and MUST NOT be reported. If that file genuinely cannot be read, say so and **abstain** rather than proceed ungrounded. (If you were spawned by a `race-audit-all` sweep — your prompt already lists the confirmed sources — skip the Source preflight and do not pause; the orchestrator ran it once.)
>
> **Coverage — floor, not ceiling.** The grep patterns and site lists in this skill are a **seed, not an exhaustive enumeration**. After running them, widen the search with full reasoning. The techniques here are **illustrative examples, not the allowed set** — use any approach your reasoning suggests, including ones not listed: e.g. semantic search (by behavior/effect, not just token), resolving macros / wrappers / typedefs / indirection the literal pattern can't match, following the call graph to callers and callees, and diffing the WH/BH/QSR variants to catch a site present in one arch and missing in another. If you can find a hazard, primitive, or site the encoded patterns don't cover — by any means — pursue and report it; do **not** clamp a stronger analysis to this list or to these techniques. State any residual coverage gaps explicitly (no silent caps).
>
> **Execution — parallel by default.** When enumeration yields more than a few sites/files, **fan out concurrent `Agent` calls by default** (one per file/subsystem, a fresh context each), saturating the available concurrency (~10–16 at once); go inline only for a trivial set. The per-file fan-out described under *Thoroughness* is the **default**, not an exhaustive-only option. The cross-referencing/synthesis of results stays sequential (it must follow the per-unit findings). The heavyweight **Workflow** tool still requires explicit multi-agent opt-in — it is the opt-in exhaustive tier, not the default. Don't over-spawn a tiny diff.
>
> **Persisting results — single writer, incremental.** Agents only **return** their findings; they never write a shared file (no concurrent-write clobbering). If findings are persisted to a file, the orchestrator/caller is the **sole writer** and **appends each wave's returns as they arrive** — incremental, never only-at-the-end — so an interrupt preserves every completed wave's findings.

## The bug class (precise)
A RISC-V baby core writes a Tensix CONFIG or GPR register via **MMIO** — a direct memory-mapped store from the RISC core, NOT an instruction issued to the Tensix coprocessor. That store does **not** pass through the Tensix wait gate, so a `STALLWAIT` cannot order or hold it back. If a later **Tensix instruction / MOP run / replay execution** depends on that register, you can get:
- **write-too-late**: the consumer reads the stale value (MMIO not landed yet), or
- **write-too-early (cross-call)**: the MMIO write overwrites a register that a *prior, still-in-flight* consumer is reading.

Especially dangerous when a MOP/replay runs, an MMIO write changes a register it reads, then the MOP/replay runs again — or runs in a loop with a per-iteration MMIO change.

## UNSAFE primitives (RISC MMIO — hunt for these)
- `cfg[...] = ...` via `get_cfg_pointer()` / `get_cfg16_pointer()`
- `reg_write(addr, data)`
- `cfg_rmw(...)` / `cfg_rmw_gpr(...)`  (do `cfg_regs[addr] = ...` internally)
- `regfile[idx] = ...`  (raw GPR MMIO store — **easy to miss in greps**)
- `addr_mod_*::set(...)` / `rv_wrcfg` when implemented as a raw `volatile cfg[...] =` (true on **Quasar**; on WH/BH `addr_mod_t::set` uses the ordered `TTI_SETC16` and is SAFE)
- any raw `volatile T* p = reinterpret_cast<...>(BASE); p[i] = ...` to cfg/GPR/MOP-cfg/TDMA register space (`cfg_write()`, `xmov_*`, `mop_cfg[]=`)

## SAFE (ordered in the thread stream — not the bug)
- `cfg_reg_rmw_tensix<>` (emits `TT_RMWCIB`), `TTI_REG2FLOP`, `TTI_WRCFG`, `TTI_SETC16`, `TTI_RMWCIB*`, `TTI_SETADC*`, `TTI_SETDMAREG` — Tensix instructions, in-order through the config unit.
- `sync_regfile_write(idx)` after a `regfile[]` block — a read-back fence that retires the GPR store before dependent instructions issue. (Syncing the LAST index drains all prior regfile writes in the block.)
- A `TTI_STALLWAIT` whose **condition operand** includes `p_stall::TRISC_CFG` (ISA condition C13 = "this thread has a RISC cfg/GPR write emitted but not yet processed") placed BEFORE the consumer. NOTE: a *trailing* `TRISC_CFG` stall only orders the write before the NEXT run — it does **not** protect a *prior* in-flight consumer.
- The consuming unit is provably idle when the write lands (context-acquire / semaphore handshake), with no consumer before the function returns or a sync.
- **RISC-blocking drains** — `mop_sync()` (read-back of `pc_buf_base[2]`: stalls the RISC core until in-flight **MOPs** complete) and `tensix_sync()` (read-back of `pc_buf_base[1]`: stalls the RISC core until the **whole Tensix thread** is idle). Unlike a `STALLWAIT` (which stalls the Tensix stream at the Wait Gate), these stall the *RISC core* until the Tensix side catches up, so they DO order a subsequent RISC MMIO write after the drained work. **Canonical use:** `mop_sync()` before reprogramming `TENSIX_MOP_CFG_BASE` (`ckernel_template::program()` on WH/BH/QSR) — without it the RISC overwrites the MOP template while a prior MOP is still expanding. Count these as SAFE **only when the drain provably covers the consumer at the site** (right primitive, on every path, including the cross-call window); when unproven, keep the flag. See the perf caveat below — recognizing them as SAFE does **not** mean endorsing their use.

## ⚠️ Architecture difference — do not false-positive on Quasar
**Quasar enables HW AutoTTSync.** The code sets it up (`set_ttsync_enables<TRACK_ALL>` / `TRACK_GLOBAL_CFG`; `tt_llk_quasar/common/inc/ckernel.h` ~L270-310; `p_ttsync` in `ckernel_instr_params.h`), with an in-code comment that it is `"turned on by default by HW"` (so the explicit call is redundant, not that the mechanism is untrustworthy). **What that mechanism actually guarantees — i.e. whether it HW-orders a RISC `cfg`/`GPR` MMIO write against the Tensix instruction that consumes it, thereby replacing the manual `STALLWAIT(TRISC_CFG)`/`REG2FLOP`/`sync_regfile_write` discipline — is a HW-semantics question the code does not answer. Read Confluence `1340276980` at audit to confirm it before treating a raw `cfg[...]=` / `addr_mod_t::set` on Quasar as SAFE.** Treat WH/BH with the manual-ordering rules.
- **Residual Quasar corner:** `TRACK_ALL` enables `EN_SUBDIVIDED_CFG_FOR_UNPACR` (`ckernel.h:285`; in-code comment at `ckernel.h:275`) — with subdivided tracking, cross-sync between unpacker-0 and unpacker-1 cfg accesses is dropped, so a hazard exists only if a TRISC write to one unpacker's reg is consumed by the *other* unpacker's UNPACR. The mechanism is **publicly documented for Blackhole**: tt-isa-docs `BlackholeA0/.../AutoTTSync.md` lists Tensix Backend Configuration as tracked in separate per-unpacker sub-resources (unpacker-0 / unpacker-1 / remainder) — ground the behaviour there and cross-check to Quasar via the shared `set_ttsync_enables` / `EN_SUBDIVIDED` code; confirm any Quasar-specific delta against Confluence `1340276980` at audit. No live LLK site trips this today → flag, don't assert (don't record as "covered").

## Method
1. **Enumerate** every MMIO write (across all three arches), including the easy-to-miss classes:
   ```bash
   cd tt_metal/tt-llk
   grep -rInE "\bcfg[0-9_]*\[[^]]+\]\s*=[^=]|reg_write\(|\bcfg_rmw(_gpr)?\(|\bregfile\s*\[[^]]+\]\s*=[^=]|get_cfg_pointer|get_cfg16_pointer|rv_wrcfg|\.set\(ADDR_MOD|cfg_write\(|XMOV_(L1_BASE|CMD)\s*\[" \
     tt_llk_* --include=*.h | grep -v /tests/
   ```
   Exclude non-bug-class targets: `pc_buf_base[]=`, `mailbox_base[]=` (semaphore/mailbox, not a register a Tensix instruction reads). **But** the `pc_buf_base[1]`/`pc_buf_base[2]` *read-backs* inside `tensix_sync()`/`mop_sync()` ARE ordering primitives — enumerate them too, both as guards (SAFE list) and as perf targets (below): `grep -rInE "\b(tensix_sync|mop_sync)\s*\(" tt_llk_* --include=*.h | grep -v /tests/`.
2. **Per site**, read the enclosing function (and callers it triggers). Determine: register written; is there a dependent Tensix instr / MOP / replay AFTER it (incl. loops & double-runs); what sync sits between; is the consuming unit idle.
3. **Confirm the dependency** — which instruction reads the register. ISA names differ slightly from `cfg_defines.h`; map by similarity + logic, and use the tt-isa-docs MCP (WH/BH) to confirm a consumer (e.g. `THCON_SEC0_REG3_Base_address` ← UNPACR addressing; `ADDR_MOD_*` ← MOP/UNPACR/PACR/MOV; `SCRATCH_SEC0_val` ← CFGSHIFTMASK).
4. **Check BOTH directions and value-invariance**:
   - within-call: is the write ordered before its consumer (e.g. a `STALLWAIT(STALL_UNPACK, TRISC_CFG)` before the MOP, with the consumer after the MOP)?
   - cross-call: with double-buffered contexts (`wait_for_next_context(2)` / `switch_config_context`), can a *prior* call's consumer still be in flight when this call's MMIO write lands? Is the written register context-banked (e.g. `REG3_Base` vs `REG3_Base_cntx1`, SAFE) or **shared** (e.g. single `SCRATCH_SEC0_val`, risky)?
   - value: even if a cross-call window exists, is the written value **invariant** across the pipelined calls (e.g. a CB `fifo_page_size` stride)? If the racing write writes the same value the reader needs, the race is **benign**. Say so explicitly.

## Verdict
- **Quasar + relies on AutoTTSync** → SAFE **only after confirming the AutoTTSync ordering guarantee for this write against Confluence `1340276980` at audit** (see the architecture note); flag the `EN_SUBDIVIDED` cross-unpacker corner.
- **WH/BH ordered (Tensix-instr write / sync_regfile_write / leading TRISC_CFG stall / idle handshake)** → SAFE.
- **WH/BH unordered with a reachable consumer** → RACE (real). Fix = convert the MMIO write to a Tensix-instruction write (`REG2FLOP`/`WRCFG`/`SETC16`), or add `sync_regfile_write`, or a `STALLWAIT(..., TRISC_CFG)` before the consumer.
- **Window exists but value-invariant or unit-idle in all current callers** → UNCERTAIN/latent hardening — say it is not a live bug; let the maintainer decide.

## Sync-primitive cost — perf findings on `mop_sync`/`tensix_sync` (in addition to the race verdict)
**Both** `mop_sync()` and `tensix_sync()` are slow RISC-blocking drains — they stall the RISC core until the Tensix side catches up — and are a **last resort**, not a default guard. Two cost classes: **cheap** = in-stream Tensix sync (`STALLWAIT`/`SEMWAIT`, SrcA/SrcB dvalid, Tensix semaphores, `sync_regfile_write`, stalls only the Tensix stream/a unit); **slow** = the two RISC-blocking drains (`mop_sync` drains in-flight MOPs; `tensix_sync` drains the whole thread — heavier, but `mop_sync` is *not* "cheap"). Scrutinize **both**. Two perf finding directions, emitted *alongside* (never replacing) the correctness verdict:
- **OVER-SYNC** — a slow RISC-blocking drain used where an **in-stream Tensix sync** would suffice (applies to both `mop_sync` and `tensix_sync`). Primary fix = the in-stream primitive (e.g. `STALLWAIT(STALL_*, unit)`) when it closes the hazard. Only when a RISC-blocking drain is genuinely required (e.g. a RISC MMIO write to MOP-cfg consumed by the MOP expander) is `tensix_sync`→`mop_sync` a still-slow improvement. Name the recommended guard.
- **REDUNDANT** — a `mop_sync`/`tensix_sync` that is entirely unnecessary: the hazard is already covered by another sync, duplicate/back-to-back drains, or no reachable consumer depends on it. Fix = delete it.

Discipline: **prove sufficiency at the site before recommending** any downgrade/removal — a `STALLWAIT` drains only its own thread/unit and cannot replace a cross-thread or RISC-side drain. Behavior-preserving suggestion; don't claim a measured speedup unless it's a hot path. (`mop_sync` and `tensix_sync` are both slow and flagged symmetrically; `tensix_sync` is merely the heavier of the two.)

## Thoroughness (optional, for a full sweep)
Fan out one agent per file. Only if the user opts into multi-agent orchestration, run a Workflow: enumerate → per-file analyze → **adversarial verify every file with an in-thread consumer** (try to refute each SAFE verdict; default to UNCERTAIN when safety can't be proven) → completeness critic over the primitive grep (it WILL miss `regfile[` / `cfg_write(` / `xmov` unless added). Be calibrated: prove value-invariance / idle-handshake before calling a flagged site benign; surface anything unproven as UNCERTAIN with reasoning.

## Reference fixes (ground truth)
- **MMIO base-address write before the unpacker MOP** — a raw `cfg[THCON_SEC0_REG3_Base_address] = address` issued before the unpacker MOP that consumes it is a race. Correct form: an ordered Tensix-instruction write — `TT_SETDMAREG + TTI_REG2FLOP` for the address, then `STALLWAIT(STALL_UNPACK, THCON)` before the MOP.
- **`unpack_get_tile`** — a known site requiring strict ordering of its config writes ahead of the unpacker that consumes them (same MMIO-vs-UNPACR class).
- BH `_llk_unpack_fast_untilize_bfp_block_` `SCRATCH_SEC0_val` → ordered `SETDMAREG+STALLWAIT(STALL_CFG,THCON)+WRCFG` (latent cross-call hardening; value-invariant today).
- `ckernel_template::program()` / `ckernel_unpack_template::program()` (WH/BH/QSR) — `mop_sync()` before the raw `mop_cfg[i] = ...` MMIO writes to `TENSIX_MOP_CFG_BASE` is the canonical guard for the MMIO-vs-MOP reprogram race. (Ground truth for the SAFE recognition + the perf scrutiny.)

## Output
For each flagged site: `file:line`, function, register + MMIO primitive, consumer (instr/MOP/replay) + whether it loops/re-runs, sync between (and direction it covers), arch (and AutoTTSync applicability), correctness verdict (SAFE / RACE / UNCERTAIN-latent) with one-line reasoning, and the one-line fix. For every `mop_sync`/`tensix_sync` site, ALSO emit the perf verdict (OK / OVER-SYNC / REDUNDANT) with the named cheaper-or-no alternative. End with totals per verdict (correctness and perf separately).
