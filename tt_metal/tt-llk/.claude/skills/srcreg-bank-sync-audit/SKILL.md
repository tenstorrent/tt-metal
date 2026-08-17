---
name: srcreg-bank-sync-audit
description: Audit the shared backend DATA registers — SrcA/SrcB bank-valid (AllowedClient) + bank-flip handshake between unpacker and Matrix Unit, and the shared-once Dst/LReg overwrite hazards not already carried by the MATH_PACK semaphore or mutex::SFPU. Use after touching unpack→math dataflow, SETDVALID/CLEARDVALID, bank-flip bookkeeping, MOVD2A/MOVA2D/MOVB2D, or any cross-thread Dst/LReg access.
user_invocable: true
---

# /srcreg-bank-sync-audit — shared backend data-register hazards

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
The backend **data** memories are shared and have their own hardware flow control, distinct from config registers, Tensix semaphores, mailboxes, and CBs:
- **`SrcA` / `SrcB`** — each has **2 banks** carrying an `AllowedClient ∈ {Unpackers, MatrixUnit}`, plus four bank-pointer bits (`MatrixUnit::SrcABank/SrcBBank`, `Unpackers[0/1]::SrcBank`). The unpacker fills a bank and hands it to the FPU (set data-valid); the FPU consumes and hands it back. The **Wait Gate enforces this in hardware**: an FPU instruction stalls until the relevant bank's `AllowedClient == MatrixUnit`; `UNPACR` can start but stalls mid-execution until `AllowedClient` is appropriate. Software must keep the two sides' bank pointers in **lockstep** and place the valid/clear at the right point.
- **`Dst`** and **`LReg`** exist **once** (not per-thread). Threads can overwrite each other's data. The math↔pack `Dst` handoff rides the `MATH_PACK` semaphore (owned by `semaphore-handshake-audit`); cross-thread `LReg` is the (declared-but-unused) `mutex::SFPU` (also that audit). THIS audit owns the parts they don't: bank-flip / dvalid correctness, and any Dst/LReg sharing not mediated by those primitives.

A desync → the FPU reads a bank the unpacker is still filling, or a thread clobbers a live Dst/LReg → **silent data corruption** (rarely deadlock).

## Ground-truth (confirm via tt-isa-docs MCP)
`SrcASrcB.md` (bank model, `AllowedClient`, the four bank bits, BH implied-format-per-bank), `WaitGate.md` (the hardware-enforced `AllowedClient` stall + the `UNPACR` mid-execution wait), `Dst.md`, `LReg.md` (shared-once). Re-read per arch — WH and BH differ (e.g. BH per-bank implied format).

## What to check
1. **Bank-flip lockstep.** Over a complete tile/op, the unpacker's `SrcBank` increments and the FPU's `SrcABank`/`SrcBBank` increments must match 1:1. A conditional `UNPACR`, an op that flips one side but not the other, or a face/tile-count mismatch desyncs them → FPU reads the wrong (still-being-written) bank. Walk every branch.
2. **Valid/clear placement (dvalid handshake).** Data-valid handed to the MatrixUnit only after the unpack of that bank completes; handed back to the unpackers only after the FPU has consumed it. Flag a set-valid before the fill is complete, or a clear/reuse before the FPU drains.
3. **Single-thread ownership of the bank state.** The ISA requires "each relevant backend execution unit is only in use by one thread at a time." Two threads both driving the unpackers, or both issuing FPU ops, corrupt the shared bank-pointer bits. Flag any cross-thread contention on the unpacker or FPU bank state that isn't excluded by a handshake.
4. **Dst/LReg overwrite outside the known primitives.** A raw FPU/SFPU/pack access to `Dst`, or cross-thread `LReg`, that is NOT ordered by `MATH_PACK` / `mutex::SFPU` → flag and hand the semaphore half to `semaphore-handshake-audit`; this audit confirms the data-register access itself.

## Method
1. Enumerate the handshake primitives and bank bookkeeping:
   ```bash
   cd tt_metal/tt-llk
   grep -rInE "SETDVALID|CLEARDVALID|CLEARSRC|set_dvalid|clear_src|SrcA?Bank|unpack.*bank|MOV[AB]2D|MOVD2[AB]|TTI_UNPACR|get_valid" tt_llk_* --include=*.h | grep -v /tests/
   ```
2. Per unpack→math op, pair the unpacker's fill/flip with the FPU's consume/flip; trace the bank pointer on both sides across the tile loop. Confirm lockstep, valid/clear ordering, and single-thread ownership.
3. For Dst/LReg, identify the accessing threads and the mediating primitive (or its absence).

## Verdict
- **Bank pointers lockstep on every path, valid/clear correctly ordered, single owner per unit** → SAFE.
- **Bank-flip desync reachable** (counts diverge on a branch) → CORRUPTION (FPU reads unfilled/over-written bank).
- **dvalid set/cleared at the wrong point** → CORRUPTION or stall.
- **Cross-thread contention on bank state / unmediated Dst|LReg sharing** → RACE (hand the semaphore half to `semaphore-handshake-audit`).
- **Risk only on an experimental/unused path or value-invariant** → LATENT — say so.

## Architecture note
WH/BH share the bank model; BH adds per-bank implied data format (`ImpliedSrcAFmt/BFmt`) written by the unpacker — verify the implied-format and the data land in the same bank the FPU will read. **On BH a raw `SETDVALID` is ISA-unsupported** (it corrupts `ImpliedSrcBFmt` to an unpredictable value); the supported form is `UNPACR_NOP(...,SET_DVALID,...)`. Flag a raw `TTI_SETDVALID` on BH, and check whether `DISABLE_IMPLIED_SRCB_FMT_Base` is set when MOVB2D/MOVA2D consume that bank. Quasar's unpack→dest path has its own semaphores (`UNPACK_TO_DEST` / the QSR semaphore map) plus HW AutoTTSync — confirm the model before extending verdicts.

**Do NOT dismiss a Quasar-specific data lane by analogy to the WH/BH 2-bank SrcA/SrcB model.** Quasar adds a third unpacker / `SrcS` lane (`llk_srcs.h`, `UNPACKER2`): audit its dvalid lifecycle in full — both the **set** (producer, e.g. `UNPACR2`) **and** the **clear/consume** (consumer, e.g. `PACR1`) — and whether the lane's interlock fences (e.g. `*_SRCS_RDY` stall conditions) are actually *invoked*. A fence that is **defined but never used** is itself a finding (the lane is unprotected — safe only while it stays unwired/test-only), not grounds to call the lane SAFE. "It's a separate lane, so it doesn't participate in the SrcA/SrcB handshake" is a hypothesis to verify against the QSR ISA/Confluence and to trace in code — never a closure by analogy.

## Output
For each op/site: `file:line` of the unpacker fill/flip and the FPU consume/flip, bank-pointer lockstep result (per branch), dvalid set/clear placement, single-owner check, Dst/LReg mediation, arch, verdict (SAFE / CORRUPTION / RACE / LATENT) + one-line fix. End with totals per arch.
