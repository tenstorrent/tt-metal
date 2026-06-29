---
name: race-audit-all
description: Run all nine LLK hazard audits (mmio-race, reconfig-stall, cfg-word-overlap, semaphore-handshake, mailbox-sync, dataflow-cb-sync, srcreg-bank-sync, noc-sync, instruction-latency) across four synchronization surfaces, and add a cross-class JOIN pass that catches emergent races no single audit can see — where one audit's verdict is "safe because <invariant owned by another audit>". Use for a full hazard sweep of an LLK change, or before merging anything touching config writes, reconfig/uninit, inter-thread/cross-core sync, the SrcA/SrcB-Dst data path, or hand-written instruction sequences.
user_invocable: true
---

# /race-audit-all — Orchestrated LLK race sweep + cross-class synthesis

## Purpose
The per-class audits each cover one hazard mechanism. Most real bugs are category-local and any one audit finds them. But some are **emergent**: each audit individually says SAFE because its safety *depends on an invariant that lives in another audit's domain*, and no single audit verifies the join. This skill runs all nine and adds a **JOIN pass** that discharges those cross-references — without ever losing what the individual audits found.

The nine sub-audits span **four synchronization surfaces** (not just cross-thread races — the suite also covers cross-core, RISC↔Tensix, and intra-thread micro-architectural hazards). Run each fully — see its own SKILL.md:
- **Cross-thread (shared backend state):**
  - `cfg-word-overlap-audit` — two threads write the same 32-bit CONFIG word.
  - `reconfig-stall-audit` — config rewrite without draining the consuming execution unit.
  - `semaphore-handshake-audit` — inter-thread semaphore/mutex protocol (incl. SEMINIT-vs-usage).
  - `srcreg-bank-sync-audit` — SrcA/SrcB `AllowedClient`+bank-flip handshake (unpacker↔Matrix Unit) and shared-once Dst/LReg overwrite.
- **RISC↔Tensix ordering:**
  - `mmio-race-audit` — RISC MMIO write vs Tensix-instruction/MOP/replay ordering; also `mop_sync`/`tensix_sync` drains (incl. OVER-SYNC/REDUNDANT perf findings).
  - `mailbox-sync-audit` — RISC↔RISC mailbox FIFO handshakes (push/pop balance, call-count symmetry, fence=nop ordering).
- **Cross-core (NoC):**
  - `dataflow-cb-sync-audit` — circular-buffer producer/consumer credits (reserve/push/wait/pop balance, data-before-credit ordering, capacity, remote CBs).
  - `noc-sync-audit` — raw `noc_semaphore_*` + barrier data-before-signal ordering and multicast fan-out (the non-CB half of dataflow).
- **Intra-thread (micro-architectural):**
  - `instruction-latency-audit` — pipeline result-latency / NOP padding on hand-written instruction sequences (compiler-grounded, arch-divergent).

## The monotonic contract (non-negotiable — this is what makes the sweep a true superset)
A naive "run them + concatenate" can catch *less* than the audits alone (summarization loss, dedup collapse, over-resolution). To prevent that, the JOIN is **additive-only**:
1. **Preserve every per-audit finding verbatim.** The output *includes* all nine raw reports (full finding lists, not summaries). Nothing is dropped, merged-away, or reworded.
2. **The JOIN may only ADD findings or ESCALATE severity.** It must never silently delete or downgrade a per-audit verdict.
3. **No silent downgrades.** If the JOIN verifies a cross-reference and believes a flagged item is actually safe, it attaches an *annotation* next to the original flag with the shown evidence — it does **not** replace the flag. Default to keeping the flag; never upgrade "probably safe" to SAFE without proof at that exact site.
4. **No summarization at the fan-out boundary.** Sub-audits return their full enumerations + verdicts; the JOIN reasons over those, not over compressed digests. If you fan out to agents, instruct each to return the complete finding list (and its candidate count) so the JOIN can re-judge dismissed sites if a cross-reference implicates them.
5. **No silent caps.** If coverage is bounded anywhere (top-N, sampling, agent budget), `log`/state it — a bounded sweep must not read as exhaustive.

## Method
1. **Run all nine audits** (faithful execution — invoke each skill or run its deterministic enumeration; don't approximate). Collect each one's complete finding list with per-finding `file:line`, verdict, and — critically — its **stated assumption** for every SAFE/LATENT verdict (the "safe because …" clause).
2. **Build the cross-reference worklist.** Extract every verdict whose safety is conditional on another class's invariant. The known seams (starting set — not exhaustive):

   | Audit says… | …safe because (other-class invariant) | JOIN must verify |
   |---|---|---|
   | `cfg-word-overlap`: shared SrcA/SrcB ALU-format word is LATENT | "pipeline semaphores keep the reconfig from overlapping the other thread's op" | the *specific* format RMW sits inside the region that `semaphore-handshake` proved ordered — same semaphore, write between wait and post/get, on every branch |
   | `mmio-race`: MMIO config write SAFE | "a semaphore / STALLWAIT(TRISC_CFG) orders it before the consumer" | `semaphore-handshake` shows that semaphore is balanced+init'd AND the MMIO store is sequenced relative to the post; or the stall's condition actually covers the consumer |
   | `reconfig-stall`: per-thread drain present (e.g. `STALLWAIT(STALL_CFG, PACK)`) | (drains *this* thread's unit only) | does another **thread** write the same word? → hand to `cfg-word-overlap`; a per-thread drain never excludes a cross-thread writer |
   | `semaphore-handshake`: semaphore protocol SAFE | (verifies counting, not payload) | which config words/dest/src rely on this semaphore for mutual exclusion? → confirm each such write is actually inside the ordered window |
   | `mailbox-sync`: mailbox handshake SAFE | "the memory the mailbox value refers to is ready, and all threads reach the broadcast equally" | the referenced memory (L1 tile, dest offset) is ordered-ready — `fence`=nop means the mailbox write does NOT imply a prior store landed, so cross with `mmio-race`/memory-ordering AND hand the "is the CB page ready?" half to `dataflow-cb-sync`; and the call-count symmetry holds on every branch (same control-flow that `semaphore-handshake` balance depends on) |
   | `dataflow-cb-sync`: CB credit SAFE | "the page write is ordered before the credit, and reserve/wait gates the access" | the data-before-credit barrier (NOC flush before `cb_push_back`) is present → cross with `mmio-race`/NOC ordering; the address `mailbox-sync` broadcasts derives from `fifo_rd_ptr` gated by this `cb_wait_front`; and `tile_regs_*` interleaving is `semaphore-handshake`'s `MATH_PACK` |
   | `mmio-race`: MMIO-vs-MOP write SAFE | "a `mop_sync()`/`tensix_sync()` drains it" | the drain provably covers the consumer at the site (right primitive, every path, cross-call window). ALSO: is the drain heavier than needed (OVER-SYNC) or unnecessary (REDUNDANT)? → perf finding, never suppresses the race verdict |
   | `srcreg-bank-sync`: SrcA/SrcB bank handoff SAFE | "the FPU op waits for `AllowedClient`, and Dst/LReg is ordered" | bank-flip is lockstep on both sides; the Dst/LReg half rides `MATH_PACK`/`mutex::SFPU` → hand that half to `semaphore-handshake`; single-thread ownership of the bank state holds |
   | `noc-sync`: cross-core credit SAFE | "the remote write is flushed before the credit, and the wait count matches the fan-out" | the `noc_async_write_barrier`/`writes_flushed` sits before the `noc_semaphore_inc`; cross with `dataflow-cb-sync` when the same buffer is also a CB page |
   | `instruction-latency`: sequence SAFE | "the compiler scheduled the NOPs" / "Blackhole HW scoreboards it" | the code is actually sfpi-compiled (provenance lens), not raw `TTI_*`; and for BH the consuming insn is NOT in the freshly-derived `xtt_dynamic_bug` errata set — re-derive from the pinned `sfpi-gcc`, never a baked list |
   | any: "value-invariant / unit-idle / single-thread" | (assumption about another class's state) | re-confirm the assumption at the site with the other audit's lens |

3. **Discharge each cross-reference at its actual site** (read the code; don't trust the summary). Trace the one physical resource (a CFG word, dest/src bank, a semaphore) across *all* mechanisms that are supposed to guard it. A race exists when the guarantees, composed, leave a gap — even if each guarantee is individually valid.
4. **Emit** per the monotonic contract: nine raw reports untouched, plus a new **EMERGENT** findings section, plus any escalations (annotated).

## Verdict (combined)
- **EMERGENT-RACE** — individually all-SAFE, but the join condition is unmet at a reachable site (e.g. format RMW moved outside the semaphore window; MMIO write not actually sequenced to its gating post). Real; report with the full cross-class chain.
- **Per-class verdicts** — passed through unchanged from each sub-audit (SAFE / RACE / LATENT / HARDENING-GAP / INIT-BUG …).
- **ANNOTATED-SAFE** — a per-class flag the JOIN cross-checked and believes is discharged; keep the original flag, attach evidence, mark for maintainer confirmation. Never silently resolve.

## Ground-truth freshness contract (non-negotiable)
Two sub-audits depend on sources that are **dynamic** and must be consulted live each run, never from data baked into a SKILL.md:
- `instruction-latency-audit` grounds in the **`sfpi-gcc` version this build is pinned to** (latency tables, `xtt_delay`, `xtt_dynamic_bug` errata) — re-derive every run; optionally diff the tip and flag divergence; mark coverage bounded if the pinned compiler can't be resolved.
- All HW-semantics verdicts ground in the **tt-isa-docs MCP** (fetched live), not cached prose.
Any instruction list / latency number / errata set appearing in a sub-skill is a *dated illustration*, subordinate to fresh derivation. The JOIN must not treat a stale baked list as authority.

**ISA-precedence policy (applies to every sub-audit):** the live ISA doc (and, for `instruction-latency`, the pinned `sfpi-gcc`) **outranks** any rule, table, or example baked into a skill. When a live source **contradicts** a baked rule, do NOT silently resolve it — surface the conflict to the user and ask whether the baked rule should be overwritten, discarded, or kept; default to the live source. This holds whether a sub-audit is run standalone or inside this sweep.

**Coverage — floor, not ceiling (applies to every sub-audit and the JOIN):** the grep patterns, site lists, and seam tables in this suite are a **seed, not an exhaustive enumeration**. Treat them as a minimum: after running them, widen with full reasoning. The techniques named here are **illustrative, not the allowed set** — use any approach your reasoning suggests, including ones not listed: e.g. semantic search (by behavior/effect, not token), resolving macros/wrappers/typedefs/indirection the literal patterns miss, following the call graph across files/layers, and diffing WH/BH/QSR variants. Pursue and report any hazard, primitive, seam, or site the encoded method doesn't cover — by any means; a more capable analysis must **not** be clamped to what is written here or to these techniques. The encoded patterns lower-bound coverage and reduce variance — they do not cap it. State residual coverage gaps explicitly (no silent caps).

**Execution — parallel by default.** Default to **concurrent `Agent` fan-out** (the nine audits concurrently, and per-file/subsystem within each), scaled to the candidate set and saturating the ~10–16 concurrency cap; inline only for a trivial diff. Concurrent `Agent` fan-out needs no opt-in; the heavyweight **Workflow** tool remains the explicit-opt-in exhaustive tier. The JOIN/synthesis stays sequential. Don't over-spawn small work.

**Persisting results — single writer, incremental.** Agents only **return** their findings; they never write a shared file (no concurrent-write clobbering). If findings are persisted to a file, the orchestrator/caller is the **sole writer** and **appends each wave's returns as they arrive** — incremental, never only-at-the-end — so an interrupt preserves every completed wave's findings (you lose at most the in-flight wave).

## Architecture note
WH/BH: all nine classes apply; cross-references as above. **Quasar**: HW Auto-TTSync (auto-orders RISC↔Tensix MMIO cfg/GPR writes against their consumers; WH/BH need manual ordering) changes the MMIO-ordering class, so seams touching `mmio-race` resolve differently; `instruction-latency` is also arch-divergent (BH/QSR scoreboarding vs WH always-pad). The cfg-word / semaphore / reconfig / mailbox / dataflow-cb / srcreg-bank / noc seams still apply (verify Quasar mailbox + NoC + unpack-to-dest HW semantics before extending verdicts there; the CB API is arch-agnostic but its NOC ordering primitives are arch-specific). Each sub-audit carries its own Quasar caveat — honor them in the join.

## Thoroughness (optional, full sweep)
**Default = parallel.** Fan out the nine audits as **concurrent `Agent` calls** (each a fresh context), and within each audit fan out per-file/subsystem concurrently, saturating the ~10–16 concurrency cap; then do the JOIN inline (it must follow the per-audit results). Run fully inline only for a trivial diff. Concurrent `Agent` fan-out does **not** require multi-agent opt-in. For an exhaustive adversarial pass at scale, **only if the user opts into multi-agent orchestration**, run a Workflow:
`phase 1` nine audits in parallel (each returns its FULL finding list + assumptions, schema-structured) → `phase 2` one JOIN agent per cross-reference that **adversarially tries to prove the join condition is violated** (default to EMERGENT-RACE / keep-flag when it can't prove safety) → `phase 3` a completeness critic over the cross-reference worklist ("which 'safe because' clauses were never discharged? which resource wasn't traced across all mechanisms?"). Keep it monotonic: the Workflow's synthesis stage unions the raw findings and only adds/escalates.

## Output
1. **Nine raw reports**, verbatim, one section each.
2. **Cross-reference worklist** — every "safe because <other class>" clause and whether it was discharged.
3. **EMERGENT-RACE findings** — cross-class chain (`file:line` → resource → the composed guarantees → the gap) + fix.
4. **Escalations / ANNOTATED-SAFE** — additive only.
5. **Totals** per verdict per class + emergent count, and an explicit note of any coverage bound. State plainly that no per-class finding was dropped or downgraded.
