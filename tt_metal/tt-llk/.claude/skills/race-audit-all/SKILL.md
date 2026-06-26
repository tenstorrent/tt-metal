---
name: race-audit-all
description: Run all six LLK race audits (mmio-race-audit, reconfig-stall-audit, cfg-word-overlap-audit, semaphore-handshake-audit, mailbox-sync-audit, dataflow-cb-sync-audit) and add a cross-class JOIN pass that catches emergent races no single audit can see — where one audit's verdict is "safe because <invariant owned by another audit>". Use for a full race sweep of an LLK change, or before merging anything touching config writes, reconfig/uninit, or inter-thread sync.
user_invocable: true
---

# /race-audit-all — Orchestrated LLK race sweep + cross-class synthesis

## Purpose
The per-class audits each cover one race mechanism. Most real bugs are category-local and any one audit finds them. But some races are **emergent**: each audit individually says SAFE because its safety *depends on an invariant that lives in another audit's domain*, and no single audit verifies the join. This skill runs all six and adds a **JOIN pass** that discharges those cross-references — without ever losing what the individual audits found.

The six sub-audits (run each fully — see its own SKILL.md):
- `mmio-race-audit` — RISC MMIO write vs Tensix-instruction/MOP/replay ordering.
- `reconfig-stall-audit` — config rewrite without draining the consuming execution unit.
- `cfg-word-overlap-audit` — two threads write the same 32-bit CONFIG word.
- `semaphore-handshake-audit` — inter-thread semaphore/mutex protocol (incl. SEMINIT-vs-usage).
- `mailbox-sync-audit` — RISC↔RISC mailbox FIFO handshakes (push/pop balance, call-count symmetry, fence=nop ordering).
- `dataflow-cb-sync-audit` — circular-buffer producer/consumer credits (reserve/push/wait/pop balance, data-before-credit ordering, capacity, remote CBs).

## The monotonic contract (non-negotiable — this is what makes the sweep a true superset)
A naive "run them + concatenate" can catch *less* than the audits alone (summarization loss, dedup collapse, over-resolution). To prevent that, the JOIN is **additive-only**:
1. **Preserve every per-audit finding verbatim.** The output *includes* all six raw reports (full finding lists, not summaries). Nothing is dropped, merged-away, or reworded.
2. **The JOIN may only ADD findings or ESCALATE severity.** It must never silently delete or downgrade a per-audit verdict.
3. **No silent downgrades.** If the JOIN verifies a cross-reference and believes a flagged item is actually safe, it attaches an *annotation* next to the original flag with the shown evidence — it does **not** replace the flag. Default to keeping the flag; never upgrade "probably safe" to SAFE without proof at that exact site.
4. **No summarization at the fan-out boundary.** Sub-audits return their full enumerations + verdicts; the JOIN reasons over those, not over compressed digests. If you fan out to agents, instruct each to return the complete finding list (and its candidate count) so the JOIN can re-judge dismissed sites if a cross-reference implicates them.
5. **No silent caps.** If coverage is bounded anywhere (top-N, sampling, agent budget), `log`/state it — a bounded sweep must not read as exhaustive.

## Method
1. **Run all six audits** (faithful execution — invoke each skill or run its deterministic enumeration; don't approximate). Collect each one's complete finding list with per-finding `file:line`, verdict, and — critically — its **stated assumption** for every SAFE/LATENT verdict (the "safe because …" clause).
2. **Build the cross-reference worklist.** Extract every verdict whose safety is conditional on another class's invariant. The known seams (starting set — not exhaustive):

   | Audit says… | …safe because (other-class invariant) | JOIN must verify |
   |---|---|---|
   | `cfg-word-overlap`: shared SrcA/SrcB ALU-format word is LATENT | "pipeline semaphores keep the reconfig from overlapping the other thread's op" | the *specific* format RMW sits inside the region that `semaphore-handshake` proved ordered — same semaphore, write between wait and post/get, on every branch |
   | `mmio-race`: MMIO config write SAFE | "a semaphore / STALLWAIT(TRISC_CFG) orders it before the consumer" | `semaphore-handshake` shows that semaphore is balanced+init'd AND the MMIO store is sequenced relative to the post; or the stall's condition actually covers the consumer |
   | `reconfig-stall`: per-thread drain present (e.g. `STALLWAIT(STALL_CFG, PACK)`) | (drains *this* thread's unit only) | does another **thread** write the same word? → hand to `cfg-word-overlap`; a per-thread drain never excludes a cross-thread writer |
   | `semaphore-handshake`: semaphore protocol SAFE | (verifies counting, not payload) | which config words/dest/src rely on this semaphore for mutual exclusion? → confirm each such write is actually inside the ordered window |
   | `mailbox-sync`: mailbox handshake SAFE | "the memory the mailbox value refers to is ready, and all threads reach the broadcast equally" | the referenced memory (L1 tile, dest offset) is ordered-ready — `fence`=nop means the mailbox write does NOT imply a prior store landed, so cross with `mmio-race`/memory-ordering AND hand the "is the CB page ready?" half to `dataflow-cb-sync`; and the call-count symmetry holds on every branch (same control-flow that `semaphore-handshake` balance depends on) |
   | `dataflow-cb-sync`: CB credit SAFE | "the page write is ordered before the credit, and reserve/wait gates the access" | the data-before-credit barrier (NOC flush before `cb_push_back`) is present → cross with `mmio-race`/NOC ordering; the address `mailbox-sync` broadcasts derives from `fifo_rd_ptr` gated by this `cb_wait_front`; and `tile_regs_*` interleaving is `semaphore-handshake`'s `MATH_PACK` |
   | any: "value-invariant / unit-idle / single-thread" | (assumption about another class's state) | re-confirm the assumption at the site with the other audit's lens |

3. **Discharge each cross-reference at its actual site** (read the code; don't trust the summary). Trace the one physical resource (a CFG word, dest/src bank, a semaphore) across *all* mechanisms that are supposed to guard it. A race exists when the guarantees, composed, leave a gap — even if each guarantee is individually valid.
4. **Emit** per the monotonic contract: four raw reports untouched, plus a new **EMERGENT** findings section, plus any escalations (annotated).

## Verdict (combined)
- **EMERGENT-RACE** — individually all-SAFE, but the join condition is unmet at a reachable site (e.g. format RMW moved outside the semaphore window; MMIO write not actually sequenced to its gating post). Real; report with the full cross-class chain.
- **Per-class verdicts** — passed through unchanged from each sub-audit (SAFE / RACE / LATENT / HARDENING-GAP / INIT-BUG …).
- **ANNOTATED-SAFE** — a per-class flag the JOIN cross-checked and believes is discharged; keep the original flag, attach evidence, mark for maintainer confirmation. Never silently resolve.

## Architecture note
WH/BH: all six classes apply; cross-references as above. **Quasar**: HW Auto-TTSync changes the MMIO-ordering class (memory `quasar-auto-ttsync`), so seams touching `mmio-race` resolve differently; the cfg-word / semaphore / reconfig / mailbox / dataflow-cb seams still apply (verify Quasar mailbox HW semantics before extending mailbox verdicts there; the CB API is arch-agnostic but its NOC ordering primitives are arch-specific). Each sub-audit already carries its own Quasar caveat — honor them in the join.

## Thoroughness (optional, full sweep)
Default: run the four inline (or one agent each), then do the JOIN inline. For an exhaustive pass, **only if the user opts into multi-agent orchestration**, run a Workflow:
`phase 1` six audits in parallel (each returns its FULL finding list + assumptions, schema-structured) → `phase 2` one JOIN agent per cross-reference that **adversarially tries to prove the join condition is violated** (default to EMERGENT-RACE / keep-flag when it can't prove safety) → `phase 3` a completeness critic over the cross-reference worklist ("which 'safe because' clauses were never discharged? which resource wasn't traced across all mechanisms?"). Keep it monotonic: the Workflow's synthesis stage unions the raw findings and only adds/escalates.

## Output
1. **Six raw reports**, verbatim, one section each.
2. **Cross-reference worklist** — every "safe because <other class>" clause and whether it was discharged.
3. **EMERGENT-RACE findings** — cross-class chain (`file:line` → resource → the composed guarantees → the gap) + fix.
4. **Escalations / ANNOTATED-SAFE** — additive only.
5. **Totals** per verdict per class + emergent count, and an explicit note of any coverage bound. State plainly that no per-class finding was dropped or downgraded.
