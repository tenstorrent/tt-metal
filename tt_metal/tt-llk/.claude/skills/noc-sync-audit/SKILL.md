---
name: noc-sync-audit
description: Audit cross-core NoC synchronization in dataflow kernels — noc_semaphore_wait/set/inc balance and direction, multicast fan-out counts, and data-before-signal NoC ordering (noc_async_write_barrier / noc_async_writes_flushed before a remote credit). The half of dataflow that dataflow-cb-sync-audit (CB credits) does not reach. Use after touching reader/writer kernels, noc_semaphore_*, noc_async_*_barrier, or any cross-core handshake not expressed as a CB.
user_invocable: true
---

# /noc-sync-audit — cross-core NoC handshake correctness

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
Cores coordinate across the NoC with **NoC semaphores** (L1 counters bumped by remote atomic increments / `noc_semaphore_set`) and **async NoC transactions** whose completion is not implied by the issuing instruction returning. Misuse → **data corruption** (a consumer reads a remote write that hasn't landed, because the credit raced ahead of the data) or **deadlock** (a `noc_semaphore_wait` whose increment never arrives, or a multicast whose fan-out count is wrong). `dataflow-cb-sync-audit` covers the CB-credit abstraction (`cb_*`, RemoteCB); THIS audit covers the raw `noc_semaphore_*` + barrier handshakes those same kernels use directly, which the CB audit's grep does not reach.

## Ground-truth mechanism (`tt_metal/hw/inc/api/dataflow/dataflow_api.h`, NoC docs)
- `noc_async_read` / `noc_async_write` return before the transfer completes. Completion is established only by `noc_async_read_barrier` / `noc_async_write_barrier` / `noc_async_writes_flushed`.
- `noc_semaphore_wait(ptr, val)` spins until the L1 semaphore reaches `val`; `noc_semaphore_inc` / `noc_semaphore_set` bump it (often via remote NoC atomic). Multicast writes/incs fan out to N receivers.
- `fence` / memory-ordering caveats are NoC- and arch-specific; the barrier primitive must match the transfer type and NoC in use.

## What to check
1. **Data-before-signal ordering (the main race).** When a producer writes a remote core's L1 and then signals readiness (a `noc_semaphore_inc`/`set` to that core), the **data write must be flushed before the credit write** — a `noc_async_write_barrier` / `noc_async_writes_flushed` between them. Otherwise the receiver's `noc_semaphore_wait` releases and it reads stale/partial L1. (CB analog of "flush before `cb_push_back`"; same discipline, raw form.)
   **Do NOT flag "no flush between write and credit" as a race by default** (the #48480 false-positive class). A flush is *not* needed when HW already orders the two — same initiator, same destination(s), same NoC, and same VC (multicast writes default to `NOC_MULTICAST_WRITE_VC` / static VC, or are coupled via `NOC_CMD_VC_LINKED`): same-VC packets are delivered in issue order, so a credit *write* cannot overtake the payload write. A flush **is** required when a precondition breaks: the credit is an **atomic** (`noc_semaphore_inc` — no write→atomic ordering rule even same-VC; #48478), the two ride **different NoCs** (e.g. payload on default noc, flag on `noc=1`; #48479), or a **different/unpinned VC**. Cross-check `1001357404`: a `linked=true` "fix" can itself hang (linked multicast + concurrent atomic on another cmd buffer). Ground the ordering guarantee (ISA `<arch>/NoC/Ordering.md`) and the Blackhole page-absent case **per `race-audit-all`'s source ladder + Ground-or-abstain** (missing living source ⇒ **UNCERTAIN**, never confirmed-RACE) — do not re-derive that policy here.
2. **Semaphore balance & direction.** Each `noc_semaphore_wait(val)` must be satisfiable by exactly the set of `inc`/`set` that reach it; an increment on a path whose wait is unconditional (or vice-versa) hangs or releases early. Producer waits for space / consumer waits for data — confirm the direction.
3. **Multicast fan-out count.** A `noc_semaphore_wait(N)` gated on a multicast must match the **actual number of receivers** that increment; an off-by-one in the grid/fan-out count → permanent stall.
4. **Reuse-before-complete.** A buffer reused (re-written, or freed) before its outstanding NoC transaction completes → corruption; confirm a barrier gates reuse.
5. **Cross-kernel pairing.** Producer (writer/reader on RISCV B/NC) and consumer live in **different kernels** — trace across the op's kernel set; a wait with no reachable remote incrementer is a deadlock, but only after chasing all participating kernels (a model/compute-layer kernel may issue the inc).

## Method
1. Enumerate (scope reaches beyond tt-llk):
   ```bash
   cd tt_metal && grep -rInE '\bnoc_semaphore_(wait|set|inc|set_multicast)\b|\bnoc_async_(read|write)_barrier\b|\bnoc_async_writes_flushed\b' \
     tt_metal/hw/inc/api ttnn/cpp models --include=*.h --include=*.cpp | grep -v '/tests/'
   ```
   **Exhaustive run — in scope, no sampling.** The tt-llk header tree has ~0 NoC sites — that is *expected* (the primitives live one layer up) and is **NOT** a reason to report "no findings / out of layer." The sites are in `tt_metal/hw/inc/api/dataflow` + `ttnn/cpp` + `models`, which the grep above (and the frontmatter) target. Enumerate **all** matching kernels into the run's coverage ledger and fan out per kernel-family; do not sample.
2. Per cross-core handshake, pair the signaller (write + inc/set) with the waiter (`noc_semaphore_wait` + read). Confirm a flush/barrier sits between the data write and the credit, and between the read and any buffer reuse.
3. Run checks 1–5; for multicast, resolve the receiver count from the grid.

## Verdict
- **Data flushed before credit, balanced wait/inc with correct direction, multicast count matches, reuse gated by completion** → SAFE.
- **Credit reachable before the data write is flushed** → DATA RACE — fix = add the missing `noc_async_write_barrier`/`writes_flushed` before the inc/set.
- **Wait/inc imbalance or wrong fan-out count on a reachable path** → DEADLOCK or early-release.
- **Buffer reused before transaction completion** → CORRUPTION — gate reuse with a barrier.
- **Risk only in a specific op's kernel set** → LATENT/author-level — name the kernel.

## Architecture note
The dataflow API is largely arch-agnostic, but the **ordering primitives and NoC atomic semantics differ per NoC/arch** — verify the barrier used matches the transfer type and that multicast/atomic behavior holds on the target arch (WH/BH/QSR). Don't transfer a verdict across arches without checking the NoC layer.

## Output
For each cross-core handshake: `file:line` of signaller (write+inc) and waiter (wait+read), barrier present/missing between data and credit, semaphore balance + direction, multicast fan-out count check, reuse-after-completion check, verdict (SAFE / DATA-RACE / DEADLOCK / CORRUPTION / LATENT) + one-line fix. End with totals.
