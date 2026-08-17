---
name: mailbox-sync-audit
description: Audit LLK/compute-API use of the RISC↔RISC hardware mailboxes (mailbox_write/mailbox_read/mailbox_not_empty, TENSIX_MAILBOX*) for races/deadlock — push/pop balance per directed channel, call-count symmetry across threads, correct write-dest/read-src addressing, FIFO overflow, and the fence=nop ordering caveat. Use after touching any mailbox_write/mailbox_read, the CB tile-address/value broadcast (circular_buffer.h / cb_api.h get_tile_address/read_tile_value), unpack-to-dest dst_index passing, or the debug halt/unhalt handshake.
user_invocable: true
---

# /mailbox-sync-audit — RISC↔RISC mailbox handshake correctness

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
Mailboxes are **point-to-point FIFOs between pairs of baby-RISCV cores** (T0/unpack, T1/math, T2/pack, B) — a sync path entirely separate from Tensix semaphores (`semaphore-handshake-audit`), config registers, and MMIO. A misused mailbox → **deadlock** (blocking read on an empty FIFO that never gets written, or a writer stalled forever on a full FIFO) or **stale/lost data** (ordering not enforced; fence is a nop).

## Ground-truth HW semantics (confirm via tt-isa-docs MCP: `BabyRISCV/Mailboxes.md`)
- **16 directed FIFOs**: one *from* each of {B,T0,T1,T2} *to* each. None to/from NC. Each is single-writer / single-reader / order-preserving.
- **Depth = 4** 32-bit values per FIFO; **aggregate 4** across all four FIFOs a given core can write to. Pushing beyond that spills to a shared buffer and then **stalls the writer** until space frees.
- **Read of an empty FIFO BLOCKS** (no read-response until a write arrives) — this blocking pop *is* the handshake. A non-blocking "is anything present?" query is the `address & 4` form (`mailbox_not_empty`, reading `mailbox_base[t][1]`).
- **Addressing is asymmetric and resolved by HW** from (issuing core, index, read-vs-write). Convention the LLK follows: **write with the DESTINATION thread id, read with the SOURCE thread id** — `mailbox_write(dest)` on the source core and `mailbox_read(src)` on the dest core name the *same* physical FIFO. (`ThreadId`: Unpack=1/T0, Math=2/T1, Pack=3/T2; `mailbox_base[idx]` → `TENSIX_MAILBOX{idx}_BASE`.) So writer-index ≠ reader-index for the same channel — do **not** flag that asymmetry as a bug.
- **`fence` is a nop and the Load/Store unit reorders** (`MemoryOrdering.md`): a mailbox access is **not** ordered against *other* memory by default.

## LLK primitives (`ckernel.h`)
- `mailbox_write(thread, data)` → `mailbox_base[thread][0] = data` (push, RISC store).
- `mailbox_read(thread)` → `mailbox_base[thread][0]` (blocking pop, RISC load).
- `mailbox_not_empty(thread)` → `mailbox_base[thread][1] > 0` (non-blocking query).
- (`debug_mailbox_base` / `record_mailbox_value` / `clear_mailbox_values` are the **debug scratch** mailbox in L1 — a different thing; not a sync FIFO.)

## What to check
1. **Push/pop balance per directed channel.** Each `mailbox_write(dest)` on the source must be matched 1:1 by a `mailbox_read(src)` on the destination, over any complete iteration. Imbalance → reader blocks on empty (deadlock) or writer stalls at full (after 4 unconsumed).
2. **Call-count symmetry across threads (the deadlock trap).** The broadcast pattern relies on all participating threads invoking the same function the same number of times. In the compute API this is guaranteed because one symmetric call is split by the `UNPACK()/MATH()/PACK()` macros. Flag any control-flow path where one thread issues the write (or read) but a peer conditionally skips its half — that desyncs the FIFO. This is the cross-layer analog of semaphore post/get imbalance: a **user kernel** can introduce it even when the LLK is correct.
3. **Direction / single-writer-single-reader.** Confirm each channel has exactly one writer and one reader. Two readers popping one FIFO would race (each value goes to only one). Verify via the write-dest/read-src convention that math vs pack read *different* FIFOs (T0→T1 vs T0→T2), not the same one.
4. **Overflow.** Depth-4; if a producer can push >4 without the consumer popping, the writer stalls. Confirm steady-state is ≤ depth (normally 1-deep).
5. **Ordering (fence=nop).** If a mailbox handshake is used to imply that *other* memory is ready (e.g. "wrote tile data to L1, then signal via mailbox"), the producer needs explicit ordering (a read-back fence) — the mailbox write alone does not fence the prior store. SAFE when the mailbox only carries a self-contained value consumed by data dependency AND data readiness is enforced by separate sync (CB/NOC). Flag any site that leans on the mailbox to order unrelated memory.
6. **Signal-before-blocking-write ordering + completion fence (deadlock).** When a producer notifies a consumer through **two** mechanisms in the same function — a `semaphore_post` (consumer-release) AND a `mailbox_write` carrying related data — two things must hold:
   - **(a) Issue order:** the `semaphore_post` must come **before** the `mailbox_write`. `mailbox_write` **stalls the producer when the FIFO is full** (depth-4), so issuing it first can wedge the producer on a full FIFO *before* it posts, while the consumer is still blocked on the un-posted semaphore and never drains the mailbox → **deadlock**.
   - **(b) Completion fence:** program order alone is NOT enough — RISC stores are not ordered against each other (`fence` is a nop; the load/store unit reorders). Force the post to actually *land* before the mailbox write with a **read-back fence**: store-then-load the semaphore's `pc_buf_base[PC_BUF_SEMAPHORE_BASE + <sem>]` address and consume the loaded value (e.g. `andi x0, <load>, 0`) between the post and the write.

   Canonical correct form (`_llk_unpack_get_tile_`): `semaphore_post(UNPACK_OPERAND_SYNC)` → read-back fence on that semaphore → `mailbox_write(<consumer>, byte_address)`. Flag **both** the reversed order and a post that lacks a completion fence ahead of the mailbox write.

## Method
1. Enumerate every site (exclude the debug-scratch mailbox):
   ```bash
   cd tt_metal
   grep -rInE '\bmailbox_(write|read|not_empty)\(' tt-llk tt_metal/hw/inc/api 2>/dev/null \
     | grep -vE 'record_mailbox|clear_mailbox|debug_mailbox|mailbox_base\[|inline |//'
   ```
2. Per site, decode the channel via the convention (writer's `dest` id + issuing thread → directed FIFO; reader's `src` id + issuing thread → same FIFO). Pair writes with reads.
3. Run checks 1–5. For balance/symmetry, read the enclosing function and confirm all threads reach it equally (watch `if constexpr`/runtime branches and the `UNPACK/MATH/PACK` split).

## Verdict
- **Balanced 1:1 per channel, symmetric across threads, 1-writer/1-reader, ≤depth, value self-contained (no cross-memory ordering assumed)** → SAFE.
- **A reachable path where write and read counts diverge across threads** → DEADLOCK (blocking-read hang or full-FIFO stall).
- **Mailbox used to imply other memory is ready, with no explicit ordering** → ORDERING-RACE (fence=nop).
- **A (blocking) `mailbox_write` issued before the `semaphore_post`/credit that releases the consumer** → DEADLOCK-ORDERING — producer can stall on a full FIFO before signalling, consumer waits on the un-posted release. Fix = post the release signal first, then write the mailbox.
- **Risk only on a user-kernel / cross-layer path, LLK itself balanced** → LATENT — say so; it's an author-level invariant.

## Architecture note
WH/BH share the model (ISA confirms both); the compute-API files (`circular_buffer.h`, `cb_api.h`) are arch-agnostic, so the same usage applies across arches — verify Quasar's mailbox HW semantics before extending verdicts there.

## Verified non-bugs (don't re-flag)
- **CB tile address/value broadcast** (`tt_metal/hw/inc/api/dataflow/circular_buffer.h`, `tt_metal/hw/inc/api/compute/cb_api.h`, `get_tile_address`/`read_tile_value`): unpack(T0) `mailbox_write(MathThreadId)` + `mailbox_write(PackThreadId)` → two **distinct** FIFOs (T0→T1, T0→T2); math reads T0→T1, pack reads T0→T2 via `mailbox_read(UnpackThreadId)`. 1 push / 1 pop each per call, value consumed by data dependency, symmetric via the macro split. SAFE.
- **Unpack-to-dest dst_index**: math `cmath_common.h` `mailbox_write(UnpackThreadId, dst_index)` (T1→T0) ↔ unpack `cunpack_common.h` `mailbox_read(MathThreadId)` (T1→T0), both gated on `unpack_destination == DestReg`. Matched pair. SAFE.
- **Debug halt/unhalt** (`ckernel_debug.h` `dbg_thread_halt`/`dbg_thread_unhalt`): unpack→math idle-notify (T0→T1) + math→unpack unhalt (T1→T0), 1 write / 1 read each side, blocking. Debug-only. SAFE.

## Output
For each mailbox site: `file:line`, decoded directed channel (src→dst FIFO), the matching read/write site, push/pop balance + cross-thread call-count symmetry, 1-writer/1-reader check, overflow check, ordering check, verdict (SAFE / DEADLOCK / ORDERING-RACE / LATENT) with one-line reasoning + fix. End with the channel map and totals per verdict per arch.
