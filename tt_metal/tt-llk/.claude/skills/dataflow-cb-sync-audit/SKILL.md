---
name: dataflow-cb-sync-audit
description: Audit circular-buffer (CB) producer/consumer flow control between data-movement (reader/writer) and compute kernels — cb_reserve_back/cb_push_back/cb_wait_front/cb_pop_front credit balance, data-write-before-credit ordering (NOC flush), reserve/wait-before-access, capacity vs num_pages, single-producer/consumer, counter cache-coherency, and remote/sharded CB credits. Use after touching any cb_* call, a reader/writer/compute kernel, fifo_rd_ptr/fifo_wr_ptr/pages_received/pages_acked, or RemoteSender/ReceiverCBInterface. Scope reaches beyond tt-llk into tt_metal/hw/inc/api/dataflow and ttnn/models kernels.
user_invocable: true
---

# /dataflow-cb-sync-audit — Circular-buffer producer/consumer sync

> **Ground-truth precedence:** the live ISA doc (tt-isa-docs MCP, fetched each run) and the live dataflow API source outrank every rule, table, and example baked into this skill — treat those as dated illustrations. If a live source **contradicts** a baked rule here, do NOT silently proceed: surface the conflict to the user and ask whether the baked rule should be overwritten, discarded, or kept. Default to the live source.
>
> **MANDATORY — before any verdict, read the shared grounding policy.** The per-architecture **source ladder** (which docs to consult), the **ground-or-abstain** rule, and the **Source preflight** (list the sources you'll consult with their reachability + hierarchy, then PAUSE for the user) are defined once in `race-audit-all` → `.claude/skills/race-audit-all/SKILL.md`. **Your FIRST action is to `Read` that file and follow its "Ground-truth source ladder", "Ground-or-abstain", and "Source preflight" sections** — they are load-bearing: a verdict produced without them is ungrounded and MUST NOT be reported. If that file genuinely cannot be read, say so and **abstain** rather than proceed ungrounded. (If you were spawned by a `race-audit-all` sweep — your prompt already lists the confirmed sources — skip the Source preflight and do not pause; the orchestrator ran it once.)
>
> **Coverage — floor, not ceiling.** The grep patterns and site lists in this skill are a **seed, not an exhaustive enumeration**. After running them, widen the search with full reasoning. The techniques here are **illustrative examples, not the allowed set** — use any approach your reasoning suggests, including ones not listed: e.g. semantic search (by behavior/effect, not just token), resolving macros / wrappers / typedefs / indirection the literal pattern can't match, following the call graph to callers and callees, and diffing the WH/BH/QSR variants to catch a site present in one arch and missing in another. If you can find a hazard, primitive, or site the encoded patterns don't cover — by any means — pursue and report it; do **not** clamp a stronger analysis to this list or to these techniques. State any residual coverage gaps explicitly (no silent caps).
>
> **Execution — parallel by default.** When enumeration yields more than a few sites/files, **fan out concurrent `Agent` calls by default** (one per file/subsystem, a fresh context each), saturating the available concurrency (~10–16 at once); go inline only for a trivial set. The per-file fan-out described under *Thoroughness* is the **default**, not an exhaustive-only option. The cross-referencing/synthesis of results stays sequential (it must follow the per-unit findings). The heavyweight **Workflow** tool still requires explicit multi-agent opt-in — it is the opt-in exhaustive tier, not the default. Don't over-spawn a tiny diff.
>
> **Persisting results — single writer, incremental.** Agents only **return** their findings; they never write a shared file (no concurrent-write clobbering). If findings are persisted to a file, the orchestrator/caller is the **sole writer** and **appends each wave's returns as they arrive** — incremental, never only-at-the-end — so an interrupt preserves every completed wave's findings.

## The bug class (precise)
CBs are the credit-based FIFOs connecting **producers** (a reader on RISCV B, or the packer on T2) to **consumers** (compute on T0/T1, or a writer on NC). Flow control is two L1 counters per CB: `tiles_received` (a.k.a. `pages_received`, bumped by the producer in `cb_push_back`) and `tiles_acked` (`pages_acked`, bumped by the consumer in `cb_pop_front`). Misuse → **data corruption** (consumer reads a page before the producer finished writing it, or producer overwrites a page the consumer hasn't read) or **deadlock** (a wait whose credit never arrives). This is the dataflow layer the `mailbox-sync` / `cfg-word-overlap` "is the referenced data ready?" question hands off to.

## Ground-truth mechanism (`tt_metal/hw/inc/api/dataflow/dataflow_api.h`, verified)
- **Producer:** `cb_reserve_back(cb, n)` spins until `fifo_num_pages - (received - acked) >= n` (free space) — reads the consumer's `pages_acked` via `invalidate_l1_cache()` + `reg_read` (it may be updated remotely). Then writes the `n` pages into L1 at `fifo_wr_ptr`. Then `cb_push_back(cb, n)` does `pages_received += n` and advances `fifo_wr_ptr`.
- **Consumer:** `cb_wait_front(cb, n)` spins until `(received - acked) >= n` (reads `pages_received` uncached). Then reads the `n` pages at `fifo_rd_ptr`. Then `cb_pop_front(cb, n)` does `pages_acked += n` and advances `fifo_rd_ptr`.
- Counters are **16-bit wrapping**; comparisons use `uint16_t` subtraction (wrap-safe) — preserve that.
- Compute view (`cb_api.h`): unpack/T0 consumes input CBs (`cb_wait_front`/`cb_pop_front`), pack/T2 produces output CBs (`cb_reserve_back`/`cb_push_back`); the `UNPACK()/MATH()/PACK()` macros split one symmetric call. Dest-register handoff inside compute (`tile_regs_acquire/commit/wait/release`) is the **`MATH_PACK` semaphore** at API level → that's `semaphore-handshake-audit`'s domain, not this one.
- Remote/sharded CBs: `RemoteSenderCBInterface` / `RemoteReceiverCBInterface` (`circular_buffer_interface.h`) propagate credits **across cores via NOC** atomic-increments / semaphore writes.

## What to check
1. **Ordering — data fully written/flushed BEFORE the credit (the main race).** Producer must complete the page write before `cb_push_back`. For pages filled by NOC reads, a `noc_async_read_barrier` must precede `cb_push_back`; for cross-core credit (remote CB), the data NOC write must be **flushed before** the credit NOC write. Otherwise the consumer's `cb_wait_front` sees the page and reads stale/partial data. Symmetric on the consumer: finish reading before `cb_pop_front` (else the producer reuses the page). This is the CB analog of the `mailbox`/`fence=nop` ordering caveat.
2. **Credit balance.** Every `cb_reserve_back(n)` is matched by a `cb_push_back(n)` with the **same n**; every `cb_wait_front(n)` by a `cb_pop_front(n)`. Pushing more than reserved (or popping more than waited) drifts the counters → producer overruns unread data / consumer consumes unwritten. Walk every branch and early-return.
3. **Reserve-before-write / wait-before-read.** A write to the CB region without a preceding `cb_reserve_back`, or a read without `cb_wait_front`, touches an unsynchronized page.
4. **Capacity.** `n <= fifo_num_pages` for every reserve/wait — asking for more pages than the CB holds can never be satisfied → deadlock.
5. **Single producer / single consumer.** The `+=` on each counter is non-atomic and assumes one incrementer per side. Two producers or two consumers on one CB → counter race. (Multi-core fan-in/out must use the remote-CB interfaces, not raw shared counters.)
6. **Counter cache-coherency.** Each side reads the *other* side's counter, which a remote core may update via NOC — must use `invalidate_l1_cache()` / uncached `reg_read` (as the library does). A cached read → spin forever (deadlock) or proceed early (corruption).
7. **Remote/sharded CBs.** Verify the credit semaphore addresses, the NOC ordering (data-before-credit), and that sender/receiver page counts agree.

## Method
1. Enumerate CB sites:
   ```bash
   cd tt_metal && grep -rInE '\bcb_(reserve_back|push_back|wait_front|pop_front)\b|pages_(received|acked)|fifo_(rd|wr)_ptr|Remote(Sender|Receiver)CBInterface' \
     tt_metal/hw/inc/api ttnn/cpp models --include=*.h --include=*.cpp | grep -v '/tests/'
   ```
   **Exhaustive run — no sampling.** That grep yields the full kernel set (typically ~150–200 files across the CCL/dataflow families — all_gather, reduce_scatter, all_to_all, llama, moe, deepseek, broadcast, sdpa, matmul, prefetcher, unary/binary readers-writers). Enumerate them **all** into the run's coverage ledger and fan out (≥) one cell per kernel-family; **do not sample** — "sampled 6 of 186" is a blocking incompleteness per `race-audit-all`, not a caveat. This class is **in scope** (frontmatter): "the CB races live above tt-llk, run it separately" is not a valid skip — audit them here, now.
2. Per CB (per kernel pair), pair the producer (`reserve_back`+`push_back`) with the consumer (`wait_front`+`pop_front`) — they live in **different kernels** (reader↔compute, compute↔writer), so trace across the kernel set of the op.
3. Run checks 1–7. For ordering, confirm a NOC/read barrier sits between the data fill and `cb_push_back` (and between the read and `cb_pop_front`).

## Verdict
- **Balanced credits, reserve/wait before access, barrier orders data before the credit, n≤capacity, 1-producer/1-consumer, uncached counter reads** → SAFE.
- **`cb_push_back` reachable before the page write is flushed** → DATA RACE (consumer reads stale/partial) — fix = add the missing `noc_async_*_barrier` / flush before the credit bump.
- **reserve/push or wait/pop count mismatch on a reachable path** → corruption or deadlock.
- **n > capacity, or a credit that no producer/consumer ever bumps** → DEADLOCK.
- **Risk only in a specific op's kernel set, library primitives correct** → LATENT/author-level — name the kernel.

## Architecture note
The CB API is arch-agnostic (one copy in `tt_metal/hw/inc/api`); roles map as reader→RISCV B, writer→RISCV NC, compute→T0/T1/T2. Mechanism is the same across WH/BH/Quasar at this layer; the **ordering** primitives differ per NoC/arch — verify the barrier used matches the transfer type.

## Verified ground truth (don't re-derive)
- `cb_reserve_back`/`cb_wait_front`/`cb_push_back`/`cb_pop_front` semantics above are from `dataflow_api.h` (credit math, uncached counter read, 16-bit wrap). Canonical compute pattern (e.g. `eltwise_bw_gelu_poly.cpp`): `cb_reserve_back(out)` + `cb_wait_front(in)` → per-tile `tile_regs_acquire/commit/wait` + `pack_tile` → `cb_pop_front(in)` + `cb_push_back(out)`. The `tile_regs_*` half is `MATH_PACK` (defer to `semaphore-handshake-audit`); this audit owns the `cb_*` half.

## Cross-references (where this audit meets the others)
- **`mailbox-sync` / `cfg-word-overlap` "is the referenced memory ready?"** → discharged HERE: the address the mailbox broadcasts is derived from `fifo_rd_ptr`, valid only if `cb_wait_front` gated correctly AND the producer ordered its write before `cb_push_back`.
- **Data-before-credit ordering** ↔ `mmio-race` / NOC memory-ordering (the barrier discipline).
- **Dest-register `tile_regs_*` interleaved with `cb_*`** ↔ `semaphore-handshake-audit` (`MATH_PACK`).

## Output
For each CB / kernel pair: `file:line` of producer (reserve/push) and consumer (wait/pop), credit balance (n match), the barrier ordering data before the credit (present/missing, both sides), capacity check, producer/consumer cardinality, counter-read coherency, remote-CB credit path if applicable, verdict (SAFE / DATA-RACE / DEADLOCK / LATENT) + one-line fix. End with totals.
