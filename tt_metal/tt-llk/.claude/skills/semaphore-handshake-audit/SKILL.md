---
name: semaphore-handshake-audit
description: Audit LLK inter-thread synchronization (Tensix semaphores + ATGETM/ATRELM mutexes) for races/deadlock — SEMINIT correctness vs usage, post/get balance, wait-direction, RISC-MMIO-vs-Tensix ordering, and mutex acquire/release balance. Use after touching any t6_semaphore_*/semaphore_post/semaphore_get/SEMINIT/SEMWAIT/SEMPOST/SEMGET/t6_mutex_* or any math↔pack / unpack↔math handshake (MATH_PACK, UNPACK_TO_DEST, UNPACK_SYNC, MATH_DONE, FPU_SFPU).
user_invocable: true
---

# /semaphore-handshake-audit — Inter-thread semaphore & mutex correctness

## The bug class (precise)
The three Tensix threads coordinate through **8 hardware semaphores** (`Semaphores[0..7]`, each a 4-bit `Value` 0–15 plus a `Max`) and **two ATGETM/ATRELM mutexes**. Unlike config-register races (covered by `cfg-word-overlap-audit`, `reconfig-stall-audit`, `mmio-race-audit`), these bugs are in the *handshake protocol*: a mis-initialized, unbalanced, wrong-direction, or wrongly-ordered post/wait/get → **lost synchronization (silent data corruption)** or **deadlock (TENSIX TIMED OUT)**.

## Ground-truth HW semantics (confirm via tt-isa-docs MCP: `SEMINIT.md`, `SEMWAIT.md`, `SEMPOST.md`, `SyncUnit.md`)
- `SEMPOST` → `Value++` **capped at 15** (HW). `SEMGET` → `Value--` **floored at 0**. Both silently no-op at the limit.
- `SEMWAIT(block_mask, sem_mask, cond)`: stalls the thread's instruction issue while the condition holds. `STALL_ON_MAX` = block **while `Value == Max`** (wait for a consumer to drain). `STALL_ON_ZERO` = block **while `Value == 0`** (wait for a producer to post).
- `SEMINIT(NewMax, NewValue, sem_mask)` sets **both `Value` and `Max`** atomically. **Crucial: `Max` is used ONLY by `SEMWAIT`; `SEMPOST` ignores it and still caps at 15.** So `Max` is purely the wait threshold — it does **not** bound posting. Backpressure exists only if the *producer* calls `wait_on_max` **before** each post.
- LLK wrappers (`ckernel.h`): `t6_semaphore_init(idx, min, max)` → `TTI_SEMINIT(max, min, …)` (so wrapper `min` = initial `Value`, `max` = `Max`). `t6_semaphore_post/get` are Tensix `SEMPOST/SEMGET` (in-stream, ordered). `semaphore_post/get` (no `t6_`) are **RISC MMIO** writes to `pc_buf_base[PC_BUF_SEMAPHORE_BASE+idx]` — asynchronous to the Tensix stream. The `LLK_ASSERT(value<MAX)` / `(value>0)` guards are **debug-only** → release builds silently over/underflow.

## Semaphore map (WH/BH — `ckernel_structs.h`)
| # | Name | Roles |
|---|------|-------|
|0|`FPU_SFPU`| fpu↔sfpu; also reused (aliased `SFPU_FPU`) in some experimental kernels |
|1|`MATH_PACK`| math↔pack on **dest** register (the main double-buffer) |
|2|`UNPACK_TO_DEST`| unpack↔math, unpack-direct-to-dest |
|3|`UNPACK_OPERAND_SYNC`| unpack↔pack/math operand get/release |
|4|`PACK_DONE`| pack iteration begin/end (perf) |
|5|`UNPACK_SYNC`| RISC↔unpack, config-context acquire/release |
|6|`UNPACK_MATH_DONE`| (see name reuse note below) |
|7|`MATH_DONE`| math-done barrier for unpack-to-dest |

## What to check — and the rules

### 1. INIT correctness vs usage (do this FIRST — it's the most-missed)
- **Every semaphore used with `wait_on_max`/`STALL_ON_MAX` MUST be `SEMINIT`'d with `Max` = the intended buffer depth**, by one thread, *before any thread uses it*. If it relies on the HW-reset default `Max`, the producer's backpressure gate is wrong → over-post (silent corruption) or premature/never block.
  - In-tree, only `MATH_PACK` is `SEMINIT`'d inside tt-llk: `_llk_math_pack_sync_init_` (`llk_math_common.h`) sets `Max=1` (`DstSync::SyncFull`) or `Max=2` (`SyncHalf`), `Value=0`. **`UNPACK_TO_DEST` (waited-on-max at `cunpack_common.h`) and `MATH_DONE` have no `SEMINIT` in tt-llk** — trace the kernel/firmware startup (e.g. `tt_metal/hw/inc/api/compute/.../*_hw_startup.h`, `compute_kernel_hw_startup`) to confirm their `Max`. Flag any wait-on-max semaphore with no reachable `SEMINIT`.
- **Initial `Value` must match the empty/full convention** — producer/consumer semaphores start at `Value=0` (empty). A stale non-zero value (no re-init across kernels, or across a `DstSync` mode switch) desyncs from cycle one.
- **The issuing thread + ordering**: semaphores are thread-agnostic (single bank), but `SEMINIT` executes in one thread's stream. It must land before any thread's first wait/post/get. Verify the init barrier (e.g. `_llk_math_pack_sync_init_` does `tensix_sync()` + `while(semaphore_read(MATH_PACK)>0){}` to drain prior packs **before** re-`SEMINIT`). Missing barrier → init races a peer still using the old value.
- **Mode-switch re-init**: `SyncFull`↔`SyncHalf` need `Max` 1↔2. Changing dest-sync mode without re-`SEMINIT` (and a dest drain) leaves `Max` mismatched to the real buffer depth.
- `Max` ≤ 15 and ≥ the deepest outstanding count the producer can post between waits. Since `SEMPOST` ignores `Max`, ALSO confirm usage rule #2.

### 2. Post/get (producer/consumer) balance
- Exactly **one `SEMGET` per `SEMPOST`** over any complete iteration. Walk every branch: a post inside an `if` whose get is unconditional (or vice-versa), or an early `return`/`continue` between them, drifts `Value` → producer eventually wedges at `Max` forever (deadlock) or consumer drains a slot that was never filled.
- Because `SEMPOST` ignores `Max`, **every producer post must be preceded (in that thread's program order) by the `wait_on_max` gate** — that gate is the only thing preventing overshoot past the intended depth. A post on a path that skips the wait → silent overflow toward 15.

### 3. Wait-direction
- Producer side waits `STALL_ON_MAX` (block while full); consumer side waits `STALL_ON_ZERO` (block while empty), then `SEMGET`. Swapped condition → either no synchronization (proceeds immediately) or permanent stall. Canonical good pair: math `wait_on_max`→compute→`post(MATH_PACK)`; pack `wait_on_zero`→pack→`get(MATH_PACK)`.

### 4. RISC-MMIO vs Tensix-instruction ordering
- `semaphore_post`/`semaphore_get` (no `t6_`) are RISC stores that execute asynchronously to the Tensix stream (overlaps `/mmio-race-audit`). When a RISC `semaphore_post` is meant to gate a Tensix consumer, there must be a `SEMWAIT`/`TTI_SEMGET` that actually stalls on it (the `UNPACK_SYNC` context-acquire idiom: RISC `semaphore_post` → `STALLWAIT(STALL_UNPACK, TRISC_CFG)` → MOP → `t6_semaphore_get`). A bare MMIO post/get with no in-stream wait can land early/late. On **Quasar**, Auto-TTSync changes these ordering rules (see memory `quasar-auto-ttsync`) — don't apply WH/BH manual-ordering verdicts blindly.

### 5. Mutex (ATGETM/ATRELM) balance & deadlock
- `t6_mutex_acquire(idx)`/`t6_mutex_release(idx)` (`mutex::REG_RMW`=0, `mutex::SFPU`=4) must be **balanced on every path** — an early return between acquire and release holds the mutex and **deadlocks every other thread** that acquires it. The mutex only helps parties that take it (`REG_RMW` is **not** taken by the math thread — see `cfg-word-overlap-audit`).
- `mutex::SFPU` is **declared but currently never acquired** in tt-llk. It exists because SFPU instructions can issue from both T1 and T2; if a kernel ever issues SFPU from pack concurrently with math without taking it → race. Flag any new cross-thread SFPU issue that doesn't.

### 6. Cross-thread deadlock cycle
- Build the wait-for graph: math waits `MATH_PACK`-max (needs pack `get`) and `UNPACK_TO_DEST`-zero (needs unpack `post`); pack waits `MATH_PACK`-zero (needs math `post`); unpack waits `UNPACK_TO_DEST`-max (needs math `get`). Any cycle with no thread able to make progress = deadlock. Most often introduced by reordering wait-before-work, or an op that posts on one thread whose consumer is conditionally skipped, or mismatched init/uninit pairing across threads.

## Method
1. **Enumerate** every site, tagged by thread via filename (`*unpack*`/`cunpack`=T0, `*math*`/`cmath`=T1, `*pack*`/`cpack`=T2):
   ```bash
   cd tt_metal/tt-llk
   grep -rInE "t6_semaphore_(post|get|wait_on_max|wait_on_zero|init)|semaphore_(post|get)\(|TTI_SEM(POST|GET|WAIT|INIT)|t6_mutex_(acquire|release)|TTI_ATGETM|TTI_ATRELM" \
     tt_llk_* --include=*.h | grep -v /tests/
   # init sites (incl. above tt-llk — kernels/firmware do some SEMINIT):
   grep -rInE "SEMINIT|t6_semaphore_init" tt_metal/hw tt_metal/tt-llk --include=*.h
   ```
2. **Per semaphore**, list every (thread, op, condition, file:line). Classify ops: `SEMINIT`(Max,Value) / `SEMPOST`(producer) / `SEMGET`(consumer) / `SEMWAIT`(max|zero) / RISC-MMIO post|get.
3. **Run the rules**: init present + correct Max/Value (#1) → balance over all branches (#2) → directions (#3) → MMIO ordering (#4) → mutex balance (#5) → deadlock graph (#6).
4. **Confirm semantics** against tt-isa-docs (WH/BH) when unsure what a `SEMWAIT` condition or `Max` does.

## Verdict
- **Init present (right thread, right Max/Value, ordered before use) + balanced post/get on all paths + correct directions + MMIO posts gated by an in-stream wait + mutexes balanced + no wait cycle** → SAFE.
- **Wait-on-max semaphore with no reachable `SEMINIT`, or Max ≠ buffer depth, or stale initial Value** → INIT-BUG (corruption or deadlock depending on default).
- **Post/get imbalance or skipped wait-before-post on a reachable branch** → RACE/DEADLOCK (real).
- **Window exists only on an unused/experimental path, or value-invariant** → LATENT — say so; let the maintainer decide.

## Architecture note
WH/BH share this model. **Quasar** adds HW Auto-TTSync (memory `quasar-auto-ttsync`) that auto-orders RISC↔Tensix, so rule #4 verdicts differ; the semaphore-protocol rules (#1–#3, #5–#6) still apply. Verify the semaphore map/wrappers resolve for Quasar before concluding.

## Verified non-bugs (don't re-flag)
- `deepseek_compute_kernel_init` (`tt_metal/hw/inc/api/compute/experimental/deepseek_compute_kernel_hw_startup.h`): `MATH` inits `FPU_SFPU` (sem 0), `PACK` inits `SFPU_FPU` (= `UNPACK_MATH_DONE`, sem 6). Different semaphores **on purpose** — a two-semaphore bidirectional FPU↔SFPU handshake (documented in its `@note`), not a typo.

## Thoroughness (optional, full sweep)
Best done as: deterministic enumeration (grep → per-semaphore table) → one agent per semaphore to run rules #1–#6 → **adversarial verify**: for each SAFE verdict, try to construct a branch/order that breaks balance or deadlocks; for each wait-on-max semaphore, try to prove its `SEMINIT` is unreachable. Only run a Workflow if the user opts into multi-agent orchestration. Never upgrade "balanced in the common path" to SAFE without checking early-return/conditional paths.

## Output
For each semaphore/mutex: the per-thread op table (`file:line`), `SEMINIT` Max/Value + issuing thread + whether ordered before use, balance result, wait-directions, MMIO-ordering, and verdict (SAFE / INIT-BUG / RACE-DEADLOCK / LATENT) with one-line reasoning + fix. End with a wait-for-graph note (any cycle) and totals per verdict per arch.
