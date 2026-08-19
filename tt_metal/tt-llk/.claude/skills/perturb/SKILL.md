---
name: perturb
description: Reproduce a flaky/timing-dependent kernel failure (suspected undocumented HW race) by injecting NOPs/delays to shift inter-thread / inter-kernel / inter-core timing until the failure becomes frequent enough to isolate and minimize into a deterministic reproducer. Sweeps NOP count × injection position × actor, records the max-error scenario PER actor to a report file (rewritten after each actor finishes so you can stop early), then (on request) minimizes the test around a chosen scenario. Works for tt-metal ttnn op tests and tt-llk kernel tests.
user_invocable: true
---

# /perturb — timing-perturbation reproducer for flaky kernel races

## What this does & when to use it
Some failures are timing races (often undocumented-HW-race), invisible to static audits and rare enough that plain re-runs won't reproduce them. This skill deliberately shifts the *relative timing* of the actors in the op's kernel pipeline — the compute threads, the dataflow kernels, and (for a multi-core op) the participating cores relative to one another — by injecting NOPs (compute threads) or RISC delays (dataflow kernels) at synchronization points, sweeping the amount, to widen the race window until the failure becomes frequent — ideally deterministic — so it can be minimized into a small, known-offset reproducer.

Use when: a test fails intermittently (bad output / NaN / hang), static race audits came back clean, and you need a small, high-frequency, *known-offset* reproducer.

Two failure modes are detected and reported separately: **data mismatch** (wrong/NaN output vs golden) and **hang** (device timeout/wedge).

## The actors
A Tensix op pipeline has up to five perturbable actors **per core**:
- **Compute (Tensix coprocessor threads):** `unpack` (TRISC0), `math` (TRISC1), `pack` (TRISC2).
- **Dataflow (baby-RISC kernels):** `reader` (BRISC), `writer` (NCRISC).

The compute threads and the dataflow kernels need **different NOP primitives** (below).

For a **multi-core op** — multicast, sharded, or any producer/consumer split across cores via a remote CB — the same five actors exist independently on every participating core, and the **relative timing between cores** (mcast sender vs receiver, a producer core vs its remote-CB consumer core) is its own perturbation axis that no on-core sweep can move. Single-core races are the common case, so sweep one core first and escalate to a second core's actors (see *Escalations*) when the on-core sweep is clean but a cross-core handshake sits in the op's call graph.

## Perturbation primitives
- **Compute threads** — Tensix NOP `TTI_NOP` (`ckernel_ops.h`), emitted into exactly one thread via the `UNPACK(...)`/`MATH(...)`/`PACK(...)` pass-through macros (`compute_kernel_api.h`).
- **Reader/Writer** — these are RISC kernels doing NoC ops, NOT Tensix threads, so `TTI_NOP` does not reach them. Use a **RISC-V `asm volatile("nop")`**.
  - Do **NOT** use `riscv_wait(cycles)` for the fine sweep: it busy-spins reading the WALL_CLOCK MMIO register (`risc_common.h`), so its unit is AICLK cycles (not ms) but the per-iteration MMIO-read overhead is its resolution floor — small N (1..100) becomes coarse/non-linear. Keep `riscv_wait(cycles)` only as a **coarse fallback** (100/500/1k/2k/5k...) to reach large offsets when the fine `asm nop` sweep finds nothing.

**ALWAYS UNROLL** the NOP loop so the injected cycle count is exact (no loop counter/branch overhead) — this matters for a clean monotonic sweep and for pinning the precise cycle offset in the reproducer. Because the count is a compile-time constant, force full unroll with `_Pragma("GCC unroll N")` (N a literal > max sweep count):
```c
// compute (Tensix):
#define EMIT_NOPS()      do { _Pragma("GCC unroll 128") for (uint32_t _i=0;_i<(NOP_COUNT);++_i){ TTI_NOP; } } while(0)
// reader/writer (RISC):
#define EMIT_RISC_NOPS() do { _Pragma("GCC unroll 128") for (uint32_t _i=0;_i<(NOP_COUNT);++_i){ asm volatile("nop"); } } while(0)
```

## Sync-point anchors (injection positions)
Anchors = the actor's start + immediately AFTER each synchronization point it participates in. Derive them from BOTH the **ISA docs** (what the primitive is) and the **kernel/LLK code** (WHERE it actually sits in this op's call graph). Sync points fall on four surfaces — anchor every one the op contains, so the sweep spans the same hazard space a full static race audit does, not just the CB/semaphore handshakes:

- **Cross-thread, within a core** — `SEMWAIT`/`SEMPOST` on the SyncUnit; `STALLWAIT` at the Wait Gate; `SETDVALID`/`CLEARDVALID` (the SrcA/SrcB bank + `AllowedClient` handshake); and **config/reconfig writes** — a reconfig (`reconfig_data_format`, `llk_*_reconfig`, `*_uninit`) or a shared-CFG-word RMW is an ordering point between the thread that writes it and the thread/unit that consumes the old value, so anchor right after it (this is where the cfg-word-overlap and reconfig-stall hazards live).
- **RISC↔Tensix** — a baby-RISC **MMIO write** to a Tensix config/GPR register vs the Tensix instruction/MOP that consumes it: anchor after the store on the RISC side and at the consumer on the Tensix side, and around the `mop_sync()`/`tensix_sync()` drains that order them.
- **RISC↔RISC** — **mailbox** push/pop / `fence` handshakes between `reader` (BRISC) and `writer` (NCRISC) — and RISC↔TRISC where the op broadcasts a tile address/value through one.
- **Cross-core (NoC)** — CB credits `cb_reserve_back`/`cb_push_back` (producer side) and `cb_wait_front`/`cb_pop_front` (consumer side); raw `noc_semaphore_wait`/`noc_semaphore_inc`; multicast fan-out; and the **data-before-credit barrier** (`noc_async_write_barrier`/`writes_flushed` before a remote credit). For a multi-core op these anchor on the actors of *both* cores.

Progression: **start → after 1st sync point → … → after last sync point** for one actor; then move to the next actor. Actor order: `unpack → math → pack → reader → writer` (repeated per core for a multi-core op).

Example anchor set (untilize wide, single-core reference run — yours will differ per op):
- unpack: start; after `wait_for_next_context` (UNPACK_SYNC acquire); after `UNPACR`/`SETDVALID`; after any mid-pipeline `reconfig_data_format`.
- math: start; after `math_dest_wait` (SEMWAIT MATH_PACK); after the A2D datacopy (MOVA2D+SETRWC); after `math_dest_section_done` (SEMPOST).
- pack: start; after `packer_wait_for_math_done` (SEMWAIT); after `PACR`; after `pack_dest_section_done`.
- reader: start; after `cb_reserve_back`; after `read_barrier` (before `push_back`); after any RISC↔RISC mailbox push.
- writer: start; after `cb_wait_front`; after per-row `write_barrier`; before `cb_pop_front`.

That example is single-core with no MMIO/mailbox/mcast traffic, so those anchors don't appear; an op that reconfigures formats, coordinates the two RISCs through a mailbox, or fans out over a multicast/remote-CB handshake adds the corresponding config, mailbox, and NoC-semaphore anchors from the list above.

## Injection mechanics (JIT, no host rebuild)
Control everything with compile-time `#define`s so the JIT kernel hash changes per value → the kernel **recompiles per count with no host rebuild**. Verify a recompile actually happened via the `BuildKernels` "JIT cache stats" line showing MISSES on a fresh count.

- **Compute kernel:** put `NOP_COUNT` / `NOP_THREAD` (0/1/2) / `NOP_POS` `#define`s at the **TOP of the compute .cpp, BEFORE the includes**, so the LLK headers and the compute-API orchestration header see them. `EMIT_NOPS` uses `TTI_NOP` (defined deeper in the include chain) — fine, macro bodies expand lazily at the use site.
- **Orchestration hub:** the compute-API block function that drives the op (e.g. `pack_untilize_block` in `pack_untilize.h`) is where all three compute threads' sync points are visible at the `UNPACK()/MATH()/PACK()` level — inject math/pack "after sync point" anchors there rather than deep in the LLK. Anchors that live only inside an LLK function (e.g. unpack's `UNPACK_SYNC`) go in that LLK header (e.g. `llk_unpack_A.h`).
- **Reader/Writer:** each is its own translation unit that does NOT include the compute kernel — give each its **own self-contained** `#define`s (e.g. `R_NOP_COUNT`/`R_NOP_POS`, `W_NOP_COUNT`/`W_NOP_POS`) + its own `EMIT_RISC_NOPS`.
- **Guard every site** so other kernels/TUs are unaffected: `#if (NOP_THREAD==X) && (NOP_POS==Y)` (undefined macros evaluate to 0 in `#if` → inactive). At count 0 everything is behaviorally identical to the original.

## Run every test in slow dispatch mode
**All test invocations in this skill — every count, every actor, and the `NOP_COUNT=0` baseline — MUST run with fast dispatch disabled:**
```bash
export TT_METAL_SLOW_DISPATCH_MODE=1   # or prefix each run: TT_METAL_SLOW_DISPATCH_MODE=1 pytest ...
```
Slow dispatch bypasses the asynchronous command-queue path: the host issues one operation at a time and waits for each to complete before the next, using the synchronous `ReadFromBuffer`/`WriteToBuffer` APIs instead of the `Enqueue*` async ones. This removes the fast-dispatch CQ overlap so on-device execution ordering is host-serialized and deterministic. Perturbing timing under slow dispatch therefore isolates the *kernel / coprocessor / NoC* timing race itself rather than a host-dispatch scheduling artifact, and keeps the `(actor, position, NOP_COUNT)` offset meaningful and stable when you replay the reproducer. It also makes the sweep more reproducible (less host-side timing jitter between counts).

Set it once for the whole sweep so every data point is comparable, and confirm it took effect (the run should not create command queues). If a test *only* fails under fast dispatch and never under slow dispatch, that itself is a finding: the race is in the host dispatch layer, not the kernel — note it in the report rather than continuing to perturb the kernel.

## The sweep
For each (actor, position):
1. Set that actor+position active.
2. Sweep `NOP_COUNT = 0 .. MAXN` (default 100). For each count: `sed` the `#define` in the target source (forces JIT recompile), run the test **10×** in one process **with `TT_METAL_SLOW_DISPATCH_MODE=1`** (see above), record `fails/10` (data-mismatch or NaN) and any hang/wedge.
3. If a count **hangs/wedges** the device (timeout), stop that sweep and reset (`tt-smi -r`) before continuing — a wedge blocks all further runs.
Per-count wall time ≈ device-open + JIT recompile + 10 runs. Note: `NOP_COUNT=0` is the baseline (should match the test's normal pass/fail rate).

## Failure detection
Use the test's own correctness check. For a **lossless** op (e.g. untilize TILE→ROW_MAJOR), a correct output equals the input bit-exactly, so `torch.equal(out, in)` + a NaN check is a strict, sufficient detector. Otherwise use the test's `assert_with_pcc`/`assert_equal`. Count a run as a failure on any mismatch/NaN; track hangs separately.

## Output — per-actor max-error report (incremental; written BEFORE minimizing)
Maintain a single report file recording, **per actor and per position**, the count(s) with the **highest failure frequency** (and the frequency), plus any hang-inducing counts. Include the exact `(actor, position, NOP_COUNT)` and the failure kind (mismatch/NaN/hang). This lets the user attempt reduction themselves. If NOTHING reproduced anywhere, state that plainly and list the escalations tried / not tried.

**Write it INCREMENTALLY — create the file when the first sweep starts and REWRITE it as soon as EACH actor finishes its full sweep, never only once at the very end.** The full `unpack → math → pack → reader → writer` sweep is long (an actor can take hours), and the user may stop early to start working from partial results — so after any actor completes, the file must already hold every result gathered so far. Mark which actors are done vs still pending each time you rewrite it, so a partial file is unambiguous.

**File name & location:** use the canonical name `perturb_report.md` (or `perturb_report_<op>.md`). It is a generated artifact — write it **outside the tracked repo tree** (e.g. beside the checkout or in a scratch/tmp dir), not inside `tt-llk`, so it can't be committed by accident. There is **no `.gitignore` rule** for it (deliberately — we don't add report-file globs to the repo `.gitignore`); if you do place it inside the repo, do not `git add` it and do not add an ignore rule for it.

**Then PROMPT the user**: ask whether to minimize the test around a chosen max-error scenario. **Do not auto-minimize.** Only minimize when the user asks, and keep iterating on the user's direction.

## Minimization (only when the user asks)
For the chosen `(actor, position, NOP_COUNT)`:
- Find the earliest point in the kernel where an intermediate result can be checked against expected; if you can already conclude "it has failed" there, delete/short-circuit everything downstream → smaller kernel.
- Keep the perturbation pinned; shrink shapes/loops/tiles while the failure frequency holds.
- Deliverable: minimal kernel + fixed `(actor, position, NOP_COUNT)` reproducing with high probability. If it doesn't fire immediately on replay, nudge NOP_COUNT (or switch to the coarse `riscv_wait` fallback) until it does.

## Escalations (if the fine 0..100 single-actor sweep finds nothing)
1. **Coarse magnitude** — larger NOP counts / `riscv_wait(cycles)` at 100/500/1k/2k/5k/10k to reach much bigger offsets.
2. **Two-actor combinations** — inject in TWO actors at once to set a specific *relative* phase (e.g. reader+unpack, math+pack). CB/semaphore handshakes self-correct single-sided delays, so a specific relative offset is a genuinely different perturbation the single-actor sweep cannot create.
3. **Cross-core** — for a multi-core op, perturb an actor on a *second* core (e.g. delay a multicast receiver's `unpack`/`reader`, or a remote-CB consumer core) to shift the sender-vs-receiver / producer-core-vs-consumer-core phase. On-core sweeps cannot move inter-core timing, so a cross-core handshake window (raw `noc_semaphore_*`, mcast fan-out, remote CB) only widens this way.
4. Consider that the failure may be board/topology/load-specific (e.g. only reproduces on the exact CI hardware) — a clean single board may never exhibit it regardless of timing.

## Applicability: tt-metal AND tt-llk
The METHOD is identical for both (same Tensix 3-thread + dataflow model); only the MECHANICS differ:
- **tt-metal (ttnn op tests):** kernels live under `ttnn/cpp/ttnn/operations/.../device/kernels/{compute,dataflow}/`; JIT-compiled when the op runs; test = pytest ttnn test; failure = output/PCC check. Perturb the op's compute kernel + its reader/writer. Run with `TT_METAL_SLOW_DISPATCH_MODE=1` (see "Run every test in slow dispatch mode").
- **tt-llk (kernel tests):** kernels/harness under `tt_metal/tt-llk/tests/` (`tests/sources/`, `tests/python_tests/`); two-phase compile-producer/consumer flow (`CHIP_ARCH` selects arch); failure = the harness's golden comparison. Perturb the test's unpack/math/pack kernels (and its data-movement) the same way. The LLK harness drives kernels directly rather than through the fast-dispatch command queue, but still set `TT_METAL_SLOW_DISPATCH_MODE=1` so any metal-dispatch path it touches stays synchronous. See this repo's `run-test`/`debug-kernel` skills for driving LLK tests.
In both, the compute-thread injection uses `UNPACK/MATH/PACK`+`TTI_NOP`; dataflow injection uses `asm("nop")`, and every run is in slow dispatch mode.

## Gotchas
- **NOP injection can MASK an intra-thread pipeline-latency hazard, not only expose a race.** A missing result-latency pad (the instruction-latency class — a dependent instruction consuming a multi-cycle result before it's ready) is *fixed* by adding a NOP. So if a failure vanishes the moment you add a few NOPs at one thread's start, with no relative-timing / cross-actor story, suspect a scheduling/latency hazard in that thread and hand it to `instruction-latency-audit` rather than reading the sweep as "timing race reproduced." A clean perturb sweep does not clear that class.
- A hang wedges the device → `tt-smi -r` before continuing (interactive; ask the user if reset is gated). Do NOT reset for compile errors.
- The instrumentation is **temporary**: guard it, keep it inert at count 0, and **restore the kernels** (git checkout / targeted revert) when done — never commit the NOP hooks.
- Verify the JIT actually recompiled per count (cache MISS), else the sweep is inert.
- `NOP_COUNT=0` baseline must reproduce the test's normal behavior; if base rate is already nonzero, the peak is the *increase* over baseline.
