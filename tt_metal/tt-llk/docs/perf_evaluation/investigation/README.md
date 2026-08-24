# Investigation: why matmul `L1_TO_L1` measurements are bistable

The main manual (`../README.md`) establishes *that* some matmul measurements land
on one of two discrete values. These notes are the hunt for *why*.

One file per question asked, each with the method, the numbers, what it rules
out, and what it does not.

| # | question | verdict |
|---|---|---|
| [01](01-not-a-single-thread.md) | Does the jump come from one thread's own work? | **No.** A median of 0.2% of the pipeline jump is explained by any single thread |
| [02](02-counters-cannot-see-it.md) | Can hardware counters show which stall it is? | **No.** The counter build does not reproduce the effect at all |
| [03](03-per-thread-zones.md) | Inside the real pipeline, is the time lost in a thread or between threads? | **Unanswerable with these zones.** All three span the whole loop |
| [04](04-repetition-does-not-help.md) | Can repeated runs average the instability away? | **No.** Median-of-5 still fails every gate run |

## Where this stands

**Established, and not in doubt:**

- The effect is real, discrete, bounded, and confined to matmul.
- It is not the configuration, not a fixed set of tests, not a per-run state, not
  core placement, not execution order, not concurrency, not the build state.
  (`../README.md` §8.5–8.8.)
- It is not any single thread's own work. (01)
- It vanishes when the profiling instrumentation changes, while the total work
  stays the same. (02)

**The strongest inference available:** a race with a window narrow enough that a
few instructions of profiling code close it. A fixed data-dependent cost would
survive that change; this does not.

**Not established:** the mechanism. Which hardware interaction resolves two ways,
and why. That is a silicon-level question.

## The correction, now made

§8.9 of `../README.md` used to conclude that *"the packer's completion time is
bistable"*. That has been **withdrawn**. The per-configuration join in 01 shows
`PACK_ISOLATE`'s 1,457 flagged configurations and `L1_TO_L1`'s 42 are disjoint,
which is what independence predicts — plausibly two separate phenomena, and the
gate question concerns the second.

What replaces it is weaker and better supported: when an affected measurement
lands slow, the whole pipeline stalls together, no thread's isolated work varies,
and the instruments available cannot localise it further.

## What is left to try

Every instrument available has now been tried.

| approach | outcome |
|---|---|
| Isolate run types (01) | Each thread is stable alone |
| Hardware counters (02) | **Ruled out.** The effect is absent from that build |
| Per-thread zones (03) | **Ruled out.** The zones span the whole loop, so they cannot localise |
| Repetition — median, min (04) | **Ruled out.** Cannot suppress it |
| A zone around a single handshake | Kernel change, in the exact region under suspicion. 02 suggests it may remove the effect |
| RTL simulation | ttsim is functional, so it cannot see timing; the VCS flow is Quasar-only. RTL is deterministic and would not reproduce a probabilistic race in any case |
| Hand it to the packer path owner | The realistic route to a root cause |

Note also `marko/dvalid-vs-semaphore-perf`, which is measuring dvalid against
semaphore synchronisation — the same handshake this investigation keeps landing
on.

## What this does block

Matmul cannot be gated. 04 shows repetition does not help, and no threshold
absorbs a 2-6% bistable jump. Matmul is 88% of the `L1_TO_L1` sweep, so a gate
that excludes it watches 7,890 of 100,971 measurements. That is the honest cost,
and it is why this is a blocking bug rather than a footnote.

## What this does not block

The threshold itself does not depend on any of it. `2%` on `TILE_LOOP` and
`KERNEL`, excluding matmul, has zero false failures on 71,152 Blackhole and 7,890
non-matmul Wormhole measurements. See `../README.md` §10.
