# Investigation: why matmul `L1_TO_L1` measurements are bistable

The main manual (`../README.md`) establishes *that* some matmul measurements land
on one of two discrete values. These notes are the hunt for *why*.

One file per question asked, each with the method, the numbers, what it rules
out, and what it does not.

| # | question | verdict |
|---|---|---|
| [01](01-not-a-single-thread.md) | Does the jump come from one thread's own work? | **No.** A median of 0.2% of the pipeline jump is explained by any single thread |
| [02](02-counters-cannot-see-it.md) | Can hardware counters show which stall it is? | **No.** The counter build does not reproduce the effect at all |
| [03](03-per-thread-zones.md) | Inside the real pipeline, is the time lost in a thread or between threads? | in progress |

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

## A correction pending against the main manual

§8.9 and §8.10 of `../README.md` conclude that *"the packer's completion time is
bistable"*, reasoning from the population fact that `PACK_ISOLATE` and `L1_TO_L1`
both show flags while `MATH_ISOLATE` and `UNPACK_ISOLATE` show none.

**That inference does not survive the per-configuration join in 01.**
`PACK_ISOLATE` has 1,457 flagged configurations and `L1_TO_L1` has 42, and they
are not the same configurations. There are plausibly two separate phenomena, and
the gate question concerns the second one.

Those sections are left unrevised until 03 lands, so that the correction is made
once and correctly.

## What is left to try

| approach | cost | prospect |
|---|---|---|
| Per-thread zones inside the real pipeline (03) | ~7 min | The last cheap decisive step |
| Hardware counters | — | **Ruled out** by 02 |
| Waveform capture | high, and the tooling is Quasar-only | Would answer it, if it existed for Wormhole |
| Hand it to the packer path owner | — | The realistic route to a root cause |

Note also `marko/dvalid-vs-semaphore-perf`, which is measuring dvalid against
semaphore synchronisation — the same handshake this investigation keeps landing
on.

## What this does not block

The gate threshold does not depend on any of it. `2%` on `TILE_LOOP` and
`KERNEL`, excluding matmul, has zero false failures on 71,152 Blackhole and 7,890
non-matmul Wormhole measurements. See `../README.md` §10.
