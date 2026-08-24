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
| [05](05-the-cost-is-per-tile.md) | Is the extra time an event, or a rate? | **A rate.** About two extra cycles per tile, sustained |

## Where this stands

**Established, and not in doubt:**

- The effect is real, discrete, bounded, and confined to matmul.
- It is not the configuration, not a fixed set of tests, not a per-run state, not
  core placement, not execution order, not concurrency, not the build state.
  (`../README.md` §8.5–8.8.)
- It is not any single thread's own work. (01)
- It vanishes when the profiling instrumentation changes, while the total work
  stays the same. (02)

- The cost is **a rate, not an event** — roughly two extra cycles per tile,
  sustained across the whole loop, agreeing across two tests that differ 16-fold
  in loop factor and 20-fold in total cycles. (05)

**The strongest inference available:** the pipeline has two stable steady
rhythms differing by about two cycles per tile, and which one it settles into is
decided by something with a very narrow timing window — narrow enough that a few
instructions of profiling code change the answer.

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
| **Loop-factor sweep (05)** | **The open lead.** `.claude/scripts/perf_loop_factor_sweep.sh` |
| RTL simulation | Was closed: a 200,000-cycle kernel is hours to days, and RTL is deterministic. **05 may reopen it** -- if the cost is per tile, a ~1,900-cycle reproducer is about a minute of simulation, and at that size you read the handshake directly rather than hoping to catch a race |
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
