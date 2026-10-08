# r03-b04-a03 result: 1.3318 (ok)

## What happened vs expected
Valid on every shape. PCC is 0.9999985 and max_abs is 0.0204-0.0240, the same values as r03-b04-a01: same data, same math,
only the time gamma reaches compute changed. This node is r03-b04-a01's code plus the streamed gamma (the parent's waves
are reverted), so a01 is the right comparison (µs, chip mean):

| shape | r03-b04-a01 | parent r03-b04-a02 (waves) | this node | vs a01 |
|---|---|---|---|---|
| h3584 | 12.89 | 13.76 | 12.77 | -0.12 (-0.9%, noise) |
| h4096 | 14.20 | 14.78 | 14.20 | 0.00 |
| h6144 | 17.83 | 18.78 | 17.75 | -0.08 (noise) |
| h7168 | 19.53 | 20.73 | **18.72** | **-0.81 (-4.1%)** |

Score 1.3318: +1.4% over a01 (1.3131) and +7.1% over the parent (1.2440). It ties the campaign best r03-b02-a02 (1.3333)
within noise, without that node's L1 scratch or fused add_rsqrt. h7168 at 18.72 µs is the fastest h7168 of any node
(previous best 19.32). The prediction was +0.5-1.5% geomean, with the gain on h6144/h7168 from removing a straggler.
h7168 delivered more than expected. h6144 delivered nothing measurable on the chip mean.

## Why (profiler evidence)
`analysis/late2.py` per shape x device, over 10 measured calls (outputs: `late2_out.txt`, `late2_r03-b04-a01.txt`,
`late2_r03-b02-a02.txt`). A "straggler call" is a call where one core's POST starts >0.5 µs later after go than the
median core's. "last-med drain" is the last core's drain end minus the median core's.

| | a01 dev0 h7168 | this dev0 h7168 | a01 dev0 h6144 | this dev0 h6144 | r03-b02-a02 dev0 h7168 |
|---|---|---|---|---|---|
| straggler calls | 9/10 | **0/10** | 8/10 | **0/10** | 9/10 |
| last-med drain end | 1.67 µs | 0.40 µs | 0.71 | 0.62 | 1.58 |
| max POST lag after go (all h7168 cores, chips) | 3.88 µs | 1.68 µs | | | |

- **The dev-0 straggler loop is gone.** In a01 (and in the best node r03-b02-a02), one dev-0 core, usually (2,2), launches
  ~1.4 µs late every call because it finished last in the previous call. Its gamma landed after its PRE, its 3.3 µs
  x*gamma pre-pass ran after the AG, and its POST started 2-2.5 µs after every other core's. So it finished last again
  and kept the loop going. Its late stick also gated F_COLLECT for the chip: in a01, dev-0 F_COLLECT ended at
  8.96 µs on that core's push. With gamma streamed, compute starts x*gamma as soon as PRE ends and the first chunks have
  landed. The late core's POST lag shrank enough that it no longer finishes last, and the next call's launch skew
  doesn't build up. In the sampled call (`core_dev0_h7168_call.txt`), dev 0 now waits 3.1 µs in F_FABRIC for the other
  chips: it is no longer the late chip.
- **Why every chip's h7168 got faster, not only dev 0:** the AG couples the chips, so dev 0's late stick held up the
  gather on all four. The chip-mean kernel at h7168 dropped on all devices (per-dev kernel ~19.55-19.73 -> 18.57-18.80 µs).
- **Healthy cores:** per h7168 measured core, relative to its own BRISC start (median), a01 -> this node:
  - gamma loop end: 6.87 -> 7.98. The loop now also contains the trid waits and the in-loop stick push.
  - stick push start: 6.72 -> 6.43.
  - input end: 5.64 -> 5.58.
  - POST lag after go: 1.28 -> 1.28.

  So the streamed gamma costs nothing on the healthy path. The later loop end doesn't matter because compute consumes
  chunk by chunk. The earlier push is because the push no longer waits behind the tail of the gamma loop on cores where
  PRE finished during it.
- The narrow shapes had 2-3 µs of gamma slack already and no straggler, so they are unchanged, as predicted.

## Classification
win: +1.4% geomean over the code it modifies (r03-b04-a01), h7168 -4.1%, the other shapes within noise. It removes a
cross-call straggler feedback loop, a cost no earlier node had identified. It ties the campaign best rather than beating
it, because this lineage lacks the best node's two small wins (L1 scratch, fused add_rsqrt).

## What a child of this node should try next
1. **Port this gamma streaming onto the best node r03-b02-a02.** It is the cheapest likely new best. The change is
   writer-only and confined to the W_GAMMA block, so it applies cleanly. r03-b02-a02 still has the dev-0 h7168 straggler
   in 9/10 calls (`late2_r03-b02-a02.txt`, last-med drain 1.58 µs). Expected h7168 ~18.5 µs, score ~1.35.
2. **Look for other cross-call feedback loops with `analysis/late2.py` / `start.py`.** The chip-mean metric charges
   every per-core lateness twice: once in this call's tail and once as next call's launch skew. Any phase whose slack is
   under ~1.5 µs (a typical launch skew) can lock a core into finishing last. The candidate now is the drain. The last
   core's drain still ends 0.4-0.7 µs after the median's on every chip. A per-core drain order that makes the
   historically late cores (high-x, y=3) drain first or on the short NoC would attack it (r02-b02-a01 #1).
3. The gamma loop is issue-bound at ~50 ns per 32 B read (112 reads at h7168). It is off the critical path now, but it
   becomes the gate again if PRE gets faster. The cheap fix is one TensorAccessor address per page instead of one per
   face row.
4. Keep the sticky-trid read pattern: set the trid once per chunk, use plain one-packet reads, then a trid barrier. It
   avoids the TXN_ID path's per-read set_trid and outstanding-count poll.
