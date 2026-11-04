# The persistent KDA+MoE layer program (the "megakernel" route), design note 2026-11-04

Why: after the fabric exchanges the KDA+MoE layer is still 22 programs at 776 us on chip (0,0) (4 users per row,
32 users on the 8x4). Every program pays its own pipeline fill, drain and DRAM round trip; per-cluster fusion keeps
removing them one at a time and lands near 30 ms per step. The reference (aperezvicente-TT, 8 chips) runs the whole
layer, and the layer loop, inside one resident program at 283 us. This note sizes that design on our 8x4 hybrid.

## The floor on our layout (per KDA+MoE layer, per chip, bfp8, 32 users at 4 per row)

| weights read per chip | MB | at 400 GB/s |
|---|---|---|
| qkv projection (TP4 shard 4096 x 6144) | 26.7 | 67 us |
| out projection (2048 x 4096) | 8.9 | 22 |
| the two 288-wide KDA projections | 2.5 | 6 |
| routed experts, ~28 active eighth-slices of 72 (3 x 4096 x 256 each) | 93 | 232 |
| shared expert + router | 7.8 | 20 |
| total | ~139 | **~350 us** |

Activations stay in L1; the KDA recurrent state (2.1 MB per user) stays in DRAM and streams per user as today. So
a perfect persistent program on this layout is ~350-400 us per KDA layer and ~600 per DSA layer: 34 x 0.4 + 11 x
0.6 = ~20 ms per step = 50 tok/s per user at 32 users, 1,600 tok/s aggregate. Fewer users lower the expert bytes
(8 users: ~150 us of experts) and bfp4 would halve them, but neither is needed to reach 50 at bfp8 x 32.

## Phases of the resident program (one chip; every chip of the mesh runs the same program)

| phase | what runs | cores | activation in L1 after the phase |
|---|---|---|---|
| P1 pre-block (attn wrap) | the fused mHC pre-block compute (dest Sinkhorn) | 120 (column tile per core) | hn [32 x 4096] bf16, pre/post/comb lane tiles |
| P2 qkv + the two small projections | DRAM-streaming matmul phase (the expert stream op's reader pattern, bank-local readers) | 96 + readers | qkv [32 x 6144], gates |
| P3 KDA recurrence | the kda_decode compute as a phase (5 parts x 16 cores, the state per user from DRAM) | 80 | o [32 x 2048] |
| P4 out projection | streaming matmul phase | 22-64 | y partial [32 x 4096] |
| P5 TP all-reduce | the row_exchange all-reduce phase (owner sum + fan-out, fabric) | 1 exchange + 8 workers | y [32 x 4096] replicated |
| P6 post-mix + pre-block (ffn wrap) | the fused post-mix then the pre-block on the new streams, no DRAM hop | 120 | streams, hn |
| P7 router | router matmul + fused top-k phase | 9 + 32 | routing [32 x 288] |
| P8 row gather | the row_exchange gather phase (fabric) | 1 + 9 | x_g [32 x 4096], routing_g |
| P9 experts | the expert stream gate\|up + down kernels as a phase (bank-local weight streaming, prefetch of the next phase) | 112 + 9 | y partial [32 x 4096] |
| P10 row combine | the row_exchange reduce phase | 1 + 8 | out [4 x 4096] |
| P11 shared expert + add | streaming matmul x2 + the glu + add | 16-64 | y |
| P12 TP all-reduce | as P5 | 1 + 8 | y replicated |
| P13 post-mix | the fused post-mix into the resident streams | 120 | streams for the next layer |

The exchange phases keep their protocol (persistent receive staging in DRAM, a global arrival semaphore per phase
site, owner sums in fp32 DST); the per-call-site instance rule becomes per-phase instances. Phase hand-off on a chip:
one program semaphore per phase, counted by the producing cores, waited by the consumers (the fused expert op's
arrival counters are the template). The weight ring: a reader core per DRAM bank streams the next phase's weight
tiles into a per-core prefetch CB while the current phase computes (the expert stream op's bank-local readers are
this, restricted to one phase today).

## What we reuse and what is new

Reused as phases with their kernels: the mHC pre-block and post-mix computes, the KDA decode compute, the router
top-k, the expert stream gate|up / down kernels, the three row_exchange modes. New: the phase scheduler (a per-core
phase list and semaphore chain), the resident activation plan (which core holds which tiles between phases: the
mHC's column-tile ownership is the natural one, the KDA's head ownership and the experts' bank ownership differ,
so two re-layouts stay inside L1 via local NoC copies), a generic DRAM-streaming matmul phase for the five
projections, the per-layer argument table in DRAM (weights' addresses, the layer's constants) and the layer loop.

## Risks

* L1 budget: the expert phase uses most of L1 on 112 cores today (weight stream CBs, the act buffer); resident
  streams (2 MB fp32 per chip = 17 KB per core over 120 cores) and hn / qkv / y (each < 4 KB per core) fit, but the
  exchange staging (288 KB per instance on 10 cores) must move to DRAM or shrink to the per-phase packet size.
* Core-role conflicts: the KDA recurrence wants 5 x 16 cores with a part-0 multicast, the experts 112 + 9 next to the
  banks, the mHC 120 column cores; a resident program needs one grid plan, so some phases run on fewer cores than
  their standalone op (the KDA decode's 80 and the mHC's 120 are compatible; the experts' 121 are the whole grid).
* Debuggability: a hang inside one resident program is a tt-triage case, not a failed op; every phase keeps a
  standalone unit test and a "phase mask" argument that lets the program stop after phase k and write L1 to DRAM.
* The served contract: 32 users, the 1 M context (the KDA snapshot ring and the paged DSA cache are DRAM-resident and
  unaffected), prefill untouched (its own path), MTP later inside the loop.

## Staging, each step gated on the 45-layer chain (per-call agreement for rounding-moving phases, the prefill
chain, a served smoke) and measured on the per-layer profile

1. Two phases in one program: P1 + P2 (pre-block feeding the qkv matmul from L1). Validates the resident hand-off
   and the streaming matmul phase. ~1 day.
2. The attention half as one program: P1-P6 (8 programs, ~330 us today; target ~200). 2-3 days.
3. The MoE half as one program: P7-P13 (14 programs, ~450 us today; target ~300). 2-3 days.
4. One program per layer (join the halves), then the layer loop with the argument table (one launch per KDA stack).
   2 days. Expected step after 1-4: ~24-26 ms (38-42 tok/s per user), the remaining gap being the expert phase's
   own efficiency (215 us vs the 232 us floor is already close) and the DSA layers.
5. The DSA layer on the same scheme (the fused decode-state block already exists behind a knob). 3 days.
6. Head and sampling inside the step, then MTP. After that the 50-100 band.

About two weeks of focused kernel work in its own worktree (`wt_perf_mhc` is ready), while the per-cluster branch
keeps serving the gates and the release.

## Step 1 as built (2026-11-04 04:30, `custom_ops/moe_persistent`, knob `GLM53_MOE_PERSIST=1`)

One program per chip: the gather phase on the 10 cores the fused expert placement leaves idle, a multicast release
(a 1 written into semaphores 12-15 of every core of the grid: one per expert dataflow kernel) once the gathered
tiles are written, the fused gate|up + down kernels unchanged except for a wait at their start, the down cores
counting their written output tiles on the combine workers' semaphore 2, and the reduce phase on the same 10 cores.
Program semaphores are re-initialised by the dispatch per launch and reset by their waiters; the two exchanges keep
their persistent receive stagings and arrival semaphores. Data still crosses the gathered and expert-output
tensors.

Step 2 (next): the gathered rows multicast straight into the expert cores' L1. Every core's first CB becomes a
256 KB landing zone (the gate|up cores' resident x; a dummy on the down and exchange cores), so one multicast per
worker chunk to the whole grid rectangle lands x in every gate|up core at the same address, and the gate|up writer
skips its 256 KB read of the x tensor. The routing matrix (9 tiles) rides the same way into a second zone.

## Status 2026-11-04 05:10 (step 1 gates)

| gate | tree | verdict |
|---|---|---|
| gate84_B4_persist | wt_persist @dea92a9fd6d | construction: 8 free cores < 10 (12x10 grid; 12 groups x 8 banks + 16 down = 112 cores) |
| gate84_B4_persist_g11 | @4bb7e41c2c7 (11 groups: 88 + 16, 16 free) | construction: the persistent op read `down_xy` / groups / semaphores off the `ExpertStream` wrapper; they live on its `fused_op` |
| gate84_B4_persist_g11b | @9b4aeaf46b5 | queued behind gate84_B4_fabar_g11 (ids reference at 11 groups) and gate84_B1_fabar (8 users, today's path) |

Lessons: the Galaxy chips' compute grid is 12 x 10 (120 cores), so every worker the persistent program adds comes out of
the 8 cores the fused expert placement leaves at 12 groups -> `persistent_fused_groups` (11 groups when
`GLM53_MOE_PERSIST=1`). The step-2 tree (wt_persist2 @b986e535e83) carries both fixes and passes the landing offset
as a runtime arg (one exchange-writer variant per chunk width, not per chunk).

### Lesson (2026-11-04): grid multicasts need Tensix-only rects

The Galaxy Blackhole chips' 12 x 10 worker grid maps to NOC columns 2-7 and 10-15 (8-9 are DRAM / non-Tensix), so a
single multicast rect from logical (0, 0) to (11, 9) covers non-Tensix cores and never completes (the release reached
four cores; the exchange core spun in its write barrier). `grid_mcast_rects` derives the physically contiguous rects
from `worker_core_from_logical_core`, and the kernels loop over them (`n_rects, (x0, y0, x1, y1, num_dests,
loopback) x n` runtime args; the loopback variant only for the rect holding the sender). Every later grid-wide
multicast (the weight-prefetch ring's releases, the layer loop's phase barriers) must use the same rects.

### Finding (2026-11-04 06:56): the program's launch config is exposed

Layer-4 profile (device 0), fabric knobs vs the persistent block:

| op | three programs | persistent |
|---|---|---|
| MoE block device time | 56.0 + 215.3 + 37.9 = 309.2 us | 305.4 us |
| op-to-op gap before the block | 0 + 0 + 0 | 20 us |
| MoE all-reduce after the block | 24.4 us | 40.9 us (cross-chip skew lands here) |
| layer total (device + gaps) | 775.7 + 67 = 843 us | 786.2 + 79 = 865 us |

The block itself is 4 us faster than the three programs, but its launch config (per-core runtime args of ~115 cores,
28 kernels, 20+ CBs, 7 semaphores) is written by the dispatcher while the PREVIOUS program runs, and the previous
program here is the 12 us router top-k: ~20 us of the write is exposed. The three-program flow hid the expert
program's config behind the 56 us gather. Consequences for the design:
- an isolated persistent block only pays off when the op before it is longer than its config write (or when the
  block is replayed back-to-back with another long program);
- the full persistent layer hides its config behind the previous layer's program (~700 us): the exposure vanishes
  only at that end state, so intermediate steps must be judged by device time + skew, not by the step alone;
- the cross-chip skew of the block (the reduce waits for the slowest row) moves into the next all-reduce: a
  persistent layer must keep the rows in lock-step (the exchanges already do) and should fold the all-reduce in.
Next: step 2 (resident x: no x_g write / read), then fold the router top-k and the shared expert (two always-active
"experts" with weight 1) and the MoE all-reduce into the block so the preceding op is the 52 us MHC post-mix.
