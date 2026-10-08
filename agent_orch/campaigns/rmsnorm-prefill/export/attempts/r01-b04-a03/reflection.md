# r01-b04-a03 result: 1.2075 (ok)

## What happened vs expected
Valid on all shapes. Accuracy is identical to the lineage (PCC 0.9999985, max_abs 0.022-0.024). Per shape: h3584
1.175, h4096 1.196, h6144 1.226, h7168 1.234. That is a new campaign best (previous best r01-b01-a01, 1.109), +44% over
the parent (0.840) and +12% over the grandparent r01-b04-a01 (1.076). It landed above the expected 1.12-1.18. The wide
shapes gained the most, because they were the ones where late gamma and the latency-bound read hurt most.

## Why (profiler evidence, device 1, µs from TRISC kernel start; grandparent r01-b04-a01 in brackets)
Script: /tmp/r01b04a03/zones.py (per run host id, zone start/end, min-max over the 20 workers).

| zone end | h3584 (call idx 10) | h7168 (call idx 58) |
|---|---|---|
| R_INPUT (trid-pipelined, input only) | 2.6-3.7 (3.8-4.9) | 5.9-6.9 (7.4-8.8) |
| W_GAMMA (new, BRISC) | 3.2-4.2 | 5.6-6.5 |
| NCRISC kernel end | 2.7-3.7 (7.7-8.7) | 5.9-7.0 (15.3-16.7) |
| W_PUSH (PRE + stick) | 4.0-5.5 (4.9-6.0) | 7.2-8.8 (8.5-10.1) |
| AG wait end | 8.3-8.6 (9.0-9.3) | 11.2-11.6 (13.9-14.2) |
| TRISC end | 12.1-12.5 (12.8-13.2) | 16.8-17.2 (20.5-21.9) |
| W_DRAIN end | 12.9-14.6 (13.7-15.1) | 17.8-21.0 (22.1-25.2) |

1. **The trid-pipelined input read works once nothing else is interleaved on NCRISC.** It is ~1.2-1.9 µs faster
   at h7168 and ~1.2 µs faster at h3584, matching r01-b02-a02's gain on the root lineage. This node is the first
   isolated measurement of the parent's `read_input_pass_pipelined` (lookahead 4 blocks).
2. **Gamma on BRISC lands in parallel with the input.** The 2 x 56 face-row reads take ~4.5 µs on BRISC at h7168
   (~40 ns each, still slow: issue-bound or same-page DRAM hot spot; the per-worker start-page rotation did not make
   them cheap). That doesn't matter because BRISC is otherwise idle. Gamma is resident ~at the input end, so the
   x*gamma pre-pass hides fully under the AG wait. Post-AG compute (AG end -> TRISC end) is 5.6 µs at h7168, vs
   6.6-7.7 µs in r01-b04-a01 where x*gamma spilled past the AG.
3. W_PUSH starts right as W_GAMMA ends (BRISC is serial). On these shapes PRE finishes after gamma, so the stick
   push is not delayed. But gamma finishes only ~0.3-0.5 µs before PRE at h3584. Any faster PRE would expose it.
4. What remains (h7168): AG section ~3-4 µs (F_COLLECT waits for the slowest worker's PRE, then F_FABRIC ~2.3 µs),
   post-AG single pass ~5.6 µs (~100 ns/tile), and a drain tail 1-3.8 µs after compute. The drain spread across
   workers (W_DRAIN end 17.8-21.0) is now the largest per-core variance.

## Classification
win (+20.8% geomean vs baseline, every shape far outside the ±1% noise). It repairs the parent: the bug was gamma
reads interleaved on the NCRISC input queue. The fix moved them to the idle BRISC.

## What a child of this node should try next
1. **Port this lineage onto the column split (r01-b03-a02, 1.094: k=4, 80 workers, one kernel group).** The pieces
   are orthogonal. The split cuts read, PRE, POST and drain per core 4x, and this node's x*gamma pre-pass plus BRISC
   gamma removes a POST pass and keeps gamma off the input path. With 7-14 tiles/core the gamma read on BRISC is only
   14-28 reads. Watch the leader-combine hop and the drain bank-phase issue at h4096 that b03-a02 found.
2. **Shrink the drain tail.** Compute finishes 1-3.8 µs before the last W_DRAIN. NCRISC is now idle from ~6-7 µs
   (h7168) to the end, so let the reader drain half of each row's output tiles on its NoC (output_cb is resident, and
   compute pushes per block: e.g. even blocks on BRISC, odd blocks on NCRISC, each popping cumulatively). Or rotate
   each worker's write start column (by tile_row) to de-phase DRAM banks.
3. **Cheaper PRE.** PRE (x*x HiFi4 + per-tile L1-acc pack) ends 1.3-1.9 µs after the input lands at h7168, and the AG
   start waits on the slowest worker's PRE. Accumulate x^2 in DST across a block and pack once per block.
   If PRE gets faster, make the gamma read cheaper too (it would become the gate on the stick push). Options:
   precompute the NoC address once per page and issue both face-row reads from it, or barrier gamma only right before
   the first use instead of before W_PUSH (use a trid or move the barrier after the stick push).
