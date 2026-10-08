# r04-b02-a01 result: 1.2988 (ok)

## What happened vs expected
Valid on every shape. PCC is 0.9999985 and max_abs is 0.0204-0.0240, identical to the parent. So these all work:
- the forwarder-core-sharded scratch;
- the slot tile-row-0 layout;
- the 64 B-page gathered CB;
- compute's offset tile indexing.

It is slower than the parent r03-b02-a02 (µs, chip mean):

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 12.59 | 12.85 | +2.0% |
| h4096 | 14.05 | 14.52 | +3.3% |
| h6144 | 17.55 | 18.43 | +5.0% |
| h7168 | 19.32 | 19.38 | +0.3% |

Score is 1.2988 vs 1.3333 (-2.6%), outside the ±1% noise band. I expected -0.4..-0.7 µs per shape and got +0.06..+0.9.
(The first eval attempt was a JIT compile error: I had appended the new compute CT args to the reader's arg vector,
which ends the same way. That was fixed before the only device run.)

## Why (profiler evidence)
`ag.py` in this dir is the parent's script plus an F_MCAST zone. Run it as `python3 ag.py $DREAM_HOME/rmsnorm-prefill/reports/r04-b02-a01`;
the output is in `ag_out.txt`. Values are medians over measured calls and all 4 chips, in µs. Parent (r03-b02-a02/ag_out.txt) -> this node, on every shape:

| | parent | this node |
|---|---|---|
| F_MCAST (3 remote pages x 3328 B + go flag, multicast to 22 cores, + ack barrier) | — | **1.21** |
| go arrival after F_FABRIC end, first / last worker | 0.17 / 0.52 | **1.03 / 1.15** |
| go -> W_DRAIN start (stick in CB) | 0.60 | **0.04** |
| go -> C_COMB start (TRISC_0) | 0.62 | **0.065** |
| F_FABRIC end -> C_POST start, max over workers | 1.64 | 1.74 |

1. **The worker side of the mechanism did what was planned.** From go to the compute combine now takes 0.065 µs
   instead of 0.62 µs. No reads, and compute unpacks the multicast pages in place. The go-order spread fell from
   0.35 µs to 0.11 µs.
2. **The forwarder multicast costs more than it removes.** Moving ~10 KB of remote pages, plus a 4 B flag, from the
   forwarder to the 2-row rectangle takes ~1.0 µs before the flag lands. The barrier adds another ~0.2 µs. That is
   ~10 GB/s, far below a NoC link (64 B/cycle, ~86 GB/s).
   - Candidate causes, which this run can't separate:
     - The loopback-src multicast on NoC1 spans the non-Tensix x=8/9 columns, and each of the 3 data packets
       reserves the multicast path.
     - The forwarder's BRISC issues the 3 multicasts serially, each with 22-destination ack accounting.
   - The own-page multicast before the fabric wait is hidden, as intended. It lies inside F_FABRIC, which is
     unchanged at a median of 2.4-2.7 µs.
3. Net effect: the combine starts ~0.1 µs later than in the parent (1.74 vs 1.64 µs after F_FABRIC end), so the
   whole post-AG chain moves later by that amount.
   - h3584 and h7168 move by about that (+0.06..+0.26 µs).
   - h4096 and h6144 lost more (+0.5..+0.9 µs). Their drain_e - F_FABRIC end grew 0.4-0.5 µs beyond the combine
     shift. h6144 is also the shape with the host launch stagger in the parent's ops CSV (op-to-op 0.5 / 1.7 / 2.8 /
     3.3 µs on d0..d3). So part of that is call-to-call / chip skew, but the direction is consistently worse.
4. The 3328 B fabric packets, up from 2560 B, did not measurably change F_FABRIC.

## Classification
Flawed as executed. An on-chip forwarder multicast of the gathered pages, issued as one serial burst after out_ready,
is slower (~1.0-1.2 µs) than the per-worker pull it replaces (~0.6 µs plus 0.17-0.52 µs go fan-out).

The idea is still worth something: the worker side drops the 0.6 µs go -> combine window to 0.065 µs. The cost moved
to the forwarder's serial multicast, and that cost is now the thing to cut.

Reusable plumbing, all validated on HW:
- the height-sharded scratch on the forwarder core;
- the slot tile-row-0 layout (`L(s) = (s/16)*2048 + (s%16)*64`);
- the 64 B-page gathered CB with offset tile indexing in compute;
- the forked RMS forwarder with a `gather_mcast` CT switch.

## What a child of this node should try next
1. **Overlap the multicasts with the cross-chip skew instead of serialising them after the last arrival.** F_FABRIC's
   median is 2.4-2.7 µs, but the last chip's real latency is ~1 µs. So 2 of the 3 remote pages are typically in L1
   well before out_ready completes.
   - Give each source chip its own landing flag. Point the fused write+inc's semaphore address at a word in that
     page's slot of the forwarder-core shard (e.g. the last 16 B of the 4352 B page). flush=true already orders
     payload before inc.
   - The forwarder then polls the per-page flags and multicasts each page as soon as it lands.
   - Only the last page's multicast (~0.35 µs) plus the flag stays on the critical path. Expected: go at ~0.4-0.5 µs
     after the last arrival, which beats the parent's 0.17-0.52 + 0.6 µs.
   - Keep everything else from this node.
2. **Measure the multicast itself before any other variant.** Put zones around each multicast issue and the barrier.
   Try exact worker rectangles without loopback (the forwarder is outside them if its row is excluded; row-1 workers
   need a second rect), or NoC0 for the multicast. r02-b02-a04 measured ~0.1 µs per small set_multicast, so 1.2 µs
   for 3 x 3.3 KB looks anomalous and may be fixable (e.g. `linked=true` across the 3 data packets + flag, so the path
   is reserved once).
3. If the multicast can't be made fast, revert to r03-b02-a02's per-worker pull. The only thing left to win there is
   the 0.17-0.52 µs go fan-out, which is smaller than option 1.
4. Unrelated levers are unchanged: gamma streaming port (r03-b04-a03), PRE fidelity, and the drain.
