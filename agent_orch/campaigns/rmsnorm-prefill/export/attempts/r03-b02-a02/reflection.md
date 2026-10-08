# r03-b02-a02 result: 1.3333 (ok)

## What happened vs expected
Valid on all shapes. PCC 0.9999985 and max_abs 0.0204-0.0240 are the parent's values, so the sticks land where they
should in the L1-interleaved scratch on every chip (the bank->core map is uniform across the mesh). Per shape, parent
r03-b02-a01 -> this node (µs, chip mean): h3584 12.72 -> 12.59 (-1.0%), h4096 14.13 -> 14.05 (-0.6%), h6144 17.72 ->
17.55 (-1.0%), h7168 19.41 -> 19.32 (-0.5%). Score 1.3333 vs 1.3231 (+0.8%). That is inside the ±1% noise band, so it
is NOT a measured improvement, even though every shape moved the right way. I expected -0.3..-0.6 µs per shape and got
~-0.1.

## Why (profiler evidence)
`ag.py` (in this dir; `python3 ag.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`; both outputs in `ag_out.txt`).
Medians over the measured calls on all 4 chips, µs:

| shape | go -> W_DRAIN start (med) | go -> C_COMB start, TRISC_0 (med) | F_FABRIC end -> C_POST start (max) | F_FABRIC dur, min over calls |
|---|---|---|---|---|
| h3584 | 0.688 -> 0.601 | 0.713 -> 0.625 | 1.769 -> 1.644 | 0.74 -> 1.05 |
| h4096 | 0.673 -> 0.568 | 0.696 -> 0.593 | 1.743 -> 1.606 | 1.26 -> 1.33 |
| h6144 | 0.687 -> 0.600 | 0.708 -> 0.625 | 1.766 -> 1.645 | 1.17 -> 0.73 |
| h7168 | 0.689 -> 0.599 | 0.714 -> 0.623 | 1.763 -> 1.643 | 0.96 -> 1.02 |

1. **The mechanism engaged, and the stick read did get faster, but only by ~0.09 µs.** go -> sticks in the gathered
   CB fell from ~0.69 to ~0.60 µs on every shape, and the first POST unpack starts ~0.12 µs earlier relative to the
   forwarder's F_FABRIC end. So DRAM vs L1 latency was only ~120 cycles of the ~0.69 µs. The other ~0.6 µs (~810
   cycles) is NOT the read's memory latency. It is the go-sem poll exit, the CB reserve, 8 TensorAccessor-addressed
   64 B read issues to 4 different remote cores, the read barrier, the push, and compute's CB wait/unpack start.
   r02-b02-a04's address pre-staging didn't help either, so the dominant part is probably the NoC round trip of 8
   serial small reads plus the cross-RISC CB handoff, not address math.
2. **The AG itself did not get faster.** F_FABRIC duration (both its median, which is cross-chip skew, and the min
   over calls, the last-arriving chip's real latency) is unchanged within noise, and go - F_FABRIC end is the same
   0.165 / 0.52 µs. The landing write's DRAM-vs-L1 ack time is not visible in the fabric path. The fabric hops and
   the EDM forwarding dominate.
3. The ~0.1 µs that POST gains moves the kernel end on h3584/h6144 by about that, and less on h4096/h7168. That
   fits the drain being throughput-bound from its start (r03-b03-a01).

## Classification
neutral (within noise: +0.8%, every shape -0.5..-1.0%). The diagnosis "DRAM round trip is the 0.67 µs" (r02-b02-a04
#3) was wrong. Only ~0.09 µs of it was DRAM latency. The L1 scratch is correct and free (2 x 4352 B per L1 bank), so
keep it as the base. It also makes the next step below possible.

## What a child of this node should try next
1. **Remove the worker-side gathered-stick read entirely.** The scratch is now in L1, so the forwarder can push the
   data instead of each worker pulling it:
   - after out_ready, the forwarder (fork it into the op dir like r02-b02-a04 did) multicasts the 4 pages (4 x
     pc*128 B = 10 KB) into a grid-uniform L1 region on the worker rows (2 rectangles, NoC write + mcast). The
     go inc then rides behind the data on the same NoC.
   - each worker copies its 8 x 64 B face-rows locally (L1 -> L1 on-core, or have compute unpack a tile built in
     place).
   - Or let each worker's OWN-device stick come from its local stats_transposed_local copy, so only ring-1 remote
     reads are needed. That saves 2 of the 8 reads, so it is small.
   Measure first: put zones around the 8 reads + barrier in W_AGWAIT..W_DRAIN, to split the ~0.6 µs into issue,
   round trip, and CB handoff.
2. Unchanged larger levers: the drain is the post-AG tail (drain end ~4.8-7.5 µs after F_FABRIC end; per-core NoC0
   share tuning, r02-b02-a01 #1). The remaining combine gap is 0.38 µs (siblings are on it). The AG fabric section's
   cross-chip skew (F_FABRIC median 2.4-2.9 µs vs ~1 µs on the last chip) is the largest fixed cost, but most of it is
   launch skew between chips.
3. Don't spend more on DRAM-vs-L1 placement of small CCL scratch. The memory latency difference is ~0.1 µs.
