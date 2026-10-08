# r03-b01-a02 result: 1.3220 (ok)

## What happened vs expected
Valid on every shape. PCC 0.9999985 and max_abs 0.0204-0.0240 are bit-identical in kind to the parent (same data, it
only lands in a different memory). The L1-interleaved scratch works end to end: allocation from
`create_stats_buffer`, fabric fused write + atomic into a remote Tensix L1 bank, worker read through the same
accessor. No hang, no CB/L1 clash.

Per shape, parent r03-b01-a01 -> this node (µs, chip mean):

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 12.97 | 12.66 | -0.31 (-2.4%) |
| h4096 | 14.24 | 14.09 | -0.15 (-1.1%) |
| h6144 | 17.73 | 17.92 | +0.19 (+1.1%) |
| h7168 | 19.48 | 19.41 | -0.07 (-0.4%) |

Geomean 1.3220 vs 1.3129 (+0.7%), inside the ±1% noise band, and below the campaign best r03-b02-a01 (1.3231).
I expected ~0.3-0.5 µs per shape. The measured window shrank by only ~0.09 µs.

## Why (profiler evidence)
`stick.py` (in this dir; `python3 stick.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`): medians over measured calls
and all 4 chips. `stick` = W_AGWAIT end -> W_DRAIN start per worker (CB reserve, 8 x 64 B stick reads, barrier,
push); `ffab` = forwarder F_FABRIC duration.

| shape | stick median parent -> this | stick max parent -> this | ffab parent -> this |
|---|---|---|---|
| h3584 | 0.689 -> 0.601 | 0.724 -> 0.649 | 2.62 -> 2.36 |
| h4096 | 0.673 -> 0.567 | 0.707 -> 0.621 | 2.35 -> 2.38 |
| h6144 | 0.687 -> 0.601 | 0.713 -> 0.653 | 2.37 -> 2.37 |
| h7168 | 0.691 -> 0.599 | 0.719 -> 0.650 | 2.95 -> 2.81 |

- **The DRAM part of the stick read was only ~0.09 µs.** The window is still ~0.6 µs with the sticks in L1. So the
  ~0.67 µs that r02-b02-a04 attributed to "the DRAM round trip" is mostly something else: 8 serially issued NoC reads
  on BRISC's NoC1 (in DM_DYNAMIC_NOC mode each issue updates L1 counters), the NoC1 path from the worker to the bank
  core, the read barrier, the CB push, and the profiler zone overhead around W_AGWAIT end / W_DRAIN start. A pure
  L1-to-L1 64 B read round trip is ~0.2 µs, so the issue/sync side is ~0.4 µs.
- **F_FABRIC did not change systematically** (two shapes down 0.14-0.26, two flat). The flush=true fabric write
  commits about as fast to a DRAM NIU as to an L1 bank; the AG is fabric latency, not destination commit.
- The per-shape spread (-2.4% .. +1.1%) is the usual call-to-call / chip skew noise; the -0.09 µs fixed saving
  can't be separated from it.

## Classification
neutral (within noise). The mechanism is correct and harmless (and removes 4 KB of DRAM traffic per call), but the
premise was wrong: the post-go stick read is not DRAM-latency bound. Not worth a retry by itself.

## What a child of this node should try next
1. **Port r03-b02-a01's combine (fused add_rsqrt on row 0 + POST unpack init hoisted before the 1/rms wait)** onto
   this node. It beat this lineage's 3-pass row-0 SFPU by ~0.8%; that and this node's L1 scratch are orthogonal.
2. **If the ~0.6 µs go -> combine window is attacked again, measure inside it first** (zone around the 8 reads +
   barrier only, separate from reserve/push). Candidates that remove issue/sync cost, not latency:
   - with the scratch in L1, read the 4 sticks with 4 x 128 B reads into a contiguous staging CB page and let
     compute's unpack read the face rows from there (needs compute-side layout change), or
   - have the forwarder (it already holds the sticks' arrival) write each worker's 4 x 128 B gathered sticks into the
     worker's gathered CB directly before the go inc (forwarder fork from r02-b02-a04), so the worker does no reads
     at all after go. Watch the forwarder's serial cost: 20 workers x 8 writes.
3. Larger levers are unchanged: the drain tail after POST (1.1-1.9 µs; per-core NoC0 share tuning, r02-b02-a01 #1),
   the AG section (~2.4-2.9 µs F_FABRIC), and the column split on this lineage.
