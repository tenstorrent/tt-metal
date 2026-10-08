# r05-b04-a01 result: 1.1939 (ok)

## What happened vs expected
Valid on every shape. PCC is 0.9999985 and max_abs is 0.0205-0.0240, the parent's values. So the half-row slices,
the sibling-stick mapping (8 partials per row), the 1/H scaling, the two-round forwarder geometry and the two-phase
compute order are all correct, and nothing hung across 13 calls x 4 shapes. But the node is much slower. Chip mean in
µs, parent r04-b04-a02 -> this node:

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 11.96 | 15.72 | +3.76 (+31%) |
| h4096 | 13.62 | 16.66 | +3.04 (+22%) |
| h6144 | 16.91 | 17.96 | +1.05 (+6%) |
| h7168 | 17.93 | 19.83 | +1.90 (+11%) |

Score 1.1939 vs 1.3997. I expected -1 to -2.5 µs. Narrow shapes lost most, which is the signature of a fixed
per-round cost: a second AG round added ~3-4 µs, and only half a row of drain came back.

## Why (profiler evidence)
Scripts are in `analysis/`; run each as `python3 X.py $DREAM_HOME/rmsnorm-prefill/reports/r05-b04-a01 [dev]`.
- `tl2.py`: every zone instance per worker core, so round 0 and round 1 are separate. Output: `tl2_out_dev3.txt`.
- `fwd.py`: forwarder F_COLLECT/F_FABRIC per round. Output: `fwd_out.txt`.
- `tl.py`: single-instance timeline used for the parent in the proposal.

Values are medians, in µs from the call's first kernel start.

**h3584, dev3 (the latest-launched chip):**

| step | round 0 | round 1 |
|---|---|---|
| stick push end | 3.87 | 8.72 |
| forwarder collect end -> fabric end (`fwd.py`) | 4.21 -> 6.83 | 9.05 -> 11.37 |
| worker go (W_AGWAIT end) | 7.19 | 11.73 |
| gathered sticks in CB (C_COMB start) | 8.45 | 12.96 |
| POST | 9.02 -> 9.78 | 13.52 -> 14.28 |
| drain end | 10.80 | 15.30 |

The two rounds serialize completely, and each round's chain is ~4.6 µs:

1. **The fabric AG itself is ~2.3-2.6 µs per round, even when the chips are already in sync.** Round 1's F_FABRIC
   (send -> out_ready) is 2.3 µs on dev3 h3584, and 2.2-3.3 µs across devices and shapes. That is the same as round 0,
   which also absorbs launch skew. So the ~2 µs "fabric latency" floor is real mechanism latency: a 3-hop line
   multicast plus out_ready, not skew. My model assumed ~2 µs including skew and almost nothing for a synced round.
2. **The release fan-out costs 0.67 µs per round.** That is F_FABRIC end -> next F_COLLECT start: 20 serial go incs
   plus the atomic barrier.
3. **The gathered-stick read doubled: 1.26 µs for 16 x 64 B reads, vs 0.6 µs for 8.** It costs ~75 ns per read,
   so the reads behave as serialized round trips, not pipelined.
4. **The stock forwarder forces push(r+1) after go(r).** Its arrival count is cumulative, and it only starts round r+1's
   collect after round r's out_ready wait + go incs. In my writer, push1 also comes after the round-0 stick read. So
   round 1's AG could only start at go0 + 1.3 µs, even though compute had stat1 ready at ~4.4 µs (PRE1 tracks the read).

So the second round added collect wait + fabric + fan-out + stick read, ~4.6 µs, to the critical path. It saved
half a drain: 1.6 µs at h3584 and 2.6 µs at h7168. Net at h7168 is +1.9 µs.

Wide shapes lose less for two reasons: the saved half-drain is larger, and the compute two-phase order (all PRE, then
all x*gamma) hides fine. C_POST is 0.8 µs per 14-tile half at h3584 and 1.7 µs per 28-tile half at h7168, so compute
was never the limit. The drain still trails POST by ~1 µs per round.

## Classification
flawed idea in this form (stock forwarder). The overlap premise fails because the per-round AG fixed cost (~4.6 µs
with serialized rounds, ~2.3 µs fabric alone) is larger than the work it hides (half a drain, 1.6-2.6 µs). The
plumbing is correct and reusable: `pick_col_splits` shared by `compute_sizing` and `create_at`, the half-row reader
stream, the writer's sibling-stick gather, and the two-phase compute block.

## What a child of this node should try next
1. **Don't retry multi-round pipelining with the stock forwarder.** If you retry it at all, the rounds' AGs must
   overlap:
   - Fork the forwarder (as r03-b04-a02 / r02-b02-a04 did) with 16-bit per-round arrival and out_ready fields.
   - Send round r+1's packet as soon as its arrivals complete, without waiting for round r's out_ready.
   - Let the writer push stat1 as soon as compute has it (~read end + 0.3 µs) instead of after go0.

   Best case at h3584: go1 ~= push1 (~4.4) + 2.5 + 0.7 ~= 7.6 µs, about where today's single go lands. Only then can
   the half-row tail (POST 0.8 + drain 1.6) beat the parent's full-row tail. Even then the gain is bounded by
   ~1-1.5 µs per shape, and every extra round costs one more stick read and one more fan-out.
2. **The ~2.3 µs fabric round is the real floor of the AG.** It is the same in a synced round. The forwarder's F_FABRIC
   has two candidate shrinks:
   - flush-then-inc instead of `async_write_barrier` + `async_atomic_barrier` before the go release (r04-b03-a01 #2);
   - the 0.67 µs serial go fan-out (multicast, or flush-then-inc go).

   Both apply to the parent's single round. They are cheaper and safer than more rounds.
3. **The gathered-stick read costs ~75 ns per 64 B read** (0.6 µs for the parent's 8 reads). It sits on every node's
   post-go critical path. Try reading each device's 128 B stick with one read into a contiguous scratch, then let
   compute unpack it from a layout where face_00/face_01 row 0 are adjacent. Or issue the reads on both NoCs. Either
   could take ~0.3 µs off every shape on the parent.
4. Keep `analysis/tl2.py` (multi-instance zones) and `fwd.py` (per-round forwarder zones) for any multi-round work.
