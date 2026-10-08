# r03-b03-a03 result: 1.3023 (ok)

## What happened vs expected
All shapes are valid. PCC is 0.9999985, and max_abs is 0.0223-0.0245, bit-for-bit the parent's values (the same data, written
on different VCs). Per shape, parent r03-b03-a02 -> this node (µs, chip mean): h3584 12.90 -> 12.70 (-1.5%), h4096
14.26 -> 14.57 (+2.2%), h6144 17.83 -> 18.25 (+2.4%), h7168 19.51 -> 19.51 (0.0%). The score is 1.3023 vs 1.3120 (-0.7%), inside
the ±1% noise band. I expected the drain to speed up by 0.8-1.7 µs/shape. **It did not change.**

## Why (profiler evidence)
`drain.py` (in this dir: `python3 drain.py <report>/reports/<ts>/profile_log_device.csv`) gives medians over the
measured and warmup calls x 4 chips. Times are µs from each worker's W_DRAIN start. Parent -> this node:

| shape | W_DRAIN dur | first POST unpack | pack POST end | drain end - pack end |
|---|---|---|---|---|
| h3584 (28 t) | 3.633 -> 3.589 | 0.570 -> 0.563 | 2.555 -> 2.547 | 1.077 -> 1.041 |
| h4096 (32 t) | 4.061 -> 4.063 | 0.563 -> 0.557 | 2.812 -> 2.806 | 1.249 -> 1.257 |
| h6144 (48 t) | 5.778 -> 5.705 | 0.587 -> 0.605 | 3.870 -> 3.889 | 1.908 -> 1.816 |
| h7168 (56 t) | 6.263 -> 6.192 | 0.645 -> 0.643 | 4.447 -> 4.446 | 1.816 -> 1.746 |

- The per-tile drain rate is the same within 0-1.5% (~100-109 ns/tile after the first POST tile). It still trails
  the pack (~64-68 ns/tile) by 1.0-1.8 µs at the end of the row. The per-shape kernel-time moves (+2% on h4096/h6144,
  -1.5% on h3584) are larger than the drain change and go in both directions, so they are AG / cross-chip variance, not this change.
- So **single-VC serialization is not what limits a core's write injection.** With 4 VCs available, consecutive 2 KB
  packets to different banks still go out at the same ~14 B/cycle. The ~100 cycles per tile that the BRISC spends
  polling `NOC_CMD_CTRL` (JIT disassembly: ~50 instructions/tile of issue code, ~147 cycles/tile total) are therefore spent
  on something shared by all VCs of the NIU. Candidates: the NIU's L1 read port for the write payload, the injection
  port into the router (one flit/cycle shared by all VCs, then backpressure from the path), or per-NIU write
  bookkeeping. This run can't tell them apart.
- This is consistent with earlier facts: the rate doesn't depend on bank phasing (h3584 is quad-phased, the others are
  lockstepped) or on how many cores write (r03-b04-a02 waves).

## Classification
Neutral (within noise). The premise was wrong: per-VC serialization isn't the drain limit. The change is harmless (the
same bytes and the same flush semantics) but is not worth keeping. A child can revert it or keep it; the measured
difference is nil.

## What a child of this node should try next
1. **Measure where a drain tile's ~147 cycles go before trying another drain mechanism.** In the writer, accumulate
   two cycle counters across the drain: one around `cb_output.wait_front` (compute-gated) and one around the
   cmd-buf-ready spin (put a local copy of the issue with the poll timed via `read_wall_clock()`). Dump both
   via a DeviceZoneScopedN-free path, e.g. write them into an unused L1 scratch word or a profiler `DeviceTimestampedData`.
   If the spin dominates, the limit is in the NIU. Then test:
   - a second **command buffer** on the same NoC (BRISC's dynamic-NoC buf 1 is idle during the drain; restore its
     RET_ADDR_COORDINATE to the local xy afterwards, because reads rely on it). That tests a per-cmd-buf limit.
   - more **NoC0 share** (today ~25% of tiles), to test a per-NIU limit. r02-b02-a01's path model says which
     destinations are cheap on NoC0.
2. Don't retry VC tricks for the drain (this node), bank de-phasing for the drain (r01-b03-a03), or waves
   (r03-b04-a02).
3. Other levers on this lineage: the 0.38 µs combine gap (siblings), the 0.6 µs go -> stick-in-CB window
   (r03-b02-a02 #1: forwarder pushes the gathered sticks), and a lower PRE x*x fidelity (r03-b03-a02 #1).
