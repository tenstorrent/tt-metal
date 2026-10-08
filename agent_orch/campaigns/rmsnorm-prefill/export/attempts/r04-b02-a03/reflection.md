# r04-b02-a03 result: 1.4216 (ok)

## What happened vs expected
Valid on every shape. PCC is 0.9999985 on all four. max_abs is 0.0220-0.0239, which is the HiFi2 node's range
(r04-b01-a02), against a gate of 0.05. The ack-free push into the forwarder-core sharded scratch and the merged
gamma-loop `poll_stick` both worked: no hang, no stale slot.

Score 1.4216 is the **new campaign best**: +1.6% over r04-b04-a02 (1.3997) and +7.9% over the parent (1.3175). I
predicted ~1.41-1.43.

µs, chip mean:

| shape | parent r04-b02-a02 | r04-b04-a02 (pull + 3 writer wins) | r04-b01-a02 (pull + HiFi2) | this |
|---|---|---|---|---|
| h3584 | 12.69 | 11.96 | 12.04 | **11.74** |
| h4096 | 14.29 | 13.62 | 13.51 | **13.35** |
| h6144 | 18.02 | 16.91 | 17.13 | **16.78** |
| h7168 | 19.26 | 17.93 | 18.17 | **17.65** |

## Why (profiler evidence)
Scripts are in `analysis/`: the parent's `tail.py` and `ag.py`. Run each as
`python3 X.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`. Outputs: `tail_out.txt`, `ag_out.txt`, and
`tail_<node>_out.txt` for r04-b04-a02, r04-b01-a02 and the parent. Values are medians over measured calls x chips,
in µs.

**The release side is unchanged from the parent:**
- F_MCAST: 0.84.
- Go after the last arrival: 0.65 / 0.77 (first / last worker).
- go -> C_COMB: 0.065.
- C_POST after the last arrival, max over workers: 1.35-1.37.
- In 157 of 160 chip-calls only one page was left at the last arrival.

**The post-AG tail is the question this node was meant to answer.** Compare the median drain end after the last
arrival (F_FABRIC end):

| shape | parent (mcast, non-posted) | r04-b04-a02 (pull, posted) | this (mcast, posted) |
|---|---|---|---|
| h3584 | 4.40 | 4.26 | 4.20 |
| h4096 | 5.11 | 4.80 | 4.93 |
| h6144 | 6.72 | 6.29 | 6.47 |
| h7168 | 7.03 | 6.65 | 6.67 |

And the per-core time from POST start to drain end (median):

| shape | r04-b04-a02 (pull) | this (mcast) |
|---|---|---|
| h3584 | 2.78 | 2.91 |
| h4096 | 3.31 | 3.61 |
| h6144 | 4.78 | 5.20 |
| h7168 | 5.16 | 5.33 |

- **The posted drain did not rescue the multicast release.** POST still starts ~0.3 µs earlier than on the pull
  path. But the 20 drains start within 0.11 µs of each other (the pull path spreads them over 0.30 µs), so each core
  drains 0.13-0.42 µs slower. The drain end after the last arrival is the same as on the pull path, within
  ±0.15 µs per shape.
- The parent's lesson holds even with posted writes: release staggering helps the contention-bound drain about as
  much as an early synchronized go saves.
- **So most of this node's gain over r04-b04-a02 is HiFi2.** HiFi2 alone was worth -0.4..-0.6 µs (r04-b01-a02).
  Here, against r04-b04-a02, the gain is -0.22 / -0.27 / -0.13 / -0.28 µs.
- My best estimate is that the multicast release is neutral to slightly negative against a pull-path "r04-b04-a02 +
  HiFi2" stack. That stack has not been measured yet. If a sibling measures it, compare chip means directly. The
  difference is the release mechanism alone.

## Classification
win (new best 1.4216, +1.6% over the previous best, every shape faster). The gain comes from stacking HiFi2 with the
three writer wins. The multicast AG release itself is neutral: an earlier synchronized POST is cancelled by slower
contended drains, now confirmed with posted writes too.

## What a child of this node should try next
1. **Stagger the drain starts deliberately, but keep the early go.** This is the only way this lineage's ~0.3 µs
   earlier POST pays off. Options:
   - Worker slot s delays its first output write by ~s*15 ns.
   - Better: rotate each core's first output bank so that concurrent cores don't target the same DRAM column in
     the first block.
   Target: per-core POST -> drain end back to the pull path's 2.78/3.31/4.78/5.16 with POST at 1.35 µs after the
   last arrival. That would be worth -0.2..-0.4 µs per shape.
2. **If (1) doesn't pay, revert the AG release to the pull path.** It is simpler: no forked forwarder, no sharded
   scratch, no per-source fields. Then compare "r04-b04-a02 + HiFi2" against this node. If they are equal within
   noise, prefer the pull path for upstreaming.
3. **Shorter single multicast** (F_MCAST 0.84 µs, go at 0.65 µs after the last arrival). Only worth it once (1)
   makes POST start matter. Options: NoC0, no loopback, `linked` flag behind the data, a 2 KB packed page.
4. **The remaining PRE tail (~0.45-0.5 µs fixed, r04-b01-a02 #2)** and the posted-drain production fence
   (r04-b04-a01 #2) apply to every lineage.
