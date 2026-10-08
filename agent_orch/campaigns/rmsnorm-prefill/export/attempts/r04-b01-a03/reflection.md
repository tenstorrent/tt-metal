# r04-b01-a03 result: 1.4475 (ok)

## What happened vs expected
Valid on every shape. PCC is 0.9999985 and max_abs 0.0220-0.0239, the parent's values (HiFi2 PRE; the writer change
sends the same bytes). Score 1.4475 is the new campaign best: +4.1% over the parent r04-b01-a02 (1.3911) and +3.4% over
the previous best r04-b04-a02 (1.3997). I predicted 1.43-1.44, and it beat that slightly. µs, chip mean:

| shape | parent r04-b01-a02 (HiFi2) | r04-b04-a02 (writer wins) | this (both) | vs r04-b04-a02 |
|---|---|---|---|---|
| h3584 | 12.04 | 11.96 | **11.67** | -0.29 (-2.4%) |
| h4096 | 13.51 | 13.62 | **13.05** | -0.57 (-4.2%) |
| h6144 | 17.13 | 16.91 | **16.35** | -0.56 (-3.3%) |
| h7168 | 18.16 | 17.93 | **17.35** | -0.58 (-3.2%) |

Every shape is the campaign's fastest, and every move is 2.4-4.2x the ±1% noise band. The two lineages' gains are
additive.

## Why (profiler evidence)
The script is the parent's `pre.py`. `pre_out.txt` has this node's output and `pre_best_out.txt` has r04-b04-a02's.
Values are medians over measured calls x 4 chips, in µs from the call's first worker BRISC start. Each cell is
r04-b04-a02 -> this:

| shape | stat ready - input end, med / max | push end max | F_FABRIC end max | AG-wait end max | drain end max |
|---|---|---|---|---|---|
| h3584 | 0.67/0.85 -> 0.45/0.54 | 4.42 -> 4.06 | 6.91 -> 6.65 | 7.43 -> 7.17 | 11.56 -> 11.41 |
| h4096 | 0.79/1.07 -> 0.46/0.54 | 5.27 -> 4.72 | 7.82 -> 7.32 | 8.34 -> 7.84 | 13.25 -> 12.72 |
| h6144 | 0.83/1.15 -> 0.50/0.58 | 7.10 -> 6.54 | 9.65 -> 9.04 | 10.17 -> 9.56 | 16.60 -> 16.07 |
| h7168 | 0.82/1.13 -> 0.50/0.56 | 7.75 -> 7.13 | 10.32 -> 9.63 | 10.84 -> 10.15 | 17.58 -> 16.66 |

- **HiFi2 does on this writer what it did on the parent's.** The PRE tail after the last input lands shrinks by
  0.22-0.33 µs at the median and 0.3-0.57 µs on the slowest core. R_INPUT end is unchanged (±0.04 µs).
- The push stays at 0.24-0.25 µs, so the ack-free handshake is intact. Push end max moves 0.36-0.62 µs earlier.
  F_COLLECT and F_FABRIC end move by the same amount, which means the AG is still gated by the local pushes.
- This includes h4096. In r04-b04-a02, its F_FABRIC end didn't move. Here it moves 0.50 µs, so that was cross-chip
  noise in that run, not a fabric floor.
- The post-AG window (AG-wait end max -> drain end max) is unchanged at 4.2/4.9/6.5/6.5 µs vs 4.1/4.9/6.4/6.7 µs in
  r04-b04-a02. The whole gain is the earlier AG start, carried straight through to the kernel end.

## Classification
win (+3.4% vs the previous best, every shape outside noise). Pure combination: the compute-only HiFi2 PRE edit and the
writer-only posted drain + ack-free push are orthogonal and additive.

## What a child of this node should try next
Where h7168's 16.7 µs goes now (max over cores): the input read ends at 6.4, the PRE tail is 0.5, the push 0.25, the
local collect 0.1, the fabric AG 2.4 (F_COLLECT -> F_FABRIC end), and AG release -> drain end is 6.5.
1. **The post-AG drain tail is now ~40% of the kernel** (4.2-6.5 µs, POST ~2.5-4.3 µs, and the drain trails POST by
   0.8-1.4 µs). Issue, VC, cmd-buf, ack and aggregate bandwidth are all ruled out. Re-measure the per-core drain rate
   against core position and destination DRAM column with posted writes on, then re-tune the per-core NoC0 share
   (r02-b02-a01 #1). Also consider deliberately staggering drain start order per core: r04-b02-a02 found that
   synchronized drains contend.
2. **The fabric AG costs ~2.4 µs (F_COLLECT end -> F_FABRIC end)** on every shape, the second-largest fixed block. Try
   the forwarder-side flush-then-inc (no write-ack round trip before the fabric's go/inc, r04-b03-a01 #2), and break
   F_FABRIC into per-hop zones to see whether it is fabric latency or the slowest remote chip.
3. **The input read (3.4-6.4 µs) is the largest pre-AG block.** It is width-proportional, so it is bandwidth-bound per
   core. Overlapping x*gamma and POST-input prefetch is already done. The remaining lever is more reading cores or
   NoCs per row. The column split (r01-b03-a02) was neutral on older lineages, but now that the AG path is short it
   may pay off. This is expensive to try.
4. The PRE tail is now a flat ~0.5 µs. Waiting per tile for the last block could cut ~0.1 µs at most. Low priority.
5. Production caveat carried over: the posted drain needs a final per-bank non-posted fence (r04-b04-a01 #2).
