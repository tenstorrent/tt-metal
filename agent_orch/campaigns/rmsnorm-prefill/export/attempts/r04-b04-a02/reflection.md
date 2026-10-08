# r04-b04-a02 result: 1.3997 (ok)

## What happened vs expected
Valid on every shape. PCC 0.9999985 and max_abs 0.0204-0.0240 match the parent: same bytes, and the same-VC
ordering of the stick push held. Parent r04-b04-a01 -> this node (µs, chip mean):

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 12.32 | 11.96 | -0.35 (-2.9%) |
| h4096 | 13.64 | 13.62 | -0.03 (noise) |
| h6144 | 17.35 | 16.91 | -0.44 (-2.5%) |
| h7168 | 18.65 | 17.93 | -0.71 (-3.8%) |

The score is 1.3997, up from 1.3666 (+2.4%). That is the new campaign best, and it lands at the bottom of the
predicted 1.39-1.40. Three shapes moved well outside the ±1% band. h4096 did not move.

## Why (profiler evidence)
Scripts are in `analysis/`: `push.py` is from r04-b03-a01 and `late2.py` is from r04-b01-a01. Outputs are in
`push_out.txt` and `late2_out.txt`, each with this node and the parent. Values are µs from the first worker start,
medians over measured calls and chips.

| shape | push dur | push end max | F_FABRIC end max | AG-wait end max | drain end max |
|---|---|---|---|---|---|
| h3584 | 0.63 -> 0.25 | 4.99 -> 4.42 | 7.24 -> 6.91 | 7.76 -> 7.43 | 11.92 -> 11.56 |
| h4096 | 0.63 -> 0.25 | 5.55 -> 5.27 | 7.86 -> 7.82 | 8.39 -> 8.34 | 13.29 -> 13.25 |
| h6144 | 0.62 -> 0.24 | 7.83 -> 7.10 | 10.18 -> 9.65 | 10.70 -> 10.17 | 17.22 -> 16.60 |
| h7168 | 0.63 -> 0.25 | 8.31 -> 7.75 | 10.98 -> 10.32 | 11.50 -> 10.84 | 18.19 -> 17.58 |

- **The push handshake port behaves as it did alone.** The push takes 0.25 µs instead of 0.63 µs on every core, and the
  forwarder sees the inc ~0.1 µs after the slowest push returns.
- **h6144 now gains.** In r04-b03-a01, where the drain was non-posted, h6144 absorbed the earlier AG end. With the
  posted drain, the 0.53 µs earlier AG end reaches the kernel end (-0.62 µs drain end max). The two changes compound
  on h6144, as the proposal guessed.
- **h4096 is the exception.** Its slowest push ended only 0.28 µs earlier, and F_FABRIC end moved just 0.04 µs, so the
  AG wasn't gated by the local pushes. This run can't tell whether the cause was a remote chip's arrival or fabric
  latency. Every other shape gained 0.33-0.66 µs at F_FABRIC end.
- **Gamma streaming port.** On the parent, dev0 had straggler calls in 8/10 h7168 runs (last-med drain 1.36 µs,
  maxlag 1.53) and dev0 in 5/10 h6144 runs. Here dev0 has 0/10 on both. A new, smaller straggler appears on h7168
  dev3: 7/10 calls, maxlag-med 0.67 µs, last-med drain 0.60. Per-device h7168 kernel spans are 19.15 / 18.16 /
  17.31 / 16.98 µs (dev0..3). That spread looks like launch skew from dev0 starting first, not something inside a
  chip.

## Classification
win (+2.4% geomean, new best). Combining the three round-4 writer wins is close to additive on h3584, h6144 and
h7168, and neutral on h4096.

## What a child of this node should try next
1. **Cross-device launch skew on h7168** (dev0 span 19.15 vs dev3 16.98 µs). The chip mean pays for dev0 waiting at
   the AG for later-launching chips. Use `late2.py` / a start-time script to check whether dev0 launches first in
   every call. If the skew is host dispatch, it is outside the op. If it comes from a per-chip phase, see #3.
2. **h4096 AG not gated by local pushes:** F_FABRIC end did not move even though the pushes ended 0.28 µs earlier.
   Measure per-chip F_COLLECT end and the fabric arrival order on h4096 (r04-b02-a01's `ag.py`) to find which chip
   or hop gates it.
3. **Dev3 h7168 straggler (7/10, 0.67 µs POST lag).** Find the core with late2's per-core lag (likely the same
   high-x / y=3 position class named in r03-b04-a03 #2). Test a per-core drain order, or a NoC0 share that favours
   it.
4. Still untried levers: PRE x*x at HiFi3/HiFi2 (PRE tail ~0.7 µs, suggested 5+ times), and forwarder-side ack-free
   release (r04-b03-a01 #2).
5. Production caveat carried over: the posted drain needs a final per-bank non-posted fence (r04-b04-a01 #2).
