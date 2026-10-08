# r04-b03-a02 result: 1.3906 (ok)

## What happened vs expected
Valid on every shape. PCC is 0.9999985 and max_abs is 0.0204-0.0240, the same as both source nodes. Score 1.3906 is the
new campaign best: +2.7% over the parent (1.3543) and +1.8% over r04-b04-a01 (1.3666), the best node so far. I predicted
~1.38, and it beat that slightly.

µs, chip mean:

| shape | parent r04-b03-a01 (push) | r04-b04-a01 (posted) | this (both) | vs parent |
|---|---|---|---|---|
| h3584 | 12.33 | 12.31 | **12.25** | -0.6% (noise) |
| h4096 | 13.76 | 13.64 | **13.42** | -2.4% |
| h6144 | 17.54 | 17.35 | **16.98** | -3.2% |
| h7168 | 18.94 | 18.65 | **18.15** | -4.2% |

Every shape is the campaign's fastest. The two gains are roughly additive on h4096/h7168 and super-additive on h6144.
On h6144 the parent's earlier AG end was absorbed by the drain, and the faster posted drain now lets it carry through.
That is what the proposal guessed.

h3584 barely moved on the chip mean. Its chip max is 14.20 µs (parent 13.09), and us_min 9.83 vs p50 12.03 shows a
larger cross-chip/launch skew in this run. At 28 tiles the posted gain is only ~0.18 µs anyway.

## Why (profiler evidence)
`push.py` (parent's script, copied here; output in `push_out.txt`). Medians over measured calls x 4 chips, µs from the
call's first worker BRISC start. Parent values from r04-b03-a01/push_out.txt:

| shape | push dur med | AG-wait end max | drain end max | AG-wait end -> drain end max |
|---|---|---|---|---|
| h3584 | 0.24 / 0.24 | 7.45 -> 7.76 | 11.71 -> 11.89 | 4.26 -> **4.13** |
| h4096 | 0.25 / 0.24 | 8.23 -> 8.32 | 13.31 -> 13.32 | 5.08 -> **5.00** |
| h6144 | 0.25 / 0.25 | 10.12 -> 10.32 | 17.10 -> 16.75 | 6.98 -> **6.43** |
| h7168 | 0.25 / 0.24 | 11.14 -> 11.01 | 18.23 -> 17.57 | 7.09 -> **6.56** |

- The push stays at 0.24 µs, so the parent's handshake is intact. The posted drain does not interfere with it, because
  it runs strictly after the push.
- The post-AG window (AG release -> last drain end) shrinks by 0.13/0.08/0.55/0.53 µs. That matches r04-b04-a01's
  measured drain-tail cuts (0.18/0.20/0.43/0.46 µs) on the wide shapes.
- AG-wait end max varies by ±0.2-0.3 µs between runs. That is cross-chip fabric/launch skew, not this change. It
  explains why h3584/h4096 "drain end max" are flat even though the window shrank.

## Classification
win (+2.7% vs parent, +1.8% vs previous best; 3 of 4 shapes outside the ±1% noise band). It confirms that the AG-start
push cut and the drain-ack cut are orthogonal, and that on h6144 the push gain was hidden behind the drain.

Production caveat, inherited from r04-b04-a01: the posted drain writes have no completion ack. The kernel ends once
they leave L1, not once they land. A production version needs a per-bank fence at the end (one non-posted write per
bank + barrier).

## What a child of this node should try next
1. **Port r03-b04-a03 / r04-b01-a01's streamed gamma** (8-page sticky-trid chunks; writer W_GAMMA block only). It is
   the third orthogonal win. It removed the dev-0 h7168 cross-call straggler (-0.6 µs h7168 on its lineage). This
   lineage still has the old blocking gamma read. Merge carefully: `push_stick` is called from inside the gamma loop
   (r03-b01-a03's poll-while-gamma-lands). Expected ~1.40-1.41.
2. **Make the posted drain production-safe** with a per-bank non-posted fence at kernel end, and measure that it costs
   ≤0.05 µs. Needed before this lineage is upstreamable.
3. **The drain is still the post-AG tail:** 4.1-6.6 µs after AG release, against a POST of ~2.5-4.3 µs. Issue, VC,
   cmd-buf, ack and aggregate bandwidth are now all ruled out, so the remaining suspect is the path toward the two
   DRAM columns. Re-measure the per-core drain rate vs core position with posted writes on, then re-tune the per-core
   NoC0 share (r02-b02-a01 #1).
4. **Forwarder-side "flush, then inc on the same VC"** (parent #2) in F_FABRIC/go release. It is a small AG-path cut
   that now shows up on every shape except where the drain dominates.
5. **PRE x*x at lower fidelity** (HiFi3/HiFi2 for the PRE ELWMUL only) is still untried after 6 suggestions. The PRE
   tail is ~0.7 µs after the last input lands.
