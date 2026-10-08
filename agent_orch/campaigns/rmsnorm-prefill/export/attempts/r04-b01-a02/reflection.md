# r04-b01-a02 result: 1.3911 (ok)

## What happened vs expected
Valid on every shape. PCC 0.9999985 on all four, as before. max_abs is 0.0220-0.0239, inside the parent's
0.0204-0.0240 range: the HiFi2 bias on sum(x^2) (~-0.3% in emulation) did not show up in max_abs at all, and the gate
(0.05) has lots of room. Parent r04-b01-a01 -> this node (µs, chip mean):

| shape | parent | this | change |
|---|---|---|---|
| h3584 | 12.43 | 12.04 | -0.39 (-3.1%) |
| h4096 | 14.12 | 13.51 | -0.61 (-4.3%) |
| h6144 | 17.73 | 17.13 | -0.60 (-3.4%) |
| h7168 | 18.68 | 18.17 | -0.51 (-2.8%) |

Score 1.3911 vs 1.3436 (+3.5%), every shape 3-4x the ±1% noise band. New campaign best (previous best r04-b04-a01,
1.3666). I expected -0.1..-0.4 µs per shape; the measured -0.4..-0.6 µs is above that range.

## Why (profiler evidence)
`pre.py` (r04-b03-a01's push.py plus the NCRISC R_INPUT end; `python3 pre.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`).
Outputs: `pre_out.txt` (this node), `pre_parent_out.txt` (r04-b01-a01). Medians over measured calls x 4 chips, µs from
the call's first worker BRISC start:

| shape | R_INPUT end max | stat ready - input end, med | ... max over cores | push end max | AG-wait end max | drain end max |
|---|---|---|---|---|---|---|
| h3584 | 3.38 -> 3.36 | 0.64 -> **0.45** | 0.85 -> **0.53** | 4.83 -> 4.54 | 7.69 -> 7.26 | 11.95 -> 11.46 |
| h4096 | 3.96 -> 3.94 | 0.79 -> **0.47** | 1.05 -> **0.53** | 5.67 -> 5.12 | 8.42 -> 8.01 | 13.58 -> 13.11 |
| h6144 | 5.69 -> 5.69 | 0.83 -> **0.50** | 1.19 -> **0.57** | 7.57 -> 6.96 | 10.41 -> 9.77 | 17.33 -> 16.65 |
| h7168 | 6.31 -> 6.28 | 0.83 -> **0.50** | 1.05 -> **0.57** | 8.06 -> 7.51 | 10.76 -> 10.38 | 17.72 -> 17.28 |

("stat ready" = W_PUSH start: the writer enters the push once compute's row-0 stat is in the CB.)
- **PRE was math bound.** The input read is unchanged (R_INPUT end within 0.03 µs), but the time from the last input
  landing to the stat being ready fell by 0.2-0.33 µs at the median and 0.3-0.6 µs on the slowest core. The core-to-core
  spread of that tail collapsed (median-to-max 0.2-0.36 -> 0.03-0.07 µs): HiFi4 compute was lagging the read by an
  amount that varied per core, and the slowest core gated F_COLLECT.
- So the slowest push ends 0.29-0.61 µs earlier. The AG starts and ends that much earlier (F_COLLECT and F_FABRIC
  ends move by the same amount), and POST and the drain are unchanged in duration, so the drain end and kernel end move
  0.44-0.68 µs. That is the score gain.
- What's left of the PRE tail is a flat ~0.45-0.5 µs on every shape and every core. With the math halved, that looks
  like the fixed part: the last block's wait + 4 muls, the S pack, the ones*S^T matmul init/unpack/pack, and the CB
  handoff to BRISC.
- Accuracy: the matmul part is exact (ones in SrcB). The x*x part drops SrcB's last bf16 mantissa bit. max_abs moved
  by < 0.002 per shape, within the run-to-run spread of earlier nodes.

## Classification
win (+3.5% geomean, all four shapes -2.8..-4.3%, mechanism confirmed: PRE tail -0.2..-0.6 µs with an unchanged read).

## What a child of this node should try next
1. **Stack the two writer-only round-4 wins on this node:** r04-b04-a01's posted output-drain writes (-0.2..-0.7 µs, drain
   rate) and r04-b03-a01's flush-then-inc stick push (W_PUSH 0.63 -> 0.25 µs, still 0.63 here). Both are in
   `dit_rmsnorm_fused_worker_writer.cpp` and orthogonal to this compute-only change (r04-b01-a01's gamma streaming calls
   `push_stick` from inside the gamma loop, so merge carefully). If a sibling already combined them, port this
   compute diff onto that node instead: it's 3 local edits in the PRE block. Expected ~1.42-1.44.
2. **The remaining ~0.47 µs PRE tail is fixed per call.** Ideas, cheapest first:
   - wait per tile (not per 4-tile block) for the last input block, so the muls overlap the last block's arrival;
   - drop the S pack -> unpack -> matmul -> pack hop: r03-b03-a02's DST-resident transpose_dest + SFPU column sum was
     neutral at HiFi4, but the trade-off may differ now that the muls are shorter. Add a TRISC_1 zone first.
3. **Fidelity elsewhere.** The x*gamma pre-pass (bf16 x bf16, under the AG wait) is off the critical path, so it doesn't
   matter. POST (fp32 x*gamma times bcast 1/rms) is unpack-bound and the drain trails it, so lower POST fidelity would
   probably not show up in the kernel end; only try it after the drain is faster. Don't try LoFi for PRE: it truncates
   SrcA to 5 mantissa bits.
