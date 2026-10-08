# r04-b02-a02 result: 1.3175 (ok)

## What happened vs expected
Valid on every shape. PCC 0.9999985 and max_abs 0.0204-0.0240 are the parent's values. So the per-source-device
fields in out_ready work: the 32-bit fused inc carries `1 << (8*d)`, the same-VC write-then-inc ordering holds, and
no page was multicast before it landed. There was no hang. Per shape (µs, chip mean):

| shape | r03-b02-a02 (grandparent, pull) | r04-b02-a01 (parent, burst mcast) | this node |
|---|---|---|---|
| h3584 | 12.59 | 12.85 | 12.69 |
| h4096 | 14.05 | 14.52 | 14.29 |
| h6144 | 17.55 | 18.43 | 18.02 |
| h7168 | 19.32 | 19.38 | 19.26 |

Score 1.3175: +1.4% over the parent (1.2988), but -1.2% vs the grandparent r03-b02-a02 (1.3333). So it repairs most
of the parent's loss, but does not beat the per-worker pull. I expected about -0.6 µs vs the parent and a small gain
over r03-b02-a02. The AG-release part delivered: POST now starts 0.25-0.30 µs earlier than in r03-b02-a02. That gain
did not reach the kernel end.

## Why (profiler evidence)
Scripts in this dir, each run as `python3 X.py $DREAM_HOME/rmsnorm-prefill/reports/<node>`:
- `ag.py`: the parent's script plus the F_MCASTN zone.
- `tail.py`: new; splits the post-AG tail per core.

Outputs: `ag_out.txt`, `ag_r03-b02-a02_out.txt`, `tail_out.txt`. Medians are over measured calls x 4 chips, in µs.

| | r03-b02-a02 | parent | this |
|---|---|---|---|
| F_MCAST (after the last arrival) | — | 1.21 (3 pages + go) | **0.83-0.85 (1 page + go)** |
| go min / max after last arrival (F_FABRIC end) | 0.17 / 0.52 | 1.03 / 1.15 | **0.64 / 0.75** |
| go -> C_COMB start | 0.62 | 0.065 | 0.066 |
| C_POST start after last arrival, max over workers | 1.64 | 1.74 | **1.34-1.39** |
| POST start spread across workers | 0.31 | 0.11 | 0.11 |
| per-core drain end - POST start, h3584/h4096/h6144/h7168 | 2.94/3.51/5.17/5.56 | 3.12/3.82/5.42/5.73 | 3.11/3.83/5.41/5.72 |
| median drain end after last arrival | 4.45/4.98/6.70/7.04 | 4.79/5.48/7.08/7.41 | 4.40/5.11/6.72/7.03 |

1. **The streaming works.** In 158 of 160 chip-calls exactly one peer page was still unsent at the last arrival
   (F_MCAST). Only 2 took F_MCASTN (1.31 µs). So the other pages were hidden under the cross-chip skew, as planned.
2. **One multicast is expensive in fixed latency.** A single 3328 B page plus the 4 B go flag takes 0.64 µs to
   land go at the first worker, and 0.83 µs to the barrier. The parent's 3-page burst was 1.21, so each extra page
   costs ~0.19 µs, and the first page + flag carry a ~0.45-0.6 µs fixed cost. Even so, the combine still starts
   0.25-0.3 µs earlier than with the pull path (go + 0.6 µs read).
3. **The earlier POST is eaten by the drain.** The per-worker POST-start -> drain-end time grew by 0.17-0.32 µs on
   every shape, exactly as in the parent. That matches the change in POST-start spread from 0.31 µs to 0.11 µs.
   - In r03-b02-a02 the 20 serial go incs staggered the cores over ~0.35 µs, so their drains, which are
     contention-bound at the DRAM columns (r03-b04-a02, r04-b04-a01), overlapped less.
   - With a multicast go, all 20 drains start within 0.11 µs and contend harder.
   - Net: the median drain end after the last arrival is the same as r03-b02-a02 (±0.1 µs on every shape).
4. The remaining chip-mean gap to r03-b02-a02 (h6144 +0.47) comes from the pre-AG side: F_COLLECT end is 7.77 vs
   7.27 µs. This lineage still has the dev-0 cross-call straggler (no gamma streaming), so that is call-to-call
   variance, not this change.

## Classification
repairable / neutral. As a repair of the parent's serial burst it is a win (+1.4%, mechanism confirmed: 1 page
left at the last arrival in 158/160 chip-calls). Against the pull path it gives no net gain (-1.2%, mostly pre-AG
noise). The post-AG chain is drain-bound: removing ~0.3 µs of fixed release latency only moves POST, while
synchronizing the drains costs about the same. Lesson: **a faster, synchronized go doesn't pay while the drain is
contention-bound; release staggering was helping the drain.**

## What a child of this node should try next
1. **Port r04-b04-a01's posted drain writes onto this node** (writer W_DRAIN only, orthogonal). With posted writes the
   per-core drain is ~7% faster and less ack-bound, so the 0.3 µs earlier POST may survive. Also port r04-b03-a01's
   push handshake (flush then inc, no acks) and r04-b01-a01's gamma streaming, which kill the pre-AG noise. Compare
   the per-core drain end - POST start with `tail.py`.
2. **Stagger the drains deliberately, keep the early POST.** For example, worker slot s skips/delays the first
   output block by s * ~15 ns, or alternate the bank start order per half. Go itself should stay early. Target:
   per-core drain end - POST start back to r03-b02-a02's 2.94/3.51/5.17/5.56 with this node's 1.34 µs POST start.
3. **Cut the single-multicast latency (0.64 µs to go).** Options: NoC0 instead of NoC1, a rectangle without
   loopback, the flag sent with `linked=true` behind the data, or a smaller payload (the slot layout has a 768 B
   hole: slots 16-19 could pack into rows 1 of faces 0/1, making the page 2 KB). Measure with an F_MCAST zone around
   the data multicast alone.
4. Don't go back to the burst-after-out_ready multicast (parent). Per-source fields are cheap and work.
