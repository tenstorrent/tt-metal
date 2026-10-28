# Channel allocation: free depths let the slots follow the credit loops

Status: data note for the case for the new router (2026-10-28). Not a decision: the VC1 split is not hardcoded (user,
2026-10-28); `alloc_model.py` is the proposed way to arrive at it, and its inputs move with the router's shape.
Data: RESULTS §98 (VC1 allocation sweeps, jobs 109-112; 1D slot split, jobs 076-078; worker window, jobs 104-105).

## 1. The constraint main's router puts on allocation

- An ERISC has a fixed number of packet slots (Blackhole: 81 of 4,448 B in 2D / VC1 builds, 83 in 1D). Allocation is
  zero-sum: a slot given to one channel is taken from another.
- Main's allocator picks the first row of a per-architecture tier table that fits: one depth for every channel of a VC,
  all powers of 2, then every spare slot to the worker channel. On a VC1 router that is VC0 forwarding 4, VC0 receiver
  8, VC1 forwarding 2, VC1 receiver 4, and the worker 51.
- Main's router needs those powers of 2: its sender channels' index arithmetic (`wrap_increment`, the 2N pointer wrap
  in `edm_fabric_flow_control_helpers.hpp`) has power-of-2 paths, and a sender channel of another depth costs the flows
  through it 2-6 cycles/packet (RESULTS §98, 1D slot split: a 26- or 19-slot worker +1.6 to +3.3, a 25- or 19-slot
  forwarding channel +6.2).
- So with main's router the only choices are 2x steps per channel kind, and the table takes the first that fits.

The new router (v4) has no power-of-2 path in its sender channels: any worker or forwarding depth costs the same
(RESULTS §98 slot sweep: worker 19, 27, 35 and forwarding 15, 20, 24, 28 all fine), and completion groups of 1, 2 or 4
slots need not divide the ring. Its receiver rings still prefer a power of 2: another size makes `ring_add` a compare
and a branch, ~8 cycles/packet on E1 wherever E1 binds (rings: 24, 40, 48 all +8). So sender channels can take any
depth, and receivers move in powers of 2 (where E1 does not bind, e.g. the VC1 ring on a VC1 router, any size).

## 2. What the slots are worth: each channel is a credit window

A channel of W slots whose credit loop takes RT cycles carries at most one packet per RT / W cycles (Little's law).
Measured on the VC1 split (two 2x2 meshes, v4gi -O3), a row runs at the slowest window on its path, or at its service
bound when none binds; a fit of `RT0 / W + b` per window kind (b: what each packet in flight adds to the loop) matches
all 112 measured points within 3.8% (mean 0.6%; `alloc_model.py fit`):

| window (VC1 router) | round trip at the window sizes measured (cycles, 16 B / 4352 B) | slots it needs to stop binding the 1-hop rate (~158) |
| --- | ---: | ---: |
| VC0 ring behind an intra-mesh link (groups of 2) | 1,468 / 1,584 | ~10, so 16 at groups of 4 |
| VC0 ring behind the inter-mesh link | 1,860 / 1,894 | ~12, so 16 |
| VC0 forwarding channel (into a ring with groups of 4) | 1,911 + 53 W / 2,131 + 46 W | ~18 at 1-hop rate (never reached) |
| VC1 forwarding channel (groups of 1) | 1,496 + 29 W / 1,744 + 24 W | ~6 (against the VC1 path's ~290) |
| VC1 ring (groups of 1) | 1,280 / 1,499 | ~5, then the VC1 path's own bound (~290) |
| worker channel | ~2,700 (one binding point: 15 slots, bursts) | ~18-19 |

Wanted in all: ~18 + 3 x 18 + 16 + 3 x 6 + 5 = ~111 slots against 81. Every split is a trade, and main's table
makes it badly: 51 worker slots where ~19 do, and every forwarding window and ring at a quarter to a half of its loop.

## 3. Main's allocation against the ones the new router can use (VC1 split)

cycles/packet; worker, VC0 forwarding, VC0 receiver, VC1 forwarding, VC1 receiver. Main's router as shipped (kernel
v1, flags 0, firmware keepalive, the builder's -Os); v4gi at -O3. Jobs 110-112 (job 112 repeats job 111's 19,9,16,5,4
within 0.4).

| row | main's router, main's table 51,4,8,2,4 | main's router, 25,8,16,4,4 | v4gi, main's table | v4gi, best all-power-of-2 25,8,16,4,4 | v4gi 18,8,16,6,5 (model 1:1) | v4gi 18,9,16,5,5 (model 3:2) | v4gi 19,10,16,4,4 (model 2:1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-hop E/W 16 B / 4352 B | 307.2 / 310.8 | 303.3 / 306.5 | 183.4 / 198.0 | 158.1 / 157.9 | 157.9 / 158.0 | 157.9 / 158.0 | 157.8 / 158.0 |
| 2-hop 16 B / 4352 B | 454.1 / 518.4 | 305.8 / 303.4 | 482.2 / 530.0 | 290.6 / 310.6 | 292.5 / 310.9 | 266.1 / 285.0 | 240.5 / 258.2 |
| multicast 4352 B | 277.0 | 200.5 | 289.5 | 169.6 | 169.7 | 156.4 | 144.8 |
| inter-mesh, direct link 16 B / 4352 B | 333.3 / 333.5 | 330.3 / 330.7 | 232.4 / 236.7 | 206.3 / 207.1 | 206.4 / 207.3 | 206.4 / 207.3 | 206.3 / 207.2 |
| inter-mesh, then VC1 hops, 16 B (3 rows) | 795-899 | 436-459 | 781-853 | 398-446 | 299-326 | 335-373 | 398-446 |
| same, 4352 B | 982-1,035 | 500-523 | 894-966 | 456-500 | 326-353 | 377-412 | 456-500 |
| mixed load, 16 B: VC0 worker / VC1 flow on one link | 353 / 855 | 345 / 475 | 227 / 843 | 221 / 454 | 212 / 340 | 216 / 386 | 221 / 453 |

What it shows:
- **Main's table costs main's router too**, even within powers of 2: 25,8,16,4,4 takes its 2-hop row 454 -> 306,
  multicast 277 -> 201, and its inter-mesh flows with VC1 hops ~800-1,035 -> ~440-520. Main's 1-hop rate (~305, its
  E0) then caps it: its forwarded rows can not get below that.
- **The new router's faster pass makes the windows matter more.** At ~158 per packet a window binds at half the round
  trip it bound at with main's router, so main's table costs v4 more: 1-hop 183 / 198 instead of 158, 2-hop 482.
- **Off powers of 2, the same VC0 with a faster VC1:** 18,8,16,6,5 against the best all-power-of-2 split 25,8,16,4,4
  has identical VC0 rows (1-hop, 2-hop, multicast within 2) and VC1 rows 97-130 faster (24-29%). The power-of-2 split
  has to give the worker 25 slots it can not use (~18-19 do), because the next step for each forwarding or ring
  window costs 3-12 slots.
- **Every point between is reachable:** the VC0 / VC1 trade moves in 3-slot steps (one slot per forwarding channel),
  and the model's weight picks the point.
- Against main as shipped (first column), the new router with an allocation from the model: 1-hop 307 -> 158,
  2-hop 454 / 518 -> 266 / 285 (3:2), inter-mesh with VC1 hops 795-1,035 -> 335-412 (3:2) or 299-353 (1:1).

## 4. The same in 1D and in 2D (VC0 only)

Data: RESULTS §98 slot allocation sweep (jobs 070-075, router v4us, an earlier v4; 4352 B suites; worker / forwarding
/ receiver):
- 1D (83 slots, one forwarding channel): the tier's 16 / 16 / 32 against v4's even split 26 / 25 / 32: 1-hop 131.8 ->
  124.5, bidirectional 140.8 -> 124.8, 3-hop 144.3 -> 132.6 (worker 16 binds 1-hop, forwarding binds 3-hop up to ~28).
  Main's router at 26 / 25 gains on multicast but loses 2-6 on every row through a non-power-of-2 channel; its best is
  16 / 32 / 32 (job 078: forwarded multicast rows -5 to -30, nothing else moves), which gives up v4's 1-hop gain.
- 2D (81 slots, three forwarding channels): today's 17 / 16 / 16 uses every slot, and each window binds somewhere:
  worker 17 costs 1-hop rows 2-9 (saturated from 20), forwarding under 16 costs 3-hop 6-123, a receiver of 8 costs every
  row 31-53. So the 2D split is a trade between 1-hop and forwarded rows, not free gain.
- Rings and tori (first-level acks): no window binds anywhere (worker 19-43, forwarding 24: within 1.6); only the
  receiver ring's size matters, as E1's cost when it is not a power of 2.

The same model and rule (`alloc_model.py fit 2d`, `fit 1d`, `alloc 2d`, `alloc 1d`), fitted to that sweep alone:

| family | fit | rule's allocation | measured there | today |
| --- | --- | --- | --- | --- |
| 2D | 60 points, mean error 0.3%, max 2.3% | 20 / 15 / 16 | the sweep's best 2D split: stream mixes mean -3.33 (33 rows), 4352 B rows -0.40 (1-hop -2 to -9, 3-hop +6.2) | 17 / 16 / 16 |
| 1D | 35 points, mean 0.6%, max 7.5% (1-hop scatter at 16 workers) | 19 / 32 / 32 | mean -5.92 over 12 rows; 23 / 28 / 32 -6.60, 27 / 24 / 32 -6.46 (within ~0.7) | 26 / 25 / 32 (v4), 16 / 16 / 32 (main) |
| VC1 | 112 points, mean 0.6%, max 3.8% | 18,8,16,6,5 (1:1) to 19,10,16,4,4 (2:1) | every weight's pick measured (jobs 110-112) | 51,4,8,2,4 |

- 2D: receiver 16 (8 binds every row; 16 is the smallest power of 2 that does not), worker 18 from its round trip
  (~2,250 for bidirectional rows / 126.7), forwarding 15 from the 47 left, and the 2 slots 15 x 3 leaves to the worker.
- 1D: receiver 32 (16 binds the 1-hop scatter rows, whose ring loop is ~2,340), worker 19, and all 32 left to the one
  forwarding channel (3-hop still gains up to ~28).
- 2D's pick moves today's split the way the rule's priority says: 1-hop rows first (their windows are sized before the
  forwarding side), so 3-hop pays 6 for 1-hop's 2-9. That priority is a policy, like VC1's weight.

## 5. Arriving at the split: `alloc_model.py`

`alloc_model.py fit` fits the windows above from the sweeps; `alloc_model.py alloc VC0:VC1` allocates:
1. VC0 receivers at the largest completion group (T = 4: fewest credits; E0's pass binds the 1-hop rows, and T 4 -> 2
   costs them 9.5) and a power of 2 (E1's cost otherwise), the smallest whose window binds no row.
2. The worker: the smallest window that binds no 1-hop row (from its fitted round trips, or an input where no point
   bound it).
3. Everything else, a unit at a time (one slot per forwarding channel of a VC: 3 in 2D, 1 in 1D; 1 for the VC1 ring),
   to the window that binds the forwarded row with the worst weighted cycles/packet (max-min water-filling; rows at
   their service bound are frozen); leftovers to the worker.
Policy inputs: the order (1-hop windows before the forwarding side) and, with two VCs, the VC0:VC1 weight.

Section 6 has its outputs for four weights; three of them were run on the device (job 112 and jobs 110-111).

The weight is the one policy input (how much inter-mesh traffic matters next to intra-mesh); everything else comes
from round trips, which change with the router (E0's pass, E1's work, the completion groups) and with the topology.
They can be measured per router version (`router_rt.hpp`-style timing, or one window-bound point per kind: rate x W),
so the split is re-derived, not re-tuned. Predictions for unmeasured allocations were written before job 112
(`vcd_predictions.txt`); job 112 checked them (section 6).

## 6. Check of the model on allocations it was not fitted to (job 112)

The model was fitted to jobs 109-111 and its predictions for five allocations written down before job 112
(`vcd_predictions.txt`); `alloc_model.py check` compares (FIT_JOB112=0 for the fit of jobs 109-111 only).
- Every VC0 row (1-hop, 2-hop, inter-mesh direct) within 1.3%, including the 18-slot worker (bursts within 1.3 of 19);
  the all-power-of-2 point within 0.3% (every window in it was inside the measured range).
- The VC1 rows it had to extrapolate were optimistic: VC1 ring 5 and forwarding 6 predicted 271 / 309, measured 301 /
  326; VC1 ring 7 and forwarding 8 predicted 207 / 235, measured 290 / 293. The VC1 path has a service bound of its
  own (~290 / 293: its packets cross the inter-mesh router's crossover and two receivers) that no earlier allocation
  reached; the model assumed the direct link's (206).
- Refitted with job 112 (that bound now measured): 112 points, mean error 0.6%, max 3.8%. The allocations it gives:

| VC0:VC1 weight | allocation | model 2-hop | model inter + VC1 | measured |
| --- | --- | ---: | ---: | --- |
| 1:1 | 18, 8, 16, 6, 5 | 292 / 312 | 290 / 315 | 2-hop 293 / 311, VC1 299-326 / 326-353 |
| 3:2 | 18, 9, 16, 5, 5 | 265 / 282 | 329 / 373 | 2-hop 266 / 285, VC1 335-373 / 377-412 |
| 2:1 | 19, 10, 16, 4, 4 | 244 / 259 | 403 / 460 | 2-hop 241 / 258, VC1 398-446 / 456-500 |
| 1:2 | 20, 6, 16, 7, 6 | 371 / 401 | 290 / 293 | (not run) |

(16 B / 4352 B. The measured VC1 ranges are the three inter-mesh rows; the model's figure is the one-VC1-hop row.)

## 7. What the experiments run with (2026-10-28, job 113)

The allocation stays offline (user). The builder carries the slots the study's router (kernel v2) runs with, so
experiments are not window-bound by the tier table:
- VC1 routers (Blackhole, 4 VC0 + 3 VC1 senders): 19, 9, 16, 5, 4 by default (`TT_FABRIC_EXP_SLOTS_VC1` overrides,
  `TT_FABRIC_EXP_TIER_SLOTS=1` restores 51, 4, 8, 2, 4).
- 1D: 26 / 25 / 32 (the builder's even split; the rule's 19 / 32 / 32 is equal within noise).
- 2D: 17 / 16 / 16, the tier table plus spare to the worker (kept by the user's call; the rule's 20 / 15 / 16 trades
  3-hop rows for 1-hop rows).
- Main's router: the tier tables everywhere.
When the router's shape changes, re-run `alloc_model.py` on a fresh sweep (a few window-bound points per kind) before
changing these.

## 8. Limits

- 2x2 meshes: at most 2 hops a side; single flows and one mixed pair (VC0 + VC1 on one link: each flow keeps the
  split's effect, the shared link costs both 25-70). Galaxy-sized meshes have longer paths and more merging.
- The worker's round trip has one binding point (15 slots); the 2D numbers are from the 17-slot worker study.
- The service bounds of the forwarded classes were never reached within 81 slots, so the model's floor for them is a
  lower bound.
- The 2D and 1D fits are from v4us (jobs 070-075), the VC1 fit from v4gi: round trips are per router version, so
  adopting the rule for VC0-only builds means one calibration run with today's router first.
- Rings and tori are outside the model: no window binds there, so the rule only fixes their receivers (power of 2).
