# Searching for a better in1 reader placement (offline)

Production in1 buffering is unchanged (cb1 depth 4; the `TT_REGIME_A_CB1_DEPTH` knob is diagnostic-only and
default-off - verified: default path measures 92.16 us on 256x2048x6144, the production baseline, with no
override log). This document is about placement only.

Tool: `in1_place_search.py`. Static geometry + traffic accounting on the device-validated route model.

## 1. What makes a placement good, structurally

The in1 read response goes DRAM endpoint -> core on the reader's NoC, dimension-ordered (x then y) and
strictly unidirectional with torus wrap. So for a given (bank, noc) the cheap cells form a region DOWNSTREAM
of the endpoint, and everything else costs most of a lap:

- banks 0-3 (endpoint physical x=0): NOC_0 (+x) is cheap on the LEFT half (logical x=0..5, 1..6 hops) and
  costs 11-15 hops on the right; NOC_1 (-x) is the exact opposite (cheap on the RIGHT half via the wrap).
- banks 4-7 (endpoint physical x=9): NOC_0 cheap on the RIGHT half (2..6 hops), NOC_1 cheap on the LEFT.
- The same applies in y: NOC_0 wants y >= endpoint row, NOC_1 wants y <= endpoint row. Being ONE row on the
  wrong side costs 11 hops, which is why the current placement shows 9-25 hop readers even on NOC_0.

Two consequences:

1. **The ideal layout is a CROSS**: each bank's NOC_0 readers cluster on one side of its DRAM column and its
   NOC_1 readers on the other. A bank's readers should NOT all be clustered next to the bank, which is what
   `find_near` does today.
2. **Row alignment matters more than column proximity**: put readers in the endpoint's ROW and walk outward
   in x only. That gives 0 y-hops and 1..6 x-hops (mean ~3-4) instead of paying an 11-hop y wrap.

There is also a hard floor on peak link load: all readers sharing one endpoint must traverse that endpoint's
first link, so `peak_in1 >= readers_per_endpoint * in1_bytes`. With 12 readers per bank split 50/50 across the
two NoCs, that is 6 * in1_bytes.

## 2. How to search

**The main objective is exactly solvable, not a heuristic search.** With the reader NoC fixed, "minimise total
in1 read hop-bytes" is a LINEAR ASSIGNMENT problem: slots = (bank, kk, nn) readers, cells = logical cores,
cost[slot][cell] = response hops * in1_bytes. Hungarian solves 96x110 in milliseconds
(`scipy.optimize.linear_sum_assignment`). No search heuristics, no local minima.

Peak link load is a min-max objective and not linear - but it turns out not to need a search either, because the
assignment already lands exactly on the endpoint-egress floor on every shape tested. An iterative
reweighting pass (penalise cells whose path uses already-hot links) was implemented and is **useless to
harmful**: it cannot beat the floor and it perturbs the placement, making the ring and reduction worse.

So the right formulation is two-stage:

- **Stage 1 (exact):** Hungarian on in1 hop-bytes. This both minimises in1 and achieves the peak floor.
- **Stage 2 (the interesting part):** the stage-1 optimum is highly degenerate - many placements share the
  same in1 cost - so spend that freedom on the OTHER traffic classes. Constrain in1 cost to the stage-1
  optimum (or optimum x (1+eps)) and minimise reduction-chain + in0-ring cost by pairwise swaps. This is the
  same budget discipline that failed for the ring-order work, but here the budget is anchored on a PROVABLE
  optimum rather than on production's arbitrary choice.

## 3. Results (all at DEPLOYED picker configs)

`current` = production `find_near` spiral. `cross` = the a priori structure from section 1. `hung` = exact
assignment. `alt` = production's alternating NoC rule; `block` = NoC by contiguous kk blocks.

| shape (config) | variant | in1 hops | mean | in1 MB-hops | peak_in1 | red hops | ring hops | ALL-peak |
|---|---|---|---|---|---|---|---|---|
| 512x6144x2304 Ns1/Pk12/Sm1 | current | 1260 | 13.1 | 372 | 3.54 | 1164 | 683 | 5.34 |
| | **cross/alt** | 395 | 4.1 | 116 | **1.77** | 933-959 | **533** | **2.98** |
| | hung/alt | **386** | 4.0 | 114 | **1.77** | **933** | 574 | 3.11 |
| 256x2048x6144 Ns3/Pk4/Sm1 | current | 1260 | 13.1 | 330 | 3.15 | 781 | 683 | 3.90 |
| | **cross/alt** | 395 | 4.1 | 104 | **1.57** | **732** | **533** | **2.16** |
| 32x6144x1536 Ns1/Pk6/Sm1 | current | 510 | 10.6 | 201 | 2.36 | 421 | 323 | 2.42 |
| | **cross/alt** | 146 | 3.0 | 57 | **1.18** | **373** | **280** | **1.25** |
| 256x15360x768 Ns1/Pk6/Sm2 | current | 660 | 13.8 | 324 | 3.44 | **776** | 683 | 4.19 |
| | **cross/alt** | 146 | 3.0 | 72 | **1.47** | 901 | **526** | **2.70** |
| 256x6144x4608 Ns1/Pk6/Sm2 | current | 660 | 13.8 | 779 | 8.26 | **776** | 683 | 8.55 |
| | **cross/alt** | 146 | 3.0 | 172 | **3.54** | 901 | **526** | **4.29** |

Universal, on every shape:

- **in1 read hops -69% to -78%** (mean 10.6-13.8 -> 3.0-4.1 hops).
- **peak in1 link load exactly hits the endpoint-egress floor** (-50% to -57%). Provably optimal.
- **whole-op peak link load -36% to -50%** - the metric most likely to track wall time.
- **the in0 ring gets 13-23% cheaper for free**, because per-bank clusters end up compact.
- the exact assignment beats the a priori cross heuristic by only 0-2% on in1, so the structure captures
  essentially all of the win and is worth preferring for being deterministic and explainable.

## 4. A priori ideas: which hold up

Good:
1. **Cross layout** (per (bank, noc) downstream region). The dominant idea; -69..-78% in1 hops.
2. **Row alignment to the endpoint row**, walking outward in x only. This is what turns 13 mean hops into 3-4.
3. **Keep the 50/50 NoC split per bank.** Both DRAM subchannels and both NoC networks stay in use, and it
   halves readers-per-endpoint, which halves the peak floor. Do not put a bank's readers all on one NoC.
4. **Stop optimising in1 peak once at the floor**; spend the remaining degeneracy on ring + reduction.

Wrong (measured, my reasoning did not survive):
5. **NoC by contiguous kk blocks** so each reduction chain crosses the chip once. Consistently WORSE for the
   reduction chain than production's alternating rule (927 vs 732, 1111 vs 901, 527 vs 373 hops). The chain's
   cost depends on the *direction* the chain walks relative to the writer NoC, not on the number of crossings,
   so it has to be an explicit objective term rather than an a priori structure.
6. **Peak-link reweighting on top of the assignment.** Cannot beat the floor; only damages ring/reduction.

Open risk: the reduction chain gets 16% WORSE on the two Sm=2 shapes (776 -> 901 hops) while everything else
improves. That is the stage-2 objective's job and is a reason not to ship stage 1 alone unexamined.

## 5. Why this should convert to wall time better than the cb1 lever did

The cb1-depth experiment showed the in1 read on nsb=1 shapes is bound by latency x concurrency: raising
in-flight bytes 8x bought +7.7% on 256x2048x6144 but COST 3-4.6% on the in0-dominated shapes, because
prefetching harder steals DRAM/NoC bandwidth from the in0 gather. Placement attacks the same product from the
other side - it cuts latency ~3.3x (13.1 -> 4.0 mean hops) - but it *removes* traffic instead of moving it
earlier, so it should not have the same downside on in0-dominated shapes. Indeed the model says the in0 ring
gets cheaper too.

Caveat: this is a static footprint/latency model. It does not prove wall-time gains, and the ring-order
campaign is a standing reminder that predicted link improvements can fail to convert (or invert). The
difference here is that the change is large (-69% hops, -50% peak, both classes improving together) rather
than a re-routing of a fixed amount.

## 6. Proposed next step

Implement `cross/alt` behind a diagnostic bit (host-only, correctness-preserving: it only changes
`P.cores[i].coord`, exactly like the existing `place_m_split_workers`), A/B on these five shapes, and only
then add stage 2 (ring + reduction refinement within the in1-optimal set) and the ring re-order on top.

## Reproduction

```
python3 tools/mm_sweep/picker_gen/in1_place_search.py 16 192 72 1 12 1 2 1   # Mt Kt Nt Ns Pk Sm kb nsb
```
