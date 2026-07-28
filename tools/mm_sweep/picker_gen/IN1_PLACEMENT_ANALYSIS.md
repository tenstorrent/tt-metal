# in1 read: core placement analysis (multi-reader-per-bank)

Static analysis of the CURRENT placement from the in1-read point of view, at DEPLOYED picker configs.
Tool: `in1_placement_probe.py` (replicates build_plan's find_near + place_m_split_workers, uses the
device-validated route model from `ring_link_model.py`). No matmul runs; this is geometry + traffic accounting.

## 1. How the current placement works

`build_plan` (regime_a_matmul_plan.hpp), one pass, bank-major / slice-minor:

```
for b in 0..7:
  for p in 0..preaders-1:                       # preaders = Pk * Ns * Sm
    i   = b*preaders + p
    noc = (Sm > 1) ? ((p / Sm) & 1) : (p & 1)    # alternate reader NoC for balance
    tgt = noc ? opt1[b] : opt0[b]                # bank-adjacent worker from the tt-metal API
    coord = find_near(tgt)                       # expanding Manhattan spiral, FIRST-FIT, marks used
```

`find_near` walks d = 0,1,2,... and within each d walks dx = -d..+d, dy = +rem then -rem, taking the first
free in-grid cell. Then for Sm>1, `place_m_split_workers` overrides coords: pass 1 re-places every mm==0
reader with the same spiral (readers first), pass 2 puts each slave on the free core minimising the directed
reader->slave hop on the reader's NoC (IN1_NEAR).

Provenance / how "optimal" was determined:

- `get_optimal_dram_bank_to_logical_worker_assignment(noc)` is a tt-metal API returning ONE optimal worker
  per bank. It is built for the one-reader-per-bank case.
- For preaders > 1 per bank, nothing was optimised: we spiral outward from that single point and take
  whatever is free. There is no direction awareness, no per-bank distance equalisation, no spreading across
  the bank's NoC endpoints, and no choice of which slice index lands on which core.
- The ONLY placement lever ever measured is IN1_NEAR for M-split slaves (-6.3/-7.2% on the sole production
  Sm=2 shape). Multi-reader-per-bank placement itself has never been A/B'd.
- On BH p150b `opt0 == opt1 == [(0,9),(0,0),(0,7),(0,3),(6,9),(6,1),(6,6),(6,4)]`, so the `noc` term in
  `tgt` is a no-op: NOC_0 and NOC_1 readers of a bank are placed in the SAME spiral around the SAME point.

## 2. Three hardware facts the placement does not account for

1. **Each BH DRAM channel has THREE NoC endpoints (subchannels).** `dram_views.worker_endpoint = [a, b]`
   selects subchannel a for NOC_0 and b for NOC_1, so a core's endpoint depends on which NoC it reads on,
   and the two endpoints are at DIFFERENT rows. One subchannel per bank is never used:

   | bank | NOC_0 endpoint (phys) | NOC_1 endpoint (phys) | unused |
   |---|---|---|---|
   | 0 | (0,11) | (0,1) | (0,0) |
   | 1 | (0,2) | (0,10) | (0,3) |
   | 2 | (0,9) | (0,4) | (0,8) |
   | 3 | (0,5) | (0,7) | (0,6) |
   | 4 | (9,11) | (9,1) | (9,0) |
   | 5 | (9,3) | (9,10) | (9,2) |
   | 6 | (9,8) | (9,4) | (9,9) |
   | 7 | (9,6) | (9,7) | (9,5) |

2. **Each NoC is strictly unidirectional per dimension, with torus wrap.** NOC_0 = +x/+y, NOC_1 = -x/-y.
   Device-verified: logical (3,0)->(3,1) is **1 hop on NOC_0 and 11 hops on NOC_1**; (0,5)->(1,5) likewise
   flips. So a core placed ONE cell in the wrong direction pays a full lap (11 in y, 16 in x on the 17x12
   torus), not one hop.

3. **A read's DATA travels from the DRAM endpoint to the worker on the same NoC**, so the response path
   length depends on which side of the DRAM column the worker sits. DRAM columns are physical x=0
   (banks 0-3) and x=9 (banks 4-7); workers are at physical x=1..6 and 11..15. Therefore:
   - banks 0-3, NOC_0 (+x from x=0): cheap for workers at x=1..6, expensive (11-16 hops) for x>=11.
   - banks 0-3, NOC_1 (-x from x=0): wraps immediately; cheap only for x=15,14,... i.e. the RIGHT half.
   - banks 4-7, NOC_0: cheap for x=11..15; banks 4-7, NOC_1: cheap for x=6..1.

   The optimal layout is therefore a CROSS: each bank's NOC_0 readers on one side of its DRAM column and its
   NOC_1 readers on the other. The current code puts both in the same spiral on the same side.

## 3. What the current placement actually produces

512x6144x2304 at its deployed config (Ns1, Pk12, Sm1; 96 cores, 12 readers per bank, 288 KB in1 per reader).
Response path length, DRAM endpoint -> reader, on that reader's NoC:

| bank | NOC_0 readers (hops) | NOC_1 readers (hops) |
|---|---|---|
| 0 | 1, 2, 9, 10, 13, 14 | 15, 16, 19, 19, 20, 20 |
| 1 | 1, 2, 3, 4, 4, 5 | 20, 21, 21, 22, 22, 23 |
| 2 | 3, 5, 5, 11, 14, 14 | 18, 20, 21, 23, 24, 24 |
| 3 | 3, 4, 7, 14, 15, 16 | 13, 13, 14, 14, 16, 17 |
| 4 | 2, 4, 11, 12, 13, 13 | 5, 6, 7, 16, 17, 18 |
| 5 | 2, 3, 3, 4, 4, 25 | 9, 10, 12, 20, 22, 23 |
| 6 | 3, 5, 14, 14, 24, 25 | 12, 21, 22, 24, 24, 25 |
| 7 | 7, 7, 14, 14, 15, 16 | 12, 12, 13, 13, 14, 15 |

- **NOC_0 readers mean 9.0 hops; NOC_1 readers mean 17.2 hops.** NOC_1 is half the readers but **66% of the
  in1 read hop-bytes**.
- Even on NOC_0 the spread is extreme (bank 6: 3, 5, 14, 14, 24, 25). The 9-25 hop entries are cores the
  direction-blind spiral placed on the wrong side in y: +y wrap costs 11 hops, so "one row above the target"
  is 11 hops, not 1.
- Busiest in1-only link is 3.5 MB and every one of the top 8 is a **NOC_1 x-link in the middle columns**
  (phys x=7,8,9) - i.e. wrap-around traffic crossing the chip.

## 4. How much juice

Total in1 read response hops summed over all DRAM readers (Sm>1: only mm==0 cores read), and the same in
MB-hops of link traffic. (C) and (D) are ACHIEVABLE greedy variants that honour cell collisions and reserve
the IN1_NEAR slave cells; (B) is an unconstrained lower bound.

| deployed config | current | (A) side-aware NoC, same placement | (C) direction-aware placement | (D) both | (B) lower bound |
|---|---|---|---|---|---|
| 512x6144x2304 Ns1/Pk12/Sm1 | 1260 hops, 372 MB-hops | 797, -37% | 408, **-68%** | 401, -68% | 144, -89% |
| 256x15360x768 Ns1/Pk6/Sm2 | 510 hops, 251 MB-hops | 284, -44% | 175, **-66%** | 155, -70% | 72, -86% |
| 256x6144x4608 Ns1/Pk6/Sm2 | 510 hops, 602 MB-hops | 284, -44% | 175, **-66%** | 155, -70% | 72, -86% |
| 32x6144x1536 Ns1/Pk6/Sm1 | 510 hops, 201 MB-hops | 284, -44% | 146, **-71%** | 140, -73% | 72, -86% |

Scale relative to the thing I have been optimising (in0 ring payload x ~3.6 hops):

| shape | in0 ring | in1 read | ratio |
|---|---|---|---|
| 512x6144x2304 | 159 MB-hops | 372 MB-hops | **2.3x** |
| 256x15360x768 | 198 MB-hops | 251 MB-hops | 1.3x |
| 256x6144x4608 | 79 MB-hops | 602 MB-hops | **7.6x** |
| 32x6144x1536 | 10 MB-hops | 201 MB-hops | **20.2x** |

So in1 reads generate 1.3x to 20x the in0 ring's link traffic, and roughly two thirds of it is avoidable
distance rather than payload.

## 5. Why in0 and in1 must be optimised together

- Flipping a core's reader NoC also flips its WRITER NoC (the split-NoC design forces them opposite), so
  variant (A) moves no cores but changes the in0 ring's routing direction, the reduction chain and the output
  writes for every affected core. It also unbalances the writer NoC (e.g. all left-bank readers on NOC_0
  puts all their in0 ring traffic on NOC_1).
- Variants (C)/(D) move cores, which changes ring edges and the reduction chain (`red_next = same bank,
  kk+1`, currently intra-blob and short - worth preserving).
- The vehicle already exists: `ring_link_model.py` charges in1 reads, the in0 own-shard read, the reduction
  chain and output writes onto one validated link map. It needs the placement/NoC assignment opened up as
  free variables instead of only the ring order.
- This also explains the ring-balancing result: the "background" my ring optimiser was routing around is
  mostly in1 wrap-around traffic, i.e. self-inflicted and removable, not fundamental. Removing it is worth
  more than routing around it.

## 6. Honest limits of this analysis

- It is a static link-footprint count. It does NOT prove wall-time gains. The in1 read may be
  DRAM-bandwidth-bound rather than NoC-bound; prior tt-npe work on minimal_matmul found link congestion
  non-gating (0.02% impact) even with a 731% peak-demand link.
- What it does establish: a large avoidable footprint, and a **3x latency asymmetry** between NOC_0 and
  NOC_1 readers. Latency matters here because `cb1` is only 4 blocks deep, which caps requests in flight.
- The ring corpus result is a cautionary precedent: predicted link-footprint improvement was ANTI-correlated
  with measured delta for the ring. The situation differs (here we would remove 60-70% of a footprint rather
  than re-route a fixed amount) but the lesson holds - measure on 2-3 shapes before building the optimiser.
- The DRAM endpoint row is taken from the soc descriptor and the response path is modelled with the same
  dimension-ordered rule validated for worker-to-worker pairs (0/277 mismatches); it is not directly
  measured for DRAM-to-worker pairs.

## 7. Proposed order of work

1. **Add a SKIP_IN1_READ diagnostic bit** and measure in1-read exposure at deployed configs on 4-6 shapes.
   This sizes the total prize; without it any placement work is blind. (The 22-mode ablation matrix never had
   an in1 bit - the largest DRAM consumer is the one term never ablated.)
2. **Variant (A), side-aware NoC assignment** - host-only, ~20 lines, moves no cores, -37..-44% of in1
   footprint. Evaluate on the whole-op model first (it changes writer-NoC balance), then A/B.
3. **Variant (C), direction-aware find_near** - replace the direction-blind spiral with one that walks
   downstream of the endpoint in the NoC's direction. -66..-71%. Moves cores, so re-check ring + reduction.
4. The unused third subchannel per bank is a real hardware resource with no metal API to select it; note it,
   do not plan on it.

## Reproduction

```
python3 tools/mm_sweep/picker_gen/in1_placement_probe.py 16 192 72 1 12 1 2 1   # Mt Kt Nt Ns Pk Sm kb nsb
python3 tools/mm_sweep/picker_gen/in1_placement_probe.py 8 480 24 1 6 2 2 3
```
