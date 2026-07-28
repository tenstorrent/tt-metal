# in1-optimal placement (diag bit12) measured: the in1 read is already DRAM-bound

Implements the stage-1 CROSS placement from `IN1_PLACE_SEARCH.md` as diag bit12 (host-only, only writes
`P.cores[i].coord`; supersedes the IN1_NEAR pass and re-runs the ring order on the new coords). PCC-clean on
every shape (bit-exact on Sm=1; PCC-identical on Sm=2, where the ring order recomputes and re-associates the
K sum). Offline it cuts in1 read hops 69-78% and puts peak in1 link load exactly on the endpoint-egress floor.

Also found while implementing: **`IDevice::get_optimal_dram_bank_to_logical_worker_assignment(NOC)` caches its
result in a single member without keying on the NoC** (`device.cpp`: `if
(optimal_dram_bank_to_logical_worker_assignment_.empty())`), so the second call returns the first NoC's answer
and `opt1 == opt0`. That is the mechanical root cause of production placing NOC_1 readers at NOC_0-optimal
cores. The kernel-side `dram_bank_to_noc_xy` table *is* built per NoC, so the endpoints really do differ.

## Four isolations, both placements (median us, 2 relaunches with mode order reversed)

| shape | cfg (Pk,Ns,Sm,kb,nsb) | FULL OP prod -> in1opt | mask 21: in1+compute+out (out is Pk-fold) | mask 5: in1+compute+red+out | **mask 53: in1+compute ONLY** |
|---|---|---|---|---|---|
| 512x6144x2304 | (12,1,1,2,1) | 134.45 -> 144.39 **-7.4%** | 193.80 -> 191.00 +1.4% | 97.6 -> 102.9 -5.5% | 73.10 -> 73.11 **+0.0%** |
| 512x6144x4608 | (12,1,1,2,1) | 207.26 -> 227.58 **-9.8%** | 409.25 -> 392.82 +4.0% | 172.0 -> 188.4 -9.5% | 144.9 -> 144.9 **+0.0%** |
| 256x15360x768 | (6,1,2,2,3) | 95.13 -> 80.70 **+15.2%** | 55.69 -> 54.46 +2.2% | 56.3 -> 57.1 -1.5% | 50.40 -> 47.98 **+4.8%** |
| 256x6144x4608 | (6,1,2,4,2) | 141.33 -> 139.24 +1.5% | 180.98 -> 186.85 -3.2% | 126.9 -> 126.4 +0.4% | 114.2 -> 112.7 **+1.3%** |
| 32x6144x1536 | (6,1,1,4,2) | 40.50 -> 41.04 -1.4% | 39.55 -> 39.30 +0.6% | 39.9 -> 40.2 -0.7% | 37.6 -> 37.4 **+0.5%** |
| 256x2048x2048 | (4,1,3,2,4) | 37.88 -> 36.91 +2.6% | 49.23 -> 41.78 +15.1% | 34.6 -> 34.4 +0.6% | 27.4 -> 25.8 **+5.9%** |
| 256x2048x6144 | (4,3,1,2,1) | 92.53 -> 99.88 **-7.9%** | 146.39 -> 120.30 +17.8% | 80.3 -> 84.0 -4.6% | 51.3 -> 51.3 **+0.0%** |

Mask semantics: 21 = in0 read + in0 ring forward + reduction skipped; 5 = in0 read + forward skipped
(reduction kept, so output is written ONCE); 53 = 21 + output skipped, i.e. the pure in1 isolation.

**Caveat on mask 21 (the configuration originally asked for):** `SKIP_REDUCTION` makes every split-K band
write its own partial to the same output pages, so output traffic is `Pk * M*N*2`, not `M*N*2`. At Pk=12 that
is half the isolated bytes and the mode is output-dominated, which is why its walls exceed the full op's and
why its two big "wins" (+15.1%, +17.8%) do not reproduce in mask 5 or mask 53. Those wins are the output-write
path, not in1.

## DRAM utilisation (bytes moved / wall, vs 512 GB/s)

| shape | mask 53 in1-read util prod -> in1opt | mask 5 (in1+out) util | mask 21 (in1 + Pk*out) util |
|---|---|---|---|
| 512x6144x2304 | **76% -> 76%** | 61% -> 58% | 57% -> 58% |
| 512x6144x4608 | **76% -> 76%** | 70% -> 64% | 54% -> 56% |
| 256x15360x768 | **91% -> 96%** | 83% -> 82% | 91% -> 93% |
| 256x6144x4608 | **97% -> 98%** | 91% -> 91% | 76% -> 74% |
| 32x6144x1536 | **98% -> 99%** | 93% -> 92% | 96% -> 97% |
| 256x2048x2048 | **60% -> 63%** | 54% -> 54% | 50% -> 59% |
| 256x2048x6144 | **96% -> 96%** | 68% -> 66% | 50% -> 61% |

## Conclusion: the in1 read is not NoC-limited, so placement cannot pay there

**In isolation the in1 read already runs at 76-98% of peak DRAM bandwidth under the PRODUCTION placement.**
Cutting its NoC hops by 69-78% and halving its peak link load therefore buys **+0.0% to +5.9%**, and only on
the two shapes that were not already at the wall (256x15360x768 at 91%, 256x2048x2048 at 60%). On the two
512x6144 shapes it buys exactly nothing despite 76% util - they are limited by something else in the read
pipeline (nsb=1 means single 2 KB tile reads, so burst size / issue rate, consistent with the cb1-depth result
and the "SP1 read is burst-size bound" record).

So the geometric finding ("66-75% of in1 read link traffic is avoidable wrap-around") is TRUE but nearly
worthless: the wasted hops were not on the critical path. in1 is not the tall nail for placement.

**And in the full op the placement is net NEGATIVE on three shapes (-7.4%, -9.8%, -7.9%)** because it trades
in0-delivery and reduction geometry for an in1 gain that does not exist. The one large full-op win,
`256x15360x768 +15.2%`, is the shape with the largest in0 ring exposure (+21.7%), and the offline model
predicted its ring hops dropping 683 -> 526 (-23%) - so that win comes from the RING getting shorter as a side
effect, not from in1.

## What this redirects to

1. **Placement does pay - through the in0 ring, not in1.** +15.2% on 256x15360x768 came from ring geometry.
   The right objective is therefore to place for the in0 ring + reduction (with in1 as a constraint only where
   its util is below ~80%), which is the inverse of the priority we assumed.
2. **The remaining in1 lever is request size / concurrency, not distance**: the 512x6144 shapes sit at 76%
   util with nsb=1 single-tile reads and do not respond to shorter paths, while cb1 depth (concurrency) bought
   +7.7% on a similar shape. Coalescing in1 into larger bursts is the untested candidate.
3. bit12 must not be promoted as-is.

## Reproduction

```
export TT_METAL_DEVICE_PROFILER=1 ARCH_NAME=blackhole
python3 tools/mm_sweep/picker_gen/in1_isolated_ab.py --relaunches 2       # full op + mask 21
W=tools/mm_sweep/picker_gen/corpus_ab_worker.py
python3 $W 512 6144 2304 5,4101 0     # in1+compute+reduction+output, prod vs bit12
python3 $W 512 6144 2304 53,4149 0    # in1+compute only, prod vs bit12
```
