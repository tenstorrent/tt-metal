# 63-shape corpus A/B of the link-balanced in0 ring order (bit10) — DO NOT PROMOTE

Promotion gate for `RING_BALANCED_BG` (diag bit10). Measured at **config=None**, i.e. the DEPLOYED picker
config — what would actually ship. 60-shape Mt<=8 corpus + the 3 golden shapes not already in it (the corpus
is M<=256; the two 512-row shapes where the win was originally measured are Mt=16, outside it). Two relaunches
per shape with the mask order reversed on the second, 2 warmup + 12 timed iterations each, resumable JSONL.

## Verdict

| metric | value |
|---|---|
| shapes measured | 63 |
| **wins >= +2%** | **0** |
| neutral (within +-2%) | 61 |
| **regressions <= -2%** | **2** — `256x15360x768` **-13.65%**, `256x15360x1536` **-10.58%** |
| median / mean | +0.05% / **-0.28%** |
| best | +1.67% (`512x6144x4608`) |
| relative PCC vs mask 0 | 0.99999-1.00014 on every shape (correctness preserved) |
| gate adopted a reorder on | 15 / 63 shapes (decisions bit-identical across relaunches) |

**Not promoted.** Net negative, with two double-digit regressions and no win above the ~1.5% noise floor.

## Why the +6.1% evaporated

The golden four configs are hand-specified and are **not** picker picks:

| shape | deployed picker (Pk,Ns,Sm,kb,nsb) | golden config used in the earlier study |
|---|---|---|
| 512x6144x2304 | Pk12, **Ns1**, Sm1, kb2, nsb1 | Ns2, Pk6, Sm1, kb2, nsb1 |
| 512x6144x4608 | Pk12, **Ns1**, Sm1, kb2, nsb1 | Ns2, Pk6, Sm1, kb4, nsb1 |
| 256x2048x2048 | Pk4, **Ns1**, Sm3, kb2, nsb4 | Ns2, Pk2, Sm3, kb4, nsb4 |
| 256x2048x6144 | Pk4, Ns3, Sm1, kb2, nsb1 | Ns3, Pk2, Sm2, kb2, nsb4 |

The deployed configs are also substantially FASTER: `512x6144x2304` runs **133.3 us** at the picker's config
vs **170.5 us** at the golden config (-22%). Production already exploits `in0 delivered bytes = 8*Ns*|in0|` by
choosing Ns=1 on the deep shapes, which halves in0 delivery and drops the ring's contribution to the critical
link below the adopt gate. At the deployed config that shape measures **+0.53%** and the gate correctly
declines (predicted balanced peak == production peak, i.e. the search finds no better order at all).

⚠️ **Consequence for the earlier evidence:** the golden-config ablation matrix reported in0 delivery at 35-48%
of the wall. Re-measured at DEPLOYED configs it is materially smaller:

| shape (deployed cfg) | base us | ring-forward skip | in0-read skip |
|---|---|---|---|
| 512x6144x2304 | 133.25 | +17.4% (golden: +30.9%) | +8.8% |
| 512x6144x4608 | 207.40 | +11.4% (golden: +18.5%) | +6.0% |
| 256x15360x768 | 94.96 | **+21.7%** | +6.3% |
| 256x6144x4608 | 141.35 | +5.6% | +3.3% |
| 256x2048x2048 | 37.76 | +4.5% (golden: +13.1%) | +2.4% |

So the in0 ring IS still a real production target (11-22% on deep/narrow shapes), but the golden-config
numbers overstate it by roughly 2x. Future in0 work should be measured at deployed configs.

## Why it regressed: the model is anti-correlated where the ring matters most

Predicted peak-load reduction vs measured outcome, over the adopted shapes:

| shape | ring share of peak link | predicted peak reduction | measured |
|---|---|---|---|
| 256x15360x768 | 43% | **-18.4%** (largest in corpus) | **-13.65%** |
| 256x15360x1536 | 31% | -13.4% | **-10.58%** |
| 128x15360x768 | 30% | -15.0% | -1.24% |
| 256x2048x512 | 37% | -9.6% | +1.49% |
| 64x2048x512 | 28% | -11.4% | +1.50% |
| 512x6144x4608 | 8% | -4.1% | +1.67% |
| 256x6144x2304 | 11% | -5.3% | +0.74% |

On the deep-K / narrow-N family the ordering with the biggest predicted peak improvement is the biggest
actual loss. `256x15360x768` is the corpus shape with the LARGEST ring exposure (+21.7% from skipping the
forward) and the highest ring share of the peak link (43%) — precisely where the model should be most
trustworthy, and it is most wrong.

**Mechanism.** Its shard is 80 KB (W=5, M_block=4, kb=2), above the 64 KB "bandwidth regime" threshold, so
bit10 left the worst-edge cap OFF and the search was free to inflate the worst directed edge. But the ring is
7 SERIAL steps, and per-step time is set by the worst edge across the ring — so an inflated max edge adds to
every step regardless of shard size. The shard-size crossover was the wrong model: **the max-edge term is
always on the critical path.** Production's existing objective (minimize aggmax first, then aggtot) is
therefore already the right one, and every measured win from link balancing came from violating it:

- bit9, no edge cap: +4% at one golden config, -10% on a 24 KB shard, -1.8% on a 128 KB shard.
- bit10, cap conditional on shard size: +6.1% at one golden config, -13.7% at a deployed config.
- bit10 with the cap ON (small shards): neutral everywhere.

## Conclusion: ring ORDERING is exhausted as a lever

Link-load balancing can only be applied *within* the set of orders that tie on max-edge — i.e. as a
tie-break, never as a trade — and in that form it is neutral by construction (the 48 shapes where the gate
declined are all within +-1.5%). There is no measured headroom left in choosing the ring's visiting order.

What remains for in0, given D1's finding that ~30% of the forward cost survives even at ~1 hop (so
topology alone cannot get below it):

1. **Byte-reducing levers** — cross-Ns dedup / multicast / the two-phase bisection-minimal all-gather. Note
   production already picks Ns=1 on most deep shapes, which removes the cross-Ns opportunity there; the 11
   corpus shapes with Ns>=2 are where dedup could still apply.
2. **in1 prefetch depth** (`cb1` = 4 blocks; the in1 DRAM stream is paced by compute which is paced by the
   gather) — independent of everything above, and now the highest-value untested item.

## Reproduction

```
export TT_METAL_DEVICE_PROFILER=1 ARCH_NAME=blackhole
python3 tools/mm_sweep/picker_gen/corpus_ab.py --relaunches 2      # full 63-shape A/B (~20 min, resumable)
python3 tools/mm_sweep/picker_gen/corpus_ab.py --summary
python3 tools/mm_sweep/picker_gen/corpus_ab_worker.py 256 15360 768 0,4,1 0   # deployed-config ring exposure
```
Raw: `results_v2/corpus_ab.jsonl` (one record per shape per relaunch, incl. the factory's balance decision).
