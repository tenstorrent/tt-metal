# SKIP_IN1_READ ablation at DEPLOYED configs (diag bit11)

Sizes the prize before any in1 placement work. in1 is the largest DRAM consumer and was the one term the
22-mode critical-path matrix never ablated.

New diagnostic bit11 `SKIP_IN1_READ` drops ONLY the in1 DRAM read payload in `in1_reader.cpp` (both the
coalesced-contiguous branch and the per-row fallback). Preserved: CB reserve/push/pop, the rotated shard
order, `noc_async_read_barrier`, the K-tail `zero_l1`, M-split reader->slave forwarding, semaphores, and all
downstream compute / reduction / output work. Blocks keep stale L1, so output is intentionally invalid
(verified: relative PCC vs mask 0 = 0.056, i.e. the read really is gone; mask 0 unaffected; no hang).

Measured at **config=None** (deployed picker config), 2 warmup + 12 timed iterations, two relaunches with the
mode order reversed. Relaunch agreement is within 1pp on every shape.

## Results

| shape | wall us | in1 MB | DRAM floor us | %peak | **in1 read exp** | in1 exp us | slack us | ADDR us | in0 ring | in0 read |
|---|---|---|---|---|---|---|---|---|---|---|
| 512x6144x2304 | 134.1 | 28.3 | 72.2 | 54% | **+5.4%** | 7.2 | 61.9 | 7.2 | +17.9% | +9.8% |
| 512x6144x4608 | 208.6 | 56.6 | 132.1 | 63% | **+5.0%** | 10.4 | 76.5 | 10.4 | +11.7% | +6.8% |
| 256x15360x768 | 95.1 | 23.6 | 62.2 | 65% | **+14.2%** | 13.5 | 32.9 | 13.5 | +21.4% | +7.4% |
| 256x6144x4608 | 141.4 | 56.6 | 121.3 | 86% | **+32.1%** | 45.4 | 20.1 | 20.1 | +5.2% | +3.4% |
| 32x6144x1536 | 40.6 | 18.9 | 37.8 | 93% | **+70.7%** | 28.7 | 2.8 | 2.8 | +0.2% | +1.2% |
| 256x2048x2048 | 38.0 | 8.4 | 20.5 | 54% | **+19.9%** | 7.6 | 17.6 | 7.6 | +5.0% | +3.8% |
| 256x2048x6144 | 91.8 | 25.2 | 57.3 | 63% | **+36.9%** | 33.9 | 34.4 | 33.9 | +6.9% | +6.8% |

`DRAM floor` = all DRAM traffic (in0 + in1 + out) / 512 GB/s. `%peak` = floor / wall, i.e. how close the shape
already runs to its own DRAM-bandwidth bound. `slack` = wall - floor. `ADDR` = min(in1 exposure, slack), an
upper bound on the part of the in1 read cost that is NOT irreducible DRAM time.

## Two regimes

1. **Already at the DRAM wall (no placement juice):** `32x6144x1536` (93% of peak) and `256x6144x4608` (86%).
   Their in1 exposure is huge (71%, 32%) but mostly irreducible - 32x6144x1536 has 28.7 us of in1 exposure
   against only 2.8 us of slack, and achieves 466 GB/s on in1 alone. This confirms the existing record
   ("deep-K DRAM-read-bound at 476-498 GB/s, a genuine floor"); placement cannot move it.
2. **Exposed but NOT bandwidth-limited (this is the juice):** the other five run at only 54-65% of their DRAM
   floor with 18-77 us of slack, yet their in1 read is still 5-37% exposed. Something other than DRAM
   bandwidth is costing that time, which is what the placement analysis targets. Best candidates:
   `256x2048x6144` (36.9% exposed, 63% of peak, 33.9 us addressable), `256x2048x2048` (19.9%, 54%),
   `256x15360x768` (14.2%, 65%).

## The two levers are complementary

in1 exposure and in0-ring exposure are ANTI-correlated across the corpus:

- ring-dominated: 512x6144x2304 (ring +17.9% vs in1 +5.4%), 256x15360x768 (+21.4% vs +14.2%)
- in1-dominated: 32x6144x1536 (+0.2% vs +70.7%), 256x2048x6144 (+6.9% vs +36.9%), 256x6144x4608
  (+5.2% vs +32.1%), 256x2048x2048 (+5.0% vs +19.9%)

So neither lever alone covers the corpus, and a shape is rarely limited by both at once. That also explains
why the ring work found so little at deployed configs: the shapes where the ring is exposed are a minority.

## Hypothesis for regime 2: latency x concurrency, not bandwidth

`cb1` holds `4 * kb * N_sub` tiles = **4 blocks**, so a reader can have at most `4 * kb * N_sub * 2 KB` of
in1 in flight. Combined with the read round-trip latency L, throughput per reader is capped at
`bytes_in_flight / L` regardless of DRAM bandwidth:

| shape | kb, nsb | bytes in flight | in1 per reader | implied time at L=2us | measured in1 exp |
|---|---|---|---|---|---|
| 256x2048x6144 | 2, 1 | 16 KB | 256 KB | ~32 us | 33.9 us |
| 256x2048x2048 | 2, 4 | 64 KB | 128 KB | ~4 us | 7.6 us |
| 32x6144x1536 | 4, 2 | 64 KB | 384 KB | ~12 us | 28.7 us (but DRAM-bound at 466 GB/s) |

The nsb=1 shapes issue single-tile (2 KB) reads and can only keep ~16 KB outstanding, so they are bound by
`latency x concurrency`, not bandwidth. This is consistent with the placement finding: L is inflated by the
16-24 hop wrap-around response paths, and 66-75% of that distance is avoidable. Both halves multiply:

- **reduce L** -> direction-aware placement / side-aware NoC (66-75% of the distance is removable)
- **raise concurrency** -> deeper `cb1` (fill leftover L1; it also fills the idle DRAM during the in0 gather)

Caveat: this is an arithmetic consistency check, not a measurement. The competing prior is per-RISC issue
rate (the minimal_matmul record found a ~296 GB/s per-RISC read-pipeline limit for an M=1 pattern). The two
are cheap to separate: deepening `cb1` alone raises concurrency without touching latency or issue rate, so if
the exposure shrinks roughly in proportion, the concurrency half is confirmed.

## Recommended order

1. **cb1 depth** (host-side CB sizing, fill leftover L1, cannot make a config infeasible). Cheapest test of
   the concurrency half, and independently motivated by the idle-DRAM-during-gather argument.
2. **Side-aware NoC / direction-aware placement** (the latency half), evaluated jointly with in0 on the
   whole-op link model, then A/B'd on the regime-2 shapes above.
3. Skip both on regime-1 shapes (`32x6144x1536`-like, >=86% of peak): nothing to win there.

## Reproduction

```
export TT_METAL_DEVICE_PROFILER=1 ARCH_NAME=blackhole
W=tools/mm_sweep/picker_gen/corpus_ab_worker.py
python3 $W 256 2048 6144 0,2048,4,1 1     # base / skip in1 read / skip in0 ring fwd / skip in0 read
```
