# Regime-A matmul hyperoptimization log

Plain-English running log. Every entry says what was tried, exactly which shapes moved and by how much, and
whether it shipped. Numbers are median kernel wall time in microseconds at the DEPLOYED picker config
(`config=None`), measured with 2 warmup + 12 timed iterations and at least 2 relaunches with the mode order
reversed. No change here reduces precision, fidelity or correctness: everything stays BF16 in/out, HiFi2,
FP32 accumulation, and every candidate is PCC-checked against a CPU FP32 reference.

## Reference shapes and their roofline headroom

"DRAM floor" is all DRAM bytes (in0 read + in1 read + output write) divided by 512 GB/s. "% of floor" is how
close the shape already runs to that floor - the higher the number, the less headroom is left. Exposure
columns are what we measured by ablation: how much faster the kernel gets if that stage is deleted.

| shape | wall (start of log) | DRAM floor | % of floor | in1 read exp | in0 ring exp | in0 read exp | headroom verdict |
|---|---|---|---|---|---|---|---|
| 512x6144x2304 | 134.0 | 72.2 | 54% | +5.4% | +17.4% | +9.8% | LOTS - NoC bound |
| 512x6144x4608 | 207.5 | 132.1 | 63% | +5.0% | +11.4% | +6.8% | lots - NoC bound |
| 256x2048x6144 | 92.2 | 57.3 | 62% | +36.9% | +6.9% | +6.8% | lots - in1 latency |
| 256x15360x768 | 95.1 | 62.2 | 65% | +14.2% | +21.7% | +7.4% | lots - ring bound |
| 256x2048x2048 | 37.9 | 20.5 | 54% | +19.9% | +5.0% | +3.8% | lots, but small shape |
| 256x6144x4608 | 141.4 | 121.3 | 86% | +32.1% | +5.2% | +3.4% | little - near DRAM wall |
| 32x6144x1536 | 40.5 | 37.8 | 93% | +70.7% | +0.2% | +1.2% | none - at DRAM wall |

Priority follows headroom: the four "lots" shapes are where work should go. `32x6144x1536` and
`256x6144x4608` are close to the DRAM bandwidth limit, so almost nothing can help them and a change that
lengthens their DRAM paths will hurt them badly.

## Shipped so far

### S1. Fixed a tt-metal bug: DRAM-worker assignment cache ignored the NoC (commit 8bf29bc2e5b)

`get_optimal_dram_bank_to_logical_worker_assignment(NOC)` cached one answer and returned it for both NoCs, so
the op placed its NOC_1 readers at the NOC_0-optimal core. The two answers really are different on Blackhole
because each DRAM bank exposes a different sub-endpoint per NoC. Keyed the cache by NoC.

Effect (3 samples each, spread under 0.5%):

| shape | before | after | change |
|---|---|---|---|
| 512x6144x2304 | 134.0 | 123.3 | **8.0% faster** |
| 512x6144x4608 | 207.5 | 198.2 | **4.5% faster** |
| 256x15360x768 | 95.1 | 95.8 | 0.7% slower |
| 256x2048x6144 | 92.2 | 92.7 | 0.6% slower |
| 256x6144x4608 | 141.4 | 143.6 | 1.5% slower |
| 32x6144x1536 | 40.5 | 41.3 | 1.9% slower |
| 256x2048x2048 | 37.9 | 39.3 | 3.7% slower |

Kept it: the API should not lie about the NoC, and the two big shapes gain a lot. The small regressions are a
side effect of half of each bank's cores moving to a second, correct target row.

### S2. 2D bank-by-slice mesh placement - SHIPPED as production default, gated

Ring traffic wants the 8 cores of one slice close together; the reduction chain wants the Pk cores of one bank
close together. Those are different groupings, so clustering cannot help both. Putting banks along the x axis
and slices along the y axis makes both a single hop. Offline: ring hops down 70%, reduction hops down 19-40%,
peak link load down 11-15%.

Effect (2 relaunches, all PCC-clean), production placement vs mesh:

| shape | production | mesh | change |
|---|---|---|---|
| 256x2048x6144 | 92.7 | 80.3 | **13.4% faster** |
| 512x6144x2304 | 123.9 | 112.3 | **9.4% faster** |
| 256x15360x768 | 95.9 | 88.1 | **8.2% faster** |
| 512x6144x4608 | 197.8 | 185.1 | **6.4% faster** |
| 256x2048x2048 | 39.1 | 41.4 | 5.9% slower |
| 32x6144x1536 | 41.2 | 49.1 | 19.4% slower |
| 256x6144x4608 | 143.5 | 175.7 | 22.5% slower |

Wins where the ring is exposed; loses where the shape is already at the DRAM wall (those two shapes have
almost no ring exposure, so they only pay the cost of moving cores away from their DRAM bank).

**Gate fitted on all 63 corpus shapes.** Measuring every shape both ways showed the deciding factor is whether
the mesh FILLS the grid. `preaders` (= Pk x Ns x Sm) is the number of slices, and the mesh puts one slice per
grid row. When `preaders >= 10` the cores spread over the whole 11x10 array; when it is smaller they all get
packed into the top few rows and every DRAM path piles into one corner:

| preaders | shapes | mean change | best | worst |
|---|---|---|---|---|
| 1-5 | 7 | **69.6% slower** | -48% | -89% |
| 6-7 | 14 | 16.2% slower | -6% | -23% |
| 8-9 | 7 | 2.3% slower | +10% | -11% |
| 10-13 | 35 | 1.7% faster | +15% | -22% |

Shipped gate: `preaders >= 10 AND Sm == 1 AND (Ns == 1 OR Pk >= 4)`. It adopts **24 of 63 shapes, mean 5.06%
faster, best 14.98%, worst 1.27% slower**. Every declined shape keeps exactly the old placement. Verified
after shipping (mask 0 vs bit14 which forces the old placement):

| shape | shipped | old placement | gain |
|---|---|---|---|
| 256x6144x768 | 35.72 | 42.01 | **14.98%** |
| 256x2048x6144 | 81.28 | 94.53 | **14.02%** |
| 512x6144x2304 | 112.1 | 124.0 | **9.6%** |
| 512x6144x4608 | 185.2 | 198.0 | **6.5%** |
| 128x2048x1536 | 20.79 | 22.16 | **6.2%** |
| not gated (256x2304x6144, 32x6144x1536, ...) | unchanged | unchanged | 0 (+-1.5% noise) |

111/111 correctness tests pass. Diagnostic bit13 forces the mesh on for shapes outside the gate, bit14 forces
it off so the default can always be A/B'd.

### Note on task 3 (ring ORDER on the mesh): closed by analysis, no experiment needed

On the mesh, the 8 cores of a slice sit in one row at x=0..7. Each NoC only moves one way and wraps, so ANY
directed cycle through 8 cores in a row costs exactly one full lap of the x dimension (17 hops) - every visiting
order that goes in increasing x order costs the same, and any other order costs more. So the ring order has no
freedom left to exploit on the mesh; the existing optimiser already picks a minimal order. Worth recording:
60% of that 17-hop cost is the single wrap-around edge that closes the cycle (10 of 17 hops), which is a
future target - a two-direction ring using both NoCs would avoid it.

## Work queue

1. ~~Fit an adoption gate for the mesh and ship it.~~ DONE (S2).
2. ~~Re-run ring ORDER optimisation on the mesh.~~ CLOSED by analysis (see note above).
3. Mesh v2: spread slices evenly across rows instead of packing rows 0..preaders-1, so shapes with fewer than
   10 slices can also use it. Those are 28 of 63 corpus shapes and they currently lose 6-89% with the mesh, so
   the upside is large. [next]
4. Revisit the three regressions from S1.
5. Then keep going where headroom remains.
