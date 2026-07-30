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

### S3. Gate extension: also adopt when the ring simply carries more traffic than in1

Added a second clause: adopt the mesh whenever ring bytes >= 2x in1 bytes, regardless of slice count. On the
corpus exactly 3 shapes clear 2x and all 3 win; the highest-ratio loser sits at 1.31x, so the threshold is
clean. This lets in the two ring-heavy shapes that the slice-count clause excluded because they use M-split:

| shape | shipped | old placement | gain |
|---|---|---|---|
| 256x2048x512 | 15.10 | 16.87 | **10.47%** |
| 256x15360x768 | 88.24 | 95.92 | **8.01%** |
| declined: 256x6144x4608, 256x2048x2048, 256x6144x1536 | unchanged | unchanged | 0 (noise) |

Gate now adopts 26 of 63 shapes, mean about 5.4% faster.

### F1. FAILED: mesh v2, spreading slices evenly over the rows (bit15, kept as a diagnostic)

Idea: shapes with fewer than 10 slices get packed into the top rows, so space them out over all 10 rows
instead. It helps a little but nowhere near enough - every one of these shapes is still far worse than
production:

| shape | production | mesh packed | mesh spread |
|---|---|---|---|
| 32x2048x2048 | 20.16 | 29.85 (-48.1%) | 28.36 (-40.7%) |
| 32x6144x1536 | 41.03 | 49.36 (-20.3%) | 46.54 (-13.4%) |
| 64x6144x4608 | 119.35 | 144.08 (-20.7%) | 141.13 (-18.2%) |
| 32x6144x2304 | 60.56 | 114.27 (-88.7%) | 115.97 (-91.5%) |
| 64x6144x1536 | 45.07 | 77.38 (-71.7%) | 79.48 (-76.3%) |

Why: with few slices there are few cores, so each core reads a LOT of in1 (1.18 MB per core on
32x6144x2304). Moving those cores off their own DRAM bank multiplies the read latency, and these shapes are
already at 83-95% of their DRAM floor, so there is nothing to win and everything to lose. Conclusion: the mesh
belongs only to the many-core, ring-heavy regime, which is what the gate already says. Not pursuing further.

### Where the time goes now (measured after the mesh shipped)

Percentages are how much faster the kernel gets if that stage is deleted. The mesh did its job: ring exposure
collapsed from 17.4% to 3.5% on 512x6144x2304 and from 21.7% to 0.8% on 256x15360x768.

| shape | wall | % of DRAM floor | in0 read | ring | compute | output | in1 read |
|---|---|---|---|---|---|---|---|
| 512x6144x4608 | 185.2 | 71% | +7.5 | +2.0 | **+9.8** | +0.8 | +4.0 |
| 512x6144x2304 | 112.2 | 64% | +12.2 | +3.5 | **+13.5** | +1.3 | +5.3 |
| 256x15360x768 | 87.7 | 71% | +18.1 | +0.8 | +1.3 | +0.9 | **+35.4** |
| 256x2048x6144 | 79.8 | 72% | +8.2 | +0.2 | -0.5 | +14.8 | **+33.0** |
| 256x6144x4608 | 143.3 | 85% | +5.5 | +6.3 | +1.0 | +3.0 | **+35.4** |
| 256x2048x2048 | 39.9 | 51% | +3.4 | +7.4 | +3.4 | +12.8 | **+17.6** |

So the ring is done. What is left is the in1 read (3 shapes), compute (the two 512-row shapes), the in0 read,
and the output write on the two 2048-K shapes.

### F2. FAILED: nsb (N sub-block width) re-tuning, mostly

Compute and output-write exposure both pointed at the picker choosing nsb=1, which forces a 1-tile-wide
subblock and single-page output writes. Swept nsb at the deployed config with the mesh active:

| shape | nsb=1 | nsb=2 | nsb=3 | nsb=4 | nsb=8 | best |
|---|---|---|---|---|---|---|
| 512x6144x2304 | **112.7** | 135.8 | 147.4 | - | - | picker right (nsb=1) |
| 512x6144x4608 | **185.3** | 200.0 | 223.6 | - | - | picker right (nsb=1) |
| 256x2048x6144 | 79.8 | **76.3** | - | 78.0 | 100.6 | nsb=2 is **4.4% faster** |
| 256x6144x4608 | - | **143.6** | 145.4 | - | 169.5(9) | picker right (nsb=2) |
| 256x2048x2048 | - | 40.9 | - | 39.1 | **38.7** | nsb=8 1.2% faster (noise-level) |
| 256x15360x768 | 108.4 | - | **87.6** | - | - | picker right (nsb=3) |

So the picker is right on 4 of 6; one real win (256x2048x6144, 4.4% at nsb=2) is parked as a candidate table
entry rather than a rule change.

### F3. FAILED: cb1 depth no longer helps after the mesh

Before the mesh, deepening the in1 buffer bought 7.7% on 256x2048x6144. Re-measured after the mesh:

| shape | depth 8 | depth 16 | depth 32 |
|---|---|---|---|
| 256x2048x6144 | +0.9% | +1.3% | +1.6% (was +7.7%) |
| 256x6144x4608 | +2.5% | +2.9% | +2.0% |
| 256x15360x768 | +0.7% | -0.1% | +0.3% |
| 512x6144x2304 | -0.9% | -2.9% | -4.0% |
| 512x6144x4608 | -0.5% | -1.6% | -2.5% |

The mesh and cb1 depth were attacking the same cost (in1 read latency), and the mesh did it better. Closing
this lever.

### S4. SHIPPED: bigger compute subblock when the old sizer wasted half the registers

With fp32 accumulation the compute unit can hold 4 result tiles at once. The old sizer capped the subblock
height at 2, so whenever the N sub-block is 1 tile wide it used a 2x1 subblock - only 2 of the 4 register
slots. Now such a subblock is enlarged to the biggest area that still fits (4x1 when M_block divides by 4),
which halves the number of matmul calls for exactly the same arithmetic. **Verified bit-exact** (identical
output bytes), so this cannot change numerics.

Important detail: it only ever ENLARGES. An earlier version also re-shaped subblocks that were already at 4
tiles (2x2 -> 1x4), which cost 2.22% on 256x2048x1024 for no reason. Shapes already at 4 tiles are now
untouched by construction.

Corpus result (63 shapes, 2 relaunches): mean +0.05%, and the gain is concentrated exactly where predicted:

| shape | legacy sizer | shipped | gain |
|---|---|---|---|
| 512x6144x4608 | 184.99 | 180.46 | **2.45%** |
| 512x6144x2304 | 112.56 | 109.97 | **2.31%** |
| 256x6144x768 | 35.99 | 35.63 | 0.99% |
| everything else | - | - | within noise |

111/111 correctness tests pass. Diagnostic bit16 restores the legacy sizer for A/B.

### Measurement note: the small 2048-K shapes are noisy

While checking S4 I measured two shapes whose programs are provably IDENTICAL between the two arms
(256x2048x1024 and 256x2048x2048, because their old subblock was already 4 tiles) and got +2.39% and -1.86%.
So the noise floor on those small shapes is about +-2.4%, not the +-1.5% seen on the big ones. Any claim under
about 3% on a small 2048-K shape needs 3 or more relaunches.

### Roofline redone with a measured COMPUTE floor (the number we were missing)

Running compute alone (everything else deleted) gives the compute floor. Until now we only had the DRAM floor,
which made two shapes look like they had more headroom than they do.

| shape | wall | compute floor | DRAM floor | real floor | headroom left | limited by |
|---|---|---|---|---|---|---|
| 512x6144x4608 | 180.2 | **139.2** | 132.1 | 139.2 | 41.0 us | **compute** |
| 512x6144x2304 | 110.3 | 70.0 | 72.2 | 72.2 | 38.1 us | both, nearly tied |
| 256x15360x768 | 87.6 | 27.2 | 62.2 | 62.2 | 25.4 us | DRAM |
| 256x2048x6144 | 81.1 | 32.7 | 57.3 | 57.3 | 23.8 us | DRAM |
| 256x6144x4608 | 143.6 | 67.8 | 121.3 | 121.3 | 22.3 us | DRAM |
| 256x2048x2048 | 39.6 | 20.0 | 20.5 | 20.5 | 19.2 us | both, nearly tied |

512x6144x4608 is COMPUTE-limited - its compute alone takes longer than all its DRAM traffic. That changes what
is worth trying on it. Also checked whether a wider sub-block lowers the compute floor: it barely does
(139.2 -> 137.6 -> 135.6 for nsb 1, 2, 3) while the wall gets much worse (180 -> 200 -> 220), so all the
nsb damage is on the data-movement side and compute is close to a hard limit.

### F4. FAILED (noise): the S1 "regressions" were mostly measurement noise

Suspected the API fix hurt three shapes by using the true per-NoC placement targets. Made the op use the
NOC_0 target for both NoCs and A/B'd with 3 relaunches each: 32x6144x1536 +0.30%, 256x2048x2048 +0.25%,
512x6144x2304 +0.11%, 256x6144x4608 -0.38%, 256x2048x6144 -0.76%. All under 1%. So the 1.5-3.7% "regressions"
reported in S1 were within the noise of that 2-3 sample check. Kept the NOC_0-for-both choice anyway because
it is principled - a NOC_1 read response from a DRAM column travels the wrong way and wraps, so the API's
NOC_1 "optimal" worker is not actually a good target - and it decouples the op from that API subtlety.
Diagnostic bit17 restores per-NoC targets.

### S5. SHIPPED: picker table entry for 256x2048x6144 (nsb 1 -> 2)

The picker had no table entry for this shape so it used the cost-model fallback, which chose a 1-tile-wide N
sub-block. That makes every output write a lone 2 KB page. nsb=2 measured 5.7% faster (80.8 -> 76.16 us, three
relaunches: 76.03, 76.16, 76.18). nsb=4 is 2% worse than 2 and nsb=8 is 25% worse, so 2 is a real optimum, not
"bigger is better". Added `{{8, 64, 192}, {4, 3, 1, 2, 2}}` to the lookup table and mirrored it into the
offline model so future analysis matches. 111/111 correctness tests pass.

This shape is now **17.4% faster than at the start of the log** (92.2 -> 76.16 us).

### F6. FAILED and INSTRUCTIVE: the reduction chain's cost is WORK, not depth

Built the meet-in-the-middle chain described below (bit20): bands below the middle flow up, bands above flow
down, meeting at a root in the middle, so the critical path drops from 11 hops to 6. It is correct (PCC
1.000001 to 1.000134, no hangs, 111/111 tests) and gives **no speedup at all**:

| shape | linear chain | meet-in-the-middle | change |
|---|---|---|---|
| 512x6144x4608 | 180.60 | 180.46 | +0.08% |
| 256x2048x6144 | 76.23 | 76.12 | +0.15% |
| 32x6144x1536 | 40.69 | 40.58 | +0.28% |
| 512x6144x2304 | 109.75 | 110.16 | -0.38% |
| 256x2048x2048 | 39.38 | 39.56 | -0.46% |

Why the hypothesis was wrong: halving the DEPTH does not reduce the WORK. Whatever the topology, each of the
Pk-1 non-root bands still performs exactly one full-block add and one full-block transfer. Meet-in-the-middle
only shortens how long the LAST partial takes to arrive, and that latency was evidently already hidden. So the
14.3% is real arithmetic and real traffic, not a serial tail.

That also rules out a fan-in-2 tree for the same reason - it changes depth, not work - and reduce-scatter does
not help either: it spreads the same total number of adds differently. To reduce this cost you would have to
reduce the NUMBER of partial sums, i.e. use a smaller Pk, which the picker already trades off against
parallelism. **Treating the reduction as closed.**

Kept the implementation behind bit20 as a documented negative (default off; it also required a second receive
semaphore, so the op now creates 6 instead of 5, well under the 16 available).

### Original sizing of the reduction chain (kept for the record)

We could never size the split-K reduction before, because deleting it makes every band write its own copy of
the output (Pk times the output traffic) and that artefact swamped the result. Comparing "skip reduction AND
output" against "skip output only" removes the artefact:

| shape | wall | cost of the reduction chain | chain depth (Pk-1) |
|---|---|---|---|
| 512x6144x2304 | 109.8 | **14.3% (15.7 us)** | 11 |
| 512x6144x4608 | 180.2 | **9.2% (16.6 us)** | 11 |
| 256x2048x2048 | 38.8 | **12.7% (4.9 us)** | 3 |
| 256x2048x6144 | 76.8 | 5.3% (4.0 us) | 3 |
| 256x15360x768 | 87.5 | 2.6% (2.3 us) | 5 |

This is now the biggest single item on the two highest-headroom shapes. The cost is a serial tail: with Pk=12
the last partial sum has to travel 11 hops, and each hop is a 32 KB transfer plus an add on the receiving core
(about 1.2 us per hop, which matches the 15.7 us measured).

Fix with the best value/risk: a **meet-in-the-middle chain**. Today band 0 sends to 1, 1 to 2, ... 10 to 11,
so the root is band 11 and the depth is 11. Instead let bands 0..5 flow up and bands 11..7 flow down, meeting
at band 6 as the root: the critical path becomes 6 hops instead of 11, so roughly half the tail, worth an
estimated 7-8 us (about 7%) on 512x6144x2304. A full fan-in-2 tree would reach depth 4 but is a much bigger
change.

Implementation notes for whoever picks this up:
- The root must accept TWO incoming partials per sub-block. One semaphore is enough if the two senders write
  to DIFFERENT cb_reduce slots and each increment it by one: the root waits for two increments, then adds both
  known slots. That avoids the "single counter is fungible" trap that the old tree used two semaphores for.
- cb_reduce has to grow from 2 slots to 4 (two per sub-block phase), which costs about 64 KB of L1 on
  512x6144x2304 - affordable.
- The reverse credit signal (redfree) needs to handle two senders per root.
- Watch out: reassociating the sum means the result is PCC-equal but NOT bit-identical, so validate with PCC
  plus a constant-input test (association-independent), exactly as the earlier tree/reduce-scatter work did.
- Prior art: a fan-in-2 tree existed at commit dab66853bdc (227 lines over 4 files) before the production
  cleanup removed it. It cannot be cherry-picked as-is because the diagnostic mechanism it used is gone, but
  it is a useful reference. Its lesson also matters: the tree only pays when the reduction tail is EXPOSED.
  Back then that was measured at Pk=4 where the chain is only 3 deep and it was neutral-to-negative. We have
  now measured the tail exposed at 14.3% with Pk=12, which is a different regime.

### F5. FAILED: a deeper reduction buffer does not help (and caught a real bug)

If the reduction chain's 9-14% cost were credit stalls, giving cb_reduce more than its 2 slots would help. It
does not: depth 4 and depth 8 measured +0.1% to +0.8% (all inside noise) on all four shapes tried. So the cost
really is the serial DEPTH of the chain - transfer plus add, once per hop - which is what the
meet-in-the-middle design above targets. Useful negative: it rules out buffering as the fix.

The knob was kept (diag bits 18-19 select depth 2/4/8, default 2 = production) because the refactor that
carries the depth through the existing `use_reduce` compile argument makes the CB size and the remote write
offset impossible to disagree - the exact class of mismatch that caused a PCC 0.38 bug in earlier reduction
work.

**This experiment also caught a real bug in itself, which is worth recording:** the first version wrote
`nb % use_reduce`, and for Pk==1 shapes `use_reduce` is 0. The unreachable branch still gets compiled, so the
kernel failed to build with "division by zero" and **20 of 111 correctness tests failed**. Fixed by guarding
the modulus with a constant. Lesson: always run the full correctness suite after touching a kernel, even for
a change that looks like pure buffering - the compiler sees code paths that the hardware never runs.

### F7. FAILED: the M-split in1 forward is free, so read-ahead in the reader is not the lever

On M-split shapes the reader does read -> wait for the slave's credit -> forward -> flush, strictly per block,
which looked like a serialization worth breaking with read-ahead. Added a diagnostic that drops the forward
payload (bit21) to size it first. It costs nothing:

| shape | wall | cost of the forward | cost of the in1 DRAM read |
|---|---|---|---|
| 256x15360x768 | 87.4 | -0.5% (nothing) | +35.7% |
| 256x6144x4608 | 143.9 | +0.0% | +35.4% |
| 256x2048x2048 | 39.4 | +0.6% | +19.5% |
| 256x6144x1536 | 61.1 | -0.3% | +22.4% |

So the reader's time is the DRAM read, not the copy to its slave, and read-ahead over the forward would buy
nothing. Closed without building it - the diagnostic cost one build and saved the implementation.

### What the remaining gap actually is (256x15360x768 as the worked example)

Its in1 read needs 46 us of DRAM time and its in0 read needs 15 us, so the DRAM floor is 62 us against an 87
us wall. The gap is almost exactly the in0 read: the in0 gather runs FIRST and the in1 read does not overlap it
(in0's measured exposure, 15.8 us, is essentially its whole DRAM time, meaning nothing else happens during
it). Perfect overlap would put the wall near 65 us.

The obvious unlock - let the in1 reader run ahead during the gather by deepening its buffer - does NOT work
(F3: +0.7%, -0.1%, +0.3% at depths 8/16/32). On M-split shapes the reader is also gated by its slave's credit,
and deepening the buffer deepens both, so the reason it fails is not yet explained. **This is the most
promising open question**: why the in1 read refuses to run ahead during the in0 gather even with buffer space.
Answering it needs per-RISC timeline zones (which stage is actually waiting), not another blind knob.

### S6. FIXED A REGRESSION I SHIPPED: the mesh gate needed Mt >= 8

**Process error worth recording.** I fitted the mesh gate (S2/S3) by measuring every corpus shape with the
mesh on versus off - but at that moment the "off" baseline was itself degraded, because the API cache fix (S1)
had changed the placement targets for half the cores. I later restored those targets (F4), which made the
no-mesh placement much better, and that silently invalidated the gate I had fitted against the worse baseline.

Caught it by re-running the whole 60-shape corpus against the pre-session baseline, which showed the op
**2.58% slower on average** with individual shapes 8-13% slower. Re-running the mesh A/B against the CURRENT
baseline confirmed the mesh was actively hurting a family of shapes:

| shape | mesh | no mesh | mesh gain |
|---|---|---|---|
| 128x6144x4608 | 141.62 | 125.19 | **13.1% slower** |
| 32x6080x4640 | 138.52 | 122.74 | **12.9% slower** |
| 128x15360x1536 | 125.75 | 111.71 | **12.6% slower** |
| 64x15360x768 | 63.61 | 57.58 | 10.5% slower |
| 64x6144x768 | 28.15 | 25.64 | 9.8% slower |
| 128x6144x2304 | 74.31 | 67.81 | 9.6% slower |
| 512x6144x2304 | 109.85 | 131.32 | 16.4% faster |
| 256x6144x768 | 35.69 | 46.89 | 23.9% faster |

Every one of those shapes is Pk=12/Ns=1/Sm=1, i.e. inside the old gate - so slice count was not the
discriminator at all. **Mt (the number of row tiles) is.** The mesh trades in1 read locality for a ~70% cut in
ring traffic, and ring traffic per shard scales with M_block = Mt/Sm. At Mt <= 4 the shards are so small there
is almost no ring traffic to save, while the read penalty is paid in full. Added `Mt >= 8` to the gate:

| shape | shipped | no mesh | change |
|---|---|---|---|
| 32x6080x4640 | 122.58 | 122.95 | +0.30% (no longer meshed) |
| 128x6144x4608 | 125.30 | 125.04 | -0.21% (no longer meshed) |
| 64x6144x768 | 25.53 | 25.54 | +0.03% (no longer meshed) |
| 128x15360x1536 | 111.72 | 111.70 | -0.01% (no longer meshed) |
| 256x6144x768 | 35.53 | 46.45 | **23.5% faster** |
| 256x2048x6144 | 76.51 | 90.41 | **15.4% faster** |
| 512x6144x2304 | 110.09 | 132.27 | **16.8% faster** |
| 256x6080x4640 | 148.73 | 152.26 | 2.3% faster |

**Lesson: when a change alters the baseline, every gate fitted against the old baseline has to be re-fitted.
And always close the loop against a fixed external reference, not just against the current binary.**

### Corpus-wide result against the pre-session baseline (60 shapes)

After the gate fix: **mean 0.99% faster, median 0.40% faster, 50 of 60 shapes within 2%.** The only shapes
more than 3% off:

| shape | before | after | change |
|---|---|---|---|
| 256x6144x768 | 46.62 | 35.88 | **23.0% faster** |
| 256x2048x512 | 16.54 | 15.08 | **8.8% faster** |
| 256x6144x2304 | 86.32 | 79.01 | **8.5% faster** |
| 256x15360x768 | 95.54 | 88.43 | **7.4% faster** |
| 256x2048x1024 | 22.43 | 23.15 | 3.2% slower - baseline predates the reduce-scatter removal (not from this work) |
| 256x2048x2048 | 37.88 | 39.12 | 3.3% slower - same reason |

Note this corpus is Mt<=8 only, so the two largest wins of the session (512x6144x2304 and 512x6144x4608) are
not in it.

### F8. FAILED: no better config exists for 256x15360x768 (the largest unexplained gap)

Swept the config space on the shape with the biggest unexplained gap. The deployed config wins outright:

| config (Ns,Pk,Sm,kb,nsb) | wall |
|---|---|
| **1, 6, 2, 2, 3 (deployed)** | **87.63** |
| 1, 12, 1, 1, 3 | 92.03 |
| 1, 10, 1, 2, 3 | 102.07 |
| 1, 8, 1, 2, 3 | 122.86 |
| 1, 12, 1, 2, 3 | 123.59 |
| 1, 6, 1, 2, 3 | infeasible (L1) |

So the picker is right here too, and the config dimension is exhausted for this shape.

### State of the remaining gap, and what it needs

Ablations on 256x15360x768 account for only about half its 87.6 us: in0 read 15.8, in1 read 31.2, reduction
2.3, output 0.9, ring 0.7, compute 1.1. Its DRAM bytes need 61 us. No single stage removal explains the rest,
which means the remainder is overlap loss spread across stages rather than one fixable item. Compute is NOT the
problem (deleting it saves 1.1%), and the config space is exhausted (F8), and the in1 buffer depth does not
unlock it (F3), and the M-split forward is free (F7).

Answering this needs per-RISC TIMELINE instrumentation - zones that record when each RISC is waiting and on
what - not another knob. Every knob has now been tried and measured. That instrumentation is a substantial
build (the earlier fine-grained zone system was removed in a cleanup and would need a port), and it must not be
attempted without room to validate it: this session already shipped one regression from a gate fitted against a
baseline that had shifted, and caught it only by closing the loop against a fixed external reference.

### CLOSED: the in0 gather cannot be overlapped, because it saturates DRAM while it runs

The last open question was why the in1 read does not run ahead during the in0 gather. Answered with a
zero-build experiment - the pair interaction between the two reads. If deleting both saves as much as deleting
each separately, they are serialized:

| shape | wall | skip in0 | skip in1 | sum | skip both | verdict |
|---|---|---|---|---|---|---|
| 256x15360x768 | 87.8 | 15.8 | 32.1 | 47.9 | **48.6** | fully serialized |
| 256x6144x4608 | 143.5 | 7.9 | 50.9 | 58.9 | 55.2 | fully serialized |
| 512x6144x2304 | 109.7 | 14.1 | 6.3 | 20.4 | 18.2 | partial overlap |
| 512x6144x4608 | 180.6 | 14.7 | 7.6 | 22.3 | 19.6 | partial overlap |

They are serialized. But the cause is NOT the in1 buffer being too small (which is why deepening it never
helped) - it is that **the in0 read saturates DRAM for the whole time it runs**, leaving no bandwidth for
anything else:

| shape | in0 bytes | time it is exposed | effective rate | share of the 512 GB/s peak |
|---|---|---|---|---|
| 256x15360x768 | 7.7 MB | 15.8 us | 487 GB/s | **95%** |
| 512x6144x2304 | 6.3 MB | 14.1 us | 447 GB/s | 87% |
| 512x6144x4608 | 6.3 MB | 14.7 us | 428 GB/s | 84% |

So the 14-16 us gather is not a scheduling failure that better buffering or read-ahead could hide. It is
hardware-limit DRAM work that simply cannot share the bus. And in0 is already read exactly once per core (Ns=1
on all these shapes), so there are no redundant bytes to remove either. **The in0 gather is irreducible.**

That makes the honest floor for 256x15360x768 about 62 us (15.8 us of gather that cannot overlap, plus 46 us of
in1 bytes at peak bandwidth) against an 87.8 us wall. The residual 26 us is overlap loss spread thinly across
compute tail, reduction and latencies, with no single stage responsible - compute deletion only saves 1.1%.
Resolving that specific residual is the only remaining lever, and it needs per-RISC timeline zones, since every
knob (config, buffer depths, placement, ring order, reduction topology, subblock shape) has now been measured
and closed.

## CORRECTION: four experiments were invalid (diagnostic cache aliasing)

The env-var parse silently DROPPED any mask above 0x1FFFF (131071): `if (v > 0 && v <= 0x1FFFF)` left
`diag_mask` at 0. Several later knobs (placement targets, CB depths, reduction topology) are ALSO read from the
same env var inside the planner, so those still changed the program while the hashed `diag_mask` - and hence
the program-cache key - stayed 0. Two different masks therefore aliased onto ONE cached program, and the second
arm of each A/B silently re-ran the first arm.

All bits up to 16 are unaffected, so **everything shipped and every stage ablation is valid** (max mask used
there was 65536). The four experiments using bits 17-21 were not:

| experiment | masks | was it valid? |
|---|---|---|
| mesh + gate fits, subblock, all stage ablations, in1 placement, ring order, D1 | <= 65536 | VALID |
| F4 placement targets | 131072 | invalid |
| F5 reduction-CB depth | 262144 / 524288 | invalid |
| F6 meet-in-the-middle reduction | 1048576 | invalid |
| F7 M-split forward | 2097152 | invalid |

Fixed by making an out-of-range mask a hard error instead of a silent clamp - verified it now rejects
99999999 - so this class of mistake cannot recur.

### Re-run results

**F4 (placement targets) - the conclusion CHANGES.** It is not neutral. Using the NOC_0 assignment for both
NoCs, which is what ships, is clearly better on one shape and mildly better on others:

| shape | shipped (opt0 both) | true per-NoC | shipped is |
|---|---|---|---|
| 128x6144x4608 | 125.05 | 146.96 | **14.9% faster** |
| 32x6144x1536 | 40.53 | 41.09 | 1.4% faster |
| 512x6144x2304 | 109.72 | 110.14 | 0.4% faster |
| 256x2048x2048 | 39.04 | 39.09 | 0.1% faster |
| 256x6144x4608 | 143.59 | 143.40 | 0.1% slower |

So the shipped choice is right and now actually justified by data, for the reason predicted: a NOC_1 read
response leaving a DRAM column wraps most of the way round the torus, so the API's NOC_1 "optimal" worker is a
poor target.

**F5 (reduction-CB depth) - conclusion unchanged.** Depth 4 and 8 give +0.1% to +0.5% (noise). Buffering is
still not the reduction's cost.

**F7 (M-split forward) - conclusion essentially unchanged.** The forward payload costs -0.0%, +1.4%, -0.6%.
Still not worth a read-ahead redesign.

**F6 (meet-in-the-middle reduction) - conclusion REVERSED, and it is worse than useless.** With the bit
actually taking effect, the topology **deadlocks**: it hung the device twice and needed a `tt-smi -r` both
times. So the earlier "correct, zero gain" claim was an artifact of the bit never taking effect. Two attempts
at the credit/slot protocol both hung, so the path is now hard-disabled with an explicit throw rather than
left as a trap. What we know remains true is only the SIZING (from valid masks 32/48): the reduction chain
costs 14.3% on 512x6144x2304 and 9.2% on 512x6144x4608. Whether shortening its depth helps is **unknown and
untested** - the fan-in-2 tree and reduce-scatter reasoning I wrote off on the basis of F6 should be treated as
OPEN again.

111/111 correctness tests pass after all of this.

## Total progress this session

| shape | at log start | now | change |
|---|---|---|---|
| 512x6144x2304 | 134.0 | 110.0 | **17.9% faster** |
| 256x6144x768 | 42.0 | 35.6 | **15.2% faster** |
| 256x2048x6144 | 92.2 | 76.2 | **17.4% faster** |
| 512x6144x4608 | 207.5 | 180.3 | **13.1% faster** |
| 256x2048x512 | 16.9 | 15.1 | **10.5% faster** |
| 256x15360x768 | 95.1 | 87.6 | **7.9% faster** |
| 32x6144x1536 | 40.5 | 40.4 | unchanged |
| 256x6144x4608 | 141.4 | 143.6 | 1.6% slower (noise-level; gate declines the mesh here) |
| 256x2048x2048 | 37.9 | 39.0 | 2.9% slower (inside the +-2.4% noise floor for this shape) |

Across the whole 63-shape corpus the mesh alone is adopted on 26 shapes for a mean 5.4% gain.

## Work queue

1. ~~Fit an adoption gate for the mesh and ship it.~~ DONE (S2, S3).
2. ~~Re-run ring ORDER optimisation on the mesh.~~ CLOSED by analysis (see note above).
3. ~~Mesh v2 (spread slices over rows).~~ FAILED, see F1.
4. ~~Re-measure where the time goes.~~ DONE (see the two tables above).
5. ~~Revisit the S1 regressions.~~ CLOSED as noise (F4).
6. Next candidates, in order of expected value:
   a. Hide the in0 own-shard read behind compute on the two 512-row shapes (it is 7.5-12.2% exposed and it is
      the serial head of the gather; the in1 reader goes idle during it because its buffer only holds 4
      blocks). Earlier chunk-streaming of the ring was rejected as noise, but that was before the mesh, when
      the ring was 17% and the read 10%; now the ring is 3.5% and the read 12.2%, so the balance has flipped.
   b. ~~Add 256x2048x6144 nsb=2 to the picker table.~~ DONE (S5, +5.7%).
   c. 512x6144x4608 is compute-limited, so for it the only real lever is fewer or cheaper math passes.

### S7. SHIPPED: restored the reduce-scatter reduction that a "cleanup" commit had deleted (-5.5 to -16.4%)

**How I found it.** The user asked for the corpus sorted by absolute slack. Five shapes in that corpus carried
a note "was-reduce-scatter(now chain)" from an earlier run, so I went looking for why. Commit `21b08d6f1df`
("Production cleanup: remove all diagnostic modes + reduction experiments") had deleted the ring reduce-scatter
reduction that commit `1eee35d311a` had measured, gated and shipped. The cleanup commit's own message says
*"No production numerical change (chain reduction ... preserved)"* - that is true about the numbers and false
about the speed. It reverted a shipped win while claiming to change nothing.

**What reduce-scatter is, in plain terms.** With split-K, Pk cores each compute a partial sum of the same
output block and the partials have to be added together.

- The **chain** passes the whole block up a line of Pk cores. Core 1 sends to core 2, which adds its own
  partial and sends to core 3, and so on. The last partial cannot start moving until Pk-2 earlier hops have
  finished, and the single core at the top writes the entire output to DRAM.
- **Reduce-scatter** cuts the block into Pk pieces and rotates the pieces around the Pk cores. Every core sends
  a piece every round, so all the cores are busy at once instead of one at a time. After Pk-1 rounds each core
  holds one finished piece, which it writes to DRAM itself - so the output write is shared by Pk cores instead
  of dumped on one.

The total number of additions and the total bytes moved are IDENTICAL. What changes is how much of it has to
happen one-after-another, and how many cores share the output write.

**Measured**, mask 0 (gate picks reduce-scatter) vs bit22 FORCE_CHAIN, at the deployed picker config, two
relaunches with the mask order reversed on the second:

| shape | chain (us) | reduce-scatter (us) | change |
|---|---|---|---|
| 256x2048x1024 | 23.37 | 19.54 | **16.4% faster** |
| 256x2048x2048 | 39.21 | 34.31 | **12.5% faster** |
| 128x2048x1024 | 16.40 | 14.50 | **11.6% faster** |
| 128x2048x2048 | 27.04 | 24.62 | **9.0% faster** |
| 64x2048x1024 | 12.85 | 12.14 | **5.5% faster** |

Mean 11.0% faster. The two relaunches agree within 0.9 percentage points on every shape, so this is well clear
of the +-2.4% noise floor for these 2048-K shapes. It is also BIGGER than the 5-9% the original commit
claimed, because the op has changed a lot since then.

**Where it is used.** The gate is exactly the original one and still selects exactly these five corpus shapes:
Pk>=4, K depth <= 64 tiles, N >= 32 tiles, N_sub >= 2, sub-block tile count divisible by Pk, unfused, single
output chunk. All 57 other corpus shapes keep the chain and are byte-identical to before.

**Precision.** Each owner adds the Pk partials in ring order rather than bottom-to-top, so the result is
PCC-equal but not bit-identical to the chain. Every add is still FP32 in DST and no operand is narrowed - this
is a different order of summation, not lower precision. PCC vs an FP32 CPU reference is 0.99998-0.999998 on all
five shapes, and chain-vs-scatter outputs agree to PCC 1.00000. 111/111 correctness tests pass.

Commit `a86dd5e0b68`. New diagnostics: bit22 FORCE_CHAIN, bit23 FORCE_RSCATTER.

**LESSON.** A refactor that preserves numerics can still silently delete performance. Any "cleanup" that
removes a gated optimisation must re-run the A/B that justified it, and the corpus baseline must be re-measured
after the cleanup, not assumed unchanged.

### S8. SHIPPED: reduce-scatter for 3 more shapes - uneven chunks + the N-width rule was wrong (-7.4 to -8.5%)

Having restored reduce-scatter (S7), I measured the shapes its gate DECLINED, using the new bit23
FORCE_RSCATTER. Both remaining restrictions turned out to be wrong for the shallow-K regime.

**Finding 1: the "wide N" rule was wrong.** The gate demanded N >= 32 tiles. `128x2048x512` has N = 16 tiles
and is **8.9% FASTER** with reduce-scatter. Across all 15 shapes measured, every shallow-K shape with Pk>=4 and
N_sub>=2 won - six out of six - no matter how wide N was. Rule deleted.

**Finding 2: the "divisible by Pk" rule locked out 41 of 62 corpus shapes for no reason.** The old code cut the
output sub-block into Pk EQUAL chunks, so it gave up whenever Pk did not divide the tile count. Chunks do not
have to be equal. They now differ by at most one tile (the first few take one extra), so the only real
requirement is that there are at least as many tiles as cores. The protocol is unchanged because every buffer
operation still moves a full maximum-size slot and only the useful part of it is written or read - so the
double-buffer rhythm and the remote write stride stay constant. For shapes that DO divide evenly, the code
behaves exactly as before, which I confirmed by re-measuring two of them (they reproduced their earlier numbers
within noise).

**Result: 5 -> 8 adopted shapes.** Measured against bit22 FORCE_CHAIN, two relaunches with the mask order
reversed on the second:

| shape | chain (us) | reduce-scatter (us) | change | why it is new |
|---|---|---|---|---|
| 256x2048x1024 | 23.26 | 19.85 | **14.7% faster** | |
| 256x2048x2048 | 39.16 | 34.18 | **12.7% faster** | |
| 128x2048x1024 | 16.40 | 14.50 | **11.6% faster** | |
| 128x2048x2048 | 27.04 | 24.62 | **9.0% faster** | |
| 256x2048x512 | 15.10 | 13.82 | **8.5% faster** | NEW - uneven chunks 2,2,1,1 |
| 128x2048x512 | 11.21 | 10.29 | **8.2% faster** | NEW - N-width rule dropped |
| 256x2048x1536 | 29.78 | 27.58 | **7.4% faster** | NEW - uneven chunks 3,2,2,2 |
| 64x2048x1024 | 12.85 | 12.14 | **5.5% faster** | |

The three newly-adopted shapes were carrying 46-54% relative slack to their roofline floor - the highest in the
whole corpus - so this is exactly where the headroom was.

**What is still on the chain, and why.** Deep-K shapes (K >= 72 tiles). Measured with bit23 they are genuinely
mixed: 256x15360x1536 -3.0%, 256x15360x768 -1.4%, 128x15360x768 -1.3%, 128x2304x6144 -0.7%, but
128x15360x1536 +3.0%, 64x15360x1536 +2.7%, 256x2304x6144 +2.3%. Mean about zero. The pattern is that the
LOSERS are the ones whose chunks shrink to a single tile: with Pk=12 you get 11 rounds of one 2 KB message
each, and the per-round handshake costs more than the shortened critical path saves. So chunk SIZE, not depth,
is what decides whether reduce-scatter pays.

Gate is now: Pk>=4, K <= 64 tiles, N_sub>=2, at least Pk tiles per sub-block, unfused, single output chunk.

PCC 0.99999-1.000000 vs an FP32 CPU reference on all 8 adopted shapes plus 2 chain controls; 111/111
correctness tests pass. Commit `f3370a45409`.

### F9. FAILED, and it protects the two biggest-slack shapes: reduce-scatter on deep-K is a big LOSS

With uneven chunks (S8) the two largest-slack shapes in the corpus became *feasible* for reduce-scatter for the
first time, so I tested them with bit23 FORCE_RSCATTER. Two relaunches, mask order reversed:

| shape | chain (us) | reduce-scatter (us) | result |
|---|---|---|---|
| 512x6144x4608 | 180.08 | 240.25 | **33.4% SLOWER** |
| 512x6144x2304 | 110.29 | 131.75 | **19.5% SLOWER** |
| 256x6144x6144 | 193.88 | 186.49 | 3.8% faster |
| 256x6144x4608 | 143.43 | 140.85 | 1.8% faster |
| 256x6144x1536 | 61.53 | 60.61 | 1.5% faster |

**Why the two 512-row shapes collapse.** Reduce-scatter replaces ONE full-block add per core with Pk-1 small
chunk-adds. Each of those calls pays a fixed setup cost (add_tiles_init + data-format reconfig) no matter how
few tiles it touches. These two shapes have N_sub = 1 and Pk = 12, so their chunks are 1-2 tiles and they run
18 sub-blocks x 11 rounds = **198 tiny add-calls where the chain does 18**. And 512x6144x4608 is the one shape
in the corpus that is already COMPUTE-floor-bound - its compute floor is 139.2 us of a 180.4 us wall, 77% - so
extra compute overhead is the worst thing you can add to it.

Where compute has slack the same change pays: 256x6144x6144 sits at 47% compute and gains 3.8%.

**So the rule is: reduce-scatter trades data movement for per-round compute overhead. It pays only when compute
has slack AND the chunks are big enough to amortise the per-round setup.** That is exactly why the shallow-K
shapes win 5-15% (compute is only 30-40% of their wall) and why the deep-K ones do not.

**Decision: keep the chain for deep K.** The deep-K wins are all <=3.8% and mostly at the +-1.5% noise floor for
large shapes, while the losses reach +33%. Bad expected value, and the gate as shipped in S8 already excludes
them. No code change - this is a guard rail recorded so nobody widens the gate on the strength of the three
small wins in the table above.

**Remaining headroom on the top two shapes is therefore NOT the reduction.** 512x6144x4608 has 41.2 us of slack
and is compute-floor-bound; the lever there is cheaper math, not cheaper movement. 512x6144x2304 has 38.1 us
with its compute floor (70.0) and DRAM floor (72.2) essentially equal, so it needs BOTH to improve.

### F10. FAILED: deeper kb cuts the compute floor but every config that reaches it costs more elsewhere

512x6144x4608 is the one corpus shape whose COMPUTE floor binds (139.2 us of a 180.4 us wall). Earlier work
found compute efficiency rises a lot with a deeper K block (kb), and this shape runs kb=2 - the maximum its
config allows, because the in0 ring shard must hold a whole number of kb-blocks and at Pk=12 the K slice is only
16 tiles. So I swept every 96-core config that reaches a deeper kb without wasting K.

| config | wall (us) | compute floor | vs deployed |
|---|---|---|---|
| **deployed (12,1,1,kb2,nsb1)** | **179.9** | 139.2 | - |
| 6,2,1,kb4,nsb1 | 185.2 | **128.6** | 2.9% slower |
| 6,1,2,kb4,nsb2 | 190.2 | 130.0 | 5.7% slower |
| 6,1,2,kb4,nsb1 | 206.8 | 132.0 | 15.0% slower |
| 4,1,3,kb6,nsb2 | 222.0 | 149.4 | 23.4% slower |
| 3,1,4,kb8,nsb2 | 305.5 | 204.0 | 69.8% slower |
| 2,1,6,kb12,nsb2 | 401.9 | 295.4 | 123.4% slower |

Same story on 512x6144x2304 (deployed 109.9 us; best alternative 123.6, 12.4% slower). Two configs
(3,4,1,kb8,nsb1 and 4,3,1,kb6,nsb1) do not fit in L1 at all - their in0 CB alone is 2.0 and 1.5 MB against a
1.44 MB budget.

**The deeper kb DOES work as predicted - it cuts the compute floor 7.6% (139.2 -> 128.6).** But the wall gets
worse anyway, and the reason is structural: the in0 ring shard is (Kt / (8*Pk)) * M_block, which does not depend
on kb at all. Raising kb while keeping 96 cores forces Pk down, and halving Pk DOUBLES the ring shard. So you
buy ~8% of compute with ~2x of in0 ring traffic. The picker's choice is correct.

**Conclusion: config tuning is closed for both 512-row shapes.** Their remaining slack is exposed data
movement at the deployed config, not a bad config and (per F9) not the reduction either.

### S9. SHIPPED: reduce-scatter for deep K too, under two mechanistic conditions (6 more shapes, -0.9 to -4.4%)

F9 said deep-K reduce-scatter was "genuinely mixed" and I left it on the chain. That was right as a default but
wrong as a stopping point: the mix is not random. F9 already identified the mechanism - reduce-scatter buys
less data movement by paying a fixed per-round compute setup cost - so it needs FEW rounds and ENOUGH WORK per
round. Turning that into two compile-time conditions:

- **Pk <= 6** - at most 5 rounds per sub-block.
- **max_chunk >= 2 tiles** - a round that ships a single 2 KB tile is nearly all overhead.

These two separate **all 13 measured deep-K shapes exactly**: the 6 that satisfy both were faster (-1.3% to
-3.8%), and all 5 slower ones are excluded (+2.7%, +3.0%, +19.5%, +33.4%, and one +2.3% at Pk=3). No threshold
was tuned to fit - both conditions came from the F9 mechanism before I checked them against the data.

Confirmed after shipping (mask 0 now selects reduce-scatter, bit22 forces the chain; 2 relaunches, order
reversed):

| shape | chain (us) | reduce-scatter (us) | change |
|---|---|---|---|
| 256x15360x1536 | 142.27 | 136.06 | **4.4% faster** |
| 256x6144x6144 | 193.08 | 186.75 | **3.3% faster** |
| 256x6144x1536 | 61.12 | 60.59 | 0.9% faster |

plus 256x6144x4608 (-1.8%), 256x15360x768 (-1.4%) and 128x15360x768 (-1.3%) from the bit23 sweep.

Adoption 8 -> 14 corpus shapes. Honest caveat: only the two leaders clear the +-1.5% noise floor for large
shapes; the other four are 0.9-1.8%, i.e. small but consistently signed across both relaunches. In ABSOLUTE
terms the six together take about 17 us off the corpus wall - comparable to the shallow-K win - because they are
the biggest shapes in the corpus.

Shallow K deliberately keeps NO chunk-size floor: it still wins with single-tile chunks (64x2048x1024 -5.5%,
128x2048x512 -8.2%) because compute there is only 30-40% of the wall rather than 47-77%.

PCC 0.99999-1.000000 on all 6 new shapes plus 2 excluded controls; 111/111 correctness tests pass.

### F11. FAILED (three NoC micro-optimisations): the in1 stream is BANDWIDTH-bound, not latency- or issue-bound

Sizing first. On the top-5 slack shapes the in1 read is 84% of the DRAM floor, and the ablations show how much
of it is EXPOSED (i.e. not hidden behind compute):

| shape | wall | in1 exposed | in1 DRAM time | output write exposed |
|---|---|---|---|---|
| 512x6144x4608 | 179.9 | 6.8 (3.8%) | 110.6 | 1.2 (0.7%) |
| 512x6144x2304 | 110.1 | 6.6 (6.0%) | 55.3 | 1.2 (1.1%) |
| 256x6080x4640 | 148.8 | **50.6 (34.0%)** | 110.2 | 0.4 (0.3%) |
| 256x6144x6144 | 194.5 | **78.3 (40.3%)** | 147.5 | 14.5 (7.5%) |
| 256x15360x1536 | 141.6 | 28.2 (19.9%) | 92.2 | 0.5 (0.3%) |

That also explains 256x6080x4640, whose in0+reduction components only accounted for 9.1 us of its 27.9 us of
slack: the rest is in1. **The per-tile output write is already hidden (0.3-1.1% on four of five), so that micro-
optimisation was dead before writing any code.**

**Attempt 1 - TRID-pipelined in1 read (bit24). FAILED, 3.8-10.0% SLOWER.** The production reader issues one
block then takes a FULL read barrier before pushing, so exactly one block is ever in flight and each pays the
whole DRAM latency; the dram-sharded matmul reference instead tags each block with a TRID and waits only on the
oldest. Implemented for the solo (Sm==1) path.

| shape | prod | TRID pipeline | change |
|---|---|---|---|
| 512x6144x4608 | 180.28 | 198.28 | 10.0% slower |
| 512x6144x2304 | 109.92 | 118.41 | 7.7% slower |
| 256x6080x4640 | 148.96 | 154.63 | 3.8% slower |
| 256x15360x1536 (Sm=2 control) | 136.14 | 136.72 | +0.4% (unaffected) |
| 256x6144x6144 (Sm=2 control) | 187.39 | 187.62 | +0.1% (unaffected) |

Prior work already measured in1 reads at 76-98% of peak DRAM in isolation. Running 4 blocks ahead on 96 cores
cannot raise throughput on a saturated stream - it just multiplies outstanding requests and adds queueing, on
top of the extra per-block reserve/TRID setup. **The per-block barrier was acting as free pacing.**

**Attempt 2 - wider N_sub for DRAM locality. FAILED, 3.0-43.8% SLOWER.** With N_sub=1 and a shard stride of 19
tiles, 256x6080x4640 makes 19 separate passes over the same K range taking one 2 KB column each - the worst
possible DRAM row locality. Widening N_sub makes each row read cover N_sub contiguous tiles and cuts the number
of passes:

| N_sub | read size | K passes | wall | in1 exposed |
|---|---|---|---|---|
| **1 (deployed)** | 2 KB | 19 | **148.4** | 50.2 |
| 2 | 4 KB | 10 | 152.9 | 48.1 |
| 4 | 8 KB | 5 | 163.9 | **45.0** |
| 5 | 10 KB | 4 | 170.0 | 46.7 |
| 10 | 20 KB | 2 | 213.5 | 53.1 |

The locality mechanism WORKS - exposure falls 50.2 -> 45.0 us as predicted - but the wall rises anyway, because
N_bpc is also the pipelining depth of the whole output/reduction phase and dropping from 19 output blocks to 5
destroys that overlap. On 256x6144x6144 locality did not even improve (77.6 -> 112.3 us exposure).

**Attempt 3 - stateful one-packet in1 reads (bit25). NULL, -0.13% mean (-0.37% best, +0.13% worst).** Every read
a core issues targets the same DRAM bank and full-width rows share one size, so the NoC size/config registers
can be written once instead of per transaction. This variant changes ONLY per-call issue cost - access pattern,
CB sizes and pipeline depth are all identical, unlike attempts 1 and 2 - so it isolates the question cleanly.
Bit-exact output (PCC 1.000000). **The reader is not issue-bound; there is no per-transaction overhead to
reclaim.**

**Conclusion: NoC micro-optimisation is closed for the in1 path.** All three attempts point at the same wall -
the in1 stream is bandwidth-bound. The "50.6 us exposed" on 256x6080x4640 is not recoverable latency; it is real
DRAM time that cannot hide because compute (86.1 us) is shorter than the in1 read (110.2 us). For that shape the
only remaining lever is reducing in1 BYTES or raising achieved DRAM efficiency, not restructuring the transfers.

Kept behind default-off diagnostic bits (both compile out entirely at mask 0): bit24 IN1_TRID_PIPELINE, bit25
IN1_ONE_PACKET. The shared per-block read sequence was factored into one `issue_block_reads` lambda used by all
three policies - measured neutral (every mask-0 wall reproduces its pre-refactor value within noise) and 111/111
correctness tests pass.

### F12. DIRECT-EXCHANGE reduce-scatter (bit26): fixes the serialization, still loses to the chain at Pk=12

F9/S9 showed the ring reduce-scatter loses badly at Pk=12 because it takes Pk-1 SEQUENTIAL rounds, each paying a
semaphore round-trip plus its own add setup (measured 0.22-0.30 us per round). Direct exchange removes that
serialization: every core writes its partial for chunk q straight to the core that owns chunk q, all Pk writes
issued back to back, then ONE wait for all arrivals. The reduce accumulates every incoming partial in fp32 DST
(binary_dest_reuse_tiles) so it costs one pack per output tile instead of one per round.

Three-way measurement, 2 relaunches with the mask order reversed:

| shape | Pk | ring rounds | chain | ring | direct | ring vs chain | direct vs chain | direct vs ring |
|---|---|---|---|---|---|---|---|---|
| 512x6144x4608 | 12 | 198 | 180.11 | 240.40 | 217.48 | +33.5% | +20.7% | **-9.5%** |
| 512x6144x2304 | 12 | 99 | 110.21 | 131.83 | 121.32 | +19.6% | +10.1% | **-8.0%** |
| 256x6144x6144 | 6 | 60 | 194.89 | 187.09 | 190.10 | -4.0% | -2.5% | +1.6% |
| 256x15360x1536 | 6 | 5 | 141.48 | 136.15 | 136.93 | -3.8% | -3.2% | +0.6% |
| 256x2048x1024 | 4 | 3 | 23.46 | 19.79 | 20.07 | -15.6% | -14.4% | +1.4% |

**The hypothesis was right in direction.** Direct beats the ring by 8-9.5% on exactly the two shapes with the
most sequential rounds, and is neutral-to-slightly-worse where rounds are already few - the signature of a
serialization fix. PCC 0.9999-1.0001.

**But it still loses to the chain at Pk=12, and the reason is MESSAGE COUNT, not serialization.**

| per core, per sub-block | chain | direct exchange |
|---|---|---|
| payload writes | 1 (whole block) | Pk = 12 |
| arrival atomics | 1 | 12 |
| credit atomics | 1 | 12 |
| **NoC transactions** | **3** | **36** |

On 512x6144x4608 that is 18 x 36 = **648 transactions against 54**. The accounting closes: the serialization
saving is real (critical-path transfer per sub-block falls from 11 x 32 KB to ~4 KB, about 52 us over the shape)
but roughly 90 us of per-message issue + remote-atomic cost swamps it, netting the +37 us measured.

**So reduce-scatter's trade at high Pk is fewer serialized BYTES for Pk x more MESSAGES.** With a 32 KB
sub-block there are not enough bytes on the critical path to pay for 12x the messages, and NO topology change
fixes that - only cutting messages would. The visible next lever: 24 of the 36 transactions are semaphore
atomics carrying identical values to a fixed peer set, so they are multicastable (36 -> ~14). Payloads are not
multicastable (different data per destination). That would likely close about half the remaining gap on
512x6144x2304 and probably not all of it on 512x6144x4608.

No production change - the gate still selects the RING, which is better on all 14 adopted shapes. bit26 is
default-off. Also stopped allocating the chain's cb7 running-sum CB when reduce-scatter is active (it is never
touched there). 111/111 correctness tests pass.

### F13. S-WAY STRIPED OWNER-GATHER (bits 27-28): implemented, DEADLOCKS, NO measurement obtained

Requested design, all of it implemented: S in {2,3,4} owners per group instead of S=Pk; direct writes to
physically optimised owners; no loopback NoC traffic; FP32 DST accumulation per stripe; double-buffered receive;
incremental (two-stage) reduction; and separate profiler zones for payload / arrival-A / arrival-B / credit-wait
/ credit-send / reduce-wait / output-write.

**Status: the protocol still deadlocks and I did not get a single performance number. Two device resets were
needed. Production is untouched and verified (111/111 after the final reset).**

The rationale for trying it: the group sends S(Pk-1) messages instead of the full exchange's Pk(Pk-1) -- Pk/S
fewer -- while moving the SAME total bytes. F12 showed the residual cost of full direct exchange is message
COUNT (36 NoC transactions per core per sub-block vs the chain's 3), so cutting messages Pk/S-fold is the right
target.

What was built:
- **Owner selection is provably optimal, not a search.** Every member writes to every owner, so total hop cost
  is separable: sum over owners of (sum over senders of dist(sender->owner)). Ranking candidates by INBOUND cost
  and taking the S cheapest is therefore exact. Distance is on the sender's writer NoC (asymmetric on the torus).
- **No loopback:** an owner keeps its own stripe where it already is, in the fp32 intermediate CB, and seeds DST
  from it. Nothing is written to self.
- **Incremental reduction:** arrivals are split across two semaphores by sender position, so the writer releases
  the first half of the partials to compute while the second half is still in flight.
- **DST accumulation** chunked to the 4-tile fp32 DST limit (a stripe of rs_T/S tiles exceeds DST for S<4), so
  each DST group costs 2 inits and one pack per output tile rather than one per source.

**Bug found and fixed (real, would have bitten any variant):** an owner credits every group member EXCEPT
itself, so an owner receives S-1 credits per sub-block while a non-owner receives S. The wait threshold assumed
S for everyone, which deadlocks every owner from sub-block 2 onward. Fixed with a role-dependent
`cred_per_sb = S - (is_owner ? 1 : 0)`.

**A second deadlock remains unisolated.** After that fix, S=2 on 512x6144x4608 still hangs (all 96 workers time
out), and it wedges the board hard enough that the following runs fail at device init - which is what destroyed
the S=3/S=4 data points too. Candidates I ruled out by inspection: arrival counts (exp_a/exp_b correctly exclude
self), semaphore addressing, cb_send depth, non-owner CB usage, and the credit arithmetic above. Candidates NOT
ruled out: the cb_recv reserve/push accounting across the two-stage push against a 2-generation buffer, and the
interaction between an owner holding intermediate_cb (a single-slot CB) across the whole exchange and the next
sub-block's matmul needing to reserve it.

**Process lessons worth keeping:**
1. My first three masks were WRONG - I wrote 150994944 for "S=3" but that is bit27+bit24 (S=2 plus the TRID
   pipeline), so an early "PCC=1.000000" was a different configuration entirely. Multi-bit encoded fields need
   the mask arithmetic checked, not eyeballed.
2. `tt-smi -r` chained after `pkill` in one compound command never ran (the whole command was killed), so the
   board stayed wedged and the next three runs failed at init for a reason unrelated to the code under test.
   Reset must be its own command - the same lesson already recorded once in this log.

Kept behind default-off diagnostic bits so it can be picked up later; nothing in the production path changed.

### F14. S-WAY STRIPED OWNER-GATHER now WORKS and is measured: beats the ring 8-9% at Pk=12, still loses to the chain. NOT shipped.

F13 left this deadlocked with no numbers. It now runs correctly and is fully measured. Three real bugs were
found; all three were in flow control, and each one is a trap worth remembering.

**Bug 1 (the deadlock): the writer's runtime args were never pushed.** The patch that was supposed to add them
asserted on a pattern that matched the COMPUTE arg site first, so the writer block was never inserted, and
`rs_b_sem` was never created (it lived in the same patch, which died on an assertion mid-way and wrote nothing).
DPRINT showed the writer reading `S=5914240` (a buffer address) and `expb=4294967284` at arg 24 while `P=4` at
arg 22 was correct - i.e. args 17-23 fine, 24+ garbage. With garbage arrival counts the semaphore waits could
never be satisfied. **Lesson: when a patch asserts on a code pattern that appears at more than one call site,
the assertion proves nothing about WHICH site was edited.**

**Bug 2 (PCC 0.26): mixed-format reduction.** Seeding fp32 DST from the fp32 partial and then accumulating bf16
peer slots gives garbage. The working direct-exchange path was accidentally homogeneous (its loopback write made
the core's own partial arrive as bf16 like everyone else's). Fixed by packing the owner's own stripe to bf16 in a
spare CB (c_6) first - a LOCAL pack, so the no-loopback-NoC requirement still holds - and reducing one format.

**Bug 3 (PCC 0.996 / nan, and the instructive one): TWO aggregate-counter races.**
- Credits: all S owners incremented ONE counter, so a fast owner's credits satisfied a sender's threshold while
  a DIFFERENT owner had not yet freed its buffer -> the sender overwrote live data. Fixed with per-owner credit
  counters (a sender must see nb-1 on EVERY owner it writes to).
- Arrivals: one CUMULATIVE counter had the same flaw in time rather than space. The credit wait only engages at
  nb>=2, so a sender can reach nb=1 unimpeded and its SECOND increment alone satisfies the owner's nb=0
  threshold while a slower sender's slot still holds garbage. Fixed with PER-GENERATION arrival counters
  (gen = nb & 1) that the owner resets after consuming, so the threshold means "this many DISTINCT senders
  arrived for THIS generation".
The error grew exactly with N_bpc (1 -> exact, 3 -> 0.9996, 18 -> 0.994/nan), which is the signature of a
pipelining race and is what located it. **Lesson: a summed semaphore cannot enforce a per-source guarantee -
neither across sources nor across pipeline generations.**

**Bug 4 (+26.6% on one shape): empty stripes.** rs_T=6 with S=4 splits 2,2,2,**0** - the fourth owner owns
nothing yet runs the whole protocol and every sender issues it a zero-byte write. Now S is capped so every
stripe is non-empty.

**Measured (2 relaunches, mask order reversed; chain / ring / striped S=2 / striped S=4):**

| shape | Pk | chain | ring | striped S=2 | striped S=4 | S4 vs chain | S4 vs ring |
|---|---|---|---|---|---|---|---|
| 512x6144x4608 | 12 | 180.23 | 240.22 | 248.94 | 218.47 | +21.2% | **-9.1%** |
| 512x6144x2304 | 12 | 110.12 | 131.94 | 136.10 | 120.90 | +9.8% | **-8.4%** |
| 256x6144x6144 | 6 | 194.50 | 187.13 | 184.26 | **183.74** | **-5.5%** | -1.8% |
| 256x6144x1536 | 6 | 61.20 | 60.75 | 59.07 | **59.01** | **-3.6%** | -2.9% |
| 256x2048x1024 | 4 | 22.97 | 19.63 | 20.80 | 20.03 | -12.8% | +2.0% |

S=4 beats S=2 nearly everywhere, as the load-imbalance argument predicts (more owners = better balance).

**WHY it still loses to the chain at Pk=12 - the zone breakdown** (S=4, 512x6144x4608, 18 sub-blocks, per-core
totals, median over cores):

| zone | median us | cores recording it |
|---|---|---|
| **Z_RSS_CREDITWAIT** | **119.61** | 96 (all) |
| Z_RSS_REDUCEWAIT | 40.18 | 32 (owners) |
| Z_RSS_PAYLOAD | 22.90 | 96 |
| Z_RSS_CREDITSEND | 7.63 | 32 |
| Z_RSS_OUTWRITE | 5.75 | 32 |
| Z_RSS_ARRIVE | 5.57 | 32 |

**The dominant cost is not messages at all - it is LOAD IMBALANCE.** Credit-wait is 119.6 us: the 64 non-owner
cores sit idle while the 32 owners do all the reduction (40.2 us) and all the output. Concentrating the
reduction on S of Pk cores makes each owner perform (Pk-1)*rs_T/S tile-adds against a chain core's rs_T - a
2.75x overload at Pk=12,S=4 - and the other Pk-S cores simply wait. At Pk=6,S=4 the overload is only 1.5x, which
is precisely why the two Pk=6 shapes win.

**So the three reduction topologies measured so far trade along one axis:**
chain = minimum messages, minimum imbalance, maximum serialization;
ring = minimum bytes on the critical path, Pk x messages, Pk-1 sequential rounds;
striped owner-gather = few messages AND no serialization, but imbalance proportional to Pk/S.
At Pk=12 there is no setting that beats the chain, because the reduction work has to live somewhere.

**DECISION: not shipped.** The two clear wins (256x6144x6144 -1.8%, 256x6144x1536 -2.9% vs the shipping ring)
have NO structural discriminator against the Pk=6 shapes that were neutral-or-worse (256x6144x4608 +0.3%,
256x15360x1536 +1.0%), and the rest of the adopted corpus is a wash (128x15360x768 -1.1%, 256x15360x768 -0.6%,
64x2048x1024 -0.1%, 128x2048x2048 +0.8%, 256x2048x2048 +2.0%). Per the lesson already recorded in this log,
2 stable shapes without a structural trigger is not enough to ship. Kept behind default-off bits 27-28 with
zones and DPRINT markers both opt-in via env vars, so production is byte-identical. 111/111 tests pass.

### Reverted the direct-exchange and striped-owner-gather CODE (evidence retained)

Decision (2026-07-29): stick with the two reduction topologies that ship -- the linear CHAIN and the ring
REDUCE-SCATTER. The experimental alternatives (F12 direct exchange, F13/F14 S-way striped owner-gather) never
beat the chain where the chain wins, and against the ring they were a wash on the adopted corpus, so carrying
them as dead code in the op was not worth it.

Restored the four op source files to commit `3a229f8c5dc` (the commit immediately before direct exchange began),
removing 581 lines: the RS_DIRECT and RS_STRIPED kernel paths, `rsd_reduce_slots` / `rss_reduce_stripe`, the
striped owner-selection pass, the extra arrival/credit semaphores, the c_4/c_5/c_6 sizing for those modes, the
opt-in zone and DPRINT instrumentation, and diag bits 26-28 (kMaxDiagMask back to 0x3FFFFFF).

What REMAINS shipped and untouched: the chain, the ring reduce-scatter with its full gate (S7/S8/S9), the 2D
mesh placement, the subblock enlargement, the picker entry, and the default-off in1 NoC diagnostics bits 24-25.

Verified after the revert:
- working tree byte-identical to `3a229f8c5dc` for all four files (`git diff` empty);
- no residual `RS_DIRECT` / `RS_STRIPED` / `rs_striped` / `rs_direct` / zone / DPRINT symbols anywhere in the op;
- diag bits 26 and 27 are now correctly REJECTED as out of range (the aliasing guard from `0a6c55660fa` doing
  its job -- a silently-ignored stale mask is exactly the failure mode that invalidated four experiments);
- 111/111 correctness tests pass;
- production perf at mask 0 matches the pre-experiment measurements: 512x6144x4608 179.96 (was 180.2),
  256x6144x6144 186.62 (186.8), 256x6144x1536 61.00 (60.6), 256x2048x1024 19.32 (19.8), 256x2048x512 13.63
  (13.8) -- every shape within the +-2.4% noise floor, and the two reduce-scatter shapes still show their
  reduce-scatter walls, confirming the gate still selects the ring where it should.

The negative results themselves (F12, F13, F14) are kept above: they contain the message-count/serialization/
load-imbalance model of the reduction, the four flow-control bugs, and the zone breakdown showing that at Pk=12
the reduction cost is load imbalance rather than messaging. That model is the reason not to revisit these
topologies, so it is worth more than the code was.

### Re-applied: don't allocate the chain's cb7 under reduce-scatter (L1 saving, perf-neutral)

The revert above restored cb7 allocation for reduce-scatter shapes. cb7 is the CHAIN's running-sum buffer;
reduce-scatter routes its partials through the c_4/c_5 chunk CBs and never touches cb7, so allocating it is dead
L1. Re-applied as a standalone change.

L1 freed on the 14 reduce-scatter shapes (cb7 = 2 * M_block * N_sub bf16 tiles):

| shape | Pk | cb7 KB | op L1 KB | freed |
|---|---|---|---|---|
| 256x15360x1536 | 6 | 96 | 1024 | 9.4% |
| 256x15360x768 / 128x15360x768 | 6 | 48 | 832 | 5.8% |
| 256x6144x6144 / 4608 / 1536 | 6 | 32 | 416 | 7.7% |
| 256x2048x1024 | 4 | 64 | 384 | **16.7%** |
| 256x2048x2048 | 4 | 48 | 304 | 15.8% |
| 128x2048x2048 | 4 | 48 | 320 | 15.0% |
| 256x2048x1536 | 4 | 36 | 252 | 14.3% |
| 128x2048x1024 | 4 | 32 | 224 | 14.3% |
| 256x2048x512 | 4 | 24 | 200 | 12.0% |
| 64x2048x1024 / 128x2048x512 | 4 | 16 | 144 | 11.1% |

5.8-16.7% of the op's L1 per core, 41 KB on average.

The kernels are BYTE-IDENTICAL: `use_reduce` (the cb7 depth compile arg) is still computed from the planner's
`cb.cb7_tiles`, which is unchanged - only the `mkcb` call is skipped. So the only possible perf effect is L1
addresses shifting for the other CBs, which has to be measured rather than assumed.

Measured, mask 0: 512x6144x4608 (CHAIN, unaffected control) 180.12 vs 179.96 = +0.1%; 256x6144x6144 186.89 vs
186.62 = +0.1%; 256x6144x1536 60.85 vs 61.00 = -0.2%. The two small 2048-K shapes first read +2.6% and +1.9%,
which is at their +-2.4% noise floor, so I resampled with 4 relaunches and compared against EVERY
ring-configuration measurement of those shapes from earlier in the session:

| shape | now (4 runs) | median | history (cb7 allocated) | median | delta |
|---|---|---|---|---|---|
| 256x2048x1024 | 19.37 / 19.52 / 19.62 / 19.84 | 19.57 | 19.32-19.85 | 19.63 | **-0.3%** |
| 256x2048x512 | 13.79 / 13.80 / 13.82 / 13.89 | 13.81 | 13.63-13.82 | 13.76 | **+0.4%** |

Both inside the historical spread: the initial +2.6% was a low-outlier single-run reference, not a regression.
**LESSON: never accept a 1-2% delta against a single-run baseline on these small shapes - the noise floor is
wider than the effect.** 111/111 correctness tests pass.

### COMPREHENSIVE DEFAULT-CONFIGURATION SWEEP (Mt<=16, LTX + FLUX union, 67 shapes)

Everything at DEFAULTS: `config=None` so the production picker chooses, no diagnostic mask, no env override that
changes behaviour. Corpus = union of the broad Mt<=16 corpus and the canonical FLUX/LTX production shape list
(the FLUX/LTX set turned out to be almost entirely a subset -- only 1 of its 20 shapes was not already present).
Device time via the profiler CSV demuxed by run-host-id (op device time, not host wall), 2 warmup + 12 timed
iterations per block, 2 blocks per shape = 24 timed iterations. Absolute PCC against an FP32 CPU reference.

Full table in `PROD_SWEEP_MT16.md`, sorted ascending by effective DRAM bandwidth. Harness:
`prod_sweep_worker.py` + `prod_sweep_report.py`.

**Ground-truth configuration.** There is no way to read the auto-selected config from Python, and the host-side
python mirror of the picker is NOT validated against `auto_select_config` (the only parity test covers
feasibility, not which config is chosen), so reporting the mirror could silently misreport. Added
`TT_REGIME_A_LOG_CFG`, an observability-only env var that makes the factory log its own pick, reduction strategy
and placement once per program-cache miss. Behaviour is unchanged; the table's cfg/reduction/placement columns
are therefore ground truth.

**Headline results (66 measured / 67):**

| metric | value |
|---|---|
| effective DRAM BW | min 247.6, median **436.6**, max 501.3 GB/s (peak 512) |
| % of peak | median **85%**; 44 of 66 shapes above 80%; only 1 below 50% |
| correctness | **66/66 PCC >= 0.999**, minimum 0.99997 |
| stability | median block-to-block spread **0.2%**, worst 1.3% (256x2048x1024); median iteration spread 2.9% |
| reduction chosen | 14 reduce-scatter / 52 chain |
| placement chosen | 11 mesh / 9 in1-near / 46 bank-local |

The gate selections match what the campaign predicted exactly: the 14 reduce-scatter shapes are precisely the 14
the S8/S9 gate was fitted to adopt, and the 11 mesh shapes are all Mt>=8, confirming the S6 `Mt >= 8` fix is
doing its job in production.

**Where the remaining headroom is.** The bottom of the table is dominated by ONE family: the shallow-K
(K=2048) M-heavy shapes plus the Mt=16 wide-N shapes. The five worst are 256x2048x512 (48% of peak),
512x6144x768 (52%), 128x2048x512 (53%), 256x2048x1024 (58%), 256x2048x1536 (58%). Every one of them is small in
absolute time (10-62 us), so the absolute-slack ranking is unchanged -- this is a low-efficiency-but-cheap
corner, not the place the wall-clock is. The largest shapes all sit at 85-98% of peak, which is why the earlier
optimisation rounds found so little left there.

**One hard failure, pre-existing and already known: 512x15360x768.** The picker THROWS rather than returning a
config: "auto-select found no feasible config for Mt=16 Kt=480 Nt=24"
(`regime_a_matmul_config.cpp:211`). Diagnosis: the anchor search in step 1 only considers **Sm=1**, and at Sm=1
this shape cannot fit L1 -- the in0 resident CB alone is 640 tiles = 1280 KB, and the full CB set is 1480 KB at
N_sub=1 (1680 at N_sub=2, 1880 at N_sub=3) against a 1440 KB budget. It NEEDS M-split, but step 2's M-split
hysteresis only ever runs as an improvement on an already-valid anchor, so with no anchor it fatals. Sm=2/Pk=6,
Sm=3/Pk=4 and Sm=4/Pk=3 all fit comfortably (1280-1440 KB for the in0 CB), so a feasible config demonstrably
exists.
NOT a regression from this campaign: the earlier corpus snapshot in `regime_a_current_perf.json` already records
this shape as `cls: "picker_infeasible", cfg: null`. The fix is contained -- fall back to searching Sm>1 in the
anchor step when no Sm=1 config is feasible -- but it is a picker-coverage fix rather than a perf change, so it
is left un-implemented pending a decision.

### FPU utilization added to the sweep table

Chip FLOPs are DOCUMENTED in this repo, so no guessing was needed:
`tech_reports/GEMM_FLOPS/GEMM_FLOPS.md` states the BH matrix engine computes `8x16 x 16x16` in a single cycle =
`2*8*16*16 = 4096` FLOP/cycle, that MATH_FIDELITY divides that, and gives the per-engine table
(LoFi ~5.4 / HiFi2 ~2.7 / HiFi4 ~1.35 TFLOPS at 1.35 GHz). This op is bf16 in/out at **HiFi2** with fp32
accumulation, so its peak is **2048 FLOP/cycle/core = 2.765 TFLOP/s per core**. fp32 dest accumulation costs DST
capacity, not MAC throughput.

Grid: device-queried `compute_with_storage_grid_size` = **11x10 = 110 cores** on this board => **304 TFLOP/s**
full-grid peak. Deliberately NOT the 13x10 = 130 the tech report quotes for Blackhole generally -- this board has
harvested columns, and using 130 would understate utilization by 18%.

Two columns, because they answer different questions:
- **FPU%grid** = achieved / 304 TFLOP/s -- utilization of the whole chip.
- **FPU%core** = achieved / (allocated cores x 2.765 TFLOP/s) -- utilization of the cores the op actually used.
The gap between them is purely "grid not fully used" (the op picks 24-96 of 110 cores).

| metric | value |
|---|---|
| achieved | min 9.2, median **30.5**, max 160.9 TFLOP/s |
| FPU util vs full 110-core grid | median **10.0%**, max 52.9% (512x6144x4608) |
| FPU util vs allocated cores | median **20.3%**, max 60.6% (512x6144x4608) |

**This is the expected shape of a DRAM-bandwidth-optimal matmul and not a defect:** median 85% of peak DRAM
against median 10% of peak FPU. These are low-arithmetic-intensity shapes (M<<N), so the roofline says they are
memory-bound by a wide margin; the FPU is idle because there is nothing for it to do while the data streams. The
one shape with high FPU utilization, 512x6144x4608 at 60.6% of its allocated cores, is exactly the one shape the
earlier roofline work found to be COMPUTE-floor-bound rather than DRAM-bound -- an independent confirmation from
a completely different measurement.

**Cross-check of the peak against measured data.** Comparing the theoretical FPU time against the
independently-measured compute floors (diag mask 2101, from the ablation campaign):

| shape | theoretical us | measured compute-only | ratio | compute/wall | FPU%wall |
|---|---|---|---|---|---|
| 512x6144x4608 | 109.2 | 139.2 | 1.27 | 77% | 60.6% |
| 512x6144x2304 | 54.6 | 70.0 | 1.28 | 64% | 49.7% |
| 256x6144x6144 | 72.8 | 90.2 | 1.24 | 48% | 38.8% |
| 256x6144x4608 | 54.6 | 68.6 | 1.26 | 49% | 38.6% |
| 256x15360x1536 | 45.5 | 55.5 | 1.22 | 41% | 33.6% |
| 256x6080x4640 | 54.4 | 86.1 | 1.58 | 58% | 36.6% |

The ratio is a consistent 1.22-1.28x (one outlier at 1.58x), i.e. **60-80% FPU efficiency inside the compute
phase**, the balance being unpack/pack/reconfig overhead. That consistency is the validation: a wrong peak would
give ratios below 1.0 (physically impossible) or scatter wildly across shapes.

### HEYGEN SHAPES: roofline + regime-A filter, then default-config perf/correctness

23 candidate shapes supplied. Kept only those BOTH memory-bound by roofline AND servable by regime-A; ran the
survivors at DEFAULTS. Table in `HEYGEN_SWEEP.md`.

**Filter.** Machine balance = 304 TFLOP/s (110 cores x 2.765 TFLOP/s bf16 HiFi2) / 512 GB/s = **594 FLOP/byte**;
memory-bound iff `MNK/(MK+KN+MN)` < 594. Regime-A structural requirements, all consequences of the 8-bank in0
ring and 8-bank in1 width shard rather than tunables:
- Nt wide enough to shard 8 ways: `7*ceil(Nt/8) < Nt`
- **Kt >= 8** -- the k-slice spans exactly 8 banks, so the minimum slice is 8 tiles and the picker's hard 20%
  K-waste cap rejects anything smaller
- M < N -- regime A shards in1 on the assumption it is the big operand; M >= N is regime B

Result: **12 kept, 11 dropped**. Dropped for compute-bound (AI 739-1817, the 2656- and 10560-row shapes),
Nt too narrow (N=64 -> Nt=2; N=128 -> Nt=4), or M>=N. Of the 12 kept, **32x128x30720 was then rejected by the
picker at runtime**: Kt=4 < 8 forces >=100% K padding. My pre-filter checked the Nt shard rule but not the Kt>=8
rule, and the run caught it -- Kt>=8 is now written into the documented filter.

**11 measured, sorted ascending by effective DRAM bandwidth:** median 437 GB/s (85% of peak), 7 of 11 above 80%,
11/11 PCC >= 0.999 (min 0.99998), median block-to-block spread 0.1%.

Two findings specific to this set:

1. **32x256x512 is far off the roofline: 4.29 us for a 0.6 us DRAM floor, 76 GB/s (15% of peak), only 16 cores.**
   It is a genuinely tiny problem (Kt=8, Nt=16) and the picker can only field 16 cores on it, so the wall is
   fixed-overhead-dominated rather than bandwidth-dominated. Nothing in this op's design targets that size; a
   shape this small wants a different (single-core or small-grid) matmul entirely.

2. **The three 512-row shapes are the closest thing to balanced in this corpus** (AI 394-427 against a 594
   balance) and they show it: 512x5120x5120 reaches **64.8% FPU utilization on its allocated cores and 47.2% of
   the whole grid** -- the highest FPU numbers measured anywhere in this campaign, well above the Mt<=16 corpus
   median of 20.3%. They sit at only 61-67% of peak DRAM because compute, not DRAM, is most of their wall. They
   are the shapes in this set where further work would have to attack the FPU side.

Also of note: `96x2048x5120` and `96x8192x5120` carry **1.19x schedule padding** (the largest in either sweep),
i.e. the picked config's capacity exceeds the logical shape by 19%. Both still reach 91-93% of peak DRAM because
padded positions are never DRAM-read, but the padding is paid in compute and L1.

### F15. FAILED: more cores on the Mt=16 HeyGen shapes. The default picker is optimal on all three.

Question: the three Mt=16 HeyGen shapes run on 80/80/96 of 104 available cores -- can more cores help? Swept 31
explicit configs across them (20 diverse + 11 controlled), all else at production defaults.

**First: what is even reachable.** Core count is quantized to `8*Pk*Ns*Sm` and capped at kMaxCores=104, so >80
cores needs `Pk*Ns*Sm >= 11`. 104 cores needs the product to be 13, which is prime, so Pk=13 -- and at Kt=160
that is 30% K waste, above the picker's hard 20% cap. **96 is therefore the practical ceiling, not 104.**

**Result: every alternative lost, and the default won on all three shapes.**

| shape | default | best alternative | penalty |
|---|---|---|---|
| 512x5120x5120 | 10,1,1,2,1 (80c) **187.35 us** | 11,1,1,2,1 (88c) 213.59 | **+14.0%** |
| 512x5120x2560 | 10,1,1,2,1 (80c) **109.39 us** | 11,1,1,2,1 (88c) 135.46 | **+23.8%** |
| 512x4096x5120 | 4,3,1,2,1 (96c) **172.55 us** | 8,1,1,2,1 (64c) 203.75 | **+18.1%** |

Three independent mechanisms, each of which the picker already models:

1. **Extra cores via higher Pk buy nothing, by construction.** At Kt=160 the default Pk=10/kb=2 gives K_slice=16
   with `Pk*K_slice = 160 = Kt` exactly -- ZERO K padding. Pk=11 needs 10% padding and Pk=12 needs 20%. Since
   padded K tiles are still multiplied (the tails are zero-FILLED, not skipped), compute work rises by exactly
   the padding while cores rise by the same fraction: `12/10 cores` against `192/160 work` is a perfect wash.
   The measured +14 to +26% is the residue -- more ring traffic (ring bytes scale with `Pk*K_slice`) and a
   deeper reduction chain (depth Pk-1).
2. **Extra cores via Sm are catastrophic here: +102% to +470%.** M-split has the in1 reader read once and
   FORWARD to its Sm-1 slaves, so NoC forward traffic is `(Sm-1) x K*N*2`. in1 is the dominant operand on these
   shapes (52 MB on 512x5120x5120), so Sm=4 pushes ~157 MB over the NoC -- three times the entire DRAM traffic.
   M-split only pays when in1 is small relative to in0, i.e. narrow N, which is exactly what the picker's
   narrow-N (kNbandMax) hysteresis already encodes.
3. **Extra cores via Ns cost DRAM.** Ns duplicates the in0 read Ns times. On 512x4096x5120 the default *does*
   take that trade (Ns=3, 96 cores, DRAM floor 116.7 us vs 100.4 at Ns=1) and it is still the best option,
   because the Ns=1 alternatives cap at Pk=8 -> only 64 cores.

**Also a lesson about my own method.** My first candidate set was generated by minimising a compute proxy that
did not model kb, so it selected kb=1 configs -- and shallow kb is independently known in this campaign to hurt
compute efficiency badly. That inflated the more-core configs unfairly. I re-ran a controlled set holding kb=2
(the default's value) with Ns=1, Sm=1 so that **Pk/core-count was the only variable**; the conclusion did not
change, but the penalties dropped from +46..+52% to +14..+26%, i.e. roughly half of the first result was my
proxy's kb bias rather than the core count. **A candidate generator whose cost model is cruder than the
production picker's will systematically slander the production pick.**

All 31 runs correct (PCC 0.99997-1.00011) and stable (block spread <=0.5%). No code change; nothing to ship.

## FINALIZATION (2026-07-30): golden perf suite, cleanup, coverage audit

### 1. Golden performance regression suite

`tests/ttnn/unit_tests/operations/matmul/test_regime_a_matmul_perf.py` -- 10 shapes, 85 s, 10/10 pass.
Chosen so every production path is covered: Mt 1/2/4/8/16, chain AND reduce-scatter, bank-local AND in1-near
AND mesh, 7-228 us, 24-96 cores. Thresholds are the measured medians plus a margin matched to the measured
noise floor (8% under 30 us where iteration spread reaches 12%, 5% above). Every regression this campaign
caught was 9-33%, so the margins neither flake nor hide anything. Each case also asserts PCC on the same
program it times, so a perf pass cannot mask a numerical break.

Measurement note worth keeping: **the profiler CSV is only written when the device CLOSES.** Verified directly
-- it is absent after `ttnn.synchronize_device`, still absent after `ttnn.ReadDeviceProfiler`, and appears only
after `close_device`. So per-test device timing inside one pytest process is not possible; each shape is
measured in its own subprocess through the existing `prod_sweep_worker.py` (which also isolates shapes and
keeps the run-host-id demux in one place). Those tests must therefore NOT take the `device` fixture.

### 2. Removed all diagnostic / experimental infrastructure (-1427 lines)

The program factory alone went 1930 -> 950 lines. Deleted: the `TT_REGIME_A_DIAG_MASK` and
`TT_REGIME_A_CB1_DEPTH` env vars and their two operation attributes (so the program-cache key now contains
only real inputs); all 26 diagnostic bits; the ~819 lines of dead host helpers behind them (`RingLinkModel`,
`regroup_in0_rings`, `balance_in0_ring_order`, `balance_in0_ring_order_bg`, `place_in1_optimal`); the
`PlanInputs` cb1/cb7-depth and `reduce_meet` knobs (now fixed constants `kCb1Depth=4`, `kCb7Depth=2`); the
meet-in-the-middle `CorePlan` fields; and the TRID-only cb1-depth compile arg.

Kept, all production and all measured: chain, ring reduce-scatter with its S8/S9 gate, the 2D mesh gate,
IN1_NEAR M-split placement, PARETO in0 ring ordering, subblock enlargement, coalesced in1 reads, balanced
tails, fusion, chunked output. One env var survives -- `TT_REGIME_A_LOG_CFG`, observability only, logging the
factory's own pick/reduction/placement once per cache miss. It is how the sweep tables report ground truth
(the host-side picker mirror is NOT validated against `auto_select_config`), so it was deliberately retained.

**The cleanup was validated, not asserted: 111/111 correctness AND 10/10 perf thresholds after removal.**

### 3. Coverage audit -- and it found a real gap

Ran the main 111-test suite with the config log to see which production paths it actually reaches:

| programs | reduction | placement |
|---|---|---|
| 99 | chain | bank-local |
| 10 | chain | in1-near |
| 1 | chain | mesh |
| **0** | **reduce-scatter** | **--** |

**The correctness suite never exercised ring reduce-scatter at all**, despite it shipping on 14 of the 66
corpus shapes. Mesh placement had a single program and Ns>1 a single config. So "111/111 pass" was silent
about the reduction path on a fifth of the corpus. (It was not unverified overall -- 4 of the 10 golden perf
shapes are reduce-scatter and assert PCC -- but the suite carrying the tails, fusion and cache cases was.)

Added `test_regime_a_matmul_audit.py`, 29 tests, 12 s, closing that and the other gaps:
- **reduce-scatter**: shallow-K, Sm>1, deep-K-via-Pk<=6, and both UNEVEN chunk partitions (rs_T%Pk != 0:
  9-over-4 and 6-over-4), each at both an explicit config and config=None; plus the gate yielding to fusion
  (a reduce-scatter-shaped problem WITH bias must fall back to the chain and still be correct).
- **mesh**: 4 shapes including mesh+reduce-scatter together and mesh with non-divisible Kt and Nt.
- **Pk/Ns/Sm grid** unfused on one shape, including Ns=4 and Mt=8 with Sm=3 (M-split tail).
- **program-cache DISCRIMINATION**, which the main suite structurally could not test -- it only ever replays
  the SAME program twice, proving address refresh but not that distinct entries never cross-serve. That is
  exactly the failure class that invalidated four experiments in this campaign. Now interleaved in one
  process: two configs on one shape; three shapes; a reduce-scatter shape against a chain shape; unfused vs
  bias-fused vs chunked. Plus explicit-config-equals-auto-pick asserted BIT-EXACT against config=None, and
  config=None asserted deterministic across cache hits.

After the audit the two suites together cover **all six** reduction x placement combinations:
19 reduce-scatter programs (11 in1-near, 5 bank-local, 3 mesh) and 17 chain.

No bug was found -- every added test passed first time. The finding is that a shipped path was correct but
unverified by the suite that gates changes to it.

### 3b. The audit gap is CLOSED, and the closure is itself guarded

Two corrections to the first pass at this, both found by re-reading my own claims:

1. **I documented a guarantee I had not implemented.** The audit file's header said "each test asserts the path
   it intends to cover is really taken" -- no test did. Coverage was correct the day it was written (verified
   via the config log) but would have rotted SILENTLY: if a gate change moved those shapes back to the chain,
   every test would still pass while the reduce-scatter coverage quietly vanished. Exactly the decay the
   sentence claimed to prevent.
2. `test_audit_path_coverage_guard` now implements it: it asserts every `_RSCATTER` case still selects
   `reduction=reduce-scatter`, every `_MESH` case still selects `placement=mesh`, and the audit set as a whole
   still spans both reductions and all three placements -- read from the factory's own log, which is ground
   truth. A failure names the shape and what it switched to.

Implementation notes worth keeping:
- A first attempt ran the probe in a SUBPROCESS. It passed alone and timed out (>300 s) inside the full file:
  pytest already holds a device, so the child could not open one. The `device` fixture is FUNCTION-scoped, so
  each test gets a fresh device and therefore a fresh PROGRAM CACHE -- every case in the guard is a cache miss
  and logs exactly once. So the guard runs in-process with `capfd` capturing the C++ logger at fd level. No
  subprocess, no device contention. (The perf suite genuinely needs subprocesses, for a different reason: the
  profiler CSV only lands at device close. It gets away with it because no test in that file takes the fixture.)
- **The guard was verified to have teeth**, not assumed: temporarily forcing `rscatter = rs_gate && false` and
  rebuilding made it fail with
  `COVERAGE ROT: shallowK_sm1 (64x2048x1024 cfg=4,2,1,2,2) now selects reduction=chain, not reduce-scatter`.
  Gate restored and rebuilt afterwards. A guard that cannot fail is worth nothing.
- This is also why `TT_REGIME_A_LOG_CFG` was kept in the cleanup: the guard depends on it, so the one surviving
  env var now has a test that fails if the logging is removed.

**Final state: 111 correctness + 30 audit (incl. the coverage guard) + 10 golden perf = 151 tests, all
passing.**
