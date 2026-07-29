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

### TOP REMAINING LEVER (measured, specified, not yet built): the reduction chain is 9-14% deep

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
