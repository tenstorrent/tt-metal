# r01-b03-a03 result: 1.1313 (ok)

Per shape (parent r01-b03-a02 in brackets): h3584 15.29 µs / 1.112 (15.38 / 1.105), h4096 16.39 / 1.114 (18.05 / 1.011),
h6144 20.44 / 1.149 (20.57 / 1.141), h7168 22.60 / 1.151 (23.15 / 1.124). PCC 0.9999985, max_abs unchanged on all
shapes. Geomean 1.1313 vs the parent's 1.0942 (+3.4%). This is the new best node (previous best r01-b01-a01, 1.1089).

## What happened vs expected
- h4096 recovered as predicted: -1.66 µs (-9.2%), now in line with the other shapes. This is the shape where every
  one of the 80 cores hit the same DRAM bank at each step (simulation: 80 cores/bank, now 10).
- h7168 -0.55 µs (-2.4%, above noise). h6144 -0.13 µs and h3584 -0.09 µs are within the ±1% noise. h3584 was already
  bank-balanced, so no change was expected there. h6144 (40 cores/bank before) gained less than I expected.

## Why (profiler evidence)
Medians over measured calls 3-12 and all 4 chips (/tmp/r01b03a03/agg2.py; µs from kernel start, max over worker cores
unless noted). Parent -> this node:

| shape | R_INPUT max end | W_PUSH max end | AG wait end | TRISC end - AG | DRAIN max end - AG | DRAIN mean end - AG |
|---|---|---|---|---|---|---|
| h3584 | 3.80 -> 3.71 | 5.58 -> 5.50 | 8.96 -> 8.73 | 4.33 -> 4.32 | 6.31 -> 6.33 | 4.56 -> 4.56 |
| h4096 | **5.29 -> 4.10** | **7.12 -> 5.95** | **10.39 -> 9.27** | 4.47 -> 4.47 | 7.36 -> 6.97 | 5.20 -> 4.97 |
| h6144 | 5.44 -> 5.77 | 7.37 -> 7.67 | 10.75 -> 10.58 | 4.98 -> 4.97 | 9.84 -> 9.26 | 6.84 -> 6.59 |
| h7168 | 6.88 -> 6.83 | 8.56 -> 8.60 | 10.91 -> 11.17 | 5.18 -> 5.18 | 10.70 -> 10.27 | 7.48 -> 7.35 |

1. **The h4096 win came from the INPUT READ, not the drain.** The slowest core's read end dropped 1.2 µs. PRE, the stick
   push and the AG start moved earlier by the same amount. With all 80 cores on one bank, that bank's queue serialized the
   ~4-tile block reads. Spreading the reads removes that.
2. **The drain is NOT bank-bound.** The drain tail after the AG shrank only 0.4-0.6 µs on h4096/h6144/h7168, even though
   per-step bank load went from 20-80 to 10 cores/bank. The slowest core still finishes its drain 2-5 µs after the
   TRISC end (h7168: compute ends AG+5.2 µs, last drain AG+10.3 µs). The parent's per-core zones showed the drain end
   growing with core x and y (h7168 dev 0: x=2 ~20-23 µs, x=14 ~26-27 µs), the signature of NoC0 write-path
   congestion (all 80 BRISC writers on NoC0, routes going +x/+y toward the DRAM columns), not DRAM bank queueing.
3. The rotation reorders the fp32 partial sum-of-squares across tiles. Accuracy is unchanged to 7 digits.

## Classification
win (+3.4% geomean over parent, new campaign best). Bank de-phasing fixes the read-side hot spot (large on h4096) and
trims the drain a little. The remaining drain tail is a different bottleneck.

## What a child of this node should try next
1. **Split the output drain across both NoCs.** NCRISC (reader, NoC1) is idle after its gamma read (ends well before the
   AG). Let BRISC write even CB positions on NoC0, and let NCRISC write odd positions on NoC1. Alternatively BRISC
   alternates the noc index per tile via the `noc` argument / a second `Noc` object on NoC1, which is simpler: no
   cross-RISC CB pop coordination. NoC1 routes go -x/-y, so the x/y-graded congestion should halve. Judge it with
   "DRAIN max end - AG" from /tmp/r01b03a03/agg2.py (it's 6.3-10.3 µs now, while TRISC ends at AG+4.3-5.2 µs).
2. Port r01-b01-a01's gamma reorder (x*gamma under the AG wait, single post-AG pass). TRISC end - AG is 4.3-5.2 µs now.
   It only pays once the drain stops being the tail (do 1 first, or stack both).
3. Keep the rotation (col_rot RT args in reader + worker writer, greedy choice in the factory). It is free and fixes
   bank-aligned shapes. If the drain is moved to two NoCs, keep the same rotated column mapping in both writers.
