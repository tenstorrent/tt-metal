# Read-only depth sweep — corrects the N-sub/split-K story (stride ≈ 0; multi-reader caps ~440)

Pure-read microbench (`sp_bankread/test_bank_read`, `bank_reader.cpp`): P readers per DRAM bank,
bank-adjacent, each reads its share of the bank shard (Kt=192, N_band=18 tiles = 6.75 MB/bank, 54 MB
total) in **contiguous** (K-slice) or **strided** (N-sub sub-band) mode, sweeping pipeline depth. No
compute. Kernel-time BW.

## Data (GB/s)
| config | d2 | d4 | d8 | d16 | d32 |
|---|--:|--:|--:|--:|--:|
| P1 contig (whole bank, 8c) | 331 | 405 | 454 | 481 | **494** |
| P2 contig (K-slice, 16c) | 382 | 371 | 352 | 343 | 337 |
| P2 strided (ns=9, 16c) | 381 | 364 | 336 | 329 | 331 |
| P3 contig (K-slice, 24c) | **436** | 403 | 389 | 400 | 397 |
| P3 strided (ns=6, 24c) | **443** | 407 | 405 | 393 | 399 |
| P6 contig/strided (48c) | 401 | 397 | — | — | — |

## Three corrections to earlier conclusions
1. **Stride penalty ≈ 0.** Contiguous (K-slice) and strided (N-sub) reads are within noise at every
   (P, depth). My earlier "N-sub is stride-capped (~28 %)" was WRONG — it was confounded with depth.
2. **Multi-reader-per-bank wants SHALLOW depth; deep HURTS.** Opposite of P=1. P2/P3 peak at d2 and
   fall monotonically with depth (P2: 382→337). P1 is the reverse (331→494) — a single reader needs a
   deep pipeline to fill its bank; multiple readers already fill it and deep just adds contention.
3. **Multi-reader read caps ~440 (P=3 sweet spot), NOT 494.** Only 1 reader/bank + deep hits 494.
   Any P≥2 tops out ~380–443 (best P=3, d2 = 443 ≈ 87 % of 510). More readers (P6) don't help (401).

## Consequence: split-K gives NO read advantage over N-sub
Since contiguous == strided, split-K's contiguous K-slices read no faster than N-sub's strided
sub-bands. So **don't build split-K** (its reduction is pure overhead). Use N-sub (no reduction).

## The real M>1 ceiling and the gap
- **M=1 (8 cores, 1/bank, deep): 494 (97 % util) — the clean win (1.25×), unaffected.**
- **M>1 (needs >8 compute cores ⇒ P≥2 readers/bank): read ceiling ~440 (87 %), stride-independent,
  shallow-depth-optimal.** So the sharded approach can at best reach ~85 % util on M>1 (a modest
  ~1.1× over the branch's ~75 %), never the 97 % of M=1.
- **Current matmul N-sub P=3 gets only 70 % / 0.94× — below the 440 read ceiling** because
  `reader_subband` pipelines too deep (~2·kb outstanding) and deep hurts multi-reader. Reaching ~440
  needs the matmul reader retuned to the shallow-optimal (few outstanding per reader), plus confirming
  compute (24 cores) isn't the co-limiter.

## Recommendation
- Ship the op as **M=1-tile Regime A (the strong, clean 1.25× win)**; **fall back to the reuse/branch
  matmul for M>1** where the sharded read ceiling (~440) only buys ~1.1× and needs delicate tuning.
- If M>1 is worth pursuing: retune `reader_subband` to shallow multi-reader depth (match the d2
  optimum), P=3, and re-measure — target ~85 % util. Do NOT pursue split-K (no read benefit).
