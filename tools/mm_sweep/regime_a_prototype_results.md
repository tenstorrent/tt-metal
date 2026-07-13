# Regime-A prototype — FLUX/LTX skinny sweep (8 bank-adjacent cores)

Standalone `regime_a_mm` prototype vs the existing branch (`bh_skinny_results.md`). 8 cores, one reader
per DRAM bank, bank-adjacent placement, in1 bank-contiguous (16 KB bursts, triple-buffered TRIDs),
NCRISC does in0+output, compute = real `compute.cpp`. Constant-input correctness (out==K).
BW-util uses the table metric 2*(MK+KN+MN)/time/500 GB/s. best kb chosen per shape.

| shape | AI | best kb | µs | BW-util % | branch µs | speedup |
|---|--:|--:|--:|--:|--:|--:|
| 32×2048×512 | 30 | 8 | 8.4 | 53.7 | 9.4 | **1.12×** |
| 32×2048×1536 | 31 | 4 | 16.7 | 78.1 | 18.8 | **1.13×** |
| 32×2048×2048 | 31 | 4 | 21.2 | 81.4 | 23.1 | **1.09×** |
| 32×6144×1536 | 31 | 4 | 45.4 | 85.3 | 51.9 | **1.14×** |
| 32×6144×2304 | 31 | 8 | 65.5 | 88.1 | 78.3 | **1.20×** |
| 32×6144×3072 | 32 | 2 | 83.0 | 92.3 | 105.5 | **1.27×** |
| 32×6144×6144 | 32 | 2 | 159.0 | 96.0 | 199.6 | **1.26×** |
| 32×6144×9216 | 32 | 2 | 235.0 | 97.2 | 293.4 | **1.25×** |
| 64×6144×1536 | 61 | 4 | 64.2 | 61.9 | 51.7 | 0.81× |
| 64×15360×1536 | 61 | 8 | 153.7 | 64.2 | 137.4 | 0.89× |
| 64×4608×6144 | 62 | 4 | 190.5 | 60.9 | 153.8 | 0.81× |
| 64×6144×4608 | 62 | 4 | 188.5 | 61.5 | 156.1 | 0.83× |
| 128×6144×768 | 108 | 8 | 65.7 | 34.1 | 37.1 | 0.56× |
| 128×15360×768 | 109 | 8 | 156.3 | 35.5 | 86.6 | 0.55× |
| 128×6144×2304 | 119 | 8 | 189.4 | 32.2 | 82.5 | 0.44× |
| 32×256×6144 | 28 | — | FAIL/L1 (shallow-K edge) | | 11.2 | |
| 64×6144×9216 | 63 | — | FAIL/L1 | | 296.1 | |
| 128×2304×6144 / 128×6144×4608 / 512×6144×1536 | | — | FAIL/L1 | | | |

## Analysis
**M=32 (1 M-tile): read-bound, WINS all shapes — 1.09–1.27× (geomean ~1.18×), BW-util up to 97 %.**
These are the hardest DRAM-bound GEMV-like skinny shapes and the prototype nails them. BW-util rises
with N (more in1 to amortize): 32×2048×512 = 54 % (small in1) → 32×6144×9216 = 97 % (at the ~494 GB/s
read ceiling). The largest = biggest wins (1.25–1.27×).

**M≥64: compute-bound at 8 cores, LOSES.** M=64 → 0.81–0.89×; M=128 → 0.44–0.56×. Exactly SP2's
prediction: min(M,N) ≥ 64 needs > 8 compute cores (M=64 ~14, M=128 ~27), but the 8-bank-reader model
gives only 8. The branch wins these via its N-slicing/K-par filling more cores. **This is the known M>1
limitation** — the fix is > 8 cores (2 readers/bank, keeping the read bank-limited ~510 while spreading
compute), not a block-size change.

**FAIL/L1:** big output blocks (M_block·N_block fp32 intermediate) + the deep in1 CB overflow L1 at
M≥2 / large N; plus a shallow-K edge (32×256×6144). Fixable with per-shape CB sizing / smaller blocks;
does not change the conclusions.

## Block size — how important? (answer: NOT the primary lever in Regime A)
kb (K-block) is the only real block knob (N_block=Nt/8 fixed by 8 cores; M_block=Mt fixed by shape;
subblock auto). Two findings:

1. **kb is mostly CONSTRAINT-PINNED, not freely tunable.** The 16 KB-burst requirement forces
   in1_blk = kb·N_block to be a multiple of 8 tiles, and L1 caps the deep CB. For most shapes only ONE
   kb is valid (e.g. 32×6144×2304 N_block=9 → only kb=8 legal; 32×6144×9216 N_block=36 → only kb=2
   fits L1). So there is little to sweep.
2. **Where multiple kb are legal it matters modestly (~1.3×) only on small/under-amortized shapes.**
   32×2048×512: kb4 = 39 % vs kb8 = 53 % BW-util (larger block → fewer pipeline barriers → helps a
   small read). On the large read-bound shapes the single feasible kb already hits 92–97 %.

## N-sub-division (P readers/bank) — MEASURED, read-capped by stride (does NOT reach peak)
Extended to 8P cores: P readers per bank, each owning a contiguous N-sub-band `[K, ns]`, ns=N_band/P,
full K, no reduction (kernels: `reader_subband.cpp`; host `--preaders P`). Goal: add compute cores for
the M≥64 shapes losing at 8 cores.

**Result: it helps modestly over P=1 (1.06–1.17×) but does NOT flip the shapes to wins — BW-util stays
~60–65 % (M=64) / ~37–52 % (M=128).** Per-RISC breakdown (64×6144×4608 P=2) shows all stages balanced
~240 K cyc, and the read (BRISC) is only ~320–355 GB/s — the limiter.

**Root cause = the sub-band read is STRIDED, not contiguous.** A k-major bank shard `[K, N_band]`
sub-divided by N gives per-row reads of `ns` contiguous tiles, then a jump (skip `N_band−ns`) to the
next row → DRAM row-buffer thrashing. Measured read (fast `one_packet_with_state` path, so not a code
artifact):
| ns (contiguous run) | read GB/s |
|---|---:|
| 8 (16 KB) | ~355 |
| 6 (12 KB) | ~347 |
| 3 (6 KB) | ~199 |
Even at 16 KB runs the stride costs ~28 % vs contiguous whole-bank (494) — the jump reopens DRAM rows.
Smaller runs drop further. This answers the two things to measure:
- **Multi-reader-per-bank penalty** ≈ 0.9× (SP1/SP5; contiguous 2/bank ~445) — mild.
- **The stride penalty** ≈ 0.78× on top (→ ~350) — dominant, and unavoidable while sub-dividing a
  k-major shard by N.

**⇒ N-sub-division is the wrong axis for M>1.** The contiguous extension is **split-K** (each of P
cores reads a contiguous K-slice of its bank → no stride → keeps ~445–490, + a cheap reduction for
small-M). Recommend pivoting to split-K for the M≥64 shapes. (Alternatively, a finer *n-major* / sub-
block DRAM shard layout would make N-sub contiguous — but that changes the in1 layout; split-K works
with the natural k-major shard.)

**Why block size matters little here:** Regime A is READ-bound, so the matmul blocking (kb, subblock)
is hidden under the in1 read. The read rate is set by **placement (bank-adjacent) + 16 KB bursts +
pipeline depth**, not by the compute block. This is the OPPOSITE of the compute-only microbench (SP2),
where deep-K was critical (79 %→91 %) — because there compute *was* the bottleneck. Lesson: in the
DRAM-bound regime, tune the read (placement/burst/pipeline); block size is secondary and largely
determined by feasibility constraints.
