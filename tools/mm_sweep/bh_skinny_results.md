# Blackhole p150b (11x10) — minimal_matmul skinny FLUX/LTX two-phase sweep

Two-phase methodology: **baseline** = optimized-main (branch build, all levers OFF: S1Pk1, no large-N
levers / auto-prefetch / auto-K-par) + full block sweep; **branch** = joint (S,Pk,blocking) sweep, levers
ON. Speedup = baseline-best us / branch-best us (each at its own optimal blocking). HiFi2, bf16.

- Compute peak: **304.1 TFLOP/s** (11x10 @ 1.35 GHz). DRAM peak assumed **500 GB/s**.
- **MAC-util %** = achieved / 304 TFLOP/s. **BW-util %** = achieved DRAM BW / 500 GB/s,
  using minimal traffic = 2*(MK+KN+MN) bytes (bf16, each operand read/written once = ideal reuse;
  K-par/slicing re-reads make true traffic higher, so BW-util here is a LOWER bound).
- **AI** = 2*MKN / [2*(MK+KN+MN)] FLOP/byte (roofline). BH ridge point = 608 FLOP/byte;
  AI << ridge => DRAM-bound (these skinny shapes), so BW-util is the meaningful occupancy metric.

**29 shapes, geomean speedup 1.72x (min 1.28x, max 2.42x). All win.**

| shape (M×K×N) | AI (F/B) | base µs | base MAC% | base BW% | branch µs | br MAC% | br BW% | best (S,Pk) | branch block | speedup |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--|---:|
| 32×2048×32 | 16 | 16.3 | 0.1 | 3.2 | 8.6 | 0.2 | 6.1 | S1Pk8 | mb1kb4nb1sb1x1 | **1.89×** |
| 32×2048×512 | 30 | 22.7 | 1.0 | 19.9 | 9.4 | 2.4 | 48.1 | S1Pk8 | mb1kb4nb2sb1x2 | **2.42×** |
| 32×2048×1536 | 31 | 40.6 | 1.6 | 32.1 | 18.8 | 3.5 | 69.4 | S4Pk2 | mb1kb4nb2sb1x2 | **2.16×** |
| 32×6144×1536 | 31 | 105.9 | 1.9 | 36.6 | 51.9 | 3.8 | 74.6 | S4Pk2 | mb1kb4nb2sb1x2 | **2.04×** |
| 32×2048×2048 | 31 | 46.2 | 1.9 | 37.4 | 23.1 | 3.8 | 74.7 | S2Pk4 | mb1kb8nb3sb1x1 | **2.00×** |
| 32×6144×2304 | 31 | 136.3 | 2.2 | 42.3 | 78.3 | 3.8 | 73.7 | S1Pk8 | mb1kb8nb4sb1x4 | **1.74×** |
| 32×6144×3072 | 32 | 174.4 | 2.3 | 44.0 | 105.5 | 3.8 | 72.7 | S1Pk4 | mb1kb4nb8sb1x4 | **1.65×** |
| 32×256×6144 | 28 | 23.9 | 1.4 | 29.7 | 11.2 | 2.9 | 63.2 | S2Pk4 | mb1kb4nb3sb1x1 | **2.13×** |
| 32×6144×6144 | 32 | 308.3 | 2.6 | 49.5 | 199.6 | 4.0 | 76.4 | S1Pk4 | mb1kb4nb2sb1x2 | **1.54×** |
| 32×6144×9216 | 32 | 454.6 | 2.6 | 50.3 | 293.4 | 4.1 | 77.9 | S1Pk4 | mb1kb4nb4sb1x4 | **1.55×** |
| 64×6144×1536 | 61 | 106.0 | 3.8 | 37.5 | 51.7 | 7.7 | 76.8 | S4Pk2 | mb2kb4nb2sb2x2 | **2.05×** |
| 64×15360×1536 | 61 | 232.5 | 4.3 | 42.5 | 137.4 | 7.2 | 71.8 | S1Pk4 | mb1kb4nb3sb1x1 | **1.69×** |
| 64×6144×4608 | 62 | 249.3 | 4.8 | 46.5 | 156.1 | 7.6 | 74.3 | S2Pk2 | mb1kb4nb8sb1x4 | **1.60×** |
| 64×4608×6144 | 62 | 236.3 | 5.0 | 49.1 | 153.8 | 7.8 | 75.4 | S2Pk2 | mb1kb4nb3sb1x1 | **1.54×** |
| 64×6144×9216 | 63 | 454.0 | 5.2 | 50.8 | 296.1 | 8.1 | 77.8 | S2Pk2 | mb1kb4nb4sb1x4 | **1.53×** |
| 128×6144×768 | 108 | 70.8 | 5.6 | 31.6 | 37.1 | 10.7 | 60.5 | S1Pk8 | mb4kb4nb3sb4x1 | **1.91×** |
| 128×15360×768 | 109 | 154.6 | 6.4 | 35.9 | 86.6 | 11.5 | 64.0 | S2Pk4 | mb4kb4nb2sb4x1 | **1.78×** |
| 128×6144×2304 | 119 | 137.0 | 8.7 | 44.5 | 82.5 | 14.4 | 73.9 | S4Pk1 | mb2kb4nb2sb2x2 | **1.66×** |
| 128×6144×4608 | 122 | 250.7 | 9.5 | 47.4 | 155.4 | 15.3 | 76.4 | S2Pk2 | mb2kb4nb8sb1x4 | **1.61×** |
| 128×2304×6144 | 119 | 132.4 | 9.0 | 46.0 | 84.9 | 14.0 | 71.8 | S2Pk2 | mb2kb4nb6sb2x2 | **1.56×** |
| 512×6144×128 | 101 | 46.1 | 5.7 | 34.7 | 28.0 | 9.4 | 57.0 | S1Pk8 | mb2kb4nb4sb1x4 | **1.65×** |
| 512×6144×1536 | 361 | 111.5 | 28.5 | 48.0 | 87.0 | 36.5 | 61.5 | S1Pk2 | mb4kb4nb6sb2x2 | **1.28×** |
| 1024×6144×128 | 112 | 71.3 | 7.4 | 40.4 | 44.2 | 12.0 | 65.3 | S1Pk2 | mb3kb8nb1sb1x1 | **1.61×** |
| 1216×4096×32 | 31 | 57.1 | 1.8 | 36.1 | 28.9 | 3.6 | 71.4 | S2Pk4 | mb2kb8nb1sb2x1 | **1.98×** |
| 2048×6144×128 | 118 | 121.5 | 8.7 | 44.9 | 74.6 | 14.2 | 73.1 | S1Pk8 | mb3kb4nb4sb1x4 | **1.63×** |
| 4096×6144×128 | 122 | 218.7 | 9.7 | 48.4 | 142.1 | 14.9 | 74.5 | S2Pk2 | mb2kb8nb2sb2x2 | **1.54×** |
| 4864×4096×32 | 32 | 173.0 | 2.4 | 46.7 | 104.6 | 4.0 | 77.3 | S2Pk4 | mb3kb8nb1sb1x1 | **1.65×** |
| 8192×6144×128 | 123 | 402.0 | 10.5 | 51.9 | 271.1 | 15.6 | 77.0 | S4Pk1 | mb2kb4nb2sb2x2 | **1.48×** |
| 16384×6144×128 | 124 | 777.0 | 10.9 | 53.3 | 524.8 | 16.1 | 78.9 | S1Pk4 | mb4kb8nb2sb4x1 | **1.48×** |
