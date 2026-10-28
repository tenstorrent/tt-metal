# Round 1 end: analysis (analyst: Opus, low, read-only; profile job 1209 on c97b6f32cae)

Reduced-layer estimate 272.5 ms against bench 272.7 ms (paper / accuracy). One mid layer is 51.34 ms: matmuls
30.9 ms (60 %), binary ops 12.6, scan 5.0, norms 2.8. The bench is traced; op-to-op gaps are about 0.5 us.

| op (layer_mid) | ms | note |
|---|---|---|
| B proj (bf16 x fp32 -> fp32, HiFi4) | 8.22 | 22 % of the HiFi4 peak; fp32-write floor about 2.1 ms |
| C proj (fp32 x fp32 -> bf16, HiFi4) | 6.46 | 28 %; bandwidth floor about 2.6 ms |
| w1 + silu | 5.48 | 16 % of HiFi2 |
| scan (generic op, 32 cores) | 5.01 | about 430 GB/s: bandwidth-bound at paper |
| w2 | 4.71 | |
| mul a*b | 3.58 | 1.6 GB round trip |
| gate + sigmoid | 3.11 | |
| w3 | 2.92 | 31 % |
| D*u, g*y, add, z+y, u+n | 1.80-1.86 each | about 445 GB/s |
| layer_norm, rms_norm | 1.43, 1.37 | about 380 GB/s |

long_context (734 ms) is scan-bound (16 series, 4x longer serial chains, 32 cores); it has no profile yet.
