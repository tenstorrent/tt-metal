# rms_norm Regime-B transport bake-off (device kernel ns, lower=better)

| shape | mcast all-gather | root-relay | speedup |
|---|---|---|---|
| 1x1x32x4096 |    24.53 |    16.21 | 1.51x |
| 1x1x32x8192 |    45.15 |    21.97 | 2.06x |
| 1x1x64x8192 |    35.37 |    27.16 | 1.30x |
| 1x1x32x16384 |    53.77 |    30.52 | 1.76x |
| 1x1x32x32768 |   118.76 |    48.68 | 2.44x |
| 1x1x64x12288 |    46.94 |    33.08 | 1.42x |
