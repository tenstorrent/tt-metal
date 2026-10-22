# Recorded device profile

Full 60-layer Gemma4 31B, CP=8 / TP=4; revision `4934945c9e6c22dd4ae4d3d673ecc9eb6ffb93e2`. 5 warmed samples per cell. Representative timeline is the median sample's longest-span chip.

Legacy: one 8192-token request. Hybrid: 512 + 512 + 1024 + 1536 + 2048 + 2560. Every request has the selected prefix; KV history is synthetic. No correctness/PCC run is included.

## Additive breakdown (ms)

| Prefix | Component | Legacy | Hybrid | Added time |
|---:|---|---:|---:|---:|
| 0 | Global SDPA | 4.890 | 9.648 | +4.758 |
| 0 | Sliding-window SDPA | 15.810 | 43.033 | +27.223 |
| 0 | Attention projection matmuls | 14.500 | 14.552 | +0.052 |
| 0 | MLP matmuls | 31.601 | 31.630 | +0.029 |
| 0 | TP collectives | 27.579 | 26.799 | -0.780 |
| 0 | KV updates | 0.942 | 2.556 | +1.614 |
| 0 | Copies / layout / typecast | 6.064 | 10.283 | +4.220 |
| 0 | Other programs | 9.398 | 11.063 | +1.665 |
| 0 | Device gaps | 0.390 | 10.308 | +9.918 |
| 0 | **Total** | **111.172** | **159.872** | **+48.699** |
| 253,952 | Global SDPA | 161.064 | 276.964 | +115.900 |
| 253,952 | Sliding-window SDPA | 16.734 | 61.713 | +44.978 |
| 253,952 | Attention projection matmuls | 14.508 | 14.541 | +0.033 |
| 253,952 | MLP matmuls | 31.609 | 31.639 | +0.030 |
| 253,952 | TP collectives | 27.787 | 26.851 | -0.935 |
| 253,952 | KV updates | 0.942 | 2.574 | +1.632 |
| 253,952 | Copies / layout / typecast | 6.069 | 10.273 | +4.205 |
| 253,952 | Other programs | 9.400 | 11.074 | +1.673 |
| 253,952 | Device gaps | 0.391 | 8.461 | +8.070 |
| 253,952 | **Total** | **268.505** | **444.091** | **+175.586** |

## Hybrid SDPA time per request

Mean op time over the 10 global / 50 sliding-window layers in the representative replay. Both include internal CCL.

| Prefix | Request | Tokens | Global ms/op | Global total ms | SWA ms/op | SWA total ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 512 | 0.148 | 1.479 | 0.080 | 4.020 |
| 0 | 2 | 512 | 0.145 | 1.454 | 0.065 | 3.272 |
| 0 | 3 | 1024 | 0.126 | 1.256 | 0.121 | 6.060 |
| 0 | 4 | 1536 | 0.188 | 1.879 | 0.149 | 7.474 |
| 0 | 5 | 2048 | 0.162 | 1.624 | 0.213 | 10.638 |
| 0 | 6 | 2560 | 0.196 | 1.956 | 0.231 | 11.568 |
| 253,952 | 1 | 512 | 3.975 | 39.752 | 0.146 | 7.285 |
| 253,952 | 2 | 512 | 3.988 | 39.879 | 0.138 | 6.910 |
| 253,952 | 3 | 1024 | 4.442 | 44.421 | 0.212 | 10.607 |
| 253,952 | 4 | 1536 | 4.393 | 43.934 | 0.206 | 10.291 |
| 253,952 | 5 | 2048 | 5.149 | 51.486 | 0.254 | 12.715 |
| 253,952 | 6 | 2560 | 5.749 | 57.492 | 0.278 | 13.905 |

## Sample ranges

| Path | Prefix | Min–max device ms | Programs per chip | Representative chip |
|---|---:|---:|---:|---:|
| legacy | 0 | 111.115–111.307 | 1985 | 3 |
| legacy | 253,952 | 268.420–268.721 | 1985 | 7 |
| hybrid | 0 | 159.776–159.908 | 3198 | 31 |
| hybrid | 253,952 | 444.018–444.268 | 3198 | 11 |

All 20 samples have complete, non-overlapping program spans on the selected chip. Category totals plus gaps equal the measured device span. Matmul and SDPA counts/order are checked against the 60-layer model. Per-chip clocks are normalized independently; the tracks are separate replays, not simultaneous execution.
