# Recorded device profile

Full 60-layer Gemma4 31B, CP=8 / TP=4; revision `c06c59b1f6a66dc1e324abdfe39a920a0b1b1045`. 5 warmed samples per cell. Representative timeline is the median sample's longest-span chip.

Legacy: one 8192-token request. Hybrid: 512 + 512 + 1024 + 1536 + 2048 + 2560. Every request has the selected prefix; KV history is synthetic. No correctness/PCC run is included.

## Additive breakdown (ms)

| Prefix | Component | Legacy | Hybrid | Added time |
|---:|---|---:|---:|---:|
| 0 | Global SDPA | 6.688 | 9.711 | +3.023 |
| 0 | Sliding-window SDPA | 14.540 | 42.997 | +28.457 |
| 0 | Attention projection matmuls | 14.514 | 14.534 | +0.021 |
| 0 | MLP matmuls | 31.605 | 31.654 | +0.049 |
| 0 | TP collectives | 26.964 | 27.091 | +0.127 |
| 0 | KV updates | 0.937 | 2.554 | +1.618 |
| 0 | Copies / layout / typecast | 6.164 | 10.385 | +4.221 |
| 0 | Other programs | 9.422 | 11.051 | +1.629 |
| 0 | Device gaps | 0.390 | 10.079 | +9.689 |
| 0 | **Total** | **111.224** | **160.056** | **+48.832** |
| 253,952 | Global SDPA | 161.010 | 398.847 | +237.837 |
| 253,952 | Sliding-window SDPA | 16.721 | 60.624 | +43.903 |
| 253,952 | Attention projection matmuls | 14.518 | 14.606 | +0.088 |
| 253,952 | MLP matmuls | 31.633 | 31.701 | +0.068 |
| 253,952 | TP collectives | 28.038 | 26.654 | -1.384 |
| 253,952 | KV updates | 0.936 | 2.590 | +1.654 |
| 253,952 | Copies / layout / typecast | 6.100 | 10.275 | +4.175 |
| 253,952 | Other programs | 9.414 | 11.071 | +1.657 |
| 253,952 | Device gaps | 0.388 | 8.487 | +8.099 |
| 253,952 | **Total** | **268.757** | **564.854** | **+296.097** |

## Hybrid SDPA time per request

Mean op time over the 10 global / 50 sliding-window layers in the representative replay. Both include internal CCL.

| Prefix | Request | Tokens | Global ms/op | Global total ms | SWA ms/op | SWA total ms |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 512 | 0.141 | 1.407 | 0.067 | 3.374 |
| 0 | 2 | 512 | 0.141 | 1.409 | 0.069 | 3.433 |
| 0 | 3 | 1024 | 0.135 | 1.350 | 0.118 | 5.903 |
| 0 | 4 | 1536 | 0.148 | 1.483 | 0.152 | 7.582 |
| 0 | 5 | 2048 | 0.150 | 1.502 | 0.220 | 11.008 |
| 0 | 6 | 2560 | 0.256 | 2.560 | 0.234 | 11.697 |
| 253,952 | 1 | 512 | 4.972 | 49.720 | 0.118 | 5.888 |
| 253,952 | 2 | 512 | 4.957 | 49.568 | 0.141 | 7.027 |
| 253,952 | 3 | 1024 | 5.046 | 50.462 | 0.212 | 10.615 |
| 253,952 | 4 | 1536 | 5.601 | 56.015 | 0.201 | 10.033 |
| 253,952 | 5 | 2048 | 5.093 | 50.931 | 0.269 | 13.467 |
| 253,952 | 6 | 2560 | 14.215 | 142.151 | 0.272 | 13.594 |

## Sample ranges

| Path | Prefix | Min–max device ms | Programs per chip | Representative chip |
|---|---:|---:|---:|---:|
| legacy | 0 | 111.192–111.482 | 1985 | 9 |
| legacy | 253,952 | 268.524–268.937 | 1985 | 7 |
| hybrid | 0 | 159.933–160.687 | 3198 | 26 |
| hybrid | 253,952 | 564.567–565.062 | 3198 | 5 |

All 20 samples have complete, non-overlapping program spans on the selected chip. Category totals plus gaps equal the measured device span. Matmul and SDPA counts/order are checked against the 60-layer model. Per-chip clocks are normalized independently; the tracks are separate replays, not simultaneous execution.
