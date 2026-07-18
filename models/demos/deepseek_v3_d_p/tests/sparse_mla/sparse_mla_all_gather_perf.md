# Sparse MLA all-gather performance

Measured on the LoudBox SP=2 × TP=4 proxy with the realtime profiler. The current
implementation includes native `ttnn.all_gather` in MLA and the indexer, plus
row-major top-k-index gathering.

Negative latency deltas mean the current implementation is faster.

| Model | Case | Legacy total (ms) | Current total (ms) | Latency delta | Legacy collective (ms) | Current collective (ms) | Collective delta |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepSeek v3.2 | warm | 9.414 | 8.955 | -4.9% | 1.227 | 0.941 | -23.3% |
| DeepSeek v3.2 | cold | 88.466 | 83.932 | -5.1% | 13.066 | 10.387 | -20.5% |
| DeepSeek v3.2 | long | 19.415 | 17.066 | -12.1% | 7.843 | 6.274 | -20.0% |
| GLM 5.1 | warm | 7.950 | 6.351 | -20.1% | 3.859 | 2.389 | -38.1% |
| GLM 5.1 | cold | 83.342 | 65.347 | -21.6% | 42.721 | 26.750 | -37.4% |
| GLM 5.1 | long | 17.442 | 13.880 | -20.4% | 10.507 | 7.715 | -26.6% |

Collective time is accounted as follows:

- Legacy: `AllGatherAsync + AllBroadcast`.
- Current: native all-gather kernels reported as `ccl`, plus `AllBroadcast`.

Warm/long collective program counts:

| Model | Legacy | Current |
|---|---:|---:|
| DeepSeek v3.2 | 5 `AllGatherAsync` + 2 `AllBroadcast` | 6 `ccl` + 1 `AllBroadcast` |
| GLM 5.1 | 7 `AllGatherAsync` + 2 `AllBroadcast` | 8 `ccl` + 1 `AllBroadcast` |

The remaining current `AllBroadcast` is the indexer weight all-reduce. Its
reduce-scatter output contains four bf16 weights per TP rank, which is TILE-padded
and too small for the native row-major path's 16-byte NoC-write alignment.
