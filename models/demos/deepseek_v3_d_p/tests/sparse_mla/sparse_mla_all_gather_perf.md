# Sparse MLA all-gather performance

Measured on the LoudBox SP=2 × TP=4 proxy with the realtime profiler. The current
implementation includes native `ttnn.all_gather` in MLA and the indexer, plus
row-major top-k-index gathering. It additionally gathers sparse-MLA KVPE directly
from the ND-sharded cache, selecting the active cache slot and transferring only
its populated prefix.

Negative latency deltas mean the current implementation is faster.

| Model | Case | Legacy total (ms) | Current total (ms) | Latency delta | Legacy collective (ms) | Current collective (ms) | Collective delta |
|---|---|---:|---:|---:|---:|---:|---:|
| DeepSeek v3.2 | warm | 9.414 | 9.020 | -4.2% | 1.227 | 0.996 | -18.8% |
| DeepSeek v3.2 | cold | 88.466 | 80.841 | -8.6% | 13.066 | 7.741 | -40.8% |
| DeepSeek v3.2 | long | 19.415 | 17.291 | -10.9% | 7.843 | 6.819 | -13.1% |
| GLM 5.1 | warm | 7.950 | 6.374 | -19.8% | 3.859 | 2.453 | -36.4% |
| GLM 5.1 | cold | 83.342 | 62.151 | -25.4% | 42.721 | 23.988 | -43.8% |
| GLM 5.1 | long | 17.442 | 14.125 | -19.0% | 10.507 | 8.309 | -20.9% |

Collective time is accounted as follows:

- Legacy: `AllGatherAsync + AllBroadcast`.
- Current: native all-gather kernels reported as `ccl`, plus `AllBroadcast`.

The cold proxy is the meaningful prefix-gather comparison: it fills an already
allocated cache over eleven iterations, reducing current total latency by 3.7 ms
(DeepSeek) and 3.2 ms (GLM) versus the prior native result. Warm and long allocate
their cache to exactly the measured context, so the populated prefix is the full
slab and they are expected to be within run-to-run noise of the prior result.

Warm/long collective program counts:

| Model | Legacy | Current |
|---|---:|---:|
| DeepSeek v3.2 | 5 `AllGatherAsync` + 2 `AllBroadcast` | 6 `ccl` + 1 `AllBroadcast` |
| GLM 5.1 | 7 `AllGatherAsync` + 2 `AllBroadcast` | 8 `ccl` + 1 `AllBroadcast` |

The remaining current `AllBroadcast` is the broadcast phase of one composite
all-gather fallback. Its tiled input has padding on the gather dimension, whereas
the native path requires that logical and padded extents match. It is unrelated to
the direct KVPE-prefix gather above.
