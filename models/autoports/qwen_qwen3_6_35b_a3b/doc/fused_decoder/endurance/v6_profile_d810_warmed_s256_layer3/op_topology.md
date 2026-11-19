# d810 layer 3: standard warmed S256 profiler capture

One complete warmed prefill and 16 complete warmed traced decode passes. Device kernel sums exclude host dispatch and profiler gaps; use separate clean benchmarks for latency ranking.

| Phase | Ops/pass | Kernel us/pass | Gap us/pass |
|---|---:|---:|---:|
| prefill | 104 | 8732.648 | 2137.068 |
| decode | 68 | 549.995 | 252106.539 |

## prefill largest operation classes

| Operation | Count/pass | Kernel us/pass |
|---|---:|---:|
| GroupedMatmulDeviceOperation | 8 | 6559.657 |
| EmbeddingsDeviceOperation | 4 | 503.672 |
| MatmulDeviceOperation | 5 | 469.599 |
| BinaryNgDeviceOperation | 33 | 290.535 |
| SliceDeviceOperation | 21 | 232.188 |
| RoutedCombineDeviceOperation | 4 | 201.288 |
| SDPAOperation | 1 | 109.000 |
| LayerNormDeviceOperation | 4 | 80.352 |
| ReshapeCodegenDeviceOperation | 1 | 55.992 |
| RoutingMetadataDeviceOperation | 4 | 55.140 |
| TransposeDeviceOperation | 3 | 52.667 |
| TopKDeviceOperation | 1 | 43.935 |

## decode largest operation classes

| Operation | Count/pass | Kernel us/pass |
|---|---:|---:|
| MatmulDeviceOperation | 5 | 189.484 |
| SparseMatmulDeviceOperation | 2 | 173.832 |
| BinaryNgDeviceOperation | 8 | 32.886 |
| LayerNormDeviceOperation | 4 | 22.917 |
| SdpaDecodeDeviceOperation | 1 | 22.454 |
| SliceDeviceOperation | 15 | 21.386 |
| NLPCreateQKVHeadsDecodeDeviceOperation | 1 | 18.803 |
| ReshapeViewDeviceOperation | 1 | 10.324 |
| UnaryDeviceOperation | 2 | 8.969 |
| PagedUpdateCacheDeviceOperation | 2 | 8.189 |
| ShardedToInterleavedDeviceOperation | 5 | 6.502 |
| RotaryEmbeddingDeviceOperation | 2 | 6.251 |

Full operation sequences, advice-bearing tt-perf-report text/CSV tables, raw source hashes, warning assessments, and strict completeness checks are retained beside this summary.
