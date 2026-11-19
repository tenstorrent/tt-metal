# ab28 layer 3: standard warmed S256 profiler capture

One complete warmed prefill and 16 complete warmed traced decode passes. Device kernel sums exclude host dispatch and profiler gaps; use separate clean benchmarks for latency ranking.

| Phase | Ops/pass | Kernel us/pass | Gap us/pass |
|---|---:|---:|---:|
| prefill | 104 | 8721.485 | 2424.688 |
| decode | 68 | 549.400 | 246372.062 |

## prefill largest operation classes

| Operation | Count/pass | Kernel us/pass |
|---|---:|---:|
| GroupedMatmulDeviceOperation | 8 | 6540.014 |
| EmbeddingsDeviceOperation | 4 | 507.943 |
| MatmulDeviceOperation | 5 | 468.872 |
| BinaryNgDeviceOperation | 33 | 289.491 |
| SliceDeviceOperation | 21 | 230.405 |
| RoutedCombineDeviceOperation | 4 | 208.606 |
| SDPAOperation | 1 | 109.390 |
| LayerNormDeviceOperation | 4 | 79.990 |
| ReshapeCodegenDeviceOperation | 1 | 55.616 |
| RoutingMetadataDeviceOperation | 4 | 55.235 |
| TransposeDeviceOperation | 3 | 54.352 |
| TopKDeviceOperation | 1 | 43.926 |

## decode largest operation classes

| Operation | Count/pass | Kernel us/pass |
|---|---:|---:|
| MatmulDeviceOperation | 5 | 189.527 |
| SparseMatmulDeviceOperation | 2 | 173.219 |
| BinaryNgDeviceOperation | 8 | 32.858 |
| LayerNormDeviceOperation | 4 | 22.893 |
| SdpaDecodeDeviceOperation | 1 | 22.464 |
| SliceDeviceOperation | 15 | 21.559 |
| NLPCreateQKVHeadsDecodeDeviceOperation | 1 | 18.810 |
| ReshapeViewDeviceOperation | 1 | 10.338 |
| UnaryDeviceOperation | 2 | 8.975 |
| PagedUpdateCacheDeviceOperation | 2 | 8.173 |
| ShardedToInterleavedDeviceOperation | 5 | 6.510 |
| RotaryEmbeddingDeviceOperation | 2 | 6.215 |

Full operation sequences, advice-bearing tt-perf-report text/CSV tables, raw source hashes, warning assessments, and strict completeness checks are retained beside this summary.
