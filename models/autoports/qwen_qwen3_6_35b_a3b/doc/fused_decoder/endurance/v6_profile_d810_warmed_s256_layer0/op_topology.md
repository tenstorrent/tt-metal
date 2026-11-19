# d810 layer 0: standard warmed S256 profiler capture

One complete warmed prefill and 16 complete warmed traced decode passes. Device kernel sums exclude host dispatch and profiler gaps; use separate clean benchmarks for latency ranking.

| Phase | Ops/pass | Kernel us/pass | Gap us/pass |
|---|---:|---:|---:|
| prefill | 11920 | 465412.496 | 120044.156 |
| decode | 161 | 1306.809 | 283649.930 |

## prefill largest operation classes

| Operation | Count/pass | Kernel us/pass |
|---|---:|---:|
| GenericOpDeviceOperation | 44 | 299298.588 |
| BinaryNgDeviceOperation | 3380 | 53996.181 |
| AccumulationDeviceOperation | 512 | 44300.683 |
| GroupedMatmulDeviceOperation | 32 | 19400.835 |
| UntilizeDeviceOperation | 524 | 9942.918 |
| TilizeWithValPaddingDeviceOperation | 1280 | 9649.713 |
| GatherDeviceOperation | 560 | 4925.220 |
| UntilizeWithUnpaddingDeviceOperation | 1028 | 4097.225 |
| SliceDeviceOperation | 1912 | 4029.256 |
| ReshapeViewDeviceOperation | 512 | 2816.929 |
| EmbeddingsDeviceOperation | 16 | 2611.349 |
| ReshapeCodegenDeviceOperation | 564 | 2397.191 |

## decode largest operation classes

| Operation | Count/pass | Kernel us/pass |
|---|---:|---:|
| GenericOpDeviceOperation | 11 | 579.590 |
| BinaryNgDeviceOperation | 62 | 238.263 |
| SparseMatmulDeviceOperation | 2 | 173.206 |
| FastReduceNCDeviceOperation | 13 | 63.602 |
| MatmulDeviceOperation | 2 | 41.170 |
| UnaryDeviceOperation | 15 | 39.681 |
| TransposeDeviceOperation | 6 | 39.015 |
| ReshapeViewDeviceOperation | 4 | 34.536 |
| SliceDeviceOperation | 15 | 24.487 |
| ReshapeCodegenDeviceOperation | 8 | 19.069 |
| TypecastDeviceOperation | 11 | 17.152 |
| PermuteDeviceOperation | 2 | 12.979 |

Full operation sequences, advice-bearing tt-perf-report text/CSV tables, raw source hashes, warning assessments, and strict completeness checks are retained beside this summary.
