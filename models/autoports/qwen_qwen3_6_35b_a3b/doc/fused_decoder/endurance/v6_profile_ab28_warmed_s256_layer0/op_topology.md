# ab28 layer 0: standard warmed S256 profiler capture

One complete warmed prefill and 16 complete warmed traced decode passes. Device kernel sums exclude host dispatch and profiler gaps; use separate clean benchmarks for latency ranking.

| Phase | Ops/pass | Kernel us/pass | Gap us/pass |
|---|---:|---:|---:|
| prefill | 11988 | 465714.035 | 115175.738 |
| decode | 207 | 1413.651 | 291813.820 |

## prefill largest operation classes

| Operation | Count/pass | Kernel us/pass |
|---|---:|---:|
| GenericOpDeviceOperation | 36 | 299284.277 |
| BinaryNgDeviceOperation | 3380 | 53957.773 |
| AccumulationDeviceOperation | 528 | 44342.511 |
| GroupedMatmulDeviceOperation | 32 | 19506.626 |
| UntilizeDeviceOperation | 524 | 9941.631 |
| TilizeWithValPaddingDeviceOperation | 1296 | 9721.453 |
| GatherDeviceOperation | 560 | 4924.986 |
| UntilizeWithUnpaddingDeviceOperation | 1044 | 4138.764 |
| SliceDeviceOperation | 1928 | 4059.122 |
| ReshapeViewDeviceOperation | 512 | 2816.073 |
| EmbeddingsDeviceOperation | 16 | 2609.000 |
| ReshapeCodegenDeviceOperation | 564 | 2399.636 |

## decode largest operation classes

| Operation | Count/pass | Kernel us/pass |
|---|---:|---:|
| GenericOpDeviceOperation | 7 | 512.409 |
| BinaryNgDeviceOperation | 65 | 249.664 |
| SparseMatmulDeviceOperation | 2 | 174.168 |
| FastReduceNCDeviceOperation | 13 | 63.757 |
| UnaryDeviceOperation | 17 | 49.508 |
| TilizeWithValPaddingDeviceOperation | 10 | 47.905 |
| MatmulDeviceOperation | 2 | 41.193 |
| TernaryDeviceOperation | 3 | 40.406 |
| TransposeDeviceOperation | 7 | 39.559 |
| SliceDeviceOperation | 24 | 35.943 |
| ReshapeViewDeviceOperation | 4 | 34.638 |
| UntilizeWithUnpaddingDeviceOperation | 7 | 19.362 |

Full operation sequences, advice-bearing tt-perf-report text/CSV tables, raw source hashes, warning assessments, and strict completeness checks are retained beside this summary.
