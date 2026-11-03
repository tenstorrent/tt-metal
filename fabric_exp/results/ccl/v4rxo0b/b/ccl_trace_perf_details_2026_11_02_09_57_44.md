# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_57_44
CCLs: sdpa_reduce_to_all

## sdpa_reduce_to_all

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_sdpa_reduce_to_all.py::test_sdpa_reduce_to_all_trace
Configured benchmark shape: [8, 4096]
Configured transport mode: fixed forwarder topology
Setup: 2 forwarder cores total; each forwarder core binds to one physical link, with BRISC/NCRISC handling the two traffic directions on that core.
Configured num_l_chunks: 4
Configured compute_block_sizes: 8
Sweep: num_l_chunks x compute_block_size x max_payload_size_bytes
For this shape, total_l_tiles per worker = 16.
Tuned payload matrix: 2048 -> (4, 8), 4096 -> (2, 4), 8192 -> (2, 8), 15232 -> (2, 8).
Invalid cells are shown as n/a when a forced L chunk payload exceeds the configured fabric max payload or a compute block size does not divide the total L tiles.
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_l_chunks=4, tiles_per_l_chunk=4, block_size=8, max_payload_size_bytes=2048

Report: generated/profiler/reports/2026_11_02_09_57_58/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5748.8     | 4258.363 | 4.258  | 4290.0     | 5758.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4593.3     | 3402.430 | 3.402  | 3908.0     | 4648.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5539.3     | 4103.185 | 4.103  | 4099.0     | 5570.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6251.2     | 4630.548 | 4.631  | 4765.0     | 6366.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7223.5     | 5350.719 | 5.351  | 5822.0     | 7281.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6368.9     | 4717.711 | 4.718  | 5528.0     | 6330.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6386.5     | 4730.711 | 4.731  | 5423.0     | 6509.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5906.2     | 4374.993 | 4.375  | 5092.0     | 5871.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4546.6     | 3367.874 | 3.368  | 3764.0     | 4440.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5630.5     | 4170.770 | 4.171  | 4802.0     | 5612.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6505.3     | 4818.726 | 4.819  | 5698.0     | 6479.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7409.9     | 5488.844 | 5.489  | 6588.0     | 7383.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6309.4     | 4673.652 | 4.674  | 5182.0     | 5991.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6200.4     | 4592.896 | 4.593  | 5707.0     | 6173.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5879.9     | 4355.481 | 4.355  | 5170.0     | 5812.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4552.8     | 3372.407 | 3.372  | 3790.0     | 4471.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5611.6     | 4156.733 | 4.157  | 4879.0     | 5556.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6489.2     | 4806.807 | 4.807  | 5789.0     | 6415.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7390.1     | 5474.178 | 5.474  | 6691.0     | 7325.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6325.9     | 4685.830 | 4.686  | 5425.0     | 5927.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6220.9     | 4608.081 | 4.608  | 5599.0     | 6175.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5724.9     | 4240.689 | 4.241  | 4187.0     | 5755.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4567.5     | 3383.333 | 3.383  | 3712.0     | 4672.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5515.4     | 4085.452 | 4.085  | 3922.0     | 5612.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6241.6     | 4623.393 | 4.623  | 4666.0     | 6331.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7197.7     | 5331.637 | 5.332  | 5702.0     | 7227.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6376.0     | 4722.985 | 4.723  | 4871.0     | 6310.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6362.5     | 4712.970 | 4.713  | 4541.0     | 6546.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5815.0     | 4307.381 | 4.307  | 5724.9                | 5814.3                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4565.0     | 3381.511 | 3.382  | 4546.6                | 4560.1                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5574.2     | 4129.035 | 4.129  | 5515.4                | 5575.4                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6371.8     | 4719.869 | 4.720  | 6241.6                | 6370.2                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7305.3     | 5411.344 | 5.411  | 7197.7                | 7306.8                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6345.1     | 4700.044 | 4.700  | 6309.4                | 6347.4                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6292.6     | 4661.165 | 4.661  | 6200.4                | 6291.7                   |

Final op time: 5.411 us (7305.3 cycles), bucket=TRISC_2 compute

