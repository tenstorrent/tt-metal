# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_58_31
CCLs: sdpa_reduce_to_all

## sdpa_reduce_to_all

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_sdpa_reduce_to_all.py::test_sdpa_reduce_to_all_trace
Configured benchmark shape: [8, 4096]
Configured transport mode: fixed forwarder topology
Setup: 2 forwarder cores total; each forwarder core binds to one physical link, with BRISC/NCRISC handling the two traffic directions on that core.
Configured num_l_chunks: 2
Configured compute_block_sizes: 8
Sweep: num_l_chunks x compute_block_size x max_payload_size_bytes
For this shape, total_l_tiles per worker = 16.
Tuned payload matrix: 2048 -> (4, 8), 4096 -> (2, 4), 8192 -> (2, 8), 15232 -> (2, 8).
Invalid cells are shown as n/a when a forced L chunk payload exceeds the configured fabric max payload or a compute block size does not divide the total L tiles.
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_l_chunks=2, tiles_per_l_chunk=8, block_size=8, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_58_45/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5893.8     | 4365.763 | 4.366  | 3977.0     | 5853.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4128.2     | 3057.904 | 3.058  | 3279.0     | 4060.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5630.4     | 4170.637 | 4.171  | 3703.0     | 5571.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6459.7     | 4784.948 | 4.785  | 4572.0     | 6468.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7392.2     | 5475.726 | 5.476  | 5481.0     | 7371.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5777.6     | 4279.681 | 4.280  | 4369.0     | 5681.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5770.5     | 4274.444 | 4.274  | 4954.0     | 5685.0        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5978.4     | 4428.422 | 4.428  | 4728.0     | 5984.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4879.7     | 3614.600 | 3.615  | 3899.0     | 4882.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5704.2     | 4225.311 | 4.225  | 4457.0     | 5704.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6578.2     | 4872.733 | 4.873  | 5336.0     | 6587.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7475.2     | 5537.178 | 5.537  | 6224.0     | 7476.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6409.9     | 4748.081 | 4.748  | 5217.0     | 6393.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6286.8     | 4656.919 | 4.657  | 4965.0     | 6437.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5938.6     | 4398.941 | 4.399  | 4637.0     | 5942.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4846.8     | 3590.230 | 3.590  | 3916.0     | 4854.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5668.3     | 4198.770 | 4.199  | 4684.0     | 5668.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6547.8     | 4850.207 | 4.850  | 4990.0     | 6556.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7444.9     | 5514.726 | 5.515  | 6070.0     | 7440.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6373.4     | 4721.052 | 4.721  | 3745.0     | 6389.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6261.3     | 4638.030 | 4.638  | 4973.0     | 6403.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5872.6     | 4350.067 | 4.350  | 3854.0     | 5899.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4098.3     | 3035.778 | 3.036  | 3215.0     | 3971.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5615.8     | 4159.830 | 4.160  | 3624.0     | 5621.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6440.7     | 4770.919 | 4.771  | 4205.0     | 6505.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7369.9     | 5459.185 | 5.459  | 5285.0     | 7419.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5767.5     | 4272.200 | 4.272  | 4883.0     | 5664.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5748.2     | 4257.956 | 4.258  | 4005.0     | 5682.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5920.8     | 4385.798 | 4.386  | 5872.6                | 5916.2                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4488.2     | 3324.628 | 3.325  | 4098.3                | 4487.5                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5654.7     | 4188.637 | 4.189  | 5615.8                | 5649.4                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6506.6     | 4819.702 | 4.820  | 6440.7                | 6503.7                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7420.5     | 5496.704 | 5.497  | 7369.9                | 7418.6                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6082.1     | 4505.254 | 4.505  | 5767.5                | 6075.5                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6016.7     | 4456.837 | 4.457  | 5748.2                | 6015.9                   |

Final op time: 5.497 us (7420.5 cycles), bucket=TRISC_2 compute

