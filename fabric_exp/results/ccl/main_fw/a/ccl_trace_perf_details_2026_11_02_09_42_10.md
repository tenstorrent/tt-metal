# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_42_10
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

Report: generated/profiler/reports/2026_11_02_09_42_22/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5977.9     | 4428.044 | 4.428  | 3953.0     | 5957.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4667.3     | 3457.230 | 3.457  | 3614.0     | 4819.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5772.3     | 4275.778 | 4.276  | 3897.0     | 5685.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6522.5     | 4831.474 | 4.831  | 4302.0     | 6552.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7469.5     | 5532.985 | 5.533  | 5401.0     | 7464.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6274.1     | 4647.444 | 4.647  | 5111.0     | 6202.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6332.0     | 4690.363 | 4.690  | 5236.0     | 6509.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6346.7     | 4701.281 | 4.701  | 5594.0     | 6326.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4769.2     | 3532.741 | 3.533  | 3418.0     | 4761.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6069.7     | 4496.074 | 4.496  | 5331.0     | 6042.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6945.1     | 5144.548 | 5.145  | 6200.0     | 6925.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7842.4     | 5809.200 | 5.809  | 7115.0     | 7822.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6177.5     | 4575.956 | 4.576  | 4858.0     | 5677.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5943.6     | 4402.667 | 4.403  | 4923.0     | 5674.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6319.5     | 4681.104 | 4.681  | 5646.0     | 6321.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4753.4     | 3521.044 | 3.521  | 3558.0     | 4768.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6050.3     | 4481.696 | 4.482  | 5361.0     | 6038.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6929.5     | 5132.948 | 5.133  | 5971.0     | 6915.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7827.6     | 5798.244 | 5.798  | 7062.0     | 7822.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6174.8     | 4573.926 | 4.574  | 4691.0     | 5718.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5967.7     | 4420.541 | 4.421  | 4852.0     | 5692.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5967.3     | 4420.237 | 4.420  | 4014.0     | 5892.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4650.2     | 3444.622 | 3.445  | 3527.0     | 4827.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5751.2     | 4260.156 | 4.260  | 3864.0     | 5615.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6505.6     | 4818.963 | 4.819  | 4396.0     | 6469.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7450.5     | 5518.874 | 5.519  | 5445.0     | 7386.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6282.8     | 4653.911 | 4.654  | 5125.0     | 6243.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6310.7     | 4674.578 | 4.675  | 3762.0     | 6478.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 6152.8     | 4557.667 | 4.558  | 5967.3                | 6148.7                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4710.0     | 3488.909 | 3.489  | 4650.2                | 4710.3                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5910.9     | 4378.426 | 4.378  | 5751.2                | 5911.3                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6725.7     | 4981.983 | 4.982  | 6505.6                | 6726.0                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7647.5     | 5664.826 | 5.665  | 7450.5                | 7648.6                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6227.3     | 4612.809 | 4.613  | 6174.8                | 6225.8                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6138.5     | 4547.037 | 4.547  | 5943.6                | 6139.2                   |

Final op time: 5.665 us (7647.5 cycles), bucket=TRISC_2 compute

