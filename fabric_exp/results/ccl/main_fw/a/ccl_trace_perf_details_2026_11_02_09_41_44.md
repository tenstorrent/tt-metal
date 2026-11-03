# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_41_44
CCLs: sdpa_reduce_to_all

## sdpa_reduce_to_all

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_sdpa_reduce_to_all.py::test_sdpa_reduce_to_all_trace
Configured benchmark shape: [8, 4096]
Configured transport mode: fixed forwarder topology
Setup: 2 forwarder cores total; each forwarder core binds to one physical link, with BRISC/NCRISC handling the two traffic directions on that core.
Configured num_l_chunks: 2
Configured compute_block_sizes: 4
Sweep: num_l_chunks x compute_block_size x max_payload_size_bytes
For this shape, total_l_tiles per worker = 16.
Tuned payload matrix: 2048 -> (4, 8), 4096 -> (2, 4), 8192 -> (2, 8), 15232 -> (2, 8).
Invalid cells are shown as n/a when a forced L chunk payload exceeds the configured fabric max payload or a compute block size does not divide the total L tiles.
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_l_chunks=2, tiles_per_l_chunk=8, block_size=4, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_02_09_41_56/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6050.3     | 4481.719 | 4.482  | 3995.0     | 6297.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4626.9     | 3427.296 | 3.427  | 3331.0     | 4715.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5797.7     | 4294.563 | 4.295  | 3832.0     | 6035.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6620.8     | 4904.281 | 4.904  | 4548.0     | 6830.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7547.3     | 5590.578 | 5.591  | 5448.0     | 7809.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6287.6     | 4657.474 | 4.657  | 4277.0     | 6329.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6267.1     | 4642.281 | 4.642  | 4956.0     | 6295.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6136.6     | 4545.630 | 4.546  | 4533.0     | 6101.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4830.6     | 3578.185 | 3.578  | 3564.0     | 4935.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5875.9     | 4352.526 | 4.353  | 4503.0     | 5826.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6710.4     | 4970.659 | 4.971  | 5059.0     | 6692.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7625.2     | 5648.267 | 5.648  | 5967.0     | 7592.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6242.7     | 4624.207 | 4.624  | 4884.0     | 6125.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6121.1     | 4534.156 | 4.534  | 4910.0     | 6060.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6085.5     | 4507.807 | 4.508  | 4612.0     | 6133.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4816.1     | 3567.496 | 3.567  | 3478.0     | 4867.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5831.1     | 4319.319 | 4.319  | 4347.0     | 5841.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6671.9     | 4942.133 | 4.942  | 5216.0     | 6712.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7587.1     | 5620.111 | 5.620  | 6130.0     | 7636.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6200.9     | 4593.222 | 4.593  | 4816.0     | 6065.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6080.0     | 4503.719 | 4.504  | 4937.0     | 5966.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6066.0     | 4493.304 | 4.493  | 4069.0     | 6299.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4621.2     | 3423.081 | 3.423  | 3358.0     | 4724.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5821.6     | 4312.296 | 4.312  | 3801.0     | 6069.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6623.2     | 4906.081 | 4.906  | 4493.0     | 6914.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7559.4     | 5599.593 | 5.600  | 5525.0     | 7814.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6256.4     | 4634.363 | 4.634  | 4751.0     | 6328.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6264.5     | 4640.378 | 4.640  | 3922.0     | 6315.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 6084.6     | 4507.115 | 4.507  | 6050.3                | 6075.8                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4723.7     | 3499.015 | 3.499  | 4621.2                | 4721.5                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5831.6     | 4319.676 | 4.320  | 5797.7                | 5826.3                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6656.6     | 4930.789 | 4.931  | 6620.8                | 6647.5                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7579.8     | 5614.637 | 5.615  | 7547.3                | 7573.3                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6246.9     | 4627.317 | 4.627  | 6200.9                | 6249.5                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6183.2     | 4580.133 | 4.580  | 6080.0                | 6192.8                   |

Final op time: 5.615 us (7579.8 cycles), bucket=TRISC_2 compute

