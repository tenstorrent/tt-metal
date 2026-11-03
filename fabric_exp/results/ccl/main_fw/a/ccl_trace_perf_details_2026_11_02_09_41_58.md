# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_41_58
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

### num_l_chunks=2, tiles_per_l_chunk=8, block_size=4, max_payload_size_bytes=4352

Report: generated/profiler/reports/2026_11_02_09_42_08/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6059.6     | 4488.585 | 4.489  | 4495.0     | 6039.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3832.9     | 2839.156 | 2.839  | 3277.0     | 3649.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5796.5     | 4293.696 | 4.294  | 4204.0     | 5768.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6646.8     | 4923.541 | 4.924  | 5080.0     | 6637.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7559.0     | 5599.252 | 5.599  | 5992.0     | 7534.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5445.9     | 4034.015 | 4.034  | 4894.0     | 5294.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5380.9     | 3985.822 | 3.986  | 4816.0     | 5279.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5995.2     | 4440.904 | 4.441  | 4751.0     | 5973.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5453.0     | 4039.281 | 4.039  | 4327.0     | 5533.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5756.6     | 4264.170 | 4.264  | 4513.0     | 5745.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6555.0     | 4855.585 | 4.856  | 5357.0     | 6527.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7482.4     | 5542.541 | 5.543  | 6233.0     | 7446.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 7092.4     | 5253.615 | 5.254  | 5364.0     | 7207.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 7032.7     | 5209.430 | 5.209  | 5187.0     | 7146.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5975.5     | 4426.319 | 4.426  | 4638.0     | 5909.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5448.1     | 4035.637 | 4.036  | 4362.0     | 5529.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5732.8     | 4246.489 | 4.246  | 4361.0     | 5657.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6535.1     | 4840.837 | 4.841  | 5236.0     | 6511.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7472.7     | 5535.326 | 5.535  | 6146.0     | 7426.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 7059.9     | 5229.541 | 5.230  | 4501.0     | 7202.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 7022.7     | 5202.000 | 5.202  | 5395.0     | 7172.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6061.4     | 4489.919 | 4.490  | 4493.0     | 6041.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3762.8     | 2787.289 | 2.787  | 3196.0     | 3551.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5795.6     | 4293.052 | 4.293  | 4218.0     | 5776.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6649.6     | 4925.593 | 4.926  | 4969.0     | 6643.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7561.9     | 5601.378 | 5.601  | 6004.0     | 7554.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5402.9     | 4002.119 | 4.002  | 4728.0     | 5204.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5371.4     | 3978.800 | 3.979  | 4288.0     | 5252.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 6022.9     | 4461.431 | 4.461  | 5975.5                | 6027.4                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4624.2     | 3425.341 | 3.425  | 3762.8                | 4640.5                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5770.4     | 4274.352 | 4.274  | 5732.8                | 5776.1                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6596.6     | 4886.389 | 4.886  | 6535.1                | 6600.9                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7519.0     | 5569.624 | 5.570  | 7472.7                | 7520.7                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6250.3     | 4629.822 | 4.630  | 5402.9                | 6252.9                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6201.9     | 4594.013 | 4.594  | 5371.4                | 6201.8                   |

Final op time: 5.570 us (7519.0 cycles), bucket=TRISC_2 compute

