# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_42_36
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

Report: generated/profiler/reports/2026_11_02_09_42_51/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5574.3     | 4129.096 | 4.129  | 5017.0     | 5531.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3827.8     | 2835.444 | 2.835  | 3494.0     | 3813.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5314.9     | 3937.000 | 3.937  | 4754.0     | 5266.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6176.1     | 4574.859 | 4.575  | 5642.0     | 6135.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7077.5     | 5242.615 | 5.243  | 6521.0     | 7029.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5678.6     | 4206.341 | 4.206  | 4416.0     | 5685.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5520.4     | 4089.193 | 4.089  | 5220.0     | 5487.0        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5632.2     | 4172.007 | 4.172  | 5342.0     | 5576.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5227.9     | 3872.556 | 3.873  | 4766.0     | 5206.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5381.6     | 3986.363 | 3.986  | 5105.0     | 5322.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6220.1     | 4607.519 | 4.608  | 5912.0     | 6173.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7132.7     | 5283.496 | 5.283  | 6830.0     | 7078.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 7026.2     | 5204.593 | 5.205  | 6590.0     | 7012.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 7028.9     | 5206.593 | 5.207  | 6395.0     | 7017.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5619.2     | 4162.356 | 4.162  | 5375.0     | 5553.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5246.2     | 3886.052 | 3.886  | 4922.0     | 5217.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5369.2     | 3977.178 | 3.977  | 5113.0     | 5320.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6199.0     | 4591.881 | 4.592  | 5969.0     | 6156.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7117.4     | 5272.133 | 5.272  | 6889.0     | 7055.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 7010.0     | 5192.600 | 5.193  | 6050.0     | 7010.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 7001.9     | 5186.600 | 5.187  | 6450.0     | 7005.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5557.0     | 4116.289 | 4.116  | 5078.0     | 5510.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3713.0     | 2750.363 | 2.750  | 3476.0     | 3683.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5299.2     | 3925.319 | 3.925  | 4828.0     | 5247.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6163.0     | 4565.215 | 4.565  | 5659.0     | 6112.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7064.6     | 5233.067 | 5.233  | 6576.0     | 7010.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5603.6     | 4150.837 | 4.151  | 5081.0     | 5589.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5478.3     | 4057.985 | 4.058  | 4601.0     | 5458.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5595.7     | 4144.937 | 4.145  | 5557.0                | 5596.7                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4503.7     | 3336.104 | 3.336  | 3713.0                | 4527.9                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5341.2     | 3956.465 | 3.956  | 5299.2                | 5342.1                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6189.6     | 4584.869 | 4.585  | 6163.0                | 6187.6                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7098.1     | 5257.828 | 5.258  | 7064.6                | 7097.5                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6329.6     | 4688.593 | 4.689  | 5603.6                | 6344.3                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6257.4     | 4635.093 | 4.635  | 5478.3                | 6261.2                   |

Final op time: 5.258 us (7098.1 cycles), bucket=TRISC_2 compute

