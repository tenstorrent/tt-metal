# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_45_04
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

### num_l_chunks=2, tiles_per_l_chunk=8, block_size=8, max_payload_size_bytes=15232

Report: generated/profiler/reports/2026_11_02_09_45_19/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5890.4     | 4363.274 | 4.363  | 5590.0     | 5823.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4489.9     | 3325.844 | 3.326  | 4108.0     | 4437.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5658.7     | 4191.600 | 4.192  | 5357.0     | 5604.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6465.7     | 4789.385 | 4.789  | 6186.0     | 6413.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7385.2     | 5470.489 | 5.470  | 7081.0     | 7325.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6512.0     | 4823.681 | 4.824  | 5160.0     | 6547.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6547.1     | 4849.711 | 4.850  | 5956.0     | 6557.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5901.6     | 4371.578 | 4.372  | 5551.0     | 5829.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4513.1     | 3343.074 | 3.343  | 4138.0     | 4458.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5694.0     | 4217.770 | 4.218  | 5335.0     | 5636.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6479.9     | 4799.948 | 4.800  | 6070.0     | 6421.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7394.2     | 5477.200 | 5.477  | 7027.0     | 7328.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6586.5     | 4878.911 | 4.879  | 5878.0     | 6585.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6520.9     | 4830.274 | 4.830  | 6097.0     | 6537.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5872.5     | 4350.022 | 4.350  | 5556.0     | 5826.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4507.7     | 3339.052 | 3.339  | 4092.0     | 4472.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5683.4     | 4209.963 | 4.210  | 5338.0     | 5627.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6461.3     | 4786.133 | 4.786  | 6149.0     | 6418.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7376.8     | 5464.274 | 5.464  | 7061.0     | 7329.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6586.8     | 4879.074 | 4.879  | 4912.0     | 6603.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6530.1     | 4837.111 | 4.837  | 6100.0     | 6572.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5885.8     | 4359.881 | 4.360  | 5617.0     | 5836.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4454.1     | 3299.348 | 3.299  | 4116.0     | 4407.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5668.0     | 4198.496 | 4.198  | 5397.0     | 5608.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6453.7     | 4780.533 | 4.781  | 6120.0     | 6412.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7377.3     | 5464.637 | 5.465  | 7086.0     | 7338.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6505.7     | 4819.067 | 4.819  | 5793.0     | 6522.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6472.2     | 4794.193 | 4.794  | 4830.0     | 6486.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5887.6     | 4361.189 | 4.361  | 5872.5                | 5888.1                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4491.2     | 3326.830 | 3.327  | 4454.1                | 4498.8                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5676.0     | 4204.457 | 4.204  | 5658.7                | 5675.7                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6465.1     | 4789.000 | 4.789  | 6453.7                | 6463.5                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7383.4     | 5469.150 | 5.469  | 7376.8                | 7381.2                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6547.7     | 4850.183 | 4.850  | 6505.7                | 6549.2                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6517.6     | 4827.822 | 4.828  | 6472.2                | 6525.5                   |

Final op time: 5.469 us (7383.4 cycles), bucket=TRISC_2 compute

