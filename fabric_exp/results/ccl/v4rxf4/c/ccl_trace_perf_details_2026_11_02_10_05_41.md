# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_05_41
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

Report: generated/profiler/reports/2026_11_02_10_05_55/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5774.6     | 4277.519 | 4.278  | 4769.0     | 5694.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5000.5     | 3704.074 | 3.704  | 4366.0     | 5017.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5541.5     | 4104.837 | 4.105  | 4541.0     | 5449.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6350.5     | 4704.089 | 4.704  | 5360.0     | 6264.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7267.7     | 5383.496 | 5.383  | 6304.0     | 7189.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6802.9     | 5039.193 | 5.039  | 4158.0     | 6849.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6789.0     | 5028.881 | 5.029  | 6066.0     | 6854.0        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5824.1     | 4314.119 | 4.314  | 4906.0     | 5716.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4070.9     | 3015.511 | 3.016  | 3734.0     | 3937.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5549.4     | 4110.681 | 4.111  | 4608.0     | 5427.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6423.2     | 4757.919 | 4.758  | 5487.0     | 6309.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7329.8     | 5429.481 | 5.429  | 6397.0     | 7219.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5891.2     | 4363.867 | 4.364  | 5499.0     | 5774.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5811.0     | 4304.415 | 4.304  | 5215.0     | 5756.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5782.2     | 4283.089 | 4.283  | 5008.0     | 5655.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4071.2     | 3015.696 | 3.016  | 3767.0     | 3976.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5515.0     | 4085.178 | 4.085  | 4739.0     | 5390.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6391.7     | 4734.615 | 4.735  | 5619.0     | 6264.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7294.9     | 5403.644 | 5.404  | 6498.0     | 7165.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5882.5     | 4357.415 | 4.357  | 5120.0     | 5812.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5793.5     | 4291.489 | 4.291  | 5297.0     | 5799.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5754.3     | 4262.437 | 4.262  | 4859.0     | 5680.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5016.1     | 3715.593 | 3.716  | 4434.0     | 4984.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5536.2     | 4100.881 | 4.101  | 4557.0     | 5465.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6342.8     | 4698.400 | 4.698  | 5441.0     | 6265.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7253.0     | 5372.607 | 5.373  | 6351.0     | 7205.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6808.9     | 5043.593 | 5.044  | 5568.0     | 6807.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6758.8     | 5006.526 | 5.007  | 3979.0     | 6855.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5783.8     | 4284.291 | 4.284  | 5754.3                | 5778.4                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4539.7     | 3362.719 | 3.363  | 4070.9                | 4535.8                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5535.5     | 4100.394 | 4.100  | 5515.0                | 5538.9                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6377.1     | 4723.756 | 4.724  | 6342.8                | 6371.1                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7286.4     | 5397.307 | 5.397  | 7253.0                | 7281.3                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6346.4     | 4701.017 | 4.701  | 5882.5                | 6347.1                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6288.1     | 4657.828 | 4.658  | 5793.5                | 6284.9                   |

Final op time: 5.397 us (7286.4 cycles), bucket=TRISC_2 compute

