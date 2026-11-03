# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_05_57
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

Report: generated/profiler/reports/2026_11_02_10_06_11/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5725.1     | 4240.815 | 4.241  | 4460.0     | 5642.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4485.3     | 3322.437 | 3.322  | 3863.0     | 4412.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5518.8     | 4087.970 | 4.088  | 4167.0     | 5368.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6232.1     | 4616.407 | 4.616  | 5030.0     | 6240.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7198.7     | 5332.378 | 5.332  | 5953.0     | 7153.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6227.6     | 4613.030 | 4.613  | 5617.0     | 6237.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6232.5     | 4616.644 | 4.617  | 5241.0     | 6073.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5905.1     | 4374.119 | 4.374  | 4894.0     | 5916.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4748.8     | 3517.593 | 3.518  | 4236.0     | 4763.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5642.0     | 4179.230 | 4.179  | 4853.0     | 5644.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6500.5     | 4815.207 | 4.815  | 5412.0     | 6523.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7403.1     | 5483.763 | 5.484  | 6339.0     | 7427.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6530.1     | 4837.096 | 4.837  | 5268.0     | 6792.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6338.8     | 4695.370 | 4.695  | 5795.0     | 6349.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5878.2     | 4354.207 | 4.354  | 4877.0     | 5975.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4742.9     | 3513.281 | 3.513  | 4240.0     | 4771.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5619.5     | 4162.585 | 4.163  | 4713.0     | 5702.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6483.5     | 4802.600 | 4.803  | 5484.0     | 6535.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7386.6     | 5471.556 | 5.472  | 6395.0     | 7471.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6506.9     | 4819.948 | 4.820  | 5082.0     | 6766.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6309.4     | 4673.593 | 4.674  | 5559.0     | 6346.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5709.4     | 4229.185 | 4.229  | 4488.0     | 5646.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4464.5     | 3307.067 | 3.307  | 3864.0     | 4422.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5491.9     | 4068.104 | 4.068  | 4300.0     | 5367.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6228.5     | 4613.681 | 4.614  | 5018.0     | 6257.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7183.2     | 5320.926 | 5.321  | 5987.0     | 7150.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6240.0     | 4622.215 | 4.622  | 5072.0     | 6256.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6235.1     | 4618.593 | 4.619  | 4683.0     | 6023.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5804.4     | 4299.581 | 4.300  | 5709.4                | 5801.6                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4610.4     | 3415.094 | 3.415  | 4464.5                | 4614.1                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5568.0     | 4124.472 | 4.124  | 5491.9                | 5569.1                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6361.2     | 4711.974 | 4.712  | 6228.5                | 6357.8                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7292.9     | 5402.156 | 5.402  | 7183.2                | 7292.7                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6376.1     | 4723.072 | 4.723  | 6227.6                | 6373.5                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6278.9     | 4651.050 | 4.651  | 6232.5                | 6272.2                   |

Final op time: 5.402 us (7292.9 cycles), bucket=TRISC_2 compute

