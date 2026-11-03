# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_06_13
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

Report: generated/profiler/reports/2026_11_02_10_06_23/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 7074.2     | 5240.156 | 5.240  | 6609.0     | 7054.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5464.7     | 4047.941 | 4.048  | 5043.0     | 5453.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6879.2     | 5095.704 | 5.096  | 6366.0     | 6853.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 7467.9     | 5531.741 | 5.532  | 6933.0     | 7457.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8537.9     | 6324.363 | 6.324  | 8021.0     | 8522.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6728.2     | 4983.874 | 4.984  | 5020.0     | 6735.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6794.2     | 5032.711 | 5.033  | 6395.0     | 6786.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6987.9     | 5176.185 | 5.176  | 6494.0     | 6981.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5865.3     | 4344.637 | 4.345  | 5348.0     | 5846.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6792.7     | 5031.600 | 5.032  | 6253.0     | 6763.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 7365.2     | 5455.674 | 5.456  | 6847.0     | 7349.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8436.8     | 6249.481 | 6.249  | 7950.0     | 8433.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6956.5     | 5152.933 | 5.153  | 6543.0     | 6936.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6864.0     | 5084.422 | 5.084  | 6347.0     | 6842.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6953.9     | 5151.044 | 5.151  | 6460.0     | 6948.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5821.6     | 4312.326 | 4.312  | 5313.0     | 5804.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6746.3     | 4997.259 | 4.997  | 6213.0     | 6747.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 7351.4     | 5445.489 | 5.445  | 6822.0     | 7354.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8419.5     | 6236.674 | 6.237  | 7912.0     | 8419.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6946.6     | 5145.600 | 5.146  | 6584.0     | 6926.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6895.8     | 5107.993 | 5.108  | 6488.0     | 6901.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 7097.4     | 5257.363 | 5.257  | 6650.0     | 7080.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5531.1     | 4097.074 | 4.097  | 5150.0     | 5505.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6894.4     | 5106.926 | 5.107  | 6538.0     | 6894.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 7478.4     | 5539.526 | 5.540  | 6959.0     | 7455.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8543.6     | 6328.600 | 6.329  | 8063.0     | 8530.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6729.0     | 4984.467 | 4.984  | 5917.0     | 6724.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6733.0     | 4987.385 | 4.987  | 4355.0     | 6772.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 7028.4     | 5206.187 | 5.206  | 6953.9                | 7031.0                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 5670.7     | 4200.494 | 4.200  | 5464.7                | 5676.3                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 6828.1     | 5057.872 | 5.058  | 6746.3                | 6835.9                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 7415.7     | 5493.107 | 5.493  | 7351.4                | 7416.5                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 8484.5     | 6284.780 | 6.285  | 8419.5                | 8487.3                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6840.1     | 5066.719 | 5.067  | 6728.2                | 6837.8                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6821.7     | 5053.128 | 5.053  | 6733.0                | 6829.1                   |

Final op time: 6.285 us (8484.5 cycles), bucket=TRISC_2 compute

