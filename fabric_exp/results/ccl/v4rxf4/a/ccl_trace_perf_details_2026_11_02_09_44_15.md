# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_44_15
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

Report: generated/profiler/reports/2026_11_02_09_44_30/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5971.0     | 4422.956 | 4.423  | 4277.0     | 5999.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4040.3     | 2992.837 | 2.993  | 3060.0     | 4006.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5708.8     | 4228.741 | 4.229  | 3980.0     | 5716.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6550.6     | 4852.289 | 4.852  | 4848.0     | 6595.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7467.7     | 5531.637 | 5.532  | 5772.0     | 7507.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5551.4     | 4112.126 | 4.112  | 3883.0     | 5371.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5481.9     | 4060.689 | 4.061  | 4796.0     | 5354.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5837.2     | 4323.844 | 4.324  | 4818.0     | 5858.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5301.8     | 3927.259 | 3.927  | 4575.0     | 5305.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5559.3     | 4117.978 | 4.118  | 4591.0     | 5576.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6421.2     | 4756.474 | 4.756  | 5570.0     | 6470.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7334.4     | 5432.859 | 5.433  | 6459.0     | 7358.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6607.9     | 4894.711 | 4.895  | 5530.0     | 6677.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6618.6     | 4902.681 | 4.903  | 4790.0     | 6891.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5785.0     | 4285.178 | 4.285  | 4371.0     | 5837.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5313.4     | 3935.881 | 3.936  | 4615.0     | 5313.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5541.1     | 4104.496 | 4.104  | 4577.0     | 5562.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6375.0     | 4722.237 | 4.722  | 5007.0     | 6428.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7289.0     | 5399.230 | 5.399  | 6085.0     | 7336.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6591.4     | 4882.541 | 4.883  | 4434.0     | 6723.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6589.8     | 4881.363 | 4.881  | 4836.0     | 6812.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5957.2     | 4412.726 | 4.413  | 4217.0     | 6008.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4007.6     | 2968.607 | 2.969  | 3137.0     | 3960.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5704.4     | 4225.444 | 4.225  | 3913.0     | 5730.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6536.5     | 4841.844 | 4.842  | 4640.0     | 6622.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7450.4     | 5518.785 | 5.519  | 5720.0     | 7517.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5537.7     | 4102.022 | 4.102  | 4792.0     | 5363.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5486.5     | 4064.081 | 4.064  | 3968.0     | 5376.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5887.6     | 4361.176 | 4.361  | 5785.0                | 5897.2                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4665.8     | 3456.146 | 3.456  | 4007.6                | 4671.1                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5628.4     | 4169.165 | 4.169  | 5541.1                | 5631.8                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6470.8     | 4793.211 | 4.793  | 6375.0                | 6478.9                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7385.3     | 5470.628 | 5.471  | 7289.0                | 7392.4                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6072.1     | 4497.850 | 4.498  | 5537.7                | 6071.4                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6044.2     | 4477.204 | 4.477  | 5481.9                | 6038.2                   |

Final op time: 5.471 us (7385.3 cycles), bucket=TRISC_2 compute

