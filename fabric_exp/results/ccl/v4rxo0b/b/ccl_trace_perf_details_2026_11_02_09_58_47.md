# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_58_47
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

Report: generated/profiler/reports/2026_11_02_09_59_01/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5871.5     | 4349.230 | 4.349  | 5439.0     | 5838.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4462.3     | 3305.378 | 3.305  | 4111.0     | 4411.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5659.1     | 4191.919 | 4.192  | 5266.0     | 5622.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6447.1     | 4775.622 | 4.776  | 6071.0     | 6429.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7365.6     | 5455.978 | 5.456  | 6959.0     | 7330.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6531.7     | 4838.281 | 4.838  | 5286.0     | 6549.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6529.1     | 4836.333 | 4.836  | 5915.0     | 6530.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5913.2     | 4380.148 | 4.380  | 5607.0     | 5839.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4503.8     | 3336.148 | 3.336  | 4157.0     | 4446.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5678.1     | 4205.985 | 4.206  | 5350.0     | 5614.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6491.0     | 4808.163 | 4.808  | 6113.0     | 6433.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7407.5     | 5487.044 | 5.487  | 7030.0     | 7342.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6591.1     | 4882.304 | 4.882  | 6043.0     | 6587.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6511.2     | 4823.081 | 4.823  | 5864.0     | 6537.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5878.4     | 4354.385 | 4.354  | 5354.0     | 5848.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4512.9     | 3342.867 | 3.343  | 4170.0     | 4467.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5675.6     | 4204.185 | 4.204  | 5208.0     | 5640.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6466.1     | 4789.681 | 4.790  | 5971.0     | 6442.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7381.7     | 5467.926 | 5.468  | 6887.0     | 7348.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6570.6     | 4867.104 | 4.867  | 4946.0     | 6587.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6523.9     | 4832.548 | 4.833  | 5707.0     | 6536.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5868.9     | 4347.370 | 4.347  | 5389.0     | 5836.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4454.2     | 3299.400 | 3.299  | 4083.0     | 4430.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5649.4     | 4184.763 | 4.185  | 5249.0     | 5619.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6443.9     | 4773.274 | 4.773  | 5975.0     | 6435.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7361.3     | 5452.793 | 5.453  | 6887.0     | 7335.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6502.3     | 4816.526 | 4.817  | 5410.0     | 6519.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6462.3     | 4786.911 | 4.787  | 4896.0     | 6488.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5883.0     | 4357.783 | 4.358  | 5868.9                | 5874.9                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4483.3     | 3320.948 | 3.321  | 4454.2                | 4483.0                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5665.6     | 4196.713 | 4.197  | 5649.4                | 5667.4                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6462.0     | 4786.685 | 4.787  | 6443.9                | 6456.6                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7379.0     | 5465.935 | 5.466  | 7361.3                | 7373.6                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6548.9     | 4851.054 | 4.851  | 6502.3                | 6551.1                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6506.6     | 4819.719 | 4.820  | 6462.3                | 6517.5                   |

Final op time: 5.466 us (7379.0 cycles), bucket=TRISC_2 compute

