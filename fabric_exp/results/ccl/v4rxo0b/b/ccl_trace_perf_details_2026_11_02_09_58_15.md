# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_58_15
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

Report: generated/profiler/reports/2026_11_02_09_58_30/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5796.9     | 4294.007 | 4.294  | 3612.0     | 6103.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4298.7     | 3184.244 | 3.184  | 3395.0     | 4254.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5555.2     | 4114.933 | 4.115  | 3329.0     | 5812.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6356.8     | 4708.748 | 4.709  | 4198.0     | 6635.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7295.2     | 5403.852 | 5.404  | 5098.0     | 7610.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5754.4     | 4262.526 | 4.263  | 4869.0     | 5716.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5815.1     | 4307.452 | 4.307  | 4924.0     | 5751.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6043.8     | 4476.852 | 4.477  | 5301.0     | 6104.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5221.8     | 3867.963 | 3.868  | 4537.0     | 5277.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5771.2     | 4274.941 | 4.275  | 5036.0     | 5822.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6628.2     | 4909.756 | 4.910  | 5898.0     | 6686.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7533.3     | 5580.244 | 5.580  | 6798.0     | 7577.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6467.9     | 4791.037 | 4.791  | 4813.0     | 6754.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5918.8     | 4384.274 | 4.384  | 4641.0     | 6111.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5990.4     | 4437.304 | 4.437  | 5072.0     | 6052.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5240.7     | 3882.030 | 3.882  | 4619.0     | 5275.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5731.9     | 4245.837 | 4.246  | 4858.0     | 5802.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6587.5     | 4879.637 | 4.880  | 5501.0     | 6637.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7495.4     | 5552.119 | 5.552  | 6585.0     | 7539.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6438.1     | 4768.978 | 4.769  | 4260.0     | 6671.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5932.9     | 4394.770 | 4.395  | 4588.0     | 6130.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5764.6     | 4270.104 | 4.270  | 3478.0     | 5967.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4240.3     | 3140.978 | 3.141  | 3283.0     | 4180.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5531.2     | 4097.178 | 4.097  | 3316.0     | 5697.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6328.1     | 4687.467 | 4.687  | 3896.0     | 6584.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7261.9     | 5379.200 | 5.379  | 4991.0     | 7483.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5700.3     | 4222.437 | 4.222  | 4076.0     | 5622.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5750.3     | 4259.511 | 4.260  | 3783.0     | 5666.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5898.9     | 4369.567 | 4.370  | 5764.6                | 5893.6                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4750.4     | 3518.804 | 3.519  | 4240.3                | 4760.2                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5647.3     | 4183.222 | 4.183  | 5531.2                | 5643.5                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6475.1     | 4796.402 | 4.796  | 6328.1                | 6472.2                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7396.5     | 5478.854 | 5.479  | 7261.9                | 7395.3                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6090.2     | 4511.244 | 4.511  | 5700.3                | 6096.3                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 5854.3     | 4336.502 | 4.337  | 5750.3                | 5866.9                   |

Final op time: 5.479 us (7396.5 cycles), bucket=TRISC_2 compute

