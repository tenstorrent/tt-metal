# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_58_00
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

Report: generated/profiler/reports/2026_11_02_09_58_14/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5958.4     | 4413.607 | 4.414  | 3918.0     | 6033.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3809.8     | 2822.104 | 2.822  | 3039.0     | 3674.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5721.2     | 4237.956 | 4.238  | 3652.0     | 5764.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6522.5     | 4831.496 | 4.831  | 4258.0     | 6570.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7449.2     | 5517.896 | 5.518  | 5344.0     | 7486.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5343.2     | 3957.948 | 3.958  | 4153.0     | 5197.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5342.6     | 3957.481 | 3.957  | 4791.0     | 5187.0        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5785.7     | 4285.733 | 4.286  | 4112.0     | 5746.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5365.0     | 3974.081 | 3.974  | 4049.0     | 5456.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5539.3     | 4103.163 | 4.103  | 3825.0     | 5519.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6351.4     | 4704.704 | 4.705  | 4703.0     | 6324.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7276.3     | 5389.881 | 5.390  | 5622.0     | 7249.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6955.7     | 5152.400 | 5.152  | 5211.0     | 7124.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6848.4     | 5072.874 | 5.073  | 4760.0     | 6977.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5735.4     | 4248.437 | 4.248  | 3872.0     | 5717.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5362.0     | 3971.881 | 3.972  | 4072.0     | 5450.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5537.1     | 4101.556 | 4.102  | 3959.0     | 5557.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6296.1     | 4663.815 | 4.664  | 4418.0     | 6294.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7231.0     | 5356.311 | 5.356  | 5503.0     | 7219.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6932.2     | 5134.993 | 5.135  | 4117.0     | 7124.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6842.2     | 5068.274 | 5.068  | 4763.0     | 7037.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5937.4     | 4398.081 | 4.398  | 3837.0     | 6013.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3790.2     | 2807.533 | 2.808  | 3043.0     | 3654.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5698.6     | 4221.207 | 4.221  | 3700.0     | 5767.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6503.0     | 4817.022 | 4.817  | 4296.0     | 6551.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7428.8     | 5502.800 | 5.503  | 5258.0     | 7504.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5327.1     | 3946.007 | 3.946  | 4792.0     | 5205.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5304.6     | 3929.311 | 3.929  | 3434.0     | 5144.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5854.2     | 4336.465 | 4.336  | 5735.4                | 5861.6                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4581.8     | 3393.900 | 3.394  | 3790.2                | 4585.9                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5624.1     | 4165.970 | 4.166  | 5537.1                | 5619.0                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6418.2     | 4754.259 | 4.754  | 6296.1                | 6427.2                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7346.3     | 5441.722 | 5.442  | 7231.0                | 7352.6                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6139.6     | 4547.837 | 4.548  | 5327.1                | 6137.7                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6084.4     | 4506.985 | 4.507  | 5304.6                | 6092.4                   |

Final op time: 5.442 us (7346.3 cycles), bucket=TRISC_2 compute

