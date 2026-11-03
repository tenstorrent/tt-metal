# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_06_57
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

Report: generated/profiler/reports/2026_11_02_10_07_11/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5580.6     | 4133.770 | 4.134  | 4949.0     | 5542.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3882.0     | 2875.563 | 2.876  | 3569.0     | 3839.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5332.6     | 3950.037 | 3.950  | 4694.0     | 5310.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6179.7     | 4577.585 | 4.578  | 5548.0     | 6137.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7081.2     | 5245.319 | 5.245  | 6464.0     | 7049.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5692.9     | 4216.933 | 4.217  | 4251.0     | 5704.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5543.2     | 4106.067 | 4.106  | 5195.0     | 5509.0        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5663.0     | 4194.815 | 4.195  | 5233.0     | 5608.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5245.1     | 3885.281 | 3.885  | 4766.0     | 5200.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5427.6     | 4020.459 | 4.020  | 4969.0     | 5360.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6247.6     | 4627.837 | 4.628  | 5839.0     | 6183.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7156.8     | 5301.348 | 5.301  | 6753.0     | 7094.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 7038.2     | 5213.452 | 5.213  | 6658.0     | 7029.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6996.3     | 5182.437 | 5.182  | 6658.0     | 7000.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5641.7     | 4179.030 | 4.179  | 5208.0     | 5580.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5248.9     | 3888.089 | 3.888  | 4844.0     | 5238.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5411.9     | 4008.800 | 4.009  | 5130.0     | 5358.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6234.7     | 4618.296 | 4.618  | 5562.0     | 6177.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7145.2     | 5292.756 | 5.293  | 6651.0     | 7086.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 7027.4     | 5205.481 | 5.205  | 5291.0     | 7029.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6968.4     | 5161.807 | 5.162  | 5677.0     | 6967.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5574.4     | 4129.222 | 4.129  | 4782.0     | 5545.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3757.8     | 2783.585 | 2.784  | 3461.0     | 3705.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5322.9     | 3942.852 | 3.943  | 4681.0     | 5300.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6179.1     | 4577.089 | 4.577  | 5405.0     | 6150.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7081.8     | 5245.807 | 5.246  | 6298.0     | 7051.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5642.8     | 4179.874 | 4.180  | 5421.0     | 5631.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5490.5     | 4067.059 | 4.067  | 3607.0     | 5488.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5614.9     | 4159.209 | 4.159  | 5574.4                | 5611.1                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4533.5     | 3358.130 | 3.358  | 3757.8                | 4563.6                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5373.7     | 3980.537 | 3.981  | 5322.9                | 5372.2                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6210.3     | 4600.202 | 4.600  | 6179.1                | 6207.2                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7116.3     | 5271.307 | 5.271  | 7081.2                | 7113.5                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6350.3     | 4703.935 | 4.704  | 5642.8                | 6360.1                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6249.6     | 4629.343 | 4.629  | 5490.5                | 6255.8                   |

Final op time: 5.271 us (7116.3 cycles), bucket=TRISC_2 compute

