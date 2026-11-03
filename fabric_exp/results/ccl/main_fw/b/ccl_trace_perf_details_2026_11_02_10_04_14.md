# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_04_14
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

Report: generated/profiler/reports/2026_11_02_10_04_24/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 7512.8     | 5565.044 | 5.565  | 6917.0     | 7474.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5309.2     | 3932.748 | 3.933  | 4369.0     | 5302.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 7418.7     | 5495.333 | 5.495  | 6787.0     | 7344.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 7879.4     | 5836.556 | 5.837  | 7270.0     | 7853.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8958.3     | 6635.785 | 6.636  | 8350.0     | 8942.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6892.4     | 5105.467 | 5.105  | 4895.0     | 6887.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6900.3     | 5111.304 | 5.111  | 6300.0     | 6851.0        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6655.9     | 4930.304 | 4.930  | 6037.0     | 6607.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5374.2     | 3980.881 | 3.981  | 4322.0     | 5377.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6404.5     | 4744.074 | 4.744  | 5756.0     | 6374.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 7251.0     | 5371.126 | 5.371  | 6624.0     | 7208.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8157.6     | 6042.630 | 6.043  | 7536.0     | 8115.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6923.8     | 5128.770 | 5.129  | 6261.0     | 6957.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6858.5     | 5080.341 | 5.080  | 6201.0     | 6884.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6602.6     | 4890.844 | 4.891  | 6118.0     | 6580.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5337.1     | 3953.407 | 3.953  | 4284.0     | 5281.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6376.3     | 4723.215 | 4.723  | 5868.0     | 6383.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 7199.6     | 5333.044 | 5.333  | 6721.0     | 7187.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8110.3     | 6007.659 | 6.008  | 7630.0     | 8092.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6916.4     | 5123.244 | 5.123  | 6166.0     | 6935.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6857.1     | 5079.363 | 5.079  | 6198.0     | 6883.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 7519.0     | 5569.607 | 5.570  | 6835.0     | 7485.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5324.0     | 3943.696 | 3.944  | 4355.0     | 5276.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 7417.1     | 5494.178 | 5.494  | 6615.0     | 7403.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 7872.4     | 5831.430 | 5.831  | 7177.0     | 7835.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 8952.6     | 6631.593 | 6.632  | 8266.0     | 8923.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6915.8     | 5122.807 | 5.123  | 6324.0     | 6903.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6881.3     | 5097.252 | 5.097  | 4900.0     | 6885.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 7072.6     | 5238.950 | 5.239  | 6602.6                | 7084.4                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 5336.1     | 3952.683 | 3.953  | 5309.2                | 5330.5                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 6904.2     | 5114.200 | 5.114  | 6376.3                | 6910.8                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 7550.6     | 5593.039 | 5.593  | 7199.6                | 7561.7                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 8544.7     | 6329.417 | 6.329  | 8110.3                | 8555.1                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6912.1     | 5120.072 | 5.120  | 6892.4                | 6916.1                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6874.3     | 5092.065 | 5.092  | 6857.1                | 6869.9                   |

Final op time: 6.329 us (8544.7 cycles), bucket=TRISC_2 compute

