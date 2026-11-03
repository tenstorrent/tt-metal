# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_43_10
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

Report: generated/profiler/reports/2026_11_02_09_43_24/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5766.1     | 4271.200 | 4.271  | 3184.0     | 5696.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4132.1     | 3060.830 | 3.061  | 3181.0     | 3971.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5511.8     | 4082.837 | 4.083  | 2897.0     | 5390.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6334.5     | 4692.230 | 4.692  | 3784.0     | 6292.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7261.1     | 5378.630 | 5.379  | 4675.0     | 7189.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5852.0     | 4334.837 | 4.335  | 5040.0     | 5752.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5838.4     | 4324.719 | 4.325  | 4919.0     | 5660.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5862.2     | 4342.341 | 4.342  | 5222.0     | 5793.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4896.8     | 3627.259 | 3.627  | 3710.0     | 4942.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5586.0     | 4137.793 | 4.138  | 4929.0     | 5516.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6441.5     | 4771.452 | 4.771  | 5805.0     | 6402.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7352.4     | 5446.215 | 5.446  | 6719.0     | 7292.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6417.1     | 4753.370 | 4.753  | 4689.0     | 6620.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6115.5     | 4530.015 | 4.530  | 4555.0     | 6265.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5824.4     | 4314.393 | 4.314  | 5236.0     | 5814.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4893.2     | 3624.593 | 3.625  | 3688.0     | 5000.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5565.4     | 4122.504 | 4.123  | 4958.0     | 5539.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6419.9     | 4755.481 | 4.755  | 5836.0     | 6416.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7331.0     | 5430.400 | 5.430  | 6751.0     | 7318.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6399.4     | 4740.274 | 4.740  | 4204.0     | 6679.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6118.9     | 4532.533 | 4.533  | 4742.0     | 6363.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5738.9     | 4251.030 | 4.251  | 3270.0     | 5634.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4081.1     | 3023.022 | 3.023  | 3136.0     | 3895.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5482.4     | 4061.000 | 4.061  | 3143.0     | 5408.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6307.8     | 4672.415 | 4.672  | 3633.0     | 6235.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7233.1     | 5357.874 | 5.358  | 4725.0     | 7132.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5831.8     | 4319.815 | 4.320  | 3867.0     | 5740.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5825.3     | 4315.015 | 4.315  | 3975.0     | 5673.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5797.9     | 4294.741 | 4.295  | 5738.9                | 5795.3                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4500.8     | 3333.926 | 3.334  | 4081.1                | 4512.7                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5536.4     | 4101.033 | 4.101  | 5482.4                | 5538.6                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6375.9     | 4722.894 | 4.723  | 6307.8                | 6377.2                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7294.4     | 5403.280 | 5.403  | 7233.1                | 7296.1                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6125.0     | 4537.074 | 4.537  | 5831.8                | 6125.7                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 5974.5     | 4425.570 | 4.426  | 5825.3                | 5976.9                   |

Final op time: 5.403 us (7294.4 cycles), bucket=TRISC_2 compute

