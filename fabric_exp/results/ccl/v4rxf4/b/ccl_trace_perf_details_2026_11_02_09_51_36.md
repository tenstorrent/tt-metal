# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_51_36
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

### num_l_chunks=2, tiles_per_l_chunk=8, block_size=8, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_09_51_50/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5803.1     | 4298.570 | 4.299  | 3902.0     | 5717.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3895.1     | 2885.274 | 2.885  | 3149.0     | 3786.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5548.6     | 4110.044 | 4.110  | 3629.0     | 5426.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6397.0     | 4738.533 | 4.739  | 4562.0     | 6299.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7304.7     | 5410.881 | 5.411  | 5476.0     | 7210.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5439.3     | 4029.133 | 4.029  | 3959.0     | 5344.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5338.5     | 3954.430 | 3.954  | 4559.0     | 5308.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5849.2     | 4332.756 | 4.333  | 4871.0     | 5875.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5225.9     | 3871.037 | 3.871  | 4198.0     | 5262.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5576.5     | 4130.733 | 4.131  | 4588.0     | 5598.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6427.5     | 4761.081 | 4.761  | 5491.0     | 6452.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7344.8     | 5440.585 | 5.441  | 6384.0     | 7377.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6752.6     | 5001.941 | 5.002  | 5240.0     | 6745.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6766.1     | 5011.904 | 5.012  | 5505.0     | 6704.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5791.4     | 4289.889 | 4.290  | 4563.0     | 5814.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5225.4     | 3870.689 | 3.871  | 4211.0     | 5265.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5567.1     | 4123.815 | 4.124  | 4461.0     | 5588.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6377.6     | 4724.185 | 4.724  | 4912.0     | 6393.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7294.5     | 5403.356 | 5.403  | 5995.0     | 7328.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6733.3     | 4987.622 | 4.988  | 3890.0     | 6778.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6777.6     | 5020.407 | 5.020  | 5556.0     | 6692.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5765.2     | 4270.526 | 4.271  | 4095.0     | 5702.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3858.0     | 2857.785 | 2.858  | 3039.0     | 3711.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5503.9     | 4076.978 | 4.077  | 3997.0     | 5427.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6354.6     | 4707.104 | 4.707  | 4418.0     | 6317.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7266.2     | 5382.348 | 5.382  | 5522.0     | 7221.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5428.4     | 4021.015 | 4.021  | 4832.0     | 5294.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5324.7     | 3944.193 | 3.944  | 3609.0     | 5304.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5802.2     | 4297.935 | 4.298  | 5765.2                | 5797.2                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4551.1     | 3371.196 | 3.371  | 3858.0                | 4560.3                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5549.0     | 4110.393 | 4.110  | 5503.9                | 5557.9                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6389.2     | 4732.726 | 4.733  | 6354.6                | 6387.3                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7302.5     | 5409.293 | 5.409  | 7266.2                | 7299.6                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6088.4     | 4509.928 | 4.510  | 5428.4                | 6086.3                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6051.7     | 4482.733 | 4.483  | 5324.7                | 6052.3                   |

Final op time: 5.409 us (7302.5 cycles), bucket=TRISC_2 compute

