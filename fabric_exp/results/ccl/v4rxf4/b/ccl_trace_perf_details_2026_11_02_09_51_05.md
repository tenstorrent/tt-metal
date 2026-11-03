# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_51_05
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

Report: generated/profiler/reports/2026_11_02_09_51_19/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5748.1     | 4257.874 | 4.258  | 3778.0     | 5678.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3816.7     | 2827.178 | 2.827  | 3000.0     | 3663.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5503.1     | 4076.363 | 4.076  | 3620.0     | 5447.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6314.6     | 4677.467 | 4.677  | 4269.0     | 6232.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7244.1     | 5365.970 | 5.366  | 5350.0     | 7135.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5394.6     | 3995.963 | 3.996  | 4877.0     | 5304.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5338.2     | 3954.222 | 3.954  | 4621.0     | 5145.0        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5757.5     | 4264.822 | 4.265  | 4644.0     | 5750.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5244.2     | 3884.570 | 3.885  | 4324.0     | 5284.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5499.1     | 4073.407 | 4.073  | 4362.0     | 5494.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6331.7     | 4690.119 | 4.690  | 5251.0     | 6335.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7250.3     | 5370.593 | 5.371  | 6140.0     | 7245.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6829.8     | 5059.141 | 5.059  | 4922.0     | 6985.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6622.1     | 4905.289 | 4.905  | 5109.0     | 6696.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5723.4     | 4239.570 | 4.240  | 4348.0     | 5755.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5247.6     | 3887.096 | 3.887  | 4136.0     | 5302.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5477.7     | 4057.556 | 4.058  | 4344.0     | 5496.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6296.7     | 4664.207 | 4.664  | 4993.0     | 6292.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7222.3     | 5349.837 | 5.350  | 6070.0     | 7219.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6817.0     | 5049.659 | 5.050  | 3536.0     | 6956.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6604.9     | 4892.489 | 4.892  | 5065.0     | 6654.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5732.4     | 4246.244 | 4.246  | 3670.0     | 5718.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3775.8     | 2796.919 | 2.797  | 3045.0     | 3612.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5481.6     | 4060.415 | 4.060  | 3509.0     | 5506.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6304.3     | 4669.822 | 4.670  | 3998.0     | 6266.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7225.4     | 5352.141 | 5.352  | 5078.0     | 7199.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5378.8     | 3984.259 | 3.984  | 4551.0     | 5251.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5320.0     | 3940.741 | 3.941  | 4268.0     | 5113.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5740.4     | 4252.128 | 4.252  | 5723.4                | 5740.3                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4521.1     | 3348.941 | 3.349  | 3775.8                | 4530.4                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5490.4     | 4066.935 | 4.067  | 5477.7                | 5490.3                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6311.8     | 4675.404 | 4.675  | 6296.7                | 6309.4                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7235.5     | 5359.635 | 5.360  | 7222.3                | 7234.7                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6105.0     | 4522.256 | 4.522  | 5378.8                | 6105.8                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 5971.3     | 4423.185 | 4.423  | 5320.0                | 5971.5                   |

Final op time: 5.360 us (7235.5 cycles), bucket=TRISC_2 compute

