# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_03_37
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

Report: generated/profiler/reports/2026_11_02_10_03_47/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5957.4     | 4412.889 | 4.413  | 3875.0     | 5912.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4442.6     | 3290.785 | 3.291  | 3461.0     | 4451.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5747.6     | 4257.467 | 4.257  | 3760.0     | 5662.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6481.3     | 4800.970 | 4.801  | 4363.0     | 6377.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7441.8     | 5512.437 | 5.512  | 5372.0     | 7368.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6074.1     | 4499.333 | 4.499  | 5152.0     | 6088.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6121.4     | 4534.363 | 4.534  | 4899.0     | 6016.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6223.7     | 4610.141 | 4.610  | 5469.0     | 6178.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5051.3     | 3741.726 | 3.742  | 4332.0     | 5053.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5945.9     | 4404.341 | 4.404  | 5360.0     | 5893.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6816.6     | 5049.363 | 5.049  | 6053.0     | 6778.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7716.1     | 5715.600 | 5.716  | 6966.0     | 7681.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6383.9     | 4728.844 | 4.729  | 4883.0     | 6665.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6123.5     | 4535.933 | 4.536  | 4994.0     | 6291.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6202.9     | 4594.756 | 4.595  | 5516.0     | 6157.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5045.7     | 3737.541 | 3.738  | 4341.0     | 5023.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5933.4     | 4395.096 | 4.395  | 5249.0     | 5895.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6804.8     | 5040.556 | 5.041  | 6060.0     | 6788.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7708.1     | 5709.726 | 5.710  | 7022.0     | 7683.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6372.4     | 4720.304 | 4.720  | 4866.0     | 6704.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6090.9     | 4511.807 | 4.512  | 5023.0     | 6147.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5963.4     | 4417.356 | 4.417  | 3986.0     | 5944.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4475.6     | 3315.296 | 3.315  | 3478.0     | 4325.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5748.1     | 4257.874 | 4.258  | 3753.0     | 5665.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6492.3     | 4809.096 | 4.809  | 4382.0     | 6477.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7441.0     | 5511.867 | 5.512  | 5466.0     | 7402.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6112.1     | 4527.452 | 4.527  | 4639.0     | 6102.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6125.3     | 4537.237 | 4.537  | 4074.0     | 5960.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 6086.9     | 4508.785 | 4.509  | 5957.4                | 6083.2                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4753.8     | 3521.337 | 3.521  | 4442.6                | 4760.7                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5843.7     | 4328.694 | 4.329  | 5747.6                | 5840.8                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6648.7     | 4924.996 | 4.925  | 6481.3                | 6648.5                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7576.8     | 5612.407 | 5.612  | 7441.0                | 7575.0                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6235.6     | 4618.983 | 4.619  | 6074.1                | 6242.2                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6115.3     | 4529.835 | 4.530  | 6090.9                | 6122.5                   |

Final op time: 5.612 us (7576.8 cycles), bucket=TRISC_2 compute

