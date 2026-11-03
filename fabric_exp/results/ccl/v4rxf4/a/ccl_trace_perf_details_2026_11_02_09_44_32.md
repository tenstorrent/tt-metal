# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_44_32
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

Report: generated/profiler/reports/2026_11_02_09_44_46/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5925.2     | 4389.037 | 4.389  | 4034.0     | 5823.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4033.2     | 2987.585 | 2.988  | 3126.0     | 3972.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5692.6     | 4216.770 | 4.217  | 3871.0     | 5700.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6495.7     | 4811.622 | 4.812  | 4721.0     | 6429.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7420.8     | 5496.889 | 5.497  | 5634.0     | 7326.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5512.4     | 4083.274 | 4.083  | 4576.0     | 5423.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5445.4     | 4033.630 | 4.034  | 4610.0     | 5283.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5934.1     | 4395.644 | 4.396  | 4419.0     | 5960.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5191.9     | 3845.874 | 3.846  | 4154.0     | 5314.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5666.1     | 4197.119 | 4.197  | 4147.0     | 5681.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6495.8     | 4811.726 | 4.812  | 5018.0     | 6507.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7423.5     | 5498.896 | 5.499  | 5932.0     | 7455.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6687.3     | 4953.585 | 4.954  | 4962.0     | 6764.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6605.9     | 4893.289 | 4.893  | 4623.0     | 6564.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5859.4     | 4340.259 | 4.340  | 4132.0     | 5888.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5185.5     | 3841.119 | 3.841  | 4102.0     | 5303.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5624.1     | 4165.993 | 4.166  | 4269.0     | 5647.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6443.3     | 4772.785 | 4.773  | 4737.0     | 6443.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7361.9     | 5453.222 | 5.453  | 5824.0     | 7393.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6645.2     | 4922.348 | 4.922  | 3297.0     | 6771.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6595.0     | 4885.215 | 4.885  | 4696.0     | 6585.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5888.2     | 4361.630 | 4.362  | 3850.0     | 5826.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3994.4     | 2958.844 | 2.959  | 3096.0     | 3944.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5640.4     | 4178.074 | 4.178  | 3849.0     | 5587.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6451.8     | 4779.119 | 4.779  | 4333.0     | 6312.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7381.6     | 5467.822 | 5.468  | 5412.0     | 7271.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5506.3     | 4078.741 | 4.079  | 4832.0     | 5409.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5422.8     | 4016.889 | 4.017  | 3727.0     | 5282.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5901.7     | 4371.643 | 4.372  | 5859.4                | 5906.7                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4601.3     | 3408.356 | 3.408  | 3994.4                | 4609.4                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5655.8     | 4189.489 | 4.189  | 5624.1                | 5653.3                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6471.6     | 4793.813 | 4.794  | 6443.3                | 6473.8                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7396.9     | 5479.207 | 5.479  | 7361.9                | 7401.2                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6087.8     | 4509.487 | 4.509  | 5506.3                | 6078.8                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6017.3     | 4457.256 | 4.457  | 5422.8                | 6020.2                   |

Final op time: 5.479 us (7396.9 cycles), bucket=TRISC_2 compute

