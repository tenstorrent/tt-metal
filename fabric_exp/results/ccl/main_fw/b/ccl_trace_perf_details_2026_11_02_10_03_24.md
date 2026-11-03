# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_03_24
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

Report: generated/profiler/reports/2026_11_02_10_03_35/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6186.1     | 4582.259 | 4.582  | 5743.0     | 6168.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4692.5     | 3475.926 | 3.476  | 4286.0     | 4676.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5965.9     | 4419.215 | 4.419  | 5637.0     | 5933.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6779.7     | 5021.985 | 5.022  | 6266.0     | 6770.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7688.0     | 5694.822 | 5.695  | 7184.0     | 7680.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6798.4     | 5035.881 | 5.036  | 6193.0     | 6809.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6806.2     | 5041.600 | 5.042  | 6424.0     | 6785.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6187.1     | 4583.000 | 4.583  | 5663.0     | 6157.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4778.2     | 3539.385 | 3.539  | 4453.0     | 4742.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5954.2     | 4410.548 | 4.411  | 5562.0     | 5917.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6775.6     | 5018.993 | 5.019  | 6237.0     | 6748.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7681.2     | 5689.785 | 5.690  | 7112.0     | 7657.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6933.4     | 5135.815 | 5.136  | 6361.0     | 6908.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6859.1     | 5080.852 | 5.081  | 6504.0     | 6830.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6179.8     | 4577.637 | 4.578  | 5784.0     | 6134.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4789.9     | 3548.059 | 3.548  | 4430.0     | 4771.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5957.0     | 4412.607 | 4.413  | 5653.0     | 5902.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6784.0     | 5025.170 | 5.025  | 6397.0     | 6738.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7687.0     | 5694.089 | 5.694  | 7305.0     | 7644.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6931.3     | 5134.289 | 5.134  | 5668.0     | 6917.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6851.8     | 5075.437 | 5.075  | 6462.0     | 6824.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6191.4     | 4586.244 | 4.586  | 5643.0     | 6148.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4680.6     | 3467.126 | 3.467  | 4364.0     | 4660.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5956.6     | 4412.311 | 4.412  | 5575.0     | 5913.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6792.9     | 5031.815 | 5.032  | 6227.0     | 6751.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7697.4     | 5701.770 | 5.702  | 7136.0     | 7657.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6779.8     | 5022.059 | 5.022  | 6215.0     | 6771.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6756.0     | 5004.430 | 5.004  | 5018.0     | 6750.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 6186.1     | 4582.285 | 4.582  | 6179.8                | 6186.6                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4735.3     | 3507.624 | 3.508  | 4680.6                | 4735.3                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5958.5     | 4413.670 | 4.414  | 5954.2                | 5956.8                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6783.1     | 5024.491 | 5.024  | 6775.6                | 6781.8                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7688.4     | 5695.117 | 5.695  | 7681.2                | 7687.5                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6860.7     | 5082.011 | 5.082  | 6779.8                | 6864.9                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6818.3     | 5050.580 | 5.051  | 6756.0                | 6829.0                   |

Final op time: 5.695 us (7688.4 cycles), bucket=TRISC_2 compute

