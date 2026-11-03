# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_05_25
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

Report: generated/profiler/reports/2026_11_02_10_05_39/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5687.7     | 4213.096 | 4.213  | 4471.0     | 5636.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4493.2     | 3328.296 | 3.328  | 3867.0     | 4469.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5480.2     | 4059.415 | 4.059  | 4221.0     | 5493.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6190.4     | 4585.481 | 4.585  | 4904.0     | 6160.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7159.4     | 5303.274 | 5.303  | 5951.0     | 7097.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6283.1     | 4654.119 | 4.654  | 4185.0     | 6287.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6283.4     | 4654.341 | 4.654  | 5565.0     | 6370.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5880.1     | 4355.607 | 4.356  | 5276.0     | 5752.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4716.1     | 3493.400 | 3.493  | 4196.0     | 4664.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5602.9     | 4150.319 | 4.150  | 4979.0     | 5468.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6478.5     | 4798.881 | 4.799  | 5899.0     | 6348.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7379.3     | 5466.133 | 5.466  | 6792.0     | 7251.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6437.8     | 4768.704 | 4.769  | 5575.0     | 6351.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6308.3     | 4672.815 | 4.673  | 5694.0     | 6273.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5851.4     | 4334.400 | 4.334  | 5294.0     | 5753.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4693.3     | 3476.489 | 3.476  | 4173.0     | 4724.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5581.6     | 4134.519 | 4.135  | 5020.0     | 5484.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6458.0     | 4783.726 | 4.784  | 5906.0     | 6341.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7362.5     | 5453.733 | 5.454  | 6784.0     | 7257.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6424.2     | 4758.696 | 4.759  | 5361.0     | 6298.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6258.6     | 4635.985 | 4.636  | 5697.0     | 6206.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5676.2     | 4204.563 | 4.205  | 4473.0     | 5648.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4483.9     | 3321.430 | 3.321  | 3889.0     | 4541.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5466.5     | 4049.289 | 4.049  | 4293.0     | 5524.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6181.0     | 4578.548 | 4.579  | 4907.0     | 6157.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7145.8     | 5293.170 | 5.293  | 5940.0     | 7112.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6325.7     | 4685.696 | 4.686  | 5607.0     | 6303.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6287.0     | 4657.022 | 4.657  | 4363.0     | 6381.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5773.8     | 4276.917 | 4.277  | 5676.2                | 5769.6                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4596.6     | 3404.904 | 3.405  | 4483.9                | 4593.2                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5532.8     | 4098.385 | 4.098  | 5466.5                | 5530.9                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6327.0     | 4686.659 | 4.687  | 6181.0                | 6324.2                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7261.8     | 5379.078 | 5.379  | 7145.8                | 7261.0                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6367.7     | 4716.804 | 4.717  | 6283.1                | 6375.0                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6284.3     | 4655.041 | 4.655  | 6258.6                | 6285.2                   |

Final op time: 5.379 us (7261.8 cycles), bucket=TRISC_2 compute

