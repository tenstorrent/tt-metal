# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_06_25
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

Report: generated/profiler/reports/2026_11_02_10_06_39/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5701.1     | 4223.022 | 4.223  | 4500.0     | 5729.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4512.2     | 3342.378 | 3.342  | 3995.0     | 4552.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5488.2     | 4065.370 | 4.065  | 4222.0     | 5540.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6187.7     | 4583.459 | 4.583  | 4933.0     | 6343.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7169.3     | 5310.578 | 5.311  | 5973.0     | 7246.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6262.1     | 4638.630 | 4.639  | 4470.0     | 6258.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6281.5     | 4652.956 | 4.653  | 5664.0     | 6464.0        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5907.4     | 4375.822 | 4.376  | 5253.0     | 5840.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4688.9     | 3473.281 | 3.473  | 4193.0     | 4659.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5628.2     | 4169.044 | 4.169  | 4962.0     | 5577.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6505.4     | 4818.852 | 4.819  | 5822.0     | 6444.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7410.7     | 5489.415 | 5.489  | 6736.0     | 7347.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6464.6     | 4788.578 | 4.789  | 5530.0     | 6648.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6289.1     | 4658.563 | 4.659  | 5838.0     | 6270.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5870.5     | 4348.496 | 4.348  | 5258.0     | 5806.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4683.7     | 3469.415 | 3.469  | 4189.0     | 4681.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5600.0     | 4148.178 | 4.148  | 4966.0     | 5692.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6485.1     | 4803.763 | 4.804  | 5891.0     | 6407.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7383.2     | 5469.067 | 5.469  | 6768.0     | 7309.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6446.7     | 4775.356 | 4.775  | 5540.0     | 6511.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6279.3     | 4651.341 | 4.651  | 5865.0     | 6260.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5695.8     | 4219.074 | 4.219  | 4514.0     | 5791.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4494.6     | 3329.326 | 3.329  | 4002.0     | 4562.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5482.9     | 4061.378 | 4.061  | 4263.0     | 5679.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6201.0     | 4593.348 | 4.593  | 4837.0     | 6266.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7168.9     | 5310.304 | 5.310  | 5916.0     | 7237.0        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6294.5     | 4662.615 | 4.663  | 5533.0     | 6288.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6259.3     | 4636.511 | 4.637  | 3898.0     | 6378.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5793.7     | 4291.604 | 4.292  | 5695.8                | 5785.8                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4594.9     | 3403.600 | 3.404  | 4494.6                | 4598.0                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5549.8     | 4110.993 | 4.111  | 5482.9                | 5544.1                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6344.8     | 4699.856 | 4.700  | 6187.7                | 6343.1                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7283.0     | 5394.841 | 5.395  | 7168.9                | 7276.3                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6367.0     | 4716.294 | 4.716  | 6262.1                | 6370.6                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6277.3     | 4649.843 | 4.650  | 6259.3                | 6280.4                   |

Final op time: 5.395 us (7283.0 cycles), bucket=TRISC_2 compute

