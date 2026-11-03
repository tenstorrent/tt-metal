# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_06_41
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

Report: generated/profiler/reports/2026_11_02_10_06_55/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5554.9     | 4114.756 | 4.115  | 5061.0     | 5538.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3907.2     | 2894.244 | 2.894  | 3519.0     | 3846.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5299.9     | 3925.852 | 3.926  | 4858.0     | 5294.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6155.9     | 4559.963 | 4.560  | 5651.0     | 6145.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7057.1     | 5227.496 | 5.227  | 6566.0     | 7044.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5724.4     | 4240.319 | 4.240  | 4857.0     | 5692.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5628.9     | 4169.570 | 4.170  | 5237.0     | 5542.0        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5660.0     | 4192.593 | 4.193  | 5207.0     | 5606.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5142.4     | 3809.185 | 3.809  | 4560.0     | 5131.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5414.6     | 4010.852 | 4.011  | 4994.0     | 5370.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6239.6     | 4621.896 | 4.622  | 5774.0     | 6178.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7154.4     | 5299.526 | 5.300  | 6650.0     | 7089.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6956.2     | 5152.726 | 5.153  | 6394.0     | 6940.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6910.6     | 5118.933 | 5.119  | 6321.0     | 6939.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5626.8     | 4167.978 | 4.168  | 5029.0     | 5600.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5145.3     | 3811.304 | 3.811  | 4583.0     | 5159.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5391.5     | 3993.674 | 3.994  | 4962.0     | 5368.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6216.3     | 4604.659 | 4.605  | 5621.0     | 6192.5        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7131.8     | 5282.793 | 5.283  | 6535.0     | 7111.0        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6932.2     | 5135.000 | 5.135  | 4952.0     | 6960.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6887.8     | 5102.044 | 5.102  | 5847.0     | 6916.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5570.1     | 4126.037 | 4.126  | 5117.0     | 5586.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3776.4     | 2797.341 | 2.797  | 3473.0     | 3686.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5314.6     | 3936.770 | 3.937  | 4839.0     | 5313.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6172.5     | 4572.207 | 4.572  | 5626.0     | 6189.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7077.8     | 5242.822 | 5.243  | 6631.0     | 7100.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5653.6     | 4187.867 | 4.188  | 5412.0     | 5616.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5564.0     | 4121.459 | 4.121  | 3908.0     | 5514.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 5603.0     | 4150.341 | 4.150  | 5554.9                | 5598.5                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4492.8     | 3328.019 | 3.328  | 3776.4                | 4524.8                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5355.2     | 3966.787 | 3.967  | 5299.9                | 5353.1                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6196.1     | 4589.681 | 4.590  | 6155.9                | 6194.4                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7105.3     | 5263.159 | 5.263  | 7057.1                | 7104.8                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6316.6     | 4678.978 | 4.679  | 5653.6                | 6328.3                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6247.8     | 4628.002 | 4.628  | 5564.0                | 6258.3                   |

Final op time: 5.263 us (7105.3 cycles), bucket=TRISC_2 compute

