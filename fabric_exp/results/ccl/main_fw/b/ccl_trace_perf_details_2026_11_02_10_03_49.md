# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_03_49
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

Report: generated/profiler/reports/2026_11_02_10_04_00/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6039.7     | 4473.881 | 4.474  | 3922.0     | 6123.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4380.4     | 3244.763 | 3.245  | 3363.0     | 4517.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5832.1     | 4320.052 | 4.320  | 3874.0     | 5917.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6564.4     | 4862.481 | 4.862  | 4328.0     | 6626.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7523.0     | 5572.578 | 5.573  | 5354.0     | 7613.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5991.2     | 4437.904 | 4.438  | 4842.0     | 6119.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6082.9     | 4505.815 | 4.506  | 5070.0     | 6049.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6181.6     | 4579.000 | 4.579  | 5093.0     | 6276.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5140.8     | 3807.963 | 3.808  | 4277.0     | 5114.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5916.9     | 4382.874 | 4.383  | 4873.0     | 5991.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6767.6     | 5013.044 | 5.013  | 5693.0     | 6876.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7671.6     | 5682.696 | 5.683  | 6585.0     | 7788.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6451.5     | 4778.874 | 4.779  | 4970.0     | 6433.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6288.5     | 4658.163 | 4.658  | 4969.0     | 6464.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6129.8     | 4540.556 | 4.541  | 5203.0     | 6205.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5128.9     | 3799.200 | 3.799  | 4335.0     | 5155.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5882.1     | 4357.074 | 4.357  | 4941.0     | 5942.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6728.0     | 4983.696 | 4.984  | 5798.0     | 6810.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7637.9     | 5657.726 | 5.658  | 6713.0     | 7700.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6451.1     | 4778.630 | 4.779  | 4374.0     | 6531.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6239.6     | 4621.896 | 4.622  | 5074.0     | 6225.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6045.0     | 4477.807 | 4.478  | 3838.0     | 6103.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4355.1     | 3225.985 | 3.226  | 3278.0     | 4393.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5855.7     | 4337.541 | 4.338  | 3848.0     | 5979.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6570.2     | 4866.837 | 4.867  | 4419.0     | 6581.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7519.6     | 5570.081 | 5.570  | 5335.0     | 7566.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5962.4     | 4416.593 | 4.417  | 4824.0     | 6092.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6020.8     | 4459.830 | 4.460  | 3904.0     | 5881.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 6099.0     | 4517.811 | 4.518  | 6039.7                | 6087.4                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4751.3     | 3519.478 | 3.519  | 4355.1                | 4754.7                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5871.7     | 4349.385 | 4.349  | 5832.1                | 5868.9                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6657.5     | 4931.515 | 4.932  | 6564.4                | 6649.1                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7588.0     | 5620.770 | 5.621  | 7519.6                | 7580.5                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6214.0     | 4603.000 | 4.603  | 5962.4                | 6221.2                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6157.9     | 4561.426 | 4.561  | 6020.8                | 6161.2                   |

Final op time: 5.621 us (7588.0 cycles), bucket=TRISC_2 compute

