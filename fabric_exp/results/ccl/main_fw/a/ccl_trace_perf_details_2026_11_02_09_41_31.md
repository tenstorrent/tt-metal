# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_41_31
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

Report: generated/profiler/reports/2026_11_02_09_41_43/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6169.4     | 4569.926 | 4.570  | 5739.0     | 6131.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4666.6     | 3456.726 | 3.457  | 4276.0     | 4636.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5939.1     | 4399.304 | 4.399  | 5633.0     | 5891.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6761.1     | 5008.237 | 5.008  | 6323.0     | 6732.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7669.8     | 5681.348 | 5.681  | 7237.0     | 7635.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6784.2     | 5025.311 | 5.025  | 6242.0     | 6786.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6788.1     | 5028.259 | 5.028  | 6455.0     | 6779.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6205.6     | 4596.778 | 4.597  | 5764.0     | 6154.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4811.2     | 3563.874 | 3.564  | 4524.0     | 4756.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5949.8     | 4407.274 | 4.407  | 5574.0     | 5877.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6795.8     | 5033.919 | 5.034  | 6371.0     | 6750.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7702.0     | 5705.156 | 5.705  | 7260.0     | 7647.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6930.3     | 5133.556 | 5.134  | 6580.0     | 6920.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6894.2     | 5106.852 | 5.107  | 6566.0     | 6886.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6163.9     | 4565.859 | 4.566  | 5794.0     | 6113.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4819.5     | 3570.015 | 3.570  | 4497.0     | 4759.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5937.3     | 4397.978 | 4.398  | 5608.0     | 5882.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6766.6     | 5012.311 | 5.012  | 6415.0     | 6724.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7670.0     | 5681.496 | 5.681  | 7304.0     | 7619.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6944.8     | 5144.326 | 5.144  | 5959.0     | 6937.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6917.4     | 5124.022 | 5.124  | 6191.0     | 6918.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6167.9     | 4568.800 | 4.569  | 5811.0     | 6136.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4641.3     | 3438.022 | 3.438  | 4294.0     | 4614.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5927.5     | 4390.748 | 4.391  | 5620.0     | 5888.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6764.1     | 5010.430 | 5.010  | 6379.0     | 6743.5        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7670.6     | 5681.933 | 5.682  | 7297.0     | 7654.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6755.3     | 5003.896 | 5.004  | 5792.0     | 6743.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6734.3     | 4988.341 | 4.988  | 5417.0     | 6745.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 6176.7     | 4575.341 | 4.575  | 6163.9                | 6168.6                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4734.7     | 3507.159 | 3.507  | 4641.3                | 4738.9                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5938.4     | 4398.826 | 4.399  | 5927.5                | 5938.2                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6771.9     | 5016.224 | 5.016  | 6761.1                | 6765.4                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7678.1     | 5687.483 | 5.687  | 7669.8                | 7670.3                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6853.6     | 5076.772 | 5.077  | 6755.3                | 6857.2                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6833.5     | 5061.869 | 5.062  | 6734.3                | 6841.2                   |

Final op time: 5.687 us (7678.1 cycles), bucket=TRISC_2 compute

