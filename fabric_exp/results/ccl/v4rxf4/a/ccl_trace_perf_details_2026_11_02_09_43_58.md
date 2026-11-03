# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_09_43_58
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

Report: generated/profiler/reports/2026_11_02_09_44_13/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6452.3     | 4779.481 | 4.779  | 4805.0     | 6359.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5006.0     | 3708.126 | 3.708  | 4154.0     | 4968.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6224.1     | 4610.407 | 4.610  | 4524.0     | 6102.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6954.1     | 5151.193 | 5.151  | 5375.0     | 6964.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7932.5     | 5875.956 | 5.876  | 6278.0     | 7859.0        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6433.9     | 4765.844 | 4.766  | 4016.0     | 6391.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6460.5     | 4785.585 | 4.786  | 5470.0     | 6563.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5921.3     | 4386.178 | 4.386  | 4420.0     | 5917.5        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5086.8     | 3768.000 | 3.768  | 3827.0     | 5170.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5655.8     | 4189.467 | 4.189  | 4291.0     | 5648.0        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6511.6     | 4823.430 | 4.823  | 4781.0     | 6505.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7426.7     | 5501.230 | 5.501  | 5844.0     | 7420.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6356.5     | 4708.511 | 4.709  | 5273.0     | 6046.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6151.4     | 4556.585 | 4.557  | 5274.0     | 6069.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5901.6     | 4371.533 | 4.372  | 4343.0     | 5893.5        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5078.0     | 3761.452 | 3.761  | 3820.0     | 5224.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5645.4     | 4181.793 | 4.182  | 4381.0     | 5622.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6505.9     | 4819.207 | 4.819  | 4873.0     | 6500.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7417.4     | 5494.333 | 5.494  | 5880.0     | 7399.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6353.4     | 4706.200 | 4.706  | 5582.0     | 6056.0        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6133.1     | 4543.000 | 4.543  | 5199.0     | 6109.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6462.0     | 4786.667 | 4.787  | 4601.0     | 6369.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5003.5     | 3706.267 | 3.706  | 4019.0     | 4987.5        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 6221.2     | 4608.267 | 4.608  | 4335.0     | 6078.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6954.0     | 5151.089 | 5.151  | 5205.0     | 6934.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7932.2     | 5875.741 | 5.876  | 6089.0     | 7860.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6437.0     | 4768.170 | 4.768  | 4702.0     | 6383.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6439.0     | 4769.615 | 4.770  | 4218.0     | 6635.5        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 6184.3     | 4580.965 | 4.581  | 5901.6                | 6186.8                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 5043.5     | 3735.961 | 3.736  | 5003.5                | 5042.0                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5936.6     | 4397.483 | 4.397  | 5645.4                | 5938.5                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6731.4     | 4986.230 | 4.986  | 6505.9                | 6732.8                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7677.2     | 5686.815 | 5.687  | 7417.4                | 7679.5                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6395.2     | 4737.181 | 4.737  | 6353.4                | 6395.2                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6296.0     | 4663.696 | 4.664  | 6133.1                | 6295.2                   |

Final op time: 5.687 us (7677.2 cycles), bucket=TRISC_2 compute

