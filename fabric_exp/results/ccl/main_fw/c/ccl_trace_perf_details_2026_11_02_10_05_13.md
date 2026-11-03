# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_05_13
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

Report: generated/profiler/reports/2026_11_02_10_05_23/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6184.3     | 4580.978 | 4.581  | 5836.0     | 6148.0        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4665.9     | 3456.230 | 3.456  | 4323.0     | 4635.5        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5955.9     | 4411.741 | 4.412  | 5648.0     | 5911.5        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6780.9     | 5022.904 | 5.023  | 6424.0     | 6755.0        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7684.2     | 5691.970 | 5.692  | 7315.0     | 7651.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6747.7     | 4998.281 | 4.998  | 6104.0     | 6736.0        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6734.6     | 4988.585 | 4.989  | 6363.0     | 6718.5        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6184.9     | 4581.444 | 4.581  | 5714.0     | 6126.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4843.9     | 3588.074 | 3.588  | 4509.0     | 4818.0        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5929.2     | 4392.015 | 4.392  | 5584.0     | 5856.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6775.9     | 5019.207 | 5.019  | 6290.0     | 6727.5        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7683.3     | 5691.326 | 5.691  | 7209.0     | 7629.0        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6996.8     | 5182.822 | 5.183  | 6612.0     | 6987.5        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6918.6     | 5124.881 | 5.125  | 6577.0     | 6899.5        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6153.6     | 4558.185 | 4.558  | 5849.0     | 6111.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4817.9     | 3568.785 | 3.569  | 4399.0     | 4774.5        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5913.8     | 4380.593 | 4.381  | 5608.0     | 5861.0        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6752.4     | 5001.763 | 5.002  | 6377.0     | 6711.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7657.5     | 5672.237 | 5.672  | 7329.0     | 7605.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 7003.7     | 5187.911 | 5.188  | 5743.0     | 6983.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6930.7     | 5133.822 | 5.134  | 6155.0     | 6935.0        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6167.8     | 4568.704 | 4.569  | 5832.0     | 6130.0        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4638.9     | 3436.237 | 3.436  | 4281.0     | 4618.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5924.7     | 4388.689 | 4.389  | 5636.0     | 5869.0        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6764.6     | 5010.844 | 5.011  | 6395.0     | 6729.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7671.7     | 5682.763 | 5.683  | 7326.0     | 7638.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6731.8     | 4986.481 | 4.986  | 5992.0     | 6745.5        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6692.2     | 4957.215 | 4.957  | 4690.0     | 6710.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 6172.6     | 4572.328 | 4.572  | 6153.6                | 6176.0                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4741.6     | 3512.331 | 3.512  | 4638.9                | 4741.9                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5930.9     | 4393.259 | 4.393  | 5913.8                | 5927.0                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6768.5     | 5013.680 | 5.014  | 6752.4                | 6770.3                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7674.2     | 5684.574 | 5.685  | 7657.5                | 7677.5                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6870.0     | 5088.874 | 5.089  | 6731.8                | 6872.2                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6819.0     | 5051.126 | 5.051  | 6692.2                | 6826.6                   |

Final op time: 5.685 us (7674.2 cycles), bucket=TRISC_2 compute

