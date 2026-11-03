# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_02_10_04_01
CCLs: sdpa_reduce_to_all

## sdpa_reduce_to_all

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_sdpa_reduce_to_all.py::test_sdpa_reduce_to_all_trace
Configured benchmark shape: [8, 4096]
Configured transport mode: fixed forwarder topology
Setup: 2 forwarder cores total; each forwarder core binds to one physical link, with BRISC/NCRISC handling the two traffic directions on that core.
Configured num_l_chunks: 2
Configured compute_block_sizes: 8
Sweep: num_l_chunks x compute_block_size x max_payload_size_bytes
For this shape, total_l_tiles per worker = 16.
Tuned payload matrix: 2048 -> (4, 8), 4096 -> (2, 4), 8192 -> (2, 8), 15232 -> (2, 8).
Invalid cells are shown as n/a when a forced L chunk payload exceeds the configured fabric max payload or a compute block size does not divide the total L tiles.
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = max(avg per top-level (RISC processor type, zone) bucket across devices).

### num_l_chunks=2, tiles_per_l_chunk=8, block_size=8, max_payload_size_bytes=8192

Report: generated/profiler/reports/2026_11_02_10_04_12/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                  | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | --------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6152.3     | 4557.281 | 4.557  | 4869.0     | 6166.5        |
| 0         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 4035.0     | 2988.859 | 2.989  | 3146.0     | 3672.0        |
| 0         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5878.9     | 4354.778 | 4.355  | 4598.0     | 5907.0        |
| 0         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6752.6     | 5001.904 | 5.002  | 5468.0     | 6782.5        |
| 0         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7655.5     | 5670.726 | 5.671  | 6384.0     | 7680.5        |
| 0         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5623.3     | 4165.393 | 4.165  | 4922.0     | 5320.5        |
| 0         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5585.9     | 4137.681 | 4.138  | 4828.0     | 5355.0        |
| 1         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6055.3     | 4485.407 | 4.485  | 3963.0     | 6075.0        |
| 1         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5320.6     | 3941.170 | 3.941  | 3394.0     | 5503.5        |
| 1         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5815.6     | 4307.859 | 4.308  | 3717.0     | 5807.5        |
| 1         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6607.2     | 4894.252 | 4.894  | 4440.0     | 6638.0        |
| 1         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7540.6     | 5585.622 | 5.586  | 5459.0     | 7553.5        |
| 1         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6962.4     | 5157.304 | 5.157  | 5193.0     | 7048.0        |
| 1         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 7022.2     | 5201.659 | 5.202  | 5257.0     | 7183.0        |
| 2         | NCRISC              | SDPA_REDUCE_READER    | 100   | 5981.4     | 4430.674 | 4.431  | 3737.0     | 6021.0        |
| 2         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 5290.7     | 3919.007 | 3.919  | 3616.0     | 5497.0        |
| 2         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5771.7     | 4275.319 | 4.275  | 3672.0     | 5800.5        |
| 2         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6561.9     | 4860.704 | 4.861  | 4275.0     | 6617.0        |
| 2         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7480.7     | 5541.281 | 5.541  | 5249.0     | 7525.5        |
| 2         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 6910.8     | 5119.104 | 5.119  | 4476.0     | 6986.5        |
| 2         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 6980.3     | 5170.600 | 5.171  | 5206.0     | 7197.5        |
| 3         | NCRISC              | SDPA_REDUCE_READER    | 100   | 6185.1     | 4581.519 | 4.582  | 4759.0     | 6199.5        |
| 3         | BRISC               | SDPA_REDUCE_WRITER    | 100   | 3968.6     | 2939.726 | 2.940  | 3218.0     | 3585.0        |
| 3         | TRISC_0             | SDPA_REDUCE_COMPUTE   | 100   | 5915.8     | 4382.074 | 4.382  | 4668.0     | 5913.5        |
| 3         | TRISC_1             | SDPA_REDUCE_COMPUTE   | 100   | 6784.5     | 5025.570 | 5.026  | 5104.0     | 6808.0        |
| 3         | TRISC_2             | SDPA_REDUCE_COMPUTE   | 100   | 7684.7     | 5692.385 | 5.692  | 6184.0     | 7708.5        |
| 3         | BRISC               | SDPA_REDUCE_FORWARDER | 100   | 5563.3     | 4120.993 | 4.121  | 4852.0     | 5249.0        |
| 3         | NCRISC              | SDPA_REDUCE_FORWARDER | 100   | 5538.7     | 4102.756 | 4.103  | 4037.0     | 5305.0        |

#### Bucket averages across devices

| RISC processor type | ZONE                  | DEVICE_COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_DEVICE_AVG_CYCLES | MEDIAN_DEVICE_AVG_CYCLES |
| ------------------- | --------------------- | ------------ | ---------- | -------- | ------ | --------------------- | ------------------------ |
| NCRISC              | SDPA_REDUCE_READER    | 4            | 6093.5     | 4513.720 | 4.514  | 5981.4                | 6103.8                   |
| BRISC               | SDPA_REDUCE_WRITER    | 4            | 4653.7     | 3447.191 | 3.447  | 3968.6                | 4662.8                   |
| TRISC_0             | SDPA_REDUCE_COMPUTE   | 4            | 5845.5     | 4330.007 | 4.330  | 5771.7                | 5847.3                   |
| TRISC_1             | SDPA_REDUCE_COMPUTE   | 4            | 6676.6     | 4945.607 | 4.946  | 6561.9                | 6679.9                   |
| TRISC_2             | SDPA_REDUCE_COMPUTE   | 4            | 7590.4     | 5622.504 | 5.623  | 7480.7                | 7598.0                   |
| BRISC               | SDPA_REDUCE_FORWARDER | 4            | 6264.9     | 4640.698 | 4.641  | 5563.3                | 6267.0                   |
| NCRISC              | SDPA_REDUCE_FORWARDER | 4            | 6281.8     | 4653.174 | 4.653  | 5538.7                | 6283.1                   |

Final op time: 5.623 us (7590.4 cycles), bucket=TRISC_2 compute

