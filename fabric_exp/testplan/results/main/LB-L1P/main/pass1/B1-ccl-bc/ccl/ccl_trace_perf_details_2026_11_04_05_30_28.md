# DeepSeek CCL trace perf matrix details

Timestamp: 2026_11_04_05_30_28
CCLs: broadcast

## broadcast

Test target: models/demos/deepseek_v3_b1/tests/unit_tests/test_ccl_broadcast.py::test_ccl_broadcast
Configured benchmark shape: [1, 7168]
Configured transport mode: configurable (1 or 2 links)
Configured num_links: 2
Trace id: 1
Source: profile_log_device.csv only (top-level zones only; no micro-profiling).
Op time = slowest device's slowest top-level bucket, using per-device averages across iterations only.

Cross-device bucket averages are omitted for this CCL.

### num_links=2, max_payload_size_bytes=4096

Report: generated/profiler/reports/2026_11_04_05_30_39/profile_log_device.csv
CHIP_FREQ[MHz]: 1350.000

#### Per-device averages

| PCIe slot | RISC processor type | ZONE                    | COUNT | AVG_CYCLES | AVG_NS   | AVG_US | MIN_CYCLES | MEDIAN_CYCLES |
| --------- | ------------------- | ----------------------- | ----- | ---------- | -------- | ------ | ---------- | ------------- |
| 0         | BRISC               | CCL_BROADCAST_READER    | 30    | 79.6       | 58.963   | 0.059  | 71.0       | 81.0          |
| 0         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10734.1    | 7951.210 | 7.951  | 10454.0    | 10738.5       |
| 1         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10767.8    | 7976.123 | 7.976  | 10506.0    | 10768.5       |
| 2         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10605.7    | 7856.049 | 7.856  | 7278.0     | 10680.0       |
| 3         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10581.0    | 7837.778 | 7.838  | 5324.0     | 10746.0       |
| 4         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10671.6    | 7904.864 | 7.905  | 9900.0     | 10666.5       |
| 5         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10762.1    | 7971.951 | 7.972  | 10504.0    | 10754.5       |
| 6         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10628.5    | 7872.963 | 7.873  | 7549.0     | 10712.0       |
| 7         | NCRISC              | CCL_BROADCAST_TRANSPORT | 30    | 10606.5    | 7856.642 | 7.857  | 5805.0     | 10755.5       |

Final op time: 7.976 us (10767.8 cycles), device=1, bucket=NCRISC transport

