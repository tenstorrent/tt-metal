# Quasar perf suite results

Started: 2026-07-11 10:22:23 UTC
Updated: 2026-07-11 10:22:35 UTC
Host: tensix-l-01-special-ndivnic-for-reservation-109931
Command pattern: `pytest -x --run-simulator --port=5556 --timeout=1000 --speed-of-light -k <PerfRunType> <test> > perf_output_<PerfRunType>_<suite_id>.txt`

| # | Run type | Test file | Status | Duration | Output | Reason |
|---|----------|-----------|--------|----------|--------|--------|
| 1 | `UNPACK_ISOLATE` | `perf_eltwise_unary_datacopy_quasar.py` | **PASSED** | 24m04s | `perf_output_UNPACK_ISOLATE_02.txt` | All collected tests in this file passed.  |
| 2 | `UNPACK_ISOLATE` | `perf_eltwise_binary_broadcast_quasar.py` | **PASSED** | 13m13s | `perf_output_UNPACK_ISOLATE_03.txt` | All collected tests in this file passed.  |
| 3 | `UNPACK_ISOLATE` | `perf_eltwise_binary_quasar.py` | **PASSED** | 4m03s | `perf_output_UNPACK_ISOLATE_04.txt` | All collected tests in this file passed.  |
| 4 | `UNPACK_ISOLATE` | `perf_unpack_tilize_quasar.py` | **PASSED** | 2m32s | `perf_output_UNPACK_ISOLATE_05.txt` | All collected tests in this file passed.  |
| 5 | `UNPACK_ISOLATE` | `perf_unpack_unary_operand_quasar.py` | **PASSED** | 2m32s | `perf_output_UNPACK_ISOLATE_06.txt` | All collected tests in this file passed.  |
| 6 | `UNPACK_ISOLATE` | `perf_transpose_dest_quasar.py` | **PASSED** | 4m43s | `perf_output_UNPACK_ISOLATE_07.txt` | All collected tests in this file passed.  |
| 7 | `UNPACK_ISOLATE` | `perf_pack_quasar.py` | **PASSED** | 16m23s | `perf_output_UNPACK_ISOLATE_08.txt` | All collected tests in this file passed.  |
| 8 | `UNPACK_ISOLATE` | `perf_pack_untilize_quasar.py` | **PASSED** | 15m16s | `perf_output_UNPACK_ISOLATE_09.txt` | All collected tests in this file passed.  |
| 9 | `UNPACK_ISOLATE` | `perf_unary_broadcast_quasar.py` | **HANG** | 21m35s | `perf_output_UNPACK_ISOLATE_10.txt` | Single test exceeded 300s (hang).  |
| 10 | `UNPACK_ISOLATE` | `perf_pack_l1_acc_quasar.py` | **FAILED** | 0m12s | `perf_output_UNPACK_ISOLATE_11.txt` |  |

## Notes

- `--speed-of-light` and `-k <PerfRunType>` applied for each of: UNPACK_ISOLATE.
- `-x` stops each file on first failure.
- tt-exalens 600s timeout: retry up to 5 times. Other failures are not retried.
- Single-test hang threshold: 300s after tt-exalens is ready (not retried).
- Emulator bring-up (tt-exalens / Zebu) is included in duration.
