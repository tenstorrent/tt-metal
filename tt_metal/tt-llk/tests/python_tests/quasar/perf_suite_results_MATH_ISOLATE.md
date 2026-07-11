# Quasar perf suite results

Started: 2026-07-11 10:02:19 UTC
Updated: 2026-07-11 14:02:07 UTC
Host: tensix-l-01-special-ndivnic-for-reservation-109931
Command pattern: `pytest -x --run-simulator --port=5556 --timeout=1000 --speed-of-light -k <PerfRunType> <test> > perf_output_<PerfRunType>_<suite_id>.txt`

| # | Run type | Test file | Status | Duration | Output | Reason |
|---|----------|-----------|--------|----------|--------|--------|
| 1 | `MATH_ISOLATE` | `perf_eltwise_unary_datacopy_quasar.py` | **FAILED** | 20m14s | `perf_output_MATH_ISOLATE_02.txt` | perf_eltwise_unary_datacopy_quasar.py::test_perf_eltwise_unary_datacopy_quasar[formats_dest_acc_data_copy_type_dims_dest_sync_dest_indices:(InputOutputFormat[A:MxInt8,B:MxInt8,out:Float16], <DestAccumulation.Yes: True>, <DataCopyType.B2D: 'B2D'>, [32, 32], <DestSync.Half: 'SyncHalf'>, 0, (16, 16))-i |
| 2 | `MATH_ISOLATE` | `perf_eltwise_binary_broadcast_quasar.py` | **EXALENS_TIMEOUT** | 55m10s | `perf_output_MATH_ISOLATE_03.txt` | tt-exalens did not become ready within 600s.  |
| 6 | `MATH_ISOLATE` | `perf_transpose_dest_quasar.py` | **PASSED** | 7m13s | `perf_output_MATH_ISOLATE_07.txt` | All collected tests in this file passed.  |
| 7 | `MATH_ISOLATE` | `perf_pack_quasar.py` | **PASSED** | 20m34s | `perf_output_MATH_ISOLATE_08.txt` | All collected tests in this file passed.  |
| 8 | `MATH_ISOLATE` | `perf_pack_untilize_quasar.py` | **PASSED** | 5m53s | `perf_output_MATH_ISOLATE_09.txt` | All collected tests in this file passed.  |
| 9 | `MATH_ISOLATE` | `perf_unary_broadcast_quasar.py` | **PASSED** | 1m32s | `perf_output_MATH_ISOLATE_10.txt` | All collected tests in this file passed.  |
| 10 | `MATH_ISOLATE` | `perf_pack_l1_acc_quasar.py` | **HANG** | 43m08s | `perf_output_MATH_ISOLATE_11.txt` | Single test exceeded 300s (hang).  |
| 13 | `MATH_ISOLATE` | `perf_unpack_reduce_col_tilizeA_strided_quasar.py` | **HANG** | 8m34s | `perf_output_MATH_ISOLATE_14.txt` | Single test exceeded 300s (hang).  |

## Notes

- `--speed-of-light` and `-k <PerfRunType>` applied for each of: MATH_ISOLATE.
- `-x` stops each file on first failure.
- tt-exalens 600s timeout: retry up to 5 times. Other failures are not retried.
- Single-test hang threshold: 300s after tt-exalens is ready (not retried).
- Emulator bring-up (tt-exalens / Zebu) is included in duration.
