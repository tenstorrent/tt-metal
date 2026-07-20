# Quasar perf suite progress summary

Updated: 2026-07-12 22:33:58 UTC
Host: (all-modes run)
Started: 2026-07-12 21:38:01 UTC

## Mode: ALL PerfRunTypes (no -k, no pytest --timeout)

Each test file runs L1_TO_L1 + UNPACK_ISOLATE + MATH_ISOLATE + PACK_ISOLATE
in a single parametrization (homogeneous CSV schema). L1_CONGESTION excluded (hangs).
On hang (~10 min no progress): clear local+Zebu, record PerfRunType skip in
perf_hang_skips.json, retry the same test.

Hang policy: open-test >600s or stall >600s after tt-exalens ready → kill local + Zebu,
record PerfRunType skip, retry.
Exalens 600s timeout → retry up to 5×.

| ID | Test | Status | Duration | Output | Reason |
|----|------|--------|----------|--------|--------|
| 02 | `perf_eltwise_unary_datacopy_quasar` | **PASSED** | 50m00s | `perf_output_all_02.txt` | All collected tests in this file passed. Hang-skips: skipped MATH_ISOLATE; skipped UNPACK_ISOLATE; |
| 03 | `perf_eltwise_binary_broadcast_quasar` | **PASSED** | 17m43s | `perf_output_all_03.txt` | All collected tests in this file passed. |
| 04 | `perf_eltwise_binary_quasar` | **PASSED** | 28m16s | `perf_output_all_04.txt` | All collected tests in this file passed. |
| 05 | `perf_unpack_tilize_quasar` | **PASSED** | 8m29s | `perf_output_all_05.txt` | All collected tests in this file passed. |
| 06 | `perf_unpack_unary_operand_quasar` | **PASSED** | 15m57s | `perf_output_all_06.txt` | All collected tests in this file passed. |
| 07 | `perf_transpose_dest_quasar` | **PASSED** | 26m22s | `perf_output_all_07.txt` | All collected tests in this file passed. |
| 08 | `perf_pack_quasar` | **PASSED** | 72m29s | `perf_output_all_08.txt` | All collected tests in this file passed. |
| 09 | `perf_pack_untilize_quasar` | **PASSED** | 18m11s | `perf_output_all_09.txt` | All collected tests in this file passed. |
| 10 | `perf_unary_broadcast_quasar` | **PASSED** | 20m01s | `perf_output_all_10.txt` | All collected tests in this file passed. |
| 11 | `perf_pack_l1_acc_quasar` | **TEST_FAILURE** | 71m20s | `perf_output_all_11.txt` | FAILED perf_pack_l1_acc_quasar.py::test_perf_pack_l1_acc_quasar[formats_dest_acc:(InputOutputFormat[A:UInt8,B:UInt8,out:Int8], <DestAccumulation.Yes: True>)-implied_math_format:Yes-dest_sync_mode:Half-input_dimensions:[512, 64]-run_types:PerfRunType.L1_TO_L1+PerfRunType.UNPACK_ISOLATE+PerfRunType.MATH_ISOLATE+PerfRunType.PACK_ISOLATE-loop_factor:32-is_perf:True] |
| 12 | `perf_reduce_quasar` | **EXALENS_TIMEOUT** | 55m57s | `perf_output_all_12.txt` | tt-exalens did not become ready within 600s. |
