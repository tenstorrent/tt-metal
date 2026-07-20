# Quasar perf suite results

Started: 2026-07-11 15:23:36 UTC
Updated: 2026-07-11 16:37:17 UTC
Host: tensix-l-01-special-ndivnic-for-reservation-109931
Command pattern: `pytest -x --run-simulator --port=5556 --timeout=1000 --speed-of-light -k PerfRunType.<PerfRunType> <test> > perf_output_<PerfRunType>_<suite_id>.txt`

| # | Run type | Test file | Status | Duration | Output | Reason |
|---|----------|-----------|--------|----------|--------|--------|
| 1 | `PACK_ISOLATE` | `perf_eltwise_unary_datacopy_quasar.py` | **EXALENS_TIMEOUT** | 55m10s | `perf_output_PACK_ISOLATE_02.txt` | tt-exalens did not become ready within 600s.  |
| 2 | `PACK_ISOLATE` | `perf_eltwise_binary_broadcast_quasar.py` | **FAILED** | 16m09s | `perf_output_PACK_ISOLATE_03.txt` | configfile: pytest.ini plugins: progress-1.4.0, cov-6.1.1, sugar-1.1.1, github-actions-annotate-failures-0.4.0, split-0.11.0, anyio-4.14.1, forked-1.6.0, timeout-2.4.0, random-order-1.2.0, xdist-3.8.0, repeat-0.9.4 timeout: 1000.0s timeout method: signal timeout func_only: False collected 365 items  |
| 3 | `PACK_ISOLATE` | `perf_eltwise_binary_quasar.py` | **FAILED** | 0m22s | `perf_output_PACK_ISOLATE_04.txt` | configfile: pytest.ini plugins: progress-1.4.0, cov-6.1.1, sugar-1.1.1, github-actions-annotate-failures-0.4.0, split-0.11.0, anyio-4.14.1, forked-1.6.0, timeout-2.4.0, random-order-1.2.0, xdist-3.8.0, repeat-0.9.4 timeout: 1000.0s timeout method: signal timeout func_only: False collected 325 items  |

## Notes

- `--speed-of-light` and `-k <PerfRunType>` applied for each of: PACK_ISOLATE.
- `-x` stops each file on first failure.
- tt-exalens 600s timeout: retry up to 5 times. Other failures are not retried.
- Single-test hang threshold: 300s after tt-exalens is ready (not retried).
- Emulator bring-up (tt-exalens / Zebu) is included in duration.
