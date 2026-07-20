# Quasar perf suite results

Started: 2026-07-11 02:34:56 UTC
Updated: 2026-07-11 02:34:56 UTC
Host: tensix-l-01-special-ndivnic-for-reservation-109931
Command pattern: `pytest -x --run-simulator --port=5556 --timeout=1000 --speed-of-light -k <PerfRunType> <test> > perf_output_<PerfRunType>_<suite_id>.txt`

| # | Run type | Test file | Status | Duration | Output | Reason |
|---|----------|-----------|--------|----------|--------|--------|

## Notes

- `--speed-of-light` and `-k <PerfRunType>` applied for each of: L1_CONGESTION.
- `-x` stops each file on first failure.
- tt-exalens 600s timeout: retry up to 5 times. Other failures are not retried.
- Single-test hang threshold: 300s after tt-exalens is ready (not retried).
- Emulator bring-up (tt-exalens / Zebu) is included in duration.
