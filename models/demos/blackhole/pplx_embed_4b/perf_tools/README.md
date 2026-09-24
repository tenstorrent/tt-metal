# perf_tools

Scripts behind the numbers in `../PERF.md` and `../README.md`. Shell scripts locate the tt-metal checkout from their own path and use
`./python_env/bin/python`; every script takes the chip from `TT_VISIBLE_DEVICES` (the scripts that take a
`<chip>` argument set it themselves).

- `e2e_run_fp.py <batch> <iters>` — the extended-trace latency run (forward + pooling + I/O in one replay).
- `ab_one.sh <bs> <chip> "<ENV_A>" "<ENV_B>" [iters]` — same-chip sequential A/B, prints `RES ab …`.
- `ab_multi.sh <bs> <chip> "<ENV_A>" "<ENV_B>" <pairs> [iters]` — alternating launches (for bs16's two clock states).
- `sustained_run.sh <bs> <chip> <iters> <tag> "<ENV>"` — one run with tt-smi sampling: cold best, sustained median, AICLK, power.
- `bench_*.py`, `test_*.py` — standalone op benches and bit-identity tests (see the guide's table).
- `bench_common_traced.py` — shared trace-timing helper (`make_traced(device)`).
