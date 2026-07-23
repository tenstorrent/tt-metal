Optimization summary — kokoro_82m · main (device_ms)
====================================================
baseline/final ms unavailable (no baseline profile found)

(no kernel attempts recorded — nothing was tried, or the run stopped before any lever)

committed wins: 0 (branch HEAD)

Limitations / suggested manual next steps:
- (none flagged automatically — see the per-op device report for remaining headroom.)

Reproduce:
  perf test:  python -m pytest models/demos/kokoro_82m/tests/e2e/test_main_perf.py::test_main_perf -svv
  per-op device report (tt-metal format): /tmp/tt_hw_planner_hexgrad_Kokoro-82M_1783376840/models/experimental/perf_automation/runs/2026-07-07T01-56-33/profiles/iter_baseline_report.csv

levels: grid -> dtype -> tt-lang -> cpp -> host   |   ✓win = beat baseline, ·try = measured no-gain, — = not attempted
