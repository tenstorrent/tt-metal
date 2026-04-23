# tt-triage Script Reliability Drill-Down

## Cohort A (First Runs on Fresh Device) — 83 jobs analyzed

For each script, a breakdown of outcomes by specific error type. Only non-PASS outcomes are shown in the detail tables.

---

### `dump_configuration.py`

**Summary**: 83 runs | PASS: 83 (100.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%)

*All runs passed — no issues detected.*

---

### `check_arc.py`

**Summary**: 83 runs | PASS: 60 (72.3%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 23 (27.7%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Unsafe ARC access on remote WH device (E05) | 23 | 27.7% |

---

### `check_cb_inactive.py`

**Summary**: 83 runs | PASS: 78 (94.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 5 (6.0%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 5 | 6.0% |

---

### `check_eth_status.py`

**Summary**: 83 runs | PASS: 83 (100.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%)

*All runs passed — no issues detected.*

---

### `check_noc_locations.py`

**Summary**: 83 runs | PASS: 78 (94.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 5 (6.0%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 5 | 6.0% |

---

### `device_info.py`

**Summary**: 83 runs | PASS: 60 (72.3%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 23 (27.7%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Unsafe ARC access in Postcode column (E05) | 23 | 27.7% |

---

### `device_telemetry.py`

**Summary**: 83 runs | PASS: 83 (100.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%)

*All runs passed — no issues detected.*

---

### `dump_running_operations.py`

**Summary**: 83 runs | PASS: 80 (96.4%) | EXPECTED: 2 (2.4%) | UNEXPECTED: 1 (1.2%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| EXPECTED | MatMul preceded hang — likely di/dt (E25) | 2 | 2.4% |
| UNEXPECTED | Current Op Name/Params N/A — metadata resolution failed (fast dispatch) | 1 | 1.2% |

---

### `check_binary_integrity.py`

**Summary**: 83 runs | PASS: 57 (68.7%) | EXPECTED: 21 (25.3%) | UNEXPECTED: 5 (6.0%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| EXPECTED | Binary corruption detected on device (E11) | 21 | 25.3% |
| UNEXPECTED | No cores available (E10) | 5 | 6.0% |

---

### `check_core_magic.py`

**Summary**: 83 runs | PASS: 78 (94.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 5 (6.0%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 5 | 6.0% |

---

### `check_noc_status.py`

**Summary**: 83 runs | PASS: 7 (8.4%) | EXPECTED: 35 (42.2%) | UNEXPECTED: 41 (49.4%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | UNEXPECTED: Missing noc_mode DWARF variable (E06) | 35 | 42.2% |
| EXPECTED | NOC transaction mismatch detected (E16) | 35 | 42.2% |
| UNEXPECTED | UNEXPECTED: No cores available (E10) | 5 | 6.0% |
| UNEXPECTED | UNEXPECTED: Cores not halted, skipped (E08) | 1 | 1.2% |

---

### `dump_aggregated_callstacks.py`

**Summary**: 83 runs | PASS: 53 (63.9%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 30 (36.1%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Missing fabric ERISC router ELF (E03) | 30 | 36.1% |

---

### `dump_fast_dispatch.py`

**Summary**: 83 runs | PASS: 81 (97.6%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 2 (2.4%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Failed to halt/read dispatch core symbols (E09) | 1 | 1.2% |
| UNEXPECTED | Unhandled exception/traceback | 1 | 1.2% |

---

### `dump_lightweight_asserts.py`

**Summary**: 82 runs | PASS: 60 (73.2%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 22 (26.8%) | ABSENT: 1

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Missing ELF file (E03) | 22 | 26.8% |
| ABSENT | Script not present | 1 | 1.2% |

---

### `dump_watcher_ringbuffer.py`

**Summary**: 82 runs | PASS: 77 (93.9%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 5 (6.1%) | ABSENT: 1

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 5 | 6.1% |
| ABSENT | Script not present | 1 | 1.2% |

---

### `firmware_versions.py`

**Summary**: 82 runs | PASS: 82 (100.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%) | ABSENT: 1

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| ABSENT | Script not present | 1 | 1.2% |

---

### `system_info.py`

**Summary**: 82 runs | PASS: 82 (100.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%) | ABSENT: 1

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| ABSENT | Script not present | 1 | 1.2% |

---

### `check_broken_components.py`

**Summary**: 82 runs | PASS: 57 (69.5%) | EXPECTED: 25 (30.5%) | UNEXPECTED: 0 (0.0%) | ABSENT: 1

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| EXPECTED | Cores broken during triage halt/resume (E04) | 25 | 30.5% |
| ABSENT | Script not present | 1 | 1.2% |

---

### `dump_risc_debug_signals.py`

**Summary**: 75 runs | PASS: 51 (68.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 24 (32.0%) | ABSENT: 8

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Unhandled exception/traceback | 24 | 32.0% |
| ABSENT | Script not present | 8 | 10.7% |

---

## Cohort B (Subsequent Runs on Contaminated Device) — 106 runs analyzed

For each script, a breakdown of outcomes by specific error type. Only non-PASS outcomes are shown in the detail tables.

---

### `dump_configuration.py`

**Summary**: 106 runs | PASS: 106 (100.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%)

*All runs passed — no issues detected.*

---

### `check_arc.py`

**Summary**: 106 runs | PASS: 71 (67.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 35 (33.0%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Unsafe ARC access on remote WH device (E05) | 35 | 33.0% |

---

### `check_cb_inactive.py`

**Summary**: 106 runs | PASS: 14 (13.2%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 92 (86.8%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 92 | 86.8% |

---

### `check_eth_status.py`

**Summary**: 106 runs | PASS: 106 (100.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%)

*All runs passed — no issues detected.*

---

### `check_noc_locations.py`

**Summary**: 106 runs | PASS: 14 (13.2%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 92 (86.8%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 92 | 86.8% |

---

### `device_info.py`

**Summary**: 106 runs | PASS: 71 (67.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 35 (33.0%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Unsafe ARC access in Postcode column (E05) | 35 | 33.0% |

---

### `device_telemetry.py`

**Summary**: 106 runs | PASS: 106 (100.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%)

*All runs passed — no issues detected.*

---

### `dump_running_operations.py`

**Summary**: 106 runs | PASS: 104 (98.1%) | EXPECTED: 2 (1.9%) | UNEXPECTED: 0 (0.0%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| EXPECTED | MatMul preceded hang — likely di/dt (E25) | 2 | 1.9% |

---

### `check_binary_integrity.py`

**Summary**: 106 runs | PASS: 11 (10.4%) | EXPECTED: 3 (2.8%) | UNEXPECTED: 92 (86.8%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | No cores available (E10) | 92 | 86.8% |
| EXPECTED | Binary corruption detected on device (E11) | 3 | 2.8% |

---

### `check_core_magic.py`

**Summary**: 106 runs | PASS: 14 (13.2%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 92 (86.8%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 92 | 86.8% |

---

### `check_noc_status.py`

**Summary**: 106 runs | PASS: 3 (2.8%) | EXPECTED: 11 (10.4%) | UNEXPECTED: 92 (86.8%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | UNEXPECTED: No cores available (E10) | 92 | 86.8% |
| EXPECTED | NOC transaction mismatch detected (E16) | 11 | 10.4% |

---

### `dump_aggregated_callstacks.py`

**Summary**: 106 runs | PASS: 101 (95.3%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 5 (4.7%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Missing fabric ERISC router ELF (E03) | 5 | 4.7% |

---

### `dump_fast_dispatch.py`

**Summary**: 106 runs | PASS: 105 (99.1%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 1 (0.9%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Failed to halt/read dispatch core symbols (E09) | 1 | 0.9% |

---

### `dump_lightweight_asserts.py`

**Summary**: 106 runs | PASS: 100 (94.3%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 6 (5.7%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Missing ELF file (E03) | 6 | 5.7% |

---

### `dump_watcher_ringbuffer.py`

**Summary**: 106 runs | PASS: 14 (13.2%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 92 (86.8%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 92 | 86.8% |

---

### `firmware_versions.py`

**Summary**: 106 runs | PASS: 106 (100.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%)

*All runs passed — no issues detected.*

---

### `system_info.py`

**Summary**: 106 runs | PASS: 106 (100.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%)

*All runs passed — no issues detected.*

---

### `check_broken_components.py`

**Summary**: 106 runs | PASS: 98 (92.5%) | EXPECTED: 8 (7.5%) | UNEXPECTED: 0 (0.0%)

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| EXPECTED | Cores broken during triage halt/resume (E04) | 8 | 7.5% |

---

### `dump_risc_debug_signals.py`

**Summary**: 105 runs | PASS: 105 (100.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%) | ABSENT: 1

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| ABSENT | Script not present | 1 | 1.0% |

---

## Week-over-Week — Cohort A

Comparison against `triage_script_drilldown_20260415.csv`. Threshold: |Δ pp| > 5 = REGRESSION/IMPROVEMENT; appeared from 0 = NEW; cleared to 0 = CLEARED.

### `dump_configuration.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 100.0% | 100.0% | +0 |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 0.0% | 0.0% | +0 |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |

---

### `check_arc.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 81.3% | 72.3% | $\color{red}{-9.0}$ |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 18.7% | 27.7% | $\color{red}{+9.0}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| UNEXPECTED | Unsafe ARC access on remote WH device (E05) | 18.7% | 27.7% | $\color{red}{+9.0}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |

---

### `check_cb_inactive.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 97.8% | 94.0% | $\color{red}{-3.8}$ |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 2.2% | 6.0% | $\color{red}{+3.8}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 2.2% | 6.0% | $\color{red}{+3.8}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |

---

### `check_eth_status.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 99.6% | 100.0% | $\color{green}{+0.4}$ |
| EXPECTED | 0.4% | 0.0% | $\color{green}{-0.4}$ |
| UNEXPECTED | 0.0% | 0.0% | +0 |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |
| EXPECTED | Ethernet link issue detected | 0.4% | 0.0% | $\color{green}{-0.4}$ |

---

### `check_noc_locations.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 97.8% | 94.0% | $\color{red}{-3.8}$ |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 2.2% | 6.0% | $\color{red}{+3.8}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 2.2% | 6.0% | $\color{red}{+3.8}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |

---

### `device_info.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 81.3% | 72.3% | $\color{red}{-9.0}$ |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 18.7% | 27.7% | $\color{red}{+9.0}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| UNEXPECTED | Unsafe ARC access in Postcode column (E05) | 18.7% | 27.7% | $\color{red}{+9.0}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |

---

### `device_telemetry.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 100.0% | 100.0% | +0 |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 0.0% | 0.0% | +0 |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |

---

### `dump_running_operations.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 80.2% | 96.4% | $\color{green}{+16.2}$ |
| EXPECTED | 0.0% | 2.4% | $\color{red}{+2.4}$ |
| UNEXPECTED | 19.8% | 1.2% | $\color{green}{-18.6}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| UNEXPECTED | Current Op Name/Params N/A — metadata resolution failed | 19.8% | 0.0% | $\color{green}{-19.8}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |
| EXPECTED | MatMul preceded hang — likely di/dt (E25) | 0.0% | 2.4% | $\color{red}{+2.4}$ |
| UNEXPECTED | Current Op Name/Params N/A — metadata resolution failed (fast dispatch) | 0.0% | 1.2% | $\color{red}{+1.2}$ |

---

### `check_binary_integrity.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 89.6% | 68.7% | $\color{red}{-20.9}$ |
| EXPECTED | 8.3% | 25.3% | $\color{red}{+17.0}$ |
| UNEXPECTED | 2.2% | 6.0% | $\color{red}{+3.8}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| EXPECTED | Binary corruption detected on device (E11) | 8.3% | 25.3% | $\color{red}{+17.0}$ |
| UNEXPECTED | No cores available (E10) | 2.2% | 6.0% | $\color{red}{+3.8}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |

---

### `check_core_magic.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 97.8% | 94.0% | $\color{red}{-3.8}$ |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 2.2% | 6.0% | $\color{red}{+3.8}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 2.2% | 6.0% | $\color{red}{+3.8}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |

---

### `check_noc_status.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 5.4% | 8.4% | $\color{green}{+3.0}$ |
| EXPECTED | 70.4% | 42.2% | $\color{green}{-28.2}$ |
| UNEXPECTED | 24.2% | 49.4% | $\color{red}{+25.2}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| EXPECTED | NOC transaction mismatch detected (E16) | 70.4% | 42.2% | $\color{green}{-28.2}$ |
| UNEXPECTED | UNEXPECTED: Missing noc_mode DWARF variable (E06) | 15.9% | 42.2% | $\color{red}{+26.3}$ |
| UNEXPECTED | UNEXPECTED: Cores not halted, skipped (E08) | 6.1% | 1.2% | $\color{green}{-4.9}$ |
| UNEXPECTED | UNEXPECTED: No cores available (E10) | 2.2% | 6.0% | $\color{red}{+3.8}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |
| ABSENT | Script not present | 0.4% | 0.0% | $\color{green}{-0.4}$ |

---

### `dump_aggregated_callstacks.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 32.5% | 63.9% | $\color{green}{+31.4}$ |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 67.5% | 36.1% | $\color{green}{-31.4}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| UNEXPECTED | FD exhaustion — Errno 24 (E01) | 36.8% | 0.0% | $\color{green}{-36.8}$ |
| UNEXPECTED | Missing fabric ERISC router ELF (E03) | 30.7% | 36.1% | $\color{red}{+5.4}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |
| ABSENT | Script not present | 0.4% | 0.0% | $\color{green}{-0.4}$ |

---

### `dump_callstacks.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 32.5% | 0.0% | $\color{red}{-32.5}$ |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 67.5% | 0.0% | $\color{green}{-67.5}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| UNEXPECTED | FD exhaustion — Errno 24 (E01) | 36.8% | 0.0% | $\color{green}{-36.8}$ |
| UNEXPECTED | Missing fabric ERISC router ELF (E03) | 30.7% | 0.0% | $\color{green}{-30.7}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |
| ABSENT | Script not present | 0.4% | 0.0% | $\color{green}{-0.4}$ |

---

### `dump_fast_dispatch.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 82.3% | 97.6% | $\color{green}{+15.3}$ |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 17.7% | 2.4% | $\color{green}{-15.3}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| UNEXPECTED | Failed to halt/read dispatch core symbols (E09) | 17.3% | 1.2% | $\color{green}{-16.1}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |
| UNEXPECTED | Unhandled exception/traceback | 0.4% | 1.2% | $\color{red}{+0.8}$ |
| ABSENT | Script not present | 0.4% | 0.0% | $\color{green}{-0.4}$ |

---

### `dump_lightweight_asserts.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 57.8% | 73.2% | $\color{green}{+15.4}$ |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 42.2% | 26.8% | $\color{green}{-15.4}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| UNEXPECTED | FD exhaustion — Errno 24 (E01) | 36.8% | 0.0% | $\color{green}{-36.8}$ |
| UNEXPECTED | Missing ELF file (E03) | 5.4% | 26.8% | $\color{red}{+21.4}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |
| ABSENT | Script not present | 0.4% | 1.2% | $\color{red}{+0.8}$ |

---

### `dump_watcher_ringbuffer.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 97.8% | 93.9% | $\color{red}{-3.9}$ |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 2.2% | 6.1% | $\color{red}{+3.9}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 2.2% | 6.1% | $\color{red}{+3.9}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |
| ABSENT | Script not present | 0.4% | 1.2% | $\color{red}{+0.8}$ |

---

### `firmware_versions.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 100.0% | 100.0% | +0 |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 0.0% | 0.0% | +0 |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |
| ABSENT | Script not present | 0.4% | 1.2% | $\color{red}{+0.8}$ |

---

### `system_info.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 63.2% | 100.0% | $\color{green}{+36.8}$ |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 36.8% | 0.0% | $\color{green}{-36.8}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| UNEXPECTED | Errno 24 cascade → can't open /etc/os-release | 36.8% | 0.0% | $\color{green}{-36.8}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |
| ABSENT | Script not present | 0.4% | 1.2% | $\color{red}{+0.8}$ |

---

### `check_broken_components.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 24.2% | 69.5% | $\color{green}{+45.3}$ |
| EXPECTED | 39.0% | 30.5% | $\color{green}{-8.5}$ |
| UNEXPECTED | 36.8% | 0.0% | $\color{green}{-36.8}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| UNEXPECTED | Errno 24 cascade → Rich library can't load unicode data | 36.8% | 0.0% | $\color{green}{-36.8}$ |
| EXPECTED | Cores broken during triage halt/resume (E04) | 39.0% | 30.5% | $\color{green}{-8.5}$ |
| ABSENT | No triage section (E15) | 3.2% | 0.0% | $\color{green}{-3.2}$ |
| ABSENT | Script not present | 0.4% | 1.2% | $\color{red}{+0.8}$ |

---

### `dump_risc_debug_signals.py`

| Status | Last Week % | This Week % | Δ pp |
|--------|------------:|------------:|-----:|
| PASS | 81.1% | 68.0% | $\color{red}{-13.1}$ |
| EXPECTED | 0.0% | 0.0% | +0 |
| UNEXPECTED | 18.9% | 32.0% | $\color{red}{+13.1}$ |

| Status | Reason | Last Week | This Week | Δ pp |
|--------|--------|----------:|----------:|-----:|
| ABSENT | Script not present | 64.5% | 10.7% | $\color{green}{-53.8}$ |
| UNEXPECTED | Unhandled exception/traceback | 18.9% | 32.0% | $\color{red}{+13.1}$ |
| ABSENT | No triage section (E15) | 5.3% | 0.0% | $\color{green}{-5.3}$ |

---
