# tt-triage Script Reliability Drill-Down
## Cohort A (First Runs) — 287 jobs analyzed

For each script, a breakdown of outcomes by specific error type.
Only non-PASS outcomes are shown in the detail tables.

---

### `dump_configuration.py`

**Summary**: 278 runs | PASS: 278 (100.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%) | ERRORED: 0 (0.0%) | ABSENT: 9

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| ABSENT | No triage section (E15) | 9 | 3.2% |

---

### `check_arc.py`

**Summary**: 278 runs | PASS: 226 (81.3%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 52 (18.7%) | ERRORED: 0 (0.0%) | ABSENT: 9

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Unsafe ARC access on remote WH device (E05) | 52 | 18.7% |
| ABSENT | No triage section (E15) | 9 | 3.2% |

---

### `check_cb_inactive.py`

**Summary**: 278 runs | PASS: 272 (97.8%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 6 (2.2%) | ERRORED: 0 (0.0%) | ABSENT: 9

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| ABSENT | No triage section (E15) | 9 | 3.2% |
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 6 | 2.2% |

---

### `check_eth_status.py`

**Summary**: 278 runs | PASS: 277 (99.6%) | EXPECTED: 1 (0.4%) | UNEXPECTED: 0 (0.0%) | ERRORED: 0 (0.0%) | ABSENT: 9

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| ABSENT | No triage section (E15) | 9 | 3.2% |
| EXPECTED | Ethernet link issue detected | 1 | 0.4% |

---

### `check_noc_locations.py`

**Summary**: 278 runs | PASS: 272 (97.8%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 6 (2.2%) | ERRORED: 0 (0.0%) | ABSENT: 9

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| ABSENT | No triage section (E15) | 9 | 3.2% |
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 6 | 2.2% |

---

### `device_info.py`

**Summary**: 278 runs | PASS: 226 (81.3%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 52 (18.7%) | ERRORED: 0 (0.0%) | ABSENT: 9

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Unsafe ARC access in Postcode column (E05) | 52 | 18.7% |
| ABSENT | No triage section (E15) | 9 | 3.2% |

---

### `device_telemetry.py`

**Summary**: 278 runs | PASS: 278 (100.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%) | ERRORED: 0 (0.0%) | ABSENT: 9

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| ABSENT | No triage section (E15) | 9 | 3.2% |

---

### `dump_running_operations.py`

**Summary**: 278 runs | PASS: 223 (80.2%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 55 (19.8%) | ERRORED: 0 (0.0%) | ABSENT: 9

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Current Op Name/Params N/A — metadata resolution failed | 55 | 19.8% |
| ABSENT | No triage section (E15) | 9 | 3.2% |

---

### `check_binary_integrity.py`

**Summary**: 278 runs | PASS: 249 (89.6%) | EXPECTED: 23 (8.3%) | UNEXPECTED: 6 (2.2%) | ERRORED: 0 (0.0%) | ABSENT: 9

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| EXPECTED | Binary corruption detected on device (E11) | 23 | 8.3% |
| ABSENT | No triage section (E15) | 9 | 3.2% |
| UNEXPECTED | No cores available (E10) | 6 | 2.2% |

---

### `check_core_magic.py`

**Summary**: 278 runs | PASS: 272 (97.8%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 6 (2.2%) | ERRORED: 0 (0.0%) | ABSENT: 9

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| ABSENT | No triage section (E15) | 9 | 3.2% |
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 6 | 2.2% |

---

### `check_noc_status.py`

**Summary**: 277 runs | PASS: 15 (5.4%) | EXPECTED: 195 (70.4%) | UNEXPECTED: 67 (24.2%) | ERRORED: 0 (0.0%) | ABSENT: 10

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| EXPECTED | NOC transaction mismatch detected (E16) | 195 | 70.4% |
| UNEXPECTED | UNEXPECTED: Missing noc_mode DWARF variable (E06) | 44 | 15.9% |
| UNEXPECTED | UNEXPECTED: Cores not halted, skipped (E08) | 17 | 6.1% |
| ABSENT | No triage section (E15) | 9 | 3.2% |
| UNEXPECTED | UNEXPECTED: No cores available (E10) | 6 | 2.2% |
| ABSENT | Script not present | 1 | 0.4% |

---

### `dump_aggregated_callstacks.py`

**Summary**: 277 runs | PASS: 90 (32.5%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 187 (67.5%) | ERRORED: 0 (0.0%) | ABSENT: 10

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | FD exhaustion — Errno 24 (E01) | 102 | 36.8% |
| UNEXPECTED | Missing fabric ERISC router ELF (E03) | 85 | 30.7% |
| ABSENT | No triage section (E15) | 9 | 3.2% |
| ABSENT | Script not present | 1 | 0.4% |

---

### `dump_callstacks.py`

**Summary**: 277 runs | PASS: 90 (32.5%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 187 (67.5%) | ERRORED: 0 (0.0%) | ABSENT: 10

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | FD exhaustion — Errno 24 (E01) | 102 | 36.8% |
| UNEXPECTED | Missing fabric ERISC router ELF (E03) | 85 | 30.7% |
| ABSENT | No triage section (E15) | 9 | 3.2% |
| ABSENT | Script not present | 1 | 0.4% |

---

### `dump_fast_dispatch.py`

**Summary**: 277 runs | PASS: 228 (82.3%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 48 (17.3%) | ERRORED: 1 (0.4%) | ABSENT: 10

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | Failed to halt/read dispatch core symbols (E09) | 48 | 17.3% |
| ABSENT | No triage section (E15) | 9 | 3.2% |
| ERRORED | Unhandled exception/traceback | 1 | 0.4% |
| ABSENT | Script not present | 1 | 0.4% |

---

### `dump_lightweight_asserts.py`

**Summary**: 277 runs | PASS: 160 (57.8%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 117 (42.2%) | ERRORED: 0 (0.0%) | ABSENT: 10

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| UNEXPECTED | FD exhaustion — Errno 24 (E01) | 102 | 36.8% |
| UNEXPECTED | Missing ELF file (E03) | 15 | 5.4% |
| ABSENT | No triage section (E15) | 9 | 3.2% |
| ABSENT | Script not present | 1 | 0.4% |

---

### `dump_watcher_ringbuffer.py`

**Summary**: 277 runs | PASS: 271 (97.8%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 6 (2.2%) | ERRORED: 0 (0.0%) | ABSENT: 10

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| ABSENT | No triage section (E15) | 9 | 3.2% |
| UNEXPECTED | No cores available — Skipping tensix/eth (E10) | 6 | 2.2% |
| ABSENT | Script not present | 1 | 0.4% |

---

### `firmware_versions.py`

**Summary**: 277 runs | PASS: 277 (100.0%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%) | ERRORED: 0 (0.0%) | ABSENT: 10

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| ABSENT | No triage section (E15) | 9 | 3.2% |
| ABSENT | Script not present | 1 | 0.4% |

---

### `system_info.py`

**Summary**: 277 runs | PASS: 175 (63.2%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%) | ERRORED: 102 (36.8%) | ABSENT: 10

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| ERRORED | Errno 24 cascade → can't open /etc/os-release | 102 | 36.8% |
| ABSENT | No triage section (E15) | 9 | 3.2% |
| ABSENT | Script not present | 1 | 0.4% |

---

### `check_broken_components.py`

**Summary**: 277 runs | PASS: 67 (24.2%) | EXPECTED: 108 (39.0%) | UNEXPECTED: 0 (0.0%) | ERRORED: 102 (36.8%) | ABSENT: 10

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| EXPECTED | Cores broken during triage halt/resume (E04) | 108 | 39.0% |
| ERRORED | Errno 24 cascade → Rich library can't load unicode data | 102 | 36.8% |
| ABSENT | No triage section (E15) | 9 | 3.2% |
| ABSENT | Script not present | 1 | 0.4% |

---

### `dump_risc_debug_signals.py`

**Summary**: 169 runs | PASS: 137 (81.1%) | EXPECTED: 0 (0.0%) | UNEXPECTED: 0 (0.0%) | ERRORED: 32 (18.9%) | ABSENT: 118

| Status | Reason | Count | % of runs |
|--------|--------|------:|----------:|
| ABSENT | Script not present | 109 | 64.5% |
| ERRORED | Unhandled exception/traceback | 32 | 18.9% |
| ABSENT | No triage section (E15) | 9 | 5.3% |

---
