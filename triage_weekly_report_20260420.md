# tt-triage Weekly Analysis Report
## Week of 2026-04-15 to 2026-04-20

### Executive Summary
- Total hang jobs: 83
- Jobs with triage output: 83
- Init failures (no triage output): 0 (0.0%)
- Triage completed: 77 (92.8%)
- Aborted runs (triage truncated mid-execution): 6 (7.2%)
- Architecture breakdown: 26 Blackhole, 57 Wormhole
- Cohort B sampled (last run of multi-run jobs): 16

### Why Triage Did Not Complete (Cohort A)

| file_key | test_function | Outcome | Last script run | Reason |
|----------|---------------|---------|-----------------|--------|
| 71769542833_run1 | test_forward_pass | ABORTED | check_broken_components.py | E14 — CI action timeout |
| 71823669970_run1 | test_demo_text | ABORTED | dump_risc_debug_signals.py | E24 — output truncated |
| 71916806681_run1 | test_sd35_pipeline | ABORTED | dump_risc_debug_signals.py | E14 — CI action timeout |
| 71933933606_run1 | test_forward_pass | ABORTED | dump_risc_debug_signals.py | E14 — CI action timeout |
| 71970422789_run1 | test_forward_pass | ABORTED | dump_risc_debug_signals.py | E14 — CI action timeout |
| 71593138151_run1 | test_reduce_scatter_async_training_shapes | ABORTED | dump_risc_debug_signals.py | E14 — CI action timeout |

### Script Reliability — Cohort A (First Runs on Fresh Device)

| Script | Total | PASS | EXPECTED | UNEXPECTED | Top Errors |
|--------|-------|------|----------|------------|------------|
| dump_configuration.py | 83 | 80 (96.4%) | 0 (0.0%) | 3 (3.6%) | E07 |
| check_arc.py | 83 | 60 (72.3%) | 0 (0.0%) | 23 (27.7%) | E05 |
| check_cb_inactive.py | 83 | 81 (97.6%) | 0 (0.0%) | 2 (2.4%) | — |
| check_eth_status.py | 83 | 81 (97.6%) | 0 (0.0%) | 2 (2.4%) | — |
| check_noc_locations.py | 83 | 81 (97.6%) | 0 (0.0%) | 2 (2.4%) | — |
| device_info.py | 83 | 58 (69.9%) | 0 (0.0%) | 25 (30.1%) | E05 |
| device_telemetry.py | 83 | 83 (100.0%) | 0 (0.0%) | 0 (0.0%) | — |
| dump_running_operations.py | 83 | 77 (92.8%) | 2 (2.4%) | 4 (4.8%) | — |
| check_binary_integrity.py | 83 | 60 (72.3%) | 21 (25.3%) | 2 (2.4%) | — |
| check_core_magic.py | 83 | 81 (97.6%) | 0 (0.0%) | 2 (2.4%) | — |
| check_noc_status.py | 83 | 10 (12.0%) | 35 (42.2%) | 38 (45.8%) | E06; E09; E08 |
| dump_aggregated_callstacks.py | 83 | 52 (62.7%) | 0 (0.0%) | 31 (37.3%) | E03; E12; E13 |
| dump_callstacks.py | 29 | 17 (58.6%) | 0 (0.0%) | 12 (41.4%) | — |
| dump_fast_dispatch.py | 83 | 78 (94.0%) | 0 (0.0%) | 5 (6.0%) | E09; E23; E24 |
| dump_lightweight_asserts.py | 83 | 63 (75.9%) | 0 (0.0%) | 20 (24.1%) | E03 |
| dump_watcher_ringbuffer.py | 83 | 80 (96.4%) | 0 (0.0%) | 3 (3.6%) | — |
| firmware_versions.py | 83 | 82 (98.8%) | 0 (0.0%) | 1 (1.2%) | — |
| system_info.py | 83 | 82 (98.8%) | 0 (0.0%) | 1 (1.2%) | — |
| check_broken_components.py | 83 | 49 (59.0%) | 31 (37.3%) | 3 (3.6%) | — |
| dump_risc_debug_signals.py | 80 | 73 (91.2%) | 0 (0.0%) | 7 (8.8%) | — |

**Scripts with >10% UNEXPECTED rate (Cohort A):**
- **check_arc.py**: 27.7% UNEXPECTED — E05
- **device_info.py**: 30.1% UNEXPECTED — E05
- **check_noc_status.py**: 45.8% UNEXPECTED — E06; E09; E08
- **dump_aggregated_callstacks.py**: 37.3% UNEXPECTED — E03; E12; E13
- **dump_callstacks.py**: 41.4% UNEXPECTED — —
- **dump_lightweight_asserts.py**: 24.1% UNEXPECTED — E03

### Cohort A vs B (degradation on contaminated device)

| Script | A UNEXPECTED% | B UNEXPECTED% | Δ pp |
|--------|--------------:|--------------:|-----:|
| dump_configuration.py | 3.6% | 0.0% | $\color{green}{-3.6}$ |
| check_arc.py | 27.7% | 12.5% | $\color{green}{-15.2}$ |
| check_cb_inactive.py | 2.4% | 31.2% | $\color{red}{+28.8}$ |
| check_eth_status.py | 2.4% | 25.0% | $\color{red}{+22.6}$ |
| check_noc_locations.py | 2.4% | 18.8% | $\color{red}{+16.4}$ |
| device_info.py | 30.1% | 12.5% | $\color{green}{-17.6}$ |
| device_telemetry.py | 0.0% | 0.0% | +0 |
| dump_running_operations.py | 4.8% | 31.2% | $\color{red}{+26.4}$ |
| check_binary_integrity.py | 2.4% | 31.2% | $\color{red}{+28.8}$ |
| check_core_magic.py | 2.4% | 18.8% | $\color{red}{+16.4}$ |
| check_noc_status.py | 45.8% | 31.2% | $\color{green}{-14.6}$ |
| dump_aggregated_callstacks.py | 37.3% | 50.0% | $\color{red}{+12.7}$ |
| dump_callstacks.py | 41.4% | 20.0% | $\color{green}{-21.4}$ |
| dump_fast_dispatch.py | 6.0% | 37.5% | $\color{red}{+31.5}$ |
| dump_lightweight_asserts.py | 24.1% | 56.2% | $\color{red}{+32.1}$ |
| dump_watcher_ringbuffer.py | 3.6% | 31.2% | $\color{red}{+27.6}$ |
| firmware_versions.py | 1.2% | 0.0% | $\color{green}{-1.2}$ |
| system_info.py | 1.2% | 0.0% | $\color{green}{-1.2}$ |
| check_broken_components.py | 3.6% | 12.5% | $\color{red}{+8.9}$ |
| dump_risc_debug_signals.py | 8.8% | 13.3% | $\color{red}{+4.5}$ |

### Error Patterns (Cohort A)

#### Triage Bugs (Actionable)

| Pattern | Jobs | % | Avg/Job |
|---------|-----:|--:|--------:|
| E03: Missing Fabric ERISC Router ELF | 29 | 34.9% | 475.4 |
| E07: ttexalens SyntaxWarning | 27 | 32.5% | 1.0 |
| E23: Triage Mid-Script Crash | 1 | 1.2% | 1.0 |

#### Environment Issues

| Pattern | Jobs | % |
|---------|-----:|--:|
| E06: Missing noc_mode DWARF Variable | 35 | 42.2% |
| E04: Cores Broken During Triage (known HW halt/resume limitation) | 25 | 30.1% |
| E05: Unsafe ARC Memory Access | 23 | 27.7% |
| E14: Test Action Timeout | 7 | 8.4% |
| E10: No Cores Available | 5 | 6.0% |
| E09: Failed to Halt Core | 3 | 3.6% |
| E08: Core Not Halted (skip) | 1 | 1.2% |
| E13: Core Is In Reset | 1 | 1.2% |
| E17: Unknown Motherboard Warning | 1 | 1.2% |
| E24: Triage Output Truncated | 1 | 1.2% |

#### Diagnostic Findings (Triage Working Correctly)

| Pattern | Jobs | % |
|---------|-----:|--:|
| E16: NOC Transaction Mismatch | 35 | 42.2% |
| E11: Binary Integrity Mismatch | 21 | 25.3% |
| E25: Likely di/dt — MatMul preceded hang | 2 | 2.4% |

> **E25 throttle breakdown**: of 2 MatMul-preceded hangs in Cohort A, 2 had `TT_MM_THROTTLE_PERF` unset, 0 had it set (level distribution: unset=2). Unset-and-hung suggests unmitigated di/dt risk; set-and-still-hung suggests the throttle level was insufficient.

#### Informational

| Pattern | Jobs | % |
|---------|-----:|--:|
| E12: PC Not in ELF Range | 50 | 60.2% |
| E18: N/A Kernel Name in Callstacks | 5 | 6.0% |

> **E12 per-risc breakdown (Cohort A)**: erisc=913, brisc=15. Erisc total: 913 (informational — PC context-switch to base ERISC firmware, expected on erisc). Tensix-core total: 15 (brisc/trisc/ncrisc — indicates tooling/ELF resolution gap on worker cores, worth investigating).

### Top Tests by UNEXPECTED Failure Count (Cohort A)

| Test | Jobs | Worst Scripts |
|------|-----:|---------------|
| test_resnet50_e2e_graph_capture | 16 | check_arc.py(16), device_info.py(16), check_noc_status.py(16) |
| test_demo_text | 6 | dump_aggregated_callstacks.py(6), dump_lightweight_asserts.py(5), check_arc.py(4) |
| test_resnet_50 | 24 | check_noc_status.py(14), dump_configuration.py(1) |
| test_forward_pass | 5 | dump_aggregated_callstacks.py(5), dump_lightweight_asserts.py(4), dump_risc_debug_signals.py(2) |
| test_matmul_2d_multiple_output_blocks_per_core | 1 | check_cb_inactive.py(1), check_eth_status.py(1), check_noc_locations.py(1) |
| TestTypecast | 1 | check_cb_inactive.py(1), check_eth_status.py(1), check_noc_locations.py(1) |
| test_reduce_scatter_async_training_shapes | 3 | check_arc.py(3), device_info.py(3), dump_aggregated_callstacks.py(2) |
| test_mochi_pipeline_performance | 5 | dump_aggregated_callstacks.py(5), dump_callstacks.py(2) |
| test_mode_decode_forward_pass_batch_8_users_per_row | 3 | dump_aggregated_callstacks.py(3), dump_callstacks.py(2), dump_lightweight_asserts.py(1) |
| test_tt_mochi_pipeline | 2 | dump_aggregated_callstacks.py(2), dump_lightweight_asserts.py(2), dump_callstacks.py(1) |

### New Errors Discovered (Cohort A)

#### Dispatch Sem Read Failure
- **Script**: `dump_fast_dispatch.py`
- **Jobs affected**: 1
- **Error text**: `Failed to read sem_minus_local for kernel cq_dispatch. There may be a problem with the dispatcher kernel.`
- **Suggested regex**: `Failed to read sem_minus_local for kernel`

#### Blackhole Postcode Not Readable
- **Script**: `device_info.py`
- **Jobs affected**: 1
- **Error text**: `Blackhole device Postcode column shows N/A (not a populated hex value, and not the usual unsafe-access error)`
- **Suggested regex**: `Postcode.*N/A`

#### dump_fast_dispatch None device AssertionError
- **Script**: `dump_fast_dispatch.py`
- **Jobs affected**: 1
- **Error text**: `Traceback (most recent call last): File "/__w/tt-metal/tt-metal/docker-job/tools/triage/dump_fast_dispatch.py", line 340, in run; location = OnChipCoordinate(x, y, "translated", device); File "/opt/venv/lib/python3.10/site-packages/ttexalens/coordinate.py", line 97, in __init__; assert device is not`
- **Suggested regex**: `OnChipCoordinate.*assert device is not None\s+AssertionError`

#### Lightweight Asserts stat() None Path
- **Script**: `dump_lightweight_asserts.py`
- **Jobs affected**: 1
- **Error text**: `Device 0: eth [6-6 (e0,14)]: erisc: Failed to dump lightweight asserts: stat: path should be string, bytes, os.PathLike or integer, not NoneType (repeated for all 15 erisc cores on device 0)`
- **Suggested regex**: `Failed to dump lightweight asserts: stat: path should be string, bytes, os\.PathLike or integer, not NoneType`

### Week-over-Week Trends (Cohort A)

#### Script UNEXPECTED% Trends

| Script | Last Week | This Week | Δ pp |
|--------|----------:|----------:|-----:|
| dump_configuration.py | 0.0% | 3.6% | $\color{red}{+3.6}$ |
| check_arc.py | 18.7% | 27.7% | $\color{red}{+9.0}$ |
| check_cb_inactive.py | 2.2% | 2.4% | $\color{red}{+0.2}$ |
| check_eth_status.py | 0.0% | 2.4% | $\color{red}{+2.4}$ |
| check_noc_locations.py | 2.2% | 2.4% | $\color{red}{+0.2}$ |
| device_info.py | 18.7% | 30.1% | $\color{red}{+11.4}$ |
| device_telemetry.py | 0.0% | 0.0% | +0 |
| dump_running_operations.py | 19.8% | 4.8% | $\color{green}{-15.0}$ |
| check_binary_integrity.py | 2.2% | 2.4% | $\color{red}{+0.2}$ |
| check_core_magic.py | 2.2% | 2.4% | $\color{red}{+0.2}$ |
| check_noc_status.py | 24.2% | 45.8% | $\color{red}{+21.6}$ |
| dump_aggregated_callstacks.py | 67.5% | 37.3% | $\color{green}{-30.2}$ |
| dump_callstacks.py | 67.5% | 41.4% | $\color{green}{-26.1}$ |
| dump_fast_dispatch.py | 17.3% | 6.0% | $\color{green}{-11.3}$ |
| dump_lightweight_asserts.py | 42.2% | 24.1% | $\color{green}{-18.1}$ |
| dump_watcher_ringbuffer.py | 2.2% | 3.6% | $\color{red}{+1.4}$ |
| firmware_versions.py | 0.0% | 1.2% | $\color{red}{+1.2}$ |
| system_info.py | 36.8% | 1.2% | $\color{green}{-35.6}$ |
| check_broken_components.py | 36.8% | 3.6% | $\color{green}{-33.2}$ |
| dump_risc_debug_signals.py | 18.9% | 8.8% | $\color{green}{-10.1}$ |

#### Error Pattern Trends (jobs%)

| Pattern | Last Week | This Week | Δ pp |
|---------|----------:|----------:|-----:|
| E01: FD Exhaustion (Errno 24) | 35.5% | 0.0% | $\color{green}{-35.5}$ |
| E02: FD Exhaustion Crashes system_info/Rich | 35.5% | 0.0% | $\color{green}{-35.5}$ |
| E03: Missing Fabric ERISC Router ELF | 29.6% | 34.9% | $\color{red}{+5.3}$ |
| E04: Cores Broken During Triage (known HW halt/resume limitation) | 37.6% | 30.1% | $\color{green}{-7.5}$ |
| E05: Unsafe ARC Memory Access | 18.1% | 27.7% | $\color{red}{+9.6}$ |
| E06: Missing noc_mode DWARF Variable | 15.3% | 42.2% | $\color{red}{+26.9}$ |
| E07: ttexalens SyntaxWarning | 39.4% | 32.5% | $\color{green}{-6.9}$ |
| E08: Core Not Halted (skip) | 5.9% | 1.2% | $\color{green}{-4.7}$ |
| E09: Failed to Halt Core | 2.1% | 3.6% | $\color{red}{+1.5}$ |
| E10: No Cores Available | 2.1% | 6.0% | $\color{red}{+3.9}$ |
| E11: Binary Integrity Mismatch | 8.0% | 25.3% | $\color{red}{+17.3}$ |
| E12: PC Not in ELF Range | 19.9% | 60.2% | $\color{red}{+40.3}$ |
| E13: Core Is In Reset | 0.7% | 1.2% | $\color{red}{+0.5}$ |
| E14: Test Action Timeout | 2.4% | 8.4% | $\color{red}{+6.0}$ |
| E15: No Triage Section | 3.1% | 0.0% | $\color{green}{-3.1}$ |
| E16: NOC Transaction Mismatch | 0.0% | 42.2% | $\color{red}{+42.2}$ |
| E17: Unknown Motherboard Warning | 28.9% | 1.2% | $\color{green}{-27.7}$ |
| E18: N/A Kernel Name in Callstacks | 0.0% | 6.0% | $\color{red}{+6.0}$ |
| E23: Triage Mid-Script Crash | 35.5% | 1.2% | $\color{green}{-34.3}$ |
| E24: Triage Output Truncated | 0.0% | 1.2% | $\color{red}{+1.2}$ |
| E25: Likely di/dt — MatMul preceded hang | 0.0% | 2.4% | $\color{red}{+2.4}$ |

### Recommendations (Priority Order)

1. **Cache fabric ERISC router ELFs** — ensure idle_erisc/subordinate_idle_erisc ELFs are written to cache — affects 29/83 (34.9%) of first runs.
2. **Fix ttexalens SyntaxWarning** — trivial escape-sequence fix — affects 27/83 (32.5%) of first runs.
3. **Add noc_mode to ERISC DWARF** — or infer NOC from other data — affects 35/83 (42.2%) of first runs.

### Appendix: Data Files
- [Script Reliability](triage_script_reliability_20260420.csv)
- [Error Patterns](triage_error_patterns_20260420.csv)
- [Per-Test Breakdown](triage_per_test_breakdown_20260420.csv)
- [New Errors](triage_new_errors_20260420.csv)
- [Script Drill-down](triage_script_drilldown_20260420.md)
