# tt-triage Weekly Analysis Report
## Week of 2026-04-07 to 2026-04-14

### Executive Summary
- Total hang jobs: 287
- Jobs with triage output: 278
- Init failures (no triage output): 9 (3.1%)
- Architecture breakdown: 194 Blackhole, 24 Wormhole

### Script Reliability — Cohort A (First Runs on Fresh Device)

| Script | Total | PASS | EXPECTED | UNEXPECTED | ERRORED | ABSENT |
|--------|-------|------|----------|------------|---------|--------|
| dump_configuration.py | 278 | 278 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 9 |
| check_arc.py | 278 | 226 (81.3%) | 0 (0.0%) | 52 (18.7%) | 0 (0.0%) | 9 |
| check_cb_inactive.py | 278 | 272 (97.8%) | 0 (0.0%) | 6 (2.2%) | 0 (0.0%) | 9 |
| check_eth_status.py | 278 | 277 (99.6%) | 1 (0.4%) | 0 (0.0%) | 0 (0.0%) | 9 |
| check_noc_locations.py | 278 | 272 (97.8%) | 0 (0.0%) | 6 (2.2%) | 0 (0.0%) | 9 |
| device_info.py | 278 | 226 (81.3%) | 0 (0.0%) | 52 (18.7%) | 0 (0.0%) | 9 |
| device_telemetry.py | 278 | 278 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 9 |
| dump_running_operations.py | 278 | 223 (80.2%) | 0 (0.0%) | 55 (19.8%) | 0 (0.0%) | 9 |
| check_binary_integrity.py | 278 | 249 (89.6%) | 23 (8.3%) | 6 (2.2%) | 0 (0.0%) | 9 |
| check_core_magic.py | 278 | 272 (97.8%) | 0 (0.0%) | 6 (2.2%) | 0 (0.0%) | 9 |
| check_noc_status.py | 277 | 15 (5.4%) | 195 (70.4%) | 67 (24.2%) | 0 (0.0%) | 10 |
| dump_aggregated_callstacks.py | 277 | 90 (32.5%) | 0 (0.0%) | 187 (67.5%) | 0 (0.0%) | 10 |
| dump_callstacks.py | 277 | 90 (32.5%) | 0 (0.0%) | 187 (67.5%) | 0 (0.0%) | 10 |
| dump_fast_dispatch.py | 277 | 228 (82.3%) | 0 (0.0%) | 48 (17.3%) | 1 (0.4%) | 10 |
| dump_lightweight_asserts.py | 277 | 160 (57.8%) | 0 (0.0%) | 117 (42.2%) | 0 (0.0%) | 10 |
| dump_watcher_ringbuffer.py | 277 | 271 (97.8%) | 0 (0.0%) | 6 (2.2%) | 0 (0.0%) | 10 |
| firmware_versions.py | 277 | 277 (100.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 10 |
| system_info.py | 277 | 175 (63.2%) | 0 (0.0%) | 0 (0.0%) | 102 (36.8%) | 10 |
| check_broken_components.py | 277 | 67 (24.2%) | 108 (39.0%) | 0 (0.0%) | 102 (36.8%) | 10 |
| dump_risc_debug_signals.py | 169 | 137 (81.1%) | 0 (0.0%) | 0 (0.0%) | 32 (18.9%) | 118 |

**Scripts with >10% unexpected+errored rate:**
- **check_arc.py**: 18.7% unexpected+errored
- **device_info.py**: 18.7% unexpected+errored
- **dump_running_operations.py**: 19.8% unexpected+errored
- **check_noc_status.py**: 24.2% unexpected+errored
- **dump_aggregated_callstacks.py**: 67.5% unexpected+errored
- **dump_callstacks.py**: 67.5% unexpected+errored
- **dump_fast_dispatch.py**: 17.7% unexpected+errored
- **dump_lightweight_asserts.py**: 42.2% unexpected+errored
- **system_info.py**: 36.8% unexpected+errored
- **check_broken_components.py**: 36.8% unexpected+errored
- **dump_risc_debug_signals.py**: 18.9% unexpected+errored

---

### Error Patterns (Cohort A)

#### Triage Bugs (Actionable)

| Pattern | Jobs Affected | % | Avg/Job |
|---------|--------------|---|---------|
| E07: ttexalens SyntaxWarning | 113 | 39.4% | 1.0 |
| E04: Cores Broken During Triage | 108 | 37.6% | 51.2 |
| E01: FD Exhaustion (Errno 24) | 102 | 35.5% | 111.0 |
| E02: FD Exhaustion Crashes system_info/Rich | 102 | 35.5% | 2.0 |
| E23: Rich Library Crash (FD cascade) | 102 | 35.5% | 1.0 |
| E03: Missing Fabric ERISC Router ELF | 85 | 29.6% | 190.7 |

#### Environment Issues

| Pattern | Jobs Affected | % |
|---------|--------------|---|
| E17: Unknown Motherboard Warning | 83 | 28.9% |
| E05: Unsafe ARC Memory Access | 52 | 18.1% |
| E06: Missing noc_mode DWARF Variable | 44 | 15.3% |
| E08: Core Not Halted (skip) | 17 | 5.9% |
| E15: No Triage Section | 9 | 3.1% |
| E14: Test Action Timeout | 7 | 2.4% |
| E09: Failed to Halt Core | 6 | 2.1% |
| E10: No Cores Available | 6 | 2.1% |
| E13: Core Is In Reset | 2 | 0.7% |

#### Diagnostic Findings (Triage Working Correctly)

| Pattern | Jobs Affected | % |
|---------|--------------|---|
| E11: Binary Integrity Mismatch | 23 | 8.0% |

---

### Top Tests by Unexpected Failures

| Test | Jobs | Worst Scripts |
|------|------|--------------|
| test_host_io_loopback | 50 | dump_running_operations.py(50), dump_aggregated_callstacks.py(50), dump_callstacks.py(50) |
| test_resnet50_e2e_graph_capture | 41 | check_arc.py(41), device_info.py(41), check_noc_status.py(41) |
| test_rs_row_nightly_ring | 29 | dump_aggregated_callstacks.py(28), dump_callstacks.py(28), dump_fast_dispatch.py(28) |
| test_dram_streaming_matmul_with_all_experts | 14 | dump_aggregated_callstacks.py(14), dump_callstacks.py(14), dump_lightweight_asserts.py(14) |
| test_dram_streaming_matmul | 13 | dump_aggregated_callstacks.py(13), dump_callstacks.py(13), dump_lightweight_asserts.py(13) |
| test_all_gather_subcore_grid | 16 | dump_aggregated_callstacks.py(16), dump_callstacks.py(16), check_noc_status.py(13) |
| test_demo_text | 17 | dump_aggregated_callstacks.py(10), dump_callstacks.py(10), check_arc.py(6) |
| test_eltwise_add_compressed_1tile_bfp4 | 9 | dump_aggregated_callstacks.py(9), dump_callstacks.py(9), dump_lightweight_asserts.py(9) |
| test_eltwise_add_compressed_1tile_bfp2 | 8 | dump_aggregated_callstacks.py(8), dump_callstacks.py(8), dump_lightweight_asserts.py(8) |
| test_eltwise_add_compressed_1tile_bfp8 | 7 | dump_aggregated_callstacks.py(7), dump_callstacks.py(7), dump_lightweight_asserts.py(7) |

---

### Recommendations (Priority Order)
1. **Fix FD leak in callstack_provider** — Close ELF handles after each core. Affects 102/287 (36%) of first runs.
2. **Cache fabric ERISC router ELFs** — Ensure idle_erisc/subordinate_idle_erisc ELFs are written to cache. Affects 85/287 (30%) of first runs.
3. **Investigate halt/resume reliability** — Cores broken during triage in 108/287 (38%) of first runs.
4. **Fix ttexalens SyntaxWarning** — Trivial escape sequence fix. 113/287 jobs.
5. **Add noc_mode to ERISC DWARF** — Or infer NOC from other data. 44/287 WH jobs.

### Appendix: Data Files
- triage_script_reliability_20260415.csv
- triage_error_patterns_20260415.csv
- triage_per_test_breakdown_20260415.csv
