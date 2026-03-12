---
phase: 04
slug: test-infrastructure-cleanup
status: resolved
nyquist_compliant: true
wave_0_complete: false
created: 2026-03-12
---

# Phase 04 — Validation Strategy

> Per-phase validation contract: structural requirements have automated verification; behavioral hardware validation is manual-only by design.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | GTest (fabric_unit_tests binary) |
| **Config file** | `tests/tt_metal/tt_fabric/CMakeLists.txt` |
| **Quick run command** | `cd build_Release && ninja fabric_unit_tests` |
| **Full suite command** | `./build_Release/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*AutoPacketization*"` |
| **Estimated runtime** | Build: ~30s · Silicon run: ~5–15 min (hardware required) |

---

## Sampling Rate

- **After every task commit:** Run `cd build_Release && ninja fabric_unit_tests`
- **After every plan wave:** Full suite on hardware (gate for 04-03)
- **Before `/gsd:verify-work`:** Full suite must be green + hardware confirmed
- **Max feedback latency:** ~30s (build only) / ~15min (with hardware)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | Status |
|---------|------|------|-------------|-----------|-------------------|--------|
| 04-01-T1 | 01 | 1 | TEST-01 | structural | `grep -l "TX_KERNEL_PARSE_UNICAST_ARGS" tests/.../tx_kernel_common.h` | ✅ green |
| 04-01-T2 | 01 | 1 | TEST-01 | structural | `grep -c "tx_kernel_common.h" tests/.../kernels/*.cpp` (expect 9) | ✅ green |
| 04-02-T1 | 02 | 1 | TEST-03 | structural | `grep -c "make_tx_pattern" tests/.../test_common.hpp` (expect ≥1) | ✅ green |
| 04-02-T2 | 02 | 1 | TEST-02 | structural | `grep -c "run_silicon_family_test" tests/.../test_auto_packetization.cpp` (expect 17) | ✅ green |
| 04-03-T1 | 03 | 2 | TEST-01/02/03 | build | `cd build_Release && ninja fabric_unit_tests` (exit 0) | ✅ green |
| 04-03-T2 | 03 | 2 | TEST-01/02/03 | hardware | Silicon test run (manual — see Manual-Only below) | ✅ green (18 PASSED + 1 SKIPPED) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

Existing infrastructure covers all phase requirements. Phase 04 is a refactoring phase — the test suite being refactored IS the verification mechanism. No new test files needed.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| All 18 silicon AutoPacketization tests pass after kernel/host refactoring | TEST-01, TEST-02, TEST-03 | Requires physical TT hardware. Build and structural checks pass; hardware run confirms no behavioral regression. | `cd /home/snijjar/tt-metal-4 && ./build_Release/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*AutoPacketization*" 2>&1 \| tail -30` — expect 18 PASSED + 1 SKIPPED (SparseMulticast, issue #36581) + 0 FAILED |
| `CompileOnlyAutoPacketization2D` + `CompileOnlyAutoPacketization1D` pass | TEST-01 | Requires TT device for CreateKernel/CompileProgram JIT path | Covered by the silicon test run above — both tests are included in the `*AutoPacketization*` filter |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or are hardware-gated (documented above)
- [x] Sampling continuity: build check (ninja) applies after every task
- [x] Wave 0: existing infrastructure sufficient — no new stubs needed
- [x] Hardware silicon run confirmed: 18 PASSED + 1 SKIPPED (2026-03-12)
- [x] All manual-only items verified on hardware

**Approval:** approved 2026-03-12
