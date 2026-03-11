---
phase: 2
slug: silicon-data-transfer-validation
status: audited
nyquist_compliant: true
wave_0_complete: true
created: 2026-03-11
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for silicon data-transfer tests.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Google Test (C++ GTest) in fabric_unit_tests binary |
| **Config file** | `tests/tt_metal/tt_fabric/sources.cmake` |
| **Quick run command** | `./build_Release/test/tt_metal/tt_fabric/fabric_unit_tests --gtest_filter="*AutoPacketization*"` |
| **Full suite command** | same (all auto-packetization tests in one binary) |
| **Estimated runtime** | ~60-180 seconds (silicon fabric tests, 9 families x 3 sizes) |

---

## Sampling Rate

- **After every task commit:** Host compile check (`./build_metal.sh -e -c --build-tests`)
- **After every plan wave:** Run silicon tests on hardware
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~180 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Behavior | Test Type | Automated Command | Status |
|---------|------|------|----------|-----------|-------------------|--------|
| 2-01-01 | 01 | 0 | Rewrite unicast runner for BaseFabricFixture | silicon | `--gtest_filter="*AutoPacket*Unicast*"` | ✅ green |
| 2-01-02 | 01 | 0 | Rewrite multicast runner for BaseFabricFixture | silicon | `--gtest_filter="*AutoPacket*Multicast*"` | ✅ green |
| 2-02-01 | 02 | 1 | Scatter/fused/sparse kernel + runner code | silicon | `--gtest_filter="*AutoPacket*Scatter*:*FusedAtomic*:*Sparse*"` | ✅ green |
| 2-03-01 | 03 | 2 | All 9 families x 3 sizes pass on silicon | silicon | `--gtest_filter="*AutoPacketization*"` | ✅ green |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [x] Rewritten unicast_runner.cpp using BaseFabricFixture pattern
- [x] Rewritten multicast_runner.cpp using BaseFabricFixture pattern
- [x] TEST_F cases in test_auto_packetization.cpp for unicast + multicast (19 TEST_F total)
- [x] Runner .cpp files added to sources.cmake

---

## Manual-Only Verifications

All phase behaviors have automated silicon verification.

---

## Validation Sign-Off

- [x] All tasks have automated silicon verify
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] Feedback latency < 180s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** validated 2026-03-11

## Validation Audit 2026-03-11

| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 0 |
| Escalated | 0 |

19 TEST_F cases across Fabric2DFixture (10) and Fabric1DFixture (9). All 9 auto-packetizing families covered. No GTEST_SKIP barriers remain (sparse multicast enabled). Only legitimate skip: device count guard (< 3 devices).
