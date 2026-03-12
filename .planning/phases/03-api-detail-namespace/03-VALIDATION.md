---
phase: 3
slug: api-detail-namespace
status: partial
nyquist_compliant: false
wave_0_complete: true
created: 2026-03-12
updated: 2026-03-12
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | GoogleTest (gtest) — integrated into tt-metal CMake build system |
| **Config file** | None — linked via CMake |
| **Quick run command** | `./build/test/tt_metal/fabric/fabric_tests --gtest_filter=*CompileOnly*` |
| **Full suite command** | `./build/test/tt_metal/fabric/fabric_tests --gtest_filter=*AutoPacketization*` |
| **Estimated runtime** | ~30 seconds (compile-only); ~5-10 minutes (full silicon suite) |

---

## Sampling Rate

- **After every task commit:** Run `./build/test/tt_metal/fabric/fabric_tests --gtest_filter=*CompileOnly*`
- **After every plan wave:** Run `./build/test/tt_metal/fabric/fabric_tests --gtest_filter=*AutoPacketization*`
- **Before `/gsd:verify-work`:** Full suite must be green (18 PASSED, 1 SKIPPED for SparseMulticast #36581)
- **Max feedback latency:** ~30 seconds (compile-only)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 1 | API-04 | compile | `--gtest_filter=*CompileOnly*` | ✅ | ✅ green (1 PASSED 1532ms) |
| 3-01-02 | 01 | 1 | API-04 | structural | file existence check | ✅ | ✅ green (8 families, 16 defs, all detail::) |
| 3-02-01 | 02 | 1 | API-04 | compile | `--gtest_filter=*CompileOnly*` | ✅ | ✅ green (1 PASSED 1572ms) |
| 3-02-02 | 02 | 1 | API-04 | structural | file existence check | ✅ | ✅ green (9 families, 18 defs, all detail::) |
| 3-03-01 | 03 | 2 | API-04 | silicon | `--gtest_filter=*AutoPacketization*` | ✅ | ⬜ pending (awaiting silicon checkpoint) |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

*Existing infrastructure covers all phase requirements.* The compile-only tests exercise all `detail::` function instantiation paths; the silicon tests exercise the full auto-packetizing wrappers that call through `detail::`. No new test stubs needed.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions | Status |
|----------|-------------|------------|-------------------|--------|
| No downstream callers outside auto_packetization break | API-04 | Grep-based audit — not easily automated | `grep -r "_single_packet" tt_metal/ --include="*.cpp" --include="*.hpp" --include="*.h" \| grep -v "detail/"` to find direct callers | ✅ Completed 2026-03-12 — 0 bare callers found (03-03 Task 1) |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter — pending 3-03-01 silicon gate

**Approval:** partial — 4/5 tasks green; awaiting silicon test checkpoint (3-03-01)

## Validation Audit 2026-03-12
| Metric | Count |
|--------|-------|
| Gaps found | 0 |
| Resolved | 4 |
| Escalated (manual-only) | 1 (3-03-01 — hardware gated) |
