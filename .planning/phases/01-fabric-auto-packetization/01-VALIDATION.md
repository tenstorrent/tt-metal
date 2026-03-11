---
phase: 1
slug: fabric-auto-packetization
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | TT-Metal kernel test harness (C++ device kernels + host runner) |
| **Config file** | none — Wave 0 creates new test directory |
| **Quick run command** | `./build_metal.sh -e -c --build-tests` |
| **Full suite command** | run `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_main` on hardware |
| **Estimated runtime** | ~60-120 seconds (hardware fabric tests) |

---

## Sampling Rate

- **After every task commit:** Compile-only check (headers are device-side; host compile validates syntax/types)
- **After every plan wave:** Run full suite on hardware
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~120 seconds (hardware run) / ~15 seconds (compile-only)

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Behavior | Test Type | Automated Command | Status |
|---------|------|------|----------|-----------|-------------------|--------|
| 1-01-01 | 01 | 0 | Wave 0: test stubs + sender kernels | build | compile linear/mesh headers | ⬜ pending |
| 1-02-01 | 02 | 1 | linear/api.h raw-size renames + wrappers | compile | `./build_metal.sh -e -c --build-tests` | ⬜ pending |
| 1-03-01 | 03 | 1 | mesh/api.h raw-size renames + wrappers | compile | same | ⬜ pending |
| 1-04-01 | 04 | 2 | mesh addrgen gap: multicast_fused_scatter_write_atomic_inc | compile | same | ⬜ pending |
| 1-05-01 | 05 | 3 | integration: unicast raw-size > MAX packetizes correctly | integration | run `test_main --test unicast_raw_chunking` on hardware | ⬜ pending |
| 1-06-01 | 06 | 3 | integration: multicast + breadth-first + fused final-chunk | integration | run `test_main --test multicast_raw_chunking` on hardware | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/unicast_tx_writer_raw.cpp` — raw-size write sender kernel (multi-chunk payload: 2*MAX + 512 bytes)
- [ ] `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/multicast_tx_writer_raw.cpp` — multicast raw-size variant
- [ ] `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_main.cpp` — host runner (pattern: addrgen_write runner)
- [ ] `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_common.hpp` — shared payload size constants, test enum

*Existing infrastructure in `tests/tt_metal/tt_fabric/fabric_data_movement/addrgen_write/` provides sender/receiver pattern to follow.*

---

## Manual-Only Verifications

| Behavior | Why Manual | Test Instructions |
|----------|------------|-------------------|
| Breadth-first ordering correctness (multi-connection) | Requires two-receiver setup with ordering probes | Set up two receivers, send 3-chunk payload via conn mgr, verify R0 gets chunk 0 before R1 gets chunk 1 |
| `atomic_inc` fires exactly once on fused ops | Semaphore count = 1 after multi-chunk fused send | Inspect semaphore value after full fused send |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
