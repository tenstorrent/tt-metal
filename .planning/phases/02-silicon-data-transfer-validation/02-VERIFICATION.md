---
phase: 02-silicon-data-transfer-validation
verified: 2026-03-11T18:00:00Z
status: passed
score: 9/9 must-haves verified
re_verification: false
---

# Phase 2: Silicon Data Transfer Validation Verification Report

**Phase Goal:** Run silicon data-transfer tests for all 9 auto-packetizing wrapper families. Rewrite runners to use BaseFabricFixture, follow addrgen_write test pattern, verify byte-for-byte data correctness with multiple payload sizes.
**Verified:** 2026-03-11T18:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                          | Status     | Evidence                                                                                                                  |
|----|-----------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------------------------------------------|
| 1  | Unicast runners use BaseFabricFixture, not MeshDeviceFixtureBase                             | VERIFIED   | `unicast_runner.cpp` line 24: `using fabric_router_tests::BaseFabricFixture;` — no MeshDeviceFixtureBase references       |
| 2  | Multicast runner uses BaseFabricFixture, not MeshDeviceFixtureBase                           | VERIFIED   | `multicast_runner.cpp` line 27: `using fabric_router_tests::BaseFabricFixture;` — no MeshDeviceFixtureBase references     |
| 3  | test_common.hpp defines all 9 families in AutoPacketFamily enum                              | VERIFIED   | All 9 families present: UnicastWrite, UnicastScatter, UnicastFusedAtomicInc, UnicastFusedScatterAtomicInc, MulticastWrite, MulticastScatter, MulticastFusedAtomicInc, MulticastFusedScatterAtomicInc, SparseMulticast |
| 4  | Runners perform L1 direct I/O (WriteToDeviceL1/ReadFromDeviceL1) not DRAM                   | VERIFIED   | `unicast_runner.cpp`: 3 WriteToDeviceL1, 3 ReadFromDeviceL1 calls. `multicast_runner.cpp`: 3 WriteToDeviceL1, 3 ReadFromDeviceL1 calls |
| 5  | Byte-for-byte data comparison is real (ADD_FAILURE on mismatch)                              | VERIFIED   | `unicast_runner.cpp` line 46: `ADD_FAILURE() << "Data mismatch at word " << i`. `multicast_runner.cpp` line 49: same pattern |
| 6  | 17 silicon TEST_F cases registered across Fabric2DFixture and Fabric1DFixture               | VERIFIED   | 17 TEST_F cases confirmed: 8 Fabric2DFixture + 9 Fabric1DFixture (8 + 1 sparse)                                         |
| 7  | Required Fabric1DFixture tests (LinearUnicastWrite, LinearMulticastWrite) are not skipped   | VERIFIED   | Both tests confirmed to call runner functions directly with no GTEST_SKIP                                                 |
| 8  | SparseMulticast has GTEST_SKIP with issue #36581 explanation (user-approved deferral)        | VERIFIED   | Line 455: `GTEST_SKIP() << "Sparse multicast silicon test deferred -- see issue #36581 (firmware limitation)"`            |
| 9  | Both runner files are in sources.cmake                                                        | VERIFIED   | sources.cmake lines 34-35: `unicast_runner.cpp` and `multicast_runner.cpp` both present                                   |

**Score:** 9/9 truths verified

---

### Required Artifacts

| Artifact                                                                                                    | Expected                                      | Status    | Details                                                                                  |
|-------------------------------------------------------------------------------------------------------------|-----------------------------------------------|-----------|------------------------------------------------------------------------------------------|
| `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/test_common.hpp`                        | AutoPacketFamily enum, RawTestParams, helpers | VERIFIED  | All 9 families in enum; family_kernel_path(), family_is_scatter(), family_is_fused()     |
| `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/unicast_runner.cpp`                     | BaseFabricFixture runner                      | VERIFIED  | run_raw_unicast_write_test(BaseFabricFixture*); L1 I/O; byte comparison                  |
| `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/multicast_runner.cpp`                   | BaseFabricFixture runner                      | VERIFIED  | run_raw_multicast_write_test(BaseFabricFixture*); L1 I/O; byte comparison per receiver   |
| `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/scatter_unicast_tx_writer_raw.cpp` | Scatter unicast kernel                     | VERIFIED  | Calls fabric_unicast_noc_scatter_write (line 78, 87)                                     |
| `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/fused_atomic_inc_unicast_tx_writer_raw.cpp` | Fused unicast+atomic_inc kernel | VERIFIED  | Calls fabric_unicast_noc_fused_unicast_with_atomic_inc (lines 71, 80)                   |
| `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/fused_scatter_atomic_inc_unicast_tx_writer_raw.cpp` | Fused scatter+atomic_inc unicast | VERIFIED | Calls fabric_unicast_noc_fused_scatter_write_atomic_inc (lines 78, 92)            |
| `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/scatter_multicast_tx_writer_raw.cpp` | Scatter multicast kernel              | VERIFIED  | Calls fabric_multicast_noc_scatter_write (5 occurrences)                                 |
| `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/fused_atomic_inc_multicast_tx_writer_raw.cpp` | Fused multicast+atomic_inc       | VERIFIED  | Calls fabric_multicast_noc_fused_unicast_with_atomic_inc (6 occurrences)                |
| `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/fused_scatter_atomic_inc_multicast_tx_writer_raw.cpp` | Fused scatter+atomic_inc multicast | VERIFIED | Calls fabric_multicast_noc_fused_scatter_write_atomic_inc                          |
| `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/kernels/sparse_multicast_tx_writer_raw.cpp` | Sparse multicast (linear-only)       | VERIFIED  | Calls fabric_sparse_multicast_noc_unicast_write (line 54)                                |
| `tests/tt_metal/tt_fabric/fabric_data_movement/test_auto_packetization.cpp`                               | 17 silicon TEST_F cases                       | VERIFIED  | 17 TEST_F cases confirmed; run_raw_unicast/multicast_write_test called 17 times total    |
| `tests/tt_metal/tt_fabric/sources.cmake`                                                                   | Both runner files in build                    | VERIFIED  | Lines 34-35 list both unicast_runner.cpp and multicast_runner.cpp                        |

---

### Key Link Verification

| From                             | To                                       | Via                                                   | Status  | Details                                                         |
|----------------------------------|------------------------------------------|-------------------------------------------------------|---------|-----------------------------------------------------------------|
| `unicast_runner.cpp`             | `fabric_fixture.hpp`                     | BaseFabricFixture* fixture param, generate_worker_mem_map | WIRED   | Lines 24, 58, 86-87 confirm usage                               |
| `multicast_runner.cpp`           | `fabric_fixture.hpp`                     | BaseFabricFixture* fixture param, get_device(), RunProgramNonblocking | WIRED | Lines 27, 61, 114, 118 confirm usage               |
| `test_auto_packetization.cpp`    | `unicast_runner.cpp`                     | TEST_F calls run_raw_unicast_write_test               | WIRED   | 8 unicast TEST_F cases call runner (lines 201, 228, 254, 277, 415, 642, 668, 691) |
| `test_auto_packetization.cpp`    | `multicast_runner.cpp`                   | TEST_F calls run_raw_multicast_write_test             | WIRED   | 9 multicast TEST_F cases call runner (lines 309, 332, 358, 381, 443, 714, 740, 763) |
| `unicast_runner.cpp`             | kernels/*.cpp (unicast families)         | CreateKernel with family_kernel_path() selection      | WIRED   | family_kernel_path() maps all 9 families to correct kernel paths |
| `multicast_runner.cpp`           | kernels/*.cpp (multicast families)       | CreateKernel with family_kernel_path() selection      | WIRED   | Same family_kernel_path() used in multicast runner              |

---

### Requirements Coverage

No REQUIREMENTS.md traceability IDs were declared for this phase (all plan frontmatter has `requirements: []`). No orphaned requirements to check — this phase operates outside the requirements traceability system.

---

### Anti-Patterns Found

No anti-patterns detected:
- No TODO/FIXME/HACK/PLACEHOLDER comments in runner files, test_common.hpp, or test file
- No empty implementations (all runner functions perform real L1 I/O and data comparison)
- No stub TEST_F bodies (all 17 tests call real runner functions)
- GTEST_SKIP in SparseMulticast is a documented, user-approved deferral for a confirmed firmware limitation (issue #36581), not a placeholder

---

### Human Verification Required

None. The silicon test results were already verified by the user (Task 2 of Plan 03 was a human-verify gate, explicitly approved).

The following silicon results are documented and approved:
- 16/17 tests PASSED on 4-chip TT hardware
- 1 SKIPPED (SparseMulticast — issue #36581 firmware limitation, user-approved as acceptable phase gate)
- All 8 active families deliver byte-for-byte correct data on Fabric2DFixture and Fabric1DFixture
- SparseMulticast confirmed to cause Ethernet core lockup when GTEST_SKIP is removed; GTEST_SKIP guard is essential

---

### Commit Verification

All 6 claimed commits verified in git log:

| Commit      | Description                                                              |
|-------------|--------------------------------------------------------------------------|
| `abf6ecea5b` | feat(02-01): create test_common.hpp and rewrite unicast_runner.cpp       |
| `4c9eb5f1c8` | feat(02-01): add 3 unicast device kernels and 4 silicon TEST_F cases    |
| `57543cbc0d` | feat(02-02): rewrite multicast_runner.cpp for BaseFabricFixture + 5 kernels |
| `9fbedc4a94` | feat(02-02): add multicast/sparse TEST_F cases + Fabric1DFixture + sources.cmake |
| `88c53d0b7f` | fix(02-03): silicon test fixes -- L1 overflow, 1D/2D kernel routing, device adjacency |
| `bdae1b837e` | fix: add FABRIC_2D ifdef guards to all auto-packetization kernels and implement 1D test cases |

---

### Summary

Phase goal achieved. All 9 auto-packetizing wrapper families have silicon test coverage:

- **8 families fully validated on silicon** (UnicastWrite, UnicastScatter, UnicastFusedAtomicInc, UnicastFusedScatterAtomicInc, MulticastWrite, MulticastScatter, MulticastFusedAtomicInc, MulticastFusedScatterAtomicInc) — byte-for-byte correct data at multiple payload sizes
- **SparseMulticast deferred** with GTEST_SKIP and documented issue #36581 — user-approved as acceptable phase gate
- Infrastructure is complete: BaseFabricFixture runners with L1 direct I/O, 10 device kernels, 17 TEST_F cases, both Fabric2DFixture and Fabric1DFixture coverage, sources.cmake wired

---

_Verified: 2026-03-11T18:00:00Z_
_Verifier: Claude (gsd-verifier)_
