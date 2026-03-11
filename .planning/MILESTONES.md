# Milestones

## v1.1 Silicon Validation (Shipped: 2026-03-11)

**Phases completed:** 1 phase (Phase 2), 3 plans, 5 tasks
**Git range:** `f70c87e585..1478b59f81` (6 code commits)
**Files:** 14 modified (+2,184 / -390 lines)

**Key accomplishments:**
- Rewrote unicast and multicast test runners for BaseFabricFixture with L1 direct I/O
- Created 8 new device kernels covering all auto-packetizing wrapper variants
- Added 17 silicon TEST_F cases across Fabric2DFixture and Fabric1DFixture
- Silicon-validated 8/9 families byte-for-byte correct on 4-chip TT hardware
- Confirmed SparseMulticast firmware limitation (issue #36581) -- test infrastructure ready

### Known Gaps

- **SparseMulticast** (issue #36581): Silicon test GTEST_SKIP'd due to firmware limitation causing Ethernet core lockup. Kernel + test infrastructure complete; blocked on firmware fix. Fix approach: decompose sparse_mask into per-device unicast atomic_incs.

---

## v1.0 Fabric Auto-Packetization (Shipped: 2026-03-11)

See: `.planning/milestones/v1.0-MILESTONES.md`

---
