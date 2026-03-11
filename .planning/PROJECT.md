# Fabric Auto-Packetization

## What This Is

Auto-packetizing wrappers for Tenstorrent fabric APIs that transparently chunk payloads larger than FABRIC_MAX_PACKET_SIZE. Covers unicast, multicast, scatter, fused atomic_inc, and sparse multicast families across both 2D mesh and 1D linear topologies. Used by device kernel authors who need to send arbitrary-sized payloads over the fabric without manual chunking.

## Core Value

Transparent auto-packetization for fabric APIs -- callers send any size, chunking is invisible.

## Current Milestone: v1.2 API & Test Cleanup

**Goal:** Reduce code bloat in API headers and test suite by moving internal APIs to detail namespace and consolidating duplicated test infrastructure.

**Target features:**
- Move `_single_packet` APIs to `detail` header/namespace
- Extract shared test runner utilities into common infrastructure
- Consolidate device kernel boilerplate via shared header and templatization
- Reduce TEST_F repetition via parameterization

## Requirements

### Validated

- ✓ Auto-packetizing wrappers for all 9 fabric API families -- v1.0
- ✓ Silicon validation for 8/9 families (byte-for-byte correct on 4-chip hardware) -- v1.1
- ✓ Compile-only tests for 2D and 1D topologies -- v1.0
- ✓ BaseFabricFixture + L1 direct I/O test pattern -- v1.1

### Active

- [ ] `_single_packet` APIs in `detail` namespace/header
- [ ] Shared test runner utilities extracted
- [ ] Device kernel boilerplate consolidated
- [ ] TEST_F cases de-duplicated

### Out of Scope

- SparseMulticast silicon fix -- blocked on firmware (issue #36581)
- New API families or features -- this milestone is cleanup only
- AddrGen test infrastructure -- separate concern

## Context

- API headers: `tt_metal/fabric/hw/inc/mesh/api.h` (5599 lines), `linear/api.h` (4205 lines)
- Test suite: `tests/tt_metal/tt_fabric/fabric_data_movement/auto_packetization/`
- 9 device kernels, 2 runner files, 1 test file with 15 near-identical TEST_F cases
- `_single_packet` functions are implementation details; power users may still want access via `detail` namespace

## Constraints

- **Backward compatibility**: Existing kernel code that calls `_single_packet` APIs must still compile (via detail include)
- **No behavior changes**: All existing tests must pass identically after refactor
- **Device toolchain**: Headers compile for both RISCV (device) and x86 (host) targets

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| detail namespace over removal | Power users need single-packet APIs for perf-critical paths | — Pending |
| Extract utilities before kernel consolidation | Runner utilities are lower-risk, establishes patterns for kernel work | — Pending |

---
*Last updated: 2026-03-11 after milestone v1.2 initialization*
