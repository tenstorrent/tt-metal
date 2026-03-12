---
gsd_state_version: 1.0
milestone: v1.2
milestone_name: API & Test Cleanup
current_plan: None yet
status: planning
stopped_at: "Completed 03-02-PLAN.md: linear/detail/api.h created, linear/api.h updated with detail:: calls"
last_updated: "2026-03-12T01:39:58.868Z"
last_activity: 2026-03-11 -- v1.2 roadmap created; ready to begin Phase 3
progress:
  total_phases: 4
  completed_phases: 0
  total_plans: 3
  completed_plans: 1
  percent: 18
---

# Project State

## Current Position
- **Phase:** 3 of 6 (api-detail-namespace) — first phase of v1.2
- **Current Plan:** None yet
- **Status:** Ready to plan
- **Last activity:** 2026-03-11 -- v1.2 roadmap created; ready to begin Phase 3

Progress: [██░░░░░░░░] 18% (phases 1-2 shipped; phases 3-6 pending)

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-11)

**Core value:** Transparent auto-packetization for fabric APIs -- callers send any size, chunking is invisible
**Current focus:** Phase 3 - api-detail-namespace (v1.2 API & Test Cleanup)

## Accumulated Context
- BaseFabricFixture + L1 direct I/O is the standard test pattern
- FABRIC_2D ifdef guards required in all device kernels
- GTEST_SKIP for hardware bugs is a safety mechanism (SparseMulticast issue #36581)
- Forward-declaring infrastructure in early plans eliminates cross-plan coupling
- Every phase MUST end with rebuild + retest before marking complete (user constraint)
- Detail headers must compile for both RISCV (device) and x86 (host) targets

## Decisions
- [v1.2]: detail namespace over removal -- power users need single-packet APIs for perf-critical paths
- [v1.2]: extract runner utilities before kernel consolidation -- lower-risk, establishes patterns
- See .planning/milestones/v1.1-ROADMAP.md for v1.1 decisions
- See .planning/milestones/v1.0-ROADMAP.md for v1.0 decisions
- [03-01]: MeshMcastRange must be defined before detail/api.h include; closed and reopened namespace to achieve correct ordering
- [03-01]: No using namespace detail in public header; all detail:: calls are explicit per user requirement
- [Phase 03-api-detail-namespace]: detail/api.h include placed after route helpers in linear/api.h to satisfy non-template name lookup requirements

## Blockers
- SparseMulticast (issue #36581): GTEST_SKIP'd, out of scope for v1.2

## Last Session
- **Timestamp:** 2026-03-12
- **Stopped At:** Completed 03-01-PLAN.md: mesh/detail/api.h created with 16 _single_packet definitions in detail:: namespace
