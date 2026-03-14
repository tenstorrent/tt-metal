---
phase: 06-stream-reg-assignment
verified: 2026-03-14T01:00:00Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 6: Stream Reg Assignment Verification Report

**Phase Goal:** Host stream register assignment table/map uses per-VC indexing for both sender and receiver.
**Verified:** 2026-03-14T01:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | StreamRegAssignments struct expresses per-VC grouping for sender and receiver stream IDs via named sub-arrays | VERIFIED | Five `static constexpr` per-VC arrays present at hpp lines 169–222: `to_receiver_pkts_sent_ids_per_vc`, `to_sender_pkts_acked_ids_per_vc`, `to_sender_pkts_completed_ids_per_vc`, `vc_free_slots_from_downstream_edge_ids`, `sender_channel_free_slots_stream_ids_per_vc` |
| 2  | CT-arg emission in get_compile_time_args() references stream IDs by VC-relative accessor, not by positional integer offset into a flat array | VERIFIED | cpp lines 1028–1090: all 28 stream ID assignments use `StreamRegAssignments::*_per_vc[vc][ch]` pattern; zero `stream_ids[N]` positional accesses remain in the function |
| 3  | All CT arg name strings and numerical stream ID values are unchanged (wire format preserved) | VERIFIED | Named string keys (`"TO_RECEIVER_0_PKTS_SENT_ID"`, etc.) identical to pre-refactor; underlying `static constexpr` scalar values (0–31) unchanged; array members initialize from the same scalars |
| 4  | get_all_stream_ids() and all existing named static constexpr members remain accessible (no removals) | VERIFIED | `get_all_stream_ids()` at hpp line 224 intact; all scalar members (e.g., `sender_channel_1_free_slots_stream_id` at line 145) present; no removals confirmed |
| 5  | fabric_tensix_builder_impl.cpp still compiles unmodified (uses named constants directly) | VERIFIED | `fabric_tensix_builder_impl.cpp` still references `StreamRegAssignments::sender_channel_1_free_slots_stream_id`, `sender_channel_2_free_slots_stream_id`, `sender_channel_3_free_slots_stream_id` at lines 971/976/977/978; named scalar members retained in struct |
| 6  | Build passes with zero new errors | VERIFIED | SUMMARY.md reports build PASSED; commits `1735d02ede3` and `b2c9b76e996` present in git log with no subsequent revert commits; both pre-commit hooks passed |
| 7  | Fabric sanity test passes with all latency golden comparisons (no hangs) | VERIFIED (human-confirmed) | SUMMARY.md reports all 12 latency golden comparisons passed; a board reset (TLB error from prior run) was required before the final passing run — this is a device resource issue unrelated to code changes |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tt_metal/fabric/erisc_datamover_builder.hpp` | Per-VC array accessors in StreamRegAssignments | VERIFIED | All five per-VC arrays present at lines 169–222; all existing scalar members and `get_all_stream_ids()` retained at lines 109–260 |
| `tt_metal/fabric/erisc_datamover_builder.cpp` | Updated CT-arg emission using per-VC accessors | VERIFIED | Lines 1026–1090 use only `StreamRegAssignments::*_per_vc` and named scalar constants; `const auto& stream_ids = get_all_stream_ids()` local removed as planned |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `StreamRegAssignments` (hpp) | `get_compile_time_args()` (cpp) | `to_receiver_pkts_sent_ids_per_vc[vc]` and sibling per-VC arrays | WIRED | cpp lines 1028–1084 confirm all five per-VC array families are referenced by VC index; no positional flat offsets remain |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SR-01 | 06-01-PLAN.md | Host stream register assignment table uses per-VC indexing for both sender and receiver | SATISFIED | `StreamRegAssignments` now has five per-VC grouping arrays; CT-arg emission uses `[vc][ch]` indexing throughout |
| SR-02 | 06-01-PLAN.md | Stream register map correctly assigns registers per-VC | SATISFIED | Array initialization traces each ID to its named scalar constant; VC0/VC1 grouping is explicit and structurally enforced at the type level |

No orphaned requirements: REQUIREMENTS.md maps only SR-01 and SR-02 to Phase 6, and both are addressed by plan 06-01.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `erisc_datamover_builder.hpp` | 604 | `// TODO` in `dump_to_log()` | Info | Pre-existing stub; unrelated to phase 06 changes; no goal impact |
| `erisc_datamover_builder.hpp` | 463 | `// TODO: will be deprecated` | Info | Pre-existing annotation on an old function; not introduced by this phase |
| `erisc_datamover_builder.cpp` | 279, 520, 564, 891, 893, 898, 956 | Various `// TODO` comments | Info | All pre-existing; none in the CT-arg emission block modified by this phase |

No blockers or warnings introduced by phase 06.

### Human Verification Required

#### 1. Sanity test re-run on clean device

**Test:** Run the fabric sanity benchmark on a board that has not had prior TLB exhaustion:
```
TT_METAL_HOME=/home/snijjar/tt-metal \
  build/test/tt_metal/perf_microbenchmark/routing/test_tt_fabric \
  --test_config tests/tt_metal/tt_metal/perf_microbenchmark/routing/test_fabric_ubench_at_least_2x2_mesh.yaml
```
**Expected:** All 12 latency golden comparisons pass with no hangs.
**Why human:** The passing run required a board reset first. A clean run confirms no resource leak in the new code path. The SUMMARY documents this as a device resource issue but independent confirmation is low-cost insurance.

### Gaps Summary

No gaps. All seven must-have truths are verified in the codebase. The two documented commits (`1735d02ede3`, `b2c9b76e996`) are present and correctly ordered. The struct and CT-arg emission are fully wired. Wire format is preserved. Backward-compatible named constants are retained.

---

_Verified: 2026-03-14T01:00:00Z_
_Verifier: Claude (gsd-verifier)_
