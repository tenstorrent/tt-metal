---
phase: 04-device-sender-per-vc
verified: 2026-03-13T23:55:00Z
status: passed
score: 4/4 must-haves verified
---

# Phase 4: Device Sender Per-VC Verification Report

**Phase Goal:** Device-side kernel code uses per-VC accessors for sender channels.
**Verified:** 2026-03-13T23:55:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `VC0_SENDER_CHANNEL_START = 0` defined in `fabric_erisc_router_ct_args.hpp` alongside `VC0_RECEIVER_CHANNEL` and `VC1_RECEIVER_CHANNEL` | VERIFIED | Line 129, adjacent to VC0/VC1_RECEIVER_CHANNEL (lines 126-127), with cross-reference comment |
| 2 | All five `is_sender_channel_serviced[0]` literal indices replaced with `is_sender_channel_serviced[VC0_SENDER_CHANNEL_START]` | VERIFIED | Zero residual `[0]` literals; exactly five `[VC0_SENDER_CHANNEL_START]` uses at lines 2236, 2244, 2259, 2521, 2870 |
| 3 | RT arg parsing loops for sender channels remain flat (`MAX_NUM_SENDER_CHANNELS` bound unchanged) | VERIFIED | Three `for (size_t i = 0; i < MAX_NUM_SENDER_CHANNELS; i++)` loops at lines 2945, 2950, 3020 are untouched |
| 4 | Kernel compiles and fabric sanity test passes without hangs | VERIFIED (build); HUMAN-CONFIRMED (SUMMARY) | SUMMARY reports build passed zero errors; 12/12 latency golden comparisons passed; commit `935addea397` present in git log |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_router_ct_args.hpp` | `VC0_SENDER_CHANNEL_START` constant at line 129 | VERIFIED | `constexpr size_t VC0_SENDER_CHANNEL_START = 0;` present at line 129, after `VC1_RECEIVER_CHANNEL` at line 127 |
| `tt_metal/fabric/impl/kernels/edm_fabric/fabric_erisc_router.cpp` | Five `is_sender_channel_serviced[VC0_SENDER_CHANNEL_START]` guard expressions | VERIFIED | Exactly five occurrences confirmed at lines 2236, 2244, 2259, 2521, 2870; zero `is_sender_channel_serviced[0]` literals remain |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `fabric_erisc_router_ct_args.hpp:VC0_SENDER_CHANNEL_START` | `fabric_erisc_router.cpp:is_sender_channel_serviced[VC0_SENDER_CHANNEL_START]` | constexpr substitution | WIRED | Pattern `is_sender_channel_serviced[VC0_SENDER_CHANNEL_START]` confirmed at all five call sites; constant definition confirmed in header |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DS-01 | 04-01-PLAN.md | Device-side sender channel arrays use per-VC indexing | SATISFIED | All `is_sender_channel_serviced[0]` guard expressions replaced with named constant `VC0_SENDER_CHANNEL_START`; zero opaque literals remain |
| DS-02 | 04-01-PLAN.md | CT args parsing for sender channels uses per-VC accessors | SATISFIED | RT arg loops (`for i < MAX_NUM_SENDER_CHANNELS`) left unchanged (flat wire format intact); only `is_sender_channel_serviced` guard expressions — which ARE the per-VC access check — were updated |

### Anti-Patterns Found

No blockers or warnings found in the two modified files.

The five pre-existing `TODO` comments in `fabric_erisc_router_ct_args.hpp` (lines 197, 224, 492, 519, 559) are unrelated to this phase and were not introduced by these changes.

### Human Verification Required

#### 1. Sanity test pass

**Test:** Run the fabric sanity test after a clean build
**Expected:** All 12 latency test cases pass golden comparison; no hangs
**Why human:** Requires physical TT hardware; cannot be verified programmatically

The SUMMARY reports this was executed and passed. A separate human re-run is not required to accept the phase, but the sanity test result is not mechanically re-verifiable here.

## Gaps Summary

No gaps. All four observable truths are satisfied:

- The constant `VC0_SENDER_CHANNEL_START = 0` exists at the correct location in the header, grouped with the receiver channel constants to establish naming symmetry.
- All five opaque `[0]` literals are replaced — zero residual literals confirmed by grep.
- The flat RT arg loop bounds are untouched, preserving the wire format (DS-02).
- Commit `935addea397` is present in the git log, confirming the changes are committed.

---

_Verified: 2026-03-13T23:55:00Z_
_Verifier: Claude (gsd-verifier)_
