---
phase: 03-host-sender-per-vc
verified: 2026-03-13T21:00:00Z
status: passed
score: 5/5 must-haves verified
re_verification: false
---

# Phase 3: Host Sender Per-VC Verification Report

**Phase Goal:** Host Sender Per-VC — Make host-side sender channel arrays consistently per-VC indexed; remove dead flat-only structs and fix confusing dimension naming.
**Verified:** 2026-03-13T21:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth                                                                                                       | Status     | Evidence                                                                                          |
|----|-------------------------------------------------------------------------------------------------------------|------------|---------------------------------------------------------------------------------------------------|
| 1  | No dead flat-only `AllocatorConstructionParams` struct exists in `fabric_builder_config.hpp`                | VERIFIED   | grep returns zero matches in all of `tt_metal/` — struct fully removed by commit `052a7a884e8`   |
| 2  | `is_sender_channel_serviced_` is sized `[MAX_RISC_CORES_PER_ETH_CHAN][channel_id]` not `[MAX_NUM_VCS][...]` | VERIFIED   | Lines 620-622 of `erisc_datamover_builder.hpp` use `MAX_RISC_CORES_PER_ETH_CHAN`; no `MAX_NUM_VCS` variant present |
| 3  | `num_used_sender_channels_per_vc` is documented as canonical; `num_used_sender_channels` documented as derived total | VERIFIED | Lines 319-320 of `erisc_datamover_builder.hpp`: comment reads "Derived total: sum(num_used_sender_channels_per_vc). Use num_used_sender_channels_per_vc for per-VC logic." |
| 4  | Builder loops that branch on per-VC sender counts use `num_used_sender_channels_per_vc` directly            | VERIFIED   | `compute_mesh_router_builder.cpp` L260-261: explicit `per_vc[0] + per_vc[1]` sum; zero occurrences of flat `num_used_sender_channels` in that file |
| 5  | `compute_mesh_router_builder.cpp` derives `erisc_num_channels` from per-VC array, not flat total            | VERIFIED   | Lines 259-261 of `compute_mesh_router_builder.cpp` confirmed; committed as `287e6a59ab9`         |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact                                               | Expected                                                  | Status   | Details                                                                           |
|--------------------------------------------------------|-----------------------------------------------------------|----------|-----------------------------------------------------------------------------------|
| `tt_metal/fabric/builder/fabric_builder_config.hpp`    | Builder config types — `AllocatorConstructionParams` removed; `MAX_RISC_CORES_PER_ETH_CHAN` added | VERIFIED | Struct absent (grep clean); constant present at line 49 with explanatory comment |
| `tt_metal/fabric/erisc_datamover_builder.hpp`          | `FabricEriscDatamoverConfig` and `FabricEriscDatamoverBuilder` declarations with correct array dimensions | VERIFIED | `is_sender/receiver_channel_serviced_` use `MAX_RISC_CORES_PER_ETH_CHAN`; `num_used_sender_channels` comment updated |
| `tt_metal/fabric/erisc_datamover_builder.cpp`          | Builder constructor and `get_compile_time_args` implementation uses `is_sender_channel_serviced_[risc_id]` | VERIFIED | Four usages at lines 724, 730, 751, 1118 all index as `[risc_id][channel_id]`   |
| `tt_metal/fabric/compute_mesh_router_builder.cpp`      | `erisc_num_channels` derived from `num_used_sender_channels_per_vc` sum                         | VERIFIED | Lines 259-261 confirmed; zero flat `num_used_sender_channels` references remain  |

---

### Key Link Verification

| From                                              | To                                          | Via                                            | Status  | Details                                                                                       |
|---------------------------------------------------|---------------------------------------------|------------------------------------------------|---------|-----------------------------------------------------------------------------------------------|
| `FabricEriscDatamoverBuilder::is_sender_channel_serviced_` | `get_compile_time_args` risc_id loop   | `is_sender_channel_serviced_[risc_id]`         | WIRED   | Line 1118: `this->is_sender_channel_serviced_[risc_id][i]` inside CT args emission loop      |
| `FabricEriscDatamoverConfig::num_used_sender_channels_per_vc` | `compute_mesh_router_builder.cpp erisc_num_channels` | `edm_config.num_used_sender_channels_per_vc sum` | WIRED | Lines 260-261: `edm_config.num_used_sender_channels_per_vc[0] + edm_config.num_used_sender_channels_per_vc[1]` |
| `actual_sender_channels_vc0 / vc1`               | `emit_channel_allocations_ct_args` call     | already correct (no change needed per plan)    | WIRED   | Pre-existing correct wiring; plan explicitly deferred CT arg emission loop changes            |

---

### Requirements Coverage

| Requirement | Source Plan | Description                                                                           | Status    | Evidence                                                                                        |
|-------------|-------------|---------------------------------------------------------------------------------------|-----------|-------------------------------------------------------------------------------------------------|
| HS-01       | 03-01, 03-02 | Sender channel arrays in allocators and builder use per-VC indexing where appropriate | SATISFIED | `is_sender_channel_serviced_` uses `MAX_RISC_CORES_PER_ETH_CHAN`; `erisc_num_channels` uses per-VC sum; zero flat refs in mesh router builder |
| HS-02       | 03-01       | `num_used_sender_channels_per_vc` typing and naming is consistent and correct         | SATISFIED | `num_used_sender_channels` comment documents it as derived; `num_used_sender_channels_per_vc` is `std::array<std::size_t, MAX_NUM_VCS>` as expected |

No orphaned requirements: REQUIREMENTS.md maps only HS-01 and HS-02 to Phase 3, both claimed by plans.

---

### Anti-Patterns Found

No anti-patterns detected in modified files.

| File | Pattern Checked | Result |
|------|-----------------|--------|
| `tt_metal/fabric/builder/fabric_builder_config.hpp` | TODO/FIXME/placeholder, dead struct | Clean |
| `tt_metal/fabric/erisc_datamover_builder.hpp`       | TODO/FIXME, stub arrays, wrong dim | Clean |
| `tt_metal/fabric/erisc_datamover_builder.cpp`       | Flat index misuse, stub handlers | Clean |
| `tt_metal/fabric/compute_mesh_router_builder.cpp`   | Flat `num_used_sender_channels` refs | Clean — zero occurrences |

---

### Human Verification Required

None. All changes are structural (type declarations, constant naming, field derivation) and fully verifiable by static inspection. Build and sanity test were run by the executor (213/213 and 127/127 targets, 12/12 latency tests passed golden comparison).

---

### Gaps Summary

No gaps. All 5 observable truths verified, all 4 artifacts substantive and wired, both requirements satisfied by concrete code evidence.

---

### Commit Evidence

| Commit       | Description                                                                   | Files Modified                                                               |
|--------------|-------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| `052a7a884e8` | Remove dead `AllocatorConstructionParams`; add `MAX_RISC_CORES_PER_ETH_CHAN`; fix array dimensions | `fabric_builder_config.hpp`, `erisc_datamover_builder.hpp` |
| `287e6a59ab9` | Derive `erisc_num_channels` from per-VC sum in `compute_mesh_router_builder` | `compute_mesh_router_builder.cpp` |

Both commits verified present in branch HEAD (confirmed via `git log`).

---

_Verified: 2026-03-13T21:00:00Z_
_Verifier: Claude (gsd-verifier)_
