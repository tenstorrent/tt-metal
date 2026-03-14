# Requirements: PR #39538 Decomposition

**Defined:** 2026-03-12
**Core Value:** Each PR is self-contained, correct, and independently reviewable

## v1 Requirements

### Host Receiver Channels (PR 1)

- [ ] **HR-01**: `num_used_receiver_channels_per_vc` renamed to `is_receiver_channel_active_per_vc` with type `std::array<bool, MAX_NUM_VCS>`
- [ ] **HR-02**: All receiver channel arrays in allocators collapsed from 2D `[vc][channel]` to per-VC scalar `[vc]`
- [ ] **HR-03**: `receiver_channels_base_address` and `remote_receiver_channels_base_address` are per-VC scalars
- [ ] **HR-04**: `FabricEriscDatamoverConfig` and `FabricBuilderContext` updated to use bool receiver field
- [ ] **HR-05**: Dead builder fields removed (`receiver_channels_num_buffers`, `local_receiver_channels_buffer_address`, etc.)
- [ ] **HR-06**: All usages compile and tests pass

### Device Receiver Channels (PR 2)

- [ ] **DR-01**: Device-side receiver channel arrays use per-VC indexing (no flat 2D indexing)
- [ ] **DR-02**: CT args parsing for receiver channels uses per-VC accessors
- [ ] **DR-03**: Receiver channel pointer initialization uses per-VC template parameters
- [ ] **DR-04**: No change to CT args wire format (host/device remain compatible)

### Host Sender Channels (PR 3)

- [x] **HS-01**: Sender channel arrays in allocators and builder use per-VC indexing where appropriate
- [x] **HS-02**: `num_used_sender_channels_per_vc` typing and naming is consistent and correct

### Device Sender Channels (PR 4)

- [x] **DS-01**: Device-side sender channel arrays use per-VC indexing
- [x] **DS-02**: CT args parsing for sender channels uses per-VC accessors

### Channel Allocator (PR 5)

- [x] **CA-01**: Channel allocator uses both per-VC receiver and sender channel data
- [x] **CA-02**: Allocator API is consistent — no mixed flat/per-VC indexing

### Stream Reg Assignment (PR 6)

- [ ] **SR-01**: Host stream register assignment table uses per-VC indexing for both sender and receiver
- [ ] **SR-02**: Stream register map correctly assigns registers per-VC

## Out of Scope

| Feature | Reason |
|---------|--------|
| Other PR #39538 changes | Only decomposing the per-VC indexing refactor |
| Performance optimizations | Correctness/clarity refactor only |
| Sender-only changes in PR 1 | Only mechanical changes forced by receiver side |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| HR-01 through HR-06 | Phase 1 | In Progress |
| DR-01 through DR-04 | Phase 2 | Pending |
| HS-01 through HS-02 | Phase 3 | Pending |
| DS-01 through DS-02 | Phase 4 | Pending |
| CA-01 through CA-02 | Phase 5 | Pending |
| SR-01 through SR-02 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 18 total
- Mapped to phases: 18
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-12*
