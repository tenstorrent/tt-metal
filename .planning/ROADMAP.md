# Roadmap: Fabric Auto-Packetization

## Milestones

- **v1.0 Fabric Auto-Packetization** — Phase 1 (shipped 2026-03-11)
- **v1.1 Silicon Validation** — Phase 2 (in progress)

## Phases

<details>
<summary>v1.0 Fabric Auto-Packetization (Phase 1) — SHIPPED 2026-03-11</summary>

- [x] Phase 1: fabric-auto-packetization (8/8 plans) — completed 2026-03-11

</details>

### v1.1 Silicon Validation

- [ ] Phase 2: silicon-data-transfer-validation — Validate all auto-packetizing wrappers deliver correct data on silicon with payloads exceeding FABRIC_MAX_PACKET_SIZE

**Goal:** Run silicon data-transfer tests for all 9 auto-packetizing wrapper families. Rewrite runners to use BaseFabricFixture, follow addrgen_write test pattern, verify byte-for-byte data correctness with multiple payload sizes.

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. fabric-auto-packetization | v1.0 | 8/8 | Complete | 2026-03-11 |
| 2. silicon-data-transfer-validation | v1.1 | 0/0 | Not started | - |
