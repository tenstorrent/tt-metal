# PR #39538 Decomposition: Per-VC Channel Indexing Refactor

## What This Is

Decompose the monolithic PR #39538 into 6 smaller, independently reviewable PRs that correctly migrate fabric router channel arrays to per-VC indexing. Each VC (Virtual Channel) can have at most 0 or 1 receiver channel and a fixed number of sender channels — arrays that suggest multiple channels per VC are semantically wrong and must be fixed in both host (builder) and device (kernel) code.

## Core Value

Each PR is self-contained, correct, and independently reviewable — no partial or mixed migrations that leave the codebase in an inconsistent state.

## Requirements

### Validated

(None yet — in progress)

### Active

- [ ] Host receiver channels migrated to per-VC (bool `is_receiver_channel_active_per_vc`, scalar base addresses)
- [ ] Device receiver channels migrated to per-VC indexing
- [ ] Host sender channels migrated to per-VC indexing
- [ ] Device sender channels migrated to per-VC indexing
- [ ] Channel allocator updated to use both per-VC receiver and sender
- [ ] Host stream reg assignment table/map updated for both sender/receiver per-VC

### Out of Scope

- PR #39538 content beyond the per-VC indexing refactor — other features/fixes in that PR are separate
- Device kernel performance changes — this is a correctness/clarity refactor only
- Sender channel side changes except where mechanically forced by receiver changes (within the first PR)

## Context

- PR #39538: https://github.com/tenstorrent/tt-metal/pull/39538 (the monolithic change being decomposed)
- Branch: `snijjar/convert-flat-to-per-vc-indexing-receiver-channels`
- Each VC (VC0, VC1) has at most 1 receiver channel — 2D arrays indexed `[vc][channel_id]` are always accessed at `[vc][0]`, so they collapse to per-VC scalars
- Host-side PR 1 (receiver) is partially complete: allocator files done, `erisc_datamover_builder` done
- Device-side changes involve CT args parsing, channel pointer initialization, and per-VC template accessors

## Constraints

- **Scope**: Each PR touches only its designated layer (host or device) and channel type (sender or receiver)
- **Correctness**: No change to CT args wire format — only host-side naming/typing and device-side indexing logic
- **Device builds**: ERISC kernels auto-recompile on next test run; no `ninja` rebuild needed for device-only changes

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| `num_used_receiver_channels_per_vc` → `is_receiver_channel_active_per_vc` (bool) | Each VC has 0 or 1 receiver channels, bool is semantically correct | — Pending |
| Keep `actual_receiver_channels_per_vc` as `size_t` (0 or 1) in builder API | Interacts with bitfield arithmetic in channel trimming functions | — Pending |
| Dead fields removed from `FabricEriscDatamoverBuilder` | Never accessed after allocator was introduced | — Pending |

---
*Last updated: 2026-03-12 after project initialization*
