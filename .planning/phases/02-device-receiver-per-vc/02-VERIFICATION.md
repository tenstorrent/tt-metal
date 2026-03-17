---
phase: 02-device-receiver-per-vc
status: passed
verified: 2026-03-13
method: git-evidence
commit: 7132db45f5d2d8e81065fca41039b6205e0c450e
---

# Phase 2 Verification: Device Receiver Per-VC

**Status:** passed
**Score:** 4/4 must-haves verified
**Method:** Git evidence (phase executed before GSD phase directories were set up)

## Evidence

**Commit:** `7132db45f5d` — "Phase 2 validated: build passes, all 12 latency tests pass golden comparison, no hangs."

**Truth 1 — DR-01: Device-side receiver arrays use per-VC indexing:**
All `is_receiver_channel_serviced[0/1]`, `to_receiver_packets_sent_streams[0/1]`, `receiver_channel_response_credit_senders[0]`, and `receiver_channel_forwarding_*_cmd_buf_ids[0/1]` replaced with `[VC0_RECEIVER_CHANNEL]` / `[VC1_RECEIVER_CHANNEL]` constants.

**Truth 2 — DR-02: CT args parsing uses per-VC accessors:**
`run_receiver_channel_step<0,` / `<1,` template params replaced with `<VC0_RECEIVER_CHANNEL,` / `<VC1_RECEIVER_CHANNEL,`.
`ChannelPointersTuple::get<0/1>()` and `ReceiverChannelBuffers::get<0>()` use per-VC constants.

**Truth 3 — DR-03: Receiver channel pointer init uses per-VC template params:**
`receiver_channel_pointers.template get<0/1>()` → `get<VC0/1_RECEIVER_CHANNEL>()`.
`local_receiver_channels.template get<0>()` → `get<VC0_RECEIVER_CHANNEL>()`.

**Truth 4 — DR-04: No CT args wire format change:**
Purely naming/constant substitution — no reordering, no new args, no value changes. Build passed, 12/12 latency tests pass golden comparison, no hangs.

## Requirements Coverage

| Requirement | Status | Evidence |
|-------------|--------|---------|
| DR-01 | satisfied | All `[0]`/`[1]` receiver array accesses use VC constants in `fabric_erisc_router.cpp` |
| DR-02 | satisfied | `run_receiver_channel_step` template params use VC constants |
| DR-03 | satisfied | `get<>()` template params use VC constants for pointer init |
| DR-04 | satisfied | 29 insertions / 29 deletions — pure rename, no wire format change |
