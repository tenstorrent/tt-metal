<!--
SUMMARY: Deep-dive analysis of the ETH handshake protocol in T3K fabric ERISC firmware and design for an ETH DMA pre-ping rendezvous to eliminate the MMIO/non-MMIO race condition deadlock.
KEYWORDS: ETH handshake, ERISC, fabric, T3K, race condition, deadlock, pre-ping, DMA, MMIO, nonce exchange, handshake_info_t, eth_send_packet
SOURCE: Local code analysis of nsexton-0-racecondition-hunt worktree (tt-metal fabric firmware)
SCOPE: Current handshake protocol step-by-step, ETH DMA primitives, pre-ping insertion points, buffer conflicts, ETH link readiness, local ERISC barrier interaction
USE WHEN: Designing or reviewing the ETH DMA pre-ping rendezvous fix for #42429 race condition in fabric init
-->

# ETH DMA Pre-Ping Rendezvous — Researcher A Draft

## 1. How the Current Handshake Works (step by step)

The handshake is a nonce-exchange between paired ERISC cores over ETH DMA.

### Setup (both sides)

- `init_handshake_info()` at `edm_handshake.hpp:61` allocates a `handshake_info_t` struct at `handshake_addr` (a compile-time arg from `fabric_erisc_router_ct_args.hpp:130`)
- It flushes any stale ETH TX queue state (`edm_handshake.hpp:78` — guards against prior firmware crash leaving TXQ busy)
- Sets `local_value = 0` (`edm_handshake.hpp:85`)
- Writes the `session_nonce` into `scratch[0]` (`edm_handshake.hpp:86`)
- Writes `mesh_id/device_id` into `scratch[1]` (`edm_handshake.hpp:90`) — this is how peers exchange identity

### Sender Side (`fabric_router_eth_handshake.hpp:24-78`)

- Computes `local_val_addr = &handshake_info->local_value / 16` (word address for ETH DMA)
- Computes `scratch_addr = &handshake_info->scratch / 16`
- Enters spin loop: `while local_value != session_nonce AND no termination signal`
  - On each iteration: calls `internal_::eth_send_packet(0, scratch_addr, local_val_addr, 1)` — sends 16 bytes from local scratch to REMOTE `local_value`. Writes the nonce + identity to the peer's `handshake_info_t` bytes 0-15.
  - Periodically calls `run_routing()` for context switch
  - Has a watchdog counter that emits `WAYPOINT("HSST")` every ~100M iterations
- After loop exit (nonce received from peer): sends one final `eth_send_packet` to handle the simultaneous-sender race (FIX HS2, line 64-77)

### Receiver Side (`fabric_router_eth_handshake.hpp:83-131`)

- Same `init_handshake_info` setup
- Enters spin loop: `while local_value != session_nonce AND no termination signal`
  - Does NOT send packets in the loop (key difference from sender)
  - Just polls `local_value` with cache invalidation
  - Has watchdog counter emitting `WAYPOINT("HSRT")`
- After loop exit (nonce received): sends one `eth_send_packet` back to the sender to deliver its own nonce + identity

### The Caller (`fabric_erisc_router.cpp:3687-3704`)

- `is_handshake_sender` is a compile-time constant (CT arg at `ct_args.hpp:129`) determined by tie-break: lower ID = sender (`erisc_datamover_builder.cpp:933`)
- Since MMIO devices are IDs 0-3 and non-MMIO are 4-7, MMIO is ALWAYS the sender
- After handshake completes, `neighbor_mesh_id/neighbor_device_id` are extracted from the handshake struct and stored in telemetry (lines 3707-3715)

### Before the Handshake (`fabric_erisc_router.cpp:3647-3685`)

1. `wait_for_other_local_erisc()` barrier at line 3658 (only when `NUM_ACTIVE_ERISCS > 1`)
2. Write `HANDSHAKE_READY` to `edm_status_ptr` (line 3663) — FIX CU
3. If `host_gate_enabled` (MMIO cores only): spin on `HOST_GATE_OPEN` (lines 3673-3684) — FIX CY

## 2. The ETH DMA Mechanism

The core primitive is `internal_::eth_send_packet()` at `tunneling.h:82`:

- Parameters: `q_num` (TX queue, always 0 for handshake), `src_word_addr`, `dest_word_addr`, `num_words`
- All addresses are in 16-byte words (actual byte addr = word_addr * 16)
- It writes to ETH TXQ registers: `TRANSFER_START_ADDR`, `DEST_ADDR`, `TRANSFER_SIZE_BYTES`, then `CMD=START_DATA`
- The busy-wait spins on `eth_txq_is_busy()` before sending (with optional context switch)
- Each `eth_send_packet` sends `num_words * 16` bytes from local L1 to peer L1 at the corresponding address

The "unsafe" variant (`tunneling.h:98`) skips the busy-wait (asserts queue is free).

Key insight: `eth_send_packet` operates at the RAW HARDWARE LEVEL. It writes directly to the ETH MAC TX queue registers. There is NO software protocol layer — it is a raw L1-to-L1 DMA over the ethernet link.

### Buffer Addressing in the Handshake

`handshake_info_t` is 32 bytes total (`edm_handshake.hpp:49-56`):

- Bytes 0-3: `local_value` (where remote nonce lands)
- Bytes 4-7: `neighbor_mesh_id` (2B) + `neighbor_device_id` (1B) + padding (1B)
- Bytes 8-15: `padding[2]`
- Bytes 16-31: `scratch[4]` (source of outgoing nonce)

The sender does: `eth_send_packet(0, scratch_addr, local_val_addr, 1)` — sends 16B from local scratch (bytes 16-31) to remote `local_value` (bytes 0-15). This overwrites the remote's `local_value` with the nonce and identity fields.

## 3. Where the Pre-Ping Would Be Inserted

### Receiver Side (non-MMIO ERISC)

Insert AFTER line 3663 (`HANDSHAKE_READY` write) and BEFORE the `host_gate_enabled` block at line 3673. The receiver has just:

- Completed all channel/stream init
- Passed the `wait_for_other_local_erisc()` barrier
- Written `HANDSHAKE_READY` status

Pseudo-insertion point: between lines 3664 and 3666.

The receiver (non-MMIO) would:

1. Write a known magic value (e.g. `0xPREP1NG`) to a dedicated L1 address
2. Call `internal_::eth_send_packet(0, preping_src_word_addr, preping_dst_word_addr, 1)` to send 16B to the peer
3. Then fall through to the normal handshake code

### Sender Side (MMIO ERISC)

Insert AFTER line 3663 (`HANDSHAKE_READY` write), REPLACING the `host_gate_enabled` spin (lines 3673-3684). Instead of waiting for `HOST_GATE_OPEN` from the host, the MMIO ERISC would:

1. Spin reading a dedicated L1 address for the pre-ping magic value
2. Once received, proceed directly to `fabric_sender_side_handshake`

This completely replaces FIX CY. The sender no longer needs the host in the critical path.

## 4. Buffer Conflicts

The pre-ping CANNOT use the handshake buffer (`handshake_addr`).

`init_handshake_info()` (`edm_handshake.hpp:83-93`) zeroes `local_value` and writes scratch. If the pre-ping wrote to `handshake_addr` before `init_handshake_info()` runs, the init would overwrite it. And `init_handshake_info()` is called INSIDE `fabric_sender_side_handshake`/`fabric_receiver_side_handshake` — that's AFTER the pre-ping check.

The pre-ping needs a DEDICATED address. Options considered:

- **A new CT arg `PRE_PING_ADDR`** pointing to a reserved 16-byte slot in L1 (cleanest) -- RECOMMENDED
- ~~`AERISC_FABRIC_SCRATCH_BASE`~~ — REJECTED: 28 bytes at `TELEMETRY_BASE + 156`. Since `156 % 16 = 12`, this address is NOT 16-byte aligned. `eth_send_packet` does `src_word_addr << 4` / `dest_word_addr << 4` (tunneling.h:91-93), so an unaligned address would be silently truncated and corrupt adjacent memory. (Cross-review with Researcher B confirmed this.)
- Reuse part of `edm_status_ptr`'s region (risky — that address is polled by host)
- Use a stream scratch register (like `wait_for_other_local_erisc` does with `MULTI_RISC_TEARDOWN_SYNC_STREAM_ID`) — but stream registers are LOCAL only, not reachable via ETH DMA

**Consensus recommendation**: allocate via `erisc_datamover_builder` with a new `PRE_PING_ADDR` CT arg, using `tt::round_up(next_free_addr, 16)` for alignment — same pattern as `handshake_address` (`erisc_datamover_builder.cpp:716-717`). Reserve 16 bytes (one `eth_send_packet` transfer unit), not just 4, since `eth_send_packet` transfers whole 16B words.

## 5. ETH DMA Availability

YES, the ETH link is ready for DMA before the handshake. Evidence:

- `init_handshake_info()` at `edm_handshake.hpp:78-81` does an ETH TXQ flush if busy — this implies the hardware is already accessible
- The existing handshake itself uses `eth_send_packet`, which writes to ETH TXQ registers. If the ETH link weren't ready, the handshake itself would fail.
- There is NO "ETH link init" step visible in the ERISC firmware. The ETH physical link is established by hardware/firmware at a lower level (the ethernet MAC) before the ERISC application firmware runs.
- The WAYPOINT and POSTCODE calls before the handshake (e.g., `EDM_VCS_SETUP_COMPLETE` at line 3586, `STREAM_REG_INITIALIZED` at line 3192) confirm that by the time we reach the handshake area, all software init is done and hardware is ready.

**Caveat**: the TXQ flush in `init_handshake_info` guards against stale state from a prior kernel run. The pre-ping runs BEFORE `init_handshake_info`, so the pre-ping itself could hit a stale TXQ. The pre-ping code should include the same TXQ flush guard:

```cpp
if (eth_txq_is_busy()) {
    eth_txq_reg_write(0, ETH_TXQ_CMD, ETH_TXQ_CMD_FLUSH);
    eth_txq_reg_read(0, ETH_TXQ_CMD);
    while (eth_txq_is_busy()) {}
}
```

Or alternatively, call `init_handshake_info()` earlier (before the pre-ping), moving it out of `fabric_sender/receiver_side_handshake`. But that's a larger refactor.

## 6. The `wait_for_other_local_erisc()` Barrier

Located at `fabric_erisc_router.cpp:3658`, runs BEFORE the `HANDSHAKE_READY` write (line 3663) and before the handshake.

### How It Works (`fabric_erisc_router.cpp:2931-2958`)

- Uses stream scratch registers (`MULTI_RISC_TEARDOWN_SYNC_STREAM_ID`) for local inter-ERISC sync
- ERISC 0 (teardown master) writes `0x0fed`, waits for peer to write `0x1bad`
- ERISC 1 (subordinate) waits for `0x0fed`, then writes `0x1bad`
- Has a bounded timeout of 100M iterations (~2-4 seconds)
- This is LOCAL only — syncs the two ERISC cores on the SAME chip, not across ETH

### Impact on Pre-Ping

- The barrier COMPLETES before the pre-ping insertion point. It does not block the pre-ping.
- The barrier ensures both local ERISCs have finished channel/stream init before either enters handshake. Important because the ETH link is shared between two local ERISC cores.
- The pre-ping approach does NOT interact with this barrier — the barrier is about local core sync, the pre-ping is about cross-link peer readiness.
- If `NUM_ACTIVE_ERISCS > 1`, both local ERISCs must pass this barrier before EITHER sends a pre-ping. This is already the case with the current code structure.
- The barrier at line 3167-3169 (further up in init) is a DIFFERENT invocation — after stream register initialization. The one at 3658 is the pre-handshake barrier. Both complete before the handshake entry point.

## Summary: Pre-Ping Design

NON-MMIO (receiver):
1. [existing] `wait_for_other_local_erisc()` — line 3658
2. [existing] Write `HANDSHAKE_READY` — line 3663
3. [NEW] Flush TXQ if busy
4. [NEW] Write pre-ping magic to local L1 slot, `eth_send_packet` to peer's pre-ping L1 slot
5. [existing] Enter `fabric_receiver_side_handshake` — line 3697

MMIO (sender):
1. [existing] `wait_for_other_local_erisc()` — line 3658
2. [existing] Write `HANDSHAKE_READY` — line 3663
3. [NEW] Spin waiting for pre-ping magic in local L1 slot (with cache invalidation + termination check)
4. [REMOVED] No more `HOST_GATE_OPEN` spin (FIX CY eliminated)
5. [existing] Enter `fabric_sender_side_handshake` — line 3689

Requirements:
- New CT arg `PRE_PING_ADDR` for the 16B L1 slot
- Builder allocates this slot via `tt::round_up(next_free_addr, 16)` adjacent to `handshake_addr`
- The pre-ping magic value must differ from the handshake nonce and must not be 0 (since L1 might retain stale zeros)
- TXQ flush before pre-ping send (or restructure `init_handshake_info` to run earlier)

## Cross-Review Consensus (Researcher A + Researcher B)

Items agreed upon after cross-review:

1. **Buffer allocation**: `PRE_PING_ADDR` via `erisc_datamover_builder` with `tt::round_up()` alignment. `AERISC_FABRIC_SCRATCH_BASE` rejected due to 16B alignment failure (offset 156, 156%16=12).

2. **Termination check in pre-ping spin**: Must use the WH-specific two-path pattern (`#ifndef ARCH_WORMHOLE` guard) — raw `*termination_signal_ptr` deref on WH, `got_immediate_termination_signal()` elsewhere. Reference pattern: FIX CY loop at `fabric_erisc_router.cpp:3673-3685`.

3. **Teardown edge case**: If non-MMIO ERISC dies before sending pre-ping, MMIO spins forever unless termination signal check is in the spin loop. Use WAYPOINT-only watchdog (no hard break-out) — breaking out would recreate the race at the handshake layer.

4. **Commit strategy**: Two separate commits for clean revert path:
   - Commit 1: Add pre-ping mechanism
   - Commit 2: Remove FIX CY artifacts (~80 lines across 8 files: `host_gate_enabled` CT arg, `HOST_GATE_OPEN` enum, `open_erisc_handshake_gate()` in device.cpp, Phase C in `fabric_firmware_initializer.cpp`)

5. **16B transfer size**: The pre-ping L1 slot must be 16 bytes (not 4), since `eth_send_packet` transfers whole 16B words.
