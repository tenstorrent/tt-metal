<!--
SUMMARY: Failure modes, edge cases, and gotchas for the ETH DMA pre-ping approach to fix the T3K fabric ERISC handshake race condition (#42429)
KEYWORDS: erisc, handshake, race-condition, pre-ping, eth-dma, teardown, termination, watchdog, scratch-buffer, wormhole, blackhole, t3k, fabric
SOURCE: Code analysis of nsexton-0-racecondition-hunt branch in tt-metal (fabric_erisc_router.cpp, fabric_router_eth_handshake.hpp, tt_fabric_utils.h, fabric_firmware_initializer.cpp, device.cpp, eth_l1_address_map.h)
SCOPE: All failure modes of the proposed ETH DMA pre-ping fix: termination handling, ETH link state, teardown asymmetry, buffer placement, watchdog/timeout, multi-ERISC interactions, and FIX CY cleanup surface
USE WHEN: Implementing the ETH DMA pre-ping fix for #42429, reviewing the implementation, or debugging issues with the pre-ping mechanism
-->

# ETH DMA Pre-Ping: Failure Modes and Edge Cases

## Context

**Problem**: MMIO ERISCs (chips 0-3, always handshake senders due to tie-break) enter the handshake loop before non-MMIO ERISCs (chips 4-7, always receivers) have reached their receiver loop. Nonce dropped, deadlock.

**Proposed fix**: Non-MMIO ERISC sends a 4-byte "peer_ready" pre-ping over raw ETH DMA to its MMIO peer after reaching `HANDSHAKE_READY`. MMIO ERISC spins waiting for this pre-ping before entering `fabric_sender_side_handshake`. Eliminates host from sync path entirely.

---

## 1. Termination Handling

The FIX CY spin loop (fabric_erisc_router.cpp:3673-3685) establishes the pattern:

```cpp
if constexpr (host_gate_enabled) {
    WAYPOINT("HGWT");
    while (*edm_status_ptr != tt::tt_fabric::EDMStatus::HOST_GATE_OPEN) {
        router_invalidate_l1_cache<ENABLE_RISC_CPU_DATA_CACHE>();
        if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(termination_signal_ptr) ==
            static_cast<uint32_t>(tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE)) {
            WAYPOINT("HGXT");
            *edm_status_ptr = tt::tt_fabric::EDMStatus::TERMINATED;
            return;
        }
    }
    WAYPOINT("HGOP");
}
```

**CRITICAL GOTCHA ON WORMHOLE**: `got_immediate_termination_signal()` (tt_fabric_utils.h:24-31) does MORE than check termination_signal_ptr — it also checks `launch_msg->kernel_config.exit_erisc_kernel` from the mailbox. But in the FIX CY loop, a RAW dereference of termination_signal_ptr is used instead. This is intentional — on WH, the full function is compiled out behind `#ifndef ARCH_WORMHOLE` guards. The handshake code (fabric_router_eth_handshake.hpp:42-48) shows the canonical two-path pattern:

```cpp
#ifndef ARCH_WORMHOLE
    && !tt::tt_fabric::got_immediate_termination_signal<RISC_CPU_DATA_CACHE_ENABLED>(termination_signal_ptr)
#else
    && *termination_signal_ptr != static_cast<uint32_t>(tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE)
#endif
```

**RECOMMENDATION**: The pre-ping spin must use this EXACT two-path pattern. On early exit, set `*edm_status_ptr = EDMStatus::TERMINATED` and return. Must call `router_invalidate_l1_cache()` each iteration.

---

## 2. ETH Link Up/Down

There is NO explicit ETH link-up check in the ERISC init path before HANDSHAKE_READY. The sequence:

- fabric_erisc_router.cpp:3600-3660: Channel/buffer init (all local L1 operations)
- Line 3658: `wait_for_other_local_erisc()` (same-device barrier, no ETH)
- Line 3663: `*edm_status_ptr = HANDSHAKE_READY` (local L1 write)
- Line 3673: FIX CY gate spin (local L1 poll)
- Line 3689-3704: ETH handshake (first actual ETH traffic)

The ETH link is managed at the hardware/firmware level. By the time the ERISC fabric router runs, ethernet link training is already complete — it happens during ERISC ROM boot. The retrain_count/CRC addresses at bottom of eth_l1_address_map.h (lines 147-152) are for monitoring an already-trained link.

The ETH link SHOULD be up before HANDSHAKE_READY. Subtle risk: `eth_send_packet()` writes to ETH TXQ registers expecting the remote receiver to be listening. If the ETH link drops between HANDSHAKE_READY and the pre-ping send, the pre-ping silently fails. MMIO side spins forever.

**RECOMMENDATION**: This is NOT a new risk — the existing handshake has the same vulnerability. No special handling needed, but the watchdog (point 5) is essential as a safety net.

---

## 3. Asymmetry on Teardown (MOST DANGEROUS)

Scenario:
1. Non-MMIO ERISC has NOT yet reached HANDSHAKE_READY (still initializing)
2. Host decides to terminate (test failure, Ctrl+C)
3. Host writes IMMEDIATELY_TERMINATE to non-MMIO ERISC's termination_signal_ptr
4. Non-MMIO ERISC terminates without ever sending the pre-ping
5. MMIO ERISC spins forever waiting for the pre-ping

The teardown path (fabric_firmware_initializer.cpp:597+) writes IMMEDIATELY_TERMINATE to ALL ERISCs. The MMIO ERISC's pre-ping spin MUST check termination signal (same as FIX CY). This is covered by point 1.

Additional concern: non-MMIO ERISC crashes (hard fault, infinite loop in init) and never sends pre-ping, AND host hasn't sent terminate yet. MMIO ERISC spins indefinitely.

**RECOMMENDATION**: Pre-ping spin MUST have bounded watchdog (see point 5). Termination check is the primary escape hatch. The `wait_for_notification` pattern (tt_fabric_utils.h:81-115) shows both termination check AND max_iterations — the pre-ping spin should do both.

No reverse direction concern: non-MMIO ERISC is always the SENDER of the pre-ping. It sends and proceeds to the handshake receiver loop without spinning.

---

## 4. Pre-Ping Buffer Placement

**~~RETRACTED~~: `AERISC_FABRIC_SCRATCH_BASE` is NOT 16-byte aligned.**

Verified math: `SCRATCH_BASE = TELEMETRY_BASE + 160 - 4 = TELEMETRY_BASE + 156`. TELEMETRY_BASE is 32-byte aligned (dev_mem_map.h:260, `& ~31` mask). 156 % 16 = 12, so SCRATCH_BASE is misaligned. `eth_send_packet` does `src_word_addr << 4` / `dest_word_addr << 4` internally — an unaligned address would be silently truncated, writing to the wrong L1 location and corrupting adjacent memory.

(Original analysis: tunneling.h:24-27 defines `fabric_scratch_ptr` and `ROUTER_SCRATCH_WRITE(id, val)` macro at SCRATCH_BASE, but ZERO actual calls found — unused debug infrastructure. The 28 bytes would have been sufficient space-wise.)

**RECOMMENDED APPROACH**: Allocate via `erisc_datamover_builder` with a new `PRE_PING_ADDR` CT arg. The builder already uses `tt::round_up()` for alignment — same pattern as `handshake_address` (erisc_datamover_builder.cpp:716-717):
```cpp
handshake_address(tt::round_up(
    tt::tt_metal::hal::get_erisc_l1_unreserved_base(), FabricEriscDatamoverConfig::eth_channel_sync_size)),
```

The pre-ping allocation should reserve 16 bytes (one `eth_send_packet` transfer unit), not just 4, since ETH DMA operates in 16B words:
```cpp
pre_ping_address = tt::round_up(next_free_addr, 16);
// reserve 16 bytes
```

**DO NOT** reuse `handshake_addr` — `init_handshake_info()` (called later in the handshake) would overwrite it.

**IMPORTANT**: Pre-ping is written by NON-MMIO ERISC via `eth_send_packet` into MMIO ERISC's L1. Buffer address must be in MMIO ERISC's address space. Since both share the same memory map, the same CT arg value works on both sides.

---

## 5. Watchdog / Timeout Behavior

Consistent codebase pattern:
- fabric_router_eth_handshake.hpp:39: `kWatchdogIter = 100,000,000` (~2-4 seconds on ERISC)
- wait_for_other_local_erisc:2933: `kSyncMaxIter = 100,000,000`
- wait_for_notification:92: `kWatchdogIter = 100,000,000`

Two distinct behaviors:
1. **WAYPOINT-only** (handshake, wait_for_notification): Emits WAYPOINT postcode, keeps spinning. Termination check is real exit.
2. **Break-out** (wait_for_other_local_erisc:2944): Actually breaks after timeout. Used for same-device peers only.

**RECOMMENDATION**: Use WAYPOINT-only watchdog (no hard break-out). If we hard break-out, we enter the handshake with no guarantee the peer is ready — recreating the original race. Termination signal is the escape. If peer is truly dead, host WILL send IMMEDIATELY_TERMINATE during teardown.

---

## 6. Multi-ERISC per Device

Each WH device has up to 16 ethernet channels. `NUM_ACTIVE_ERISCS` (CT arg, erisc_datamover_builder.cpp:1293): on BH can be 2, on WH typically 1.

Each ETH channel runs its own ERISC instance. Each MMIO ERISC paired with exactly ONE non-MMIO ERISC across that ETH link. Pre-ping is point-to-point per link. No fan-out, no broadcast.

`wait_for_other_local_erisc()` (line 3658) runs BEFORE HANDSHAKE_READY (line 3663). Synchronizes two RISC cores WITHIN a single ETH channel (NUM_ACTIVE_ERISCS > 1), not across channels.

- Pre-ping is per-ETH-link, completely independent across channels
- `wait_for_other_local_erisc` has no interaction with pre-ping
- Pre-ping spin should be gated by same CT args as handshake (`enable_ethernet_handshake`, `is_handshake_sender`, and a new CT arg replacing `host_gate_enabled`)

**NO INTERACTION** with `wait_for_other_local_erisc` needed.

---

## 7. FIX CY Cleanup

Complete artifact list to remove when replacing FIX CY with ETH DMA pre-ping:

**FIRMWARE SIDE:**
- `fabric_erisc_router.cpp:3666-3685` — entire `if constexpr (host_gate_enabled)` block (HGWT/HGOP/HGXT waypoints)
- `fabric_erisc_router_ct_args.hpp:268-272` — `host_gate_enabled` CT arg definition and comment
- `fabric_edm_packet_header.hpp:62` — `HOST_GATE_OPEN = 0xA0B0C0D2` enum value in EDMStatus

**HOST SIDE:**
- `erisc_datamover_builder.cpp:1226` — `named_args["HOST_GATE_ENABLED"]` assignment
- `device.cpp:2878-2925` — entire `Device::open_erisc_handshake_gate()` method
- `device_impl.hpp:195-199` — declaration of `open_erisc_handshake_gate()`
- `fabric_firmware_initializer.cpp:3540-3553` — Phase C block calling `open_erisc_handshake_gate()` on MMIO devices

**UTILITY:**
- `edm_status_utils.hpp:61,87` — `HOST_GATE_OPEN` cases in status-to-string mappers

**TOTAL**: ~80 lines removed across 8 files.

Preserve FIX CY comment references (#42429) where they document the race condition, updated to reference the new pre-ping mechanism.

**NOTE on Pass A / Pass B**: `fabric_firmware_initializer.cpp:3528-3538` — Pass B waits for non-MMIO ERISCs to reach HANDSHAKE_READY via host polling. With pre-ping, Pass B is no longer needed for correctness (non-MMIO ERISCs self-signal). Consider keeping as telemetry-only or removing entirely.
