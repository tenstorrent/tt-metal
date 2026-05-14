// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "fabric/fabric_edm_packet_header.hpp"

// Shared EDMStatus → string helpers used by both device.cpp and
// fabric_firmware_initializer.cpp.  Keep all cases here in sync when adding
// new EDMStatus enumerators in tt_metal/fabric/fabric_edm_packet_header.hpp.
//
// edm_status_name(EDMStatus)  — typed enum → name, used for enum-aware callers
//                               and by is_known_edm_status().
// edm_status_str(uint32_t)    — raw uint32_t → name, handles EthDiagSentinel
//                               values (0x49706550, 0xDEAD****) in addition to
//                               all EDMStatus enumerators.
// is_known_edm_status(uint32_t) — returns true iff the raw value is a recognised
//                               EDMStatus enumerator (used by probe / quiesce logic).

namespace tt::tt_metal {

// Sentinel values used to represent ETH channel status fields when no valid
// EDMStatus enumerator applies.  Includes both firmware-written sentinels and
// host-side diagnostic placeholders.
enum class EthDiagSentinel : uint32_t {
    // Written by base UMD relay firmware to erisc_sync_addr once it has
    // completed .bss init and entered its polling loop.  ASCII "iPeP".
    // The host uses this value to distinguish:
    //   - "UMD relay never launched" (still 0x49706550 → leave alone)
    //   - "launch sent but ERISC crashed before STARTED" (also 0x49706550 →
    //     detect via the HOST_PRE_LAUNCH_CANARY probe written before launch)
    // Firmware writes this.  Do not use as a host-side placeholder.
    BASE_UMD_FIRMWARE_SENTINEL = 0x49706550u,

    // Host wrote this value to router_sync_address before sending the launch
    // message.  If the field still reads this value after launch, ERISC never
    // polled — the channel is stuck in base firmware or crashed.
    HOST_PRE_LAUNCH_CANARY = 0xDEADB07Eu,

    // A read_core() call for this channel threw an exception.  The status field
    // is set to this value as a placeholder so the channel is included in
    // diagnostic output with a recognisable sentinel rather than a stale value.
    READ_EXCEPTION = 0xDEADBEEFu,

    // Phase 5b per-iteration deadline exceeded — the read was skipped to avoid
    // blocking subsequent channels.  The actual hardware value was not observed.
    PHASE5B_DEADLINE_SKIPPED = 0xDEAD5B5Bu,

    // Phase 5b relay read threw an exception.  Equivalent to READ_EXCEPTION but
    // specifically in the Phase 5b polling path (distinct value aids grep).
    PHASE5B_READ_EXCEPTION = 0xDEADECE7u,
};

inline const char* edm_status_name(tt::tt_fabric::EDMStatus s) {
    switch (s) {
        case tt::tt_fabric::EDMStatus::STARTED:                      return "STARTED";
        case tt::tt_fabric::EDMStatus::REMOTE_HANDSHAKE_COMPLETE:    return "REMOTE_HANDSHAKE_COMPLETE";
        case tt::tt_fabric::EDMStatus::LOCAL_HANDSHAKE_COMPLETE:     return "LOCAL_HANDSHAKE_COMPLETE";
        case tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC:            return "READY_FOR_TRAFFIC";
        case tt::tt_fabric::EDMStatus::TERMINATED:                   return "TERMINATED";
        case tt::tt_fabric::EDMStatus::INITIALIZATION_STARTED:       return "INITIALIZATION_STARTED";
        case tt::tt_fabric::EDMStatus::TXQ_INITIALIZED:              return "TXQ_INITIALIZED";
        case tt::tt_fabric::EDMStatus::STREAM_REG_INITIALIZED:       return "STREAM_REG_INITIALIZED";
        case tt::tt_fabric::EDMStatus::DOWNSTREAM_EDM_SETUP_STARTED: return "DOWNSTREAM_EDM_SETUP_STARTED";
        case tt::tt_fabric::EDMStatus::EDM_VCS_SETUP_COMPLETE:       return "EDM_VCS_SETUP_COMPLETE";
        case tt::tt_fabric::EDMStatus::WORKER_INTERFACES_INITIALIZED: return "WORKER_INTERFACES_INITIALIZED";
        case tt::tt_fabric::EDMStatus::ETHERNET_HANDSHAKE_COMPLETE:  return "ETHERNET_HANDSHAKE_COMPLETE";
        case tt::tt_fabric::EDMStatus::VCS_OPENED:                   return "VCS_OPENED";
        case tt::tt_fabric::EDMStatus::ROUTING_TABLE_INITIALIZED:    return "ROUTING_TABLE_INITIALIZED";
        case tt::tt_fabric::EDMStatus::INITIALIZATION_COMPLETE:      return "INITIALIZATION_COMPLETE";
        default: return "(unknown)";
    }
}

// Maps a raw uint32_t edm_status to a human-readable name.  Handles the two
// host-side sentinel values written before device-side firmware runs.
inline const char* edm_status_str(uint32_t v) {
    switch (static_cast<tt::tt_fabric::EDMStatus>(v)) {
        case tt::tt_fabric::EDMStatus::INITIALIZATION_STARTED:        return "INITIALIZATION_STARTED";
        case tt::tt_fabric::EDMStatus::STARTED:                       return "STARTED";
        case tt::tt_fabric::EDMStatus::LOCAL_HANDSHAKE_COMPLETE:      return "LOCAL_HANDSHAKE_COMPLETE";
        case tt::tt_fabric::EDMStatus::REMOTE_HANDSHAKE_COMPLETE:     return "REMOTE_HANDSHAKE_COMPLETE";
        case tt::tt_fabric::EDMStatus::READY_FOR_TRAFFIC:             return "READY_FOR_TRAFFIC";
        case tt::tt_fabric::EDMStatus::TERMINATED:                    return "TERMINATED";
        case tt::tt_fabric::EDMStatus::TXQ_INITIALIZED:               return "TXQ_INITIALIZED";
        case tt::tt_fabric::EDMStatus::STREAM_REG_INITIALIZED:        return "STREAM_REG_INITIALIZED";
        case tt::tt_fabric::EDMStatus::DOWNSTREAM_EDM_SETUP_STARTED:  return "DOWNSTREAM_EDM_SETUP_STARTED";
        case tt::tt_fabric::EDMStatus::EDM_VCS_SETUP_COMPLETE:        return "EDM_VCS_SETUP_COMPLETE";
        case tt::tt_fabric::EDMStatus::WORKER_INTERFACES_INITIALIZED: return "WORKER_INTERFACES_INITIALIZED";
        case tt::tt_fabric::EDMStatus::ETHERNET_HANDSHAKE_COMPLETE:   return "ETHERNET_HANDSHAKE_COMPLETE";
        case tt::tt_fabric::EDMStatus::VCS_OPENED:                    return "VCS_OPENED";
        case tt::tt_fabric::EDMStatus::ROUTING_TABLE_INITIALIZED:     return "ROUTING_TABLE_INITIALIZED";
        case tt::tt_fabric::EDMStatus::INITIALIZATION_COMPLETE:       return "INITIALIZATION_COMPLETE";
        default: break;
    }
    if (v == static_cast<uint32_t>(EthDiagSentinel::BASE_UMD_FIRMWARE_SENTINEL)) return "(base-umd-relay)";
    if (v == static_cast<uint32_t>(EthDiagSentinel::HOST_PRE_LAUNCH_CANARY))    return "(host-pre-launch-canary)";
    if (v == static_cast<uint32_t>(EthDiagSentinel::READ_EXCEPTION))            return "(read-exception)";
    if (v == static_cast<uint32_t>(EthDiagSentinel::PHASE5B_DEADLINE_SKIPPED))  return "(deadline-skipped)";
    if (v == static_cast<uint32_t>(EthDiagSentinel::PHASE5B_READ_EXCEPTION))    return "(phase5b-read-exception)";
    return "(unknown)";
}

// Returns true iff `status` is one of the well-known EDMStatus sentinel values
// written by a live fabric ERISC router.  Any other value indicates L1 is
// corrupt or has been overwritten — the TERMINATE handshake will not complete.
inline bool is_known_edm_status(uint32_t status) {
    const char* name = edm_status_name(static_cast<tt::tt_fabric::EDMStatus>(status));
    return name[0] != '(';
}

// Returns true iff `status` is a valid, non-corrupt ERISC state that does NOT
// correspond to a live fabric router (i.e. not an EDMStatus enumerator), but is
// also NOT random/corrupt L1 garbage.  Callers should check is_known_edm_status()
// first; this function handles the complementary benign cases:
//
//   BASE_UMD_FIRMWARE_SENTINEL (0x49706550, "iPeP")
//     Written by the base UMD relay firmware once it has completed .bss init
//     and entered its polling loop.  The channel is healthy — it just rebooted
//     into base firmware and EDM has not been launched yet.
//
//   ROM postcode family (0x4970xxxx, excluding 0x49706550)
//     The ERISC BRISC ROM writes a series of intermediate postcodes during
//     boot (0x49705180 → 0x49705530 → … → 0x49706550).  The channel is
//     mid-boot and will complete; it is NOT corrupt.
//
//   HOST_PRE_LAUNCH_CANARY (0xDEADB07E)
//     Written intentionally by our own host code to router_sync_address before
//     sending the EDM launch message.  If the field still reads this value the
//     ERISC has not yet polled — the channel is not corrupt.
inline bool is_benign_erisc_state(uint32_t status) {
    if (status == static_cast<uint32_t>(EthDiagSentinel::BASE_UMD_FIRMWARE_SENTINEL)) {
        return true;
    }
    if (status == static_cast<uint32_t>(EthDiagSentinel::HOST_PRE_LAUNCH_CANARY)) {
        return true;
    }
    // ROM postcode family: 0x4970xxxx (covers all intermediate boot postcodes).
    // BASE_UMD_FIRMWARE_SENTINEL (0x49706550) is also in this range and already
    // handled above; the check is harmless but kept separate for clarity.
    constexpr uint32_t kRomPostcodeBase = 0x49700000u;
    constexpr uint32_t kRomPostcodeMask = 0xFFFF0000u;
    if ((status & kRomPostcodeMask) == kRomPostcodeBase) {
        return true;
    }
    return false;
}

}  // namespace tt::tt_metal
