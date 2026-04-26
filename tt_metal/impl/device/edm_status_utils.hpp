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
// edm_status_str(uint32_t)    — raw uint32_t → name, handles host-side sentinel
//                               values 0xDEAD5B5B and 0xDEADECE7 in addition to
//                               all EDMStatus enumerators.
// is_known_edm_status(uint32_t) — returns true iff the raw value is a recognised
//                               EDMStatus enumerator (used by probe / quiesce logic).

namespace tt::tt_metal {

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
    if (v == 0xDEAD5B5B) return "(deadline-skipped)";
    if (v == 0xDEADECE7) return "(read-exception)";
    return "(unknown)";
}

// Returns true iff `status` is one of the well-known EDMStatus sentinel values
// written by a live fabric ERISC router.  Any other value indicates L1 is
// corrupt or has been overwritten — the TERMINATE handshake will not complete.
inline bool is_known_edm_status(uint32_t status) {
    const char* name = edm_status_name(static_cast<tt::tt_fabric::EDMStatus>(status));
    return name[0] != '(';
}

}  // namespace tt::tt_metal
