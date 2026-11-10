// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The names of the registered enums enchantum can't reach. Enchantum finds enumerators by searching a range of values
// (±256 by default), and EDMStatus's start at 0xA0B0C0D0, so they're listed here once.

#pragma once

#include <optional>
#include <string>

#include "tt_metal/fabric/debug/visualizer/manifest/fabric_manifest_names.hpp"
#include "tt_metal/hw/inc/hostdev/fabric_edm_packet_header.hpp"

namespace tt::tt_fabric::decode {

// The EDMStatus enumerator `status` names, in lower case, or null when it names none. Each case spells its name from
// the enumerator itself, so no name is typed by hand, and the switch has no default, so -Wswitch fails the build when
// an enumerator is added.
inline std::optional<std::string> find_edm_status_name(EDMStatus status) {
#define FABRIC_DECODE_EDM_STATUS_NAME(enumerator) \
    case EDMStatus::enumerator: return manifest::lower_name(#enumerator)
    switch (status) {
        FABRIC_DECODE_EDM_STATUS_NAME(STARTED);
        FABRIC_DECODE_EDM_STATUS_NAME(REMOTE_HANDSHAKE_COMPLETE);
        FABRIC_DECODE_EDM_STATUS_NAME(LOCAL_HANDSHAKE_COMPLETE);
        FABRIC_DECODE_EDM_STATUS_NAME(READY_FOR_TRAFFIC);
        FABRIC_DECODE_EDM_STATUS_NAME(TERMINATED);
        FABRIC_DECODE_EDM_STATUS_NAME(INITIALIZATION_STARTED);
        FABRIC_DECODE_EDM_STATUS_NAME(TXQ_INITIALIZED);
        FABRIC_DECODE_EDM_STATUS_NAME(STREAM_REG_INITIALIZED);
        FABRIC_DECODE_EDM_STATUS_NAME(DOWNSTREAM_EDM_SETUP_STARTED);
        FABRIC_DECODE_EDM_STATUS_NAME(EDM_VCS_SETUP_COMPLETE);
        FABRIC_DECODE_EDM_STATUS_NAME(WORKER_INTERFACES_INITIALIZED);
        FABRIC_DECODE_EDM_STATUS_NAME(ETHERNET_HANDSHAKE_COMPLETE);
        FABRIC_DECODE_EDM_STATUS_NAME(VCS_OPENED);
        FABRIC_DECODE_EDM_STATUS_NAME(ROUTING_TABLE_INITIALIZED);
        FABRIC_DECODE_EDM_STATUS_NAME(INITIALIZATION_COMPLETE);
    }
#undef FABRIC_DECODE_EDM_STATUS_NAME
    return std::nullopt;
}

}  // namespace tt::tt_fabric::decode
