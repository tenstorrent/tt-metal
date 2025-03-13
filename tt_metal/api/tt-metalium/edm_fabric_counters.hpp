// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_fabric {

struct EdmFabricReceiverChannelCounters {
    uint32_t n_pkts_processed = 0;
    uint32_t n_pkts_fwded = 0;
    uint32_t n_pkts_written_locally = 0;
    uint32_t n_pkts_rx_acked = 0;
    uint32_t n_pkts_completion_acked = 0;

    uint32_t n_fabric_mcast_noc_atomic_processed = 0;
    uint32_t n_fabric_mcast_noc_write_processed = 0;
    uint32_t n_fabric_unicast_noc_atomic_processed = 0;
    uint32_t n_fabric_unicast_noc_write_processed = 0;

    EdmFabricReceiverChannelCounters() = default;
};
static constexpr uint32_t receiver_channel_counters_l1_size = sizeof(EdmFabricReceiverChannelCounters);

struct EdmFabricSenderChannelCounters {
    uint32_t n_lifetime_connections = 0;

    uint32_t n_lifetime_pkts_received = 0;
    uint32_t n_lifetime_pkts_fwded = 0;
    uint32_t n_lifetime_pkts_acked = 0;
    uint32_t n_lifetime_pkts_complete = 0;

    uint32_t n_lifetime_fabric_mcast_noc_atomic_pkts = 0;
    uint32_t n_lifetime_fabric_mcast_noc_write_pkts = 0;
    uint32_t n_lifetime_fabric_unicast_noc_atomic_pkts = 0;
    uint32_t n_lifetime_fabric_unicast_noc_write_pkts = 0;

    uint32_t n_connection_pkts_received = 0;
    uint32_t n_connection_pkts_fwded = 0;
    uint32_t n_connection_pkts_acked = 0;
    uint32_t n_connection_pkts_complete = 0;

    uint32_t n_connection_fabric_mcast_noc_atomic_pkts = 0;
    uint32_t n_connection_fabric_mcast_noc_write_pkts = 0;
    uint32_t n_connection_fabric_unicast_noc_atomic_pkts = 0;
    uint32_t n_connection_fabric_unicast_noc_write_pkts = 0;

    void add_connection() volatile {
        this->n_lifetime_connections++;
        this->reset_connection_counters();
    }

    void add_pkt_received() volatile {
        this->n_lifetime_pkts_received++;
        this->n_connection_pkts_received++;
    }

    void add_pkt_sent() volatile {
        this->n_lifetime_pkts_fwded++;
        this->n_connection_pkts_fwded++;
    }

    void reset_connection_counters() volatile {
        this->n_connection_pkts_received = 0;
        this->n_connection_pkts_fwded = 0;
        this->n_connection_pkts_acked = 0;
        this->n_connection_pkts_complete = 0;

        this->n_connection_fabric_mcast_noc_atomic_pkts = 0;
        this->n_connection_fabric_mcast_noc_write_pkts = 0;
        this->n_connection_fabric_unicast_noc_atomic_pkts = 0;
        this->n_connection_fabric_unicast_noc_write_pkts = 0;
    }
};
static constexpr uint32_t sender_channel_counters_l1_size = sizeof(EdmFabricSenderChannelCounters);

}  // namespace tt::tt_fabric
