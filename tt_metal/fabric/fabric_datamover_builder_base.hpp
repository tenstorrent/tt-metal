// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <hostdevcommon/fabric_common.h>

namespace tt::tt_fabric {

struct SenderWorkerAdapterSpec;

// Base class for all fabric datamover builders
class FabricDatamoverBuilderBase {
public:
    virtual ~FabricDatamoverBuilderBase() = default;

    size_t get_noc_x() const { return noc_x_; }
    size_t get_noc_y() const { return noc_y_; }
    eth_chan_directions get_direction() const { return direction_; }
    virtual SenderWorkerAdapterSpec build_connection_to_fabric_channel(uint32_t channel_id) const = 0;

    void set_sender_channel_injection_flags(std::vector<bool>&& flags) {
        this->sender_channel_is_traffic_injection_channel_array = std::move(flags);
    }

protected:
    FabricDatamoverBuilderBase(size_t noc_x, size_t noc_y, eth_chan_directions direction)
        : noc_x_(noc_x), noc_y_(noc_y), direction_(direction) {}

    size_t noc_x_;
    size_t noc_y_;
    eth_chan_directions direction_;
    std::vector<bool> sender_channel_is_traffic_injection_channel_array;
};

}  // namespace tt::tt_fabric
