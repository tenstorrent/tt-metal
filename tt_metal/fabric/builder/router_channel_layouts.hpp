// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <hostdevcommon/fabric_common.h>  // chan_id_t
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

namespace tt::tt_metal::distributed::multihost {
class DistributedContext;
}  // namespace tt::tt_metal::distributed::multihost

namespace tt::tt_fabric {

// A router's channel buffers: each channel's first slot address and slot count, by VC.
struct RouterChannelLayout {
    struct Channel {
        uint32_t base_address = 0;
        uint32_t num_slots = 0;
    };
    std::array<std::vector<Channel>, builder_config::MAX_NUM_VCS> senders;
    std::array<std::vector<Channel>, builder_config::MAX_NUM_VCS> receivers;
};

// Every router's channel layout, by fabric node and Ethernet channel. Device builds publish in parallel; the exchange
// then gives each host the receivers of the routers across its links, before any router kernel is emitted.
class RouterChannelLayouts {
public:
    using Key = std::pair<FabricNodeId, chan_id_t>;
    using Map = std::map<Key, RouterChannelLayout>;

    // One router as it crosses hosts: its key and its receivers, all the far end needs.
    struct Record {
        uint32_t mesh_id = 0;
        uint32_t chip_id = 0;
        uint32_t eth_chan = 0;
        uint32_t num_receivers = 0;
        struct Receiver {
            uint32_t vc = 0;
            uint32_t base_address = 0;
            uint32_t num_slots = 0;
        };
        std::array<Receiver, builder_config::num_max_receiver_channels> receivers{};
    };
    static_assert(std::is_trivially_copyable_v<Record>);

    // What each rank says about its launch, gathered before the records. num_records is filled in by exchange().
    struct LaunchHeader {
        uint32_t num_records = 0;
        uint32_t status = 0;
        uint32_t fabric_config = 0;
        uint32_t launch_generation = 0;  // set by exchange(): this registry's launch count
    };
    static_assert(std::is_trivially_copyable_v<LaunchHeader>);
    // Sees every rank's header, in rank order, before the records gather; throws to stop the launch.
    using HeaderCheck = std::function<void(const std::vector<LaunchHeader>&)>;

    // Start of each fabric launch.
    void begin_launch();
    void publish(FabricNodeId node, chan_id_t eth_chan, const RouterChannelLayout& layout);
    // Collective over `context`: every rank calls it once per launch, after its devices have published. A rank with a
    // nonzero status sends no records; `check` decides what that means.
    void exchange(
        const tt::tt_metal::distributed::multihost::DistributedContext& context,
        LaunchHeader header,
        const HeaderCheck& check);
    const RouterChannelLayout& get(FabricNodeId node, chan_id_t eth_chan) const;
    // nullptr when no layout was published for the router.
    const RouterChannelLayout* find(FabricNodeId node, chan_id_t eth_chan) const;

    // The two halves of the exchange, without the collectives.
    static std::vector<Record> pack(const Map& layouts);
    // `gathered` is every rank's records, each padded to the same stride; num_records is each rank's real count.
    // `layouts` holds this rank's own routers; a router built by two ranks is fatal.
    static void merge(
        Map& layouts, const std::vector<Record>& gathered, const std::vector<uint32_t>& num_records, size_t this_rank);

private:
    mutable std::mutex mutex_;
    Map layouts_;
    uint32_t launch_generation_ = 0;
};

}  // namespace tt::tt_fabric
