// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_pool_channel_allocator.hpp"
#include <tt_stl/assert.hpp>
#include "builder/fabric_static_sized_channels_allocator.hpp"
#include "builder/fabric_remote_channels_allocator.hpp"

namespace tt::tt_fabric {

static bool implements_static_channel_allocator(FabricChannelAllocator* allocator) {
    return dynamic_cast<FabricStaticSizedChannelsAllocator*>(allocator) != nullptr || dynamic_cast<FabricRemoteChannelsAllocator*>(allocator) != nullptr;
}

MultiPoolChannelAllocator::MultiPoolChannelAllocator(
    std::vector<std::shared_ptr<FabricChannelAllocator>> pool_allocators,
    std::vector<FabricChannelPoolType> pool_types) :
    pool_allocators_(std::move(pool_allocators)), pool_types_(std::move(pool_types)) {
    TT_FATAL(!pool_allocators_.empty(), "MultiPoolChannelAllocator requires at least one pool allocator");
    TT_FATAL(
        pool_allocators_.size() == pool_types_.size(),
        "Number of pool allocators ({}) must match number of pool types ({})",
        pool_allocators_.size(),
        pool_types_.size());

    // Validate that all pool allocators are non-null
    for (size_t i = 0; i < pool_allocators_.size(); ++i) {
        TT_FATAL(pool_allocators_[i] != nullptr, "Pool allocator at index {} is null", i);
    }
}

void MultiPoolChannelAllocator::emit_ct_args(
    std::vector<uint32_t>& ct_args,
    size_t num_fwd_paths,
    size_t num_used_sender_channels,
    size_t num_used_receiver_channels) const {
    // Step 0: Emit special tag (replaces the tag that was in static allocator)
    ct_args.push_back(0xabcd1234);

    auto get_static_channel_allocator_num_channels =
        [](FabricChannelAllocator* allocator) -> std::pair<size_t, size_t> {
        size_t num_sender_channels = 0;
        size_t num_receiver_channels = 0;
        if (auto static_allocator = dynamic_cast<FabricStaticSizedChannelsAllocator*>(allocator)) {
            num_sender_channels += static_allocator->get_num_sender_channels();
            num_receiver_channels += static_allocator->get_num_receiver_channels();
        } else if (auto remote_allocator = dynamic_cast<FabricRemoteChannelsAllocator*>(allocator)) {
            num_receiver_channels += remote_allocator->get_num_receiver_channels();
        } else {
            TT_FATAL(false, "Allocator is not a static or remote allocator");
        }
        return std::make_pair(num_sender_channels, num_receiver_channels);
    };

    // Step 1: Emit number of pools
    // a bit hacky for now
    size_t num_pools = 0;
    for (size_t i = 0; i < pool_allocators_.size(); ++i) {
        if (implements_static_channel_allocator(pool_allocators_[i].get())) {
            auto [num_sender_channels, num_receiver_channels] =
                get_static_channel_allocator_num_channels(pool_allocators_[i].get());
            num_pools += num_sender_channels + num_receiver_channels;
        } else {
            num_pools++;
        }
    }
    ct_args.push_back(static_cast<uint32_t>(num_pools));

    // Step 2: Emit array of pool types
    for (size_t i = 0; i < pool_types_.size(); ++i) {
        FabricChannelPoolType pool_type = pool_types_[i];
        if (pool_type == FabricChannelPoolType::STATIC) {
            auto [num_sender_channels, num_receiver_channels] =
                get_static_channel_allocator_num_channels(pool_allocators_[i].get());
            size_t num_channels = num_sender_channels + num_receiver_channels;
            for (size_t j = 0; j < num_channels; ++j) {
                ct_args.push_back(static_cast<uint32_t>(FabricChannelPoolType::STATIC));
            }
        } else {
            ct_args.push_back(static_cast<uint32_t>(pool_type));
        }
    }

    // Step 3: Emit individual pool CT args
    // Each pool allocator emits its own compile-time arguments (WITHOUT special tags)
    for (const auto& pool_allocator : pool_allocators_) {
        pool_allocator->emit_ct_args(ct_args, num_fwd_paths, num_used_sender_channels, num_used_receiver_channels);
    }

    // Emit the sender channel to pool index
    for (size_t i = 0; i < pool_allocators_.size(); ++i) {
        TT_FATAL(
            dynamic_cast<FabricStaticSizedChannelsAllocator*>(pool_allocators_[i].get()) ||
                dynamic_cast<FabricRemoteChannelsAllocator*>(pool_allocators_[i].get()),
            "Non static sized channel allocators not supported in the code below yet");
    }
    for (size_t i = 0; i < num_used_sender_channels; ++i) {
        ct_args.push_back(static_cast<uint32_t>(i));
    }
    for (size_t i = 0; i < num_used_receiver_channels; ++i) {
        ct_args.push_back(static_cast<uint32_t>(i + num_used_sender_channels));
    }

    // Emit the sender channel to pool type -- NVM

    // Emit the receiver channel to pool index

    // Emit the receiver channel to pool type -- NVM
}

std::shared_ptr<FabricChannelAllocator> MultiPoolChannelAllocator::get_pool(size_t pool_index) const {
    TT_FATAL(
        pool_index < pool_allocators_.size(),
        "Pool index {} out of range (max {})",
        pool_index,
        pool_allocators_.size() - 1);
    return pool_allocators_[pool_index];
}

FabricChannelPoolType MultiPoolChannelAllocator::get_pool_type(size_t pool_index) const {
    TT_FATAL(
        pool_index < pool_types_.size(), "Pool index {} out of range (max {})", pool_index, pool_types_.size() - 1);
    return pool_types_[pool_index];
}

}  // namespace tt::tt_fabric
