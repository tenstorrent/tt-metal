// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_pool_channel_allocator.hpp"
#include <tt_stl/assert.hpp>
#include "builder/fabric_static_sized_channels_allocator.hpp"
#include "builder/fabric_remote_channels_allocator.hpp"
#include "tt_metal/fabric/builder/fabric_builder_config.hpp"

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
    size_t num_used_vc0_sender_channels,
    size_t num_used_vc1_sender_channels,
    size_t num_used_receiver_channels) const {
    // Step 0: Emit special tag (replaces the tag that was in static allocator)
    ct_args.push_back(0xabcd1234);

    auto get_static_channel_allocator_num_channels =
        [](FabricChannelAllocator* allocator) -> std::pair<size_t, size_t> {
        size_t num_sender_channels = 0;
        size_t num_receiver_channels = 0;
        if (auto* static_allocator = dynamic_cast<FabricStaticSizedChannelsAllocator*>(allocator)) {
            num_sender_channels += static_allocator->get_num_sender_channels();
            num_receiver_channels += static_allocator->get_num_receiver_channels();
        } else if (auto* remote_allocator = dynamic_cast<FabricRemoteChannelsAllocator*>(allocator)) {
            num_receiver_channels += remote_allocator->get_num_receiver_channels();
        } else {
            TT_FATAL(false, "Allocator is not a static or remote allocator");
        }
        return std::make_pair(num_sender_channels, num_receiver_channels);
    };

    // Step 1: Emit number of pools
    // a bit hacky for now
    size_t num_pools = 0;
    for (const auto& pool_allocator : pool_allocators_) {
        if (implements_static_channel_allocator(pool_allocator.get())) {
            auto [num_sender_channels, num_receiver_channels] =
                get_static_channel_allocator_num_channels(pool_allocator.get());
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
        pool_allocator->emit_ct_args(ct_args);
    }

    // Emit the sender channel to pool index
    TT_FATAL(
        pool_allocators_.size() == 1,
        "Multi-pool channel allocator with channel-to-pool mapping currently only supports a single pool. Got {} "
        "pools.",
        pool_allocators_.size());

    for (const auto& pool_allocator : pool_allocators_) {
        TT_FATAL(
            dynamic_cast<FabricStaticSizedChannelsAllocator*>(pool_allocator.get()) ||
                dynamic_cast<FabricRemoteChannelsAllocator*>(pool_allocator.get()),
            "Non static sized channel allocators not supported in the code below yet");
    }

    // Calculate total sender channels from the first (and only) pool allocator
    auto [total_sender_channels, total_receiver_channels] =
        get_static_channel_allocator_num_channels(pool_allocators_[0].get());

    // Combine VC0 and VC1 counts to get total used sender channels
    size_t num_used_sender_channels = num_used_vc0_sender_channels + num_used_vc1_sender_channels;

    // Determine if we need to skip pool mappings due to unused channels
    // This occurs when the allocator has more channels than the router uses
    // For example to support Z dimension in 2D fabric, we need 9 sender channels over
    // 2 VCs. Z routers can potentially use all 9 channels. Mesh routers on a chip
    // with Z routers uses 8 channels. Mesh routers on a chip without Z routers use
    // 7 channels. But because the allocator is common for all routers, we need to skip
    // the unused channels on mesh routers.
    size_t num_unused_channels = total_sender_channels - num_used_sender_channels;
    bool has_unused_channels = (num_unused_channels > 0) && num_used_sender_channels > 0;

    // Emit sender channel to pool index mapping
    // If there are unused channels, we need to skip their pool mappings and add padding
    if (has_unused_channels) {
        // Map used channels to their pools, skipping the unused channel pools
        // If we have to skip, pool at index (num_used_vc0_sender_channels) will be skipped.

        for (size_t i = 0; i < num_used_sender_channels; ++i) {
            if (i < num_used_vc0_sender_channels) {
                // VC0 channels map directly (0 to num_used_vc0_sender_channels-1 → pools 0 to
                // num_used_vc0_sender_channels-1)
                ct_args.push_back(static_cast<uint32_t>(i));
            } else {
                // VC1 channels skip the unused VC0 channel(s)
                // e.g., channels 4-7 → pools 5-8 (skipping pool 4)
                ct_args.push_back(static_cast<uint32_t>(i + 1));
            }
        }
        // Add padding entries for the unused channels
        for (size_t i = 0; i < num_unused_channels; ++i) {
            ct_args.push_back(static_cast<uint32_t>(0));  // Dummy padding
        }
    } else {
        // No unused channels - direct mapping
        for (size_t i = 0; i < num_used_sender_channels; ++i) {
            ct_args.push_back(static_cast<uint32_t>(i));
        }
    }

    // Emit receiver channel to pool index mapping
    // Receiver pools start after sender pools
    size_t receiver_pool_base =
        has_unused_channels ? (num_used_sender_channels + num_unused_channels) : num_used_sender_channels;

    for (size_t i = 0; i < num_used_receiver_channels; ++i) {
        ct_args.push_back(static_cast<uint32_t>(i + receiver_pool_base));
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
