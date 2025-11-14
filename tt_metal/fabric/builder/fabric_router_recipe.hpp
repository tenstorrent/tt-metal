// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <tt_stl/assert.hpp>

namespace tt::tt_fabric {

/**
 * Types of channel pools supported by the fabric router.
 * Must match the enum in fabric_static_channels_ct_args.hpp
 */
enum class FabricChannelPoolType : uint32_t {
    STATIC = 0,
    ELASTIC = 1,
    INVALID = 2,
};

/**
 * Definition of a single channel pool in the fabric router.
 */
struct ChannelPoolDefinition {
    FabricChannelPoolType type;

    // For validation: track which channels are mapped to this pool
    std::vector<size_t> mapped_sender_channels;
    std::vector<size_t> mapped_receiver_channels;

    ChannelPoolDefinition(FabricChannelPoolType type) : type(type) {}
    ChannelPoolDefinition(
        FabricChannelPoolType type,
        std::vector<size_t> mapped_sender_channels,
        std::vector<size_t> mapped_receiver_channels) :
        type(type),
        mapped_sender_channels(mapped_sender_channels),
        mapped_receiver_channels(mapped_receiver_channels) {}
};

/**
 * FabricRouterRecipe defines the complete configuration for a fabric router's channel pools.
 *
 * This includes:
 * - The number and types of channel pools
 * - Mappings from channels to pools (independently for sender and receiver channels)
 *
 * Key Constraints:
 * - Static pools: Multiple channels can map to a single static pool, but each channel
 *   gets a dedicated 1-to-1 allocation within that pool (no sharing)
 * - Elastic pools: Multiple channels can map to the same elastic pool and share memory dynamically
 *
 * The recipe validates these constraints during construction.
 */
class FabricRouterRecipe {
public:
    /**
     * Construct a recipe with explicit pool and mapping configurations.
     *
     * @param pool_definitions Vector of pool definitions
     * @param sender_channel_to_pool_index Mapping from sender channel ID to pool index
     * @param receiver_channel_to_pool_index Mapping from receiver channel ID to pool index
     * @param num_sender_channels Total number of sender channels
     * @param num_receiver_channels Total number of receiver channels
     */
    FabricRouterRecipe(
        std::vector<ChannelPoolDefinition> pool_definitions,
        std::vector<size_t> sender_channel_to_pool_index,
        std::vector<size_t> receiver_channel_to_pool_index,
        size_t num_sender_channels,
        size_t num_receiver_channels);

    /**
     * Create a default recipe with a single static pool that all channels map to.
     *
     * @param num_sender_channels Total number of sender channels
     * @param num_receiver_channels Total number of receiver channels
     */
    static FabricRouterRecipe create_default_single_static_pool_recipe(
        size_t num_sender_channels, size_t num_receiver_channels);

    // Accessors
    size_t get_num_pools() const { return pool_definitions_.size(); }
    const std::vector<ChannelPoolDefinition>& get_pool_definitions() const { return pool_definitions_; }
    const std::vector<size_t>& get_sender_channel_to_pool_index() const { return sender_channel_to_pool_index_; }
    const std::vector<size_t>& get_receiver_channel_to_pool_index() const { return receiver_channel_to_pool_index_; }

    FabricChannelPoolType get_sender_channel_pool_type(size_t channel_id) const {
        TT_FATAL(channel_id < sender_channel_to_pool_index_.size(), "Invalid sender channel ID {}", channel_id);
        size_t pool_idx = sender_channel_to_pool_index_[channel_id];
        return pool_definitions_[pool_idx].type;
    }

    FabricChannelPoolType get_receiver_channel_pool_type(size_t channel_id) const {
        TT_FATAL(channel_id < receiver_channel_to_pool_index_.size(), "Invalid receiver channel ID {}", channel_id);
        size_t pool_idx = receiver_channel_to_pool_index_[channel_id];
        return pool_definitions_[pool_idx].type;
    }

    size_t get_num_sender_channels() const { return sender_channel_to_pool_index_.size(); }
    size_t get_num_receiver_channels() const { return receiver_channel_to_pool_index_.size(); }

    void expand_static_channels(size_t num_sender_channels) {
        auto expand = [this](size_t offset, std::vector<size_t>& channel_to_pool_index) {
            std::vector<size_t> channel_to_pool_index_expanded;
            size_t num_inserted = offset;
            for (size_t i = 0; i < channel_to_pool_index.size(); i++) {
                auto mapped_pool_idx = channel_to_pool_index[i];
                if (this->pool_definitions_[mapped_pool_idx].type == FabricChannelPoolType::STATIC) {
                    auto new_index = mapped_pool_idx + num_inserted;
                    num_inserted++;
                    channel_to_pool_index_expanded.push_back(new_index);
                }
            }
            return channel_to_pool_index_expanded;
        };

        auto expanded_senders = expand(0, sender_channel_to_pool_index_);
        // log_info(tt::LogFabric, "\tunexpanded_senders: {}", expanded_senders);
        // log_info(tt::LogFabric, "\texpanded_senders: {}", expanded_senders);
        sender_channel_to_pool_index_ = std::move(expanded_senders);
        for (size_t i = 1; i < sender_channel_to_pool_index_.size(); i++) {
            TT_FATAL(
                sender_channel_to_pool_index_[i] != 0,
                "Sender channel {} maps to invalid pool index {}",
                i,
                sender_channel_to_pool_index_[i]);
        }
        // This will not work for elastic channels
        auto expanded_receivers = expand(num_sender_channels, receiver_channel_to_pool_index_);
        // log_info(tt::LogFabric, "\tunexpanded_receivers: {}", expanded_receivers);
        // log_info(tt::LogFabric, "\texpanded_receivers: {}", expanded_receivers);
        receiver_channel_to_pool_index_ = std::move(expanded_receivers);

        std::vector<ChannelPoolDefinition> expanded_pool_definitions;
        // Need to expand the pool definitions as well
        size_t expanded_pool_index = 0;
        for (size_t i = 0; i < pool_definitions_.size(); i++) {
            auto& pool_definition = pool_definitions_[i];
            bool is_static = pool_definition.type == FabricChannelPoolType::STATIC;
            if (is_static) {
                for (size_t j = 0; j < pool_definition.mapped_sender_channels.size(); j++) {
                    auto new_pool_def = ChannelPoolDefinition(FabricChannelPoolType::STATIC, {expanded_pool_index}, {});
                    expanded_pool_definitions.push_back(new_pool_def);
                    expanded_pool_index++;
                }
                for (size_t j = 0; j < pool_definition.mapped_receiver_channels.size(); j++) {
                    auto new_pool_def = ChannelPoolDefinition(FabricChannelPoolType::STATIC, {}, {expanded_pool_index});
                    expanded_pool_definitions.push_back(new_pool_def);
                    expanded_pool_index++;
                }

            } else {
                expanded_pool_definitions.push_back(pool_definition);
                expanded_pool_index++;
            }
        }

        // for (size_t i = 0; i < expanded_pool_definitions.size(); i++) {
        //     log_info(tt::LogFabric, "Expanded pool definition {}: type={}, mapped_sender_channels={},
        //     mapped_receiver_channels={}", i, expanded_pool_definitions[i].type,
        //     expanded_pool_definitions[i].mapped_sender_channels,
        //     expanded_pool_definitions[i].mapped_receiver_channels);
        // }
        pool_definitions_ = std::move(expanded_pool_definitions);
    }

    /*
     * This is a hack to insert empty pool definitions at the front of the pool definitions to
     * keep indexing consistent for remote channel views. This is super hacky.
     */
    void insert_empty_pool_definitions_front(size_t num_pools) {
        for (size_t i = 0; i < num_pools; i++) {
            pool_definitions_.insert(
                pool_definitions_.begin(), ChannelPoolDefinition(FabricChannelPoolType::INVALID, {}, {}));
        }
    }

private:
    /**
     * Validate the recipe configuration.
     * Ensures all constraints are met:
     * - All channels are mapped to valid pool indices
     * - Pool indices are in valid range
     * - No invalid configurations
     */
    void validate() const;

    std::vector<ChannelPoolDefinition> pool_definitions_;
    std::vector<size_t> sender_channel_to_pool_index_;
    std::vector<size_t> receiver_channel_to_pool_index_;
};

}  // namespace tt::tt_fabric
