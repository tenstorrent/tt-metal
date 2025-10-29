// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

namespace tt::tt_fabric {

/**
 * Channel traits system for extracting compile-time properties from channel types.
 * This allows functions to query channel properties without requiring explicit
 * template parameters for buffer counts.
 */

// Forward declarations for supported channel types
template <uint8_t EDM_NUM_BUFFER_SLOTS>
struct RouterStaticSizedChannelWriterAdapter;

template <uint8_t SLOTS_PER_CHUNK, uint16_t CHUNK_SIZE_BYTES>
struct RouterElasticChannelWriterAdapter;

/**
 * Primary template for channel traits.
 * Specialized for each channel type to expose their properties.
 */
template <typename T, typename Enable = void>
struct ChannelTraits;

/**
 * Specialization for RouterStaticSizedChannelWriterAdapter
 */
template <uint8_t NUM_BUFFERS>
struct ChannelTraits<RouterStaticSizedChannelWriterAdapter<NUM_BUFFERS>> {
    static constexpr uint8_t num_buffers = NUM_BUFFERS;
    static constexpr bool is_elastic = false;
    using channel_type = RouterStaticSizedChannelWriterAdapter<NUM_BUFFERS>;
};

/**
 * Specialization for RouterElasticChannelWriterAdapter
 */
template <uint8_t SLOTS_PER_CHUNK, uint16_t CHUNK_SIZE_BYTES>
struct ChannelTraits<RouterElasticChannelWriterAdapter<SLOTS_PER_CHUNK, CHUNK_SIZE_BYTES>> {
    static constexpr uint8_t num_buffers = SLOTS_PER_CHUNK;
    static constexpr bool is_elastic = true;
    using channel_type = RouterElasticChannelWriterAdapter<SLOTS_PER_CHUNK, CHUNK_SIZE_BYTES>;
};

/**
 * Helper to detect if a type has channel traits
 */
template <typename T, typename = void>
struct has_channel_traits : std::false_type {};

template <typename T>
struct has_channel_traits<T, std::void_t<decltype(ChannelTraits<T>::num_buffers)>> : std::true_type {};

/**
 * Helper function to get num_buffers from a channel type at compile time
 */
template <typename ChannelT>
constexpr uint8_t get_num_buffers() {
    return ChannelTraits<ChannelT>::num_buffers;
}

/**
 * Helper function to check if a channel type is elastic
 */
template <typename ChannelT>
constexpr bool is_elastic_channel() {
    return ChannelTraits<ChannelT>::is_elastic;
}

}  // namespace tt::tt_fabric
