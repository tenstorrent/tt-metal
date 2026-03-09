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
    using channel_type = RouterStaticSizedChannelWriterAdapter<NUM_BUFFERS>;
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

}  // namespace tt::tt_fabric
