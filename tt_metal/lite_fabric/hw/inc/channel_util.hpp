// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <type_traits>

#include "tt_metal/lite_fabric/hw/inc/header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"

namespace lite_fabric {

template <typename T>
struct get_num_buffers;

template <uint8_t NumBuffers>
struct get_num_buffers<tt::tt_fabric::EthChannelBuffer<lite_fabric::FabricLiteHeader, NumBuffers>> {
    static constexpr uint8_t value = NumBuffers;
};

template <size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type init_receiver_headers_impl(std::tuple<Tp...>& t) {}

template <size_t I = 0, typename... Tp>
    inline typename std::enable_if < I<sizeof...(Tp), void>::type init_receiver_headers_impl(std::tuple<Tp...>& t) {
    using ChannelType = std::tuple_element_t<I, std::tuple<Tp...>>;
    for (uint8_t i = 0; i < get_num_buffers<ChannelType>::value; i++) {
        auto buffer_header =
            std::get<I>(t).template get_packet_header<lite_fabric::FabricLiteHeader>(tt::tt_fabric::BufferIndex{i});
        buffer_header->command_fields.noc_read.event = 0xdeadbeef;
    }
    init_receiver_headers_impl<I + 1, Tp...>(t);
}

// Dirty header addresses to ensure read events are not from previous runs
template <auto& ChannelBuffers>
inline void init_receiver_headers(
    tt::tt_fabric::EthChannelBufferTuple<lite_fabric::FabricLiteHeader, ChannelBuffers>& channels) {
    init_receiver_headers_impl(channels.channel_buffers);
}

}  // namespace lite_fabric
