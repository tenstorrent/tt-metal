// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compile_time_args.h"
#include "dataflow_api.h"

#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"

#include <array>
#include <utility>

namespace tt::tt_fabric::elastic_channels {

template <size_t SLOT_SIZE_BYTES, size_t N_SLOTS_PER_CHUNK, size_t N_CHUNKS>
constexpr bool validate_elastic_channel_chunk_addresses(const std::array<size_t, N_CHUNKS>& chunk_base_addresses) {
    constexpr size_t chunk_size = N_SLOTS_PER_CHUNK * SLOT_SIZE_BYTES;
    for (size_t i = 0; i < N_CHUNKS; ++i) {
        for (size_t j = i + 1; j < N_CHUNKS; ++j) {
            auto start1 = chunk_base_addresses[i];
            auto end1 = start1 + chunk_size;
            auto start2 = chunk_base_addresses[j];
            auto end2 = start2 + chunk_size;
            if (end1 > start2 && end2 > start1) {
                return false;
            }
        }
    }
    return true;
}

template <size_t NEXT_CT_ARG_IDX, size_t SLOT_SIZE_BYTES>
struct RouterElasticChannelsCtArgs {
    static_assert(get_compile_time_arg_val(NEXT_CT_ARG_IDX) == 0xe1a571cFF, "NEXT_CT_ARG_IDX must be 0");

    static constexpr size_t N_CHUNKS = get_compile_time_arg_val(NEXT_CT_ARG_IDX + 1);
    static constexpr size_t N_SLOTS_PER_CHUNK = get_compile_time_arg_val(NEXT_CT_ARG_IDX + 2);

    static constexpr std::array<size_t, N_CHUNKS> CHUNK_BASE_ADDRESSES =
        fill_array_with_next_n_args<size_t, NEXT_CT_ARG_IDX + 3, N_CHUNKS>();

    // This helps report overlapped addresses in the event that `CHUNK_BASE_ADDRESSES` has overlapping addresses
    template <bool NoOverlap, size_t... Addresses>
    struct AssertNoOverlapOrPrintAddresses {};
    template <size_t... Addresses>
    struct AssertNoOverlapOrPrintAddresses<true, Addresses...> {
        using type = int;
    };

    template <std::size_t... I>
    static constexpr auto get_printer(std::index_sequence<I...>) -> AssertNoOverlapOrPrintAddresses<
        validate_elastic_channel_chunk_addresses<SLOT_SIZE_BYTES, N_SLOTS_PER_CHUNK, N_CHUNKS>(CHUNK_BASE_ADDRESSES),
        CHUNK_BASE_ADDRESSES[I]...>;

    using printer_t = decltype(get_printer(std::make_index_sequence<N_CHUNKS>{}));
    using assert_t = typename printer_t::type;
};

};  // namespace tt::tt_fabric::elastic_channels
