// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compile_time_args.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/compile_time_arg_tmp.hpp"

// Per-channel buffer allocation data, read directly from compile-time args.
// Each entry contains 4 args: base_address, num_slots, remote_address, remote_num_slots.
template <size_t CT_ARG_IDX_BASE>
struct ChannelBufferEntry {
    static constexpr size_t base_address = get_compile_time_arg_val(CT_ARG_IDX_BASE);
    static constexpr size_t num_slots = get_compile_time_arg_val(CT_ARG_IDX_BASE + 1);
    static constexpr size_t remote_address = get_compile_time_arg_val(CT_ARG_IDX_BASE + 2);
    static constexpr size_t remote_num_slots = get_compile_time_arg_val(CT_ARG_IDX_BASE + 3);
    static constexpr size_t ARGS_PER_ENTRY = 4;
};

// Parses a block of channel allocation CT args emitted by the host.
//
// CT arg layout:
//   [0xabcd1234]                                         <- alignment tag
//   [num_entries]                                        <- total channel entries
//   [entry_0: base, slots, remote_base, remote_slots]   <- 4 args per entry
//   [entry_1: ...]
//   ...
//   [sender_channel_to_entry_idx[NumSenderChannels]]     <- index mapping
//   [receiver_channel_to_entry_idx[NumReceiverChannels]] <- index mapping
template <size_t CT_ARG_IDX_BASE, size_t NumSenderChannels, size_t NumReceiverChannels>
struct ChannelAllocations {
    static constexpr size_t special_tag_idx = CT_ARG_IDX_BASE;
    static constexpr size_t special_tag = get_compile_time_arg_val(special_tag_idx);
    static_assert(
        special_tag == 0xabcd1234,
        "Special tag not found. This implies some arguments were misaligned between host and device. Double check the "
        "CT args.");

    static constexpr size_t num_entries = get_compile_time_arg_val(CT_ARG_IDX_BASE + 1);
    static constexpr size_t ARGS_PER_ENTRY = 4;
    static constexpr size_t entries_base_idx = CT_ARG_IDX_BASE + 2;

    // Access a specific entry by index
    template <size_t EntryIdx>
    using Entry = ChannelBufferEntry<entries_base_idx + EntryIdx * ARGS_PER_ENTRY>;

    // Channel-to-entry index mappings
    static constexpr size_t mappings_base_idx = entries_base_idx + num_entries * ARGS_PER_ENTRY;
    static constexpr std::array<size_t, NumSenderChannels> sender_channel_to_entry_index =
        fill_array_with_next_n_args<size_t, mappings_base_idx, NumSenderChannels>();
    static constexpr std::array<size_t, NumReceiverChannels> receiver_channel_to_entry_index =
        fill_array_with_next_n_args<size_t, mappings_base_idx + NumSenderChannels, NumReceiverChannels>();

    static constexpr size_t GET_NUM_ARGS_CONSUMED() {
        return 2 + num_entries * ARGS_PER_ENTRY + NumSenderChannels + NumReceiverChannels;
    }
};
