// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include <tuple>

#include "dataflow_api.h"
#include "stream_interface.h"
#include "tests/tt_metal/tt_metal/test_kernels/dataflow/streams/stream_io_kernel_helpers.hpp"
#include "tt_metal/hw/inc/wormhole/noc/noc_overlay_parameters.h"

// THESE TWO FUNCTIONS WERE ONLY VALID FOR WORMHOLE_B0 AND MAY NOT WORK WITH BLACKHOLE!!!
// STREAM_RECEIVER_ENDPOINT_MULTI_TILE_CLEAR_REG_INDEX is aliased to STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX for
// whb0
inline bool is_stream_receiver_endpoint_tile_clearing_finished(uint32_t stream_id) {
    return (NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX) == 0);
}
inline void stream_receiver_endpoint_tiles_clear_b0(uint32_t stream_id, uint32_t num_tiles) {
    uint32_t clr_val = num_tiles;
    clr_val *= 2;
    clr_val = (~clr_val) + 1;
    NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX, clr_val);
}
//////////////////////////////////////////////////////////////////////////////////////////

uint32_t get_receiver_stream_config_reg(uint32_t data_noc_id, uint32_t update_noc, bool drain_after_phase_send) {
    uint32_t stream_cfg_reg = 0;
    bool next_phase_src_dest_change = drain_after_phase_send ? 1 : 0;
    stream_cfg_reg |= STREAM_CFG(INCOMING_DATA_NOC, data_noc_id) | STREAM_CFG(REMOTE_SRC_UPDATE_NOC, update_noc) |
                      STREAM_CFG(RECEIVER_ENDPOINT, 1) | STREAM_CFG(REMOTE_SOURCE, 1) |
                      STREAM_CFG(NEXT_PHASE_SRC_CHANGE, next_phase_src_dest_change) |
                      STREAM_CFG(NEXT_PHASE_DEST_CHANGE, next_phase_src_dest_change) |
                      STREAM_CFG(PHASE_AUTO_ADVANCE, 0) | STREAM_CFG(DATA_AUTO_SEND, 0) |
                      STREAM_CFG(REG_UPDATE_VC_REG, 1);

    return stream_cfg_reg;
}

FORCE_INLINE bool messages_are_available(uint32_t stream_id, stream_state_t &stream_state) {
    uint32_t wrptr = NOC_STREAM_READ_REG(stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX);
    uint32_t rdptr = NOC_STREAM_READ_REG(stream_id, STREAM_MSG_INFO_PTR_REG_INDEX);
    uint32_t internal_rdptr = stream_state.local_msg_info_ptr >> 4;
    bool messages_available = internal_rdptr < wrptr;
    return messages_available;
}

FORCE_INLINE void flush_message_from_stream_buffer(
    uint32_t stream_id, stream_state_t &stream_state, uint32_t msg_size_bytes) {
    stream_receiver_endpoint_tiles_clear_b0(stream_id, 1);
    while (!is_stream_receiver_endpoint_tile_clearing_finished(stream_id)) {
        asm volatile("");
    }
}

FORCE_INLINE uint32_t
get_next_available_stream_message_size_in_bytes(stream_state_t &stream_state, uint32_t stream_id) {
    uint32_t msg_info_byte_ptr = stream_state.local_msg_info_ptr;
    uint32_t msg_size_bytes = *reinterpret_cast<volatile uint32_t *>(msg_info_byte_ptr) << 4;
    ASSERT(msg_size_bytes > 0);
    return msg_size_bytes;
}

FORCE_INLINE std::tuple<uint32_t, uint32_t> get_next_message_info(uint32_t stream_id, stream_state_t &stream_state) {
    uint32_t rdptr_offset = NOC_STREAM_READ_REG(stream_id, STREAM_RD_PTR_REG_INDEX) << 4;
    uint32_t addr = rdptr_offset + stream_state.local_data_buffer_base_address;
    ASSERT((rdptr_offset & 0xF) == 0);
    ASSERT((addr & 0xF) == 0);
    return {addr, get_next_available_stream_message_size_in_bytes(stream_state, stream_id)};
}

FORCE_INLINE void advance_stream_state_struct(
    uint32_t stream_id, stream_state_t &stream_state, uint32_t msg_size_bytes) {
    uint32_t next_offset = stream_state.local_buffer_read_offset + msg_size_bytes;
    if (next_offset >= stream_state.local_buffer_size) {
        next_offset -= stream_state.local_buffer_size;
    }
    stream_state.local_buffer_read_offset = next_offset;
    stream_state.local_msg_info_ptr += (1 << 4);
}

FORCE_INLINE void advance_phase(
    noc_endpoint_info_t const &remote_endpoint_info, stream_state_t &state, uint32_t stream_id) {
    // This is remote receiver, so it sends messages (updates) to remote source, NOT data, so it uses
    // the update noc to communicate to remote src instead of the data noc. Therefore, we need to set remote
    // src x/y based on the update noc.
    uint32_t translated_remote_noc_x = remote_endpoint_info.update_noc_id == 0
                                           ? remote_endpoint_info.noc_x
                                           : noc_size_x - 1 - remote_endpoint_info.noc_x;
    uint32_t translated_remote_noc_y = remote_endpoint_info.update_noc_id == 0
                                           ? remote_endpoint_info.noc_y
                                           : noc_size_y - 1 - remote_endpoint_info.noc_y;

    NOC_STREAM_WRITE_REG(stream_id, STREAM_CURR_PHASE_BASE_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_CURR_PHASE_REG_INDEX, ((uint32_t)state.local_phase_id));
    NOC_STREAM_WRITE_REG(stream_id, STREAM_BUF_START_REG_INDEX, ((uint32_t)state.local_buffer_base_addr) >> 4);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_BUF_SIZE_REG_INDEX, state.local_buffer_size >> 4);
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_REMOTE_SRC_REG_INDEX,
        STREAM_REMOTE_SRC(translated_remote_noc_x, translated_remote_noc_y, stream_id));
    NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_SRC_PHASE_REG_INDEX, ((uint32_t)state.remote_phase_id));

    NOC_STREAM_WRITE_REG(stream_id, STREAM_MEM_BUF_SPACE_AVAILABLE_ACK_THRESHOLD_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_MSG_INFO_PTR_REG_INDEX, ((uint32_t)state.local_msg_info_ptr) >> 4);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX, ((uint32_t)state.local_msg_info_ptr) >> 4);

    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_MISC_CFG_REG_INDEX,
        get_receiver_stream_config_reg(remote_endpoint_info.data_noc_id, remote_endpoint_info.update_noc_id, true));

    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, AUTO_CFG_HEADER(0, state.messages_per_phase, 0));
    NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_ADVANCE_REG_INDEX, 0x1);
}

FORCE_INLINE void advance_stream_to_next_message(
    noc_endpoint_info_t const &remote_endpoint_info,
    stream_state_t &state,
    uint32_t stream_id,
    uint32_t msg_size_bytes,
    phase_iterator_t &local_phase_iterator,
    phase_iterator_t &remote_phase_iterator) {
    advance_stream_state_struct(stream_id, state, msg_size_bytes);
    flush_message_from_stream_buffer(stream_id, state, msg_size_bytes);

    if (state.num_tiles_sent == state.tile_header_num_msgs - 1) {
        remote_phase_iterator.increment();
        state.remote_phase_id = remote_phase_iterator.get();
        local_phase_iterator.increment();
        state.local_phase_id = local_phase_iterator.get();
        state.num_tiles_sent = 0;
        state.local_msg_info_ptr = state.local_msg_info_ptr_base_address;

        advance_phase(remote_endpoint_info, state, stream_id);
        state.local_buffer_read_offset = 0;
    } else {
        state.num_tiles_sent++;
    }
}

FORCE_INLINE void copy_message_to_cb_blocking(
    uint32_t cb, uint32_t msg_addr, uint32_t msg_size_bytes, stream_state_t &stream_state) {
    uint32_t cb_write_addr = get_write_ptr(cb);
    uint64_t dest_noc_addr = get_noc_addr(cb_write_addr);
    ASSERT((dest_noc_addr & 0xF) == 0);
    ASSERT((msg_addr & 0xF) == 0);
    uint32_t distance_until_end =
        stream_state.local_buffer_size - (msg_addr - stream_state.local_data_buffer_base_address);
    uint32_t bytes_to_copy = std::min(distance_until_end, msg_size_bytes);

    noc_async_write(msg_addr, dest_noc_addr, bytes_to_copy);
    if (bytes_to_copy < msg_size_bytes) {
        uint32_t bytes_to_copy_second = msg_size_bytes - bytes_to_copy;
        noc_async_write(
            stream_state.local_data_buffer_base_address, dest_noc_addr + bytes_to_copy, bytes_to_copy_second);
        uint32_t num_words = bytes_to_copy_second >> 2;
    }
    noc_async_write_barrier();
}

void kernel_main() {
    uint32_t arg_idx = 0;

    uint32_t num_messages_to_forward = get_arg_val<uint32_t>(arg_idx++);

    uint32_t stream_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t stream_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t stream_buffer_size = get_arg_val<uint32_t>(arg_idx++);
    uint32_t stream_tile_header_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t stream_tile_header_max_num_messages = get_arg_val<uint32_t>(arg_idx++);

    uint32_t remote_src_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_src_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_src_noc_stream_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_src_data_noc_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_src_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_src_buffer_size_4B_words = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_src_tile_header_buffer_addr = get_arg_val<uint32_t>(arg_idx++);

    uint32_t relay_done_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t other_relay_core_to_signal_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t other_relay_core_to_signal_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t other_relay_done_semaphore = get_arg_val<uint32_t>(arg_idx++);

    uint32_t sender_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t sender_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t sender_wait_finish_semaphore = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_src_start_phase_addr = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t first_phase_remote_src_phase =
        wait_for_remote_source_starting_phase(reinterpret_cast<volatile uint32_t *>(remote_src_start_phase_addr));
    const uint32_t second_phase_remote_src_phase = first_phase_remote_src_phase + 1;
    const uint32_t local_first_phase = get_first_available_phase_out_of_reset(stream_id);
    const uint32_t local_second_phase = local_first_phase;

    auto local_phase_iterator = phase_iterator_t(local_first_phase, local_second_phase);
    auto remote_phase_iterator = phase_iterator_t(first_phase_remote_src_phase, second_phase_remote_src_phase);

    stream_state_t stream_state{
        stream_buffer_addr,
        stream_tile_header_buffer_addr,

        local_phase_iterator.get(),  // phase_id
        stream_tile_header_max_num_messages,

        stream_tile_header_buffer_addr,  // msg_info_wrptr_addr;

        0,                                    // num_tiles_sent;
        stream_tile_header_max_num_messages,  // tile_header_num_msgs;

        stream_buffer_addr,              // dest_buffer_base_addr;
        stream_buffer_size,              // dest_buffer_size;
        stream_tile_header_buffer_addr,  // dest_msg_info_ptr;

        0,  // src_buffer_read_offset;

        remote_src_buffer_addr,              // src_buffer_base_addr;
        remote_src_buffer_size_4B_words,     // src_buffer_size;
        remote_src_tile_header_buffer_addr,  // src_msg_info_ptr;

        0,                            // dest_buffer_write_offset;
        remote_phase_iterator.get(),  // receiver start phase
    };

    ASSERT((stream_state.local_data_buffer_base_address & 0xf) == 0);

    auto remote_noc_info_desc =
        noc_endpoint_info_t{remote_src_data_noc_id, 1 - remote_src_data_noc_id, remote_src_noc_x, remote_src_noc_y};

    advance_phase(remote_noc_info_desc, stream_state, stream_id);

    auto cb = tt::CB::c_in0;
    stream_state.local_buffer_base_addr = stream_buffer_addr;

    for (uint32_t i = 0; i < num_messages_to_forward; i++) {
        cb_reserve_back(cb, 1);

        while (!messages_are_available(stream_id, stream_state)) {
            asm volatile("nop");
        }

        auto const &[msg_addr, msg_size_bytes] = get_next_message_info(stream_id, stream_state);
        ASSERT(msg_size_bytes > 0);
        ASSERT(msg_size_bytes <= stream_state.local_buffer_size);

        copy_message_to_cb_blocking(cb, msg_addr, msg_size_bytes, stream_state);

        cb_push_back(cb, 1);

        stream_relay_tiles(stream_id, 1, msg_size_bytes >> 4);
        advance_stream_to_next_message(
            remote_noc_info_desc, stream_state, stream_id, msg_size_bytes, local_phase_iterator, remote_phase_iterator);
    }

    noc_semaphore_inc(get_noc_addr(sender_noc_x, sender_noc_y, sender_wait_finish_semaphore), 1);

    while ((NOC_STREAM_READ_REG(stream_id, STREAM_DEBUG_STATUS_REG_INDEX + 9) >> MEM_WORD_ADDR_WIDTH) != 0) {
        asm volatile("nop");
    }

    stream_reset(stream_id);

    noc_semaphore_inc(
        get_noc_addr(remote_noc_info_desc.noc_x, remote_noc_info_desc.noc_y, relay_done_semaphore_addr), 1);
    noc_semaphore_inc(
        get_noc_addr(other_relay_core_to_signal_x, other_relay_core_to_signal_y, other_relay_done_semaphore), 1);

    ASSERT(!assert_check(stream_id, false));
}
