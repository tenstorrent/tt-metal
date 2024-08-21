// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "stream_interface.h"
#include "tests/tt_metal/tt_metal/test_kernels/dataflow/streams/stream_io_kernel_helpers.hpp"
#include "tt_metal/hw/inc/wormhole/noc/noc_overlay_parameters.h"

//////////
///  FUTURE OPTIMIZATIONS
///////////
// 1) Don't update message info rd/wrptrs. Instead, just write message size into the next corresponding message info
// buffer entry 2) Use stream registers to track # messages sent 3) For contiguous messages, use a single stream phase
// to send them back to back then only do one wait for flush at the end

//////////
// Q/A W/ Djordje + Extra Notes
//
// 1) DON'T set any of the STREAM_REMOTE_DEST_* registers if NEXT_PHASE_SRC_CHANGE is false
// 2) stream_phase_advance_wait can be used to wait for the current phase to complete
//    -> in the scheme for this producer, it'll end up waiting until the message is sent out of L1
// 3) How does initial stream handshake happen?
//    -> Stream has hidden registers: curr_phase_src/dest_change. When comming out of reset, these are set true
//       This value is sticky and the next_phase_src/dest_change will override it for the next phase
///////

uint32_t get_sender_stream_config_reg(uint32_t tx_noc_id, uint32_t rx_src_update_noc, bool drain_after_phase_send) {
    uint32_t stream_cfg_reg = 0;
    bool next_phase_src_dest_change = drain_after_phase_send ? 1 : 0;
    stream_cfg_reg |= STREAM_CFG(OUTGOING_DATA_NOC, tx_noc_id) | STREAM_CFG(REMOTE_SRC_UPDATE_NOC, rx_src_update_noc) |
                      STREAM_CFG(SOURCE_ENDPOINT, 1) | STREAM_CFG(REMOTE_RECEIVER, 1) |
                      STREAM_CFG(NEXT_PHASE_SRC_CHANGE, next_phase_src_dest_change) |
                      STREAM_CFG(NEXT_PHASE_DEST_CHANGE, next_phase_src_dest_change) |
                      STREAM_CFG(PHASE_AUTO_ADVANCE, 0) | STREAM_CFG(DATA_AUTO_SEND, 0) |
                      STREAM_CFG(REG_UPDATE_VC_REG, 1);

    return stream_cfg_reg;
}

FORCE_INLINE void write_message_size_to_message_info_buffer(
    stream_state_t const &stream_state, uint32_t message_size_noc_words) {
    ASSERT((message_size_noc_words << 4) <= stream_state.local_buffer_size);
    if (!((message_size_noc_words << 4) <= stream_state.local_buffer_size)) {
        DPRINT << "YIKES\n";
    }
    *reinterpret_cast<volatile uint32_t *>(stream_state.local_msg_info_ptr) = message_size_noc_words;
}

FORCE_INLINE void reset_stream_message_info_buffer_rdptr(stream_state_t &stream_state, uint32_t stream_id) {
    stream_state.local_msg_info_ptr = stream_state.local_msg_info_ptr_base_address;
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_MSG_INFO_PTR_REG_INDEX, ((uint32_t)(stream_state.local_msg_info_ptr_base_address >> 4)));
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX, (((uint32_t)stream_state.local_msg_info_ptr_base_address >> 4)));
}
FORCE_INLINE void advance_stream_message_info_buffer_wrptr(
    stream_state_t &stream_state, uint32_t stream_id, uint32_t message_size) {
    stream_state.local_msg_info_ptr += (1 << 4);
    stream_state.local_buffer_read_offset += message_size;
    if (stream_state.local_buffer_read_offset >= stream_state.local_buffer_size) {
        stream_state.local_buffer_read_offset -= stream_state.local_buffer_size;
    }
}

FORCE_INLINE void wait_for_stream_write_complete(uint32_t sender_stream_id) {
    while (!stream_phase_advance_wait(sender_stream_id)) {
        asm volatile("nop");
    }
}

FORCE_INLINE void copy_from_cb_to_stream_buffer(
    stream_state_t &stream_state, uint32_t message_base, uint32_t message_size_noc_words) {
    ASSERT((message_size_noc_words << 4) <= stream_state.local_buffer_size);
    if (!((message_size_noc_words << 4) <= stream_state.local_buffer_size)) {
        DPRINT << "YIKES2\n";
    }
    uint32_t message_size_size_in_bytes = message_size_noc_words << 4;
    uint32_t bytes_to_copy =
        std::min(stream_state.local_buffer_size - stream_state.local_buffer_read_offset, message_size_size_in_bytes);
    noc_async_write(message_base, get_noc_addr(stream_state.get_current_local_buffer_address()), bytes_to_copy);
    ASSERT(stream_state.local_buffer_size + stream_state.local_buffer_read_offset >= bytes_to_copy);
    if (!(stream_state.local_buffer_size + stream_state.local_buffer_read_offset >= bytes_to_copy)) {
        DPRINT << "YIKES3\n";
    }

    if (bytes_to_copy < message_size_size_in_bytes) {
        uint32_t second_bytes_to_copy = message_size_size_in_bytes - bytes_to_copy;
        noc_async_write(
            message_base + bytes_to_copy, get_noc_addr(stream_state.local_buffer_base_addr), second_bytes_to_copy);
    }
    noc_async_write_barrier();
}

FORCE_INLINE void hang_toggle(volatile uint32_t *hang_toggle_semaphore) {
    return;
    while (*hang_toggle_semaphore == 0) {
        asm volatile("");
    }
    *hang_toggle_semaphore = 0;
}

FORCE_INLINE void stream_noc_write(
    stream_state_t &stream_state,
    uint32_t message_base,
    uint32_t sender_stream_id,
    uint32_t dest_addr,
    uint32_t remote_noc_x,
    uint32_t remote_noc_y,
    uint32_t dest_noc_id,
    uint32_t dest_tile_header_buffer_addr,
    uint32_t local_start_phase,
    bool very_first_message,
    volatile uint32_t *hang_toggle_semaphore,
    uint32_t message_id) {
    const uint32_t tiles_per_phase = stream_state.messages_per_phase;

    uint32_t message_size_noc_words = *reinterpret_cast<volatile uint32_t *>(message_base);

    uint32_t dest_noc_reg = 0;
    uint32_t num_tiles = stream_state.num_tiles_sent;
    const bool send_last_message_and_drain = num_tiles == (stream_state.tile_header_num_msgs - 1);

    bool first_message = num_tiles == 0;

    NOC_STREAM_WRITE_REG(sender_stream_id, STREAM_CURR_PHASE_BASE_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(sender_stream_id, STREAM_CURR_PHASE_REG_INDEX, stream_state.local_phase_id);

    if (first_message) {
        reset_stream_message_info_buffer_rdptr(stream_state, sender_stream_id);
        stream_state.local_buffer_read_offset = 0;
    }
    copy_from_cb_to_stream_buffer(stream_state, message_base, message_size_noc_words);

    if (message_id < 10) {
        hang_toggle(hang_toggle_semaphore);
    }

    uint32_t rx_src_update_noc = 1 - dest_noc_id;
    if (send_last_message_and_drain) {
        NOC_STREAM_WRITE_REG(
            sender_stream_id,
            STREAM_MISC_CFG_REG_INDEX,
            get_sender_stream_config_reg(dest_noc_id, rx_src_update_noc, true));

    } else if (first_message) {
        // ASSERT(stream_state.remote_buffer_base_addr + stream_state.local_buffer_size <=
        // stream_state.remote_buffer_size ||
        //   stream_state.remote_buffer_size + (stream_state.tile_header_num_msgs << 4) <=
        //   stream_state.remote_buffer_base_addr);

        uint32_t rx_src_update_noc = 1 - dest_noc_id;
        uint32_t translated_remote_noc_x = dest_noc_id == 0 ? remote_noc_x : noc_size_x - 1 - remote_noc_x;
        uint32_t translated_remote_noc_y = dest_noc_id == 0 ? remote_noc_y : noc_size_y - 1 - remote_noc_y;
        uint32_t dest_stream_id = sender_stream_id;

        NOC_STREAM_WRITE_REG(
            sender_stream_id,
            STREAM_BUF_START_REG_INDEX,
            ((uint32_t)stream_state.get_current_local_buffer_address()) >> 4);
        NOC_STREAM_WRITE_REG(sender_stream_id, STREAM_BUF_SIZE_REG_INDEX, stream_state.local_buffer_size >> 4);

        NOC_STREAM_WRITE_REG(
            sender_stream_id,
            STREAM_REMOTE_DEST_REG_INDEX,
            STREAM_REMOTE_DEST(translated_remote_noc_x, translated_remote_noc_y, dest_stream_id));
        NOC_STREAM_WRITE_REG(sender_stream_id, STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_HI_REG_INDEX, 0);
        NOC_STREAM_WRITE_REG(
            sender_stream_id, STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX, stream_state.remote_msg_info_ptr >> 4);

        // DPRINT << "STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX: " << (uint32_t)(stream_state.remote_msg_info_ptr >>
        // 4) << "\n";
        NOC_STREAM_WRITE_REG(
            sender_stream_id, STREAM_REMOTE_DEST_BUF_START_REG_INDEX, stream_state.remote_buffer_base_addr >> 4);
        // Inserting an assert here causes test to pass
        NOC_STREAM_WRITE_REG(
            sender_stream_id,
            STREAM_REMOTE_DEST_BUF_START_HI_REG_INDEX,
            (stream_state.remote_buffer_base_addr / MEM_WORD_WIDTH) >> MEM_WORD_ADDR_WIDTH);
        NOC_STREAM_WRITE_REG_FIELD(
            sender_stream_id,
            STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX,
            REMOTE_DEST_BUF_SIZE_WORDS,
            stream_state.remote_buffer_size >> 4);

        NOC_STREAM_WRITE_REG(
            sender_stream_id,
            STREAM_MISC_CFG_REG_INDEX,
            get_sender_stream_config_reg(dest_noc_id, rx_src_update_noc, false));
    }

    if (first_message) {
        // DPRINT << "Msg info ptr: " << (uint32_t)stream_state.local_msg_info_ptr << "\n";
    }
    if (very_first_message) {
        hang_toggle(hang_toggle_semaphore);
    }

    write_message_size_to_message_info_buffer(stream_state, message_size_noc_words);
    advance_stream_message_info_buffer_wrptr(stream_state, sender_stream_id, message_size_noc_words << 4);

    NOC_STREAM_WRITE_REG(
        sender_stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, AUTO_CFG_HEADER(0, 1 /*tiles_per_phase*/, 1));
    NOC_STREAM_WRITE_REG(sender_stream_id, STREAM_PHASE_ADVANCE_REG_INDEX, 0x1);

    if (first_message) {
        // wait for handshake to complete
        while (!stream_phase_is_active(sender_stream_id)) {
            asm volatile("");
        }
    }

    if (very_first_message) {
        hang_toggle(hang_toggle_semaphore);
    }

    if (send_last_message_and_drain) {
        // We only wrap around to 0 when the remote receiver relay stream has finished its second phase. We need to do
        // this to avoid any handshake bugs we might hit if the second phase of relay must sync with phase 1 of the
        // producer (this) since the relay will handshake with phase 1 of the producer (this) stream for relay stream's
        // first phase too
        num_tiles = 0;
        stream_state.remote_phase_id = 3 - stream_state.remote_phase_id;  // will alternate between 1 and 2
        // Remote phase was already updated so the condition is inverted
        stream_state.local_phase_id =
            (stream_state.remote_phase_id == 1) ? local_start_phase : stream_state.local_phase_id + 1;
    } else {
        num_tiles++;
        stream_state.local_phase_id++;
    }

    stream_relay_tiles(sender_stream_id, 1, message_size_noc_words);
    wait_for_stream_write_complete(sender_stream_id);

    if (very_first_message) {
        hang_toggle(hang_toggle_semaphore);
    }

    stream_state.num_tiles_sent = num_tiles;
}

void kernel_main() {
    uint32_t arg_idx = 0;

    uint32_t num_messages_to_forward = get_arg_val<uint32_t>(arg_idx++);

    uint32_t stream_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t stream_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t stream_buffer_size = get_arg_val<uint32_t>(arg_idx++);
    uint32_t stream_tile_header_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t stream_tile_header_max_num_messages = get_arg_val<uint32_t>(arg_idx++);

    uint32_t remote_dest_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_dest_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_dest_noc_stream_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_dest_noc_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_dest_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_dest_buffer_size_4B_words = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_dest_tile_header_buffer_addr = get_arg_val<uint32_t>(arg_idx++);

    uint32_t relay_done_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t other_relay_core_to_signal_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t other_relay_core_to_signal_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t other_relay_done_semaphore = get_arg_val<uint32_t>(arg_idx++);

    uint32_t wait_receiver_semaphore = get_arg_val<uint32_t>(arg_idx++);
    *reinterpret_cast<volatile uint32_t *>(wait_receiver_semaphore) = 0;

    uint32_t first_relay_remote_src_start_phase_addr = get_arg_val<uint32_t>(arg_idx++);
    volatile uint32_t *hang_toggle_semaphore = reinterpret_cast<volatile uint32_t *>(get_arg_val<uint32_t>(arg_idx++));

    uint32_t local_starting_phase =
        notify_remote_receiver_of_starting_phase(
            stream_id,
            stream_buffer_addr,
            get_noc_addr(remote_dest_noc_x, remote_dest_noc_y, first_relay_remote_src_start_phase_addr)) -
        1;

    // clear the buffers
    for (uint32_t i = 0; i < stream_buffer_size / sizeof(uint32_t); i++) {
        reinterpret_cast<volatile uint32_t *>(stream_buffer_addr)[i] = 0;
    }
    for (uint32_t i = 0; i < stream_tile_header_max_num_messages * 4; i++) {
        reinterpret_cast<volatile uint32_t *>(stream_tile_header_buffer_addr)[i] = 0;
    }

    stream_state_t stream_state{
        stream_buffer_addr,
        stream_tile_header_buffer_addr,

        local_starting_phase,                 // phase_id
        stream_tile_header_max_num_messages,  // messages_per_phase

        stream_tile_header_buffer_addr,  // msg_info_wrptr_addr;

        0,                                    // num_tiles_sent;
        stream_tile_header_max_num_messages,  // tile_header_num_msgs;

        stream_buffer_addr,              // src_buffer_base_addr;
        stream_buffer_size,              // src_buffer_size;
        stream_tile_header_buffer_addr,  // src_msg_info_ptr;
        0,                               // src_buffer_read_offset;

        remote_dest_buffer_addr,              // dest_buffer_base_addr;
        remote_dest_buffer_size_4B_words,     // dest_buffer_size;
        remote_dest_tile_header_buffer_addr,  // dest_msg_info_ptr;
        0,                                    // dest_buffer_write_offset;

        1,  // receiver_phase; // receiver start phase // don't need the true value
    };

    DPRINT << "hang_toggle_semaphore: " << (uint32_t)hang_toggle_semaphore << "\n";

    hang_toggle(hang_toggle_semaphore);

    auto cb = tt::CB::c_in0;
    bool very_first_message = true;

    uint32_t message_id = 0;
    uint32_t count = 0;
    for (uint32_t i = 0; i < num_messages_to_forward; i++) {
        cb_wait_front(cb, 1);
        uint32_t src_addr = get_read_ptr(cb);
        stream_noc_write(
            stream_state,
            src_addr,
            stream_id,
            stream_state.remote_buffer_base_addr,
            remote_dest_noc_x,
            remote_dest_noc_y,
            remote_dest_noc_id,
            remote_dest_tile_header_buffer_addr,
            local_starting_phase,
            very_first_message,
            hang_toggle_semaphore,
            message_id);

        cb_pop_front(cb, 1);
        // if (count == 1000) {
        //   DPRINT << "Sent " << i << " messages\n";
        //   count = 0;
        // } else {
        //   count++;
        // }
        very_first_message = false;
        message_id++;
    }

    // Reset sequence is that both the remote sender and remote receiver streams of the relay
    // should reset first so that no data is in flight. Sender and receiver must ensure that no
    // payloads are in flight to the relay stream(s) before sending the reset signal to the relay
    // core
    noc_semaphore_wait(reinterpret_cast<volatile uint32_t *>(wait_receiver_semaphore), 1);

    stream_reset(stream_id);

    noc_semaphore_inc(
        get_noc_addr(other_relay_core_to_signal_x, other_relay_core_to_signal_y, other_relay_done_semaphore), 1);
    noc_semaphore_inc(get_noc_addr(remote_dest_noc_x, remote_dest_noc_y, relay_done_semaphore_addr), 1);

    ASSERT(!assert_check(stream_id, false));
}
