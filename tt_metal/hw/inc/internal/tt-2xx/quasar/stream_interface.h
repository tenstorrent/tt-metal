// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _STREAM_INTERFACE_H_
#define _STREAM_INTERFACE_H_

#include "noc_overlay_parameters.h"
#include "noc_parameters.h"
#include "noc_nonblocking_api.h"

// Low-level chip-dependent stream/NOC functions

#define STREAM_PTR_REG_MASK ((uint32_t)0xFFFF)
#define EPOCH_SHIFT 15
#define MAX_TILES_MSG_INFO_BUF_PER_PHASE 2048
#define USE_2K_TILE_HEADER_BUFFER_RESET
#define MULTI_MSG_TILES_STREAM_THESH \
    12  // Note streams 6 and 7 are not capable of multi-msg tiles, so dont use them for inputs

inline __attribute__((always_inline)) void stream_phase_blob_run(
    uint32_t stream_id, uint32_t blob_start_addr, uint32_t start_phase_num_cfg_regs) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_MULTI_MSG_CLEAR_REG_INDEX, 0);  // Prevent accidental clearing
    NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX, blob_start_addr);
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, start_phase_num_cfg_regs << NEXT_PHASE_NUM_CFG_REG_WRITES);

    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_MISC_CFG_REG_INDEX, (1 << NEXT_PHASE_SRC_CHANGE) | (1 << NEXT_PHASE_DEST_CHANGE));
    NOC_STREAM_WRITE_REG(stream_id, STREAM_ONETIME_MISC_CFG_REG_INDEX, (0x1 << PHASE_AUTO_CONFIG));
}

inline __attribute__((always_inline)) void stream_phase_blob_run_offset(
    uint32_t stream_id, uint32_t blob_base_addr, uint32_t blob_start_addr, uint32_t blob_size) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_MULTI_MSG_CLEAR_REG_INDEX, 0);  // Prevent accidental clearing
    uint32_t blob_offset = NOC_STREAM_READ_REG(stream_id, STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX) - blob_base_addr;
    while (blob_offset >= blob_size) {
        blob_offset -= blob_size;
    }
    NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX, blob_start_addr + blob_offset);
    uint32_t misc_cfg_reg = NOC_STREAM_READ_REG(stream_id, STREAM_ONETIME_MISC_CFG_REG_INDEX);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_ONETIME_MISC_CFG_REG_INDEX, (0x1 << PHASE_AUTO_CONFIG) | misc_cfg_reg);
}

inline __attribute__((always_inline)) uint32_t stream_get_auto_cfg_ptr(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX);
}

inline __attribute__((always_inline)) uint32_t stream_get_auto_cfg_header(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX);
}

inline __attribute__((always_inline)) uint32_t stream_get_auto_cfg_header_phase_num_cfg_regs(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX) >> NEXT_PHASE_NUM_CFG_REG_WRITES;
}

inline __attribute__((always_inline)) void stream_set_auto_cfg_header(uint32_t stream_id, uint32_t val) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, val);
}

inline __attribute__((always_inline)) uint32_t stream_phase_is_active(uint32_t stream_id) {
    return NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_WAIT_STATUS_REG_INDEX, MSG_FWD_ONGOING);
}

inline __attribute__((always_inline)) uint32_t stream_get_curr_phase(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_CURR_PHASE_REG_INDEX);
}

inline __attribute__((always_inline)) void set_fork_scatter_inner_loop_count(uint32_t stream_id, uint32_t val) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_SRC_REG_INDEX, val);
}

inline __attribute__((always_inline)) uint32_t get_fork_scatter_inner_loop_count(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_SRC_REG_INDEX);
}

inline __attribute__((always_inline)) void set_fork_num_msgs_in_block(uint32_t stream_id, uint32_t val) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_SRC_PHASE_REG_INDEX, val);
}

inline __attribute__((always_inline)) uint32_t get_fork_num_msgs_in_block(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_SRC_PHASE_REG_INDEX);
}

inline __attribute__((always_inline)) bool stream_phase_id_is_active(uint32_t stream_id, uint32_t phase_id) {
    uint32_t curr_phase = stream_get_curr_phase(stream_id);
    bool phase_active = stream_phase_is_active(stream_id);
    return (curr_phase == phase_id) && phase_active;
}

inline __attribute__((always_inline)) uint32_t stream_phase_advance_wait(uint32_t stream_id) {
    uint32_t advance_wait =
        NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_WAIT_STATUS_REG_INDEX, WAIT_SW_PHASE_ADVANCE_SIGNAL);
    uint32_t num_tiles_pending =
        NOC_STREAM_READ_REG(stream_id, STREAM_DEBUG_STATUS_REG_INDEX + 9) >> MEM_WORD_ADDR_WIDTH;
    return advance_wait && (num_tiles_pending == 0);
}

inline __attribute__((always_inline)) uint32_t stream_get_input_noc(uint32_t stream_id) {
    return NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_MISC_CFG_REG_INDEX, INCOMING_DATA_NOC);
}

inline __attribute__((always_inline)) void stream_get_remote_src_coord(uint32_t stream_id, uint32_t& x, uint32_t& y) {
    x = NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_REMOTE_SRC_REG_INDEX, STREAM_REMOTE_SRC_X);
    y = NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_REMOTE_SRC_REG_INDEX, STREAM_REMOTE_SRC_Y);
}

inline __attribute__((always_inline)) uint32_t stream_get_output_noc(uint32_t stream_id) {
    return NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_MISC_CFG_REG_INDEX, OUTGOING_DATA_NOC);
}

inline __attribute__((always_inline)) uint32_t stream_get_output_unicast_vc(uint32_t stream_id) {
    return NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_MISC_CFG_REG_INDEX, UNICAST_VC_REG);
}

inline __attribute__((always_inline)) void stream_get_remote_dest_coord(uint32_t stream_id, uint32_t& x, uint32_t& y) {
    x = NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_REMOTE_DEST_REG_INDEX, STREAM_REMOTE_DEST_X);
    y = NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_REMOTE_DEST_REG_INDEX, STREAM_REMOTE_DEST_Y);
}

inline __attribute__((always_inline)) uint32_t stream_get_msg_info_rd_ptr(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_MSG_INFO_PTR_REG_INDEX);
}

inline __attribute__((always_inline)) uint32_t stream_get_data_buf_addr(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_BUF_START_REG_INDEX);
}

inline __attribute__((always_inline)) uint32_t stream_get_remote_data_buf_addr(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_START_REG_INDEX);
}

inline __attribute__((always_inline)) uint32_t stream_get_data_buf_size(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_BUF_SIZE_REG_INDEX);
}

inline __attribute__((always_inline)) uint32_t stream_get_remote_data_buf_size(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX);
}

inline __attribute__((always_inline)) uint32_t stream_get_remote_data_buf_space_available(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
}

inline __attribute__((always_inline)) uint32_t stream_phase_next_recved_tile_addr(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_NEXT_RECEIVED_MSG_ADDR_REG_INDEX);
}

inline __attribute__((always_inline)) uint32_t stream_phase_next_recved_tile_size(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_NEXT_RECEIVED_MSG_SIZE_REG_INDEX);
}

inline __attribute__((always_inline)) uint32_t
stream_phase_tiles_received(uint32_t stream_id, uint32_t msg_info_buf_start_addr) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX) - msg_info_buf_start_addr;
}

inline __attribute__((always_inline)) uint32_t stream_rec_endpoint_get_phase_tiles_count(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_REG_INDEX) &
           0xffff;  // used as scratch reg for receiver endpoint streams
}

inline __attribute__((always_inline)) void stream_rec_endpoint_set_phase_tiles_count(uint32_t stream_id, uint32_t val) {
    uint32_t rmw = NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_REG_INDEX) & ~0xffff;
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_REMOTE_DEST_REG_INDEX, (rmw | val));  // used as scratch reg for receiver endpoint streams
}

inline __attribute__((always_inline)) uint32_t stream_src_endpoint_get_phase_tiles_count(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_SRC_PHASE_REG_INDEX) &
           0xffff;  // used as scratch reg for source endpoint streams
}

inline __attribute__((always_inline)) void stream_src_endpoint_set_phase_tiles_count(uint32_t stream_id, uint32_t val) {
    uint32_t rmw = NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_SRC_PHASE_REG_INDEX) & ~0xffff;
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_REMOTE_SRC_PHASE_REG_INDEX, (rmw | val));  // used as scratch reg for source endpoint streams
}

inline __attribute__((always_inline)) uint32_t stream_get_buf_space_available_words(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_BUF_SPACE_AVAILABLE_REG_INDEX);
}

inline __attribute__((always_inline)) void stream_signal_flushed_tiles(
    uint32_t stream_id, uint32_t num_tiles, uint32_t num_words) {
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_NUM_MSGS_RECEIVED_INC_REG_INDEX,
        (num_words << SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE) | num_tiles);
}

inline __attribute__((always_inline)) bool stream_is_dram_read_opt_enabled(uint32_t stream_id) {
    return !NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_MISC_CFG_REG_INDEX, NEXT_PHASE_SRC_CHANGE);
}

inline __attribute__((always_inline)) bool stream_next_phase_src_change(uint32_t stream_id) {
    return NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_MISC_CFG_REG_INDEX, NEXT_PHASE_SRC_CHANGE) ||
           !NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_ONETIME_MISC_CFG_REG_INDEX, PHASE_AUTO_CONFIG);
}

inline __attribute__((always_inline)) int stream_get_curr_phase_num_msgs(uint32_t stream_id) {
    return NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, CURR_PHASE_NUM_MSGS);
}

inline __attribute__((always_inline)) void stream_set_curr_phase_num_msgs(uint32_t stream_id, uint32_t val) {
    uint32_t rmw = NOC_STREAM_READ_REG(stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX) & ~0xFFFFFF;
    NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, (rmw | (val << CURR_PHASE_NUM_MSGS)));
}

// used by unpacker fracture
inline __attribute__((always_inline)) void stream_relay_tiles(
    uint32_t stream_id, uint32_t num_tiles, uint32_t num_words) {
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_NUM_MSGS_RECEIVED_INC_REG_INDEX,
        (num_words << SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE) | num_tiles);
}

// used by packer
inline uint32_t stream_get_free_words(uint32_t stream_id) {
    uint32_t wait_status = NOC_STREAM_READ_REG(stream_id, STREAM_WAIT_STATUS_REG_INDEX);
    uint32_t tiles_left_in_phase = NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_SRC_PHASE_REG_INDEX) &
                                   0xffff;  // used as scratch reg for source endpoint streams
    uint32_t buf_space_available = NOC_STREAM_READ_REG(stream_id, STREAM_BUF_SPACE_AVAILABLE_REG_INDEX);
    wait_status &= (0x1 << MSG_FWD_ONGOING);
    return (wait_status && tiles_left_in_phase) ? buf_space_available : 0;
}

inline uint32_t stream_should_packer_reset_pointers(uint32_t stream_id) {
    uint32_t should_packer_reset_pointers = NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_SRC_REG_INDEX);
    if (should_packer_reset_pointers) {
        NOC_STREAM_WRITE_REG(
            stream_id, STREAM_REMOTE_SRC_REG_INDEX, 0);  // used as scratch reg for source endpoint streams
    }
    return should_packer_reset_pointers;
}

inline uint32_t stream_dram_write_should_reset_pointers(uint32_t stream_id) {
    uint32_t rmw = NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_REG_INDEX);
    uint32_t should_reset_pointers = rmw >> 16;
    if (should_reset_pointers) {
        NOC_STREAM_WRITE_REG(
            stream_id,
            STREAM_REMOTE_DEST_REG_INDEX,
            (rmw & 0xffff));  // used as scratch reg for receiver endpoint streams
    }
    return should_reset_pointers;
}

inline uint32_t stream_dram_read_should_reset_pointers(uint32_t stream_id) {
    uint32_t rmw = NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_SRC_PHASE_REG_INDEX);
    uint32_t should_reset_pointers = rmw >> 16;
    if (should_reset_pointers) {
        NOC_STREAM_WRITE_REG(
            stream_id,
            STREAM_REMOTE_SRC_PHASE_REG_INDEX,
            (rmw & 0xffff));  // used as scratch reg for receiver endpoint streams
    }
    return should_reset_pointers;
}

template <bool fracture = false, bool with_rd_ptr = false>
static __attribute__((unused)) __attribute__((noinline)) bool stream_get_push_flushed(
    uint32_t stream_id, uint32_t exp_rd_ptr = 0) {
    uint32_t prev_phase = stream_get_curr_phase(stream_id);
    uint32_t wait_status = NOC_STREAM_READ_REG(stream_id, STREAM_WAIT_STATUS_REG_INDEX);
    wait_status &= (0x1 << MSG_FWD_ONGOING);

    if (wait_status) {
        uint32_t buf_size = NOC_STREAM_READ_REG(stream_id, STREAM_BUF_SIZE_REG_INDEX);
        uint32_t buf_space_available = NOC_STREAM_READ_REG(stream_id, STREAM_BUF_SPACE_AVAILABLE_REG_INDEX);
        uint32_t num_tiles;
        if constexpr (fracture) {
            num_tiles = 0;
        } else {
            num_tiles = stream_get_curr_phase_num_msgs(stream_id);
        }
        uint32_t rd_ptr;
        if constexpr (with_rd_ptr) {
            rd_ptr = NOC_STREAM_READ_REG(stream_id, STREAM_RD_PTR_REG_INDEX);
        }
        uint32_t cur_phase = stream_get_curr_phase(stream_id);
        if (cur_phase == prev_phase) {
            if constexpr (with_rd_ptr) {
                return (buf_space_available != 0 && rd_ptr == exp_rd_ptr) ||
                       (buf_size == buf_space_available &&
                        num_tiles >
                            0);  // For this case we might be resending next phase so we need the num_tiles > 0 check
            } else if constexpr (fracture) {
                return buf_size ==
                       buf_space_available;  // We dont need num_tiles > 0 as there is no resend concept for fracture
            } else {
                return buf_size == buf_space_available &&
                       num_tiles >
                           0;  // For this case we might be resending next phase so we need the num_tiles > 0 check
            }
        }
    }

    return stream_phase_advance_wait(stream_id);
}

inline __attribute__((always_inline)) uint32_t stream_get_buf_space_available(uint32_t stream_id) {
    uint32_t buf_space_available = NOC_STREAM_READ_REG(stream_id, STREAM_BUF_SPACE_AVAILABLE_REG_INDEX);
    return buf_space_available;
}

// used by packer
inline __attribute__((always_inline)) void stream_push_tiles(
    uint32_t stream_id, uint32_t num_tiles, uint32_t num_words) {
    uint32_t rmw = NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_SRC_PHASE_REG_INDEX);
    uint32_t tiles_left_in_phase = rmw & 0xffff;
    rmw = rmw & ~0xffff;
    tiles_left_in_phase -= num_tiles;
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_REMOTE_SRC_PHASE_REG_INDEX,
        (rmw | tiles_left_in_phase));  // used as scratch reg for source endpoint streams
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_NUM_MSGS_RECEIVED_INC_REG_INDEX,
        (num_words << SOURCE_ENDPOINT_NEW_MSGS_TOTAL_SIZE) | num_tiles);
}

inline void stream_set_tiles_left_in_phase(uint32_t stream_id, uint32_t num_tiles) {
    uint32_t rmw = NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_SRC_PHASE_REG_INDEX);
    uint32_t tiles_left_in_phase = rmw & 0xffff;
    rmw = rmw & ~0xffff;
    tiles_left_in_phase -= num_tiles;
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_REMOTE_SRC_PHASE_REG_INDEX,
        (rmw | tiles_left_in_phase));  // used as scratch reg for source endpoint streams
}

#define STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE(dest_num, words_free_inc) \
    (((dest_num) << REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_DEST_NUM) |          \
     ((words_free_inc) << REMOTE_DEST_BUF_WORDS_FREE_INC))

#define STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_ADDR(stream_id) \
    (STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX))

inline __attribute__((always_inline)) void stream_update_remote_dest_buf_space_available(
    uint32_t stream_id, uint32_t dest_num, uint32_t inc_val) {
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX,
        STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE(dest_num, inc_val));
}

inline __attribute__((always_inline)) bool stream_is_receiver_endpoint(uint32_t stream_id) {
    return NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_MISC_CFG_REG_INDEX, RECEIVER_ENDPOINT);
}

inline __attribute__((always_inline)) void stream_receiver_endpoint_single_clear_op(
    uint32_t stream_id, uint32_t num_tiles) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_MSG_INFO_CLEAR_REG_INDEX, num_tiles);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_MSG_DATA_CLEAR_REG_INDEX, num_tiles);
}

inline __attribute__((always_inline)) uint32_t stream_tiles_outstanding(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_NUM_MSGS_RECEIVED_REG_INDEX);
}

inline void stream_receiver_endpoint_tiles_clear(uint32_t stream_id, uint32_t num_tiles) {
    while (num_tiles > 0) {
        uint32_t num_to_clear = (num_tiles == 1) ? 1 : 2;

        // Bug fix for streams. Flow ctrl messages are sent out of order, must clear one message at the end of phase.
        int32_t num_msgs_left_in_phase = stream_get_curr_phase_num_msgs(stream_id);
        if (num_msgs_left_in_phase <= 2) {
            num_to_clear = 1;
        }

        stream_receiver_endpoint_single_clear_op(stream_id, num_to_clear);
        num_tiles -= num_to_clear;
    }
}

inline bool stream_receiver_endpoint_tile_clearing_finished(uint32_t stream_id) {
    return (NOC_STREAM_READ_REG(stream_id, STREAM_MULTI_MSG_CLEAR_REG_INDEX) == 0);
}

inline void stream_receiver_endpoint_tiles_clear_b0(uint32_t stream_id, uint32_t num_tiles) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_MULTI_MSG_CLEAR_REG_INDEX, num_tiles);
}

inline __attribute__((always_inline)) void stream_reset(uint32_t stream_id) {
    uint32_t val = NOC_STREAM_READ_REG(stream_id, STREAM_ONETIME_MISC_CFG_REG_INDEX);
    val &= (~(1 << PHASE_AUTO_CONFIG));
    NOC_STREAM_WRITE_REG(stream_id, STREAM_ONETIME_MISC_CFG_REG_INDEX, val);  // disable auto-config
    NOC_STREAM_WRITE_REG(stream_id, STREAM_RESET_REG_INDEX, 0x1);
}

inline __attribute__((always_inline)) void stream_force_next_phase(uint32_t stream_id) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_RESET_REG_INDEX, 0x1);
}

inline bool assert_check(uint32_t stream_id, bool hang) {
    uint32_t debug_assert = NOC_STREAM_READ_REG(stream_id, STREAM_DEBUG_ASSERTIONS_REG_INDEX);
    if (debug_assert > 0 && hang) {
        while (true) {
        };
    }
    return debug_assert > 0;
}

inline bool stream_done_hint() {
    uint32_t stream_done = NOC_STREAM_READ_REG(0, STREAM_BLOB_AUTO_CFG_DONE_REG_INDEX) |
                           NOC_STREAM_READ_REG(0, STREAM_BLOB_AUTO_CFG_DONE_REG_INDEX + 1);
    if (stream_done) {
        NOC_STREAM_WRITE_REG(0, STREAM_BLOB_AUTO_CFG_DONE_REG_INDEX, 0xFFFFFFFF);
        NOC_STREAM_WRITE_REG(0, STREAM_BLOB_AUTO_CFG_DONE_REG_INDEX + 1, 0xFFFFFFFF);
        return true;
    } else {
        return false;
    }
}

inline bool should_stall_for_tile_header_buffer_reset(
    uint32_t stream_id, uint32_t msg_info_buf_addr, uint32_t buf_size_tiles, uint32_t& prev_ack_thresh) {
    uint32_t is_remote_src = NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_MISC_CFG_REG_INDEX, REMOTE_SOURCE);
    uint32_t msg_info_wr_ptr = NOC_STREAM_READ_REG(stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX);

    if (is_remote_src &&
        (msg_info_wr_ptr - msg_info_buf_addr >= MAX_TILES_MSG_INFO_BUF_PER_PHASE - 2 * buf_size_tiles)) {
        prev_ack_thresh = NOC_STREAM_READ_REG(stream_id, STREAM_MEM_BUF_SPACE_AVAILABLE_ACK_THRESHOLD_REG_INDEX);
        NOC_STREAM_WRITE_REG(stream_id, STREAM_MEM_BUF_SPACE_AVAILABLE_ACK_THRESHOLD_REG_INDEX, 0);
        return true;
    }

    return false;
}

inline bool reset_tile_header_buffer(
    uint32_t stream_id,
    uint32_t msg_info_buf_addr,
    uint32_t buf_size_tiles,
    uint32_t& prev_phases_tiles_received_inc,
    uint32_t& prev_ack_thresh,
    uint32_t num_iter_tiles) {
    uint32_t msg_info_full = NOC_STREAM_READ_REG(stream_id, STREAM_MSG_INFO_FULL_REG_INDEX);
    uint32_t num_msgs_recv = NOC_STREAM_READ_REG(stream_id, STREAM_NUM_MSGS_RECEIVED_REG_INDEX);

    if (msg_info_full || (num_msgs_recv == buf_size_tiles)) {
        uint32_t buf_space_available = NOC_STREAM_READ_REG(stream_id, STREAM_BUF_SPACE_AVAILABLE_REG_INDEX);

        if (buf_space_available == 0) {
            uint32_t msg_info_rd_ptr = NOC_STREAM_READ_REG(stream_id, STREAM_MSG_INFO_PTR_REG_INDEX);
            uint32_t msg_info_wr_ptr = NOC_STREAM_READ_REG(stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX);
            num_msgs_recv = NOC_STREAM_READ_REG(stream_id, STREAM_NUM_MSGS_RECEIVED_REG_INDEX);
            uint32_t msg_info_num_tiles = msg_info_wr_ptr - msg_info_rd_ptr + num_msgs_recv;
            prev_phases_tiles_received_inc = msg_info_rd_ptr - num_msgs_recv - msg_info_buf_addr;
            NOC_STREAM_WRITE_REG(stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX, msg_info_buf_addr);
            NOC_STREAM_WRITE_REG(stream_id, STREAM_MSG_INFO_PTR_REG_INDEX, msg_info_buf_addr + num_msgs_recv);
            NOC_STREAM_WRITE_REG(stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX, msg_info_buf_addr + msg_info_num_tiles);
            NOC_STREAM_WRITE_REG(stream_id, STREAM_MEM_BUF_SPACE_AVAILABLE_ACK_THRESHOLD_REG_INDEX, prev_ack_thresh);
            return true;
        }
    }

    if (num_iter_tiles <= buf_size_tiles) {
        prev_phases_tiles_received_inc = 0;
        NOC_STREAM_WRITE_REG(stream_id, STREAM_MEM_BUF_SPACE_AVAILABLE_ACK_THRESHOLD_REG_INDEX, prev_ack_thresh);
        return true;
    }

    return false;
}

inline void check_dummy_phase(uint32_t stream_id) {
    if (stream_phase_is_active(stream_id)) {
        uint32_t cur_phase = stream_get_curr_phase(stream_id) >> EPOCH_SHIFT;
        if (cur_phase == 0x1F) {
            if (NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_MISC_CFG_REG_INDEX, SOURCE_ENDPOINT)) {
                uint32_t buf_size = NOC_STREAM_READ_REG(stream_id, STREAM_BUF_SIZE_REG_INDEX);
                uint32_t buf_space_available = NOC_STREAM_READ_REG(stream_id, STREAM_BUF_SPACE_AVAILABLE_REG_INDEX);
                uint32_t num_tiles = stream_get_curr_phase_num_msgs(stream_id);

                if (buf_space_available == buf_size && num_tiles > 0) {
                    stream_relay_tiles(stream_id, 1, 1);
                }
            }
        }
    }
}

inline bool is_dummy_phase(uint32_t stream_id) {
    uint32_t cur_phase = stream_get_curr_phase(stream_id) >> EPOCH_SHIFT;
    return cur_phase == 0x1F;
}

inline void stream_dram_write_init(uint32_t stream_id, uint32_t tile_header_buffer_addr) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_CURR_PHASE_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_HI_REG_INDEX,
        tile_header_buffer_addr >>
            21);  // todo remove this when noc0/noc1 NIU_CFG_0_TILE_HEADER_STORE_OFF is set for all dram cores
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX,
        tile_header_buffer_addr >>
            4);  // todo remove this when noc0/noc1 NIU_CFG_0_TILE_HEADER_STORE_OFF is set for all dram cores
    NOC_STREAM_WRITE_REG(stream_id, STREAM_MCAST_DEST_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_MCAST_DEST_NUM_REG_INDEX, 1);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_SCRATCH_0_REG_INDEX, (1 << 0) | (1 << 2));
    NOC_STREAM_WRITE_REG(stream_id, STREAM_SCRATCH_1_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_SCRATCH_2_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_SCRATCH_3_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_SCRATCH_4_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_SCRATCH_5_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_TRAFFIC_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_MEM_BUF_SPACE_AVAILABLE_ACK_THRESHOLD_REG_INDEX, 0);
    NOC_STREAM_WRITE_REG(stream_id, STREAM_GATHER_REG_INDEX, 0);
}

inline void stream_dram_write(
    uint32_t stream_id,
    uint32_t noc,
    uint32_t src_addr,
    uint64_t dest_addr,
    uint32_t len_bytes,
    uint32_t len_tiles,
    uint32_t vc,
    uint32_t tile_header_buf_addr_word) {
    if (len_bytes > 0) {
        uint32_t dest_buf_addr = NOC_LOCAL_ADDR_OFFSET(dest_addr);

        NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, 1 << CURR_PHASE_NUM_MSGS);

        NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_REG_INDEX, dest_addr >> NOC_ADDR_LOCAL_BITS);
        NOC_STREAM_WRITE_REG(
            stream_id,
            STREAM_REMOTE_DEST_BUF_START_HI_REG_INDEX,
            (dest_buf_addr / MEM_WORD_WIDTH) >> MEM_WORD_ADDR_WIDTH);
        NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_BUF_START_REG_INDEX, dest_buf_addr / MEM_WORD_WIDTH);
        NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, len_bytes / MEM_WORD_WIDTH);

        NOC_STREAM_WRITE_REG(stream_id, STREAM_BUF_START_REG_INDEX, src_addr / MEM_WORD_WIDTH);
        NOC_STREAM_WRITE_REG(stream_id, STREAM_BUF_SIZE_REG_INDEX, len_bytes / MEM_WORD_WIDTH);
        NOC_STREAM_WRITE_REG(stream_id, STREAM_MSG_INFO_PTR_REG_INDEX, tile_header_buf_addr_word);
        NOC_STREAM_WRITE_REG(stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX, tile_header_buf_addr_word);

        NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_TRAFFIC_REG_INDEX, (vc << UNICAST_VC_REG));

        uint32_t misc_cfg_reg = (noc << OUTGOING_DATA_NOC) | ((1 - noc) << REMOTE_SRC_UPDATE_NOC) |
                                (1 << SOURCE_ENDPOINT) | (1 << REMOTE_RECEIVER) | (1 << NEXT_PHASE_SRC_CHANGE) |
                                (1 << NEXT_PHASE_DEST_CHANGE) | (1 << DEST_DATA_BUF_NO_FLOW_CTRL);
        NOC_STREAM_WRITE_REG(stream_id, STREAM_MISC_CFG_REG_INDEX, misc_cfg_reg);
        misc_cfg_reg = (1 << PHASE_AUTO_ADVANCE) | (3 << REG_UPDATE_VC_REG);
        NOC_STREAM_WRITE_REG(stream_id, STREAM_ONETIME_MISC_CFG_REG_INDEX, misc_cfg_reg);

        NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_ADVANCE_REG_INDEX, 1);

        uint32_t src_ready_state;
        do {
            src_ready_state = (NOC_STREAM_READ_REG(stream_id, STREAM_DEBUG_STATUS_REG_INDEX + 8) >> 4) & 0x7;
        } while (src_ready_state != 4);  // SRC_READY_WAIT_ALL_DESTS

        NOC_STREAM_WRITE_REG(stream_id, STREAM_DEST_PHASE_READY_UPDATE_REG_INDEX, 1 << PHASE_READY_TWO_WAY_RESP);

        NOC_STREAM_WRITE_REG(
            stream_id,
            STREAM_SOURCE_ENDPOINT_NEW_MSG_INFO_REG_INDEX,
            ((src_addr / MEM_WORD_WIDTH) << SOURCE_ENDPOINT_NEW_MSG_ADDR) |
                ((len_bytes / MEM_WORD_WIDTH) << SOURCE_ENDPOINT_NEW_MSG_SIZE));
    }
}

inline bool stream_dram_write_ok(uint32_t stream_id) { return stream_phase_advance_wait(stream_id); }

inline bool stream_dram_writes_sent(uint32_t stream_id) { return stream_phase_advance_wait(stream_id); }

inline uint32_t stream_dram_writes_read_scratch(uint32_t stream_id, uint32_t scratch_reg_index) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_SCRATCH_0_REG_INDEX + scratch_reg_index);
}

inline void stream_dram_writes_write_scratch(uint32_t stream_id, uint32_t scratch_reg_index, uint32_t val) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_SCRATCH_0_REG_INDEX + scratch_reg_index, val);
}

inline void stream_clear_all_tiles(uint32_t stream_id) {
    uint32_t msg_info_wr_ptr;
    uint32_t msg_info_rd_ptr;
    uint32_t num_msgs_recv;
    uint32_t num_msgs_recv_in_bufs_and_mem;
    do {
        msg_info_rd_ptr = NOC_STREAM_READ_REG(stream_id, STREAM_MSG_INFO_PTR_REG_INDEX);
        num_msgs_recv = NOC_STREAM_READ_REG(stream_id, STREAM_NUM_MSGS_RECEIVED_REG_INDEX);
        msg_info_wr_ptr = NOC_STREAM_READ_REG(stream_id, STREAM_MSG_INFO_WR_PTR_REG_INDEX);
        num_msgs_recv_in_bufs_and_mem = msg_info_wr_ptr - msg_info_rd_ptr + num_msgs_recv;
        if (num_msgs_recv > 0) {
            stream_receiver_endpoint_single_clear_op(stream_id, 1);
        }
    } while (num_msgs_recv_in_bufs_and_mem > 0);
}

#endif  // ndef _STREAM_INTERFACE_H_
