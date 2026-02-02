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

inline uint32_t NOC1_X_ID(uint32_t x) { return NOC_X_SIZE - 1 - x; }

inline uint32_t NOC1_Y_ID(uint32_t y) { return NOC_Y_SIZE - 1 - y; }

inline __attribute__((always_inline)) uint32_t stream_phase_is_active(uint32_t stream_id) {
    return NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_WAIT_STATUS_REG_INDEX, MSG_FWD_ONGOING);
}

inline __attribute__((always_inline)) uint32_t stream_get_curr_phase(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_CURR_PHASE_REG_INDEX);
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
    uint32_t val = NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_SRC_REG_INDEX);
    x = NOC_STREAM_GET_REG_FIELD(val, STREAM_REMOTE_SRC_X);
    y = NOC_STREAM_GET_REG_FIELD(val, STREAM_REMOTE_SRC_Y);
}

inline __attribute__((always_inline)) uint32_t stream_get_output_noc(uint32_t stream_id) {
    return NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_MISC_CFG_REG_INDEX, OUTGOING_DATA_NOC);
}

inline __attribute__((always_inline)) uint32_t stream_get_output_unicast_vc(uint32_t stream_id) {
    return NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_MISC_CFG_REG_INDEX, UNICAST_VC_REG);
}

inline __attribute__((always_inline)) void stream_get_remote_dest_coord(uint32_t stream_id, uint32_t& x, uint32_t& y) {
    uint32_t val = NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_REG_INDEX);
    x = NOC_STREAM_GET_REG_FIELD(val, STREAM_REMOTE_DEST_X);
    y = NOC_STREAM_GET_REG_FIELD(val, STREAM_REMOTE_DEST_Y);
}

inline __attribute__((always_inline)) uint32_t stream_get_data_buf_addr(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_BUF_START_REG_INDEX);
}

inline __attribute__((always_inline)) uint32_t stream_get_data_buf_size(uint32_t stream_id) {
    return NOC_STREAM_READ_REG(stream_id, STREAM_BUF_SIZE_REG_INDEX);
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
           !NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_MISC_CFG_REG_INDEX, PHASE_AUTO_CONFIG);
}

inline __attribute__((always_inline)) int stream_get_curr_phase_num_msgs(uint32_t stream_id) {
    return NOC_STREAM_READ_REG_FIELD(stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, CURR_PHASE_NUM_MSGS);
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

template <bool fracture = false>
static __attribute__((unused)) __attribute__((noinline)) bool stream_get_push_flushed(uint32_t stream_id) {
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
        uint32_t cur_phase = stream_get_curr_phase(stream_id);
        if (cur_phase == prev_phase) {
            if constexpr (fracture) {
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

inline void stream_reset(uint32_t stream_id) {
    uint32_t val = NOC_STREAM_READ_REG(stream_id, STREAM_MISC_CFG_REG_INDEX);
    val &= (~(1 << PHASE_AUTO_CONFIG));
    NOC_STREAM_WRITE_REG(stream_id, STREAM_MISC_CFG_REG_INDEX, val);  // disable auto-config
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

inline void check_dummy_phase(uint32_t stream_id) {
    if (stream_phase_is_active(stream_id)) {
        uint32_t cur_phase = stream_get_curr_phase(stream_id);
        if (cur_phase == 0xFFFFF) {
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
    uint32_t cur_phase = stream_get_curr_phase(stream_id);
    return cur_phase == 0xFFFFF;
}

#endif  // ndef _STREAM_INTERFACE_H_
