// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "dataflow_api.h"
#include "stream_interface.h"
#include "tt_metal/hw/inc/wormhole/noc/noc_overlay_parameters.h"

struct stream_state_t {
    const uint32_t local_data_buffer_base_address;
    const uint32_t local_msg_info_ptr_base_address;

    uint32_t local_phase_id;
    uint32_t messages_per_phase;
    uint32_t msg_info_wrptr_addr;

    uint32_t num_tiles_sent;
    uint32_t tile_header_num_msgs;

    uint32_t local_buffer_base_addr;
    uint32_t local_buffer_size;
    uint32_t local_msg_info_ptr;
    uint32_t local_buffer_read_offset;

    uint32_t remote_buffer_base_addr;
    uint32_t remote_buffer_size;
    uint32_t remote_msg_info_ptr;
    uint32_t remote_buffer_write_offset;

    uint32_t remote_phase_id;

    uint32_t get_current_local_buffer_address() const {
        return local_data_buffer_base_address + local_buffer_read_offset;
    }
};

struct phase_iterator_t {
    phase_iterator_t(uint32_t start_phase, uint32_t max_phase) :
        phase_id(start_phase), max_phase(max_phase), start_phase(start_phase) {}
    uint32_t phase_id;
    uint32_t max_phase;
    uint32_t start_phase;

    FORCE_INLINE uint32_t get() const { return phase_id; }

    FORCE_INLINE void increment() { phase_id = phase_id == max_phase ? start_phase : phase_id + 1; }
};

struct noc_endpoint_info_t {
    uint32_t data_noc_id;
    uint32_t update_noc_id;
    uint32_t noc_x;
    uint32_t noc_y;
};

#define STREAM_CFG(field, val) ((val) << (field))

#define AUTO_CFG_HEADER(next_phase_num_cfg_reg_writes, curr_phase_num_msgs, phase_num_incr) \
    ((uint32_t)(((next_phase_num_cfg_reg_writes) << 24) | ((curr_phase_num_msgs) << 12) | (phase_num_incr)))

#define STREAM_REMOTE_DEST(dest_x, dest_y, dest_stream_id)                     \
    (((dest_x) << STREAM_REMOTE_DEST_X) | ((dest_y) << STREAM_REMOTE_DEST_Y) | \
     ((dest_stream_id) << STREAM_REMOTE_DEST_STREAM_ID))

#define STREAM_REMOTE_SRC(src_x, src_y, src_stream_id) \
    (((src_x) << STREAM_REMOTE_SRC_X) | ((src_y) << STREAM_REMOTE_SRC_Y) | ((src_stream_id) << REMOTE_SRC_STREAM_ID))

FORCE_INLINE uint32_t
blob_header_dw(uint32_t next_phase_num_cfg_reg_writes, uint32_t curr_phase_num_msgs, uint32_t phase_num_incr) {
    return (next_phase_num_cfg_reg_writes << 24) | (curr_phase_num_msgs << 12) | phase_num_incr;
}

FORCE_INLINE void stream_phase_blob_run(
    uint32_t stream_id, volatile uint32_t *blob_start_addr, uint32_t start_phase_num_cfg_regs) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX, reinterpret_cast<uint32_t>(blob_start_addr));
    NOC_STREAM_WRITE_REG(
        stream_id, STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX, start_phase_num_cfg_regs << NEXT_PHASE_NUM_CFG_REG_WRITES);
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_MISC_CFG_REG_INDEX,
        (0x1 << PHASE_AUTO_CONFIG) | (1 << NEXT_PHASE_SRC_CHANGE) | (1 << NEXT_PHASE_DEST_CHANGE));
}
FORCE_INLINE void stream_phase_blob_run(
    uint32_t stream_id,
    volatile uint32_t *blob_start_addr,
    uint32_t num_messages_per_phase,
    uint32_t start_phase_num_cfg_regs) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX, reinterpret_cast<uint32_t>(blob_start_addr));

    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_PHASE_AUTO_CFG_HEADER_REG_INDEX,
        blob_header_dw(start_phase_num_cfg_regs, num_messages_per_phase, 1));
    NOC_STREAM_WRITE_REG(
        stream_id,
        STREAM_MISC_CFG_REG_INDEX,
        (0x1 << PHASE_AUTO_ADVANCE) | (0x1 << PHASE_AUTO_CONFIG) | (1 << NEXT_PHASE_SRC_CHANGE) |
            (1 << NEXT_PHASE_DEST_CHANGE));
    NOC_STREAM_WRITE_REG(stream_id, STREAM_PHASE_ADVANCE_REG_INDEX, 1);
}

FORCE_INLINE uint32_t blob_cfg_dw(uint32_t reg_index, uint32_t reg_val) { return (reg_val << 8) | reg_index; }

FORCE_INLINE uint32_t set_blob_reg_field(uint32_t blob_dw, uint32_t field_width, uint32_t field_offset, uint32_t val) {
    uint32_t mask = ((1 << field_width) - 1) << field_offset;
    return (blob_dw & ~mask) | ((val << field_offset) & mask);
}

FORCE_INLINE uint32_t get_first_available_phase_out_of_reset(uint32_t stream_id) {
    uint32_t stream_phase_coming_out_of_reset = stream_get_curr_phase(stream_id);
    return (
        stream_phase_coming_out_of_reset < 4096   ? 4096 : 1);
}

FORCE_INLINE uint32_t notify_remote_receiver_of_starting_phase(
    uint32_t stream_id, uint32_t local_buffer_addr, uint64_t remote_receiver_noc_addr) {
    uint32_t starting_phase = get_first_available_phase_out_of_reset(stream_id);
    ASSERT(starting_phase > 0);
    *reinterpret_cast<volatile uint32_t *>(local_buffer_addr) = starting_phase;
    noc_async_write(local_buffer_addr, remote_receiver_noc_addr, sizeof(uint32_t));
    // noc_semaphore_set_remote(local_buffer_addr, remote_receiver_noc_addr);
    noc_async_writes_flushed();
    return starting_phase;
}

FORCE_INLINE uint32_t wait_for_remote_source_starting_phase(volatile uint32_t *addr) {
    while (*addr == 0) {
        asm volatile("nop");
    }
    return *addr;
}
