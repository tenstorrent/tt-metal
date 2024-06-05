// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "stream_interface.h"
#include "tests/tt_metal/tt_metal/test_kernels/dataflow/streams/stream_io_kernel_helpers.hpp"
#include "tt_metal/hw/inc/wormhole/noc/noc_overlay_parameters.h"

void kernel_main() {
    // Work to do before productizable:
    // - Test phase advance
    //   - test > 2k messages (and > 4k messages)
    // - Test variable sized messages
    // - Test rerun after test completion (without reset)
    //   - Currently a bug where the phase ID persists from prior run
    //

    uint32_t arg_idx = 0;

    uint32_t relay_stream_overlay_blob_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t stream_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t stream_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t stream_buffer_size = get_arg_val<uint32_t>(arg_idx++);
    uint32_t stream_tile_header_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t stream_tile_header_max_num_messages = get_arg_val<uint32_t>(arg_idx++);

    uint32_t remote_src_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_src_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_src_stream_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_src_noc_id = get_arg_val<uint32_t>(arg_idx++);

    uint32_t remote_dest_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_dest_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_dest_noc_stream_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_dest_noc_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_dest_buf_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_dest_buf_size_4B_words = get_arg_val<uint32_t>(arg_idx++);
    uint32_t remote_dest_tile_header_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    volatile uint32_t* tx_rx_done_semaphore_addr =
        reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    bool is_first_relay_stream_in_chain = get_arg_val<uint32_t>(arg_idx++) == 1;

    uint32_t remote_src_start_phase_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t dest_remote_src_start_phase_addr = get_arg_val<uint32_t>(arg_idx++);

    *tx_rx_done_semaphore_addr = 0;  // should already be set to 0, but why not...
    // use stream_buffer_addr as temporary storage just for this initial setup

    const uint32_t local_first_phase = notify_remote_receiver_of_starting_phase(
        stream_id,
        stream_buffer_addr + 16,  // local storage to hold the phase while async send in progress, 16B for noc alignment
        get_noc_addr(remote_dest_noc_x, remote_dest_noc_y, dest_remote_src_start_phase_addr));
    const uint32_t local_second_phase = local_first_phase + 1;

    // If first relay, we'd expect this to be stream_tile_header_max_num_messages + STARTING_PHASE because the
    // remote_sender (FW managed) is programmed as one phase per message and there are
    // `stream_tile_header_max_num_messages` messages in this stream's phase. If second relay, we'd expect this to be
    // SECOND_PHASE
    const uint32_t first_phase_remote_src_phase =
        wait_for_remote_source_starting_phase(reinterpret_cast<volatile uint32_t*>(remote_src_start_phase_addr));
    const uint32_t second_phase_remote_src_phase =
        is_first_relay_stream_in_chain ? stream_tile_header_max_num_messages + first_phase_remote_src_phase
                                       : first_phase_remote_src_phase + 1;

    // Setup the stream phases
    volatile uint32_t* stream_phases_start = reinterpret_cast<volatile uint32_t*>(relay_stream_overlay_blob_addr);

    //
    // phase 1
    //

    const uint32_t stream_phase_1_start = reinterpret_cast<uint32_t>(stream_phases_start);
    volatile uint32_t* stream_phase_1_reg_addr = reinterpret_cast<volatile uint32_t*>(stream_phase_1_start) + 1;

    // Local stream buffer address register
    *stream_phase_1_reg_addr = blob_cfg_dw(STREAM_BUF_START_REG_INDEX, stream_buffer_addr >> 4);
    stream_phase_1_reg_addr++;
    *stream_phase_1_reg_addr = 0;

    // Local stream buffer size register
    *stream_phase_1_reg_addr = blob_cfg_dw(STREAM_BUF_SIZE_REG_INDEX, stream_buffer_size >> 4);
    stream_phase_1_reg_addr++;
    *stream_phase_1_reg_addr = 0;

    // msg info rdptr
    *stream_phase_1_reg_addr = blob_cfg_dw(STREAM_MSG_INFO_PTR_REG_INDEX, stream_tile_header_buffer_addr >> 4);
    stream_phase_1_reg_addr++;
    *stream_phase_1_reg_addr = 0;

    // msg info wrptr
    *stream_phase_1_reg_addr = blob_cfg_dw(STREAM_MSG_INFO_WR_PTR_REG_INDEX, stream_tile_header_buffer_addr >> 4);
    stream_phase_1_reg_addr++;
    *stream_phase_1_reg_addr = 0;

    // Local stream buffer size register
    *stream_phase_1_reg_addr =
        blob_cfg_dw(STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX, remote_dest_tile_header_buffer_addr >> 4);
    stream_phase_1_reg_addr++;
    *stream_phase_1_reg_addr = 0;

    // STREAM_MISC_CFG_REG_INDEX
    const uint32_t remote_src_update_noc_id = 1 - remote_src_noc_id;
    uint32_t stream_msc_cfg_reg = 0;
    stream_msc_cfg_reg =
        set_blob_reg_field(stream_msc_cfg_reg, INCOMING_DATA_NOC_WIDTH, INCOMING_DATA_NOC, remote_src_noc_id);
    stream_msc_cfg_reg =
        set_blob_reg_field(stream_msc_cfg_reg, OUTGOING_DATA_NOC_WIDTH, OUTGOING_DATA_NOC, remote_dest_noc_id);
    stream_msc_cfg_reg = set_blob_reg_field(
        stream_msc_cfg_reg, REMOTE_SRC_UPDATE_NOC_WIDTH, REMOTE_SRC_UPDATE_NOC, remote_src_update_noc_id);
    stream_msc_cfg_reg = set_blob_reg_field(stream_msc_cfg_reg, REMOTE_SOURCE_WIDTH, REMOTE_SOURCE, 1);
    stream_msc_cfg_reg = set_blob_reg_field(stream_msc_cfg_reg, REMOTE_RECEIVER_WIDTH, REMOTE_RECEIVER, 1);
    stream_msc_cfg_reg = set_blob_reg_field(stream_msc_cfg_reg, PHASE_AUTO_CONFIG_WIDTH, PHASE_AUTO_CONFIG, 1);
    stream_msc_cfg_reg = set_blob_reg_field(stream_msc_cfg_reg, PHASE_AUTO_ADVANCE_WIDTH, PHASE_AUTO_ADVANCE, 1);
    stream_msc_cfg_reg = set_blob_reg_field(stream_msc_cfg_reg, DATA_AUTO_SEND_WIDTH, DATA_AUTO_SEND, 1);
    stream_msc_cfg_reg =
        set_blob_reg_field(stream_msc_cfg_reg, NEXT_PHASE_DEST_CHANGE_WIDTH, NEXT_PHASE_DEST_CHANGE, 1);
    stream_msc_cfg_reg = set_blob_reg_field(stream_msc_cfg_reg, NEXT_PHASE_SRC_CHANGE_WIDTH, NEXT_PHASE_SRC_CHANGE, 1);
    stream_msc_cfg_reg = set_blob_reg_field(stream_msc_cfg_reg, UNICAST_VC_REG_WIDTH, UNICAST_VC_REG, 0);
    stream_msc_cfg_reg = set_blob_reg_field(stream_msc_cfg_reg, REG_UPDATE_VC_REG_WIDTH, REG_UPDATE_VC_REG, 1);
    stream_msc_cfg_reg = set_blob_reg_field(stream_msc_cfg_reg, DATA_BUF_NO_FLOW_CTRL_WIDTH, DATA_BUF_NO_FLOW_CTRL, 0);
    stream_msc_cfg_reg =
        set_blob_reg_field(stream_msc_cfg_reg, DEST_DATA_BUF_NO_FLOW_CTRL_WIDTH, DEST_DATA_BUF_NO_FLOW_CTRL, 0);
    stream_msc_cfg_reg = set_blob_reg_field(stream_msc_cfg_reg, REMOTE_SRC_IS_MCAST_WIDTH, REMOTE_SRC_IS_MCAST, 0);
    stream_msc_cfg_reg = set_blob_reg_field(
        stream_msc_cfg_reg, NO_PREV_PHASE_OUTGOING_DATA_FLUSH_WIDTH, NO_PREV_PHASE_OUTGOING_DATA_FLUSH, 0);
    *stream_phase_1_reg_addr = blob_cfg_dw(STREAM_MISC_CFG_REG_INDEX, stream_msc_cfg_reg);
    stream_phase_1_reg_addr++;
    *stream_phase_1_reg_addr = 0;

    // remote src
    // Remote src noc x/y is based on the update noc (because it sends updates, NOT data, to src, so it needs update
    // noc)
    uint32_t stream_remote_src_reg = 0;
    uint32_t data_noc_in_src_noc_x =
        remote_src_update_noc_id == 0 ? remote_src_noc_x : noc_size_x - 1 - remote_src_noc_x;
    uint32_t data_noc_in_src_noc_y =
        remote_src_update_noc_id == 0 ? remote_src_noc_y : noc_size_y - 1 - remote_src_noc_y;
    stream_remote_src_reg = set_blob_reg_field(
        stream_remote_src_reg, STREAM_REMOTE_SRC_X_WIDTH, STREAM_REMOTE_SRC_X, data_noc_in_src_noc_x);
    stream_remote_src_reg = set_blob_reg_field(
        stream_remote_src_reg, STREAM_REMOTE_SRC_Y_WIDTH, STREAM_REMOTE_SRC_Y, data_noc_in_src_noc_y);
    stream_remote_src_reg = set_blob_reg_field(
        stream_remote_src_reg, REMOTE_SRC_STREAM_ID_WIDTH, REMOTE_SRC_STREAM_ID, remote_src_stream_id);
    *stream_phase_1_reg_addr = blob_cfg_dw(STREAM_REMOTE_SRC_REG_INDEX, stream_remote_src_reg);
    stream_phase_1_reg_addr++;
    *stream_phase_1_reg_addr = 0;

    // remote dest
    // Remote dest noc x/y is NOT based on the update noc (because it is sending data to the dest, so it needs data noc)
    uint32_t stream_remote_dest_reg = 0;
    uint32_t data_noc_out_dest_noc_x = remote_dest_noc_id == 0 ? remote_dest_noc_x : noc_size_x - 1 - remote_dest_noc_x;
    uint32_t data_noc_out_dest_noc_y = remote_dest_noc_id == 0 ? remote_dest_noc_y : noc_size_y - 1 - remote_dest_noc_y;
    stream_remote_dest_reg = set_blob_reg_field(
        stream_remote_dest_reg, STREAM_REMOTE_DEST_X_WIDTH, STREAM_REMOTE_DEST_X, data_noc_out_dest_noc_x);
    stream_remote_dest_reg = set_blob_reg_field(
        stream_remote_dest_reg, STREAM_REMOTE_DEST_Y_WIDTH, STREAM_REMOTE_DEST_Y, data_noc_out_dest_noc_y);
    stream_remote_dest_reg = set_blob_reg_field(
        stream_remote_dest_reg,
        STREAM_REMOTE_DEST_STREAM_ID_WIDTH,
        STREAM_REMOTE_DEST_STREAM_ID,
        remote_dest_noc_stream_id);
    *stream_phase_1_reg_addr = blob_cfg_dw(STREAM_REMOTE_DEST_REG_INDEX, stream_remote_dest_reg);
    stream_phase_1_reg_addr++;
    *stream_phase_1_reg_addr = 0;

    // remote_dest buf start
    uint32_t stream_remote_dest_buf_start_reg_val = 0;
    stream_remote_dest_buf_start_reg_val = set_blob_reg_field(
        stream_remote_dest_buf_start_reg_val,
        DRAM_WRITES__SCRATCH_1_PTR_LO_WIDTH,
        DRAM_WRITES__SCRATCH_1_PTR_LO,
        remote_dest_buf_addr >> 4);
    *stream_phase_1_reg_addr =
        blob_cfg_dw(STREAM_REMOTE_DEST_BUF_START_REG_INDEX, stream_remote_dest_buf_start_reg_val);
    stream_phase_1_reg_addr++;
    *stream_phase_1_reg_addr = 0;

    // remote_dest buf size
    uint32_t stream_remote_dest_buf_size_reg = 0;
    stream_remote_dest_buf_size_reg = set_blob_reg_field(
        stream_remote_dest_buf_size_reg,
        REMOTE_DEST_BUF_SIZE_WORDS_WIDTH,
        REMOTE_DEST_BUF_SIZE_WORDS,
        remote_dest_buf_size_4B_words >> 4);
    *stream_phase_1_reg_addr = blob_cfg_dw(STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, stream_remote_dest_buf_size_reg);
    stream_phase_1_reg_addr++;
    *stream_phase_1_reg_addr = 0;

    *stream_phase_1_reg_addr = blob_cfg_dw(STREAM_CURR_PHASE_BASE_REG_INDEX, 0);
    stream_phase_1_reg_addr++;
    *stream_phase_1_reg_addr = 0;

    *stream_phase_1_reg_addr = blob_cfg_dw(STREAM_REMOTE_SRC_PHASE_REG_INDEX, first_phase_remote_src_phase);
    stream_phase_1_reg_addr++;
    *stream_phase_1_reg_addr = 0;

    *stream_phase_1_reg_addr = blob_cfg_dw(STREAM_CURR_PHASE_REG_INDEX, local_first_phase);
    stream_phase_1_reg_addr++;
    *stream_phase_1_reg_addr = 0;

    *stream_phase_1_reg_addr = blob_cfg_dw(STREAM_MEM_BUF_SPACE_AVAILABLE_ACK_THRESHOLD_REG_INDEX, 0);
    stream_phase_1_reg_addr++;
    *stream_phase_1_reg_addr = 0;

    //
    // phase 2 - we're unrolling one iteration of the first phase, so the second phase is mostly identical
    //
    volatile uint32_t* const stream_phase_2_start = stream_phase_1_reg_addr;
    volatile uint32_t* stream_phase_2_stream_reg_addr = stream_phase_2_start + 1;

    *stream_phase_2_stream_reg_addr = blob_cfg_dw(STREAM_BUF_START_REG_INDEX, stream_buffer_addr >> 4);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    // Local stream buffer size register
    *stream_phase_2_stream_reg_addr = blob_cfg_dw(STREAM_BUF_SIZE_REG_INDEX, stream_buffer_size >> 4);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    // msg info rdptr
    *stream_phase_2_stream_reg_addr = blob_cfg_dw(STREAM_MSG_INFO_PTR_REG_INDEX, stream_tile_header_buffer_addr >> 4);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    // msg info wrptr
    *stream_phase_2_stream_reg_addr =
        blob_cfg_dw(STREAM_MSG_INFO_WR_PTR_REG_INDEX, stream_tile_header_buffer_addr >> 4);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    *stream_phase_2_stream_reg_addr =
        blob_cfg_dw(STREAM_REMOTE_DEST_MSG_INFO_WR_PTR_REG_INDEX, remote_dest_tile_header_buffer_addr >> 4);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    *stream_phase_2_stream_reg_addr = blob_cfg_dw(STREAM_MISC_CFG_REG_INDEX, stream_msc_cfg_reg);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    *stream_phase_2_stream_reg_addr = blob_cfg_dw(STREAM_REMOTE_SRC_REG_INDEX, stream_remote_src_reg);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    *stream_phase_2_stream_reg_addr = blob_cfg_dw(STREAM_REMOTE_DEST_REG_INDEX, stream_remote_dest_reg);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    *stream_phase_2_stream_reg_addr =
        blob_cfg_dw(STREAM_REMOTE_DEST_BUF_START_REG_INDEX, stream_remote_dest_buf_start_reg_val);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    *stream_phase_2_stream_reg_addr =
        blob_cfg_dw(STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, stream_remote_dest_buf_size_reg);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    *stream_phase_2_stream_reg_addr = blob_cfg_dw(STREAM_CURR_PHASE_BASE_REG_INDEX, 0);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    *stream_phase_2_stream_reg_addr = blob_cfg_dw(STREAM_CURR_PHASE_REG_INDEX, local_second_phase);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    *stream_phase_2_stream_reg_addr = blob_cfg_dw(STREAM_MEM_BUF_SPACE_AVAILABLE_ACK_THRESHOLD_REG_INDEX, 0);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    *stream_phase_2_stream_reg_addr = blob_cfg_dw(STREAM_PHASE_AUTO_CFG_PTR_BASE_REG_INDEX, 0);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    *stream_phase_2_stream_reg_addr = blob_cfg_dw(STREAM_REMOTE_SRC_PHASE_REG_INDEX, second_phase_remote_src_phase);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    *stream_phase_2_stream_reg_addr = blob_cfg_dw(STREAM_PHASE_AUTO_CFG_PTR_REG_INDEX, stream_phase_1_start);
    stream_phase_2_stream_reg_addr++;
    *stream_phase_2_stream_reg_addr = 0;

    const uint32_t phase_1_num_cfg_regs =
        ((reinterpret_cast<uint32_t>(stream_phase_1_reg_addr) >> 2) - (stream_phase_1_start >> 2)) - 1;
    uint32_t phase_2_num_cfg_regs = ((reinterpret_cast<uint32_t>(stream_phase_2_stream_reg_addr) >> 2) -
                                     (reinterpret_cast<uint32_t>(stream_phase_2_start) >> 2)) -
                                    1;

    // We're supposed to put the **next** phase num config registers in the **current** phase's blob header. This means
    // we need to flip the register counts between the two phases for their headers So in a sequence of 3 phases, the
    // header blob on phase 1 would need the #cfg regs for phase 2. Phase 2's cfg header blob would need the #cfg regs
    // for phase 3 and for phase 3, the #cfg regs in the header blob would be 0 (since no phase follows it) In our case,
    // we just need to point to the opposite phase's #cfg regs
    *reinterpret_cast<volatile uint32_t*>(stream_phase_1_start) =
        blob_header_dw(phase_2_num_cfg_regs, stream_tile_header_max_num_messages, 1);
    *stream_phase_2_start = blob_header_dw(phase_1_num_cfg_regs, stream_tile_header_max_num_messages, 1);

    // Now kick off the stream
    stream_phase_blob_run(
        stream_id,
        reinterpret_cast<volatile uint32_t*>(stream_phase_1_start),
        stream_tile_header_max_num_messages,
        phase_1_num_cfg_regs);

    // Wait for sender and receiver to signal completion
    while (*tx_rx_done_semaphore_addr != 2) {
        asm volatile("nop");
    }

    // Now teardown the stream
    // Unknown if it's safe to reset the stream while it's in a state before active
    while ((NOC_STREAM_READ_REG(stream_id, STREAM_DEBUG_STATUS_REG_INDEX + 9) >> MEM_WORD_ADDR_WIDTH) != 0 ||
           !stream_phase_is_active(stream_id)) {
        asm volatile("nop");
    }

    stream_reset(stream_id);
    ASSERT(!assert_check(stream_id, false));
    for (auto ptr = reinterpret_cast<volatile uint32_t*>(stream_phase_1_start); ptr != stream_phase_2_stream_reg_addr;
         ptr++) {
        *ptr = 0;
    }
}
