// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include <tools/profiler/kernel_profiler.hpp>

// ============================================================================
// Persistent multicast sender helpers (from deepseek mcast.hpp)
// ============================================================================

template <uint8_t noc>
FORCE_INLINE uint64_t get_noc_multicast_addr(
    uint32_t noc_x_start, uint32_t noc_y_start, uint32_t noc_x_end, uint32_t noc_y_end, uint32_t addr) {
    if constexpr (noc == 0) {
        return NOC_MULTICAST_ADDR(
            DYNAMIC_NOC_X(noc, noc_x_start),
            DYNAMIC_NOC_Y(noc, noc_y_start),
            DYNAMIC_NOC_X(noc, noc_x_end),
            DYNAMIC_NOC_Y(noc, noc_y_end),
            addr);
    } else {
        return NOC_MULTICAST_ADDR(
            DYNAMIC_NOC_X(noc, noc_x_end),
            DYNAMIC_NOC_Y(noc, noc_y_end),
            DYNAMIC_NOC_X(noc, noc_x_start),
            DYNAMIC_NOC_Y(noc, noc_y_start),
            addr);
    }
}

template <
    uint32_t mcast_num_cores,
    bool loopback,
    bool is_part_of_receiver_grid,
    bool linked,
    bool posted,
    bool set_addresses,
    bool set_size,
    uint8_t cmd_buf>
FORCE_INLINE void mcast_send_set_state(uint32_t src_local_addr, uint64_t dst_noc_addr, uint32_t len_bytes = 0) {
    constexpr uint32_t noc = noc_index;
    constexpr uint32_t vc = NOC_MULTICAST_WRITE_VC;
    constexpr bool multicast_path_reserve = true;

    while (!noc_cmd_buf_ready(noc, cmd_buf));
    uint32_t noc_cmd_field = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) |
                             (linked ? NOC_CMD_VC_LINKED : 0x0) | (multicast_path_reserve ? NOC_CMD_PATH_RESERVE : 0) |
                             (loopback ? NOC_CMD_BRCST_SRC_INCLUDE : 0) | NOC_CMD_BRCST_PACKET |
                             (posted ? 0 : NOC_CMD_RESP_MARKED);

    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CTRL, noc_cmd_field);
    // Handles writing to PCIe
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_MID, (uint32_t)(dst_noc_addr >> 32) & NOC_PCIE_MASK);
    NOC_CMD_BUF_WRITE_REG(
        noc, cmd_buf, NOC_RET_ADDR_COORDINATE, (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_BRCST_EXCLUDE, 0);
    if constexpr (set_size) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    }
    if constexpr (set_addresses) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_local_addr);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, (uint32_t)(dst_noc_addr));
    }
}

template <
    uint32_t mcast_num_cores,
    bool loopback,
    bool is_part_of_receiver_grid,
    bool linked,
    bool posted,
    bool set_addresses,
    bool set_size,
    uint8_t cmd_buf>
FORCE_INLINE void mcast_send_with_state(uint32_t src_local_addr, uint32_t dst_local_addr, uint32_t len_bytes = 0) {
    constexpr uint32_t noc = noc_index;
    if constexpr (loopback) {
        static_assert(is_part_of_receiver_grid, "Loopback mode is only supported for receiver grid");
    }
    constexpr uint32_t num_dests =
        loopback ? mcast_num_cores : (is_part_of_receiver_grid ? mcast_num_cores - 1 : mcast_num_cores);

    if constexpr (noc_mode == DM_DYNAMIC_NOC) {
        if constexpr (posted) {
            inc_noc_counter_val<proc_type, NocBarrierType::POSTED_WRITES_NUM_ISSUED>(noc, 1);
        } else {
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_NUM_ISSUED>(noc, 1);
            inc_noc_counter_val<proc_type, NocBarrierType::NONPOSTED_WRITES_ACKED>(noc, num_dests);
        }
    }

    while (!noc_cmd_buf_ready(noc, cmd_buf));

    if constexpr (set_size) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_AT_LEN_BE, len_bytes);
    }
    if constexpr (set_addresses) {
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_TARG_ADDR_LO, src_local_addr);
        NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_RET_ADDR_LO, dst_local_addr);
    }
    NOC_CMD_BUF_WRITE_REG(noc, cmd_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    if constexpr (noc_mode == DM_DEDICATED_NOC) {
        if constexpr (posted) {
            noc_posted_writes_num_issued[noc] += 1;
        } else {
            noc_nonposted_writes_num_issued[noc] += 1;
            noc_nonposted_writes_acked[noc] += num_dests;
        }
    }
}

template <uint32_t mcast_num_cores, bool loopback, bool is_part_of_receiver_grid>
FORCE_INLINE void init_persistent_mcast_sender(
    uint64_t mcast_flag_noc_addr,
    uint32_t data_sender_semaphore_addr,
    volatile tt_l1_ptr uint32_t* data_sender_semaphore_addr_ptr) {
    mcast_send_set_state<mcast_num_cores, loopback, is_part_of_receiver_grid, true, true, false, false, write_cmd_buf>(
        0, mcast_flag_noc_addr, 0);
    mcast_send_set_state<
        mcast_num_cores,
        loopback,
        is_part_of_receiver_grid,
        true,
        true,
        true,
        true,
        write_reg_cmd_buf>(data_sender_semaphore_addr, mcast_flag_noc_addr, 4);
    mcast_send_with_state<
        mcast_num_cores,
        loopback,
        is_part_of_receiver_grid,
        true,
        true,
        false,
        false,
        write_reg_cmd_buf>(0, 0, 0);
    noc_async_posted_writes_flushed();
    noc_semaphore_set(data_sender_semaphore_addr_ptr, VALID);
}

template <uint32_t mcast_num_cores, bool loopback, bool is_part_of_receiver_grid>
FORCE_INLINE void teardown_persistent_mcast_sender(uint64_t mcast_flag_noc_addr) {
    mcast_send_set_state<
        mcast_num_cores,
        loopback,
        is_part_of_receiver_grid,
        false,
        false,
        false,
        false,
        write_reg_cmd_buf>(0, mcast_flag_noc_addr, 0);
    mcast_send_with_state<
        mcast_num_cores,
        loopback,
        is_part_of_receiver_grid,
        false,
        false,
        false,
        false,
        write_reg_cmd_buf>(0, 0, 0);
    noc_async_write_barrier();
    riscv_wait(1000);  // This is just to guarantee safety due to posted mcast hw bug
}

FORCE_INLINE void wait_for_gather(
    volatile tt_l1_ptr uint32_t* mcast_receiver_semaphore_addr_ptr, uint32_t num_senders) {
    DeviceZoneScopedN("wait_for_gather");

    // Wait for all senders to finish sending data
    noc_semaphore_wait(mcast_receiver_semaphore_addr_ptr, num_senders);

    // Reset the local semaphore for reuse
    noc_semaphore_set(mcast_receiver_semaphore_addr_ptr, 0);
}

template <uint32_t mcast_num_cores>
FORCE_INLINE void mcast(
    uint32_t mcast_cb,
    uint32_t mcast_dest_base_addr,
    uint32_t mcast_sender_semaphore_addr,
    uint64_t mcast_sender_noc_coord_x_start,
    uint64_t mcast_sender_noc_coord_y_start,
    uint64_t mcast_sender_noc_coord_x_end,
    uint64_t mcast_sender_noc_coord_y_end,
    uint64_t mcast_semaphore_noc_addr) {
    DeviceZoneScopedN("mcast");
    // Mcast to all cores using persistent sender
    const uint64_t mcast_data_noc_addr = get_noc_multicast_addr<noc_index>(
        mcast_sender_noc_coord_x_start,
        mcast_sender_noc_coord_y_start,
        mcast_sender_noc_coord_x_end,
        mcast_sender_noc_coord_y_end,
        mcast_dest_base_addr);

    // Set up state for data send (coordinates and addresses)
    const uint32_t src_addr = get_read_ptr(mcast_cb);
    const uint32_t len_bytes = get_tile_size(mcast_cb) * mcast_num_cores;
    mcast_send_set_state<
        mcast_num_cores,
        false,  // loopback: mcast core is not part of sender grid
        false,  // is_part_of_receiver_grid: mcast core is not part of sender grid
        true,   // linked
        true,   // posted
        true,   // set_addresses: set source and destination addresses
        true,   // set_size: set transfer size
        write_cmd_buf>(src_addr, mcast_data_noc_addr, len_bytes);

    // Send data using persistent sender
    mcast_send_with_state<
        mcast_num_cores,
        false,  // loopback: mcast core is not part of sender grid
        false,  // is_part_of_receiver_grid: mcast core is not part of sender grid
        true,   // linked
        true,   // posted
        false,  // set_addresses: already set in mcast_send_set_state
        false,  // set_size: already set in mcast_send_set_state
        write_cmd_buf>(0, 0, 0);

    // Use L1 scratch to hold VALID value for multicast semaphore set
    uint32_t semaphore_valid_addr = mcast_dest_base_addr;
    volatile tt_l1_ptr uint32_t* semaphore_valid_addr_ptr = (volatile tt_l1_ptr uint32_t*)semaphore_valid_addr;
    semaphore_valid_addr_ptr[0] = VALID;

    // Multicast semaphore set to all sender cores using persistent sender
    mcast_send_with_state<
        mcast_num_cores,
        false,  // loopback
        false,  // is_part_of_receiver_grid
        true,   // linked
        true,   // posted
        true,   // set_addresses
        true,   // set_size
        write_reg_cmd_buf>(semaphore_valid_addr, (uint32_t)(mcast_semaphore_noc_addr), 4);
    mcast_send_with_state<
        mcast_num_cores,
        false,  // loopback
        false,  // is_part_of_receiver_grid
        true,   // linked
        true,   // posted
        false,  // set_addresses
        false,  // set_size
        write_reg_cmd_buf>(0, 0, 0);
    noc_async_posted_writes_flushed();
}

void kernel_main() {
    constexpr uint32_t mcast_cb = get_compile_time_arg_val(0);
    constexpr uint32_t mm2_full_cb = get_compile_time_arg_val(1);
    constexpr uint32_t mm1_full_cb = get_compile_time_arg_val(2);
    constexpr uint32_t input_buffer_base_addr = get_compile_time_arg_val(3);
    constexpr uint32_t num_senders = get_compile_time_arg_val(4);

    const uint32_t mcast_receiver_semaphore_addr = get_semaphore(get_compile_time_arg_val(5));
    volatile tt_l1_ptr uint32_t* mcast_receiver_semaphore_addr_ptr =
        (volatile tt_l1_ptr uint32_t*)mcast_receiver_semaphore_addr;

    const uint32_t mcast_sender_semaphore_addr = get_semaphore(get_compile_time_arg_val(6));
    volatile tt_l1_ptr uint32_t* mcast_sender_semaphore_addr_ptr =
        (volatile tt_l1_ptr uint32_t*)mcast_sender_semaphore_addr;
    constexpr uint64_t mcast_sender_noc_coord_x_start = get_compile_time_arg_val(7);
    constexpr uint64_t mcast_sender_noc_coord_y_start = get_compile_time_arg_val(8);
    constexpr uint64_t mcast_sender_noc_coord_x_end = get_compile_time_arg_val(9);
    constexpr uint64_t mcast_sender_noc_coord_y_end = get_compile_time_arg_val(10);
    constexpr uint32_t num_layers = get_compile_time_arg_val(11);

    // Compute base multicast coordinates (address=0) for persistent sender
    // This sets up the coordinates that will be reused for all sends
    const uint64_t mcast_base_noc_addr = get_noc_multicast_addr<noc_index>(
        mcast_sender_noc_coord_x_start,
        mcast_sender_noc_coord_y_start,
        mcast_sender_noc_coord_x_end,
        mcast_sender_noc_coord_y_end,
        0);  // Base address = 0, coordinates only

    // Compute multicast semaphore NOC address
    const uint64_t mcast_semaphore_noc_addr = mcast_base_noc_addr | (uint64_t)(mcast_sender_semaphore_addr);

    // Initialize persistent mcast sender before the loop
    init_persistent_mcast_sender<num_senders, false, false>(
        mcast_semaphore_noc_addr, mcast_sender_semaphore_addr, mcast_sender_semaphore_addr_ptr);

    for (uint32_t layer = 0; layer < num_layers; layer++) {
        DeviceZoneScopedN("gather_and_mcast");

        // First mcast: matmul+relu result -> MM2_FULL_CB
        // Use get_write_ptr for mm2_full_cb since it's not bound to input tensor
        wait_for_gather(mcast_receiver_semaphore_addr_ptr, num_senders);
        {
            const uint32_t mm2_base_addr = get_write_ptr(mm2_full_cb);
            mcast<num_senders>(
                mcast_cb,
                mm2_base_addr,
                mcast_sender_semaphore_addr,
                mcast_sender_noc_coord_x_start,
                mcast_sender_noc_coord_y_start,
                mcast_sender_noc_coord_x_end,
                mcast_sender_noc_coord_y_end,
                mcast_semaphore_noc_addr);
        }

        // Second mcast: matmul+bias result -> MM1_FULL_CB
        // Use input buffer base address for mm1_full_cb since it's bound to input tensor
        wait_for_gather(mcast_receiver_semaphore_addr_ptr, num_senders);
        mcast<num_senders>(
            mcast_cb,
            input_buffer_base_addr,
            mcast_sender_semaphore_addr,
            mcast_sender_noc_coord_x_start,
            mcast_sender_noc_coord_y_start,
            mcast_sender_noc_coord_x_end,
            mcast_sender_noc_coord_y_end,
            mcast_semaphore_noc_addr);
    }

    // Teardown persistent mcast sender after the loop
    teardown_persistent_mcast_sender<num_senders, false, false>(mcast_semaphore_noc_addr);
}
