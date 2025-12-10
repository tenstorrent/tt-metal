// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dataflow_api.h"

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

// Macro to define mcast-related variables with a given prefix
#define DEFINE_PERSISTENT_MCAST_SENDER_VARS(prefix)                                                                   \
    constexpr uint32_t prefix##_mcast_dest_noc_start_x = get_named_compile_time_arg_val(#prefix "_dest_noc_start_x"); \
    constexpr uint32_t prefix##_mcast_dest_noc_start_y = get_named_compile_time_arg_val(#prefix "_dest_noc_start_y"); \
    constexpr uint32_t prefix##_mcast_dest_noc_end_x = get_named_compile_time_arg_val(#prefix "_dest_noc_end_x");     \
    constexpr uint32_t prefix##_mcast_dest_noc_end_y = get_named_compile_time_arg_val(#prefix "_dest_noc_end_y");     \
    constexpr uint32_t prefix##_mcast_num_cores = get_named_compile_time_arg_val(#prefix "_num_cores");               \
    constexpr bool prefix##_loopback = get_named_compile_time_arg_val(#prefix "_loopback");                           \
    constexpr bool prefix##_is_part_of_receiver_grid =                                                                \
        get_named_compile_time_arg_val(#prefix "_is_part_of_receiver_grid");                                          \
    uint32_t prefix##_data_sender_semaphore_addr =                                                                    \
        get_semaphore(get_named_compile_time_arg_val(#prefix "_data_sender_semaphore"));                              \
    uint32_t prefix##_data_receiver_semaphore_addr =                                                                  \
        get_semaphore(get_named_compile_time_arg_val(#prefix "_data_receiver_semaphore"));                            \
    const uint64_t prefix##_noc_coord = get_noc_multicast_addr<noc_index>(                                            \
        prefix##_mcast_dest_noc_start_x,                                                                              \
        prefix##_mcast_dest_noc_start_y,                                                                              \
        prefix##_mcast_dest_noc_end_x,                                                                                \
        prefix##_mcast_dest_noc_end_y,                                                                                \
        0);                                                                                                           \
    uint64_t prefix##_mcast_flag_noc_addr = prefix##_noc_coord | (uint64_t)(prefix##_data_receiver_semaphore_addr);   \
    volatile tt_l1_ptr uint32_t* prefix##_data_sender_semaphore_addr_ptr =                                            \
        (volatile tt_l1_ptr uint32_t*)prefix##_data_sender_semaphore_addr;

#define INIT_PERSISTENT_MCAST_SENDER(prefix)                                                                      \
    init_persistent_mcast_sender<prefix##_mcast_num_cores, prefix##_loopback, prefix##_is_part_of_receiver_grid>( \
        prefix##_mcast_flag_noc_addr, prefix##_data_sender_semaphore_addr, prefix##_data_sender_semaphore_addr_ptr);

#define TEARDOWN_PERSISTENT_MCAST_SENDER(prefix)                                                                      \
    teardown_persistent_mcast_sender<prefix##_mcast_num_cores, prefix##_loopback, prefix##_is_part_of_receiver_grid>( \
        prefix##_mcast_flag_noc_addr);

#define DEFINE_MCAST_SENDER_VARS(pmcast_prefix, prefix, arg_idx)                                              \
    constexpr uint32_t prefix##_mcast_num_cores = get_named_compile_time_arg_val(#prefix "_num_cores");       \
    constexpr uint32_t prefix##_data_size_bytes = get_named_compile_time_arg_val(#prefix "_data_size_bytes"); \
    uint32_t prefix##_input_data_addr = get_common_arg_val<uint32_t>(arg_idx++);                              \
    uint32_t prefix##_mcast_receiver_data_addr = get_common_arg_val<uint32_t>(arg_idx++);                     \
    uint64_t prefix##_mcast_data_noc_addr = pmcast_prefix##_noc_coord | (uint64_t)prefix##_mcast_receiver_data_addr;

#define MCAST_SEND_DATA_WITH_STATE(pmcast_prefix, prefix)                                                 \
    mcast_send_with_state<                                                                                \
        prefix##_mcast_num_cores,                                                                         \
        pmcast_prefix##_loopback,                                                                         \
        pmcast_prefix##_is_part_of_receiver_grid,                                                         \
        true,                                                                                             \
        true,                                                                                             \
        true,                                                                                             \
        true,                                                                                             \
        write_cmd_buf>(prefix##_input_data_addr, prefix##_mcast_data_noc_addr, prefix##_data_size_bytes); \
    mcast_send_with_state<                                                                                \
        prefix##_mcast_num_cores,                                                                         \
        pmcast_prefix##_loopback,                                                                         \
        pmcast_prefix##_is_part_of_receiver_grid,                                                         \
        true,                                                                                             \
        true,                                                                                             \
        false,                                                                                            \
        false,                                                                                            \
        write_reg_cmd_buf>(0, 0, 0);

// Macro to define gather-related variables with a given prefix
#define DEFINE_GATHER_RECEIVER_VARS(prefix)                                                                     \
    constexpr uint32_t prefix##_noc0_num_senders = get_named_compile_time_arg_val(#prefix "_noc0_num_senders"); \
    constexpr uint32_t prefix##_noc1_num_senders = get_named_compile_time_arg_val(#prefix "_noc1_num_senders"); \
    uint32_t prefix##_noc0_receiver_semaphore_addr =                                                            \
        get_semaphore(get_named_compile_time_arg_val(#prefix "_noc0_receiver_semaphore"));                      \
    uint32_t prefix##_noc1_receiver_semaphore_addr =                                                            \
        get_semaphore(get_named_compile_time_arg_val(#prefix "_noc1_receiver_semaphore"));                      \
    volatile tt_l1_ptr uint32_t* prefix##_noc0_receiver_semaphore_addr_ptr =                                    \
        (volatile tt_l1_ptr uint32_t*)prefix##_noc0_receiver_semaphore_addr;                                    \
    volatile tt_l1_ptr uint32_t* prefix##_noc1_receiver_semaphore_addr_ptr =                                    \
        (volatile tt_l1_ptr uint32_t*)prefix##_noc1_receiver_semaphore_addr;

// Macro to wait for both NOC semaphores and reset them
#define WAIT_AND_RESET_GATHER_RECEIVER_SEMAPHORES(prefix)                                     \
    noc_semaphore_wait(prefix##_noc0_receiver_semaphore_addr_ptr, prefix##_noc0_num_senders); \
    noc_semaphore_wait(prefix##_noc1_receiver_semaphore_addr_ptr, prefix##_noc1_num_senders); \
    noc_semaphore_set(prefix##_noc0_receiver_semaphore_addr_ptr, INVALID);                    \
    noc_semaphore_set(prefix##_noc1_receiver_semaphore_addr_ptr, INVALID);
