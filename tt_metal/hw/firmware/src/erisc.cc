/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "eth_l1_address_map.h"
#include "noc_nonblocking_api.h"
#include "noc_parameters.h"
#include "risc.h"
#include "risc_attribs.h"
#include "tt_eth_api.h"

void __attribute__((section("code_l1"))) risc_init();

inline void set_noc_trans_table(
    uint32_t noc, uint8_t &noc_trans_table_en, uint8_t &my_logical_x, uint8_t &my_logical_y) {
    noc_trans_table_en = false;
}

void risc_init() {
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(n, 0, NOC_NODE_ID);
        my_x[n] = noc_id_reg & NOC_NODE_ID_MASK;
        my_y[n] = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
        if (n == 0) {
            noc_size_x = (noc_id_reg >> (NOC_ADDR_NODE_ID_BITS + NOC_ADDR_NODE_ID_BITS)) &
                         ((((uint64_t)0x1) << (NOC_ADDR_NODE_ID_BITS + 1)) - 1);
            noc_size_y = (noc_id_reg >> (NOC_ADDR_NODE_ID_BITS + NOC_ADDR_NODE_ID_BITS + (NOC_ADDR_NODE_ID_BITS + 1))) &
                         ((((uint64_t)0x1) << (NOC_ADDR_NODE_ID_BITS + 1)) - 1);
        }

        set_noc_trans_table(n, noc_trans_table_en, my_logical_x[n], my_logical_y[n]);
    }
}

extern uint32_t __erisc_jump_table;

volatile uint32_t tt_l1_ptr *test_mailbox_ptr =
    (volatile uint32_t tt_l1_ptr *)(eth_l1_mem::address_map::FIRMWARE_BASE + 0x4);

void (*rtos_context_switch_ptr)();
volatile uint32_t *RtosTable =
    (volatile uint32_t *)&__erisc_jump_table;  // Rtos Jump Table. Runtime application needs rtos function handles.;
volatile uint32_t *q_ptr = (volatile uint32_t *)(eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);
volatile uint32_t *erisck_info = (volatile uint32_t *)(eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE);

#define NOC_X(x) (loading_noc == 0 ? (x) : (noc_size_x - 1 - (x)))
#define NOC_Y(y) (loading_noc == 0 ? (y) : (noc_size_y - 1 - (y)))

volatile uint32_t noc_read_scratch_buf[32] __attribute__((aligned(32)));
uint64_t my_q_table_offset;
uint32_t my_q_rd_ptr;
uint32_t my_q_wr_ptr;
uint8_t my_x[NUM_NOCS];
uint8_t my_y[NUM_NOCS];
uint8_t my_logical_x[NUM_NOCS];
uint8_t my_logical_y[NUM_NOCS];
uint8_t loading_noc;
uint8_t noc_size_x;
uint8_t noc_size_y;
uint8_t noc_trans_table_en;

uint32_t noc_reads_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS];
uint32_t noc_nonposted_writes_acked[NUM_NOCS];
uint32_t noc_xy_local_addr[NUM_NOCS];

constexpr static uint32_t get_arg_addr(int arg_idx) {
    // args are 4B in size
    return eth_l1_mem::address_map::ERISC_L1_ARG_BASE + (arg_idx << 2);
}

template <typename T>
inline T get_arg_val(int arg_idx) {
    // only 4B args are supported (eg int32, uint32)
    static_assert("Error: only 4B args are supported" && sizeof(T) == 4);
    return *((volatile tt_l1_ptr T *)(get_arg_addr(arg_idx)));
}

void __attribute__((section("code_l1"))) risc_context_switch() {
    ncrisc_noc_full_sync();
    rtos_context_switch_ptr();
    ncrisc_noc_counters_init();
}

void __attribute__((section("erisc_l1_code"))) ApplicationHandler(void) {
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    noc_init();

    risc_init();
    int32_t src_addr = eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;
    int32_t dst_addr = eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;

    uint32_t num_bytes = get_arg_val<uint32_t>(0);
    uint32_t mode = get_arg_val<uint32_t>(1);
    uint32_t num_loops = num_bytes >> 4;
    if (mode == 0) {
        // Ethernet Send
        erisck_info[0] = 0;
        for (uint32_t i = 0; i < num_loops; i++) {
            eth_send_packet(0, i + (src_addr >> 4), i + (dst_addr >> 4), 1);
        }
        erisck_info[0] = num_bytes;
        eth_send_packet(0, ((uint32_t)(&erisck_info[0])) >> 4, ((uint32_t)(&erisck_info[0])) >> 4, 1);
        while (erisck_info[0] != 0) {
            RISC_POST_STATUS(0x10000001 | (erisck_info[0] << 12));
        }
    } else if (mode == 1) {
        // Ethernet Receive
        while (erisck_info[0] != num_bytes) {
            RISC_POST_STATUS(0x10000002 | (erisck_info[0] << 12));
        }
        erisck_info[0] = 0;
        eth_send_packet(0, ((uint32_t)(&erisck_info[0])) >> 4, ((uint32_t)(&erisck_info[0])) >> 4, 1);
    } else {
        while (true) RISC_POST_STATUS(0x1234DEAD);
    }
    q_ptr[0] = 0;
}
