// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"
#include "noc_nonblocking_api.h"
#include "noc_parameters.h"
#include "risc.h"
#include "risc_attribs.h"
#include "tt_eth_api.h"
struct erisc_info_t {
  volatile uint32_t num_bytes;
  volatile uint32_t mode;
  volatile uint32_t unused_arg0;
  volatile uint32_t unused_arg1;
  volatile uint32_t bytes_sent;
  volatile uint32_t reserved_0_;
  volatile uint32_t reserved_1_;
  volatile uint32_t reserved_2_;
};

void __attribute__((section("code_l1"))) risc_init();


volatile erisc_info_t *erisc_info = (erisc_info_t *)(eth_l1_mem::address_map::ERISC_APP_SYNC_INFO_BASE);
inline void set_noc_trans_table(
    uint32_t noc, uint8_t &noc_trans_table_en, uint8_t &my_logical_x, uint8_t &my_logical_y) {
    noc_trans_table_en = false;
}
extern uint32_t __erisc_jump_table;
volatile uint32_t *flag_disable = (uint32_t *)(eth_l1_mem::address_map::LAUNCH_ERISC_APP_FLAG);

void (*rtos_context_switch_ptr)();
volatile uint32_t *RtosTable =
    (volatile uint32_t *)&__erisc_jump_table;  // Rtos Jump Table. Runtime application needs rtos function handles.;

#define NOC_X(x) (loading_noc == 0 ? (x) : (noc_size_x - 1 - (x)))
#define NOC_Y(y) (loading_noc == 0 ? (y) : (noc_size_y - 1 - (y)))

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


void __attribute__((section("erisc_l1_code"))) ApplicationHandler(void) {
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    noc_init();

    risc_init();
    int32_t src_addr = eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;
    int32_t dst_addr = eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;

    uint32_t mode = erisc_info->mode;
    uint32_t num_loops = erisc_info->num_bytes >> 4;
    uint64_t timeout = 100000;
    if (mode == 0) {
        // Ethernet Send
        erisc_info->bytes_sent = 0;
        for (uint32_t i = 0; i < num_loops; i++) {
            eth_send_packet(0, i + (src_addr >> 4), i + (dst_addr >> 4), 1);
        }
        erisc_info->bytes_sent = erisc_info->num_bytes;
        eth_send_packet(0, ((uint32_t)(&(erisc_info->bytes_sent))) >> 4, ((uint32_t)(&(erisc_info->bytes_sent))) >> 4, 1);
        uint64_t start_time = eth_read_wall_clock();
        while (erisc_info->bytes_sent != 0) {
            RISC_POST_STATUS(0x10000001 | (erisc_info->bytes_sent << 12));
         //  if (eth_read_wall_clock() - start_time > timeout) {
         //       risc_context_switch();
         //       start_time = eth_read_wall_clock();
         //   }
        }
    } else if (mode == 1) {
        // Ethernet Receive
        uint64_t start_time = eth_read_wall_clock();
        while (erisc_info->bytes_sent != erisc_info->num_bytes) {
            RISC_POST_STATUS(0x10000002 | (erisc_info->bytes_sent << 12));
         //   if (eth_read_wall_clock() - start_time > timeout) {
         //       RISC_POST_STATUS(0x1234DEED);
         //       risc_context_switch();
         //       RISC_POST_STATUS(0x1234ABCD);
         //       start_time = eth_read_wall_clock();
         //   }
        }
        erisc_info->bytes_sent = 0;
        eth_send_packet(0, ((uint32_t)(&(erisc_info->bytes_sent))) >> 4, ((uint32_t)(&(erisc_info->bytes_sent))) >> 4, 1);
    } else {
        while (true) RISC_POST_STATUS(0x1234DEAD);
    }
    flag_disable[0] = 0;
}
