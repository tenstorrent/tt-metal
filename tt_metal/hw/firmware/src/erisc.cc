// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"
#include "ethernet/dataflow_api.h"
#include "ethernet/tunneling.h"
#include "firmware_common.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "noc_parameters.h"
#include "risc_attribs.h"
#include "tools/profiler/kernel_profiler.hpp"

#ifdef __cplusplus
extern "C" {
#endif

void ApplicationHandler(void) __attribute__((__section__(".init")));

#ifdef __cplusplus
}
#endif

namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
}

uint8_t noc_index = 0;  // TODO: remove hardcoding
uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));

void __attribute__((section("code_l1"))) risc_init() {
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(n, 0, NOC_NODE_ID);
        my_x[n] = noc_id_reg & NOC_NODE_ID_MASK;
        my_y[n] = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    }
}

void __attribute__((section("erisc_l1_code"))) ApplicationHandler(void) {
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    risc_init();
    noc_init();

    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }
    ncrisc_noc_full_sync();
    while (routing_info->routing_enabled != 1) {
        internal_::risc_context_switch();
    }

    router_init();

    while (routing_info->routing_enabled) {
        // FD: assume that no more host -> remote writes are pending
        if (erisc_info->launch_user_kernel == 1) {
            kernel_profiler::init_profiler();
            kernel_profiler::mark_time(CC_MAIN_START);
            kernel_init();
            kernel_profiler::mark_time(CC_MAIN_END);
        }
        if (my_routing_mode == EthRouterMode::FD_SRC) {
            volatile tt_l1_ptr uint32_t *eth_db_semaphore_addr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(0));
            eth_db_acquire(eth_db_semaphore_addr, ((uint64_t)eth_router_noc_encoding << 32));
            if (erisc_info->launch_user_kernel == 1) {
                continue;
            }
            if (routing_info->routing_enabled == 0) {
                break;
            }
            eth_tunnel_src_forward_one_cmd();
        } else if (routing_info->routing_mode == EthRouterMode::FD_DST) {
            // Poll until FD_SRC router sends FD packet
            // Each FD packet comprises of command header followed by command data
            internal_::wait_for_fd_packet();
            if (erisc_info->launch_user_kernel == 1) {
                continue;
            }
            if (routing_info->routing_enabled == 0) {
                break;
            }

            eth_tunnel_dst_forward_one_cmd();
        } else {
            internal_::risc_context_switch();
        }
    }
    internal_::disable_erisc_app();
}
