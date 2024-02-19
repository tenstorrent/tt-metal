// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "eth_l1_address_map.h"
#include "ethernet/dataflow_api.h"
#include "firmware_common.h"
#include "generated_bank_to_noc_coord_mapping.h"
#include "noc_parameters.h"
#include "risc_attribs.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "tt_metal/impl/dispatch/device_command.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_prefetcher.hpp"
#include "debug/dprint.h"

// Number of registers to save for early exit
#define CONTEXT_SIZE (13 * 4)

#ifdef __cplusplus
extern "C" {
#endif

void ApplicationHandler(void) __attribute__((__section__(".init")));

#ifdef __cplusplus
}
#endif

namespace kernel_profiler {
uint32_t wIndex __attribute__((used));
uint32_t device_function_sums[GLOBAL_SUM_COUNT] __attribute__((used)) = {0};
uint64_t device_function_starts[GLOBAL_SUM_COUNT] __attribute__((used)) = {0};
}

uint8_t noc_index = 0;  // TODO: remove hardcoding
uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));

FORCE_INLINE
void multicore_eth_cb_wait_front(db_cb_config_t *eth_db_cb_config, int32_t num_pages) {
    DEBUG_STATUS('C', 'R', 'B', 'W');

    uint16_t pages_received;
    do {
        pages_received = uint16_t(eth_db_cb_config->recv - eth_db_cb_config->ack);
        internal_::risc_context_switch(); // AL: hopefully we can remove this...
    } while (pages_received < num_pages);
    DEBUG_STATUS('C', 'R', 'B', 'D');
}

FORCE_INLINE
void multicore_eth_cb_pop_front(
    db_cb_config_t *eth_db_cb_config,
    const volatile db_cb_config_t *remote_db_cb_config,
    uint64_t producer_noc_encoding,
    uint32_t num_pages) {
    eth_db_cb_config->ack += num_pages;

    uint32_t pages_ack_addr = (uint32_t)(&(remote_db_cb_config->ack));
    noc_semaphore_set_remote((uint32_t)(&(eth_db_cb_config->ack)), producer_noc_encoding | pages_ack_addr);
}

FORCE_INLINE
void eth_db_acquire(volatile uint32_t *semaphore, uint64_t noc_encoding) {
    DEBUG_STATUS('D', 'B', 'A', 'W');
    while (semaphore[0] == 0 and routing_info->routing_enabled and erisc_info->launch_user_kernel == 0) {
        // Without this context switch a src router on R chip of N300 may get configured for FD
        //  and block L chip of N300 from sending config for dst router on R because the path to the dst router is through the src router
        internal_::risc_context_switch();
    }
    DEBUG_STATUS('D', 'B', 'A', 'D');
}

void __attribute__((section("code_l1"))) risc_init() {
    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(n, 0, NOC_NODE_ID);
        my_x[n] = noc_id_reg & NOC_NODE_ID_MASK;
        my_y[n] = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    }
}

void __attribute__((section("code_l1"))) router_init() {
    relay_src_noc_encoding = uint32_t(NOC_XY_ENCODING(routing_info->relay_src_x, routing_info->relay_src_y));
    relay_dst_noc_encoding = uint32_t(NOC_XY_ENCODING(routing_info->relay_dst_x, routing_info->relay_dst_y));

    eth_router_noc_encoding = uint32_t(NOC_XY_ENCODING(my_x[0], my_y[0]));
    my_routing_mode = (EthRouterMode)routing_info->routing_mode;
}

void __attribute__((section("erisc_l1_code"))) Application(void) {
    DEBUG_STATUS('I');
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    risc_init();
    noc_init();

    for (uint32_t n = 0; n < NUM_NOCS; n++) {
        noc_local_state_init(n);
    }
    ncrisc_noc_full_sync();
    DEBUG_STATUS('R', 'E', 'W');
    while (routing_info->routing_enabled != 1) {
        internal_::risc_context_switch();
    }
    DEBUG_STATUS('R', 'E', 'D');

    router_init();

    volatile tt_l1_ptr uint32_t *eth_db_semaphore_addr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t *>(eth_get_semaphore(0));

    static constexpr uint32_t command_start_addr = eth_l1_mem::address_map::ERISC_APP_RESERVED_BASE;
    static constexpr uint32_t data_buffer_size = eth_l1_mem::address_map::ERISC_APP_RESERVED_SIZE - (DeviceCommand::NUM_ENTRIES_IN_DEVICE_COMMAND * sizeof(uint32_t));

    bool db_buf_switch = false;
    db_cb_config_t *eth_db_cb_config = get_local_db_cb_config(eth_l1_mem::address_map::CQ_CONSUMER_CB_BASE, false);
    volatile db_cb_config_t *remote_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, false);
    // const db_cb_config_t *remote_dst_db_cb_config = get_remote_db_cb_config(CQ_CONSUMER_CB_BASE, true);

    erisc_info->unused_arg0 = 0;
    erisc_info->unused_arg1 = 0;

    while (routing_info->routing_enabled) {
        // FD: assume that no more host -> remote writes are pending
        if (erisc_info->launch_user_kernel == 1) {
            DEBUG_STATUS('R');
            kernel_profiler::init_profiler();
            kernel_profiler::mark_time(CC_MAIN_START);
            kernel_init();
            kernel_profiler::store_function_sums();
            kernel_profiler::mark_time(CC_MAIN_END);
            DEBUG_STATUS('D');
        }
        if (my_routing_mode == EthRouterMode::FD_SRC) {
            // DPRINT << "SRC waiting " << eth_db_semaphore_addr[0] << ENDL();
            eth_db_acquire(eth_db_semaphore_addr, ((uint64_t)eth_router_noc_encoding << 32));
            // DPRINT << "SRC done waiting " << eth_db_semaphore_addr[0] << ENDL();
            if (erisc_info->launch_user_kernel == 1) {
                continue;
            }
            if (routing_info->routing_enabled == 0) {
                break;
            }
            // DPRINT << "SRC handle cmd " << eth_db_semaphore_addr[0] << ENDL();
            noc_semaphore_inc(
                ((uint64_t)eth_router_noc_encoding << 32) | uint32_t(eth_db_semaphore_addr),
                -1);  // Two's complement addition
            noc_async_write_barrier();

            volatile tt_l1_ptr uint32_t *command_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t *>(command_start_addr);
            volatile tt_l1_ptr CommandHeader *header = (CommandHeader *)command_ptr;
            uint32_t num_buffer_transfers = header->num_buffer_transfers;
            uint32_t producer_consumer_transfer_num_pages = header->producer_consumer_transfer_num_pages;
            bool is_program = header->is_program_buffer;
            bool fwd_path = header->fwd_path;
            command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;

            // DPRINT << "SRC got" << ENDL();

            // send cmd even if there is no data associated
            internal_::send_fd_packets(); // TODO: AL, is this right?
            erisc_info->unused_arg0 = erisc_info->unused_arg0 + 1;
            // DPRINT << "SRC2DST" << ENDL();

            for (uint32_t i = 0; i < num_buffer_transfers; i++) {
                // erisc_info->unused_arg0 = 109;
                const uint32_t num_pages = command_ptr[2];
                const uint32_t src_buf_type = command_ptr[4];
                const uint32_t dst_buf_type = command_ptr[5];

                bool read_from_sysmem = (BufferType)src_buf_type == BufferType::SYSTEM_MEMORY;
                bool write_to_sysmem = (BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY;
                bool tunnel_data = (read_from_sysmem) | (write_to_sysmem & !fwd_path & !is_program);

                if (!tunnel_data) {
                    // erisc_info->unused_arg0 = 500;
                    continue;
                }

                uint32_t num_to_write = min(num_pages, producer_consumer_transfer_num_pages);

                uint32_t num_pages_tunneled = 0;
                while (num_pages_tunneled != num_pages) {
                    // DPRINT << "SRC waiting for " << num_to_write << ENDL();
                    multicore_eth_cb_wait_front(eth_db_cb_config, num_to_write);
                    internal_::send_fd_packets(); // AL: increment since idx to msg sent
                    erisc_info->unused_arg1 = erisc_info->unused_arg1 + 1;
                    multicore_eth_cb_pop_front(
                        eth_db_cb_config, remote_db_cb_config, ((uint64_t)relay_src_noc_encoding << 32), num_to_write);
                    num_pages_tunneled += num_to_write;
                    num_to_write = min(num_pages - num_pages_tunneled, producer_consumer_transfer_num_pages);
                }
                command_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;
            }

            // DPRINT << "SRCD" << ENDL();
            noc_semaphore_inc(((uint64_t)relay_src_noc_encoding << 32) | get_semaphore(0), 1);
            noc_async_write_barrier();  // Barrier for now
            // erisc_info->unused_arg1 = routing_info->relay_src_x;
            // erisc_info->unused_arg2 = routing_info->relay_src_y;

        } else if (routing_info->routing_mode == EthRouterMode::FD_DST) {

            // Poll until FD_SRC router sends FD packet
            // Each FD packet comprises of command header followed by command data
            internal_::wait_for_fd_packet();
            if (erisc_info->launch_user_kernel == 1) {
                continue;
            }
            // DPRINT << "DST FROM SRC" << ENDL();
            if (routing_info->routing_enabled == 0) {
                break;
            }
            // DPRINT << "DST routing enabled" << ENDL();

            // tell pull_and_relay core that command is available

            // DPRINT << "DST GOT" << ENDL();


            // DPRINT << "DST INFORMED" << ENDL();
            volatile tt_l1_ptr uint32_t *command_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t *>(command_start_addr);
            volatile tt_l1_ptr CommandHeader *header = (CommandHeader *)command_ptr;
            uint32_t num_buffer_transfers = header->num_buffer_transfers;
            uint32_t producer_consumer_transfer_num_pages = header->producer_consumer_transfer_num_pages;
            uint32_t consumer_cb_num_pages = header->consumer_cb_num_pages;
            uint32_t consumer_cb_size = header->consumer_cb_size;
            uint32_t page_size = header->page_size;
            bool is_program = header->is_program_buffer;
            bool fwd_path = header->fwd_path;
            command_ptr += DeviceCommand::NUM_ENTRIES_IN_COMMAND_HEADER;

            // Initially 1
            // After update_producer_consumer_sync_semaphores goes to 0
            // At some point pull_and_relay will set it to 1 once it reads in the command
            update_producer_consumer_sync_semaphores(((uint64_t)eth_router_noc_encoding << 32), ((uint64_t)relay_dst_noc_encoding << 32), eth_db_semaphore_addr, get_semaphore(1));

            // Wait until push and pull kernel has read in the command
            // Before pull and push kernel signal to DST router that it has read in a command, it also programs the DST router CB
            while (eth_db_semaphore_addr[0] == 0) {
                internal_::risc_context_switch();
            }

            // Ack because pull and push kernel signalled that it got the command
            internal_::ack_fd_packet();
            erisc_info->unused_arg0 = erisc_info->unused_arg0 + 1;

            for (uint32_t i = 0; i < num_buffer_transfers; i++) {
                const uint32_t num_pages = command_ptr[2];
                const uint32_t src_buf_type = command_ptr[4];
                const uint32_t dst_buf_type = command_ptr[5];

                bool read_from_sysmem = (BufferType)src_buf_type == BufferType::SYSTEM_MEMORY;
                bool write_to_sysmem = (BufferType)dst_buf_type == BufferType::SYSTEM_MEMORY;
                bool tunnel_data = (read_from_sysmem) | (write_to_sysmem & !is_program & !fwd_path);

                if (!tunnel_data) {
                    continue;
                }

                // producer_consumer_transfer_num_pages is the total num of data pages that could fit in a FD packet
                uint32_t num_pages_to_signal = min(num_pages, producer_consumer_transfer_num_pages);
                uint32_t num_pages_transferred = 0;

                while (num_pages_transferred != num_pages) {
                    while (routing_info->fd_buffer_msgs_sent != 1) { // wait for SRC router to send data
                        internal_::risc_context_switch();
                    }

                    erisc_info->unused_arg1 = erisc_info->unused_arg1 + 1;

                    while (not cb_consumer_space_available(eth_db_cb_config, num_pages_to_signal)); // noc stall so don't need to switch to base FW --> do we need this?
                    multicore_cb_push_back( // signal to pull and push kernel that data is ready
                        eth_db_cb_config,
                        remote_db_cb_config,
                        ((uint64_t)relay_dst_noc_encoding << 32),
                        eth_db_cb_config->fifo_limit_16B,
                        num_pages_to_signal);
                    // wait for pull and push kernel to pick up the data
                    while (eth_db_cb_config->ack != eth_db_cb_config->recv) {
                        internal_::risc_context_switch();
                    }

                    internal_::ack_fd_packet(); // signal to SRC router that more data can be sent

                    num_pages_transferred += num_pages_to_signal;
                    num_pages_to_signal = min(num_pages - num_pages_transferred, producer_consumer_transfer_num_pages);
                }
                command_ptr += DeviceCommand::NUM_ENTRIES_PER_BUFFER_TRANSFER_INSTRUCTION;  // jump to buffer transfer region
            }
        } else {
            internal_::risc_context_switch();
        }
    }
    internal_::disable_erisc_app();
}

void __attribute__((section("erisc_l1_code"), naked)) ApplicationHandler(void) {
    // Save the registers, stack pointer, return address so that we can early exit in the case of
    // an error.
    __asm__(
        "addi sp, sp, -%[context_size]\n\t"
        "sw x1, 0 * 4( sp )\n\t" // Return addr saved on stack
        "sw x8, 1 * 4( sp )\n\t"
        "sw x9, 2 * 4( sp )\n\t"
        "sw x18, 3 * 4( sp )\n\t"
        "sw x19, 4 * 4( sp )\n\t"
        "sw x20, 5 * 4( sp )\n\t"
        "sw x21, 6 * 4( sp )\n\t"
        "sw x22, 7 * 4( sp )\n\t"
        "sw x23, 8 * 4( sp )\n\t"
        "sw x24, 9 * 4( sp )\n\t"
        "sw x25, 10 * 4( sp )\n\t"
        "sw x26, 11 * 4( sp )\n\t"
        "sw x27, 12 * 4( sp )\n\t"
        "li x10, %[stack_save_addr]\n\t"
        "sw  sp, 0( x10 )\n\t"
        : /* No Inputs */
        : [context_size] "i" (CONTEXT_SIZE), [stack_save_addr] "i" (eth_l1_mem::address_map::ERISC_MEM_MAILBOX_STACK_SAVE)
        : "x10", "memory"
    );
    Application();
    __asm__(
        "lw  x1, 0 * 4( sp )\n\t"
        "lw  x8, 1 * 4( sp )\n\t"
        "lw  x9, 2 * 4( sp )\n\t"
        "lw  x18, 3 * 4( sp )\n\t"
        "lw  x19, 4 * 4( sp )\n\t"
        "lw  x20, 5 * 4( sp )\n\t"
        "lw  x21, 6 * 4( sp )\n\t"
        "lw  x22, 7 * 4( sp )\n\t"
        "lw  x23, 8 * 4( sp )\n\t"
        "lw  x24, 9 * 4( sp )\n\t"
        "lw  x25, 10 * 4( sp )\n\t"
        "lw  x26, 11 * 4( sp )\n\t"
        "lw  x27, 12 * 4( sp )\n\t"
        "addi sp, sp, %[context_size]\n\t"
        "ret\n\t"
        : /* No Inputs */
        : [context_size] "i" (CONTEXT_SIZE)
        :
    );
}
