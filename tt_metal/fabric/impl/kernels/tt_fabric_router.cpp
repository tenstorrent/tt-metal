// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_utils.h"
// clang-format on

using namespace tt::tt_fabric;

router_state_t router_state __attribute__((aligned(16)));
constexpr uint32_t fvc_data_buf_size_words = get_compile_time_arg_val(0);
constexpr uint32_t fvc_data_buf_size_bytes = fvc_data_buf_size_words * PACKET_WORD_SIZE_BYTES;
constexpr uint32_t kernel_status_buf_addr_arg = get_compile_time_arg_val(1);
constexpr uint32_t kernel_status_buf_size_bytes = get_compile_time_arg_val(2);
constexpr uint32_t timeout_cycles = get_compile_time_arg_val(3);
constexpr bool is_master = get_compile_time_arg_val(4);
constexpr uint32_t router_direction = get_compile_time_arg_val(5);
uint32_t sync_val;
uint32_t router_mask;
uint32_t master_router_chan;
uint64_t xy_local_addr;
bool terminated_subordinate_routers = false;

// careful, may be null
tt_l1_ptr uint32_t* const kernel_status = reinterpret_cast<tt_l1_ptr uint32_t*>(kernel_status_buf_addr_arg);
volatile tt_l1_ptr fabric_router_l1_config_t* routing_table =
    reinterpret_cast<tt_l1_ptr fabric_router_l1_config_t*>(eth_l1_mem::address_map::FABRIC_ROUTER_CONFIG_BASE);

#define SWITCH_THRESHOLD 0x3FFF

void kernel_main() {
    tt_fabric_init();
    fvc_inbound_push_state_t fvc_inbound_state;
    fvc_outbound_push_state_t fvc_outbound_state[4];
    volatile tt_l1_ptr fabric_push_client_queue_t* client_queue =
        reinterpret_cast<tt_l1_ptr fabric_push_client_queue_t*>(FABRIC_ROUTER_CLIENT_QUEUE_START);
    zero_l1_buf((tt_l1_ptr uint32_t*)client_queue, sizeof(fabric_push_client_queue_t));
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    uint32_t rt_args_idx = 0;
    sync_val = get_arg_val<uint32_t>(rt_args_idx++);
    router_mask = get_arg_val<uint32_t>(rt_args_idx++);
    master_router_chan = get_arg_val<uint32_t>(rt_args_idx++);

    write_kernel_status(kernel_status, TT_FABRIC_STATUS_INDEX, TT_FABRIC_STATUS_STARTED);

    router_state.sync_in = 0;
    router_state.sync_out = 0;

    write_kernel_status(kernel_status, TT_FABRIC_WORD_CNT_INDEX, (uint32_t)&router_state);
    write_kernel_status(kernel_status, TT_FABRIC_WORD_CNT_INDEX + 1, (uint32_t)&fvc_outbound_state);
    write_kernel_status(kernel_status, TT_FABRIC_STATUS_INDEX + 1, (uint32_t)&fvc_inbound_state);

    fvc_outbound_state[0].init(
        0, FABRIC_ROUTER_DATA_BUF_START, FABRIC_ROUTER_OUTBOUND_BUF_SIZE / PACKET_WORD_SIZE_BYTES);
    fvc_outbound_state[1].init(
        1,
        FABRIC_ROUTER_DATA_BUF_START + FABRIC_ROUTER_OUTBOUND_BUF_SIZE,
        FABRIC_ROUTER_OUTBOUND_BUF_SIZE / PACKET_WORD_SIZE_BYTES);
    fvc_outbound_state[2].init(
        2,
        FABRIC_ROUTER_DATA_BUF_START + 2 * FABRIC_ROUTER_OUTBOUND_BUF_SIZE,
        FABRIC_ROUTER_OUTBOUND_BUF_SIZE / PACKET_WORD_SIZE_BYTES);
    fvc_outbound_state[3].init(
        3,
        FABRIC_ROUTER_DATA_BUF_START + 3 * FABRIC_ROUTER_OUTBOUND_BUF_SIZE,
        FABRIC_ROUTER_OUTBOUND_BUF_SIZE / PACKET_WORD_SIZE_BYTES);

    fvc_inbound_state.init(
        FABRIC_ROUTER_DATA_BUF_START + 4 * FABRIC_ROUTER_OUTBOUND_BUF_SIZE,
        FABRIC_ROUTER_INBOUND_BUF_SIZE / PACKET_WORD_SIZE_BYTES);

    if (!wait_all_src_dest_ready(&router_state, timeout_cycles)) {
        write_kernel_status(kernel_status, TT_FABRIC_STATUS_INDEX, TT_FABRIC_STATUS_TIMEOUT);
        return;
    }

    if constexpr (is_master) {
        // wait for all device routers to have incremented the sync semaphore.
        // sync_val is equal to number of tt-fabric routers running on a device.
        wait_for_notification(FABRIC_ROUTER_SYNC_SEM, sync_val - 1);
        notify_subordinate_routers(router_mask, master_router_chan, FABRIC_ROUTER_SYNC_SEM, sync_val);
        // increment the sync sem to signal host that handshake is complete
        *((volatile uint32_t*)FABRIC_ROUTER_SYNC_SEM) += 1;
    } else {
        notify_master_router(master_router_chan, FABRIC_ROUTER_SYNC_SEM);
        // wait for the signal from the master router
        wait_for_notification(FABRIC_ROUTER_SYNC_SEM, sync_val);
    }

    fvc_inbound_state.register_with_routers<FVC_MODE_ROUTER>(routing_table->my_device_id);
    uint32_t curr_outbound_buffer = 0;
    uint32_t next_outbound_buffer = 1;
    uint32_t eth_outbound_wrptr = 0;
    uint64_t start_timestamp = get_timestamp();

    write_kernel_status(kernel_status, TT_FABRIC_MISC_INDEX, 0xff000001);
    uint32_t loop_count = 0;

    uint32_t launch_msg_rd_ptr = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    tt_l1_ptr launch_msg_t* const launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_msg_rd_ptr]);
    while (1) {
        // Handle Ethernet Outbound Data
        if (fvc_outbound_state[curr_outbound_buffer].forward_data_from_fvc_buffer(eth_outbound_wrptr)) {
            loop_count = 0;
        } else {
            if (*fvc_outbound_state[next_outbound_buffer].slot_credits ||
                *fvc_outbound_state[next_outbound_buffer].sender_slots_cleared) {
                curr_outbound_buffer = next_outbound_buffer;
            }
            next_outbound_buffer = (next_outbound_buffer + 1) & 0x3;
        }

        // Handle Ethernet Inbound Data
        if (fvc_inbound_state.get_curr_packet_valid<FVC_MODE_ROUTER>()) {
            fvc_inbound_state.process_inbound_packet<FVC_MODE_ROUTER, router_direction>();
            loop_count = 0;
        }
#ifdef TT_FABRIC_DEBUG
        else if (fvc_inbound_state.packet_corrupted) {
            write_kernel_status(kernel_status, TT_FABRIC_STATUS_INDEX, TT_FABRIC_STATUS_BAD_HEADER);
            return;
        }
#endif

        loop_count++;

        // need to optimize this.
        // context switch to base fw is very costly.
        if ((loop_count & SWITCH_THRESHOLD) == SWITCH_THRESHOLD) {
            internal_::risc_context_switch();

            if (*(volatile uint32_t*)FABRIC_ROUTER_SYNC_SEM == 0) {
                // terminate signal from host sw.
                if constexpr (is_master) {
                    if (!terminated_subordinate_routers) {
                        notify_subordinate_routers(router_mask, master_router_chan, FABRIC_ROUTER_SYNC_SEM, 0);
                        terminated_subordinate_routers = true;
                    }
                }
                if (loop_count >= 0x1000) {
                    break;
                }
            }
            if (launch_msg->kernel_config.exit_erisc_kernel) {
                return;
            }
        }
    }
    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    set_64b_result(kernel_status, cycles_elapsed, TT_FABRIC_CYCLES_INDEX);

    write_kernel_status(kernel_status, TT_FABRIC_STATUS_INDEX, TT_FABRIC_STATUS_PASS);
    write_kernel_status(kernel_status, TT_FABRIC_MISC_INDEX, 0xff00005);
}
