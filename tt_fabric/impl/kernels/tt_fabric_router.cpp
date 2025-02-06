// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "tt_fabric/hw/inc/tt_fabric.h"
#include "tt_fabric/hw/inc/tt_fabric_status.h"
// clang-format on

using namespace tt::tt_fabric;

router_state_t router_state __attribute__((aligned(16)));
#ifdef FVC_MODE_PULL
fvc_consumer_state_t fvc_outbound_state __attribute__((aligned(16)));  // replicate for each fvc
#else
// fvc_inbound_push_state_t fvc_inbound_state;
// fvc_outbound_push_state_t fvc_outbound_state[4];
#endif
#ifdef FVCC_SUPPORT
fvcc_inbound_state_t fvcc_inbound_state __attribute__((aligned(16)));    // inbound fabric virtual control channel
fvcc_outbound_state_t fvcc_outbound_state __attribute__((aligned(16)));  // outbound fabric virtual control channel
#endif
volatile local_pull_request_t local_pull_request_temp __attribute__((aligned(16)));  // replicate for each fvc
volatile local_pull_request_t* local_pull_request = &local_pull_request_temp;        // replicate for each fvc

constexpr uint32_t fvc_data_buf_size_words = get_compile_time_arg_val(0);
constexpr uint32_t fvc_data_buf_size_bytes = fvc_data_buf_size_words * PACKET_WORD_SIZE_BYTES;
constexpr uint32_t kernel_status_buf_addr_arg = get_compile_time_arg_val(1);
constexpr uint32_t kernel_status_buf_size_bytes = get_compile_time_arg_val(2);
constexpr uint32_t timeout_cycles = get_compile_time_arg_val(3);
uint32_t sync_val;
uint32_t router_mask;
uint32_t gk_message_addr_l;
uint32_t gk_message_addr_h;

// careful, may be null
tt_l1_ptr uint32_t* const kernel_status = reinterpret_cast<tt_l1_ptr uint32_t*>(kernel_status_buf_addr_arg);
tt_l1_ptr volatile chan_req_buf* fvc_consumer_req_buf =
    reinterpret_cast<tt_l1_ptr chan_req_buf*>(FABRIC_ROUTER_REQ_QUEUE_START);
volatile fabric_router_l1_config_t* routing_table =
    reinterpret_cast<tt_l1_ptr fabric_router_l1_config_t*>(eth_l1_mem::address_map::FABRIC_ROUTER_CONFIG_BASE);
uint64_t xy_local_addr;

#define SWITCH_THRESHOLD 0x3FFF

inline void notify_gatekeeper() {
    // send semaphore increment to gatekeeper on this device.
    // semaphore notifies all other routers that this router has completed
    // startup handshake with its ethernet peer.
    uint64_t dest_addr =
        (((uint64_t)gk_message_addr_h << 32) | gk_message_addr_l) + offsetof(gatekeeper_info_t, router_sync);
    noc_fast_atomic_increment<DM_DYNAMIC_NOC>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        dest_addr,
        NOC_UNICAST_WRITE_VC,
        1,
        31,
        false,
        false,
        MEM_NOC_ATOMIC_RET_VAL_ADDR);

    volatile uint32_t* sync_sem_addr = (volatile uint32_t*)FABRIC_ROUTER_SYNC_SEM;
    // wait for all device routers to have incremented the sync semaphore.
    // sync_val is equal to number of tt-fabric routers running on a device.
    while (*sync_sem_addr != sync_val) {
        // context switch while waiting to allow slow dispatch traffic to go through
        internal_::risc_context_switch();
    }
}

void kernel_main() {
#ifdef FVC_MODE_PULL
    fvc_producer_state_t fvc_inbound_state;
#else
    fvc_inbound_push_state_t fvc_inbound_state;
    fvc_outbound_push_state_t fvc_outbound_state[4];
#endif
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    uint32_t rt_args_idx = 0;
    sync_val = get_arg_val<uint32_t>(rt_args_idx++);
    router_mask = get_arg_val<uint32_t>(rt_args_idx++);
    gk_message_addr_l = get_arg_val<uint32_t>(rt_args_idx++);
    gk_message_addr_h = get_arg_val<uint32_t>(rt_args_idx++);

    tt_fabric_init();

    write_kernel_status(kernel_status, TT_FABRIC_STATUS_INDEX, TT_FABRIC_STATUS_STARTED);

    router_state.sync_in = 0;
    router_state.sync_out = 0;

    zero_l1_buf((tt_l1_ptr uint32_t*)fvc_consumer_req_buf, sizeof(chan_req_buf));
    zero_l1_buf((tt_l1_ptr uint32_t*)FVCC_IN_BUF_START, FVCC_IN_BUF_SIZE);
    zero_l1_buf((tt_l1_ptr uint32_t*)FVCC_OUT_BUF_START, FVCC_OUT_BUF_SIZE);
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

#ifdef FVCC_SUPPORT
    fvcc_outbound_state.init(
        FVCC_OUT_BUF_START, FVCC_SYNC_BUF_START, FVCC_IN_BUF_START, (uint32_t)&fvcc_inbound_state.inbound_wrptr);
    fvcc_inbound_state.init(
        FVCC_IN_BUF_START,
        (uint32_t)&fvcc_outbound_state.remote_rdptr,
        (((uint64_t)gk_message_addr_h << 32) | gk_message_addr_l) + offsetof(gatekeeper_info_t, gk_msg_buf));
#endif

    if (!wait_all_src_dest_ready(&router_state, timeout_cycles)) {
        write_kernel_status(kernel_status, TT_FABRIC_STATUS_INDEX, TT_FABRIC_STATUS_TIMEOUT);
        return;
    }

    notify_gatekeeper();
    fvc_inbound_state.register_with_routers<FVC_MODE_ROUTER>(routing_table->my_device_id);
    uint64_t start_timestamp = get_timestamp();

    write_kernel_status(kernel_status, TT_FABRIC_MISC_INDEX, 0xff000001);
    uint32_t loop_count = 0;

    uint32_t launch_msg_rd_ptr = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    uint32_t curr_outbound_buffer = 0;
    uint32_t next_outbound_buffer = 1;
    uint32_t eth_outbound_wrptr = 0;
    tt_l1_ptr launch_msg_t* const launch_msg = GET_MAILBOX_ADDRESS_DEV(launch[launch_msg_rd_ptr]);
    while (1) {
        // Handle Ethernet Outbound Data
#ifdef FVC_MODE_PULL
        if (!fvc_req_buf_is_empty(fvc_consumer_req_buf) && fvc_req_valid(fvc_consumer_req_buf)) {
            uint32_t req_index = fvc_consumer_req_buf->rdptr.ptr & CHAN_REQ_BUF_SIZE_MASK;
            chan_request_entry_t* req = (chan_request_entry_t*)fvc_consumer_req_buf->chan_req + req_index;
            pull_request_t* pull_req = &req->pull_request;
            if (req->bytes[47] == FORWARD) {
                // Data is packetized.
                fvc_outbound_state.pull_data_to_fvc_buffer(pull_req);
                if (fvc_outbound_state.packet_words_remaining == 0 ||
                    fvc_outbound_state.pull_words_in_flight >= FVC_SYNC_THRESHOLD) {
                    fvc_outbound_state.total_words_to_forward += fvc_outbound_state.pull_words_in_flight;
                    fvc_outbound_state.pull_words_in_flight = 0;
                    fvc_outbound_state.forward_data_from_fvc_buffer<true>();
                    // noc_async_read_barrier();
                    update_pull_request_words_cleared(pull_req);
                }
            } else if (req->bytes[47] == INLINE_FORWARD) {
                fvc_outbound_state.move_data_to_fvc_buffer(pull_req);
            }

            if (fvc_outbound_state.packet_in_progress == 1 and fvc_outbound_state.packet_words_remaining == 0) {
                // clear the flags field to invalidate pull request slot.
                // flags will be set to non-zero by next requestor.
                req_buf_advance_rdptr((chan_req_buf*)fvc_consumer_req_buf);
                fvc_outbound_state.packet_in_progress = 0;
            }
            loop_count = 0;
        }

        if (fvc_outbound_state.total_words_to_forward) {
            fvc_outbound_state.forward_data_from_fvc_buffer<false>();
        }
#else
        if (fvc_outbound_state[curr_outbound_buffer].forward_data_from_fvc_buffer(eth_outbound_wrptr)) {
            loop_count = 0;
        } else if (*fvc_outbound_state[next_outbound_buffer].noc_word_credits) {
            curr_outbound_buffer = next_outbound_buffer;
            next_outbound_buffer = (next_outbound_buffer + 1) & 0x3;
        } else {
            next_outbound_buffer = (next_outbound_buffer + 1) & 0x3;
        }
#endif

        // Handle Ethernet Inbound Data
        if (fvc_inbound_state.get_curr_packet_valid<FVC_MODE_ROUTER>()) {
            fvc_inbound_state.process_inbound_packet<FVC_MODE_ROUTER>();
            loop_count = 0;
        }
#ifdef TT_FABRIC_DEBUG
        else if (fvc_inbound_state.packet_corrupted) {
            write_kernel_status(kernel_status, TT_FABRIC_STATUS_INDEX, TT_FABRIC_STATUS_BAD_HEADER);
            return;
        }
#endif

#ifdef FVCC_SUPPORT
        fvcc_inbound_state.fvcc_handler();
        fvcc_outbound_state.fvcc_handler();
#endif

        loop_count++;

        // need to optimize this.
        // context switch to base fw is very costly.
        if ((loop_count & SWITCH_THRESHOLD) == SWITCH_THRESHOLD) {
            internal_::risc_context_switch();

            if (*(volatile uint32_t*)FABRIC_ROUTER_SYNC_SEM == 0) {
                // terminate signal from host sw.
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
}
