// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "tt_metal/hw/inc/dataflow_api.h"
#include "tt_fabric/hw/inc/tt_fabric.h"
// clang-format on

router_state_t router_state __attribute__((aligned(16)));
fvc_consumer_state_t fvc_consumer_state __attribute__((aligned(16)));                // replicate for each fvc
fvc_producer_state_t fvc_producer_state __attribute__((aligned(16)));                // replicate for each fvc
volatile local_pull_request_t local_pull_request_temp __attribute__((aligned(16)));  // replicate for each fvc
volatile local_pull_request_t* local_pull_request = &local_pull_request_temp;        // replicate for each fvc

constexpr uint32_t fvc_data_buf_size_words = get_compile_time_arg_val(0);
constexpr uint32_t fvc_data_buf_size_bytes = fvc_data_buf_size_words * PACKET_WORD_SIZE_BYTES;
constexpr uint32_t kernel_status_buf_addr_arg = get_compile_time_arg_val(1);
constexpr uint32_t kernel_status_buf_size_bytes = get_compile_time_arg_val(2);
// constexpr uint32_t sync_val = get_compile_time_arg_val(3);
// constexpr uint32_t router_mask = get_compile_time_arg_val(4);
constexpr uint32_t timeout_cycles = get_compile_time_arg_val(3);
uint32_t sync_val;
uint32_t router_mask;

constexpr uint32_t PACKET_QUEUE_STAUS_MASK = 0xabc00000;
constexpr uint32_t PACKET_QUEUE_TEST_STARTED = PACKET_QUEUE_STAUS_MASK | 0x0;
constexpr uint32_t PACKET_QUEUE_TEST_PASS = PACKET_QUEUE_STAUS_MASK | 0x1;
constexpr uint32_t PACKET_QUEUE_TEST_TIMEOUT = PACKET_QUEUE_STAUS_MASK | 0xdead0;
constexpr uint32_t PACKET_QUEUE_TEST_BAD_HEADER = PACKET_QUEUE_STAUS_MASK | 0xdead1;
constexpr uint32_t PACKET_QUEUE_TEST_DATA_MISMATCH = PACKET_QUEUE_STAUS_MASK | 0x3;

// indexes of return values in test results buffer
constexpr uint32_t PQ_TEST_STATUS_INDEX = 0;
constexpr uint32_t PQ_TEST_WORD_CNT_INDEX = 2;
constexpr uint32_t PQ_TEST_CYCLES_INDEX = 4;
constexpr uint32_t PQ_TEST_ITER_INDEX = 6;
constexpr uint32_t PQ_TEST_MISC_INDEX = 16;

// careful, may be null
tt_l1_ptr uint32_t* const kernel_status = reinterpret_cast<tt_l1_ptr uint32_t*>(kernel_status_buf_addr_arg);
tt_l1_ptr volatile chan_req_buf* fvc_consumer_req_buf =
    reinterpret_cast<tt_l1_ptr chan_req_buf*>(FABRIC_ROUTER_REQ_QUEUE_START);
tt_l1_ptr volatile tt::tt_fabric::fabric_router_l1_config_t* routing_table =
    reinterpret_cast<tt_l1_ptr tt::tt_fabric::fabric_router_l1_config_t*>(
        eth_l1_mem::address_map::FABRIC_ROUTER_CONFIG_BASE);
uint64_t xy_local_addr;

#define SWITCH_THRESHOLD 0x3FF

inline void notify_all_routers() {
    uint32_t channel = 0;
    uint32_t remaining_cores = router_mask;
    // send semaphore increment to all fabric routers on this device.
    // semaphore notifies all other routers that this router has completed
    // startup handshake with its ethernet peer.
    for (uint32_t i = 0; i < 16; i++) {
        if (remaining_cores == 0) {
            break;
        }
        if (remaining_cores & (0x1 << i)) {
            uint64_t dest_addr = ((uint64_t)eth_chan_to_noc_xy[noc_index][i] << 32) | FABRIC_ROUTER_SYNC_SEM;
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
        }
    }
    volatile uint32_t* sync_sem_addr = (volatile uint32_t*)FABRIC_ROUTER_SYNC_SEM;
    // wait for all device routers to have incremented the sync semaphore.
    // sync_val is equal to number of tt-fabric routers running on a device.
    while (*sync_sem_addr != sync_val) {
        // context switch while waiting to allow slow dispatch traffic to go through
        internal_::risc_context_switch();
    }
}

void kernel_main() {
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    sync_val = get_arg_val<uint32_t>(0);
    router_mask = get_arg_val<uint32_t>(1);

    tt_fabric_init();

    write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_STARTED);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000000);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX + 1, 0xbb000000);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX + 2, 0xAABBCCDD);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX + 3, 0xDDCCBBAA);

    router_state.sync_in = 0;
    router_state.sync_out = 0;

    zero_l1_buf((tt_l1_ptr uint32_t*)fvc_consumer_req_buf, sizeof(chan_req_buf));
    write_kernel_status(kernel_status, PQ_TEST_WORD_CNT_INDEX, (uint32_t)&router_state);
    write_kernel_status(kernel_status, PQ_TEST_WORD_CNT_INDEX + 1, (uint32_t)&fvc_consumer_state);
    write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX + 1, (uint32_t)&fvc_producer_state);

    fvc_consumer_state.init(
        FABRIC_ROUTER_DATA_BUF_START, fvc_data_buf_size_words / 2, (uint32_t)&fvc_producer_state.inbound_wrptr);
    fvc_producer_state.init(
        FABRIC_ROUTER_DATA_BUF_START + (fvc_data_buf_size_words * PACKET_WORD_SIZE_BYTES / 2),
        fvc_data_buf_size_words / 2,
        (uint32_t)&fvc_consumer_state.remote_rdptr);

    if (!wait_all_src_dest_ready(&router_state, timeout_cycles)) {
        write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
        return;
    }

    notify_all_routers();
    uint64_t start_timestamp = get_timestamp();

    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000001);
    uint32_t loop_count = 0;
    uint32_t total_words_procesed = 0;

    while (1) {
        if (!fvc_req_buf_is_empty(fvc_consumer_req_buf) && fvc_req_valid(fvc_consumer_req_buf)) {
            uint32_t req_index = fvc_consumer_req_buf->rdptr.ptr & CHAN_REQ_BUF_SIZE_MASK;
            chan_request_entry_t* req = (chan_request_entry_t*)fvc_consumer_req_buf->chan_req + req_index;
            pull_request_t* pull_req = &req->pull_request;
            bool can_pull = !fvc_consumer_state.sync_buf_full() && !fvc_consumer_state.sync_pending;
            if (req->bytes[47] == FORWARD) {
                // Data is packetized.
                if (can_pull) {
                    pull_data_to_fvc_buffer(pull_req, &fvc_consumer_state);
                }
                if (!fvc_consumer_state.sync_buf_empty()) {
                    noc_async_read_barrier();
                    if (fvc_consumer_state.pull_words_in_flight) {
                        // send words cleared count to producer/sender of pull request.
                        update_pull_request_words_cleared(pull_req);
                        fvc_consumer_state.pull_words_in_flight = 0;
                    }
                }

            } else if (req->bytes[47] == INLINE_FORWARD) {
                if (can_pull) {
                    move_data_to_fvc_buffer(pull_req, &fvc_consumer_state);
                    fvc_consumer_state.pull_words_in_flight = 0;
                }
            }

            // Flush all sync entries here.
            // There can be a previous pending sync entry plus another one
            // from current inovcation of pull_data_to_fvc_buffer()
            // current invocatoin may still result in sync pending,
            // while previous sync pending has been serviced and pushed to sync buf
            while (!fvc_consumer_state.sync_buf_empty()) {
                if (fvc_consumer_state.forward_data_from_fvc_buffer<true>() == 0) {
                    // not able to forward any data over ethernet.
                    // should break and retry.
                    break;
                }
            }
            if (!fvc_consumer_state.check_sync_pending()) {
                if (fvc_consumer_state.packet_in_progress == 1 and fvc_consumer_state.packet_words_remaining == 0) {
                    // clear the flags field to invalidate pull request slot.
                    // flags will be set to non-zero by next requestor.
                    req_buf_advance_rdptr((chan_req_buf*)fvc_consumer_req_buf);
                    fvc_consumer_state.packet_in_progress = 0;
                }
            }
            loop_count = 0;
        }

        if (fvc_req_buf_is_empty(fvc_consumer_req_buf)) {
            noc_async_read_barrier();
            while (!fvc_consumer_state.sync_buf_empty()) {
                if (fvc_consumer_state.forward_data_from_fvc_buffer<true>() == 0) {
                    // not able to forward any data over ethernet.
                    // should break and retry.
                    break;
                }
            }
        }

        fvc_producer_state.update_remote_rdptr_sent();
        if (fvc_producer_state.get_curr_packet_valid()) {
            if (total_words_procesed == 0) {
                start_timestamp = get_timestamp();
            }
            total_words_procesed += fvc_producer_state.process_inbound_packet();
            loop_count = 0;
        } else if (fvc_producer_state.packet_corrupted) {
            write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_BAD_HEADER);
            return;
        }

        loop_count++;

        // need to optimize this.
        // context switch to base fw is very costly.
        if ((loop_count & SWITCH_THRESHOLD) == SWITCH_THRESHOLD) {
            internal_::risc_context_switch();
        }
        if (*(volatile uint32_t*)FABRIC_ROUTER_SYNC_SEM == 0) {
            // terminate signal from host sw.
            if (loop_count >= 0x1000) {
                break;
            }
        }
    }
    uint64_t cycles_elapsed = fvc_producer_state.packet_timestamp - start_timestamp;

    DPRINT << "Router words processed " << total_words_procesed << ENDL();

    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000002);

    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000003);

    set_64b_result(kernel_status, cycles_elapsed, PQ_TEST_CYCLES_INDEX);

    if (fvc_consumer_state.packet_in_progress) {
        write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
    } else {
        write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_PASS);
    }
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff00005);
}
