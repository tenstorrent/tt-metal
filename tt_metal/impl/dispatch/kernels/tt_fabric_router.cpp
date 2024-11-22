// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "tt_metal/impl/dispatch/kernels/tt_fabric.hpp"
// clang-format on

router_state_t router_state __attribute__((aligned(16)));
fvc_consumer_state_t fvc_consumer_state __attribute__((aligned(16))); // replicate for each fvc
fvc_producer_state_t fvc_producer_state __attribute__((aligned(16))); // replicate for each fvc
volatile local_pull_request_t local_pull_request_temp __attribute__((aligned(16))); // replicate for each fvc
volatile local_pull_request_t *local_pull_request = &local_pull_request_temp; // replicate for each fvc

constexpr uint32_t fvc_consumer_req_buf_start = get_compile_time_arg_val(0);
constexpr uint32_t fvc_data_buf_start = get_compile_time_arg_val(1);
constexpr uint32_t fvc_data_buf_size_words = get_compile_time_arg_val(2);
constexpr uint32_t fvc_data_buf_size_bytes = fvc_data_buf_size_words * PACKET_WORD_SIZE_BYTES;
constexpr uint32_t kernel_status_buf_addr_arg = get_compile_time_arg_val(3);
constexpr uint32_t kernel_status_buf_size_bytes = get_compile_time_arg_val(4);
constexpr uint32_t timeout_cycles = get_compile_time_arg_val(5);


constexpr uint32_t PACKET_QUEUE_STAUS_MASK = 0xabc00000;
constexpr uint32_t PACKET_QUEUE_TEST_STARTED = PACKET_QUEUE_STAUS_MASK | 0x0;
constexpr uint32_t PACKET_QUEUE_TEST_PASS = PACKET_QUEUE_STAUS_MASK | 0x1;
constexpr uint32_t PACKET_QUEUE_TEST_TIMEOUT = PACKET_QUEUE_STAUS_MASK | 0xdead;
constexpr uint32_t PACKET_QUEUE_TEST_DATA_MISMATCH = PACKET_QUEUE_STAUS_MASK | 0x3;

// indexes of return values in test results buffer
constexpr uint32_t PQ_TEST_STATUS_INDEX = 0;
constexpr uint32_t PQ_TEST_WORD_CNT_INDEX = 2;
constexpr uint32_t PQ_TEST_CYCLES_INDEX = 4;
constexpr uint32_t PQ_TEST_ITER_INDEX = 6;
constexpr uint32_t PQ_TEST_MISC_INDEX = 16;

// careful, may be null
tt_l1_ptr uint32_t* const kernel_status = reinterpret_cast<tt_l1_ptr uint32_t*>(kernel_status_buf_addr_arg);
tt_l1_ptr volatile chan_req_buf* fvc_consumer_req_buf = reinterpret_cast<tt_l1_ptr chan_req_buf*>(fvc_consumer_req_buf_start);
uint64_t xy_local_addr;

#define SWITCH_THRESHOLD 16
void kernel_main() {
    rtos_context_switch_ptr = (void (*)())RtosTable[0];

    tt_fabric_init();

    write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_STARTED);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000000);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX + 1, 0xbb000000);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX + 2, 0xAABBCCDD);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX + 3, 0xDDCCBBAA);

    zero_l1_buf((tt_l1_ptr uint32_t*)fvc_consumer_req_buf, sizeof(chan_req_buf));
    router_state.sync_in = 0;
    router_state.sync_out = 0;
    write_kernel_status(kernel_status, PQ_TEST_WORD_CNT_INDEX, (uint32_t)&router_state);
    write_kernel_status(kernel_status, PQ_TEST_WORD_CNT_INDEX+1, (uint32_t)&fvc_consumer_state);
    write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX+1, (uint32_t)&fvc_producer_state);

    if (!wait_all_src_dest_ready(&router_state, timeout_cycles)) {
        write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_TIMEOUT);
        return;
    }

    uint32_t my_chip_xy = (*(volatile uint32_t *)0x1108);

    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000001);
    uint32_t loop_count = 0;
    fvc_consumer_state.init(fvc_data_buf_start, fvc_data_buf_size_words / 2, (uint32_t)&fvc_producer_state.inbound_wrptr);
    fvc_producer_state.init(fvc_data_buf_start + (fvc_data_buf_size_words * PACKET_WORD_SIZE_BYTES / 2), fvc_data_buf_size_words / 2, (uint32_t)&fvc_consumer_state.remote_rdptr);
    while (1) {
        if (!fvc_req_buf_is_empty(fvc_consumer_req_buf) && fvc_req_valid(fvc_consumer_req_buf)) {

            uint32_t req_index = fvc_consumer_req_buf->rdptr.ptr & CHAN_REQ_BUF_SIZE_MASK;
            chan_request_entry_t *req = (chan_request_entry_t *)fvc_consumer_req_buf->chan_req + req_index;

            if (req->bytes[47] & FORWARD)
            {
                // Data is packetized.
                pull_request_t *pull_req = &req->pull_request;
                if (!fvc_consumer_state.sync_buf_full() && !fvc_consumer_state.sync_pending) {
                    pull_data_to_fvc_buffer(pull_req, &fvc_consumer_state);
                }
                if (!fvc_consumer_state.sync_buf_empty()) {
                    noc_async_read_barrier();
                    if (fvc_consumer_state.pull_words_in_flight) {
                        //send words cleared count to producer/sender of pull request.
                        update_pull_request_words_cleared(pull_req);
                        fvc_consumer_state.pull_words_in_flight = 0;
                    }
                    fvc_consumer_state.forward_data_from_fvc_buffer();
                }
                fvc_consumer_state.check_sync_pending();
                if (fvc_consumer_state.packet_in_progress == 1 and fvc_consumer_state.packet_words_remaining == 0 and fvc_consumer_state.pull_words_in_flight == 0) {
                    fvc_consumer_req_buf->rdptr.ptr++;
                    fvc_consumer_state.packet_in_progress = 0;
                }
            }
        }

        fvc_producer_state.update_remote_rdptr_sent();
        if (fvc_producer_state.get_curr_packet_valid()) {
            fvc_producer_state.process_inbound_packet();
        }

        loop_count++;
        if (loop_count >= 1000000) {
            break;
        }

    }



    bool all_outputs_finished = false;
    uint64_t start_timestamp = get_timestamp();
    uint32_t switch_counter = 0;


    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000002);

    uint64_t cycles_elapsed = get_timestamp() - start_timestamp;
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff000003);

    set_64b_result(kernel_status, cycles_elapsed, PQ_TEST_CYCLES_INDEX);

    write_kernel_status(kernel_status, PQ_TEST_STATUS_INDEX, PACKET_QUEUE_TEST_PASS);
    write_kernel_status(kernel_status, PQ_TEST_MISC_INDEX, 0xff00005);
}
