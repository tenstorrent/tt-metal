// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "dataflow_api.h"
#include "noc_overlay_parameters.h"
#include "ethernet/dataflow_api.h"
#include "tt_fabric.hpp"

typedef struct tt_fabric_endpoint_sync {
    uint32_t sync_addr : 24;
    uint32_t endpoint_type : 8
} tt_fabric_endpoint_sync_t;

static_assert(sizeof(tt_fabric_endpoint_sync_t) == 4);

extern local_pull_request_t local_pull_request;

inline void fabric_async_write(
    uint32_t routing_plane,      // the network plane to use for this transaction
    uint32_t src_addr,           // source address in sender’s memory
    uint64_t dst_addr,           // destination write address
    uint32_t size,               // number of bytes to write to remote destination
    uint32_t& fvc,               // fabric virtual channel. Set to –1 for automatic selection
    uint32_t return_status_addr  // TT-Fabric returns api call status at this address
) {
    uint32_t size_in_words = (size + PACKET_WORD_SIZE_BYTES - 1) >> 4;
    local_pull_request.pull_request.wr_ptr = size_in_words;
    local_pull_request.pull_request.rd_ptr = 0;
    local_pull_request.pull_request.size = size;
    local_pull_request.pull_request.buffer_size = size_in_words;
    local_pull_request.pull_request.buffer_start = src_addr;
    local_pull_request.pull_request.ack_addr = -1;
    local_pull_request.pull_request.flags = FORWARD;

    uint64_t router_request_queue = NOC_XY_ADDR(2, 0, 0x19000);
    tt_fabric_send_pull_request(router_request_queue, &local_pull_request);
}

/**
 *  Polling for ready signal from the remote peers of all input and output queues.
 *  Blocks until all are ready, but doesn't block polling on each individual queue.
 *  Returns false in case of timeout.
 */
bool wait_all_src_dest_ready(volatile router_state_t* router_state, uint32_t timeout_cycles = 0) {
    bool src_ready = false;
    bool dest_ready = false;

    uint32_t iters = 0;

    uint32_t start_timestamp = get_timestamp_32b();
    uint32_t sync_in_addr = ((uint32_t)&router_state->sync_in) / PACKET_WORD_SIZE_BYTES;
    uint32_t sync_out_addr = ((uint32_t)&router_state->sync_out) / PACKET_WORD_SIZE_BYTES;

    uint32_t scratch_addr = ((uint32_t)&router_state->scratch) / PACKET_WORD_SIZE_BYTES;
    router_state->scratch[0] = 0xAA;
    // send_buf[1] = 0x0;
    // send_buf[2] = 0x0;
    // send_buf[3] = 0x0;

    while (!src_ready or !dest_ready) {
        if (router_state->sync_out != 0xAA) {
            internal_::eth_send_packet(0, scratch_addr, sync_in_addr, 1);
        } else {
            dest_ready = true;
        }

        if (!src_ready && router_state->sync_in == 0xAA) {
            internal_::eth_send_packet(0, sync_in_addr, sync_out_addr, 1);
            src_ready = true;
        }

        iters++;
        if (timeout_cycles > 0) {
            uint32_t cycles_since_start = get_timestamp_32b() - start_timestamp;
            if (cycles_since_start > timeout_cycles) {
                return false;
            }
        }

#if defined(COMPILE_FOR_ERISC)
        if ((timeout_cycles == 0) && (iters & 0xFFF) == 0) {
            // if timeout is disabled, context switch every 4096 iterations.
            // this is necessary to allow ethernet routing layer to operate.
            // this core may have pending ethernet routing work.
            internal_::risc_context_switch();
        }
#endif
    }
    return true;
}
