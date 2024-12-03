// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "dataflow_api.h"
#include "noc_overlay_parameters.h"
#include "ethernet/dataflow_api.h"
#include "tt_fabric_interface.h"

constexpr ProgrammableCoreType fd_core_type = static_cast<ProgrammableCoreType>(FD_CORE_TYPE);

const uint32_t SYNC_BUF_SIZE = 16;  // must be 2^N
const uint32_t SYNC_BUF_SIZE_MASK = (SYNC_BUF_SIZE - 1);
const uint32_t SYNC_BUF_PTR_MASK = ((SYNC_BUF_SIZE << 1) - 1);

extern uint64_t xy_local_addr;
extern volatile local_pull_request_t* local_pull_request;
extern volatile tt::tt_fabric::fabric_router_l1_config_t* routing_table;

uint64_t tt_fabric_send_pull_request(uint64_t dest_addr, volatile local_pull_request_t* local_pull_request);
bool tt_fabric_is_header_valid(packet_header_t* p_header);

typedef struct fvc_consumer_state {
    volatile chan_payload_ptr remote_rdptr;
    uint32_t remote_ptr_update_addr;
    uint8_t chan_num;
    uint8_t packet_in_progress;
    uint8_t sync_buf_wrptr;
    uint8_t sync_buf_rdptr;
    uint32_t packet_words_remaining;
    uint32_t fvc_out_wrptr;
    uint32_t fvc_out_rdptr;
    uint32_t fvc_pull_wrptr;
    uint32_t buffer_size;
    uint32_t buffer_start;
    uint32_t remote_buffer_start;
    uint32_t pull_words_in_flight;
    uint32_t words_since_last_sync;
    uint32_t words_to_forward;
    uint8_t sync_pending;
    uint8_t padding[3];
    uint32_t sync_buf[SYNC_BUF_SIZE];

    uint32_t get_num_words_free() {
        uint32_t rd_ptr = remote_rdptr.ptr;
        uint32_t words_occupied = 0;
        if (fvc_pull_wrptr != rd_ptr) {
            words_occupied =
                fvc_pull_wrptr > rd_ptr ? fvc_pull_wrptr - rd_ptr : buffer_size * 2 + fvc_pull_wrptr - rd_ptr;
        }
        return buffer_size - words_occupied;
    }

    uint32_t get_remote_num_words_free() {
        uint32_t rd_ptr = remote_rdptr.ptr_cleared;
        uint32_t words_occupied = 0;
        if (fvc_out_wrptr != rd_ptr) {
            words_occupied = fvc_out_wrptr > rd_ptr ? fvc_out_wrptr - rd_ptr : buffer_size * 2 + fvc_out_wrptr - rd_ptr;
        }
        return buffer_size - words_occupied;
    }

    inline void init(uint32_t data_buf_start, uint32_t data_buf_size_words, uint32_t ptr_update_addr) {
        uint32_t words = sizeof(fvc_consumer_state) / 4;
        uint32_t* ptr = (uint32_t*)this;
        for (uint32_t i = 0; i < words; i++) {
            ptr[i] = 0;
        }
        chan_num = 1;
        buffer_start = data_buf_start;
        buffer_size = data_buf_size_words;
        remote_buffer_start = data_buf_start + buffer_size * PACKET_WORD_SIZE_BYTES;
        remote_ptr_update_addr = ptr_update_addr;
    }

    inline uint32_t words_before_buffer_wrap(uint32_t ptr) {
        if (ptr >= buffer_size) {
            return buffer_size * 2 - ptr;
        } else {
            return buffer_size - ptr;
        }
    }

    inline uint32_t words_before_local_buffer_wrap() {
        if (fvc_pull_wrptr >= buffer_size) {
            return buffer_size * 2 - fvc_pull_wrptr;
        } else {
            return buffer_size - fvc_pull_wrptr;
        }
    }

    inline uint32_t get_local_buffer_pull_addr() {
        uint32_t addr = buffer_start;
        uint32_t offset = fvc_pull_wrptr;
        if (offset >= buffer_size) {
            offset -= buffer_size;
        }
        addr = addr + (offset * PACKET_WORD_SIZE_BYTES);
        return addr;
    }

    inline uint32_t get_local_buffer_read_addr() {
        uint32_t addr = buffer_start;
        uint32_t offset = fvc_out_rdptr;
        if (offset >= buffer_size) {
            offset -= buffer_size;
        }
        addr = addr + (offset * PACKET_WORD_SIZE_BYTES);
        return addr;
    }

    inline uint32_t get_remote_buffer_write_addr() {
        uint32_t addr = remote_buffer_start;
        uint32_t offset = fvc_out_wrptr;
        if (offset >= buffer_size) {
            offset -= buffer_size;
        }
        addr = addr + (offset * PACKET_WORD_SIZE_BYTES);
        return addr;
    }

    inline void advance_pull_wrptr(uint32_t num_words) {
        uint32_t temp = fvc_pull_wrptr + num_words;
        if (temp >= buffer_size * 2) {
            temp -= buffer_size * 2;
        }
        fvc_pull_wrptr = temp;
    }

    inline void advance_out_wrptr(uint32_t num_words) {
        uint32_t temp = fvc_out_wrptr + num_words;
        if (temp >= buffer_size * 2) {
            temp -= buffer_size * 2;
        }
        fvc_out_wrptr = temp;
    }

    inline void advance_out_rdptr(uint32_t num_words) {
        uint32_t temp = fvc_out_rdptr + num_words;
        if (temp >= buffer_size * 2) {
            temp -= buffer_size * 2;
        }
        fvc_out_rdptr = temp;
    }

    inline void register_pull_data(uint32_t num_words_to_pull) {
        pull_words_in_flight += num_words_to_pull;
        advance_pull_wrptr(num_words_to_pull);
        words_since_last_sync += num_words_to_pull;
        packet_words_remaining -= num_words_to_pull;
        // also check for complete packet pulled.
        if ((packet_words_remaining == 0) or (words_since_last_sync >= FVC_SYNC_THRESHOLD)) {
            sync_buf[sync_buf_wrptr] = fvc_pull_wrptr;
            if (get_num_words_free()) {
                advance_pull_wrptr(1);
                sync_buf_advance_wrptr();
            } else {
                sync_pending = 1;
            }
            words_since_last_sync = 0;
        }
    }

    inline bool check_sync_pending() {
        if (sync_pending) {
            if (get_num_words_free()) {
                advance_pull_wrptr(1);
                sync_buf_advance_wrptr();
                sync_pending = 0;
            }
            return true;
        }
        return false;
    }

    inline uint32_t forward_data_from_fvc_buffer() {
        uint32_t total_words_to_forward = 0;
        uint32_t wrptr = sync_buf[sync_buf_rdptr];

        total_words_to_forward =
            wrptr > fvc_out_rdptr ? wrptr - fvc_out_rdptr : buffer_size * 2 + wrptr - fvc_out_rdptr;

        uint32_t remote_fvc_buffer_space = get_remote_num_words_free();
        if (remote_fvc_buffer_space < (total_words_to_forward + 1)) {
            // +1 is for pointer sync word.
            // If fvc receiver buffer on link partner does not have space to receive the
            // full sync buffer entry, we skip and try again next time.
            return 0;
        }

        // Now that there is enough space in receiver buffer we will send total_words_to_forward number of words.
        // This means that we may need to break up the writes to multiple ethernet packets
        // depending on whether local buffer is wrapping, remote buffer is wrapping,
        // we are writing sync word etc.

        uint32_t src_addr = 0;
        uint32_t dest_addr = 0;  // should be second half of fvc buffer.
        uint32_t words_remaining = total_words_to_forward;
        while (words_remaining) {
            uint32_t num_words_before_local_wrap = words_before_buffer_wrap(fvc_out_rdptr);
            uint32_t num_words_before_remote_wrap = words_before_buffer_wrap(fvc_out_wrptr);
            ;
            uint32_t words_to_forward = std::min(num_words_before_local_wrap, num_words_before_remote_wrap);
            words_to_forward = std::min(words_to_forward, words_remaining);
            words_to_forward = std::min(words_to_forward, DEFAULT_MAX_ETH_SEND_WORDS);
            src_addr = get_local_buffer_read_addr();
            dest_addr = get_remote_buffer_write_addr();

            internal_::eth_send_packet(
                0, src_addr / PACKET_WORD_SIZE_BYTES, dest_addr / PACKET_WORD_SIZE_BYTES, words_to_forward);
            advance_out_rdptr(words_to_forward);
            advance_out_wrptr(words_to_forward);
            words_remaining -= words_to_forward;
        }
        // after sending all the data, send the last word which is pointer sync word.
        volatile uint32_t* sync_ptr = (volatile uint32_t*)get_local_buffer_read_addr();
        advance_out_rdptr(1);
        sync_ptr[0] = fvc_out_wrptr;
        sync_ptr[1] = 0;
        sync_ptr[2] = 0;
        sync_ptr[3] = fvc_out_rdptr;
        internal_::eth_send_packet(
            0, ((uint32_t)sync_ptr) / PACKET_WORD_SIZE_BYTES, remote_ptr_update_addr / PACKET_WORD_SIZE_BYTES, 1);
        sync_buf_advance_rdptr();
        return total_words_to_forward;
    }

    inline void sync_buf_advance_wrptr() { sync_buf_wrptr = (sync_buf_wrptr + 1) & SYNC_BUF_PTR_MASK; }

    inline void sync_buf_advance_rdptr() { sync_buf_rdptr = (sync_buf_rdptr + 1) & SYNC_BUF_PTR_MASK; }

    inline bool sync_buf_empty() { return (sync_buf_wrptr == sync_buf_rdptr); }

    inline bool sync_buf_full() {
        return !sync_buf_empty() && ((sync_buf_wrptr & SYNC_BUF_SIZE_MASK) == (sync_buf_rdptr & SYNC_BUF_SIZE_MASK));
    }

} fvc_consumer_state_t;

static_assert(sizeof(fvc_consumer_state_t) % 4 == 0);

#define FVC_MODE_ROUTER 1
#define FVC_MODE_ENDPOINT 2

// FVC Producer holds data that needs to be forwarded to other destinations.
// This producer receives data over ethernet from neighboring chip.
// Data in the producer is either destined for local chip, or has to make a noc hop
// to ethernet port enroute to final destination.
// FVC producer buffer issues pull requests to other entities in the fabric node to
// pull data from Producer buffer. Pull requests can be made to next router/consumer buffer in the route
// direction, socket receiver/consumer buffer, center worker/consumer buffer.
// Which ever entity receives the pull request is responsible draining the required amount of data from
// FVC Producer.
typedef struct fvc_producer_state {
    volatile chan_payload_ptr inbound_wrptr;
    volatile chan_payload_ptr inbound_rdptr;
    uint32_t remote_ptr_update_addr;
    uint8_t chan_num;
    uint8_t packet_in_progress;
    uint8_t pad1;
    uint8_t pad2;
    uint32_t packet_words_remaining;
    uint32_t packet_words_sent;
    uint32_t fvc_out_wrptr;
    uint32_t fvc_out_rdptr;
    volatile uint32_t fvc_pull_rdptr;
    uint32_t buffer_size;
    uint32_t buffer_start;
    uint32_t pull_words_in_flight;
    uint32_t words_since_last_sync;
    uint32_t words_to_forward;
    bool curr_packet_valid;
    bool packet_corrupted;
    uint64_t packet_dest;
    packet_header_t current_packet_header;

    inline void init(uint32_t data_buf_start, uint32_t data_buf_size_words, uint32_t ptr_update_addr) {
        uint32_t words = sizeof(fvc_producer_state) / 4;
        uint32_t* ptr = (uint32_t*)this;
        for (uint32_t i = 0; i < words; i++) {
            ptr[i] = 0;
        }
        chan_num = 1;
        buffer_start = data_buf_start;
        buffer_size = data_buf_size_words;
        remote_ptr_update_addr = ptr_update_addr;
    }

    inline uint32_t inc_ptr_with_wrap(uint32_t ptr, uint32_t inc) {
        uint32_t temp = ptr + inc;
        if (temp >= buffer_size * 2) {
            temp -= buffer_size * 2;
        }
        return temp;
    }

    inline void advance_local_wrptr(uint32_t num_words) {
        inbound_wrptr.ptr = inc_ptr_with_wrap(inbound_wrptr.ptr, num_words);
    }

    inline void advance_out_wrptr(uint32_t num_words) { fvc_out_wrptr = inc_ptr_with_wrap(fvc_out_wrptr, num_words); }

    inline void advance_out_rdptr(uint32_t num_words) { fvc_out_rdptr = inc_ptr_with_wrap(fvc_out_rdptr, num_words); }

    inline uint32_t words_before_buffer_wrap(uint32_t ptr) {
        if (ptr >= buffer_size) {
            return buffer_size * 2 - ptr;
        } else {
            return buffer_size - ptr;
        }
    }

    inline uint32_t get_num_words_available() const {
        uint32_t wrptr = inbound_wrptr.ptr;
        uint32_t words_occupied = 0;
        if (fvc_out_rdptr != wrptr) {
            words_occupied = wrptr > fvc_out_rdptr ? wrptr - fvc_out_rdptr : buffer_size * 2 + wrptr - fvc_out_rdptr;
        }
        return words_occupied;
    }

    inline uint32_t get_num_words_free() {
        uint32_t wrptr = inbound_wrptr.ptr;
        uint32_t words_occupied = 0;
        if (fvc_pull_rdptr != wrptr) {
            words_occupied = wrptr > fvc_pull_rdptr ? wrptr - fvc_pull_rdptr : buffer_size * 2 + wrptr - fvc_pull_rdptr;
        }
        return buffer_size - words_occupied;
    }

    inline bool get_curr_packet_valid() {
        if (!curr_packet_valid && (get_num_words_available() >= PACKET_HEADER_SIZE_WORDS)) {
            // Wait for a full packet header to arrive before advancing to next packet.
            this->advance_next_packet();
        }
        return this->curr_packet_valid;
    }

    inline uint32_t get_local_buffer_read_addr() {
        uint32_t addr = buffer_start;
        uint32_t offset = fvc_out_rdptr;
        if (offset >= buffer_size) {
            offset -= buffer_size;
        }
        addr = addr + (offset * PACKET_WORD_SIZE_BYTES);
        return addr;
    }

    inline uint32_t get_local_buffer_write_addr() {
        uint32_t addr = buffer_start;
        uint32_t offset = inbound_wrptr.ptr;
        if (offset >= buffer_size) {
            offset -= buffer_size;
        }
        addr = addr + (offset * PACKET_WORD_SIZE_BYTES);
        return addr;
    }

    inline uint32_t words_before_local_buffer_wrap() {
        if (inbound_wrptr.ptr >= buffer_size) {
            return buffer_size * 2 - inbound_wrptr.ptr;
        } else {
            return buffer_size - inbound_wrptr.ptr;
        }
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    inline void update_remote_rdptr_sent() {
        if (inbound_wrptr.ptr_cleared != inbound_rdptr.ptr) {
            inbound_rdptr.ptr = inbound_wrptr.ptr_cleared;
            if constexpr (fvc_mode == FVC_MODE_ROUTER) {
                internal_::eth_send_packet(
                    0,
                    ((uint32_t)&inbound_rdptr) / PACKET_WORD_SIZE_BYTES,
                    remote_ptr_update_addr / PACKET_WORD_SIZE_BYTES,
                    1);
            }
        }
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    inline void update_remote_rdptr_cleared() {
        if (fvc_pull_rdptr != inbound_rdptr.ptr_cleared) {
            inbound_rdptr.ptr_cleared = fvc_pull_rdptr;
            if constexpr (fvc_mode == FVC_MODE_ROUTER) {
                internal_::eth_send_packet(
                    0,
                    ((uint32_t)&inbound_rdptr) / PACKET_WORD_SIZE_BYTES,
                    remote_ptr_update_addr / PACKET_WORD_SIZE_BYTES,
                    1);
            }
        }
    }

    inline void advance_next_packet() {
        if (this->get_num_words_available() >= PACKET_HEADER_SIZE_WORDS) {
            tt_l1_ptr uint32_t* packet_header_ptr = (uint32_t*)&current_packet_header;
            tt_l1_ptr volatile uint32_t* next_header_ptr =
                reinterpret_cast<tt_l1_ptr uint32_t*>(get_local_buffer_read_addr());
            uint32_t words_before_wrap = words_before_buffer_wrap(fvc_out_rdptr);
            uint32_t dwords_to_copy = PACKET_HEADER_SIZE_BYTES / 4;
            if (words_before_wrap < PACKET_HEADER_SIZE_WORDS) {
                // Header spans buffer end.
                // Needs to be copied in two steps.
                uint32_t dwords_before_wrap = words_before_wrap * PACKET_WORD_SIZE_BYTES / 4;
                uint32_t dwords_after_wrap = dwords_to_copy - dwords_before_wrap;
                for (uint32_t i = 0; i < dwords_before_wrap; i++) {
                    packet_header_ptr[i] = next_header_ptr[i];
                }
                next_header_ptr = reinterpret_cast<tt_l1_ptr uint32_t*>(buffer_start);
                for (uint32_t i = 0; i < dwords_after_wrap; i++) {
                    packet_header_ptr[i + dwords_before_wrap] = next_header_ptr[i];
                }
            } else {
                for (uint32_t i = 0; i < dwords_to_copy; i++) {
                    packet_header_ptr[i] = next_header_ptr[i];
                }
            }

            this->packet_words_remaining =
                (this->current_packet_header.routing.packet_size_bytes + PACKET_WORD_SIZE_BYTES - 1) >> 4;
            this->packet_words_sent = 0;
            if (tt_fabric_is_header_valid(&current_packet_header)) {
                this->curr_packet_valid = true;
            } else {
                this->packet_corrupted = true;
            }
        }
    }

    inline void copy_header(pull_request_t* req) {
        uint32_t* dst = (uint32_t*)req;
        uint32_t* src = (uint32_t*)&current_packet_header;
        for (uint32_t i = 0; i < sizeof(pull_request_t) / 4; i++) {
            dst[i] = src[i];
        }
    }

    uint32_t get_next_hop_router_noc_xy() {
        uint32_t dst_mesh_id = current_packet_header.routing.dst_mesh_id;
        if (dst_mesh_id != routing_table->my_mesh_id) {
            uint32_t next_port = routing_table->inter_mesh_table.dest_entry[dst_mesh_id];
            return eth_chan_to_noc_xy[noc_index][next_port];
        } else {
            uint32_t dst_device_id = current_packet_header.routing.dst_dev_id;
            uint32_t next_port = routing_table->intra_mesh_table.dest_entry[dst_device_id];
            return eth_chan_to_noc_xy[noc_index][next_port];
        }
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    inline uint32_t pull_data_from_fvc_buffer() {
        uint32_t words_available = get_num_words_available();
        words_available = std::min(words_available, packet_words_remaining);
        if (packet_in_progress == 0) {
            advance_out_wrptr(words_available);
            if (current_packet_header.routing.flags == INLINE_FORWARD) {
                copy_header((pull_request_t*)&local_pull_request->pull_request);
            } else {
                local_pull_request->pull_request.wr_ptr = fvc_out_wrptr;
                local_pull_request->pull_request.rd_ptr = fvc_out_rdptr;
                local_pull_request->pull_request.size = current_packet_header.routing.packet_size_bytes;
                local_pull_request->pull_request.buffer_size = buffer_size;
                local_pull_request->pull_request.buffer_start = xy_local_addr + buffer_start;
                local_pull_request->pull_request.ack_addr =
                    xy_local_addr + (uint32_t)&local_pull_request->pull_request.rd_ptr;
                local_pull_request->pull_request.flags = FORWARD;
                packet_in_progress = 1;
            }
            packet_words_remaining -= words_available;
            advance_out_rdptr(words_available);
            // issue noc write to noc target of pull request.
            uint64_t dest_addr = ((uint64_t)get_next_hop_router_noc_xy() << 32) | FABRIC_ROUTER_REQ_QUEUE_START;
            packet_dest = tt_fabric_send_pull_request(dest_addr, local_pull_request);
            if (current_packet_header.routing.flags == INLINE_FORWARD) {
                curr_packet_valid = false;
                flush_async_writes<fvc_mode>();
                return words_available;
            }
        } else {
            // pull_request.rd_ptr is updated by remote puller when data is read out of producer's local buffer.
            // it is used to determine when it it safe to reclaim local buffer memory for more data.
            fvc_pull_rdptr = local_pull_request->pull_request.rd_ptr;
            if (packet_words_remaining) {
                if (words_available) {
                    advance_out_wrptr(words_available);
                    // packet_dest is returned by tt_fabric_send_pull_request() as the address of request q entry +
                    // pull_request.wr_ptr.
                    noc_inline_dw_write(packet_dest, fvc_out_wrptr);
                    advance_out_rdptr(words_available);
                    packet_words_remaining -= words_available;
                }
            } else if (fvc_pull_rdptr == fvc_out_rdptr) {
                // all data has been pulled and cleared from local buffer
                packet_in_progress = 0;
                curr_packet_valid = false;
            }
        }
        // send ptr cleared to ethernet sender.
        update_remote_rdptr_cleared<fvc_mode>();
        return words_available;
    }

    inline uint32_t issue_async_write() {
        uint32_t words_available = get_num_words_available();
        words_available = std::min(words_available, packet_words_remaining);
        words_available = std::min(words_available, words_before_buffer_wrap(fvc_out_rdptr));
        if (words_available) {
            noc_async_write(get_local_buffer_read_addr(), packet_dest, words_available * PACKET_WORD_SIZE_BYTES);
            packet_words_remaining -= words_available;
            advance_out_wrptr(words_available);
            advance_out_rdptr(words_available);
            packet_dest += words_available * PACKET_WORD_SIZE_BYTES;
        }
        return words_available;
    }

    inline bool packet_is_for_local_chip() {
        return (current_packet_header.routing.dst_mesh_id == routing_table->my_mesh_id) &&
               (current_packet_header.routing.dst_dev_id == routing_table->my_device_id);
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    inline uint32_t process_inbound_packet() {
        uint32_t words_processed = 0;
        if (packet_is_for_local_chip()) {
            if (current_packet_header.routing.flags == FORWARD && current_packet_header.session.command == ASYNC_WR) {
                if (packet_in_progress == 0) {
                    packet_dest = ((uint64_t)current_packet_header.session.target_offset_h << 32) |
                                  current_packet_header.session.target_offset_l;
                    packet_words_remaining -= PACKET_HEADER_SIZE_WORDS;
                    advance_out_wrptr(PACKET_HEADER_SIZE_WORDS);
                    advance_out_rdptr(PACKET_HEADER_SIZE_WORDS);
                    // subtract the header words. Remaining words are the data to be written to packet_dest.
                    // Remember to account for trailing bytes which may not be a full packet word.
                    packet_in_progress = 1;
                    words_processed = PACKET_HEADER_SIZE_WORDS;
                    words_processed += issue_async_write();
                } else {
                    flush_async_writes();
                    if (packet_words_remaining) {
                        words_processed = issue_async_write();
                    } else {
                        packet_in_progress = 0;
                        curr_packet_valid = false;
                    }
                }
            } else if (current_packet_header.routing.flags == INLINE_FORWARD) {
                uint64_t noc_addr = ((uint64_t)current_packet_header.session.target_offset_h << 32) |
                                    current_packet_header.session.target_offset_l;
                noc_fast_atomic_increment(
                    noc_index,
                    NCRISC_AT_CMD_BUF,
                    noc_addr,
                    NOC_UNICAST_WRITE_VC,
                    current_packet_header.packet_parameters.atomic_parameters.increment,
                    current_packet_header.packet_parameters.atomic_parameters.wrap_boundary,
                    false);

                packet_words_remaining -= PACKET_HEADER_SIZE_WORDS;
                advance_out_wrptr(PACKET_HEADER_SIZE_WORDS);
                advance_out_rdptr(PACKET_HEADER_SIZE_WORDS);
                words_processed = PACKET_HEADER_SIZE_WORDS;
                fvc_pull_rdptr = fvc_out_rdptr;
                update_remote_rdptr_cleared<fvc_mode>();
                curr_packet_valid = false;
            }
        } else {
            words_processed = pull_data_from_fvc_buffer<fvc_mode>();
        }
        return words_processed;
    }

    template <uint8_t fvc_mode = FVC_MODE_ROUTER>
    inline void flush_async_writes() {
        noc_async_write_barrier();
        fvc_pull_rdptr = fvc_out_rdptr;
        update_remote_rdptr_cleared<fvc_mode>();
    }

} fvc_producer_state_t;

typedef struct router_state {
    uint32_t sync_in;
    uint32_t padding_in[3];
    uint32_t sync_out;
    uint32_t padding_out[3];
    uint32_t scratch[4];
} router_state_t;

inline uint64_t get_timestamp() {
    uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return (((uint64_t)timestamp_high) << 32) | timestamp_low;
}

inline uint64_t get_timestamp_32b() { return reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void tt_fabric_add_header_checksum(packet_header_t* p_header) {
    uint16_t* ptr = (uint16_t*)p_header;
    uint32_t sum = 0;
    for (uint32_t i = 2; i < sizeof(packet_header_t) / 2; i++) {
        sum += ptr[i];
    }
    sum = ~sum;
    sum += sum;
    p_header->packet_parameters.misc_parameters.words[0] = sum;
}

bool tt_fabric_is_header_valid(packet_header_t* p_header) {
    uint16_t* ptr = (uint16_t*)p_header;
    uint32_t sum = 0;
    for (uint32_t i = 2; i < sizeof(packet_header_t) / 2; i++) {
        sum += ptr[i];
    }
    sum = ~sum;
    sum += sum;
    return (p_header->packet_parameters.misc_parameters.words[0] == sum);
}

void zero_l1_buf(tt_l1_ptr uint32_t* buf, uint32_t size_bytes) {
    for (uint32_t i = 0; i < size_bytes / 4; i++) {
        buf[i] = 0;
    }
}

static FORCE_INLINE void write_test_results(tt_l1_ptr uint32_t* const buf, uint32_t i, uint32_t val) {
    if (buf != nullptr) {
        buf[i] = val;
    }
}

static FORCE_INLINE void write_kernel_status(tt_l1_ptr uint32_t* const buf, uint32_t i, uint32_t val) {
    if (buf != nullptr) {
        buf[i] = val;
    }
}

static FORCE_INLINE void set_64b_result(uint32_t* buf, uint64_t val, uint32_t index = 0) {
    if (buf != nullptr) {
        buf[index] = val >> 32;
        buf[index + 1] = val & 0xFFFFFFFF;
    }
}

inline void req_buf_ptr_advance(chan_ptr* ptr) { ptr->ptr = (ptr->ptr + 1) & CHAN_REQ_BUF_PTR_MASK; }

inline void req_buf_advance_wrptr(chan_req_buf* req_buf) { req_buf_ptr_advance(&(req_buf->wrptr)); }

inline void req_buf_advance_rdptr(chan_req_buf* req_buf) {
    // clear valid before incrementing read pointer.
    uint32_t rd_index = req_buf->rdptr.ptr & CHAN_REQ_BUF_SIZE_MASK;
    req_buf->chan_req[rd_index].bytes[47] = 0;
    req_buf_ptr_advance(&(req_buf->rdptr));
}

inline bool req_buf_ptrs_empty(uint32_t wrptr, uint32_t rdptr) { return (wrptr == rdptr); }

inline bool req_buf_ptrs_full(uint32_t wrptr, uint32_t rdptr) {
    uint32_t distance = wrptr >= rdptr ? wrptr - rdptr : wrptr + 2 * CHAN_REQ_BUF_SIZE - rdptr;
    return !req_buf_ptrs_empty(wrptr, rdptr) && (distance >= CHAN_REQ_BUF_SIZE);
}

inline bool fvc_req_buf_is_empty(const volatile chan_req_buf* req_buf) {
    return req_buf_ptrs_empty(req_buf->wrptr.ptr, req_buf->rdptr.ptr);
}

inline bool fvc_req_buf_is_full(const volatile chan_req_buf* req_buf) {
    return req_buf_ptrs_full(req_buf->wrptr.ptr, req_buf->rdptr.ptr);
}

inline bool fvc_req_valid(const volatile chan_req_buf* req_buf) {
    uint32_t rd_index = req_buf->rdptr.ptr & CHAN_REQ_BUF_SIZE_MASK;
    return req_buf->chan_req[rd_index].pull_request.flags != 0;
}

inline uint32_t num_words_available_to_pull(volatile pull_request_t* pull_request) {
    uint32_t wr_ptr = pull_request->wr_ptr;
    uint32_t rd_ptr = pull_request->rd_ptr;
    uint32_t buf_size = pull_request->buffer_size;

    if (wr_ptr == rd_ptr) {
        // buffer empty.
        return 0;
    }
    uint32_t num_words = wr_ptr > rd_ptr ? wr_ptr - rd_ptr : buf_size * 2 + wr_ptr - rd_ptr;

    // num_words = std::min(num_words, this->get_curr_packet_words_remaining());
    return num_words;
}

inline uint32_t advance_ptr(uint32_t buffer_size, uint32_t ptr, uint32_t inc_words) {
    uint32_t temp = ptr + inc_words;
    if (temp >= buffer_size * 2) {
        temp -= buffer_size * 2;
    }
    return temp;
}

inline uint32_t words_before_buffer_wrap(uint32_t buffer_size, uint32_t rd_ptr) {
    if (rd_ptr >= buffer_size) {
        return buffer_size * 2 - rd_ptr;
    } else {
        return buffer_size - rd_ptr;
    }
}

inline uint32_t get_rd_ptr_offset_words(pull_request_t* pull_request) {
    uint32_t offset = pull_request->rd_ptr;
    if (pull_request->rd_ptr >= pull_request->buffer_size) {
        offset -= pull_request->buffer_size;
    }
    return offset;
}

inline void update_pull_request_words_cleared(pull_request_t* pull_request) {
    noc_inline_dw_write(pull_request->ack_addr, pull_request->rd_ptr);
}

inline uint32_t get_num_words_to_pull(volatile pull_request_t* pull_request, fvc_consumer_state_t* fvc_consumer_state) {
    uint32_t num_words_to_pull = num_words_available_to_pull(pull_request);
    uint32_t num_words_before_wrap = words_before_buffer_wrap(pull_request->buffer_size, pull_request->rd_ptr);

    num_words_to_pull = std::min(num_words_to_pull, num_words_before_wrap);
    uint32_t fvc_buffer_space = fvc_consumer_state->get_num_words_free();
    num_words_to_pull = std::min(num_words_to_pull, fvc_buffer_space);

    if (num_words_to_pull == 0) {
        return 0;
    }

    uint32_t fvc_space_before_wptr_wrap = fvc_consumer_state->words_before_local_buffer_wrap();
    num_words_to_pull = std::min(num_words_to_pull, fvc_space_before_wptr_wrap);
    num_words_to_pull = std::min(num_words_to_pull, DEFAULT_MAX_NOC_SEND_WORDS);

    return num_words_to_pull;
}

inline uint32_t pull_data_to_fvc_buffer(
    volatile pull_request_t* pull_request, fvc_consumer_state_t* fvc_consumer_state) {
    volatile uint32_t* temp = (volatile uint32_t*)0xffb2010c;
    if (fvc_consumer_state->packet_in_progress == 0) {
        uint32_t size = pull_request->size;
        fvc_consumer_state->packet_words_remaining = (size + PACKET_WORD_SIZE_BYTES - 1) >> 4;
        fvc_consumer_state->packet_in_progress = 1;
    }

    uint32_t num_words_to_pull = get_num_words_to_pull(pull_request, fvc_consumer_state);
    bool full_packet_sent = (num_words_to_pull == fvc_consumer_state->packet_words_remaining);
    if (num_words_to_pull == 0) {
        temp[0] = 0xdead1111;
        return 0;
    }

    uint32_t rd_offset = get_rd_ptr_offset_words((pull_request_t*)pull_request);
    uint64_t src_addr = pull_request->buffer_start + (rd_offset * PACKET_WORD_SIZE_BYTES);
    uint32_t fvc_addr = fvc_consumer_state->get_local_buffer_pull_addr();

    // pull_data_from_remote();
    noc_async_read(src_addr, fvc_addr, num_words_to_pull * PACKET_WORD_SIZE_BYTES);
    fvc_consumer_state->register_pull_data(num_words_to_pull);
    pull_request->rd_ptr = advance_ptr(pull_request->buffer_size, pull_request->rd_ptr, num_words_to_pull);

    // TODO: this->remote_wptr_update(num_words_to_forward);

    return num_words_to_pull;
}

inline uint32_t move_data_to_fvc_buffer(
    volatile pull_request_t* pull_request, fvc_consumer_state_t* fvc_consumer_state) {
    if (fvc_consumer_state->packet_in_progress == 0) {
        fvc_consumer_state->packet_words_remaining = PACKET_HEADER_SIZE_WORDS;
        fvc_consumer_state->packet_in_progress = 1;
    }

    // if fvc does not have enough space, try again later.
    if (fvc_consumer_state->get_num_words_free() < PACKET_HEADER_SIZE_WORDS) {
        return 0;
    }

    uint32_t fvc_space_before_wptr_wrap = fvc_consumer_state->words_before_local_buffer_wrap();
    uint32_t* fvc_addr = (uint32_t*)fvc_consumer_state->get_local_buffer_pull_addr();
    uint32_t* src = (uint32_t*)pull_request;

    switch (fvc_space_before_wptr_wrap) {
        case 1:
            fvc_addr[0] = src[0];
            fvc_addr[1] = src[1];
            fvc_addr[2] = src[2];
            fvc_addr[3] = src[3];
            fvc_addr = (uint32_t*)fvc_consumer_state->buffer_start;
            fvc_addr[0] = src[4];
            fvc_addr[1] = src[5];
            fvc_addr[2] = src[6];
            fvc_addr[3] = src[7];
            fvc_addr[4] = src[8];
            fvc_addr[5] = src[9];
            fvc_addr[6] = src[10];
            fvc_addr[7] = src[11];
            break;
        case 2:
            // uint32_t i = 0;
            for (uint32_t i = 0; i < (PACKET_HEADER_SIZE_WORDS - 1) * PACKET_WORD_SIZE_BYTES / 4; i++) {
                fvc_addr[i] = src[i];
            }
            fvc_addr = (uint32_t*)fvc_consumer_state->buffer_start;
            fvc_addr[0] = src[8];
            fvc_addr[1] = src[9];
            fvc_addr[2] = src[10];
            fvc_addr[3] = src[11];
            break;
        default:
            for (uint32_t i = 0; i < PACKET_HEADER_SIZE_BYTES / 4; i++) {
                fvc_addr[i] = src[i];
            }
    }

    fvc_consumer_state->register_pull_data(PACKET_HEADER_SIZE_WORDS);
    return PACKET_HEADER_SIZE_WORDS;
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

// issue a pull request.
// currently blocks till the request queue has space.
// This needs to be non blocking, so that if one fvc pull request queue is full,
// we can process other fvcs and come back to check status of this pull request later.
inline uint64_t tt_fabric_send_pull_request(uint64_t dest_addr, volatile local_pull_request_t* local_pull_request) {
    uint64_t noc_addr = dest_addr + offsetof(chan_req_buf, wrptr);
    noc_fast_atomic_increment(
        noc_index,
        NCRISC_AT_CMD_BUF,
        noc_addr,
        NOC_UNICAST_WRITE_VC,
        1,
        CHAN_REQ_BUF_LOG_SIZE /*wrap*/,
        false /*linked*/);
    while (!ncrisc_noc_nonposted_atomics_flushed(noc_index));
    uint32_t wrptr = *(volatile uint32_t*)MEM_NOC_ATOMIC_RET_VAL_ADDR;
    local_pull_request->wrptr.ptr = wrptr;
    noc_addr = dest_addr + offsetof(chan_req_buf, rdptr);
    while (1) {
        noc_async_read_one_packet(noc_addr, (uint32_t)(&local_pull_request->rdptr.ptr), 4);
        noc_async_read_barrier();
        if (!req_buf_ptrs_full(wrptr, local_pull_request->rdptr.ptr)) {
            break;
        }
    }
    uint32_t dest_wr_index = wrptr & CHAN_REQ_BUF_SIZE_MASK;
    noc_addr = dest_addr + offsetof(chan_req_buf, chan_req) + dest_wr_index * sizeof(pull_request_t);
    noc_async_write_one_packet(
        (uint32_t)(&local_pull_request->pull_request), noc_addr, sizeof(pull_request_t), noc_index);

    // compute the address to send write pointer updates to consumer buffer.
    // This will happen, if the producer did not have all the availale data in its buffer when
    // the pull request was first issued. In this case, as the producer gets more data in its buffer,
    // it updates write pointer in the consumer request buffer pull request entry.
    uint64_t wr_ptr_addr = noc_addr = offsetof(pull_request_t, wr_ptr);
    return wr_ptr_addr;
}

inline void tt_fabric_init() {
    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(noc_index, 0, NOC_NODE_ID);
    uint32_t my_x = noc_id_reg & NOC_NODE_ID_MASK;
    uint32_t my_y = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;
    xy_local_addr = NOC_XY_ADDR(my_x, my_y, 0);
}
