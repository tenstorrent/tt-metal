// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "risc_attribs.h"
#include "tt_metal/hostdevcommon/common_values.hpp"
#include "tt_metal/impl/dispatch/kernels/command_queue_common.hpp"
#include "cq_cmds.hpp"
#include "dataflow_api.h"
#include "noc_overlay_parameters.h"
#include "ethernet/dataflow_api.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"


constexpr uint32_t NUM_WR_CMD_BUFS = 4;

constexpr uint32_t DEFAULT_MAX_NOC_SEND_WORDS = (NUM_WR_CMD_BUFS-1)*(NOC_MAX_BURST_WORDS*NOC_WORD_BYTES)/PACKET_WORD_SIZE_BYTES;
constexpr uint32_t DEFAULT_MAX_ETH_SEND_WORDS = 2*1024;

constexpr uint32_t NUM_PTR_REGS_PER_QUEUE = 3;


inline uint64_t get_timestamp() {
    uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return (((uint64_t)timestamp_high) << 32) | timestamp_low;
}

inline uint64_t get_timestamp_32b() {
    return reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
}

void zero_l1_buf(tt_l1_ptr uint32_t* buf, uint32_t size_bytes) {
    for (uint32_t i = 0; i < size_bytes/4; i++) {
        buf[i] = 0;
    }
}

typedef struct dispatch_packet_header_t dispatch_packet_header_t;

static_assert(sizeof(dispatch_packet_header_t) == PACKET_WORD_SIZE_BYTES);


class packet_queue_state_t;
class packet_input_queue_state_t;
class packet_output_queue_state_t;

class packet_queue_state_t {

    volatile uint32_t* local_wptr_val;
    volatile uint32_t* local_rptr_sent_val;
    volatile uint32_t* local_rptr_cleared_val;
    volatile uint32_t* local_wptr_update;
    volatile uint32_t* local_rptr_sent_update;
    volatile uint32_t* local_rptr_cleared_update;
    volatile uint32_t* local_wptr_reset;
    volatile uint32_t* local_rptr_sent_reset;
    volatile uint32_t* local_rptr_cleared_reset;

    uint32_t remote_ready_status_addr;
    volatile uint32_t* local_ready_status_ptr;

    uint32_t remote_wptr_update_addr;
    uint32_t remote_rptr_sent_update_addr;
    uint32_t remote_rptr_cleared_update_addr;

public:

    uint8_t queue_id;
    uint32_t queue_start_addr_words;
    uint16_t queue_size_words;
    uint32_t ptr_offset_mask;
    uint32_t queue_size_mask;

    bool queue_is_input;

    uint8_t remote_x, remote_y; // remote source for input queues, remote dest for output queues
    uint8_t remote_queue_id;
    DispatchRemoteNetworkType remote_update_network_type;

    // For read/write pointers, we use stream credit registers with auto-increment.
    // Pointers are in 16B units, and we assume buffer size is power of 2 so we get
    // automatic wrapping. (If needed, we can fix the pointer advance functions later
    // to handle non-power-of-2 buffer sizes.)

    void init(uint8_t queue_id,
              uint32_t queue_start_addr_words,
              uint16_t queue_size_words,
              uint8_t remote_x,
              uint8_t remote_y,
              uint8_t remote_queue_id,
              DispatchRemoteNetworkType remote_update_network_type) {

        this->queue_id = queue_id;
        this->queue_start_addr_words = queue_start_addr_words;
        this->queue_size_words = queue_size_words;
        this->remote_x = remote_x;
        this->remote_y = remote_y;
        this->remote_queue_id = remote_queue_id;
        this->remote_update_network_type = remote_update_network_type;

        this->local_wptr_val = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        this->local_rptr_sent_val = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id+1, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));
        this->local_rptr_cleared_val = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id+2, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX));

        this->local_wptr_update = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
        this->local_rptr_sent_update = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id+1, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));
        this->local_rptr_cleared_update = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id+2, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX));

        // Setting STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX resets the credit register
        this->local_wptr_reset = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
        this->local_rptr_sent_reset = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id+1, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));
        this->local_rptr_cleared_reset = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*queue_id+2, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX));

        this->remote_wptr_update_addr =
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*remote_queue_id,
                            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
        this->remote_rptr_sent_update_addr =
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*remote_queue_id+1,
                            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
        this->remote_rptr_cleared_update_addr =
            STREAM_REG_ADDR(NUM_PTR_REGS_PER_QUEUE*remote_queue_id+2,
                            STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);

        this->remote_ready_status_addr = STREAM_REG_ADDR(remote_queue_id, STREAM_REMOTE_SRC_REG_INDEX);
        this->local_ready_status_ptr = reinterpret_cast<volatile uint32_t*>(
            STREAM_REG_ADDR(queue_id, STREAM_REMOTE_SRC_REG_INDEX));
    }

    inline uint8_t get_queue_id() const {
        return this->queue_id;
    }

    inline uint32_t get_queue_local_wptr() const {
        return *this->local_wptr_val;
    }

    inline void reset_queue_local_wptr() {
        *this->local_wptr_reset = 0;
    }

    inline void advance_queue_local_wptr(uint32_t num_words) {
        *this->local_wptr_update = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
    }

    inline uint32_t get_queue_local_rptr_sent() const {
        return *this->local_rptr_sent_val;
    }

    inline uint32_t get_queue_local_rptr_cleared() const {
        return *this->local_rptr_cleared_val;
    }

    inline void reset_queue_local_rptr_sent()  {
        *this->local_rptr_sent_reset = 0;
    }

    inline void reset_queue_local_rptr_cleared()  {
        *this->local_rptr_cleared_reset = 0;
    }

    inline void advance_queue_local_rptr_sent(uint32_t num_words)  {
        *this->local_rptr_sent_update = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
    }

    inline void advance_queue_local_rptr_cleared(uint32_t num_words)  {
        *this->local_rptr_cleared_update = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
    }

    inline uint32_t get_queue_data_num_words_available_to_send() const {
        return (this->get_queue_local_wptr() - this->get_queue_local_rptr_sent()) & this->queue_size_mask;
    }

    inline uint32_t get_queue_data_num_words_occupied() const {
        return (this->get_queue_local_wptr() - this->get_queue_local_rptr_cleared()) & this->queue_size_mask;
    }

    inline uint32_t get_queue_data_num_words_free() const {
        return this->queue_size_words - this->get_queue_data_num_words_occupied();
    }

    inline uint32_t get_num_words_sent_not_cleared() const {
        return (this->get_queue_local_rptr_sent() - this->get_queue_local_rptr_cleared()) & this->queue_size_mask;
    }

    inline uint32_t get_num_words_written_not_sent() const {
        return (this->get_queue_local_wptr() - this->get_queue_local_rptr_sent()) & this->queue_size_mask;
    }

    inline uint32_t get_queue_wptr_offset_words() const {
        return this->get_queue_local_wptr() & this->ptr_offset_mask;
    }

    inline uint32_t get_queue_rptr_sent_offset_words() const {
        return this->get_queue_local_rptr_sent() & this->ptr_offset_mask;
    }

    inline uint32_t get_queue_rptr_cleared_offset_words() const {
        return this->get_queue_local_rptr_cleared() & this->ptr_offset_mask;
    }

    inline uint32_t get_queue_rptr_sent_addr_bytes() const {
        return (this->queue_start_addr_words + this->get_queue_rptr_sent_offset_words())*PACKET_WORD_SIZE_BYTES;
    }

    inline uint32_t get_queue_rptr_cleared_addr_bytes() const {
        return (this->queue_start_addr_words + this->get_queue_rptr_cleared_offset_words())*PACKET_WORD_SIZE_BYTES;
    }

    inline uint32_t get_queue_wptr_addr_bytes() const {
        return (this->queue_start_addr_words + this->get_queue_wptr_offset_words())*PACKET_WORD_SIZE_BYTES;
    }

    inline uint32_t get_queue_words_before_rptr_sent_wrap() const {
        return queue_size_words - this->get_queue_rptr_sent_offset_words();
    }

    inline uint32_t get_queue_words_before_rptr_cleared_wrap() const {
        return queue_size_words - this->get_queue_rptr_cleared_offset_words();
    }

    inline uint32_t get_queue_words_before_wptr_wrap() const {
        return queue_size_words - this->get_queue_wptr_offset_words();
    }

    inline void remote_reg_update(uint32_t reg_addr, uint32_t val) {

        if (this->remote_update_network_type == DispatchRemoteNetworkType::NONE) {
            return;
        }
        else if (this->remote_update_network_type == DispatchRemoteNetworkType::ETH) {
            eth_write_remote_reg(reg_addr, val);
        } else {
            uint64_t dest_addr = NOC_XY_ADDR(this->remote_x, this->remote_y, reg_addr);
            noc_inline_dw_write(dest_addr, val);
        }
    }

    inline void advance_queue_remote_wptr(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = this->remote_wptr_update_addr;
        this->remote_reg_update(reg_addr, val);
    }

    inline void advance_queue_remote_rptr_sent(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = this->remote_rptr_sent_update_addr;
        this->remote_reg_update(reg_addr, val);
    }

    inline void advance_queue_remote_rptr_cleared(uint32_t num_words) {
        uint32_t val = num_words << REMOTE_DEST_BUF_WORDS_FREE_INC;
        uint32_t reg_addr = this->remote_rptr_cleared_update_addr;
        this->remote_reg_update(reg_addr, val);
    }

    inline void reset_ready_flag() {
        *this->local_ready_status_ptr = 0;
    }

    inline void send_remote_ready_notification() {
        this->remote_reg_update(this->remote_ready_status_addr,
                                PACKET_QUEUE_REMOTE_READY_FLAG);
    }

    inline void send_remote_finished_notification() {
        this->remote_reg_update(this->remote_ready_status_addr,
                                PACKET_QUEUE_REMOTE_FINISHED_FLAG);
    }

    inline bool is_remote_ready() const {
        return *this->local_ready_status_ptr == PACKET_QUEUE_REMOTE_READY_FLAG;
    }

    inline uint32_t get_remote_ready_status() const {
        return *this->local_ready_status_ptr;
    }

    inline bool is_remote_finished() const {
        return *this->local_ready_status_ptr == PACKET_QUEUE_REMOTE_FINISHED_FLAG;
    }

    void yield() {
        // TODO: implement yield for ethernet here
    }

    void dprint_object() {
        DPRINT << "  id: " << DEC() << static_cast<uint32_t>(this->queue_id) << ENDL();
        DPRINT << "  start_addr: 0x" << HEX() << static_cast<uint32_t>(this->queue_start_addr_words*PACKET_WORD_SIZE_BYTES) << ENDL();
        DPRINT << "  size_bytes: 0x" << HEX() << static_cast<uint32_t>(this->queue_size_words*PACKET_WORD_SIZE_BYTES) << ENDL();
        DPRINT << "  remote_x: " << DEC() << static_cast<uint32_t>(this->remote_x) << ENDL();
        DPRINT << "  remote_y: " << DEC() << static_cast<uint32_t>(this->remote_y) << ENDL();
        DPRINT << "  remote_queue_id: " << DEC() << static_cast<uint32_t>(this->remote_queue_id) << ENDL();
        DPRINT << "  remote_update_network_type: " << DEC() << static_cast<uint32_t>(this->remote_update_network_type) << ENDL();
        DPRINT << "  ready_status: 0x" << HEX() << this->get_remote_ready_status() << ENDL();
        DPRINT << "  local_wptr: 0x" << HEX() << this->get_queue_local_wptr() << ENDL();
        DPRINT << "  local_rptr_sent: 0x" << HEX() << this->get_queue_local_rptr_sent() << ENDL();
        DPRINT << "  local_rptr_cleared: 0x" << HEX() << this->get_queue_local_rptr_cleared() << ENDL();
    }
};


class packet_input_queue_state_t : public packet_queue_state_t {

protected:

    bool curr_packet_valid;
    tt_l1_ptr dispatch_packet_header_t* curr_packet_header_ptr;
    uint16_t curr_packet_src;
    uint16_t curr_packet_dest;
    uint16_t curr_packet_size_words;
    uint16_t curr_packet_words_sent;
    uint32_t curr_packet_tag;
    uint16_t curr_packet_flags;

    inline void advance_next_packet() {
        if(this->get_queue_data_num_words_available_to_send() > 0) {
            tt_l1_ptr dispatch_packet_header_t* next_packet_header_ptr =
                reinterpret_cast<tt_l1_ptr dispatch_packet_header_t*>(
                    (this->queue_start_addr_words + this->get_queue_rptr_sent_offset_words())*PACKET_WORD_SIZE_BYTES
                );
            this->curr_packet_header_ptr = next_packet_header_ptr;
            this->curr_packet_dest = next_packet_header_ptr->packet_dest;
            this->curr_packet_src = next_packet_header_ptr->packet_src;
            this->curr_packet_size_words = next_packet_header_ptr->packet_size_words;
            this->curr_packet_tag = next_packet_header_ptr->tag;
            this->curr_packet_flags = next_packet_header_ptr->packet_flags;
            this->curr_packet_words_sent = 0;
            this->curr_packet_valid = true;
       }
    }

public:

    void init(uint8_t queue_id,
              uint32_t queue_start_addr_words,
              uint16_t queue_size_words,
              uint8_t remote_x,
              uint8_t remote_y,
              uint8_t remote_queue_id,
              DispatchRemoteNetworkType remote_update_network_type) {

        packet_queue_state_t::init(queue_id, queue_start_addr_words, queue_size_words,
                                   remote_x, remote_y, remote_queue_id, remote_update_network_type);

        tt_l1_ptr uint32_t* queue_ptr =
            reinterpret_cast<tt_l1_ptr uint32_t*>(queue_start_addr_words*PACKET_WORD_SIZE_BYTES);
        zero_l1_buf(queue_ptr, queue_size_words*PACKET_WORD_SIZE_BYTES);

        this->ptr_offset_mask = queue_size_words - 1;
        this->queue_size_mask = (queue_size_words << 1) - 1;
        this->curr_packet_valid = false;
        this->reset_queue_local_rptr_sent();
        this->reset_queue_local_rptr_cleared();
        this->reset_queue_local_wptr();
        this->reset_ready_flag();
    }


    inline bool get_curr_packet_valid() {
        if (!this->curr_packet_valid && (this->get_queue_data_num_words_available_to_send() > 0)){
            this->advance_next_packet();
        }
        return this->curr_packet_valid;
    }

    inline tt_l1_ptr dispatch_packet_header_t* get_curr_packet_header_ptr() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_header_ptr;
    }

    inline uint32_t get_curr_packet_dest() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_dest;
    }

    inline uint32_t get_curr_packet_src() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_src;
    }

    inline uint32_t get_curr_packet_size_words() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_size_words;
    }

    inline uint32_t get_curr_packet_words_remaining() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_size_words - this->curr_packet_words_sent;
    }

    inline uint32_t get_curr_packet_tag() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_tag;
    }

    inline uint32_t get_curr_packet_flags() {
        if (!this->curr_packet_valid) {
            this->advance_next_packet();
        }
        return this->curr_packet_flags;
    }

    inline bool partial_packet_sent() const {
        return this->curr_packet_valid && (this->curr_packet_words_sent > 0);
    }


    inline bool input_queue_full_packet_available_to_send(uint32_t& num_words_available_to_send) {
        num_words_available_to_send = this->get_queue_data_num_words_available_to_send();
        if (num_words_available_to_send == 0) {
            return false;
        }
        return num_words_available_to_send >= this->get_curr_packet_words_remaining();
    }

    inline uint32_t input_queue_curr_packet_num_words_available_to_send() {
        uint32_t num_words = this->get_queue_data_num_words_available_to_send();
        if (num_words == 0) {
            return 0;
        }
        num_words = std::min(num_words, this->get_curr_packet_words_remaining());
        return num_words;
    }

    inline void input_queue_advance_words_sent(uint32_t num_words) {
        if (num_words > 0) {
            this->advance_queue_local_rptr_sent(num_words);
            this->advance_queue_remote_rptr_sent(num_words);
            this->curr_packet_words_sent += num_words;
            uint32_t curr_packet_words_remaining = this->get_curr_packet_words_remaining();
            if (curr_packet_words_remaining == 0) {
                this->curr_packet_valid = false;
                this->advance_next_packet();
            }
        }
    }

    inline void input_queue_advance_words_cleared(uint32_t num_words) {
        if (num_words > 0) {
            this->advance_queue_local_rptr_cleared(num_words);
            this->advance_queue_remote_rptr_cleared(num_words);
        }
    }

    inline void input_queue_clear_all_words_sent() {
        this->input_queue_advance_words_cleared(this->get_num_words_sent_not_cleared());
    }

    void dprint_object() {
        DPRINT << "Input queue:" << ENDL();
        packet_queue_state_t::dprint_object();
        DPRINT << "  packet_valid: " << DEC() << static_cast<uint32_t>(this->curr_packet_valid) << ENDL();
        DPRINT << "  packet_tag: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_tag) << ENDL();
        DPRINT << "  packet_src: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_src) << ENDL();
        DPRINT << "  packet_dest: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_dest) << ENDL();
        DPRINT << "  packet_flags: 0x" << HEX() << static_cast<uint32_t>(this->curr_packet_flags) << ENDL();
        DPRINT << "  packet_size_words: " << DEC() << static_cast<uint32_t>(this->curr_packet_size_words) << ENDL();
        DPRINT << "  packet_words_sent: " << DEC() << static_cast<uint32_t>(this->curr_packet_words_sent) << ENDL();
    }

};


class packet_output_queue_state_t : public packet_queue_state_t {

    uint32_t max_noc_send_words;
    uint32_t max_eth_send_words;

    struct {

        packet_input_queue_state_t* input_queue_array;
        uint32_t input_queue_words_in_flight[2*MAX_SWITCH_FAN_IN];

        uint32_t* curr_input_queue_words_in_flight;
        uint32_t* prev_input_queue_words_in_flight;
        uint32_t curr_total_words_in_flight;
        uint32_t prev_total_words_in_flight;

        uint8_t num_input_queues;

        void init(packet_input_queue_state_t* input_queue_array, uint32_t num_input_queues) {
            this->num_input_queues = num_input_queues;
            this->input_queue_array = input_queue_array;
            this->curr_input_queue_words_in_flight = &(this->input_queue_words_in_flight[0]);
            this->prev_input_queue_words_in_flight = &(this->input_queue_words_in_flight[MAX_SWITCH_FAN_IN]);
            this->curr_total_words_in_flight = 0;
            this->prev_total_words_in_flight = 0;
            for (uint32_t i = 0; i < MAX_SWITCH_FAN_IN; i++) {
                this->curr_input_queue_words_in_flight[i] = 0;
                this->prev_input_queue_words_in_flight[i] = 0;
            }
        }

        inline uint32_t get_curr_total_words_in_flight() const {
            return this->curr_total_words_in_flight;
        }

        inline uint32_t get_prev_total_words_in_flight() const {
            return this->prev_total_words_in_flight;
        }

        inline uint32_t prev_words_in_flight_flush() {

            uint32_t words_flushed = this->prev_total_words_in_flight;
            if (words_flushed > 0) {
                for (uint32_t i = 0; i < num_input_queues; i++) {
                    this->input_queue_array[i].input_queue_advance_words_cleared(this->prev_input_queue_words_in_flight[i]);
                    this->prev_input_queue_words_in_flight[i] = 0;
                }
            }

            uint32_t* tmp = this->prev_input_queue_words_in_flight;
            this->prev_input_queue_words_in_flight = this->curr_input_queue_words_in_flight;
            this->curr_input_queue_words_in_flight = tmp;
            this->prev_total_words_in_flight = this->curr_total_words_in_flight;
            this->curr_total_words_in_flight = 0;

            return words_flushed;
        }

        inline void register_words_in_flight(uint32_t input_queue_id, uint32_t num_words) {
            this->curr_input_queue_words_in_flight[input_queue_id] += num_words;
            this->curr_total_words_in_flight += num_words;
            this->input_queue_array[input_queue_id].input_queue_advance_words_sent(num_words);
        }

        void dprint_object() {
            DPRINT << "  curr_total_words_in_flight: " << DEC() << this->curr_total_words_in_flight << ENDL();
            for (uint32_t j = 0; j < MAX_SWITCH_FAN_IN; j++) {
                DPRINT << "       from input queue id " << DEC() <<
                            this->input_queue_array[j].get_queue_id() << ": "
                            << DEC() << this->curr_input_queue_words_in_flight[j]
                            << ENDL();
            }
            DPRINT << "  prev_total_words_in_flight: " << DEC() << this->prev_total_words_in_flight << ENDL();
            for (uint32_t j = 0; j < MAX_SWITCH_FAN_IN; j++) {
                DPRINT << "       from input queue id " << DEC() <<
                            this->input_queue_array[j].get_queue_id() << ": "
                            << DEC() << this->prev_input_queue_words_in_flight[j]
                            << ENDL();
            }
        }

    } input_queue_status;

public:

    void init(uint8_t queue_id,
              uint32_t queue_start_addr_words,
              uint16_t queue_size_words,
              uint8_t remote_x,
              uint8_t remote_y,
              uint8_t remote_queue_id,
              DispatchRemoteNetworkType remote_update_network_type,
              packet_input_queue_state_t* input_queue_array,
              uint8_t num_input_queues) {

        packet_queue_state_t::init(queue_id, queue_start_addr_words, queue_size_words,
                                   remote_x, remote_y, remote_queue_id, remote_update_network_type);

        this->ptr_offset_mask = queue_size_words - 1;
        this->queue_size_mask = (queue_size_words << 1) - 1;
        this->max_noc_send_words = DEFAULT_MAX_NOC_SEND_WORDS;
        this->max_eth_send_words = DEFAULT_MAX_ETH_SEND_WORDS;
        this->input_queue_status.init(input_queue_array, num_input_queues);
        this->reset_queue_local_rptr_sent();
        this->reset_queue_local_rptr_cleared();
        this->reset_queue_local_wptr();
        this->reset_ready_flag();
    }

    inline void set_max_noc_send_words(uint32_t max_noc_send_words) {
        this->max_noc_send_words = max_noc_send_words;
    }

    inline void set_max_eth_send_words(uint32_t max_eth_send_words) {
        this->max_eth_send_words = max_eth_send_words;
    }

    inline uint32_t output_max_num_words_to_forward() const {
        return (this->remote_update_network_type == DispatchRemoteNetworkType::ETH) ?
            this->max_eth_send_words : this->max_noc_send_words;
    }

    inline void send_data_to_remote(uint32_t src_addr, uint32_t dest_addr, uint32_t num_words) {
        if ((this->remote_update_network_type == DispatchRemoteNetworkType::NONE) ||
            (num_words == 0)) {
            return;
        } else if (this->remote_update_network_type == DispatchRemoteNetworkType::ETH) {
            internal_::eth_send_packet(0, src_addr/PACKET_WORD_SIZE_BYTES, dest_addr/PACKET_WORD_SIZE_BYTES, num_words);
        } else {
            uint64_t noc_dest_addr = NOC_XY_ADDR(this->remote_x, this->remote_y, dest_addr);
            noc_async_write(src_addr, noc_dest_addr, num_words*PACKET_WORD_SIZE_BYTES);
        }
    }

    inline void remote_wptr_update(uint32_t num_words) {
        this->advance_queue_remote_wptr(num_words);
    }

    inline uint32_t prev_words_in_flight_check_flush() {
        if (this->get_num_words_written_not_sent() <= this->input_queue_status.get_curr_total_words_in_flight()) {
            return this->input_queue_status.prev_words_in_flight_flush();
        } else {
            return 0;
        }
    }

    bool output_barrier(uint32_t timeout_cycles = 0) {
        uint32_t start_timestamp = 0;
        if (timeout_cycles > 0) {
            start_timestamp = get_timestamp_32b();
        }
        while (this->get_queue_data_num_words_occupied() > 0) {
            if (timeout_cycles > 0) {
                uint32_t cycles_elapsed = get_timestamp_32b() - start_timestamp;
                if (cycles_elapsed > timeout_cycles) {
                    return false;
                }
            }
            this->yield();
        }
        this->input_queue_status.prev_words_in_flight_flush();
        this->input_queue_status.prev_words_in_flight_flush();
        return true;
    }

    inline uint32_t get_num_words_to_send(uint32_t input_queue_index) {

        packet_input_queue_state_t* input_queue_ptr = &(this->input_queue_status.input_queue_array[input_queue_index]);

        uint32_t num_words_available_in_input = input_queue_ptr->input_queue_curr_packet_num_words_available_to_send();
        uint32_t num_words_before_input_rptr_wrap = input_queue_ptr->get_queue_words_before_rptr_sent_wrap();
        num_words_available_in_input = std::min(num_words_available_in_input, num_words_before_input_rptr_wrap);
        uint32_t num_words_free_in_output = this->get_queue_data_num_words_free();
        uint32_t num_words_to_forward = std::min(num_words_available_in_input, num_words_free_in_output);

        if (num_words_to_forward == 0) {
            return 0;
        }

        uint32_t output_buf_words_before_wptr_wrap = this->get_queue_words_before_wptr_wrap();
        num_words_to_forward = std::min(num_words_to_forward, output_buf_words_before_wptr_wrap);
        num_words_to_forward = std::min(num_words_to_forward, this->output_max_num_words_to_forward());

        return num_words_to_forward;
    }

    inline uint32_t forward_data_from_input(uint32_t input_queue_index, bool& full_packet_sent) {

        packet_input_queue_state_t* input_queue_ptr = &(this->input_queue_status.input_queue_array[input_queue_index]);
        uint32_t num_words_to_forward = this->get_num_words_to_send(input_queue_index);
        full_packet_sent = (num_words_to_forward == input_queue_ptr->get_curr_packet_words_remaining());
        if (num_words_to_forward == 0) {
            return 0;
        }

        uint32_t src_addr =
            (input_queue_ptr->queue_start_addr_words +
             input_queue_ptr->get_queue_rptr_sent_offset_words())*PACKET_WORD_SIZE_BYTES;
        uint32_t dest_addr =
            (this->queue_start_addr_words + this->get_queue_wptr_offset_words())*PACKET_WORD_SIZE_BYTES;

        this->send_data_to_remote(src_addr, dest_addr, num_words_to_forward);
        this->input_queue_status.register_words_in_flight(input_queue_index, num_words_to_forward);
        this->advance_queue_local_wptr(num_words_to_forward);
        this->remote_wptr_update(num_words_to_forward);

        return num_words_to_forward;
    }

    void dprint_object() {
        DPRINT << "Output queue:" << ENDL();
        packet_queue_state_t::dprint_object();
        this->input_queue_status.dprint_object();
    }
};


/**
 *  Polling for ready signal from the remote peers of all input and output queues.
 *  Blocks until all are ready, but doesn't block polling on each individual queue.
 *  Returns false in case of timeout.
 */
bool wait_all_src_dest_ready(packet_input_queue_state_t* input_queue_array, uint32_t num_input_queues,
                             packet_output_queue_state_t* output_queue_array, uint32_t num_output_queues,
                             uint32_t timeout_cycles = 0) {

    bool all_src_dest_ready = false;
    bool src_ready[MAX_SWITCH_FAN_IN] = {false};
    bool dest_ready[MAX_SWITCH_FAN_OUT] = {false};

    uint32_t iters = 0;

    uint32_t start_timestamp = get_timestamp_32b();
    while (!all_src_dest_ready) {
        iters++;
        if (timeout_cycles > 0) {
            uint32_t cycles_since_start = get_timestamp_32b() - start_timestamp;
            if (cycles_since_start > timeout_cycles) {
                return false;
            }
        }
        all_src_dest_ready = true;
        for (uint32_t i = 0; i < num_input_queues; i++) {
            if (!src_ready[i]) {
                src_ready[i] = input_queue_array[i].is_remote_ready();
                if (!src_ready[i]) {
                    input_queue_array[i].send_remote_ready_notification();
                    all_src_dest_ready = false;
                } else {
                    // handshake with src complete
                }
            }
        }
        for (uint32_t i = 0; i < num_output_queues; i++) {
            if (!dest_ready[i]) {
                dest_ready[i] = output_queue_array[i].is_remote_ready();
                if (dest_ready[i]) {
                    output_queue_array[i].send_remote_ready_notification();
                } else {
                    all_src_dest_ready = false;
                }
            }
        }
    }
    return true;
}
